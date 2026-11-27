# -*- coding: utf-8 -*-
"""eval_isaac.py — 在 Isaac Gym【训练用环境】里做策略验收：速度跟踪 / 机身姿态 / 站姿几何 / 站立抖动。

为什么需要它（2026-09-30 起成为主验收工具）：
  · MuJoCo sim2sim 对同一 ckpt 复现不出训练行为（同一策略 Isaac 里 1.03 m/s 干净 trot，
    sim2sim 里 600s 前进 0.05m、步态相位混乱；摩擦/被动参数/帧序/映射 4 个旋钮全试过无效），
    所以姿态与速度必须回到 Isaac 里量。
  · 默认关掉【全部域随机化】⇒ 32 个环境是同一台"标称机器人"，指标可比、可复现。
    ⚠️ 这是历史踩过的坑: 开着 DR 测"零动作标称姿态"会得到混合随机机器人的污染值
    （V25 那次测出 pitch +0.81°/roll −7.3°，干净值是 +2.13°/+0.06°）。

用法（在仓库根目录跑；显卡需空闲——训练中跑会 CUDA OOM，8GB 卡实测过）:
  python tools/eval_isaac.py --run V26 --checkpoint 15600 --x_vel 1.0       # 平地行走验收
  python tools/eval_isaac.py --run V26 --checkpoint 15600 --x_vel 0.0       # 零指令站立（主判据）
  python tools/eval_isaac.py --run V26 --checkpoint 15600 --zero-action     # 纯 PD 标称姿态参照
  python tools/eval_isaac.py --run V26 --checkpoint 15600 --freeze-hip      # 干预: 锁髋看外八是不是拐杖
  python tools/eval_isaac.py --run V26 --checkpoint 15600 --free-cmd --terms # 训练同款指令流 + 奖励分项

选项:
  --run <V26>           logs/<experiment>/ 下的运行目录名
  --checkpoint <15600>  checkpoint 步数
  --x_vel/--y_vel/--yaw_vel   固定指令（默认 1.0/0/0）
  --num_envs <32>       并行环境数（越大越快，但也更吃显存）
  --seconds <20>        评估时长（仿真秒；前 2s 起步过渡不计入统计）
  --terrain <plane|terrain>   默认 plane（平地）；terrain=用配置里的地形
  --dr                  保留域随机化（默认关闭；开着时各环境是不同机器人，均值含义变差）
  --free-cmd            不覆盖指令，用配置原本的指令流（含重采样 + heading_command）
  --zero-action         不跑策略、动作恒 0 = 纯 PD 保持默认姿态（分离"机构/PD 因素"）
  --freeze-hip          把髋关节动作强制归零（判断外八是不是承重的拐杖）
  --terms               额外打印各奖励分项的每秒速率

输出指标: 实际速度与跟踪率 / 机身 z / reset 比例 / pitch·roll / 前进位移 /
          左足与右足几何(髋外张角度、足端机体系横向偏移、足端偏航、站姿宽度) /
          站立抖动(动作变化率 |Δa|、足端离地时间占比、抬脚高度、机身漂移) /
          trot 相位相关矩阵(对角应 ≈+1、左右应 ≈−1)。
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'legged_gym'))

import isaacgym  # noqa: F401  （必须在 torch 之前 import）

# 注意: 必须在模块层 import，且 envs 先于 utils（legged_gym 内部有循环 import）
from legged_gym.envs import *          # noqa: F401,F403
from legged_gym.utils import task_registry  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='LgDog1_flat')
    ap.add_argument('--run', required=True, help='logs/<experiment>/ 下的运行目录名')
    ap.add_argument('--checkpoint', type=int, required=True)
    ap.add_argument('--x_vel', type=float, default=1.0)
    ap.add_argument('--y_vel', type=float, default=0.0)
    ap.add_argument('--yaw_vel', type=float, default=0.0)
    ap.add_argument('--height', type=float, default=None,
                    help='目标机身高度指令 [m]（身高指令通道，V32 起有效；缺省=环境名义值 0.28）')
    ap.add_argument('--num_envs', type=int, default=32)
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--terrain', choices=['plane', 'terrain'], default='plane')
    ap.add_argument('--dr', action='store_true',
                    help='保留域随机化（默认关闭：32 个环境=同一台标称机器人，指标干净可比）')
    ap.add_argument('--freeze-hip', action='store_true',
                    help='把髋关节动作强制归零（髋锁在默认角）——用来判断"外八是不是承重的拐杖"：'
                         '锁住后还能走/不走、会不会摔、抬头是否改善')
    ap.add_argument('--heading-hold', action='store_true',
                    help='启用【部署侧朝向保持层】: 操作杆 yaw 输入积分成朝向目标，'
                         '下发给策略 wz = 0.5×朝向误差(clip ±2)（与训练一致）。用来验证'
                         '"直行不画圈/站立不自转"这一层是否真的有效。')
    ap.add_argument('--free-cmd', action='store_true',
                    help='不覆盖指令，用配置原本的指令流（含 resampling_time 重采样 + heading_command）'
                         '——用于检查"训练里到底有多少时间处于 零/慢指令 状态"')
    ap.add_argument('--terms', action='store_true',
                    help='额外打印【各奖励分项】的每秒速率（诊断"零指令下什么在推动运动"）')
    ap.add_argument('--zero-action', action='store_true',
                    help='不跑策略，动作恒为 0（纯 PD 保持默认姿态）——用来分离'
                         '『机身前高后低』是机构/PD 本身造成的，还是策略学出来的')
    # legged_gym 的 make_env/make_alg_runner 会读这些属性，补齐以免 AttributeError
    ap.add_argument('--headless', action='store_true', default=True)
    ap.add_argument('--resume', action='store_true', default=True)
    ap.add_argument('--experiment_name', default=None)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--load_run', default=None)
    ap.add_argument('--horovod', action='store_true', default=False)
    ap.add_argument('--rl_device', default='cuda:0')
    ap.add_argument('--max_iterations', type=int, default=None)
    ap.add_argument('--run_name', default=None)
    # parse_sim_params / make_env 需要的 Isaac 参数（这里给固定默认值，无需命令行传）
    ap.add_argument('--physics_engine', default='physx')
    ap.add_argument('--use_gpu', action='store_true', default=True)
    ap.add_argument('--use_gpu_pipeline', action='store_true', default=True)
    ap.add_argument('--subscenes', type=int, default=0)
    ap.add_argument('--num_threads', type=int, default=0)
    ap.add_argument('--slices', type=int, default=0)
    args = ap.parse_args()
    args.load_run = args.run
    args.checkpoint = args.checkpoint
    args.sim_device = args.device
    import isaacgym.gymapi as _gymapi
    args.physics_engine = _gymapi.SIM_PHYSX if args.physics_engine == 'physx' else _gymapi.SIM_FLEX

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.noise.add_noise = False
    # 默认关掉【全部域随机化】→ 32 个环境是同一台"标称机器人"，指标可比、可复现。
    # 想看策略在随机化下的鲁棒性就加 --dr（此时各环境是不同机器人，均值含义变差）。
    if not args.dr:
        _d = env_cfg.domain_rand
        for _name in ['randomize_payload_mass', 'randomize_com_displacement', 'randomize_link_mass',
                      'randomize_friction', 'randomize_restitution', 'randomize_motor_strength',
                      'randomize_kp', 'randomize_kd', 'randomize_initial_joint_pos',
                      'disturbance', 'push_robots', 'delay']:
            if hasattr(_d, _name):
                setattr(_d, _name, False)
    if not args.free_cmd:
        env_cfg.commands.heading_command = False
    if args.terrain == 'plane':
        env_cfg.terrain.mesh_type = 'plane'
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        # 注意: 不改 measure_heights —— critic 观测维度 238 必须和 ckpt 一致

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = args.x_vel
    env.commands[:, 1] = args.y_vel
    env.commands[:, 2] = args.yaw_vel

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    n_steps = int(args.seconds / env.dt)
    warm = int(2.0 / env.dt)          # 前 2s 起步过渡不计入统计
    vx_all, vy_all, z_all, done_all, rew_track, dist = [], [], [], [], [], []
    pitch_all, roll_all = [], []
    x_hist, dp_hist = [], []
    hip_hist, lat_hist, fyaw_hist, wid_hist = [], [], [], []
    # 站立抖动（零指令）指标
    act_hist, dact_hist, footz_hist, contact_hist, posxy_hist = [], [], [], [], []
    gate_hist, cmdxy_hist, cmdvy_hist, cmdyaw_hist, gateval_hist, lpf_hist = [], [], [], [], [], []
    swingz_hist, yawrate_hist, swingflag_hist = [], [], []
    heading_hist, yawcmd_hist = [], []      # 【自转体检】累积朝向漂移 + 下发的 yaw 指令
    tgt_hist = []                           # 【关节限位体检】PD 目标（真机安全层看的就是它）
    prev_act = None
    x0 = env.root_states[:, 0].clone()
    y0 = env.root_states[:, 1].clone()
    y_hist = []

    def _qrot_inv(q, v):
        """把世界系向量 v 转到机体系（legged_gym 同实现，q 为 xyzw）。"""
        w = q[:, 3:4]
        qv = q[:, 0:3]
        a = v * (2 * w**2 - 1)
        b = torch.cross(qv, v, dim=1) * w * 2
        c = qv * (qv * v).sum(dim=1, keepdim=True) * 2
        return a - b + c

    def _yaw(q):
        """四元数(xyzw) → 世界系 yaw。"""
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    with torch.no_grad():
        for i in range(n_steps):
            actions = (torch.zeros(env.num_envs, env.num_actions, device=env.device)
                       if args.zero_action else policy(obs.detach()))
            if args.freeze_hip:
                actions = actions.clone()
                actions[:, [0, 3, 6, 9]] = 0.0      # Isaac 顺序: 每腿第 0 个关节 = hip
            if not args.free_cmd:
                env.commands[:, 0] = args.x_vel
                env.commands[:, 1] = args.y_vel
                env.commands[:, 2] = args.yaw_vel
                if args.heading_hold:
                    # 【部署侧朝向保持层】操作杆 yaw 输入=期望转向速率 ⇒ 积分成朝向目标；
                    # 下发给策略的是 0.5×朝向误差（与训练 _post_physics_step_callback 同式）。
                    _q = env.root_states[:, 3:7]
                    _x, _y, _z, _w = _q[:, 0], _q[:, 1], _q[:, 2], _q[:, 3]
                    _hdg = torch.atan2(2 * (_w * _z + _x * _y), 1 - 2 * (_y * _y + _z * _z))
                    # 【两种模式分开】有转向指令→直接下发并把目标同步到当前朝向；
                    # 无指令→保持模式（下发 0.5×朝向误差）。第一版"积分目标"会让转弯卡死。
                    if abs(args.yaw_vel) > 1e-3:
                        hdg_target = _hdg.clone()
                    else:
                        if i == 0:
                            hdg_target = _hdg.clone()
                        _err = hdg_target - _hdg
                        _err = torch.atan2(torch.sin(_err), torch.cos(_err))
                        env.commands[:, 2] = torch.clamp(0.5 * _err, -2.0, 2.0)
                if args.height is not None and env.cfg.commands.num_commands >= 5:
                    env.commands[:, 4] = args.height
            obs, _, rews, dones, _infos, _tids, _tobs = env.step(actions.detach())
            # 站立抖动：动作变化率 / 足端离地 / 机身横向漂移（与统计段无关的原始量）
            if i >= warm:
                _rbs_j = env.rigid_body_states.view(env.num_envs, env.num_bodies, 13)
                act_hist.append(actions.detach().abs().mean().item())
                if prev_act is not None:
                    dact_hist.append((actions.detach() - prev_act).abs().mean().item())
                prev_act = actions.detach().clone()
                footz_hist.append(_rbs_j[:, env.feet_indices, 2].clone())
                contact_hist.append((torch.norm(env.contact_forces[:, env.feet_indices, :], dim=-1) > 1.0).clone())
                posxy_hist.append(env.root_states[:, 0:2].clone())
                yawrate_hist.append(env.base_ang_vel[:, 2].clone())
                _qq = env.root_states[:, 3:7]
                _ax, _ay, _az, _aw = _qq[:, 0], _qq[:, 1], _qq[:, 2], _qq[:, 3]
                heading_hist.append(torch.atan2(2 * (_aw * _az + _ax * _ay),
                                                1 - 2 * (_ay * _ay + _az * _az)).clone())
                yawcmd_hist.append(env.commands[:, 2].clone())
                gate_hist.append((torch.norm(env.commands[:, :3], dim=1) < 0.4).float())
                cmdxy_hist.append(torch.norm(env.commands[:, :2], dim=1))
                if hasattr(env, '_hip_constraint_gate'):
                    gateval_hist.append(env._hip_constraint_gate().clone())
                    lpf_hist.append(env._hip_gate_lpf.clone())
                cmdvy_hist.append(env.commands[:, 1].abs().clone())
                cmdyaw_hist.append(env.commands[:, 2].abs().clone())
            if i >= warm:
                vx_all.append(env.base_lin_vel[:, 0].mean().item())
                vy_all.append(env.base_lin_vel[:, 1].mean().item())
                z_all.append(env.root_states[:, 2].mean().item())
                done_all.append(float(dones.float().mean().item()))
                rew_track.append(float(rews.mean().item()))
                x_hist.append(env.root_states[:, 0].clone())
                y_hist.append(env.root_states[:, 1].clone())
                dp_hist.append(env.dof_pos.clone())
                tgt_hist.append(env.joint_pos_target.clone())
                # 机身姿态：root_states 四元数为 xyzw 序（Isaac 约定）
                _q = env.root_states[:, 3:7]
                _x, _y, _z, _w = _q[:, 0], _q[:, 1], _q[:, 2], _q[:, 3]
                _fz = 2 * (_x * _z - _w * _y)                 # 机头方向 z 分量 = R[2,0]
                _fx = 1 - 2 * (_y * _y + _z * _z)             # R[0,0]
                _fy = 2 * (_x * _y + _w * _z)                 # R[1,0]
                pitch_all.append(torch.atan2(_fz, torch.sqrt(_fx**2 + _fy**2)).mean().item())
                _r21 = 2 * (_y * _z + _w * _x)
                _r22 = 1 - 2 * (_x * _x + _y * _y)
                roll_all.append(torch.atan2(_r21, _r22).mean().item())
                # ---- 外八 / 站姿几何：髋关节 abduct + 足端机体系横向偏移 + 足端朝向 ----
                _dp = env.dof_pos
                hip_hist.append(torch.stack([_dp[:, 0], _dp[:, 3], _dp[:, 6], _dp[:, 9]], dim=1))
                _rbs = env.rigid_body_states.view(env.num_envs, env.num_bodies, 13)
                _fp = _rbs[:, env.feet_indices, 0:3]           # 世界系足端位置
                _fq = _rbs[:, env.feet_indices, 3:7]           # 足端朝向（xyzw）
                _body_p = env.root_states[:, 0:3].unsqueeze(1)
                _body_q = env.root_states[:, 3:7]
                _rel = _qrot_inv(_body_q.unsqueeze(1).expand(-1, _fp.shape[1], -1).reshape(-1, 4),
                                 (_fp - _body_p).reshape(-1, 3)).view(env.num_envs, -1, 3)
                lat_hist.append(_rel[:, :, 1])                 # 机体系 y = 横向（外八→|y| 变大）
                # 摆动相足端【机身系 z】（与 _reward_foot_clearance 同口径，便于和目标比）
                swingz_hist.append(_rel[:, :, 2].clone())
                swingflag_hist.append((~contact_hist[-1]).float())
                _fy = _yaw(_fq.reshape(-1, 4)).view(env.num_envs, -1) - _yaw(_body_q).unsqueeze(1)
                _fy = torch.atan2(torch.sin(_fy), torch.cos(_fy))   # 归一化到 ±π
                fyaw_hist.append(_fy)                          # 足端相对机身航向的偏航 = 外八角
                wid_hist.append((_rel[:, 0, 1] - _rel[:, 1, 1]).abs())   # 左右足横向间距
    n = max(len(vx_all), 1)
    print('\n===== Isaac 侧评估: %s / model_%d =====' % (args.run, args.checkpoint))
    print('  地形=%s  命令 vx=%.2f m/s  并行环境=%d  评估 %.0fs (统计后 %.0fs)'
          % (args.terrain, args.x_vel, args.num_envs, args.seconds, args.seconds - 2.0))
    print('  实际 base_lin_vel_x 均值 = %+.3f m/s  (命令 %.2f → 跟踪率 %.1f%%)'
          % (sum(vx_all) / n, args.x_vel,
             100 * (sum(vx_all) / n) / args.x_vel if args.x_vel else 0.0))
    if vy_all and (abs(args.y_vel) > 1e-6 or abs(sum(vy_all) / n) > 0.02):
        _vy = sum(vy_all) / n
        print('  实际 base_lin_vel_y 均值 = %+.3f m/s  (命令 %.2f → 跟踪率 %.1f%%)'
              % (_vy, args.y_vel, 100 * _vy / args.y_vel if args.y_vel else 0.0))
    _h_extra = ''
    if args.height is not None and getattr(env, 'num_commands', 4) >= 5:
        _h_extra = '  (身高指令 %.2f m ⇒ 误差 %+.3f m)' % (args.height, sum(z_all) / n - args.height)
    print('  机身高度 z 均值 = %.3f m%s' % (sum(z_all) / n, _h_extra))
    print('  每步 reset 比例 = %.4f  (≈0 表示几乎不摔倒; >0.01 表示频繁终止)' % (sum(done_all) / n))
    print('  全部奖励项均值 = %.3f' % (sum(rew_track) / n))
    # ---- 站立抖动汇总（零指令时最关键；有速度指令时也可看）----
    if act_hist:
        import numpy as _np
        dact = _np.array(dact_hist)
        actm = _np.array(act_hist)
        fz = _np.stack([f.cpu().numpy() for f in footz_hist])            # (steps, envs, 4)
        ct = _np.stack([c.cpu().numpy() for c in contact_hist])          # (steps, envs, 4)
        pxy = _np.stack([p.cpu().numpy() for p in posxy_hist])           # (steps, envs, 2)
        legs = ['FL', 'FR', 'RL', 'RR']
        air = 1.0 - ct.mean(axis=(0, 1))                                 # 每足离地时间占比
        air_env = 1.0 - ct.mean(axis=(0, 2))                             # 每环境（四足平均）离地占比
        frac_bad = float((air_env > 0.10).mean())
        step = _np.linalg.norm(_np.diff(pxy, axis=0), axis=2).sum(axis=0)  # 每环境路径长
        drift = _np.linalg.norm(pxy[-1] - pxy[0], axis=1)                # 每环境净漂移
        print('  站立/移动抖动: |a| 均值 %.2f   动作变化率 |Δa| 均值 %.3f（越小越稳）'
              % (actm.mean(), dact.mean()))
        print('  足端离地时间占比: ' + '  '.join('%s %.1f%%' % (legs[k], 100 * air[k]) for k in range(4))
              + '   ← 站立时应≈0%（≈50% 是 trot 走路）')
        print('  足端抬升高度 均值/最大: %.3f / %.3f m' % (fz.mean(), fz.max()))
        print('  环境间分布: 离地占比 中位 %.1f%% (最小 %.1f%% / 最大 %.1f%%)  离地>10%%的环境占比 %.0f%%'
              % (100*_np.median(air_env), 100*air_env.min(), 100*air_env.max(), 100*frac_bad))
        print('  机身漂移: 平均净位移 %.3f m  平均路径长 %.3f m  (每 %.0fs)'
              % (drift.mean(), step.mean(), n * env.dt))
        if gate_hist:
            g = _np.concatenate([x.cpu().numpy() for x in gate_hist])
            c = _np.concatenate([x.cpu().numpy() for x in cmdxy_hist])
            print('  指令统计: ‖cmd[:3]‖<0.4 占空比 %.1f%%   ‖cmd_xy‖ 均值 %.3f m/s'
                  % (100 * g.mean(), c.mean()))
            _vy = _np.concatenate([x.cpu().numpy() for x in cmdvy_hist])
            _wz = _np.concatenate([x.cpu().numpy() for x in cmdyaw_hist])
            print('  指令侧: |cmd_vy| 中位 %.3f |cmd_yaw| 中位 %.3f  ‖cmd_xy‖ 中位 %.3f'
                  % (_np.median(_vy), _np.median(_wz), _np.median(_np.concatenate([x.cpu().numpy() for x in cmdxy_hist]))))
        if gateval_hist:
            _g = _np.concatenate([x.cpu().numpy() for x in gateval_hist])
            if lpf_hist and hasattr(env, '_hip_gate_lpf'):
                _l = _np.concatenate([x.cpu().numpy() for x in lpf_hist])
                print('  门控输入(低通后): |v_y| 中位 %.3f p90 %.3f   |ω_z| 中位 %.3f p90 %.3f  (阈值 %.2f / %.2f)'
                      % (_np.median(_np.abs(_l[:, 0])), _np.percentile(_np.abs(_l[:, 0]), 90),
                         _np.median(_np.abs(_l[:, 1])), _np.percentile(_np.abs(_l[:, 1]), 90),
                         env.cfg.rewards.hip_abduction_pure_vy, env.cfg.rewards.hip_abduction_pure_wz))
            print('  髋部约束门控(实测口径): 均值 %.3f   ≥0.9(约束基本满强度) 占 %.1f%%   =0(完全放开) 占 %.1f%%'
                  % (_g.mean(), 100 * (_g >= 0.9).mean(), 100 * (_g <= 0.001).mean()))
    if pitch_all:
        import numpy as _np
        p = _np.degrees(_np.array(pitch_all))
        r = _np.degrees(_np.array(roll_all))
        print('  机身 pitch 前高后低: 均值 %+.2f°  中位 %+.2f°  p5 %+.2f°  p95 %+.2f°  std %.2f°'
              % (p.mean(), _np.median(p), _np.percentile(p, 5), _np.percentile(p, 95), p.std()))
        print('  机身 roll  侧倾    : 均值 %+.2f°  中位 %+.2f°  std %.2f°'
              % (r.mean(), _np.median(r), r.std()))
    # 位移：不依赖任何速度传感器，是"到底有没有走出去"的硬证据
    if x_hist:
        x_last = x_hist[-1]
        disp = (x_last - x0)
        dur = n * env.dt
        print('  统计段内前进位移 = 均值 %+.2f m (%.2f m/s)  最小 %+.2f  最大 %+.2f'
              % (disp.mean().item(), disp.mean().item() / dur, disp.min().item(), disp.max().item()))
        if len(y_hist):
            _ylast = y_hist[-1]
            _dy = (_ylast - y0)
            print('  统计段内侧向位移 = 均值 %+.2f m (%.2f m/s)  ← 侧移指令时应≈%.2f m/s'
                  % (_dy.mean().item(), _dy.mean().item() / dur, args.y_vel))
    # ---- 各奖励分项（每秒速率，与训练日志同口径）----
    if args.terms:
        dur = max(n * env.dt, 1e-6)
        rows = []
        for k, v in env.episode_sums.items():
            rows.append((k, float(v.mean().item()) / dur))
        rows.sort(key=lambda x: x[1])
        print('  奖励分项（每秒速率，越小=罚得越狠）:')
        for k, val in rows:
            if abs(val) > 1e-4:
                print('    %-26s %+8.4f' % (k, val))
    # ---- 摆动腿净空（每次摆动的峰值高度，与 clearance_height_target 同口径）----
    if swingz_hist:
        import numpy as _np
        sz = _np.stack([x.cpu().numpy() for x in swingz_hist])        # (steps, envs, 4) 机身系 z
        sw = _np.stack([x.cpu().numpy() for x in swingflag_hist])     # (steps, envs, 4) 1=摆动
        peaks, mean_swing = [], []
        for e in range(sz.shape[1]):
            for f in range(sz.shape[2]):
                s_f = sw[:, e, f]
                z_f = sz[:, e, f]
                # 逐次摆动事件: 上升沿→下降沿之间取峰值
                idx = _np.where(_np.diff(s_f.astype(int)) == 1)[0]
                end = _np.where(_np.diff(s_f.astype(int)) == -1)[0]
                for a in idx:
                    b = end[end > a]
                    if len(b) == 0:
                        continue
                    seg = z_f[a:b[0] + 1]
                    if len(seg) >= 3:
                        peaks.append(seg.max())
                if s_f.sum() > 5:
                    mean_swing.append(z_f[s_f > 0].mean())
        # ★标定用（2026-10-04 口径重做）: 奖励已改为【真实离地高度】的二次亏欠
        #   raw = Σ_摆动 (relu(clearance_ground_target − (世界系足底 z − 足下地形高)) × 10)²
        #   ⇒ 这里用世界系足底 z（plane 地形地面=0）与配置里的目标复现同一公式。
        _gtgt = float(getattr(env.cfg.rewards, 'clearance_ground_target', 0.08))
        _hgt = _np.stack([h.cpu().numpy() for h in footz_hist])            # (steps, envs, 4) 世界系足底 z
        _clear = _hgt                                                      # plane: 地面 = 0
        _quad_raw = _np.square(_np.maximum(_gtgt - _clear, 0.0) * 10.0) * sw
        _quad_per_step = _quad_raw.sum(axis=2).mean()
        print('  [标定] 摆动相【真实离地高度】目标 %.3f m: 均值 %.3f m  峰值(p95) %.3f m  '
              '高于目标的时间占比 %.1f%%'
              % (_gtgt, _clear[sw > 0].mean(), _np.percentile(_clear[sw > 0], 95),
                 100 * (_clear[sw > 0] > _gtgt).mean()))
        print('  [标定] 新口径原始值 = %.4f /步 ⇒ 权重 w 时 = %.3f×w /秒（含 2 摆动足）'
              % (_quad_per_step, _quad_per_step))
        _tgt = env.cfg.rewards.clearance_height_target
        _relu_raw = _np.maximum(_tgt - sz, 0.0) * sw                       # (steps, envs, 4) 旧口径
        _relu_per_step = _relu_raw.sum(axis=2).mean()
        print('  [标定] （旧·机身系口径参照）relu 原始值 = %.4f /步、有摆动足低于旧目标的步数占比 %.1f%%'
              % (_relu_per_step, 100 * (_relu_raw.sum(axis=2) > 1e-6).mean()))
        if peaks:
            pk = _np.array(peaks)
            print('  摆动腿净空(机身系 z, 目标 %.3f m): 峰值 中位 %.3f  p95 %.3f  均值 %.3f  m   摆动事件 %d 次'
                  % (env.cfg.rewards.clearance_height_target, _np.median(pk), _np.percentile(pk, 95), pk.mean(), len(pk)))
            print('  摆动相平均高度 %.3f m  ⇒ 峰值比"目标"高 %.3f m / 比标称站姿高 %.3f m'
                  % (_np.mean(mean_swing), _np.median(pk) - env.cfg.rewards.clearance_height_target,
                     _np.median(pk) + 0.13))
    # ---- 偏航角速度跟踪 ----
    if yawrate_hist:
        import numpy as _np
        yr = _np.concatenate([x.cpu().numpy() for x in yawrate_hist])
        if abs(args.yaw_vel) > 1e-6:
            err = yr - args.yaw_vel
            print('  偏航跟踪: 命令 %.2f rad/s → 实测 均值 %+.3f  中位 %+.3f  |误差| 均值 %.3f  跟踪率 %.1f%%'
                  % (args.yaw_vel, yr.mean(), _np.median(yr), _np.abs(err).mean(),
                     100 * min(1.0, max(0.0, 1 - _np.abs(err).mean() / abs(args.yaw_vel)))))
        else:
            # 【2026-10-06 新增·自转体检】yaw 指令=0 时最该看的量: 它到底有没有偷偷转。
            #   为什么加: 用户 MuJoCo 实测反馈"原地旋转"（V38 直行时 |ω_z|=0.311 rad/s=17.8°/s），
            #   而此前评测只在 --yaw_vel≠0 时打印偏航，零指令自转从未被量过（盲区）。
            _dps = _np.degrees(_np.abs(yr).mean())
            print('  【自转体检】yaw 指令=0: 偏航率 均值 %+.3f 中位 %+.3f  |ω_z| 均值 %.3f rad/s (%.1f °/s)'
                  '  ⇒ 30 秒自转 %.0f°' % (yr.mean(), _np.median(yr), _np.abs(yr).mean(), _dps, _dps * 30))
            print('             最大 |ω_z| %.3f rad/s   超 0.2 rad/s 的时间占比 %.1f%%'
                  % (_np.abs(yr).max(), 100 * (_np.abs(yr) > 0.2).mean()))
            if heading_hist:
                hh = _np.stack([h.cpu().numpy() for h in heading_hist])      # (T, N)
                d = hh[-1] - hh[0]
                d = (d + _np.pi) % (2 * _np.pi) - _np.pi
                print('             【累积朝向漂移】净转 %+.1f°（均值）  std %.1f°   偏置 ω̄=%+.3f rad/s'
                      % (_np.degrees(d.mean()), _np.degrees(d).std(), yr.mean()))
            if yawcmd_hist:
                yc = _np.concatenate([c.cpu().numpy() for c in yawcmd_hist])
                print('             下发给策略的 yaw 指令: |均值| %.3f rad/s  (朝向保持层工作时非 0)'
                      % _np.abs(yc).mean())
    # 步态相位：trot 应为【对角同相 / 左右反相】。Isaac 顺序 = [FL,FR,RL,RR] x [hip,thigh,calf]
    if len(dp_hist) > 50:
        import numpy as _np
        dp = torch.stack(dp_hist).cpu().numpy()        # (steps, envs, 12)
        thigh = dp[:, :, 1::3].reshape(-1, 4)          # 拼成 (steps*envs, 4)
        dark = thigh - thigh.mean(0, keepdims=True)
        corr = (dark.T @ dark) / _np.sqrt(((dark**2).sum(0)[:, None]) * ((dark**2).sum(0)[None, :]) + 1e-12)
        legs = ['FL', 'FR', 'RL', 'RR']
        print('  thigh 相位相关矩阵（Isaac 顺序）:')
        print('        ' + '  '.join('%6s' % l for l in legs))
        for i, l in enumerate(legs):
            print('  %-4s  ' % l + '  '.join('%6.2f' % corr[i, j] for j in range(4)))
        print('  对角(FL,RR)=%.2f (FR,RL)=%.2f  ← trot 应 ≈ +1 ; 左右(FL,FR)=%.2f (RL,RR)=%.2f ← 应 ≈ -1'
              % (corr[0, 3], corr[1, 2], corr[0, 1], corr[2, 3]))
    # ---- 外八/站姿几何汇总 ----
    if hip_hist:
        import numpy as _np
        hip = _np.stack([h.cpu().numpy() for h in hip_hist])          # (steps, envs, 4) Isaac=FL,FR,RL,RR
        lat = _np.stack([l.cpu().numpy() for l in lat_hist])          # (steps, envs, 4)
        fyw = _np.stack([f.cpu().numpy() for f in fyaw_hist])
        wid = _np.stack([w.cpu().numpy() for w in wid_hist])
        legs = ['FL', 'FR', 'RL', 'RR']
        print('  髋关节(abduction)均值(度): ' + '  '.join(
            '%s %+.1f' % (legs[i], _np.degrees(hip[:, :, i].mean())) for i in range(4)))
        print('  足端机体系横向偏移(米)  : ' + '  '.join(
            '%s %+.3f' % (legs[i], lat[:, :, i].mean()) for i in range(4)))
        print('  足端相对机身偏航(外八角): ' + '  '.join(
            '%s %+.1f°' % (legs[i], _np.degrees(fyw[:, :, i].mean())) for i in range(4)))
        print('  左右足横向间距: 均值 %.3f m  (零动作标称站姿实测 ≈ 0.308 m，见 --zero-action)' % wid.mean())

    # ---- 【关节限位体检·2026-10-07 新增】真机 BRAKE 就来自这里 ----
    #   为什么加: 真机日志出现 ① safety『连续 3 帧目标超限 FL_calf:+2.314』② guard『FR_calf=1.195
    #   越出限位[-0.85,1.00]』。V41 为此加了 dof_pos_limits（进软限位外沿罚），但训练侧该项只有
    #   -0.002/s（占总奖励 0.0%）⇒ 必须能【独立量出来】才判断得出惩罚到底有没有用。
    #   两个口径都要看（它们不是一回事）:
    #     · 位置口径: 实际关节角超出软限位（0.85×半行程，与训练 _reward_dof_pos_limits 同基准）的时间占比
    #     · 目标口径: PD 目标超出 URDF 硬限位的深度 —— 真机安全层夹取/刹车看的是【目标】，不是位置
    if len(dp_hist) > 50:
        import numpy as _np
        URDF_LO = _np.array([-1.2, -1.5, -0.85] * 4)
        URDF_HI = _np.array([1.2, 1.5, 0.85] * 4)
        SOFT = float(getattr(env.cfg.rewards, 'soft_dof_pos_limit', 0.85))
        _m = (URDF_LO + URDF_HI) / 2
        _r = URDF_HI - URDF_LO
        SOFT_LO, SOFT_HI = _m - 0.5 * _r * SOFT, _m + 0.5 * _r * SOFT
        dp = torch.stack(dp_hist).cpu().numpy().reshape(-1, 12)
        names = ['%s_%s' % (l, j) for l in ['FL', 'FR', 'RL', 'RR'] for j in ['hip', 'thigh', 'calf']]
        pos_rate = ((dp < SOFT_LO) | (dp > SOFT_HI)).mean(0)
        hard_rate = ((dp < URDF_LO) | (dp > URDF_HI)).mean(0)
        print('  【关节限位体检】位置口径（软限位 ±%.4f/±%.4f/±%.4f，超出的时间占比）:'
              % (SOFT_HI[0], SOFT_HI[1], SOFT_HI[2]))
        print('    ' + '  '.join('%s %.2f%%' % (names[i], 100 * pos_rate[i]) for i in range(12)))
        print('    顶到 URDF 硬限位的时间占比: 最大 %s %.2f%%（其余 <%.2f%%）'
              % (names[int(_np.argmax(hard_rate))], 100 * hard_rate.max(),
                 100 * _np.sort(hard_rate)[-2] if len(hard_rate) > 1 else 0.0))
        if tgt_hist:
            tgt = torch.stack(tgt_hist).cpu().numpy().reshape(-1, 12)
            over = (tgt > URDF_HI)
            under = (tgt < URDF_LO)
            depth = _np.maximum(tgt - URDF_HI, URDF_LO - tgt)
            depth = _np.maximum(depth, 0)
            i_worst = int(_np.argmax(depth.max(0)))
            print('  【关节目标体检】目标口径（真机安全层夹取的依据）: 超限时间占比 最大 %s %.2f%%  '
                  '超限深度 最大 %+.3f rad（均值 %.3f）'
                  % (names[i_worst], 100 * (over | under).mean(0)[i_worst],
                     depth.max(0)[i_worst], depth[:, i_worst].mean()))
            print('    各关节目标超限时间占比: ' + '  '.join(
                '%s %.1f%%' % (names[i], 100 * (over | under).mean(0)[i]) for i in range(12)))


if __name__ == '__main__':
    main()
