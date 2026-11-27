# -*- coding: utf-8 -*-
"""swing_profile.py — 摆动相【离地高度剖面】审计：回答"抬腿看着高、实际离地不长"。

为什么要它（2026-10-04 用户反馈『虽然抬腿很高和长但是实际在地下的距离不长』）:
  · 训练奖励 _reward_foot_clearance(relu_swing) 的口径是【机身系】足端 z 相对 clearance_height_target
    （-0.13m）的亏欠量。机身系 ≠ 离地高度：机身 pitch 一变（V36@6000 直行 +14.7°）、
    机身高度一变（0.30m），同一"机身系 z"对应的真实离地高度会差好几厘米。
  · 而人眼看到的是【世界系离地高度】随时间的过程 —— 峰值高不等于过程高：
    脚可能只在峰值一瞬抬高，其余时间仍贴地擦行（这正是用户描述的现象）。
  ⇒ 本工具按【摆动事件】把两种口径都量出来: 峰值 / 均值 / 25-50-75% 剖面 /
     高于各绝对离地阈值的【时间占比】/ 摆动水平位移（步幅）/ 触地前 20% 的高度。

用法（仓库根目录；显卡需空闲）:
  python tools/swing_profile.py --run V35 --checkpoint 13600 --x_vel 1.0
  python tools/swing_profile.py --run Oct04_13-45-25_LgDog1_himloco --checkpoint 6000 --x_vel 1.0
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'legged_gym'))

import isaacgym  # noqa: F401  必须在 torch 之前

from legged_gym.envs import *          # noqa: F401,F403
from legged_gym.utils import task_registry  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='LgDog1_flat')
    ap.add_argument('--run', required=True)
    ap.add_argument('--checkpoint', type=int, required=True)
    ap.add_argument('--x_vel', type=float, default=1.0)
    ap.add_argument('--yaw_vel', type=float, default=0.0)
    ap.add_argument('--height', type=float, default=0.30)
    ap.add_argument('--num_envs', type=int, default=32)
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--headless', action='store_true', default=True)
    ap.add_argument('--max_iterations', type=int, default=None)
    ap.add_argument('--resume', action='store_true', default=True)
    ap.add_argument('--experiment_name', default=None)
    ap.add_argument('--run_name', default=None)
    ap.add_argument('--horovod', action='store_true', default=False)
    ap.add_argument('--rl_device', default='cuda:0')
    # Isaac 参数（make_env 需要，给固定默认）
    ap.add_argument('--physics_engine', default='physx')
    ap.add_argument('--use_gpu', action='store_true', default=True)
    ap.add_argument('--use_gpu_pipeline', action='store_true', default=True)
    ap.add_argument('--subscenes', type=int, default=0)
    ap.add_argument('--num_threads', type=int, default=0)
    ap.add_argument('--slices', type=int, default=0)
    args = ap.parse_args()
    args.load_run = args.run
    args.sim_device = args.device
    import isaacgym.gymapi as _gymapi
    args.physics_engine = _gymapi.SIM_PHYSX

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.noise.add_noise = False
    for n in ['randomize_payload_mass', 'randomize_com_displacement', 'randomize_link_mass',
              'randomize_friction', 'randomize_restitution', 'randomize_motor_strength',
              'randomize_kp', 'randomize_kd', 'randomize_initial_joint_pos',
              'disturbance', 'push_robots', 'delay']:
        setattr(env_cfg.domain_rand, n, False)
    env_cfg.commands.heading_command = False
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = args.x_vel
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = args.yaw_vel
    env.commands[:, 4] = args.height

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    n_steps = int(args.seconds / env.dt)
    warm = int(2.0 / env.dt)
    target_body = env.cfg.rewards.clearance_height_target

    # 逐帧记录（每帧: 接触 / 世界系足底 z / 机身系足底 z / 足端世界 xy / 机身 z / pitch）
    rec = {'c': [], 'wz': [], 'bz': [], 'xy': [], 'rootz': [], 'pitch': []}
    with torch.no_grad():
        for i in range(n_steps):
            # ⚠️ 必须【每步】重写指令: 环境每 resampling_time(7s) 会重采样指令，
            #    只设一次的话后半段测的是随机指令（步态/速度全变）——首版就踩了这个坑。
            env.commands[:, 0] = args.x_vel
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = args.yaw_vel
            env.commands[:, 4] = args.height
            obs, _, _rews, _d, _i2, _t, _o = env.step(policy(obs.detach()).detach())
            if i < warm:
                continue
            rbs = env.rigid_body_states.view(env.num_envs, env.num_bodies, 13)
            fw = rbs[:, env.feet_indices, 0:3]                       # 世界系
            rel = fw - env.root_states[:, 0:3].unsqueeze(1)
            q = env.base_quat
            x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            # 机身系 z（与奖励同口径）
            fb = torch.zeros_like(rel)
            for j in range(rel.shape[1]):
                v = rel[:, j, :]
                qv = q[:, 0:3]
                a = v * (2 * w**2 - 1).unsqueeze(1)
                b = torch.cross(qv, v, dim=1) * w.unsqueeze(1) * 2
                c = qv * (qv * v).sum(dim=1, keepdim=True) * 2
                fb[:, j, :] = a - b + c
            rec['c'].append((torch.norm(env.contact_forces[:, env.feet_indices, :], dim=-1) > 1.0).cpu().numpy())
            rec['wz'].append(fw[:, :, 2].cpu().numpy())              # 世界系"离地高度"（地面=0）
            rec['bz'].append(fb[:, :, 2].cpu().numpy())              # 机身系 z（奖励口径）
            rec['xy'].append(fw[:, :, :2].cpu().numpy())
            rec['rootz'].append(env.root_states[:, 2].cpu().numpy())
            fz = 2 * (x * z - w * y)
            fx = 1 - 2 * (y * y + z * z)
            fy = 2 * (x * y + w * z)
            rec['pitch'].append(torch.atan2(fz, torch.sqrt(fx**2 + fy**2)).cpu().numpy())

    c = np.stack(rec['c'])          # (T, N, 4)
    wz = np.stack(rec['wz'])
    bz = np.stack(rec['bz'])
    xy = np.stack(rec['xy'])
    rootz = np.stack(rec['rootz'])
    pitch = np.stack(rec['pitch'])
    dt = env.dt

    print('\n===== 摆动相离地剖面: %s / model_%d  (vx=%.2f, %d 环境, %.0fs) ====='
          % (args.run, args.checkpoint, args.x_vel, args.num_envs, args.seconds - 2.0))
    print('  机身高度 z=%.3f m   pitch=%.2f°   奖励净空目标(离地)=%.3f m'
          % (rootz.mean(), np.degrees(pitch.mean()),
             getattr(env.cfg.rewards, 'clearance_ground_target', float('nan'))))
    # 足端 link 原点在【支撑期】的平均高度 ≈ "原点离足底"的偏置 ⇒ 换算成真实足底离地高度
    stance_z = wz[c]
    sole_off = float(np.median(stance_z)) if stance_z.size else 0.0
    print('  足端 link 原点·支撑期高度(中位) = %.3f m ⇒ 本工具下方"足底离地"= 原点高度 − 该偏置'
          % sole_off)

    legs = ['FL', 'FR', 'RL', 'RR']
    ev = {k: [] for k in ['dur', 'peak_w', 'mean_w', 'p25', 'p50', 'p75', 'pre20',
                          'frac_2', 'frac_5', 'frac_8', 'frac_10', 'frac_bz', 'travel',
                          'peak_bz', 'n']}
    # 支撑相【滑移】: 触地→离地期间足端在世界系走了多远（≈0=踩住不打滑；大=擦地空转不推进）
    slip, slip_dur = [], []
    per_leg = {k: {kk: [] for kk in ev} for k in range(4)}
    for e in range(c.shape[1]):
        for f in range(4):
            cf = c[:, e, f].astype(int)
            liftoff = np.where(np.diff(cf) == -1)[0] + 1        # 离地帧
            touch = np.where(np.diff(cf) == 1)[0] + 1           # 着地帧
            for a in liftoff:
                b = touch[touch > a]
                if len(b) == 0:
                    continue
                b = b[0]
                seg_w, seg_b = wz[a:b, e, f], bz[a:b, e, f]
                if len(seg_w) < 4:
                    continue
                n = len(seg_w)
                i25, i50, i75 = int(0.25 * n), int(0.5 * n), int(0.75 * n)
                pre = seg_w[max(0, int(0.8 * n)):]
                d = dict(dur=n * dt, peak_w=seg_w.max(), mean_w=seg_w.mean(),
                         p25=seg_w[i25], p50=seg_w[i50], p75=seg_w[i75], pre20=pre.mean(),
                         frac_2=(seg_w > 0.02).mean(), frac_5=(seg_w > 0.05).mean(),
                         frac_8=(seg_w > 0.08).mean(), frac_10=(seg_w > 0.10).mean(),
                         frac_bz=(seg_b > target_body).mean(), peak_bz=seg_b.max(),
                         travel=np.linalg.norm(xy[b - 1, e, f] - xy[a, e, f]), n=1)
                for k, v in d.items():
                    ev[k].append(v)
                    per_leg[f][k].append(v)

    for e in range(c.shape[1]):
        for f in range(4):
            cf = c[:, e, f].astype(int)
            liftoff = np.where(np.diff(cf) == -1)[0] + 1
            touch = np.where(np.diff(cf) == 1)[0] + 1
            for a in touch:
                b = liftoff[liftoff > a]
                if len(b) == 0:
                    continue
                b = b[0]
                if b - a >= 3:
                    slip.append(np.linalg.norm(xy[b - 1, e, f] - xy[a, e, f]))
                    slip_dur.append((b - a) * dt)

    def m(k):
        return float(np.mean(ev[k])) if ev[k] else float('nan')

    print('\n  ── 摆动事件 %d 次 ──' % len(ev['dur']))
    print('  摆动时长            : %.3f s  (步频 %.2f Hz)'
          % (m('dur'), 2.0 / m('dur') if m('dur') else 0))
    print('  摆动水平位移(步幅)  : %.3f m' % m('travel'))
    print('\n  【世界系离地高度】← 人眼看到的就是这个')
    print('    峰值      : %.3f m' % m('peak_w'))
    print('    全程均值  : %.3f m   ← 峰值高但均值低 = 只在一瞬抬高' % m('mean_w'))
    print('    25%%/50%%/75%% 处: %.3f / %.3f / %.3f m' % (m('p25'), m('p50'), m('p75')))
    print('    触地前 20%% 均值 : %.3f m   ← 太小 ⇒ 落地前贴地擦行' % m('pre20'))
    print('    高于 2/5/8/10cm 的时间占比: %.1f%% / %.1f%% / %.1f%% / %.1f%%'
          % tuple(100 * m(k) for k in ['frac_2', 'frac_5', 'frac_8', 'frac_10']))
    print('    ⇩ 换算成【足底】离地（原点高度 − 支撑期偏置 %.3f m）: 峰值 %.3f m  均值 %.3f m'
          % (sole_off, m('peak_w') - sole_off, m('mean_w') - sole_off))
    if slip:
        print('\n  【支撑相滑移】足端触地→离地期间的世界系位移（踩住≈0；大=擦地空转不推进）')
        print('    每次滑移 中位 %.3f m  均值 %.3f m   支撑时长均值 %.3f s'
              % (np.median(slip), np.mean(slip), np.mean(slip_dur)))
        print('    ⇒ 每步"滑动距离/步幅" = %.2f   （越小越好；≈1 表示整步都在打滑）'
              % (np.mean(slip) / m('travel') if m('travel') else float('nan')))
    print('\n  【机身系 z】← 奖励 _reward_foot_clearance 的口径')
    print('    峰值      : %.3f m    高于目标(%.3f)的时间占比: %.1f%%'
          % (m('peak_bz'), target_body, 100 * m('frac_bz')))
    print('\n  分腿（世界系峰值/均值/步幅/时长）:')
    for i, lg in enumerate(legs):
        d = per_leg[i]
        if not d['dur']:
            continue
        print('    %-3s 峰值 %+.3f  均值 %+.3f  步幅 %.3f m  时长 %.3f s  >5cm 占比 %.1f%%'
              % (lg, np.mean(d['peak_w']), np.mean(d['mean_w']), np.mean(d['travel']),
                 np.mean(d['dur']), 100 * np.mean(d['frac_5'])))


if __name__ == '__main__':
    main()
