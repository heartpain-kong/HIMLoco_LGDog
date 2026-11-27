#!/usr/bin/env python3
"""measure_stand.py — 测策略的【零指令站立高度】，用于 sim2sim(Isaac) 与 MuJoCo 对照

为什么需要它：
  play.py 只打印奖励，不打印机身高度；而 base_height 奖励是 (h - target)²，
  反推 h 有【两个解】（target ± 误差），无法区分"站得高"与"蹲着走"——
  后者是 commands.md 记载过的已知失效模式。所以必须直接量 z。

输出：
  1. 世界系机身高度 root_states[:,2]
  2. 相对地形高度 _get_base_heights()（= base_height 奖励实际用的量）
  3. 直立程度（projected_gravity.z，-1 = 完全直立）、姿态倾角
  4. 各关节实际角度 vs 默认姿态（看是否蹲）

与 MuJoCo 侧对照：
  python sim2sim.py --no-render --x_vel 0.0    # 读它打印的 z=

用法：
  python measure_stand.py --task LgDog1 --resume --load_run <run> --checkpoint <n>
  python measure_stand.py --task LgDog1 --resume          # 取最新 checkpoint
可选：
  --steps 600        零指令下先跑这么多步再统计（默认 600 = 6 秒）
  --settle 200       统计前先忽略的步数（默认 200）
"""

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch


def measure(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 32)
    # 与 play.py 一致：关掉噪声/扰动，测【名义】表现
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_payload_mass = False
    # ★ play.py 漏关的那些随机化这里全关：站高必须量"同一个机器人"，
    #   否则每环境的质心/连杆质量/PD 都不同，统计出来的均值没有意义。
    for k in dir(env_cfg.domain_rand):
        if k.startswith("randomize_"):
            setattr(env_cfg.domain_rand, k, False)
    env_cfg.commands.heading_command = False

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if getattr(args, "policy", None):
        # 直接加载导出的 JIT 策略 —— 用于测【归档策略】，它的 run 目录可能已删除。
        # ⚠️ 此时 env 用的是【当前】config 而非该策略训练时的 config；奖励项不影响推理，
        #    所以只要 URDF/PD/action_scale 一致就仍可比（否则需在注释里标明差异）。
        policy = torch.jit.load(args.policy, map_location=env.device).eval()
        print(f"[策略] 直接加载 JIT: {args.policy}")
    else:
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy = ppo_runner.get_inference_policy(device=env.device)

    # 指令（默认零 = 站立；给 --cmd_x/--cmd_y 可测"走动时的机身高度"）
    env.commands[:, 0] = args.cmd_x
    env.commands[:, 1] = args.cmd_y
    env.commands[:, 2] = 0.0
    obs = env.get_observations()

    z_log, h_log, gz_log, lvz_log = [], [], [], []
    for i in range(args.steps):
        actions = policy(obs.detach())
        obs, _, _, _, _, _, _ = env.step(actions.detach())
        env.commands[:, 0] = args.cmd_x      # 每步都压住，防 resample
        env.commands[:, 1] = args.cmd_y
        env.commands[:, 2] = 0.0
        if i >= args.settle:
            z_log.append(env.root_states[:, 2].clone())
            h_log.append(env._get_base_heights().clone())
            gz_log.append(env.projected_gravity[:, 2].clone())
            lvz_log.append(env.root_states[:, 9].clone())      # base z 速度

    z = torch.stack(z_log)
    h = torch.stack(h_log)
    gz = torch.stack(gz_log)
    lvz = torch.stack(lvz_log)
    dof = env.dof_pos.clone()
    default = env.default_dof_pos.clone()

    def stat(t):
        return t.mean().item(), t.std().item(), t.min().item(), t.max().item()

    z_m, z_s, z_lo, z_hi = stat(z)
    h_m, h_s, h_lo, h_hi = stat(h)
    gz_m, gz_s, _, _ = stat(gz)
    # ★中位数 + 摔倒占比：均值会被"摔倒的环境"拉低，从而误判成"站得矮"。
    #   实测踩到: V16 末态 cmd_x=0 的 h 范围 [0.0918, 0.2609]（有环境已躺下），
    #   均值 0.2367 明显低于健康值，若不分清就会得出"站高退化"的错误结论。
    #   0.15m 作"已倒地"阈值: 正常站高 0.23~0.36、几何站高 0.354，倒地约 0.07~0.12。
    h_med = h.median().item()
    fallen = (h < 0.15).float().mean().item()
    tilt_deg = float(np.degrees(np.arccos(np.clip(-gz_m, -1.0, 1.0))))

    print("\n" + "=" * 68)
    print(f" 站立/行走机身高度测量  指令=({args.cmd_x:+.2f}, {args.cmd_y:+.2f})"
          f"  ({env_cfg.env.num_envs} 环境 × {len(z_log)} 步)")
    print("=" * 68)
    print(f"  世界系机身高度 z      mean {z_m:.4f}  std {z_s:.4f}"
          f"  [{z_lo:.4f}, {z_hi:.4f}]  m")
    print(f"  相对地形高度(base_height 用的量) 中位数 {h_med:.4f}  mean {h_m:.4f}  std {h_s:.4f}"
          f"  [{h_lo:.4f}, {h_hi:.4f}]  m")
    print(f"  已倒地比例(h<0.15m)                {fallen * 100:.1f}%"
          + ("   ← ⚠️ 有环境躺下，mean 被拉低，请看中位数" if fallen > 0.02 else ""))
    # ★V16: 站立/行走目标不同（base_height_target_stand / base_height_target），
    #   所以这里必须【按本次指令】选对应目标，否则零指令测量会拿行走目标(0.28)去算偏差，
    #   报出一个根本没在生效的偏差值。判定口径与 _reward_base_height 一致（‖cmd_xy‖<0.1）。
    _is_stand = (abs(args.cmd_x) ** 2 + abs(args.cmd_y) ** 2) ** 0.5 < 0.1
    _tgt = getattr(env_cfg.rewards, 'base_height_target_stand', None)
    if _is_stand and _tgt is not None:
        _tgt_name = 'base_height_target_stand(站立)'
    else:
        _tgt = env_cfg.rewards.base_height_target
        _tgt_name = 'base_height_target(行走)'
    print(f"  {_tgt_name:28s} {_tgt:.4f}  m"
          f"   → 偏差(中位数) {h_med - _tgt:+.4f} m"
          f"  (平方 {(h_med - _tgt) ** 2:.5f})")
    print(f"  机身竖直速度 v_z      mean {lvz.mean().item():+.4f}  std {lvz.std().item():.4f} m/s"
          f"   (std 大 = 上下弹跳)")
    print(f"  直立程度 projected_gravity.z  {gz_m:+.4f}  (完全直立 = -1.0000)"
          f"   → 倾角约 {tilt_deg:.2f}°")
    if args.geom_z:
        print(f"  默认姿态的几何站高          {args.geom_z:.4f}  m"
              f"   → 实测比几何站高 {z_m - args.geom_z:+.4f} m")

    names = env.dof_names if hasattr(env, "dof_names") else [f"dof{i}" for i in range(12)]
    print("\n  各关节实际角度 vs 默认姿态（只列偏差 > 0.05 rad 的）:")
    dev = (dof - default).mean(dim=0)
    for i, n in enumerate(names):
        if abs(dev[i]) > 0.05:
            print(f"    {n:<18} 实际 {dof[:, i].mean():+.3f}  "
                  f"默认 {default[0, i]:+.3f}  偏差 {dev[i]:+.3f} rad")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="测零指令站立高度")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--settle", type=int, default=200)
    ap.add_argument("--geom_z", type=float, default=0.0,
                    help="默认姿态的几何站高（仅用于打印对照）")
    ap.add_argument("--cmd_x", type=float, default=0.0, help="前进指令 [m/s]，0=站立")
    ap.add_argument("--cmd_y", type=float, default=0.0, help="侧向指令 [m/s]")
    ap.add_argument("--policy", default=None,
                    help="直接指定 JIT 策略文件（测归档策略用；与 --resume 二选一）")
    # ⚠️ 必须先把自定义参数从 argv 里摘掉：get_args() 用的是 isaacgym 的解析器，
    #    它遇到不认识的参数会直接报错退出（--steps 就这么被拒过一次）。
    known, rest = ap.parse_known_args()
    # ★ 强制 headless：本工具是测量用的，绝不需要窗口。
    #   不加固会【静默失败】——实测踩到过：make_env(headless=False) 会创建 viewer，
    #   而 base_task.py:124 在窗口被判定为已关闭时直接 sys.exit()，
    #   结果是进程【退出码 0、无任何输出、无 traceback】，测量结果凭空消失，
    #   白跑一次还以为策略有问题（V13 结算时就这么丢过一次数据）。
    if "--headless" not in rest:
        rest = rest + ["--headless"]
    sys.argv = [sys.argv[0]] + rest
    args = get_args()          # 让 isaacgym 的 --task/--resume/--load_run 等生效
    args.steps, args.settle = known.steps, known.settle
    args.geom_z = known.geom_z
    args.cmd_x, args.cmd_y = known.cmd_x, known.cmd_y
    args.policy = known.policy
    measure(args)
