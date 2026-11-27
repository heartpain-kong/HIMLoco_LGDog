#!/usr/bin/env python3
"""view_model.py — 在 MuJoCo 交互式 viewer 里运行机器人模型（带 PD 站姿保持）

为什么需要它：XML 本身只有刚体+执行器，没有控制器。
直接用 `python -m mujoco.viewer --mjcf=xxx.xml` 打开，机器人会立刻瘫在地上，
看起来像"模型坏了"，其实只是没有 PD。

本脚本用 launch_passive 打开 MuJoCo 官方 viewer，同时跑一个 PD 保持住目标姿态：
  * 鼠标左键拖躯干 / 右键旋转视角 / 滚轮缩放
  * 右侧 Control 面板可看各执行器状态
  * 关窗即退出

用法（conda 环境 himloco，需要桌面显示）：
  cd legged_gym/legged_gym/scripts
  python view_model.py ../../resources/robots/LgDog1/xml/LgDog1.xml
  python view_model.py ../../resources/robots/LgDog1/xml/LgDog1.xml --crouch 0.5   # 蹲一半
  python view_model.py <xml> --target 0,0.3,-0.6,0,0.3,-0.6,0,0.3,-0.6,0,0.3,-0.6  # 手写目标角
  python view_model.py <xml> --no-hold          # 不做 PD，纯自由落体（看碰撞/惯性）

无显示环境（服务器）：改用
  MUJOCO_GL=egl python check_model.py <xml>     # 离线渲染出图
"""

import argparse
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", help="MuJoCo XML 路径")
    ap.add_argument("--kp", type=float, default=40.0, help="PD 的 P 增益")
    ap.add_argument("--kd", type=float, default=1.0, help="PD 的 D 增益")
    ap.add_argument("--target", default=None,
                    help="目标关节角（逗号分隔，顺序=data.ctrl 顺序）；默认全 0 = 零位站姿")
    ap.add_argument("--crouch", type=float, default=0.0,
                    help="蹲下程度 0~1：把大腿 +0.5·x、小腿 -0.7·x（受 URDF 限位裁剪）")
    ap.add_argument("--no-hold", action="store_true", help="不加 PD（自由落体，看纯物理）")
    ap.add_argument("--settle", type=float, default=1.0, help="开窗前先在地面稳定多少秒")
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer

    xml = os.path.abspath(args.xml)
    if not os.path.exists(xml):
        sys.exit(f"找不到文件：{xml}")
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)

    nu = m.nu
    lim = m.actuator_ctrlrange[:, 1].copy()
    if args.target:
        target = np.array([float(x) for x in args.target.split(",")])
        if len(target) != nu:
            sys.exit(f"--target 需要 {nu} 个数（模型有 {nu} 个执行器），收到 {len(target)}")
    else:
        target = np.zeros(nu)
        if args.crouch:                      # 默认按 go1 式腿序 [hip,thigh,calf]×4
            for i in range(nu):
                if i % 3 == 1:
                    target[i] = 0.5 * args.crouch
                elif i % 3 == 2:
                    target[i] = -0.7 * args.crouch
    target = np.clip(target, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])

    print(f"模型 {os.path.basename(xml)}：nq={m.nq} nu={nu} 总质量={m.body_mass.sum():.3f}kg")
    print("执行器（data.ctrl 顺序）→ 目标角:")
    for i in range(nu):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0])
        print(f"  [{i:2d}] {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i):10s} "
              f"({jn:16s}) target={target[i]:+.2f}  ctrlrange={np.round(lim[i] * np.array([-1, 1]), 1)}")

    # 先无窗口稳定一会儿，避免开窗瞬间还在下坠
    if not args.no_hold:
        free = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        if free:
            d.qpos[2] += 0.01
        n = int(args.settle / m.opt.timestep)
        for _ in range(n):
            d.ctrl[:] = np.clip(args.kp * (target - d.qpos[-nu:]) - args.kd * d.qvel[-nu:], -lim, lim)
            mujoco.mj_step(m, d)
        tilt = np.degrees(2 * np.arccos(min(1.0, abs(d.qpos[3]))))
        print(f"预稳定 {args.settle:.1f}s 后：z={d.qpos[2]:.3f} 倾角={tilt:.2f}° 接触={d.ncon}")

    print("\n打开 viewer（关窗退出）...")
    try:
        viewer = mujoco.viewer.launch_passive(m, d)
    except Exception as e:
        sys.exit(f"打不开窗口（无显示环境？）：{e}\n"
                 f"服务器上请改用：MUJOCO_GL=egl python check_model.py {args.xml}")

    with viewer as v:
        while v.is_running():
            if not args.no_hold:
                d.ctrl[:] = np.clip(args.kp * (target - d.qpos[-nu:]) - args.kd * d.qvel[-nu:],
                                    -lim, lim)
            mujoco.mj_step(m, d)
            v.sync()
            time.sleep(m.opt.timestep)      # 按真实时间跑（timestep=0.002 → 500Hz）
    print("viewer 已关闭。")


if __name__ == "__main__":
    main()
