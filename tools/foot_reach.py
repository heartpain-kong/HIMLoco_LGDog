#!/usr/bin/env python3
"""摆动腿可达包线（运动学上限）—— 回答"这台机器到底能把脚抬多高/放多远"。

用途：判断"过障碍能力"到底卡在【策略/奖励】还是卡在【身体几何】。
做法：三条腿保持默认站姿并踩地（据此解出机身高度），第四条腿在关节限位内
      全枚举 (hip, thigh, calf)，用 MuJoCo 正向运动学算足底高度与前伸量，
      得到"前伸量 → 能达到的最大足底高度"这条包线。

关键判据：要迈上 H 高的台阶，摆动足底必须能抬到 ≥ H（还要留落脚余量），
          并且要在前伸量 ~0.05~0.20m 处也能达到该高度。

用法:
  python tools/foot_reach.py                 # 默认站姿，打印包线
  python tools/foot_reach.py --pitch 15      # 机身抬头 15°（爬台阶时的姿态）
  python tools/foot_reach.py --step 0.02     # 更粗的枚举（更快）
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
XML = os.path.join(PROJECT_ROOT, "legged_gym", "resources", "robots", "LgDog1", "xml", "LgDog1.xml")

DEFAULT = {"hip": 0.0, "thigh": -0.4, "calf": 0.5}      # config 里的默认姿态
LIMITS = {"hip": (-1.2, 1.2), "thigh": (-1.5, 1.5), "calf": (-0.7, 0.7)}   # MJCF joint range
FOOT_R = 0.012                                          # *_calf_collision_1 半径


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.01, help="关节枚举步长 [rad]")
    ap.add_argument("--pitch", type=float, default=0.0, help="机身抬头角 [deg]（正=抬头）")
    ap.add_argument("--leg", default="FL", choices=["FL", "FR", "RL", "RR"])
    ap.add_argument("--soft", type=float, default=1.0,
                    help="关节限位使用比例（config 的 soft_dof_pos_limit=0.9 → 传 0.9）")
    ap.add_argument("--thigh-range", type=str, default=None,
                    help="把 thigh 限制到 'min,max'（如实测摆动相范围 -0.77,0.29）——回答"
                         "『策略现在用的关节范围里是否已存在高抬腿姿态』")
    ap.add_argument("--calf-range", type=str, default=None, help="同上，限制 calf")
    ap.add_argument("--hip-range", type=str, default=None, help="同上，限制 hip")
    args = ap.parse_args()

    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)

    def jid(name):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert i >= 0, name
        return i

    def qadr(name):
        return m.jnt_qposadr[jid(name)]

    def gid(name):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert i >= 0, name
        return i

    legs = ["FL", "FR", "RL", "RR"]
    swing = args.leg
    support = [l for l in legs if l != swing]
    root_q = m.jnt_qposadr[jid("root")]

    def set_pose(pose_map, root_z=0.0, pitch=0.0):
        """pose_map: {leg: (hip,thigh,calf)}；root 只设 z 与 pitch（绕 y 轴）。"""
        d.qpos[:] = 0.0
        d.qpos[root_q + 2] = root_z
        a = np.radians(pitch) / 2.0
        d.qpos[root_q + 3:root_q + 7] = [np.cos(a), 0.0, np.sin(a), 0.0]   # wxyz, 绕 y 转 pitch
        for leg in legs:
            h, t, c = pose_map[leg]
            d.qpos[qadr(f"{leg}_hip_joint")] = h
            d.qpos[qadr(f"{leg}_thigh_joint")] = t
            d.qpos[qadr(f"{leg}_calf_joint")] = c
        mujoco.mj_kinematics(m, d)

    def foot_bottom(leg):
        return d.geom_xpos[gid(f"{leg}_calf_collision_1")].copy() - np.array([0, 0, FOOT_R])

    # ① 先解出机身高度：三条支撑腿按默认姿态踩地
    base_pose = {l: (DEFAULT["hip"], DEFAULT["thigh"], DEFAULT["calf"]) for l in legs}
    set_pose(base_pose, root_z=0.35, pitch=args.pitch)
    zs = [foot_bottom(l)[2] for l in support]
    drop = float(np.mean(zs))
    root_z = 0.35 - drop
    set_pose(base_pose, root_z=root_z, pitch=args.pitch)
    fb = {l: foot_bottom(l) for l in legs}
    stance = fb[swing]
    print(f"模型 {os.path.basename(XML)} | 摆动腿 {swing} | 机身抬头 {args.pitch:+.0f}° | 关节枚举步长 {args.step}")
    print(f"① 默认站姿解出的机身高度 z = {root_z:.4f} m；支撑足底 z = "
          f"{[round(foot_bottom(l)[2], 4) for l in support]}")
    print(f"   摆动腿默认位置 = ({stance[0]:+.3f}, {stance[1]:+.3f}, {stance[2]:+.4f})  ← 前伸量基准\n")

    # ② 枚举摆动腿关节空间（--soft 限制在柔性限位内 = 策略实际敢用的范围）
    lim = {k: (v[0] * args.soft, v[1] * args.soft) for k, v in LIMITS.items()}
    for key, val in (("hip", args.hip_range), ("thigh", args.thigh_range), ("calf", args.calf_range)):
        if val:
            lo, hi = (float(x) for x in val.split(","))
            lim[key] = (max(lim[key][0], lo), min(lim[key][1], hi))
            print(f"   [限制] {key} 收窄到 [{lim[key][0]:+.2f}, {lim[key][1]:+.2f}]")
    print(f"   枚举范围 = 关节限位 × {args.soft:.2f}: "
          + " ".join(f"{k}[{v[0]:+.2f},{v[1]:+.2f}]" for k, v in lim.items()))
    hs = np.arange(lim["hip"][0], lim["hip"][1] + 1e-9, max(args.step, 0.15))  # hip 主要影响侧向
    ts = np.arange(lim["thigh"][0], lim["thigh"][1] + 1e-9, args.step)
    cs = np.arange(lim["calf"][0], lim["calf"][1] + 1e-9, args.step)
    rec_fwd, rec_h, rec_cfg = [], [], []
    for h in hs:
        for t in ts:
            for c in cs:
                pose = dict(base_pose)
                pose[swing] = (float(h), float(t), float(c))
                set_pose(pose, root_z=root_z, pitch=args.pitch)
                p = foot_bottom(swing)
                rec_fwd.append(p[0] - stance[0])
                rec_h.append(p[2])
                rec_cfg.append((float(h), float(t), float(c)))
    rec_fwd = np.array(rec_fwd); rec_h = np.array(rec_h)
    print(f"② 枚举 {len(rec_h)} 组关节角：足底高度范围 [{rec_h.min():+.4f}, {rec_h.max():+.4f}] m，"
          f"前伸量范围 [{rec_fwd.min():+.3f}, {rec_fwd.max():+.3f}] m")
    imax = int(rec_h.argmax())
    print(f"   全空间最高: 足底 {rec_h[imax]:+.4f} m @ 前伸 {rec_fwd[imax]:+.3f} m "
          f"(hip/thigh/calf = {rec_cfg[imax][0]:+.2f}/{rec_cfg[imax][1]:+.2f}/{rec_cfg[imax][2]:+.2f})\n")

    print("③ 前伸量 → 能达到的最大足底高度（每格 5cm）:")
    print(f"   {'前伸量':>8s} {'最大足底高':>10s} {'达到它的关节角(hip/thigh/calf)':>34s}  够迈多高的台阶?")
    rows = []
    for lo in np.arange(-0.30, 0.301, 0.05):
        sel = (rec_fwd > lo - 0.0125) & (rec_fwd <= lo + 0.0125)
        if not sel.any():
            print(f"   {lo:+8.2f} {'—':>10s}")
            continue
        hs_ = rec_h[sel]
        k = int(hs_.argmax()); hmax = float(hs_[k])
        cfg = np.array(rec_cfg)[sel][k]
        rows.append((lo, hmax))
        lvl = []
        for H in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            if hmax >= H + 0.02:
                lvl.append(f"{int(H*100)}cm")
        print(f"   {lo:+8.2f} {hmax:10.4f} {cfg[0]:+8.2f}/{cfg[1]:+7.2f}/{cfg[2]:+6.2f}".ljust(62)
              + f"  {'/'.join(lvl) if lvl else '都迈不上'}")
    # ④ 结论行
    print()
    best = max(h for _, h in rows)
    print(f"④ 结论：摆动足底在【默认站姿+{args.pitch:+.0f}°抬头】下最高可到 {rec_h.max():.4f} m；"
          f"在前伸 0~0.15m 区间内最高 {best:.4f} m。")
    print("   迈台阶的经验要求：足底要能到【台阶高 + 2~5cm 余量】。"
          f"⇒ 20cm 台阶需要 ≥0.22~0.25m，25cm 需要 ≥0.27~0.30m。")


if __name__ == "__main__":
    main()
