#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mj_swing_profile.py — 从 sim2sim 录制里量【摆动相足底真实离地过程】（用户在 MuJoCo 里看到的就是它）。

动机（2026-10-04 用户反馈『抬腿很高和长但是实际在地下的距离不长』）:
  用户的判断来自 MuJoCo sim2sim 录像，而 Isaac 侧同一策略的离地高度可能完全不同
  （已知迁移差: 速度被打折、姿态偏移、连 5cm 台阶都上不去）⇒ 必须按【录制本身】量。
  本工具用 MuJoCo 正向运动学还原足底世界高度（复用 tools/foot_audit.py 的做法:
  录制里的 dof_pos 已是 MuJoCo 顺序，不要 remap），再按"支撑→摆动→支撑"切分摆动事件，
  给出: 峰值/均值/25-50-75% 剖面/高于各绝对高度的【时间占比】/触地前 20% 均值/水平位移。

用法:
  python tools/mj_swing_profile.py legged_gym/legged_gym/scripts/recorded_io/sim2sim_1004_230753.npz
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
XML = os.path.join(PROJECT_ROOT, "legged_gym", "resources", "robots", "LgDog1", "xml", "LgDog1.xml")
CTRL_DT = 0.01
LEGS_MJ = ["FR", "FL", "RR", "RL"]
FOOT_R = 0.012


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--stance-h", type=float, default=0.01, help="足底低于该高度算支撑相 [m]")
    ap.add_argument("--skip", type=int, default=200, help="跳过开头这么多步（起步过渡）")
    args = ap.parse_args()

    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    qadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in [f"{l}_{j}_joint" for l in LEGS_MJ for j in ("hip", "thigh", "calf")]}
    gid = {l: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_calf_collision_1") for l in LEGS_MJ}
    root_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")]
    adr = np.array([qadr[f"{l}_{j}_joint"] for l in LEGS_MJ for j in ("hip", "thigh", "calf")])

    R = np.load(args.rec, allow_pickle=True)
    q_is, base, quat = R["dof_pos"], R["base_pos"], R["base_quat_wxyz"]
    cmd = R.get("cmd")
    T = len(base)
    print(f"录制 {os.path.basename(args.rec)} | {T} 步（= {T*CTRL_DT:.0f}s）  起点偏移统计 {args.skip} 步")
    print(f"  note: {str(R['_meta_note'])[:120]}")
    if cmd is not None and len(cmd):
        print("  指令: vx=%.2f vy=%.2f wz=%.2f" % tuple(np.median(cmd, axis=0)))

    foot_w = np.zeros((T, 4, 3))
    for t in range(T):
        d.qpos[:] = 0.0
        d.qpos[root_q:root_q + 3] = base[t]
        d.qpos[root_q + 3:root_q + 7] = quat[t]
        d.qpos[adr] = q_is[t]
        mujoco.mj_kinematics(m, d)
        for k, l in enumerate(LEGS_MJ):
            foot_w[t, k] = d.geom_xpos[gid[l]]
    sole = foot_w[:, :, 2] - FOOT_R                 # 足底世界高度 [m]

    stance = sole < args.stance_h
    print(f"\n【自校验】支撑相样本 {stance.mean()*100:.1f}%，支撑相足底高中位 {np.median(sole[stance]):+.4f} m（应≈0）")
    print(f"  机身高度 z 均值 {base[args.skip:, 2].mean():.3f} m   前进位移 {(base[-1,0]-base[args.skip,0]):+.2f} m"
          f"   后退/侧移 y {(base[-1,1]-base[args.skip,1]):+.2f} m")

    ev = []
    for k in range(4):
        sw = ~stance[:, k]
        i = args.skip
        while i < T:
            if sw[i]:
                j = i
                while j < T and sw[j]:
                    j += 1
                seg = sole[i:j, k]
                if len(seg) >= 4:
                    n = len(seg)
                    ev.append(dict(leg=LEGS_MJ[k], dur=n * CTRL_DT, peak=seg.max(), mean=seg.mean(),
                                   p25=seg[int(.25*n)], p50=seg[int(.5*n)], p75=seg[int(.75*n)],
                                   pre=seg[max(0, int(.8*n)):].mean(),
                                   f2=(seg > .02).mean(), f5=(seg > .05).mean(),
                                   f8=(seg > .08).mean(), f10=(seg > .10).mean(),
                                   travel=np.linalg.norm(foot_w[j-1, k, :2] - foot_w[i, k, :2])))
                i = j
            else:
                i += 1
    if not ev:
        print("❌ 没找到摆动事件（阈值/录制长度问题）")
        return
    g = lambda key: float(np.mean([e[key] for e in ev]))
    print(f"\n── 摆动事件 {len(ev)} 次 ──")
    print(f"  摆动时长 %.3f s   摆动水平位移(步幅) %.3f m" % (g('dur'), g('travel')))
    print("  【足底真实离地高度】")
    print("    峰值 %.3f m   全程均值 %.3f m   ← 两者差距 = 是否只在一瞬抬高" % (g('peak'), g('mean')))
    print("    25%%/50%%/75%% 处 %.3f / %.3f / %.3f m" % (g('p25'), g('p50'), g('p75')))
    print("    触地前 20%% 均值 %.3f m   ← 太小 ⇒ 落地前贴地擦行" % g('pre'))
    print("    高于 2/5/8/10cm 的时间占比 %.1f%% / %.1f%% / %.1f%% / %.1f%%"
          % tuple(100 * g(k) for k in ('f2', 'f5', 'f8', 'f10')))
    print("\n  分腿: 峰值 / 均值 / 步幅 / 时长 / >5cm 占比")
    for k, lg in enumerate(LEGS_MJ):
        sub = [e for e in ev if e['leg'] == lg]
        if not sub:
            continue
        f = lambda key: float(np.mean([e[key] for e in sub]))
        print("    %-3s %+.3f  %+.3f  %.3f m  %.3f s  %.1f%%"
              % (lg, f('peak'), f('mean'), f('travel'), f('dur'), 100 * f('f5')))


if __name__ == "__main__":
    main()
