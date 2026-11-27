#!/usr/bin/env python3
"""步态审计：从 sim2sim 录制里用正向运动学还原【足端】状态，审计 foot_clearance 奖励。

回答两个问题：
  ①【抬腿天花板】这只策略实际把摆动腿抬多高？离"迈上 20cm 台阶"需要的 0.22m 差多少？
  ②【奖励漏洞】训练奖励 _reward_foot_clearance（legged_robot.py:1167）是
        Σ_feet (足端机身系 z − clearance_target)² × ‖足端机身系水平速度‖
     权重是【水平速度】—— 那么"把腿放慢"与"把腿抬高"哪个更能降低惩罚？
     本工具按训练同一公式复现该惩罚，并做反事实分解。

用法:
  python tools/foot_audit.py <recording.npz> [--clearance -0.20] [--proposed -0.06]
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
XML = os.path.join(PROJECT_ROOT, "legged_gym", "resources", "robots", "LgDog1", "xml", "LgDog1.xml")
# ★【关键】录制里的 dof_pos 是 sim2sim 的 `q[-rdof:]`，已经是【MuJoCo 顺序】，不要 remap！
#   判定依据（实测）: |dof_pos − target_q| = 0.136 rad（不 remap）vs 0.302 rad（remap 后），
#   而 target_q = action_mj*scale + DEFAULT_MUJOCO 确定是 MuJoCo 顺序；PD kp=30 跟踪很紧，
#   正确顺序的偏差必然小得多。remap 只管【策略输入输出】的顺序，与录制字段无关。
CTRL_DT = 0.01                                    # sim.dt 0.005 × decimation 2
LEGS_MJ = ["FR", "FL", "RR", "RL"]                # MuJoCo 顺序
FOOT_R = 0.012
J_LIMIT_MJ = np.array([1.2, 1.5, 0.7] * 4)        # MuJoCo 顺序的关节限位
SOFT = 0.9                                        # config: soft_dof_pos_limit
STA = np.array([30.0, 30.0, 30.0] * 4)            # kp（只用于打印）


def quat_rotate_inverse(q_wxyz, v):
    """把世界系向量转到机体系（q 为 wxyz）。"""
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return R.T @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--clearance", type=float, default=-0.20, help="训练里的 clearance_height_target")
    ap.add_argument("--proposed", type=float, default=-0.06, help="建议的机身系净空目标（净空=机身站高+本值）")
    ap.add_argument("--stance-h", type=float, default=0.01, help="足底低于该高度算支撑相 [m]")
    args = ap.parse_args()

    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    qadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in [f"{l}_{j}_joint" for l in LEGS_MJ for j in ("hip", "thigh", "calf")]}
    gid = {l: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{l}_calf_collision_1") for l in LEGS_MJ}
    root_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")]
    # MuJoCo 顺序的关节 qpos 地址（按 LEGS_MJ × hip/thigh/calf）
    adr = np.array([qadr[f"{l}_{j}_joint"] for l in LEGS_MJ for j in ("hip", "thigh", "calf")])

    R = np.load(args.rec, allow_pickle=True)
    q_is = R["dof_pos"]            # (T,12) Isaac 顺序
    base = R["base_pos"]           # (T,3)
    quat = R["base_quat_wxyz"]     # (T,4)
    T = len(base)
    print(f"录制 {os.path.basename(args.rec)} | {T} 步（= {T*CTRL_DT:.0f}s）")
    print(f"  note: {str(R['_meta_note'])[:110]}")

    # ① 逐帧 FK 求足端位置（世界系），并做有限差分求速度
    foot_w = np.zeros((T, 4, 3)); q_mj_all = np.zeros((T, 12))
    for t in range(T):
        q_mj = q_is[t].copy()   # 已是 MuJoCo 顺序
        q_mj_all[t] = q_mj
        d.qpos[:] = 0.0
        d.qpos[root_q:root_q + 3] = base[t]
        d.qpos[root_q + 3:root_q + 7] = quat[t]
        d.qpos[adr] = q_mj
        mujoco.mj_kinematics(m, d)
        for k, l in enumerate(LEGS_MJ):
            foot_w[t, k] = d.geom_xpos[gid[l]]
    foot_bottom_w = foot_w[:, :, 2] - FOOT_R          # 足底世界高度
    # 世界系足端速度（中心差分）→ 机身系
    vel_w = np.zeros_like(foot_w)
    vel_w[1:-1] = (foot_w[2:] - foot_w[:-2]) / (2 * CTRL_DT)
    vel_w[0] = (foot_w[1] - foot_w[0]) / CTRL_DT
    vel_w[-1] = (foot_w[-1] - foot_w[-2]) / CTRL_DT
    vel_base = R["base_lin_vel"]                       # (T,3) 世界系
    body_z = np.zeros((T, 4)); body_v = np.zeros((T, 4, 3))
    for t in range(T):
        for k in range(4):
            rel = foot_w[t, k] - base[t]
            body_z[t, k] = quat_rotate_inverse(quat[t], rel)[2]
            body_v[t, k] = quat_rotate_inverse(quat[t], vel_w[t, k] - vel_base[t])

    # ② 自校验：支撑足的足底高度应 ≈0
    stance = foot_bottom_w < args.stance_h
    print(f"\n① 【自校验】足底世界高度: 支撑相样本占 {stance.mean()*100:.1f}%，"
          f"支撑相足底高中位数 {np.median(foot_bottom_w[stance]):+.4f} m（应 ≈0，验证 FK/映射没错）")

    # ③ 抬腿天花板
    per_step_peak = foot_bottom_w.max(axis=1)      # 每控制步最高的那只脚
    print(f"\n② 【抬腿天花板】摆动足底高度（机身系 clearance_target={args.clearance} → 目标净空 "
          f"{0.33+args.clearance:.2f}m；迈 20cm 台阶需要 ≥0.22m）:")
    print(f"   逐帧四足最高值: 中位 {np.median(per_step_peak):.4f}  p90 {np.percentile(per_step_peak,90):.4f}  "
          f"峰值 {per_step_peak.max():.4f} m")
    # 按"脚"统计摆动期峰值（支撑→摆动→支撑 为一个周期）
    peaks = []
    for k in range(4):
        sw = ~stance[:, k]
        i = 0
        while i < T:
            if sw[i]:
                j = i
                while j < T and sw[j]: j += 1
                peaks.append(foot_bottom_w[i:j, k].max()); i = j
            else: i += 1
    peaks = np.array(peaks)
    print(f"   逐次摆动峰值（共 {len(peaks)} 次）: 中位 {np.median(peaks):.4f}  p90 "
          f"{np.percentile(peaks,90):.4f}  最大 {peaks.max():.4f} m")
    print(f"   ⇒ 达到 0.22m 的摆动次数: {(peaks>=0.22).sum()}/{len(peaks)}；达到 0.12m: {(peaks>=0.12).sum()}/{len(peaks)}")

    # ④ 复现训练奖励并做反事实分解
    err2 = (body_z - args.clearance) ** 2
    speed = np.linalg.norm(body_v[:, :, :2], axis=2)
    term = err2 * speed
    rew_obs = term.sum(axis=1)                       # 训练里 reward 函数返回的就是 Σ_feet
    print(f"\n③ 【foot_clearance 奖励复现】按训练公式 Σ_feet (机身系足端z − target)² × ‖水平速度‖:")
    print(f"   每步值: 中位 {np.median(rew_obs):.5f}  均值 {rew_obs.mean():.5f}  p90 {np.percentile(rew_obs,90):.5f}"
          f"  （训练权重 −1.0 → 每步惩罚 ≈ −{rew_obs.mean():.5f}）")
    # 反事实 A: 抬高到目标（height_error→0）    反事实 B: 摆动速度减半
    cf_lift = np.zeros_like(term)                    # 完全抬到位 → 误差 0 → 惩罚 0
    cf_slow = err2 * (speed * 0.5)
    print(f"   反事实 A【把腿抬到位】(误差→0): 惩罚 → 0（降幅 100%）")
    print(f"   反事实 B【只把摆动腿水平速度减半】(完全不抬腿): 惩罚 → {cf_slow.sum(axis=1).mean():.5f} "
          f"（降幅 {100*(1-cf_slow.sum(axis=1).mean()/rew_obs.mean()):.1f}%）")
    print(f"   ⇒ 速度与高度【同等有效】地降低这一项 → 策略可以选择『慢慢挪』而不是『抬起来』（奖励漏洞）")
    # 分箱：按机身系足端高度看速度
    print(f"\n   分箱（摆动相样本，按足端机身系 z 分箱）:")
    print(f"   {'z 区间':>14s} {'样本占比':>8s} {'水平速度中位':>12s} {'误差²中位':>11s} {'该项中位贡献':>13s}")
    sw_b = ~stance
    edges = [-0.35, -0.30, -0.25, -0.20, -0.15, -0.10, 0.0]
    for a, b in zip(edges[:-1], edges[1:]):
        sel = sw_b & (body_z >= a) & (body_z < b)
        if sel.sum() < 20: continue
        print(f"   [{a:+.2f},{b:+.2f}) {sel.mean()*100:7.1f}% {np.median(speed[sel]):12.4f} "
              f"{np.median(err2[sel]):11.5f} {np.median(term[sel]):13.6f}")
    # ⑤ 建议形式：摆动相门控 + relu（不带速度权重）
    prop = np.zeros((T, 4))
    need = args.proposed
    prop[sw_b] = np.maximum(need - body_z[sw_b], 0.0)
    print(f"\n④ 【建议的替换形式】摆动相门控 + relu({need:+.2f} − 足端机身系z)，不带速度权重:")
    print(f"   每步值: 均值 {prop.sum(axis=1).mean():.5f}（当前形式 {rew_obs.mean():.5f}）"
          f"  → 现在这一项会变成 {prop.sum(axis=1).mean()/rew_obs.mean():.2f}× 大，且【只由高度决定】")

    # ⑥ 关节余量（摆动相）
    print(f"\n⑤ 【关节余量】摆动相关节角 vs 限位（MuJoCo 顺序 FR,FL,RR,RL × hip,thigh,calf；soft={SOFT}）:")
    sw_all = np.repeat(~stance, 3, axis=1)           # (T,12)
    for i, nm in enumerate([f"{l}_{j}" for l in LEGS_MJ for j in ("hip", "thigh", "calf")]):
        v = q_mj_all[sw_all[:, i], i] if sw_all[:, i].any() else q_mj_all[:, i]
        lim = J_LIMIT_MJ[i]; soft = lim * SOFT
        print(f"   {nm:12s} 摆动相范围 [{v.min():+.3f}, {v.max():+.3f}] / 限位 ±{lim:.2f}"
              f"（柔性 ±{soft:.2f}）→ 用掉限位的 {max(abs(v.min()),abs(v.max()))/lim*100:5.1f}%"
              + ("  ⚠ 触到柔性限位" if max(abs(v.min()), abs(v.max())) > soft else ""))


if __name__ == "__main__":
    main()
