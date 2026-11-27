#!/usr/bin/env python3
"""sim2real ↔ sim2sim 一致性校验（开发机上跑，不需要硬件）

两类校验，任一失败就退出码非 0：

  1) 【常量】把 sim2sim.py 的模块级常量 import 进来，与 spec.py 逐项比对。
     为什么需要：sim2sim.py 自己的注释里写着 action_scale 经历过 0.25→0.5→0.25 的
     反复，V15 的三个 60000 步测试就因 0.25/0.5 错配作废重跑。sim2real 若再抄一份
     常量，必然重蹈覆辙。这个脚本就是防这件事的。

  2) 【数值】喂同一份合成状态给
       (a) sim2sim.build_obs + policy  →  观测与动作
       (b) PolicyRunner                →  观测与动作
     要求逐位相同（浮点允许 1e-6）。这验证的不只是常量，还有
     历史帧顺序 / 零帧引导 / 指令死区 / 裁剪 / last_action 回路。

用法：
    python tools/check_alignment.py                     # 自动找 policy.pt
    python tools/check_alignment.py --policy /path/to/policy.pt
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # deploy/sim2real
_REPO = os.path.dirname(os.path.dirname(_ROOT))      # LgRobotDog
sys.path.insert(0, _ROOT)

import spec  # noqa: E402
from policy.policy_runner import PolicyRunner  # noqa: E402

SIM2SIM = os.path.join(_REPO, "legged_gym", "legged_gym", "scripts", "sim2sim.py")
DEFAULT_POLICY = os.path.join(
    _REPO, "legged_gym", "logs", "flat_LgDog1", "exported", "policies", "policy.pt")

FAIL = []


def _load_sim2sim():
    """把 sim2sim.py 当模块导入（它有 __main__ 守卫，导入不会执行 main）。"""
    if not os.path.isfile(SIM2SIM):
        raise FileNotFoundError("找不到 sim2sim.py: %s" % SIM2SIM)
    spec_ = importlib.util.spec_from_file_location("sim2sim_ref", SIM2SIM)
    mod = importlib.util.module_from_spec(spec_)
    sys.modules["sim2sim_ref"] = mod
    spec_.loader.exec_module(mod)
    return mod


def section(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def cmp_const(name, a, b, tol=0.0):
    """比对单个常量。tol=0 表示要求完全相等。"""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        print("  ✗ %-26s 形状不同: sim2real%s vs sim2sim%s" % (name, a_arr.shape, b_arr.shape))
        FAIL.append(name)
        return
    if a_arr.dtype.kind in "fiu" and b_arr.dtype.kind in "fiu":
        err = float(np.abs(a_arr.astype(np.float64) - b_arr.astype(np.float64)).max())
        ok = err <= tol
        print("  %s %-26s 最大偏差 %.3e   %s" % ("✓" if ok else "✗", name, err,
              "" if ok else "sim2real=%s sim2sim=%s" % (a_arr, b_arr)))
        if not ok:
            FAIL.append(name)
    else:
        ok = (a == b)
        print("  %s %-26s sim2real=%s  sim2sim=%s" % ("✓" if ok else "✗", name, a, b))
        if not ok:
            FAIL.append(name)


# ============================================================================
def check_constants(sm):
    section("【1/2】常量比对：spec.py  vs  sim2sim.py")
    print("  （sim2sim.py: %s）" % SIM2SIM)
    print()

    # --- 维度 / 历史 ---
    cmp_const("NUM_JOINTS", spec.NUM_JOINTS, 12)
    cmp_const("HISTORY_LEN", spec.HISTORY_LEN, sm.HISTORY_LEN)
    cmp_const("NUM_ONE_STEP_OBS", spec.NUM_ONE_STEP_OBS, sm.NUM_ONE_STEP_OBS)
    cmp_const("NUM_OBS", spec.NUM_OBS, sm.HISTORY_LEN * sm.NUM_ONE_STEP_OBS)

    # --- 频率 / PD / 缩放 ---
    cmp_const("SIM_DT", spec.SIM_DT, sm.SIM_DT)
    cmp_const("DECIMATION", spec.DECIMATION, sm.DECIMATION)
    cmp_const("CONTROL_FREQ", spec.CONTROL_FREQ, 1.0 / (sm.SIM_DT * sm.DECIMATION), tol=1e-9)
    cmp_const("KP", spec.KP, sm.KP)
    cmp_const("KD", spec.KD, sm.KD)
    cmp_const("ACTION_SCALE", spec.ACTION_SCALE, sm.ACTION_SCALE)

    # --- 观测缩放 ---
    for n in ("LV", "AV", "DP", "DV", "CLIP"):
        cmp_const(n, getattr(spec, n), getattr(sm, n))

    # --- 身高指令通道（46 维布局专用）---
    #   ★ 这一项曾经漂过：训练侧 commands.height_nominal 从 0.33 改成 0.30，
    #     sim2sim.py 改了（它注释里还写了"26-10-04 修正"），spec.py 留在 0.28。
    #     后果是"不给 --height 时一切正常、一给 --height 就把指令放大一倍"——
    #     典型的静默错，所以必须逐项比。注意这里比的是【默认值】：
    #     每份策略真正的名义站高由它 policy.json 的 control_contract.height_nominal 覆盖。
    if getattr(sm, "HEIGHT_NOMINAL", None) is not None:
        cmp_const("HEIGHT_NOMINAL", spec.HEIGHT_NOMINAL, sm.HEIGHT_NOMINAL)
        cmp_const("HC", spec.HC, sm.HC)
        print("      （这只是默认值；各策略真正的名义站高见各自 policy.json 的 height_nominal）")

    # --- 关节顺序 / 姿态 / 限位 ---
    cmp_const("DEFAULT_DOF_POS", spec.DEFAULT_DOF_POS, sm.DEFAULT_DOF_POS_ISAAC)
    cmp_const("REMAP(自检)", spec.REMAP_MJCF_TO_ISAAC, sm.REMAP)
    cmp_const("TAU_LIMIT", spec.TAU_LIMIT, sm.TAU_LIMIT)
    cmp_const("VEL_LIMIT", spec.VEL_LIMIT, sm.VEL_LIMIT)

    # --- 指令死区 ---
    # sim2sim.py 里是字面量 0.2（`np.hypot(cmd[0], cmd[1]) < 0.2`），这里做源码级核对
    src = open(SIM2SIM, encoding="utf-8").read()
    literal = "np.hypot(cmd[0], cmd[1]) < 0.2"
    ok = ("np.hypot(cmd[0], cmd[1]) < %g" % spec.CMD_DEADZONE_XY) in src
    print("  %s %-26s spec=%g   (源码里找到该字面量: %s)"
          % ("✓" if ok else "✗", "CMD_DEADZONE_XY", spec.CMD_DEADZONE_XY, literal in src))
    if not ok:
        FAIL.append("CMD_DEADZONE_XY")

    # --- REMAP 必须是自逆置换（sim2real 依赖这一点做双向换算）---
    rm = spec.REMAP_MJCF_TO_ISAAC
    selfinv = all(rm[rm[i]] == i for i in range(12))
    print("  %s %-26s %s" % ("✓" if selfinv else "✗", "REMAP 自逆", selfinv))
    if not selfinv:
        FAIL.append("REMAP_selfinverse")


# ============================================================================
def check_numerics(sm, policy_path):
    section("【2/2】数值比对：同一状态 → 观测 / 动作 必须逐位相同")
    print("  policy: %s" % policy_path)
    print()

    import torch
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()
    runner = PolicyRunner(policy_path)

    rng = np.random.default_rng(20260924)
    NSTEPS = 8

    # ---- 合成一段状态序列（默认姿态附近扰动 + 中间切换指令）----
    states = []
    for i in range(NSTEPS):
        q_isaac = spec.DEFAULT_DOF_POS + rng.normal(scale=0.15, size=12)
        dq_isaac = rng.normal(scale=1.5, size=12)
        quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
        # 与 sim2sim 同约定：q_mj = q_isaac[REMAP]（REMAP 自逆）
        cmd = np.array([0.7, 0.0, 0.0]) if i >= 3 else np.zeros(3)
        states.append((q_isaac, dq_isaac, quat, rng.normal(scale=0.4, size=3), cmd))

    # ---- 参考实现：手工复刻 sim2sim.py:495-535 的控制步 ----
    # ★ 四元数约定陷阱：sim2sim.build_obs 内部用 sim2sim.quat_rotate_inverse，
    #   它取 q[-1] 当实部 → 期望【xyzw】；而我们把 sim2real 的公开约定定为
    #   【wxyz】（与 math_utils.get_gravity_orientation 一致）。这里必须显式转换，
    #   否则同一份 quat 两边理解不同，比对结果毫无意义。
    def to_xyzw(q_wxyz):
        return np.asarray(q_wxyz, dtype=np.float64)[[1, 2, 3, 0]]

    def sim2sim_rollout():
        hist = [np.zeros(sm.NUM_ONE_STEP_OBS, dtype=np.float32) for _ in range(sm.HISTORY_LEN)]
        bootstrap = True
        last_a = np.zeros(12)
        obs_seq, act_seq = [], []
        for (q_i, dq_i, quat, omega, cmd) in states:
            obs = None
            if not bootstrap:
                obs = sm.build_obs(q_i[sm.REMAP], dq_i[sm.REMAP], to_xyzw(quat), omega, cmd,
                                   last_a, sm.REMAP, sm.DEFAULT_DOF_POS_ISAAC)
                hist.insert(0, obs); hist.pop()
            oh = np.concatenate(hist)
            with torch.no_grad():
                a = policy(torch.from_numpy(oh).unsqueeze(0)).numpy().flatten()
            a = np.clip(a, -sm.CLIP, sm.CLIP)
            last_a = a.copy(); bootstrap = False
            obs_seq.append(obs); act_seq.append(a)
        return obs_seq, act_seq

    # ---- 被测实现：PolicyRunner ----
    def runner_rollout():
        runner.reset()
        obs_seq, act_seq = [], []
        for (q_i, dq_i, quat, omega, cmd) in states:
            a, one_step, _ = runner.step(q_i, dq_i, quat, omega, cmd)
            obs_seq.append(one_step); act_seq.append(a)
        return obs_seq, act_seq

    obs_ref, act_ref = sim2sim_rollout()
    obs_my, act_my = runner_rollout()

    worst_obs = worst_act = 0.0
    print("  %-6s %-8s %14s %14s" % ("step", "obs?", "|Δ观测(单帧)|", "|Δ动作|"))
    for i in range(NSTEPS):
        if obs_ref[i] is None or obs_my[i] is None:
            assert obs_ref[i] is None and obs_my[i] is None, "首帧零历史引导语义不一致"
            d_obs = 0.0
            flag = "零历史"
        else:
            d_obs = float(np.abs(obs_ref[i] - obs_my[i]).max())
            flag = "有"
        d_act = float(np.abs(act_ref[i] - act_my[i]).max())
        worst_obs = max(worst_obs, d_obs); worst_act = max(worst_act, d_act)
        print("  %-6d %-8s %14.3e %14.3e" % (i, flag, d_obs, d_act))
    print()
    print("  （动作参考序列第 0 步 = 零历史引导，|a|max = %.4f）" % np.abs(act_ref[0]).max())

    # ---- 历史累积语义（与 sim2sim 一致：首步零历史，之后逐帧累积）----
    runner.reset()
    q = spec.DEFAULT_DOF_POS.copy(); dq = np.zeros(12)
    quat = np.array([1.0, 0.0, 0.0, 0.0]); omega = np.zeros(3)
    hist_counts = []
    for _ in range(spec.HISTORY_LEN + 2):
        _, _, oh = runner.step(q, dq, quat, omega, np.array([0.5, 0.0, 0.0]))
        # 【2026-10-06 修】原来写死 45（v1 观测），跑 v2（46 维、含身高指令）策略会直接
        #   ValueError: cannot reshape array of size 276 into shape (45)
        hist_counts.append(int(np.sum(np.abs(oh.reshape(-1, sm.NUM_ONE_STEP_OBS)).sum(axis=1) > 0)))
    expect = list(range(0, spec.HISTORY_LEN))
    ok_hist = hist_counts[:spec.HISTORY_LEN] == expect
    print("  历史非零帧数逐帧: %s   期望 %s" % (hist_counts, expect + [spec.HISTORY_LEN] * 2))
    print("  %s 零历史引导 + 逐帧累积符合 sim2sim 语义" % ("✓" if ok_hist else "✗"))
    if not ok_hist:
        FAIL.append("history_semantics")

    # ---- 身高指令通道：给了 --height 时两边编码必须一致 ----
    #   这一段是"最容易被忽略、错了也不报错"的地方：不传身高时 obs[3] 恒 0，
    #   编码基准写错也看不出来；一旦传了 --height，基准差 2cm 就会让指令整体偏一倍。
    if spec.HEIGHT_CMD_ENABLED:
        print()
        print("  身高指令通道 obs[3] = (目标身高 − 名义站高) × HC")
        q_i = spec.DEFAULT_DOF_POS.copy()
        dq_i = np.zeros(12)
        quat_w = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.zeros(3)
        last_a = np.zeros(12)
        worst_h = 0.0
        print("      %-10s %-14s %-14s %s" % ("目标身高", "sim2sim obs[3]",
                                              "sim2real obs[3]", "理论值(手算)"))
        for h in (spec.HEIGHT_NOMINAL, spec.HEIGHT_NOMINAL + 0.02,
                  spec.HEIGHT_NOMINAL + 0.04, spec.HEIGHT_NOMINAL - 0.02):
            o_ref = sm.build_obs(q_i[sm.REMAP], dq_i[sm.REMAP], quat_w[[1, 2, 3, 0]], omega,
                                 np.zeros(3), last_a, sm.REMAP, sm.DEFAULT_DOF_POS_ISAAC,
                                 cmd_height=h)
            o_my = runner.build_obs(q_i, dq_i, quat_w, omega, np.zeros(3), height_cmd=h)
            hand = (h - spec.HEIGHT_NOMINAL) * spec.HC
            d = abs(float(o_ref[3]) - float(o_my[3]))
            worst_h = max(worst_h, d)
            print("      %-10.3f %-14.4f %-14.4f %+.4f" % (h, o_ref[3], o_my[3], hand))
        # 再验一次语义：不给指令时必须是 0（= 策略按自己的能力站立，与训练默认一致）
        o_none = runner.build_obs(q_i, dq_i, quat_w, omega, np.zeros(3))
        print("      不给 --height → obs[3] = %.4f（必须为 0）" % o_none[3])
        ok_h = worst_h < 1e-6 and abs(float(o_none[3])) < 1e-9
        print("  %s 身高指令编码一致（最大偏差 %.3e）" % ("✓" if ok_h else "✗", worst_h))
        if not ok_h:
            FAIL.append("height_cmd")

    print()
    print("  观测最大偏差 %.3e / 动作最大偏差 %.3e" % (worst_obs, worst_act))
    ok = worst_obs < 1e-6 and worst_act < 1e-6
    print("  %s 数值一致性（阈值 1e-6）" % ("✓" if ok else "✗"))
    if not ok:
        FAIL.append("numerics")


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=DEFAULT_POLICY)
    ap.add_argument("--skip-numerics", action="store_true",
                    help="只比对常量（无需 torch/policy.pt）")
    args = ap.parse_args()

    print(spec.obs_layout_report())
    sm = _load_sim2sim()
    check_constants(sm)
    if not args.skip_numerics:
        check_numerics(sm, args.policy)

    section("结论")
    if FAIL:
        print("  ❌ %d 项不一致：%s" % (len(FAIL), ", ".join(FAIL)))
        print("  sim2real 与 sim2sim 已经漂移，先修好再上真机。")
        return 1
    print("  ✅ 全部一致：spec.py 与 sim2sim.py 无漂移，运行时观测/动作逐位相同")
    return 0


if __name__ == "__main__":
    sys.exit(main())
