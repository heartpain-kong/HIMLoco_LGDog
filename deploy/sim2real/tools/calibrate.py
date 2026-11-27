#!/usr/bin/env python3
"""LgDog1 标定工具 —— 把 hardware_map.yaml 里那 12 行从"猜的"变成"实测的"

hardware_map.yaml 里 can_id / sign / gear_ratio / zero_offset 四个量必须实测，
猜错任何一个，策略一上电就可能把关节往反方向猛推。这个工具按【风险从低到高】
的顺序提供子命令，请务必按顺序做：

    1. check      纯静态：检查映射表自洽性（不连硬件，零风险）
    2. scan       只读：扫总线上的 CAN ID，比对映射表
    3. read       只读：读 12 个关节当前角度/温度，确认接线与新鲜度
    4. wiggle     ★ 第一次动电机：单个关节 ±0.03 rad 缓慢摆动，确认"发的是哪个、
                     动的是哪个"。幅度小到不会摔机器人
    5. direction  单关节 +0.10 rad，看实测角度往哪走 → 判定 sign
    6. gear       单关节正转一段，比较"电机报的角度变化"与"你量的关节角度变化"
                     → 判定 gear_ratio（小腿那级 2:1 减速就靠这个确认）
    7. offsets    把机器人摆成 URDF 零位姿态，读回 12 个电机角度 → 直接打印可粘贴的
                     zero_offset 表

安全约定（每一个子命令都遵守）：
  * 只使能【本次用到的】电机，其余保持失能
  * 单关节测试一律 kp 从低到高、幅度 ≤ wiggle_max_rad（默认 0.10），并且检查
    目标不超 URDF 限位
  * 任何异常（超时/NaN/越限/用户 Ctrl-C）先 damping_brake 再退出
  * 不修改任何文件，只打印建议值 —— 由你确认后再手工填回 hardware_map.yaml

用法示例：
    python tools/calibrate.py check
    python tools/calibrate.py scan --port /dev/ttyUSB0
    python tools/calibrate.py read --port /dev/ttyUSB0
    python tools/calibrate.py wiggle --port /dev/ttyUSB0 --joint FL_hip
    python tools/calibrate.py direction --port /dev/ttyUSB0 --joint FL_hip
    python tools/calibrate.py gear --port /dev/ttyUSB0 --joint FL_calf
    python tools/calibrate.py offsets --port /dev/ttyUSB0
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import spec  # noqa: E402
from interface.motor_mapping import MotorMapping  # noqa: E402


def _banner(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def _fmt_arr(a, nd=4):
    return np.array2string(np.asarray(a), precision=nd, suppress_small=True)


# ============================================================================
def cmd_check(args):
    """纯静态检查，不连硬件。"""
    _banner("1/7  check —— 映射表静态检查")
    m = MotorMapping(args.map)
    print(m.describe())
    print()

    ok = True
    # 单关节行程 vs gear_ratio 的一致性
    print("关节行程 × 减速比 = 电机侧所需行程（用户核对电机是否够转）:")
    for i, n in enumerate(spec.JOINT_LABELS):
        span = float(spec.JOINT_UPPER[i] - spec.JOINT_LOWER[i])
        need = span * m.gear_ratio[i]
        print("  %-10s 关节行程 %.3f rad  ×%.2f → 电机侧 %.3f rad (%.1f°)"
              % (n, span, m.gear_ratio[i], need, np.degrees(need)))

    # 与 URDF 速度上限的一致性：calf 应比 hip/thigh 慢一半 → gear 2.0
    print()
    print("与 URDF 速度上限交叉验证:")
    for i, n in enumerate(spec.JOINT_LABELS):
        r = spec.VEL_LIMIT[i] / spec.VEL_LIMIT[0]      # 相对 hip 的倍率
        if abs(1.0 / r - m.gear_ratio[i]) > 1e-6:
            print("  ⚠️  %-10s URDF速度上限 %.3f (相对 hip %.2f×) 与 gear_ratio %.2f 不自洽"
                  % (n, spec.VEL_LIMIT[i], r, m.gear_ratio[i]))
            ok = False
    if ok:
        print("  ✅ 12 个关节的 gear_ratio 与 URDF 速度上限全部自洽")

    print()
    if not m.verified:
        print("★ verified 仍为 false —— 标定完成后请改成 true，否则进不了策略控制。")
    print("  标定顺序：scan → read → wiggle → direction → gear → offsets")
    return 0


# ============================================================================
def _open_bus(args, mapping, enabled_joints=None):
    """打开总线。只使能 enabled_joints 指定的关节（None = 一个都不使能）。"""
    from interface.motor_driver import MotorBus
    bus = MotorBus(mapping, args.port, debug=args.debug)
    bus.connect()
    return bus


def _enable_only(bus, joint_indices):
    """只使能指定索引的电机，其余确保失能。"""
    from drivers.motor_driver import RobStrideDriver  # noqa: F401
    for i, name in enumerate(spec.JOINT_LABELS):
        try:
            if i in joint_indices:
                bus.driver.clear_warnings(name)
                bus.driver.enable(name)
            else:
                bus.driver.disable(name)
        except Exception as e:
            print("  ⚠️  %s 使能/失能失败: %s" % (name, e))
    bus._enabled = bool(joint_indices)


def _safe_brake(bus):
    try:
        bus.damping_brake()
        time.sleep(0.2)
    except Exception:
        pass
    try:
        bus.disable_all()
    except Exception:
        pass


def cmd_scan(args):
    _banner("2/7  scan —— 扫描总线上的电机 ID（只读，不使能）")
    from drivers.motor_driver import RobStrideDriver
    m = MotorMapping(args.map)
    d = RobStrideDriver(args.port, debug=args.debug)
    d.connect()
    try:
        ids = sorted(d.scan_motors(timeout=args.timeout))
    finally:
        d.disconnect()

    expect = m.can_id.astype(int)
    print("扫到 %d 个 ID: %s" % (len(ids), ids))
    print("映射表期望    : %s" % sorted(expect.tolist()))
    print()
    missing = [i for i in expect if i not in ids]
    extra = [i for i in ids if i not in expect]
    print("%-10s %s" % ("关节", "状态"))
    for i, n in enumerate(spec.JOINT_LABELS):
        cid = int(expect[i])
        st = "✓ 在线" if cid in ids else "✗ 没扫到"
        print("  %-10s can_id=%-4d %s" % (n, cid, st))
    print()
    if missing:
        print("⚠️  这些 ID 没扫到: %s —— 查接线/供电/ID 拨码" % missing)
    if extra:
        print("★ 总线上有映射表之外的 ID: %s" % ["0x%X" % e for e in extra])
        print("   若其中有下位机广播 IMU，就是它 —— 用 tools/link_sniffer.py --can 看数据变化")
    if not missing and not extra:
        print("✅ 映射表与总线上实际存在的 ID 完全一致")
    return 0 if not missing else 1


def cmd_read(args):
    _banner("3/7  read —— 读 12 关节状态（只读，不使能）")
    m = MotorMapping(args.map)
    bus = _open_bus(args, m)
    try:
        print("等反馈 ...")
        bus.wait_feedback_ready(timeout=args.timeout)
        q, dq, tau, temp, fresh = bus.read()
        print()
        print("%-10s %10s %10s %10s %8s %8s" % ("关节", "q(rad)", "dq(rad/s)", "tau(Nm)", "温度℃", "新鲜"))
        for i, n in enumerate(spec.JOINT_LABELS):
            print("  %-10s %10.4f %10.4f %10.4f %8.1f %8s"
                  % (n, q[i], dq[i], tau[i], temp[i], "✓" if fresh[i] else "✗"))
        print()
        print("提示：把机器人摆成 URDF 零位姿态（各关节角=0）后再跑 `offsets`。")
        hot = [(spec.JOINT_LABELS[i], temp[i]) for i in range(spec.NUM_JOINTS) if temp[i] > 60]
        if hot:
            print("⚠️  温度偏高的电机: %s" % hot)
    finally:
        _safe_brake(bus)
        bus.disconnect()
    return 0


# ============================================================================
def _single_joint_move(args, m, idx, delta_rad, kp, kj, seconds, label):
    """让单个关节缓慢走到 当前位置+delta，返回观测到的关节侧角度序列。

    这是所有单关节测试的公共底座：低 kp、慢速、限位检查、异常刹车。
    """
    bus = _open_bus(args, m)
    name = spec.JOINT_LABELS[idx]
    try:
        _enable_only(bus, {idx})
        bus.wait_feedback_ready(timeout=args.timeout)
        q0, dq0, _, _, _ = bus.read()
        start = float(q0[idx])
        target_joint = start + delta_rad

        lo, hi = float(spec.JOINT_LOWER[idx]), float(spec.JOINT_UPPER[idx])
        margin = 0.05
        if not (lo + margin <= target_joint <= hi - margin):
            print("✋ 目标 %.4f 超出 %s 的限位 [%.2f, %.2f]（留 %.2f 余量），拒绝执行。"
                  % (target_joint, name, lo, hi, margin))
            print("   当前角度 %.4f，请把机器人摆到更居中的姿态再试。" % start)
            return None
        if abs(delta_rad) > args.wiggle_max_rad:
            print("✋ 单步幅度 %.3f 超过 wiggle_max_rad=%.3f，拒绝执行（安全上限）。"
                  % (abs(delta_rad), args.wiggle_max_rad))
            return None

        print("[%s] %s  当前 %.4f → 目标 %.4f (Δ=%+.4f rad, kp=%.1f, kd=%.2f, %.1fs)"
              % (label, name, start, target_joint, delta_rad, kp, kj, seconds))
        print("      ⚠️ 手指放在电源开关上。异常请 Ctrl-C。")

        # 其余关节保持失能；只给本关节发位置目标
        n_steps = max(1, int(seconds / spec.CONTROL_DT))
        samples = []
        for k in range(n_steps):
            # 前 30% 时间做平滑启动，避免阶跃冲击
            a = min(1.0, (k + 1) / max(1.0, 0.3 * n_steps))
            cmd = start + delta_rad * a
            tgt = np.full(spec.NUM_JOINTS, start)
            tgt[idx] = cmd
            bus.write_targets(tgt, np.full(spec.NUM_JOINTS, kp),
                              np.full(spec.NUM_JOINTS, kj))
            bus.poll()
            q, dq, _, _, fresh = bus.read()
            samples.append((q[idx], dq[idx]))
            if not np.isfinite(q[idx]):
                print("❌ 读到 NaN，立刻刹车")
                _safe_brake(bus)
                return None
            time.sleep(spec.CONTROL_DT)
        q_end = samples[-1][0]
        print("      %s 实测 %.4f → %.4f  (实际变化 %+.4f rad，指令 %+.4f)"
              % (name, start, q_end, q_end - start, delta_rad))
        return {"start": start, "end": float(q_end), "delta_cmd": delta_rad,
                "samples": samples}
    finally:
        _safe_brake(bus)
        bus.disconnect()


def cmd_wiggle(args):
    _banner("4/7  wiggle —— 单关节 ±%.3f rad 微摆（第一次让电机动作）" % args.wiggle_max_rad)
    print("目的：确认『映射表里写的这个关节』和『实际动的那个关节』是同一个。")
    print("判据：只有指定的那个关节动；若动的是别的关节 → can_id 映射错了。")
    print("      ⚠️ 幅度刻意做得很小，正常不会让机器人失去平衡；但仍请扶稳。")
    print()
    _confirm(args)
    m = MotorMapping(args.map)
    for idx in _target_indices(args):
        r1 = _single_joint_move(args, m, idx, +args.wiggle_rad, args.kp, args.kd,
                                args.seconds, "正向")
        if r1 is None:
            continue
        time.sleep(0.3)
        r2 = _single_joint_move(args, m, idx, -args.wiggle_rad, args.kp, args.kd,
                                args.seconds, "反向")
        if r2 is None:
            continue
        moved = r1["end"] - r1["start"]
        ratio = (moved / r1["delta_cmd"]) if r1["delta_cmd"] else float("nan")
        ok = moved * r1["delta_cmd"] > 0 and abs(ratio) > 1e-3
        print("   → %s %s（跟踪比 %.3f）" % ("✅ 有响应" if ok else "⚠️ 几乎没动",
              "", ratio))
    return 0


def cmd_direction(args):
    _banner("5/7  direction —— 判定 sign（方向符号）")
    print("方法：给 %+.3f rad 指令，看【方向】是否与 URDF 正方向一致。" % args.probe_rad)
    print("  若实测角度变化与指令同号 → sign = +1；反号 → sign = -1。")
    print("  注意：这里读到的是当前映射下的关节角，所以判据是『跟不跟指令走』。")
    print()
    _confirm(args)
    m = MotorMapping(args.map)
    results = {}
    for idx in _target_indices(args):
        r = _single_joint_move(args, m, idx, +args.probe_rad, args.kp, args.kd,
                               args.seconds, "direction")
        if r is None:
            continue
        d = r["end"] - r["start"]
        sign = 1 if d * args.probe_rad > 0 else -1
        results[spec.JOINT_LABELS[idx]] = sign
        print("   → 指令 %+.3f，实测 %+.4f ⇒ 建议 sign = %+d" % (args.probe_rad, d, sign))
    if results:
        print()
        print("把下面这些填回 hardware_map.yaml：")
        for n, s in results.items():
            print("  %-10s sign: %+d" % (n, s))
        print()
        print("★ 但 sign 只能这样【相对】判定。要确认绝对方向（URDF 正方向到底对应")
        print("  电机正转还是反转），请手工把该关节往 URDF 正方向掰一点，看 read 出的")
        print("  角度是变大还是变小 —— 变大则 sign=+1，变小则 sign=-1。")
    return 0


def cmd_gear(args):
    _banner("6/7  gear —— 判定 gear_ratio（减速比）")
    print("方法：在【电机空间】走固定角度 Δθ_motor，同时量【关节空间】实际变化 Δθ_joint。")
    print("      gear_ratio = Δθ_motor / Δθ_joint。")
    print()
    print("★ 本次要验证的重点：小腿（calf）是不是真有 2:1 减速 ——")
    print("  依据是 URDF 里 calf 速度上限 9.425 正好是 hip/thigh(18.85) 的一半。")
    print("  如果实测 ratio≈1.0，说明没有外置减速，映射表要把 calf 的 gear_ratio 改回 1.0。")
    print()
    _confirm(args)
    m = MotorMapping(args.map)
    for idx in _target_indices(args):
        name = spec.JOINT_LABELS[idx]
        ratio_now = float(m.gear_ratio[idx])
        # 在电机空间走 args.gear_probe_motor_rad，换算成关节空间的指令增量
        delta_motor = args.gear_probe_motor_rad
        delta_joint_cmd = delta_motor / ratio_now
        if abs(delta_joint_cmd) > args.wiggle_max_rad:
            print("✋ 换算出的关节指令 %.3f 超过安全上限，请减小 --gear-probe-motor-rad"
                  % abs(delta_joint_cmd))
            continue
        print("[gear] %s  假设 ratio=%.2f，电机侧走 %.3f rad → 关节侧指令 %.3f rad"
              % (name, ratio_now, delta_motor, delta_joint_cmd))
        r = _single_joint_move(args, m, idx, delta_joint_cmd, args.kp, args.kd,
                               args.seconds, "gear")
        if r is None:
            continue
        d_joint = r["end"] - r["start"]
        if abs(d_joint) < 1e-4:
            print("   ⚠️ 关节几乎没动，无法判定（增大 --gear-probe-motor-rad 或查机械卡死）")
            continue
        ratio_meas = delta_motor / d_joint
        print("   → 电机 %.3f rad / 关节 %.4f rad ⇒ 实测 gear_ratio ≈ %.3f" % (
            delta_motor, d_joint, ratio_meas))
        print("     当前映射表写的是 %.2f，%s"
              % (ratio_now,
                 "一致 ✓" if abs(ratio_meas - ratio_now) < 0.25 * ratio_now
                 else "★ 不一致，请改成 %.3f 并复核" % ratio_meas))
    return 0


def cmd_offsets(args):
    _banner("7/7  offsets —— 零位标定")
    print("步骤：")
    print("  1) 把机器人摆成 URDF 零位姿态（每个关节角在物理上都对到 0 位）")
    print("     —— 本机默认站姿 = spec.DEFAULT_DOF_POS = %s" % _fmt_arr(spec.DEFAULT_DOF_POS))
    print("        如果你是按『默认站姿』摆的，下面会直接给出把该姿态当作")
    print("        DEFAULT_DOF_POS 所需的 offset。")
    print("  2) 保持不动，回车采样。")
    print()
    _confirm(args)

    m = MotorMapping(args.map)
    bus = _open_bus(args, m)
    try:
        bus.wait_feedback_ready(timeout=args.timeout)
        print("采样 %d 次取平均 ..." % args.samples)
        acc = np.zeros(spec.NUM_JOINTS)
        for _ in range(args.samples):
            q, _, _, _, _ = bus.read()
            acc += m.sim_to_real(q)      # 这里拿的是"电机原始角度"（sim 空转为 real）
            time.sleep(0.02)
        motor_angles = acc / args.samples

        # 目标：让"当前物理姿态"对应 spec.DEFAULT_DOF_POS
        #   motor = sign*(sim*gear) + offset,  sim = DEFAULT  =>  offset = motor - sign*DEFAULT*gear
        offsets = motor_angles - m.sign * (spec.DEFAULT_DOF_POS * m.gear_ratio)

        print()
        print("当前各电机原始角度 [rad]:")
        for i, n in enumerate(spec.JOINT_LABELS):
            print("  %-10s %+8.4f" % (n, motor_angles[i]))
        print()
        print("⇒ 使『当前姿态 = 默认站姿』所需的 zero_offset，可直接粘贴回 hardware_map.yaml：")
        print()
        for i, n in enumerate(spec.JOINT_LABELS):
            print("  %-10s {bus: %d, can_id: %d, sign: %+d, gear_ratio: %.1f, zero_offset: %+.4f}"
                  % (n, m.bus[i], m.can_id[i], int(m.sign[i]), m.gear_ratio[i], offsets[i]))
        print()
        print("⚠️ 这只是把『当前姿态』钉成默认站姿。若当前姿态本身就不准，")
        print("   标出来的 offset 会把误差固化进去 —— 先用水平仪/量角器确认姿态。")
        print("⚠️ 若某关节 sign 是错的，这里的 offset 也一定是错的。先做 direction 再做 offsets。")
    finally:
        _safe_brake(bus)
        bus.disconnect()
    return 0


# ============================================================================
def _target_indices(args):
    if args.joint:
        if args.joint not in spec.JOINT_LABELS:
            raise SystemExit("未知关节 %r，可选: %s" % (args.joint, list(spec.JOINT_LABELS)))
        return [spec.JOINT_LABELS.index(args.joint)]
    if args.leg:
        return [i for i, n in enumerate(spec.JOINT_LABELS) if n.startswith(args.leg + "_")]
    return list(range(spec.NUM_JOINTS))


def _confirm(args):
    if args.yes:
        return
    print("⚠️  确认：机器人已固定/扶稳、周围无人、电源开关在手边？")
    try:
        ans = input("    输入 yes 继续: ").strip().lower()
    except EOFError:
        raise SystemExit("非交互环境请加 --yes")
    if ans != "yes":
        raise SystemExit("已取消。")


def main():
    ap = argparse.ArgumentParser(
        description="LgDog1 标定工具（按 check→scan→read→wiggle→direction→gear→offsets 顺序做）")
    ap.add_argument("--map", default=None, help="hardware_map.yaml 路径")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--yes", action="store_true", help="跳过交互确认（脚本化用，慎用）")
    # 单关节选择
    ap.add_argument("--joint", default=None, help="只操作这个关节，如 FL_hip")
    ap.add_argument("--leg", default=None, choices=["FL", "FR", "RL", "RR"],
                    help="操作整条腿的 3 个关节")
    # 运动参数（默认保守）
    ap.add_argument("--wiggle-max-rad", type=float, default=0.10,
                    help="单步幅度硬上限 [rad]，默认 0.10（约 5.7°）")
    ap.add_argument("--wiggle-rad", type=float, default=0.03,
                    help="wiggle 的摆动幅度 [rad]，默认 0.03")
    ap.add_argument("--probe-rad", type=float, default=0.10,
                    help="direction 的探测幅度 [rad]")
    ap.add_argument("--gear-probe-motor-rad", type=float, default=0.30,
                    help="gear 测试在电机空间走的弧度")
    ap.add_argument("--kp", type=float, default=8.0, help="单关节测试的位置增益（默认很低）")
    ap.add_argument("--kd", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=1.5, help="每个动作持续时长 [s]")
    ap.add_argument("--samples", type=int, default=50, help="offsets 采样次数")

    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="纯静态检查映射表（不连硬件）")
    sub.add_parser("scan", help="扫描总线 CAN ID（只读）")
    sub.add_parser("read", help="读 12 关节状态（只读）")
    sub.add_parser("wiggle", help="单关节微摆，确认接的是哪个关节")
    sub.add_parser("direction", help="判定 sign")
    sub.add_parser("gear", help="判定 gear_ratio")
    sub.add_parser("offsets", help="零位标定")

    args = ap.parse_args()
    # args.map 为 None 时显式传进去会覆盖 MotorMapping 的默认值 → 在这里统一解析
    from interface.motor_mapping import DEFAULT_MAP_PATH
    if args.map is None:
        args.map = DEFAULT_MAP_PATH
    fn = {"check": cmd_check, "scan": cmd_scan, "read": cmd_read,
          "wiggle": cmd_wiggle, "direction": cmd_direction,
          "gear": cmd_gear, "offsets": cmd_offsets}[args.cmd]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
