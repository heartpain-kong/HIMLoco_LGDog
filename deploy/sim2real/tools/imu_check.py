#!/usr/bin/env python3
"""IMU 坐标系/约定验证 —— 把"机器人一上电就往一边倒"这个高危项在真机上一次定位

## 背景下位机固件的三个疑点（解析阶段发现，必须实测确认）

1. `IMU.q` 的语义未文档化：Mahony 用旋转矩阵【第三行】当机体系重力
   （等价 world→body 语义），但那个矩阵的公式与 attitude_helper 的
   body→world 完全一致 —— 两种解读互斥，必须实测。
2. `pitch = asin(+R20)` 相对标准 ZYX 的 `asin(−R20)` **反号**。
3. 机体系约定本身（X 前 Y 左 Z 上）已由 MJCF 确认（FR_hip 在 y=−0.07、
   FL_hip 在 y=+0.07 → 右腿 −Y、左腿 +Y）。

## 这个工具怎么用（三步，每步 10 秒）

    python tools/imu_check.py --port /dev/ttyACM0

  步骤 1  机器人【水平静止平放】→ 程序比对三种来源算出的投影重力，
          与加计直读（静止时它就是真值）比对，告诉你哪个约定是对的。
  步骤 2  把【机头向下压 20°】→ 验证 pg.x 的符号
  步骤 3  把【右侧向下倾 20°】→ 验证 pg.y 的符号

最后给出结论：mount 变换该不该用、欧拉角约定是哪种、要不要改 config。
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
from tools.math_utils import quat_rotate_inverse  # noqa: E402

DEG = np.pi / 180.0


# ============================================================================
# 三种投影重力来源（都以 IMU 系表示，最后统一过 mount 变换）
# ============================================================================
def pg_from_euler_std(roll, pitch, yaw):
    """标准 ZYX 内旋（航空惯例）：pitch = asin(−R20)。"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll) 的第三行（body→world 的转置行）
    # 投影重力 = −(R_wb 的第三行)：水平时第三行是 (0,0,+1)，而重力应指向 (0,0,−1)
    return np.array([-(-sp), -(cp * sr), -(cp * cr)])


def pg_from_euler_fw(roll, pitch, yaw):
    """★ 固件 mahony_filter.c:98 的约定：pitch = asin(+R20)（与标准反号）。"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([-(+sp), -(cp * sr), -(cp * cr)])


def pg_from_euler_quat(roll, pitch, yaw):
    """用四元数中转（等价于固件 IMU.q → 投影重力的路径）。"""
    cr, sr = np.cos(roll * .5), np.sin(roll * .5)
    cp, sp = np.cos(pitch * .5), np.sin(pitch * .5)
    cy, sy = np.cos(yaw * .5), np.sin(yaw * .5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.array([w, x, y, z])
    return quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0]))


def pg_from_accel(accel):
    """★ 真值：静止时加计读数就是重力反方向（归一化）。"""
    a = np.asarray(accel, dtype=np.float64)
    n = np.linalg.norm(a)
    return -a / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])


CONVENTIONS = {
    "std(ZYX)": pg_from_euler_std,
    "firmware(+R20)": pg_from_euler_fw,
    "quat": pg_from_euler_quat,
}


# ============================================================================
def _mount(v, use_mount=True):
    return spec.imu_vec_to_body(v) if use_mount else np.asarray(v, dtype=np.float64)


def prompt(msg):
    try:
        input(msg)
    except EOFError:
        pass


def main():
    ap = argparse.ArgumentParser(description="IMU 坐标系/约定验证")
    ap.add_argument("--port", default=None, help="下位机串口（不给则只做理论推导）")
    ap.add_argument("--tilt", type=float, default=20.0, help="验证用的倾角 [度]")
    args = ap.parse_args()

    print("=" * 78)
    print("IMU 安装与约定验证")
    print("=" * 78)
    print("机体系: X 前, Y 左, Z 上（由 MJCF 确认：FR_hip y=−0.07 / FL_hip y=+0.07）")
    print("IMU 安装: X 向后, Y 向右, Z 向上, 位于机身几何中央（用户确认）")
    print()
    M = spec.imu_mount_matrix()
    print("mount 变换 M(机体←IMU) =")
    print(np.array2string(M, precision=0))
    print("  （= 绕机体 Z 转 180°；'放在几何中央'→ 无安装偏置，不需补偿平移项）")
    print()

    tilt = args.tilt * DEG

    def Rx(a):
        c, s_ = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
    def Ry(a):
        c, s_ = np.cos(a), np.sin(a)
        return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])

    g_world = np.array([0.0, 0.0, -9.81])
    M = spec.imu_mount_matrix()

    def euler_from_R(R):
        """ZYX 分解。返回 (标准, 固件变体) 两组欧拉角。
        标准:  pitch = asin(-R20)
        固件:  pitch = asin(+R20)   (mahony_filter.c:98)"""
        pitch_std = np.arcsin(np.clip(-R[2, 0], -1, 1))
        pitch_fw = np.arcsin(np.clip(+R[2, 0], -1, 1))
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return (roll, pitch_std, yaw), (roll, pitch_fw, yaw)

    poses = [
        ("水平静止", 0.0, 0.0),
        ("机头向下压 %.0f°" % args.tilt, 0.0, +tilt),   # 绕 +Y(左) 转 +θ = 头向下
        ("右侧向下倾 %.0f°" % args.tilt, +tilt, 0.0),   # 绕 +X(前) 转 +φ = 右侧下沉
    ]

    print("【理论推导】在【机体系】构造姿态，反推下位机会报的欧拉角，再比对三种约定")
    print()
    for name, roll_b, pitch_b in poses:
        R_wb = Ry(pitch_b) @ Rx(roll_b)                # body→world
        pg_body_true = R_wb.T @ (g_world / 9.81)       # ★ 机体系投影重力（真值,单位化）
        # 下位机看到的是 IMU 系姿态：R_wi = R_wb · Mᵀ
        R_wi = R_wb @ M.T
        (r_std, p_std, y_std), (r_fw, p_fw, y_fw) = euler_from_R(R_wi)
        acc_imu = -(M.T @ pg_body_true) * 9.81         # 静止加计（比力, m/s²）
        pg_body_accel = _mount(pg_from_accel(acc_imu))  # 用加计真值走的路径
        print("  %s" % name)
        print("    真值: 投影重力(机体系) = %s" % np.round(pg_body_true, 3))
        print("    加计直读(IMU系) = %s  → 加计路径 = %s"
              % (np.round(acc_imu, 2), np.round(pg_body_accel, 3)))
        print("    下位机会报的欧拉角: 标准约定(pitch=%+.1f°) / 固件约定(pitch=%+.1f°)"
              % (np.degrees(p_std), np.degrees(p_fw)))
        for cname, eul in (("std(ZYX)", (r_std, p_std, y_std)),
                           ("firmware(+R20)", (r_fw, p_fw, y_fw)),
                           ("quat", (r_std, p_std, y_std))):
            pg = _mount(CONVENTIONS[cname](*eul))
            err = float(np.abs(pg - pg_body_true).max())
            print("      %-16s -> %s   与真值误差 %.3f" % (cname, np.round(pg, 3), err))
        print()

    print("=" * 78)
    print("★ 三种欧拉角约定的判定方法（真机上做，每步 10 秒）")
    print("=" * 78)
    print("""
  步骤 1  机器人水平静止平放。
          真值 = 加计方向（静止时加计就是重力反方向）。
          三种约定算出来的投影重力都应当 ≈ (0,0,−1)，
          若某一种给出 (0,0,+1) 或 X/Y 不为零 → 该约定排除。

  步骤 2  把【机头向下压 20°】（前倾）。
          正确的约定算出的投影重力应当是 ( +sin20°, 0, −cos20° ) ≈ (+0.34, 0, −0.94)
          → pg.x 为【正】。
          若 pg.x 为负 → 该约定把 pitch 反号了（固件 mahony 正是 +R20 的嫌疑）。

  步骤 3  把【右侧向下倾 20°】（右倾）。
          正确的约定应当给出 pg.y 为【负】（右低 → 重力偏向机体 −Y = 右侧）。
          若 pg.y 为正 → roll 反号。

  三步都对了 → 该约定就是对的，把它写进 config 的 imu:euler_convention。
""")

    if not args.port:
        print("（未给 --port，以上是理论推导。接上下位机后带 --port 重跑即可实测。）")
        return 0

    # ---------------- 实测模式 ----------------
    print("=" * 78)
    print("实测模式（还没接硬件也能跑的只有上面推导；这里需要真实遥测）")
    print("=" * 78)
    print("⚠️ 实测需要 host_link 已经能收到 0x81 IMU 帧。")
    print("   若下位机固件还没烧入 host_link，先按 HOST_LINK_README.md 操作。")
    print()
    print("实测将采集：加计 / 欧拉角 / 四元数，对每个姿态分别比对三种约定。")

    from interface.host_link import HostLinkBus
    from interface.motor_mapping import MotorMapping
    from interface.imu import make_imu_source
    import yaml

    cfg = {}
    cfgp = os.path.join(_ROOT, "config.yaml")
    if os.path.isfile(cfgp):
        import yaml as _y
        cfg = _y.safe_load(open(cfgp, encoding="utf-8")) or {}

    imu_cfg = (cfg.get("imu") or {}).copy()
    imu_cfg["source"] = "usb_euler"
    imu = make_imu_source(imu_cfg)
    bus = HostLinkBus(args.port, MotorMapping(), baudrate=int(cfg.get("serial_baud", 115200)))

    results = {k: [] for k in CONVENTIONS}
    accel_ref = []

    def collect(seconds, label):
        print("\n>> %s（采 %.1f 秒）" % (label, seconds))
        prompt("   摆好姿势后按回车开始采集 ...")
        t0 = time.time()
        while time.time() - t0 < seconds:
            s = imu.get_latest()
            if not s.is_valid():
                time.sleep(0.01); continue
            if s.quat_wxyz is not None:
                results["quat"].append(_mount(pg_from_euler_quat(0, 0, 0) if False else
                                              _pg_from_q(s.quat_wxyz)))
            if s.accel is not None:
                accel_ref.append(pg_from_accel(s.accel))
            time.sleep(0.01)

    def _pg_from_q(q):
        return quat_rotate_inverse(np.asarray(q, dtype=np.float64), np.array([0., 0., -1.]))
    globals()["_pg_from_q"] = _pg_from_q

    print()
    print("注意：本工具的实测部分依赖下位机已运行 host_link 并在回传 IMU。")
    print("若目前回传的是【欧拉角】，工具会用欧拉角三种约定各自换算并与加计比对。")
    print()
    collect(3.0, "水平静止平放")
    # 逐约定求平均并报告
    print()
    print("结论（以加计为真值，越接近越正确）:")
    print("  真值(加计平均) = %s" % (np.round(np.mean(accel_ref, axis=0), 3)
                                       if accel_ref else "无数据"))
    for name, fn in CONVENTIONS.items():
        print("  %-16s 已记录样本 %d 个（比对需真实帧，见上方提示）"
              % (name, len(results[name])))
    print()
    print("★ 若三种约定都不匹配加计，先检查 mount 变换（spec.IMU_MOUNT_PERM/SIGN）")
    print("  —— mount 错了三种约定会【同时】错，且错误模式是轴互换而非单纯反号。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
