#!/usr/bin/env python3
"""真机后端 —— PC(N100) ──Type-C──> 下位机(dm02/STM32H7) ──FDCAN1/2──> 12×RS-02

拓扑（用户确认，2026-09）：
    上位机不直接碰 CAN。下位机承担三件事：
      ① 把 PC 下发的 MIT 指令中继到两条 FDCAN
      ② 把 IMU（欧拉角 + 陀螺仪角速度 + 四元数）回传
      ③ 把 12 个电机的 pos/vel/tau/temp/err 回传
    协议实现在 interface/host_link.py（与固件 dm02/User_File/task/lgdog_link.h 逐字节对应）。

这个后端把三件事拼成 main.py 需要的 RobotIO 接口（read_state / write_targets / …），
并在中间补上一层【IMU 安装变换】（interface/imu_mount.py）—— 少了这一层，观测里
的角速度与投影重力都是 IMU 芯片坐标系的量，机器人一上电就会往一边倒。

★ 两个必须用真机才能确认的量（这里不猜，全部由 config 暴露 + 运行时自检报告）：
    1) 欧拉角是度还是弧度        → imu.euler_unit_is_degree
    2) 姿态该信欧拉角还是信四元数 → imu.attitude_source
  启动时会把"欧拉角推的姿态"与"下位机四元数"的夹角打出来（同一姿态应≈0°），
  并用静止时的陀螺仪模长判断单位写得对不对。这两条是上电前最该看的数字。
"""

import os
import sys
import time
from contextlib import contextmanager

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import spec  # noqa: E402
from interface.host_link import (  # noqa: E402
    HostLinkBus, MODE_PC_CONTROL,
)
from interface.imu import ImuSample, ImuSource, euler_to_quat_wxyz  # noqa: E402
from interface.imu_mount import (  # noqa: E402
    ImuMountTransform, attitude_error_deg, quat_conj, quat_normalize,
)

#: 下位机 status 帧里 err_flags 的位定义（与固件 host_link.cpp 逐位对应）
#  ★ bit4 是"零点/符号/减速比填错"最直接的硬件侧证据：固件把上位机的目标（电机侧）
#    与它自己的 Pos_min/Pos_max 比对，越界就夹住并置这一位。它是【独立于我们假设】的参考系。
BOARD_ERR_BITS = {
    0x01: ("imu_not_ready", "IMU 未就绪（固件还在等姿态，前几秒正常）"),
    0x02: ("hb_timeout",    "心跳超时：下位机 100ms 没收到我们的心跳（它已经自己刹车了）"),
    0x04: ("tilt",          "下位机判摔倒：IMU |Roll| 或 |Pitch| > 60°"),
    0x10: ("target_clip",   "目标越出下位机软限位：sign / gear_ratio / zero_offset 与固件约定不一致"),
}

#: 这些位置一旦置位就必须停（bit0 是启动阶段的正常现象，不停）
BOARD_ERR_FATAL = (0x02, 0x04, 0x10)


def decode_board_err(flags: int):
    """把 err_flags 拆成 [(key, text), ...]。"""
    return [info for bit, info in sorted(BOARD_ERR_BITS.items()) if flags & bit]


MOTOR_ERR_BITS = {
    0x01: "电压过低", 0x02: "电压过高", 0x04: "过流", 0x08: "MOS 过温",
    0x10: "线圈过温", 0x20: "通信丢失", 0x40: "过载", 0x80: "未标定/其它",
}


# ============================================================================
class HostLinkImuSource(ImuSource):
    """把 HostLinkBus 收到的 ImuPayload 变成 ImuSample（★ 仍是 IMU 系，不是机体系）。

    :param bus:              HostLinkBus（后台线程已经在往里填 self.imu / last_imu_time）
    :param unit_is_degree:   下位机发的是度还是弧度。★ 固件实测是【弧度】
                             （imu_task.c 的 mahony.roll/pitch/yaw，且回绕判据用 3.1415926f）
    :param attitude_source:  "euler"（默认，用 roll/pitch/yaw 重建）| "quat"（直接用固件四元数）
    :param euler_sign:       逐字段符号 [sr, sp, sy]，用于修正固件的欧拉角符号约定
                             （已知风险：固件 pitch=asin(+R20) 疑与标准 ZYX 反号，
                              真机上用 tools/imu_check.py 拿加计当地面真值定一下）
    :param gyro_sign:        逐轴符号 [gx, gy, gz]
    """

    def __init__(self, bus: HostLinkBus, unit_is_degree: bool = False,
                 attitude_source: str = "euler", euler_sign=(1.0, 1.0, 1.0),
                 gyro_sign=(1.0, 1.0, 1.0), max_age_ms: float = 200.0):
        if attitude_source not in ("euler", "quat"):
            raise ValueError("attitude_source 只能是 euler / quat")
        self.bus = bus
        self.unit_is_degree = bool(unit_is_degree)
        self.attitude_source = attitude_source
        self.euler_sign = np.asarray(euler_sign, dtype=np.float64)
        self.gyro_sign = np.asarray(gyro_sign, dtype=np.float64)
        self.max_age_ms = float(max_age_ms)
        # ★ 注意：本类【不做】安装变换，输出的仍是 IMU 芯片坐标系的量。
        #   安装变换由 HostLinkRobotIO 在外面套一层 ImuMountTransform 完成
        #   （这样"协议解析"和"坐标系换算"两件事互不纠缠，也避免自己包自己的递归）

        # 诊断计数（进日志）
        self.frames_seen = 0
        self.frames_not_ready = 0
        self.last_crosscheck_deg = float("nan")
        self.last_crosscheck_quatT_deg = float("nan")

    # ---------------------------------------------------------------- 底层取
    def _raw_sample(self) -> ImuSample:
        """直接从 bus 的 ImuPayload 构造（不做安装变换）。"""
        with self.bus._lock:
            p = self.bus.imu
            last = self.bus.last_imu_time
        self.frames_seen += 1
        if not p.ready():
            # 固件用 flags bit0 表示"姿态已就绪"。imu_task.c 里前 ~3s 不给姿态，
            # 所以启动阶段要等它置位，不能拿 0 当水平。
            self.frames_not_ready += 1
            return ImuSample(source_tag="usb_not_ready", recv_time=last or time.monotonic())

        euler = np.array([p.roll, p.pitch, p.yaw], dtype=np.float64) * self.euler_sign
        if self.unit_is_degree:
            euler = np.radians(euler)
        gyro = np.asarray(p.gyro, dtype=np.float64) * self.gyro_sign
        accel = np.asarray(p.accel, dtype=np.float64)

        if self.attitude_source == "euler":
            quat = euler_to_quat_wxyz(float(euler[0]), float(euler[1]), float(euler[2]))
        else:
            quat = quat_normalize(np.asarray(p.quat, dtype=np.float64))

        # ---- 交叉校验：两种姿态来源差多少（同一姿态应≈0°）----
        q_e = euler_to_quat_wxyz(float(euler[0]), float(euler[1]), float(euler[2]))
        q_q = quat_normalize(np.asarray(p.quat, dtype=np.float64))
        self.last_crosscheck_deg = attitude_error_deg(q_e, q_q)
        # 固件如果给的是"世界→机体"（转置/共轭），夹角会接近 180°
        self.last_crosscheck_quatT_deg = attitude_error_deg(q_e, quat_conj(q_q))

        return ImuSample(gyro=gyro, quat_wxyz=quat, accel=accel,
                         source_tag="usb_euler" if self.attitude_source == "euler" else "usb_quat",
                         recv_time=last or time.monotonic())

    # ---------------------------------------------------------------- 接口
    def start(self, timeout: float = 8.0):
        """等固件把 flags bit0（姿态就绪）置起来。超时不抛异常，只警告 ——
        因为"姿态没就绪"时仍可以看电机反馈做标定，硬抛会挡住合法用法。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.bus._lock:
                ready = self.bus.imu.ready()
            if ready:
                return True
            time.sleep(0.05)
        print("[IMU] ⚠️  %.1fs 内 flags bit0 未置位（固件 imu_flag 还没起来）。"
              "姿态先按不可用处理；继续等待不会阻塞其它检查。" % timeout)
        return False

    def get_latest(self) -> ImuSample:
        """★ 返回的是 IMU 系样本。要机体系量请用 HostLinkRobotIO.imu（外面套了变换）。"""
        return self._raw_sample()

    def raw_sample(self) -> ImuSample:
        """IMU 系原始样本（诊断用）。"""
        return self._raw_sample()

    def stop(self):
        pass

    # ---------------------------------------------------------------- 诊断
    def sanity_report(self, n_samples: int = 20) -> dict:
        """静止状态下的自检，返回一串"人该看一眼"的数字。

        ★ 上电前必看：
          gyro_norm      静止时应 < 0.2 rad/s。若在 5~20 之间，说明下位机发的是【度/秒】，
                         config 里 euler_unit_is_degree 之外还得换算 gyro（见返回值 hints）。
          tilt_deg       静止时 roll/pitch 应 < 5°。若 > 80°，八成是度当成弧度了。
          crosscheck     欧拉角 vs 四元数夹角，应 ≈0°。≈180° 说明固件的 q 是"世界→机体"。
        """
        gyros, quats, eulers = [], [], []
        for _ in range(n_samples):
            s = self._raw_sample()
            if s.gyro is not None:
                gyros.append(np.asarray(s.gyro, dtype=np.float64))
            if s.quat_wxyz is not None:
                quats.append(np.asarray(s.quat_wxyz, dtype=np.float64))
            time.sleep(0.02)
        out = {
            "samples": len(gyros),
            "frames_seen": self.frames_seen,
            "frames_not_ready": self.frames_not_ready,
            "hints": [],
        }
        if gyros:
            g = np.mean(gyros, axis=0)
            out["gyro_mean"] = g
            out["gyro_norm"] = float(np.linalg.norm(g))
        if quats:
            # 静止姿态：用平均四元数（同号对齐后再平均，避免 w 变号把平均打飞）
            qs = np.array(quats)
            qs[qs[:, 0] < 0] *= -1.0
            qm = quat_normalize(qs.mean(axis=0))
            out["quat_mean"] = qm
            from tools.math_utils import get_gravity_orientation
            pg = get_gravity_orientation(qm)
            out["pg"] = np.asarray(pg, dtype=np.float64)
            out["tilt_deg"] = float(np.degrees(np.arccos(
                np.clip(-float(pg[2]), -1.0, 1.0))))
        out["crosscheck_deg"] = self.last_crosscheck_deg
        out["crosscheck_quatT_deg"] = self.last_crosscheck_quatT_deg

        if out.get("gyro_norm", 0.0) > 5.0:
            out["hints"].append("陀螺仪模长 %.2f 太大：下位机大概是【度/秒】，"
                                "需要在 config 的 imu 段加 gyro 单位换算（gyro_sign 不够用）"
                                % out["gyro_norm"])
        if out.get("tilt_deg", 0.0) > 20.0:
            out["hints"].append("静止倾角 %.1f°：机器人不是水平的（或单位/符号有误）"
                                % out["tilt_deg"])
        cc = out.get("crosscheck_deg", float("nan"))
        if cc == cc and cc > 5.0:
            if out.get("crosscheck_quatT_deg", 999.0) < 5.0:
                out["hints"].append("欧拉角与四元数差 %.1f°，但与四元数的【共轭】几乎一致 → "
                                    "固件 q 是『世界→机体』，即 p.quat 应取共轭后再用" % cc)
            else:
                out["hints"].append("欧拉角与四元数差 %.1f° → 固件的欧拉角/四元数至少有一个"
                                    "符号约定不对，用 tools/imu_check.py 拿加计定标" % cc)
        return out


# ============================================================================
class HostLinkRobotIO:
    """经下位机中转的真机 IO，实现 main.py 的 RobotIO 契约。

    :param port:     下位机的 USB CDC 串口（/dev/ttyACM0 …）
    :param mapping:  interface/motor_mapping.MotorMapping
    :param imu_cfg:  config.yaml 的 imu: 段
    :param verbose:  打印链路细节
    """

    def __init__(self, port: str, mapping, imu_cfg: dict = None, baudrate: int = 921600,
                 verbose: bool = False, connect_timeout: float = 8.0,
                 clip_to_fault: int = 5, gain_correction=None):
        imu_cfg = imu_cfg or {}
        self.port = port
        self.mapping = mapping
        self.verbose = bool(verbose)
        self.connect_timeout = float(connect_timeout)

        self.bus = HostLinkBus(port, mapping, baudrate=baudrate, verbose=verbose)
        # 协议层：只负责把 ImuPayload 解析成 ImuSample（IMU 系）
        self.imu_core = HostLinkImuSource(
            self.bus,
            unit_is_degree=bool(imu_cfg.get("euler_unit_is_degree", False)),
            attitude_source=imu_cfg.get("attitude_source", "euler"),
            euler_sign=imu_cfg.get("euler_sign", (1.0, 1.0, 1.0)),
            gyro_sign=imu_cfg.get("gyro_sign", (1.0, 1.0, 1.0)),
            max_age_ms=float(imu_cfg.get("max_age_ms", 200.0)),
        )
        # 坐标层：IMU 系 → 机体系（缺了这一层，机器人一上电就往一边倒）
        apply_mount = bool(imu_cfg.get("apply_mount_transform", True))
        self.imu = ImuMountTransform(self.imu_core) if apply_mount else self.imu_core
        self.apply_mount = apply_mount
        self.enabled = False
        # ★ PD 增益换算开关：关节侧 Kp/Kd → 电机侧 MIT 增益（除以 gear_ratio²）。
        #   小腿有 1:2 减速 → 除以 4；hip/thigh 无减速 → 不变。
        #   ★★ 2026-10-08 修：这个开关以前从 imu_cfg 里读，而 gain_correction 是
        #      hardware_map.yaml 的【顶层键】、调用方传进来的却是 cfg["imu"] 那一段
        #      （backends/__init__.py）⇒ 永远取不到 ⇒ 开关一直是摆设（改成 false 也照样除 4）。
        #      现在以【映射表 MotorMapping.gain_correction】为准：那是同一份
        #      hardware_map.yaml，一个开关管到底（main.py --check 打印的也是它）。
        if gain_correction is not None:
            self.gain_correction = bool(gain_correction)
        else:
            self.gain_correction = bool(getattr(mapping, "gain_correction",
                                                imu_cfg.get("gain_correction", True)))
        self._last_motor_err = np.zeros(spec.NUM_JOINTS, dtype=np.uint8)

        # ---- 目标越界（软限位）在线检测 ----
        # 我们自己也有一份窗口表（hardware_map.yaml: board_limits_deg），
        # 于是可以【逐帧】算出"刚发出去的目标会不会被板子夹住"，并在连续若干帧后升级为故障。
        # 与安全监视器的 clip_to_brake 同一个思路：单帧尖峰不管，持续越界说明映射错了。
        self.clip_to_fault = int(clip_to_fault)
        self.clip_frames = 0
        self.max_consecutive_clip = 0
        self.total_clip_frames = 0
        self.last_clip_joints = []
        self.board_err_seen = 0           # 见到过 err_flags 致命位的次数（下位机复位才清）

    # ---------------------------------------------------------------- 连接
    def start(self, timeout: float = None):
        timeout = self.connect_timeout if timeout is None else timeout
        self.bus.connect()
        print("[Link] 已打开 %s，等待上行帧（IMU + 两组电机状态）…" % self.port)
        self.bus.wait_feedback_ready(timeout)
        # 明确让下位机进入 PC 控制模式（固件默认就是，但显式更稳）
        self.bus.set_mode(MODE_PC_CONTROL)
        self.imu_core.start(timeout=min(3.0, timeout))
        st = self.bus.status
        if st.motor_missing:
            miss = [spec.JOINT_LABELS[i] for i in range(spec.NUM_JOINTS)
                    if st.motor_missing & (1 << i)]
            print("[Link] ⚠️  下位机报告启动时缺反馈的电机: %s" % ", ".join(miss))
        return True

    def wait_feedback_ready(self, timeout: float = 5.0):
        return self.bus.wait_feedback_ready(timeout)

    def describe(self) -> str:
        st = self.bus.status
        err_txt = "; ".join(t for _, t in decode_board_err(int(st.err_flags))) or "无"
        return ("真机后端 host_link：%s @ %d\n"
                "  下位机状态: mode=%d enabled=%d pc_alive=%d err_flags=0x%02X "
                "loop_us=%d rx=%d crc_err=%d tx_drop=%d missing=0x%03X\n"
                "  下位机 err_flags = 0x%02X（%s）\n"
                "  目标越界（我们自己算）: 连续 %d 帧 / 累计 %d 帧 %s\n"
                "  IMU: 单位=%s 姿态源=%s 安装变换=%s"
                % (self.port, self.bus.baudrate, st.mode, st.enabled, st.pc_alive,
                   st.err_flags, st.loop_us, st.rx_frames, st.crc_errors, st.tx_drops,
                   st.motor_missing, int(st.err_flags), err_txt,
                   self.clip_frames, self.total_clip_frames,
                   ("最近: " + ", ".join(self.last_clip_joints[:4])) if self.last_clip_joints else "",
                   "度" if self.imu_core.unit_is_degree else "弧度",
                   self.imu_core.attitude_source,
                   "开" if self.apply_mount else "关（调试）"))

    # ---------------------------------------------------------------- 使能
    def enable_motors(self, enable: bool):
        self.enabled = bool(enable)
        if self.enabled:
            self.bus.enable_all()
        else:
            self.bus.disable_all()

    def set_zero(self):
        """把当前电机位置记为下位机的零位（★ 只在机器人处于正确零位姿态时调用）。"""
        self.bus.set_zero()

    # ---------------------------------------------------------------- 读
    def read_state(self):
        q, dq, tau, temp, fresh, imu_age_ms = self.bus.read()
        s = self.imu.get_latest()                    # ★ 已做安装变换（机体系）

        now = time.monotonic()
        motor_age_ms = ((now - self.bus.last_motor_time) * 1000.0
                        if self.bus.last_motor_time else 1e9)
        st = self.bus.status

        err = np.asarray(getattr(self.bus, "_m_err", np.zeros(spec.NUM_JOINTS)),
                         dtype=np.uint8).copy()
        if np.any(err != self._last_motor_err):
            for i in range(spec.NUM_JOINTS):
                if err[i] and err[i] != self._last_motor_err[i]:
                    bits = " | ".join(MOTOR_ERR_BITS.get(1 << b, "bit%d" % b)
                                      for b in range(8) if err[i] & (1 << b))
                    print("[MOTOR] ⚠️  %s err=0x%02X (%s)"
                          % (spec.JOINT_LABELS[i], err[i], bits or "未知"))
            self._last_motor_err = err

        return {
            "q": q,
            "dq": dq,
            "quat_wxyz": np.asarray(s.quat_wxyz if s.quat_wxyz is not None
                                    else [1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "omega": np.asarray(s.gyro if s.gyro is not None else np.zeros(3),
                                dtype=np.float64),
            "imu_age_ms": float(imu_age_ms),
            "tau": tau,
            "temp": temp,
            "link_alive": bool(motor_age_ms < 200.0),
            "enabled": bool(self.enabled),
            "rx_frames": int(st.rx_frames),
            "tx_drops": int(st.tx_drops),
            "motor_missing": int(st.motor_missing),
            # 真机特有诊断（干跑后端没有的键，日志按需取）
            "motor_age_ms": float(motor_age_ms),
            "fresh": fresh,
            "motor_err": err,
            "board_err_flags": int(st.err_flags),
            "board_err_text": [t for _, t in decode_board_err(int(st.err_flags))],
            "tgt_clip_joints": list(self.last_clip_joints),
            "tgt_clip_frames": int(self.clip_frames),
            "tgt_clip_total": int(self.total_clip_frames),
            "backend_fault": self._backend_fault(int(st.err_flags)),
            "board_loop_us": int(st.loop_us),
            "pad_alive": None,
            "imu_source_tag": s.source_tag,
            "imu_crosscheck_deg": float(self.imu_core.last_crosscheck_deg),
            "imu_crosscheck_quatT_deg": float(self.imu_core.last_crosscheck_quatT_deg),
        }

    # ---------------------------------------------------------------- 链路速率
    def link_rate_report(self, seconds: float = 2.0):
        """量【上行】的实际速率与"最新值有多旧"。

        ★ 为什么要量：固件标称 IMU 200Hz / 电机状态 100Hz，但那是【发出】的速率。
          实际到达速率取决于 USB CDC 吞吐、PC 侧读线程、以及有没有丢帧。
          如果 IMU 实际只到 60Hz，那么即使控制环严格 50Hz，obs[3:6] 用的角速度
          也可能已经"老"了 16ms —— 这是训练时（Isaac 每步都是当前值）没有的延迟。

        ★ 三个数字的分工：
            imu_hz / motor_hz      实际吞吐（丢帧就能看出来）
            imu_max_age_ms         采样时刻最坏能拿到多旧的数据（丢帧/批处理都能看出来）
            clock_ratio            下位机 1ms tick 与上位机 monotonic 的比值，应≈1.000
        """
        with self._lock_guard():
            i0, m0, s0 = self.bus.imu_frames, self.bus.motor_frames, self.bus.status_frames
            tk0 = int(self.bus.imu.tick_ms)
        t0 = time.monotonic()
        imu_ages, motor_ages = [], []
        while True:
            self.bus.poll()                      # 心跳照发，不然下位机会自己刹车
            now = time.monotonic()
            imu_ages.append((now - self.bus.last_imu_time) * 1000.0
                            if self.bus.last_imu_time else 1e9)
            motor_ages.append((now - self.bus.last_motor_time) * 1000.0
                              if self.bus.last_motor_time else 1e9)
            if now - t0 >= seconds:
                break
            time.sleep(0.001)
        dt = time.monotonic() - t0
        with self._lock_guard():
            i1, m1, s1 = self.bus.imu_frames, self.bus.motor_frames, self.bus.status_frames
            tk1 = int(self.bus.imu.tick_ms)
        # 板子的 tick 是 uint32 毫秒，用 int32 差值免掉回绕问题
        d_tk = ((tk1 - tk0) & 0xFFFFFFFF)
        if d_tk >= 0x80000000:
            d_tk -= 0x100000000
        return {
            "seconds": dt,
            "imu_hz": (i1 - i0) / dt,
            "motor_hz": (m1 - m0) / dt,
            "status_hz": (s1 - s0) / dt,
            "imu_max_age_ms": float(np.max(imu_ages)) if imu_ages else float("nan"),
            "motor_max_age_ms": float(np.max(motor_ages)) if motor_ages else float("nan"),
            "clock_ratio": (d_tk / 1000.0) / dt if dt > 0 else float("nan"),
            # ★ 电机状态是【两条总线各一帧】：固件 LGD_TX_MOTOR_PERIOD_MS=10 →
            #   每 10ms 发 2 帧（MSG_MOTOR_STATE + MSG_MOTOR_STATE_HI），
            #   所以帧率是 200Hz，而"每个电机状态被更新的周期"是 10ms。
            #   这里量的是帧率；每个电机的新鲜度看 motor_max_age_ms。
            "expected": {"imu_hz": 200.0, "motor_hz": 200.0, "status_hz": 10.0},
        }

    @contextmanager
    def _lock_guard(self):
        with self.bus._lock:
            yield

    # ---------------------------------------------------------------- 故障汇总
    def _backend_fault(self, flags: int) -> str:
        """把"硬件自己报的故障"汇总成一句话，交给 main.py 的守护层去刹车。

        ★ 为什么必须由守护层处理（而不是只在日志里记一笔）：
          · err_flags bit4 表示板子已经在【夹住目标】，策略的闭环假设失效了 ——
            继续跑等于让策略对着一个"实际到不了的位置"积分，行为不可预测。
          · bit2 心跳超时说明板子已经自己刹车了，我们还以为在控制它。
          · bit1 摔倒（固件用 IMU 判的，60° 阈值）比我们 guard 的判据更早更糙，
            两个独立实现同时报警是好事，不该只记日志。
        """
        items = []
        if flags & 0x02:
            items.append("下位机心跳看门狗超时（板子已自行刹车）")
        if flags & 0x04:
            items.append("下位机判摔倒（IMU |Roll|/|Pitch| > 60°）")
        if flags & 0x10:
            # ★ 新版固件已删除软限位 → 还能看到 bit4 就说明【板子跑的还是旧固件】。
            #   这条提示很有用：用户以为刷了新固件、其实没有（或刷错工程）。
            items.append("下位机报 err_flags bit4（软限位越界）—— 板子跑的还是【旧固件】？"
                         "新版已删除电机软限位，正常不该出现这一位")
            self.board_err_seen += 1
        if self.clip_frames >= self.clip_to_fault:
            items.append("目标连续 %d 帧越出软限位（%s）"
                         % (self.clip_frames, ", ".join(self.last_clip_joints[:4])))
        return "；".join(items)

    # ---------------------------------------------------------------- 写
    def _gains_to_motor(self, kp, kd):
        if not self.gain_correction:
            return kp, kd
        return self.mapping.kp_kd_to_motor(kp, kd)

    def write_targets(self, q_target, kp, kd, tau_ff=None):
        kp, kd = self._gains_to_motor(kp, kd)
        # ★ 先算"这个目标会不会被板子夹住"——用的是我们自己的窗口表 + 映射表。
        #   这不是重复劳动：板子的位是【闩锁】的（只有下位机复位才清），
        #   我们这份是【逐帧】的，能给出"哪一帧、哪个关节、差多少"的精确定位。
        q_target = np.asarray(q_target, dtype=np.float64)
        m = self.mapping
        # ★ 只在新固件还没删除软限位时才做这个"逐帧越界"检测：
        #   固件已无软限位 → 板子不会夹目标，板侧那条独立证据链不存在了，
        #   再按窗口算就是拿一张作废的表在报假警。
        if getattr(m, "board_limits_active", True) and np.all(np.isfinite(m.board_lo)):
            motor = m.sim_to_real(q_target)
            bad = np.where((motor > m.board_hi + 1e-6) | (motor < m.board_lo - 1e-6))[0]
            if bad.size:
                self.clip_frames += 1
                self.total_clip_frames += 1
                self.max_consecutive_clip = max(self.max_consecutive_clip, self.clip_frames)
                self.last_clip_joints = [spec.JOINT_LABELS[int(i)] for i in bad]
                if self.clip_frames == self.clip_to_fault or self.clip_frames == 1:
                    over = ["%s %+.0f°→窗[%+.0f°,%+.0f°]"
                            % (spec.JOINT_LABELS[int(i)], np.degrees(motor[int(i)]),
                               np.degrees(m.board_lo[int(i)]), np.degrees(m.board_hi[int(i)]))
                            for i in bad[:4]]
                    print("[HostLink] ⚠️  目标越出下位机软限位（第 %d 帧连续）: %s"
                          % (self.clip_frames, "; ".join(over)))
                    print("           下位机会把目标夹到边界 → 关节到不了指令位置 → 策略的"
                          "闭环假设失效。根因见 hardware_map.yaml 的 board_limits_deg 注释。")
            else:
                self.clip_frames = 0
                self.last_clip_joints = []
        self.bus.write_targets(q_target, kp, kd, tau_ff)

    def hold_pose(self, q_target, kp, kd):
        kp, kd = self._gains_to_motor(kp, kd)
        self.bus.write_targets(q_target, kp, kd)

    def damping_brake(self, kd=None):
        self.bus.damping_brake(kd)

    def poll(self):
        self.bus.poll()

    def close(self):
        # HostLinkBus.disconnect() 自己会先发一帧阻尼刹车再关串口
        try:
            self.enable_motors(False)
        except Exception:
            pass
        self.bus.disconnect()


# ============================================================================
def _selftest():
    """不接真机：用假的 serial 把 HostLinkBus 骗过去，验证后端能收帧、能变换、能写。"""
    import struct
    import threading
    from interface import host_link as HL

    print("=" * 72)
    print("HostLinkRobotIO 自测（假串口，无硬件）")
    print("=" * 72)

    sent = []
    q12 = np.array([0.0, -0.4, 0.5] * 4)

    class FakeSerial:
        def __init__(self, *a, **k):
            self.timeout = 0.01
            self.buf = bytearray()
            self._stop = False
            self._t = threading.Thread(target=self._produce, daemon=True)
            self._t.start()

        def _produce(self):
            tick = 0
            while not self._stop:
                tick += 1
                # IMU：机体系下水平静止，但★故意按"IMU 装反"来发（绕 Z 转 180°）
                # 即真机的 IMU 系里 roll/pitch/yaw 应为 (0,0,π)
                m = ImuSample(gyro=np.array([0.0, 0.0, 0.0]),
                              quat_wxyz=np.array([0.0, 0.0, 0.0, 1.0]))
                imu_payload = struct.pack(
                    "<13fIB", 0.0, 0.0, np.pi,
                    0.0, 0.0, 0.0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0, 1.0, tick, 1)
                self.buf += HL.encode(HL.SOF_BOARD2HOST, HL.MSG_IMU, imu_payload)
                for base in (0, 6):
                    mc = (struct.pack("<I", tick)
                          + struct.pack("<6f", *q12[base:base + 6])
                          + struct.pack("<6f", *np.zeros(6))
                          + struct.pack("<6f", *np.zeros(6))
                          + struct.pack("<6f", *np.full(6, 30.0))
                          + bytes(6))
                    mid = HL.MSG_MOTOR_STATE if base == 0 else HL.MSG_MOTOR_STATE_HI
                    self.buf += HL.encode(HL.SOF_BOARD2HOST, mid, mc)
                st = struct.pack("<4BH5I", 1, 1, 1, 0, 0, tick * 100, tick, 0, 0, tick)
                self.buf += HL.encode(HL.SOF_BOARD2HOST, HL.MSG_STATUS, st)
                time.sleep(0.005)

        @property
        def in_waiting(self):
            return len(self.buf)

        def read(self, n=1):
            n = min(n, len(self.buf)) or 1
            if not self.buf:
                time.sleep(0.002)
                return b""
            out = bytes(self.buf[:n])
            del self.buf[:n]
            return out

        def write(self, data):
            sent.append(bytes(data))
            return len(data)

        def close(self):
            self._stop = True

    # pyserial 只在这个自测里被"假装"用到：没装也照样能验证后端逻辑
    # （真机上当然必须装，install.sh 会装）
    try:
        import serial
    except ImportError:
        import types
        serial = types.ModuleType("serial")
        sys.modules["serial"] = serial
        print("  （本机没装 pyserial，用假模块替代；这只影响自测，不影响真机）")

    orig = getattr(serial, "Serial", None)
    serial.Serial = FakeSerial
    try:
        from interface.motor_mapping import MotorMapping
        m = MotorMapping()
        io = HostLinkRobotIO("/dev/fake", m, {"attitude_source": "euler"},
                             verbose=False, connect_timeout=2.0)
        io.start(timeout=2.0)
        io.enable_motors(True)
        io.write_targets(spec.DEFAULT_DOF_POS, spec.KP, spec.KD)
        st = None
        for _ in range(20):
            io.poll()                      # 心跳（真机主循环也是每周期调一次）
            st = io.read_state()
            time.sleep(0.01)

        print("  q   = %s" % np.round(st["q"], 4))
        print("  quat= %s   omega=%s" % (np.round(st["quat_wxyz"], 6), st["omega"]))
        print("  imu_age_ms=%.1f  link_alive=%s  rx=%d  temp_max=%.1f"
              % (st["imu_age_ms"], st["link_alive"], st["rx_frames"], st["temp"].max()))
        print("  欧拉角 vs 四元数交叉校验 = %.2f°（应为 0）" % st["imu_crosscheck_deg"])

        # ★ 关键断言：下位机发的是"IMU 装反"的姿态，经安装变换后必须是水平的
        from tools.math_utils import quat_rotate_inverse
        pg = quat_rotate_inverse(st["quat_wxyz"], np.array([0.0, 0.0, -1.0]))
        print("  变换后 pg = %s（应 ≈ (0,0,-1)）" % np.round(pg, 6))
        assert np.allclose(pg, [0, 0, -1], atol=1e-6), \
            "IMU 安装变换没生效：IMU 系里的 180° 偏航没有换算成机体系的水平姿态"
        assert abs(st["imu_crosscheck_deg"]) < 1e-3, "欧拉角与四元数应当一致"

        # q 的映射：下位机上报的是【电机侧】角度，回读到关节空间要按映射表换算。
        # 映射表里 calf 的 gear_ratio=2.0（URDF 证据：calf 速度上限是 hip/thigh 的一半），
        # 所以电机侧的 0.5 换回关节侧应当是 0.25，不能直接相等。
        q_expect = m.real_to_sim(q12)
        print("  期望关节角 = %s（calf 带 2:1 减速 → 电机侧 0.5 换回 0.25）"
              % np.round(q_expect, 4))
        assert np.allclose(st["q"], q_expect, atol=1e-3), "关节映射回读不对"
        assert st["rx_frames"] > 0, "下行状态的 rx_frames 没读到"

        # ★ PD 增益换算：按 hardware_map.yaml 的 gain_correction 开关走
        #   2026-10-08 决定部署侧【不换算】（以现场树实机标定表为准，见 hardware_map.yaml 的说明），
        #   所以这里不再写死 ÷4 —— 期望值一律由开关与 spec 推导，两种设置都能自测通过。
        import interface.host_link as HL2
        mit = None
        for f in sent:
            msgs = HL2.LinkParser(HL2.SOF_HOST2BOARD).parse(f)
            if msgs and msgs[0].msg_id == HL2.MSG_MIT_CMD:
                mit = msgs[0].payload
        assert mit is not None
        import struct as _st
        kp_sent = np.array(_st.unpack_from("<12f", mit, 48))
        kd_sent = np.array(_st.unpack_from("<12f", mit, 96))
        # ★ 期望值一律从 spec 推导，不写死数字 —— 2026-10-02 spec.KD 由 0.8 改成 1.2 时，
        #   写死 0.8/0.2 的旧写法当场变红（正是这类"改了常量没改测试"的漂移要暴露的）。
        _gc = bool(m.gain_correction)
        _f = 4.0 if _gc else 1.0
        exp_kp_ht, exp_kd_ht = spec.KP, spec.KD                       # gear=1 → 不变
        exp_kp_calf, exp_kd_calf = spec.KP / _f, spec.KD / _f         # gear=2 → 除以 gear²（开关关掉就不除）
        print("  下发到电机的 MIT 增益（关节侧 Kp=%.2f/Kd=%.2f，gain_correction=%s）:"
              % (spec.KP, spec.KD, _gc))
        print("    hip/thigh : Kp=%.2f Kd=%.2f" % (kp_sent[0], kd_sent[0]))
        print("    calf      : Kp=%.2f Kd=%.2f  （gear=2 → %s）"
              % (kp_sent[2], kd_sent[2], "除以 4" if _gc else "不换算（2026-10-08 决定）"))
        assert abs(kp_sent[0] - exp_kp_ht) < 1e-6 and abs(kd_sent[0] - exp_kd_ht) < 1e-6, \
            "hip/thigh 的增益不该被改（gear=1）"
        assert abs(kp_sent[2] - exp_kp_calf) < 1e-6 and abs(kd_sent[2] - exp_kd_calf) < 1e-6, \
            "calf 的增益与 gain_correction=%s 不符（应为 %.3f / %.3f，实际 %.3f / %.3f）" % (
                _gc, exp_kp_calf, exp_kd_calf, kp_sent[2], kd_sent[2])

        io.damping_brake()
        # 心跳 + MIT 指令 + 刹车，至少各发过一帧
        ids = [HL.LinkParser(HL.SOF_HOST2BOARD).parse(f)[0].msg_id for f in sent]
        print("  已发出帧类型: %s" % sorted(set(ids)))
        assert HL.MSG_MIT_CMD in ids and HL.MSG_HEARTBEAT in ids and HL.MSG_DAMPING_BRAKE in ids
        io.close()
        print()
        print("✅ host_link_backend 自测通过")
    finally:
        serial.Serial = orig


if __name__ == "__main__":
    _selftest()
