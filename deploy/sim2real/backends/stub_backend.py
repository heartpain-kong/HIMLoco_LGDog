#!/usr/bin/env python3
"""干跑后端 —— 不接任何硬件的"假机器人"，用来在 N100 上把整条链路跑通

★★ 移植到新机器时，这个后端是你的第一个验证目标 ★★
   在真实电机通电之前，先用它确认：
     1) Python 环境 / torch 策略能加载（270 维输入、12 维输出）
     2) 观测构造、状态机、起立流程、安全监视、日志 全部走得通
     3) 故障注入能真的触发刹车（见下面 FAULTS）
   它【只依赖 numpy/stdlib】，不需要 pyserial、不需要手柄、不需要真机映射表。

它实现的是 RobotIO 接口（与 backends/host_link_backend.py 完全同形）：

    start(timeout) / read_state() / write_targets(q,kp,kd,tau_ff=None)
    hold_pose(q,kp,kd) / damping_brake() / enable_motors(bool)
    poll() / wait_feedback_ready(timeout) / close() / describe()

read_state() 返回的 dict（★ 这是全栈的公共契约，真机后端必须给齐同样的键）：

    q            (12,)  关节角 [rad]，Isaac 顺序（spec.JOINT_LABELS）
    dq           (12,)  关节速度 [rad/s]
    quat_wxyz    (4,)   机身姿态（机体系 → 世界系），已归一化
    omega        (3,)   机体系角速度 [rad/s]（★ 已做 IMU 安装变换）
    imu_age_ms   float  IMU 数据龄 [ms]（安全判据用，取【真实到达时刻】算）
    tau          (12,)  关节侧力矩估计 [Nm]（只用于日志/检查）
    temp         (12,)  电机温度 [°C]
    link_alive   bool   与下位机的链路是否活着
    enabled      bool   电机是否使能
    rx_frames    int    收到的上行帧数（链路诊断）
    tx_drops     int    发送失败/丢弃计数
    motor_missing int   缺反馈的电机掩码（真机由下位机上报；干跑恒 0）
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import spec  # noqa: E402
from interface.imu import ImuSample, ImuSource  # noqa: E402


# ============================================================================
class _StubImuSource(ImuSource):
    """干跑 IMU：姿态由外部给定（默认永远水平），数据龄由外部虚拟时钟驱动。

    ★ 时间用【虚拟时钟】而不是墙钟：干跑的自测循环不按控制周期节拍跑，
      用墙钟的话 50 步只花几微秒，"IMU 停更 200ms" 这种故障注入永远不会触发。
      虚拟时钟每 read_state 加一个 dt，于是"停更 10 步 = 200ms"是确定性的。
      为了让 ImuSample.age_ms() 仍然有意义，recv_time 记成
      monotonic() - 虚拟数据龄。
    """

    def __init__(self, tilt_deg=(0.0, 0.0), gyro_noise=0.0, seed=0):
        self.tilt_deg = (float(tilt_deg[0]), float(tilt_deg[1]))
        self.gyro_noise = float(gyro_noise)
        self._rng = np.random.default_rng(seed)
        self._age_s = 0.0        # 虚拟数据龄，由 StubRobotIO 维护

    @staticmethod
    def _quat_from_rpy(roll, pitch, yaw=0.0):
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        return np.array([cr * cp * cy + sr * sp * sy,
                         sr * cp * cy - cr * sp * sy,
                         cr * sp * cy + sr * cp * sy,
                         cr * cp * sy - sr * sp * cy])

    def start(self, timeout: float = 8.0):
        self._age_s = 0.0

    def get_latest(self) -> ImuSample:
        g = self._rng.normal(scale=self.gyro_noise, size=3) if self.gyro_noise else np.zeros(3)
        return ImuSample(
            gyro=g,
            quat_wxyz=self._quat_from_rpy(*np.radians(self.tilt_deg)),
            accel=np.array([0.0, 0.0, 9.81]),
            source_tag="stub",
            recv_time=time.monotonic() - self._age_s)

    def stop(self):
        pass


# ============================================================================
class StubRobotIO:
    """一阶惯性假机器人。所有关节独立、无耦合，够用来验证链路与安全逻辑。

    :param q_start:        初始关节角（默认一个"趴着"的姿态，好让起立流程有活干）
    :param tau_joint:      关节一阶时间常数 [s]（越大越"软"）
    :param sag_rad:        重力下垂（每关节常量偏差），让静止时目标与实测有差异
    :param noise:          关节角/速度量测噪声 [rad]
    :param dt:             积分步长（= 控制周期 0.02s）
    :param faults:         故障注入 dict，见 FAULTS
    """

    #: 可用故障注入（值为触发时刻 [s]，None/0 表示关闭）
    FAULTS = ("nan_after_s", "imu_freeze_after_s", "motor_missing_after_s",
              "drop_link_after_s", "backend_fault_after_s")

    def __init__(self,
                 q_start=None,
                 tau_joint: float = 0.12,
                 sag_rad: float = 0.02,
                 noise: float = 2e-4,
                 dt: float = spec.CONTROL_DT,
                 tilt_deg=(0.0, 0.0),
                 faults: dict = None,
                 seed: int = 0,
                 verbose: bool = False):
        self.dt = float(dt)
        self.tau_joint = float(tau_joint)
        self.noise = float(noise)
        self.verbose = bool(verbose)
        self.faults = dict(faults or {})
        self._rng = np.random.default_rng(seed)

        # 初始姿态：默认"趴着"（hip 外张、thigh 前伸、calf 收紧）
        if q_start is None:
            q_start = np.array([0.0, -1.10, 0.65] * spec.NUM_LEGS, dtype=np.float64)
        self.q = np.clip(np.asarray(q_start, dtype=np.float64).reshape(spec.NUM_JOINTS),
                         spec.JOINT_LOWER, spec.JOINT_UPPER)
        self.dq = np.zeros(spec.NUM_JOINTS)
        self.q_cmd = self.q.copy()
        # 下垂：thigh/calf 各按 sag_rad 的分配（hip 不承重）
        self.sag = np.zeros(spec.NUM_JOINTS)
        self.sag[1::3] = float(sag_rad)
        self.sag[2::3] = -float(sag_rad)

        self.kp = np.zeros(spec.NUM_JOINTS)
        self.kd = np.zeros(spec.NUM_JOINTS)
        self.tau = np.zeros(spec.NUM_JOINTS)
        self.temp = np.full(spec.NUM_JOINTS, 26.0)
        self.enabled = False
        self.braked = False
        self.n_writes = 0
        self.tx_drops = 0
        self._t = 0.0            # ★ 虚拟时钟 [s]：每个 read_state 加一个 dt
        self._steps = 0
        self._imu_frozen = False
        self._motor_missing = 0

        self.imu = _StubImuSource(tilt_deg=tilt_deg)
        self._backend_fault = ""

    # ---------------------------------------------------------------- 连接
    def start(self, timeout: float = 8.0):
        self.imu.start(timeout)
        self._t = 0.0
        self._steps = 0
        if self.verbose:
            print("[Stub] 假机器人就绪（无硬件）")

    def wait_feedback_ready(self, timeout: float = 5.0):
        time.sleep(min(0.2, timeout))
        return True

    def describe(self) -> str:
        return ("干跑后端 stub：一阶惯性假机器人，无硬件依赖\n"
                "  初始姿态 = %s\n"
                "  关节时间常数 = %.3f s，重力下垂 = %s\n"
                "  故障注入 = %s"
                % (np.round(self.q, 3), self.tau_joint,
                   np.round(self.sag, 3), self.faults or "无"))

    # ---------------------------------------------------------------- 使能
    def enable_motors(self, enable: bool):
        self.enabled = bool(enable)
        if not self.enabled:
            self.kp[:] = 0.0
            self.kd[:] = 0.0

    # ---------------------------------------------------------------- 读
    def read_state(self):
        self._steps += 1
        self._t += self.dt          # ★ 虚拟时间前进一个控制周期
        t = self._t

        # ---- 故障注入（判据用虚拟时间，与是否按节拍跑无关）----
        fu = self.faults
        if fu.get("imu_freeze_after_s") and t > fu["imu_freeze_after_s"]:
            self._imu_frozen = True      # 数据龄从此每步累加，不再清零
        if self._imu_frozen:
            self.imu._age_s += self.dt
        else:
            self.imu._age_s = 0.0
        if fu.get("drop_link_after_s") and t > fu["drop_link_after_s"]:
            self.tx_drops += 1
            return self._state_dict(link_alive=False, rx_frames=0)
        if fu.get("motor_missing_after_s") and t > fu["motor_missing_after_s"]:
            self._motor_missing = 0x03F   # CAN1 的 6 个全丢
        if fu.get("backend_fault_after_s") and t > fu["backend_fault_after_s"]:
            self._backend_fault = ("下位机把目标夹到了软限位上"
                                   "（sign/gear_ratio/zero_offset 与固件约定不一致）")

        # ---- 关节响应：一阶惯性（kinematic）----
        # ★ 为什么不用"双积分 + PD 力矩"的真动力学：kp=30/kd=0.8 配单位惯量是
        #   欠阻尼的（ζ≈0.07，整定时间 ~10s），干跑时会一直小幅振荡，看起来像
        #   "控制环坏了"。干跑后端的职责是验证【链路/状态机/安全逻辑】，
        #   不是验证整定 —— 所以用必然稳定的一阶跟踪，另把 PD 律算出的力矩
        #   照常输出到 tau（供日志与力矩检查用，保证安全逻辑有真实输入）。
        if self.enabled and not self.braked:
            err = (self.q_cmd - self.sag) - self.q
            self.dq = err / self.tau_joint
            self.tau = np.clip(self.kp * err - self.kd * self.dq,
                               -spec.TAU_LIMIT, spec.TAU_LIMIT)
        else:
            self.tau = np.zeros(spec.NUM_JOINTS)
            self.dq = np.zeros(spec.NUM_JOINTS)
        if self.braked:
            # 阻尼刹车：速度指数衰减（电机限流后的典型表现）
            self.dq *= max(0.0, 1.0 - self.dt / 0.15)
        self.q = np.clip(self.q + self.dq * self.dt, spec.JOINT_LOWER, spec.JOINT_UPPER)
        # 温度随 |力矩| 缓慢上升（给日志一个像样的 temp_max）
        self.temp = np.clip(self.temp + np.abs(self.tau) * 2e-4 * self.dt, 0.0, 90.0)

        if fu.get("nan_after_s") and t > fu["nan_after_s"]:
            self.q[0] = np.nan      # 故障注入：NaN 进观测 → 必须立刻刹车

        return self._state_dict(link_alive=True, rx_frames=self._steps)

    def _state_dict(self, link_alive, rx_frames):
        s = self.imu.get_latest()
        q = self.q.copy()
        dq = self.dq.copy()
        if self.noise:
            q = q + self._rng.normal(scale=self.noise, size=spec.NUM_JOINTS)
            dq = dq + self._rng.normal(scale=self.noise * 10, size=spec.NUM_JOINTS)
        return {
            "q": q,
            "dq": dq,
            "quat_wxyz": np.asarray(s.quat_wxyz, dtype=np.float64),
            "omega": np.asarray(s.gyro, dtype=np.float64),
            "imu_age_ms": float(self.imu._age_s * 1000.0),
            "tau": self.tau.copy(),
            "temp": self.temp.copy(),
            "link_alive": bool(link_alive),
            "enabled": bool(self.enabled),
            "rx_frames": int(rx_frames),
            "tx_drops": int(self.tx_drops),
            "motor_missing": int(self._motor_missing),
            # 后端自报故障（真机后端由 err_flags / 目标越界产生）。
            # main.py 的 GuardedIO 会把它上升为守护急停 —— 这里注入就是为了测那条链。
            "backend_fault": self._backend_fault,
        }

    # ---------------------------------------------------------------- 写
    def write_targets(self, q_target, kp, kd, tau_ff=None):
        self.n_writes += 1
        self.q_cmd = np.clip(np.asarray(q_target, dtype=np.float64).reshape(spec.NUM_JOINTS),
                             spec.JOINT_LOWER, spec.JOINT_UPPER).copy()
        self.kp = np.broadcast_to(np.asarray(kp, dtype=np.float64), (spec.NUM_JOINTS,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, dtype=np.float64), (spec.NUM_JOINTS,)).copy()
        self.braked = False

    def hold_pose(self, q_target, kp, kd):
        self.write_targets(q_target, kp, kd)

    def damping_brake(self, kd=None):
        self.braked = True
        self.kp[:] = 0.0
        self.kd[:] = 2.5 if kd is None else float(kd)
        self.q_cmd = self.q.copy()

    def poll(self):
        """干跑无心跳需求；保留接口与真机后端同形。"""
        return

    def close(self):
        self.enable_motors(False)
        self.imu.stop()


# ============================================================================
def _selftest():
    from safety.runtime_guard import RuntimeGuard, GuardLevel

    print("=" * 72)
    print("StubRobotIO 自测")
    print("=" * 72)
    io = StubRobotIO()
    io.start()
    io.wait_feedback_ready()
    print(io.describe())
    print()

    io.enable_motors(True)
    # 下发默认站姿，跑 3 秒看是否收敛
    tgt = spec.DEFAULT_DOF_POS.copy()
    for _ in range(int(3.0 / spec.CONTROL_DT)):
        io.write_targets(tgt, spec.KP, spec.KD)
        st = io.read_state()
    err = float(np.max(np.abs(st["q"] - tgt)))
    print("收敛 3s 后 |q - target|max = %.4f rad（含 %.3f 静态下垂）" % (err, io.sag[1]))
    assert err < 0.1, "一阶模型没收敛"
    print("  一阶收敛 + 静止下垂 ✓")

    # guard 应判 OK
    g = RuntimeGuard()
    lvl, why = g.check(st)
    print("  guard.check -> %s %s" % (lvl.name, why or "-"))
    assert lvl in (GuardLevel.OK, GuardLevel.WARN), "正常状态不该停"

    # 干跑 IMU 的水平姿态 → pg 必须是 (0,0,-1)
    from tools.math_utils import quat_rotate_inverse
    pg = quat_rotate_inverse(st["quat_wxyz"], np.array([0.0, 0.0, -1.0]))
    print("  pg = %s" % np.round(pg, 6))
    assert np.allclose(pg, [0, 0, -1], atol=1e-6)

    # ---- 故障注入必须真的触发 STOP ----
    print()
    print("故障注入（确认安全链能真的跳闸）:")
    for label, faults, secs, expect in (
            ("NaN 进关节角", {"nan_after_s": 0.5}, 2.0, "NaN"),
            ("IMU 停更",     {"imu_freeze_after_s": 0.2}, 2.0, "IMU stale"),
            ("链路中断",     {"drop_link_after_s": 0.3}, 2.0, "链路"),
            ("后端自报故障", {"backend_fault_after_s": 0.3}, 2.0, "backend_fault"),
    ):
        io2 = StubRobotIO(faults=faults)
        io2.start()
        io2.enable_motors(True)
        g2 = RuntimeGuard()
        hit = ""
        for _ in range(int(secs / spec.CONTROL_DT)):
            io2.write_targets(tgt, spec.KP, spec.KD)
            s2 = io2.read_state()
            if not s2["link_alive"]:
                # 链路断了由主循环负责刹车；这里用 guard 只能测到数据类故障
                hit = "链路中断（由 main.py 的 link_alive 判据处理）"
                break
            if s2.get("backend_fault"):
                # 后端自报故障由 GuardedIO 转成守护急停，裸 guard 看不到，这里只确认它被带出来了
                hit = "backend_fault（由 main.py 的 GuardedIO 转成急停）"
                break
            lvl, why = g2.check(s2)
            if lvl is GuardLevel.STOP:
                hit = why
                break
        print("  %-14s -> %s" % (label, hit or "★没触发（故障注入失效）"))
        assert expect in hit, "%s 没被 guard 抓到" % label
        io2.close()

    # ---- 阻尼刹车后速度必须衰减 ----
    io3 = StubRobotIO(sag_rad=0.0)
    io3.start()
    io3.enable_motors(True)
    for _ in range(20):
        io3.write_targets(tgt, spec.KP, spec.KD)
        io3.read_state()
    v_before = float(np.max(np.abs(io3.dq)))
    io3.damping_brake()
    for _ in range(25):
        io3.read_state()
    v_after = float(np.max(np.abs(io3.dq)))
    print("  阻尼刹车: |dq|max %.4f -> %.4f" % (v_before, v_after))
    assert v_after < v_before * 0.2, "刹车没起作用"

    print()
    print("✅ stub_backend 自测通过")


if __name__ == "__main__":
    _selftest()
