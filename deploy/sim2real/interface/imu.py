#!/usr/bin/env python3
"""IMU 接入层 —— 数据由下位机处理成【欧拉角】后经 Type-C 直接返回

硬件事实（用户确认）：
    PC(N100) ──Type-C──> 下位机 ──CAN──> 12× 灵足 RS-02
    下位机自己读 IMU 并解算姿态，把【欧拉角】通过 Type-C 回传。
    同一条 Type-C 链路上还跑着电机的 CAN 透传。

═══════════════════════════════════════════════════════════════════════════
★ 关键缺口：策略需要【机体角速度】，但欧拉角给不出角速度
═══════════════════════════════════════════════════════════════════════════
策略的观测布局（spec.build_obs）：
    obs[3:6] = base_ang_vel(机体系) * 0.25     ← 必须来自陀螺仪
    obs[6:9] = projected_gravity               ← 可由欧拉角算出 ✓

只给欧拉角的话，obs[3:6] 会恒为 0（或不准），策略输入直接偏离训练分布。
这不是"精度差一点"的问题，是【观测缺一段】。

所以本模块提供三种 `source`：

  gyro_from_board  ★首选：下位机同时回传角速度（陀螺仪原始值或在板解算值）。
                   大部分下位机固件都会带，只是你没提到 —— 先确认一下。

  gyro_derived     ⚠️ 兜底：对欧拉角做差分反推机体角速度。
                   能让你先把机器人立起来做低速测试，但它有本质缺陷：
                     · 有量化噪声（下位机回传频率/分辨率决定）
                     · 有一拍延迟（差分需要两个样本）
                     · 快速摆动时误差大
                   训练时的 omega 是真陀螺仪；差分值与它分布不同，
                   所以【高速/动态动作会明显退化】。只当过渡手段用。

  stub             干跑/自检：常量水平姿态。

无论哪种，帧格式（帧头/帧类型/字段/量纲/欧拉角旋序）目前都【未知】，
不要猜 —— 先用 `tools/link_sniffer.py` 抓真实帧，确认后填 config.yaml。

为什么不能直接信下位机的欧拉角旋序：欧拉角有 12 种约定（内旋/外旋 × 排列），
猜错会让 projected_gravity 完全错位，机器人一上电就往一边倒。
"""

import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)


# 支持的欧拉角约定：名字 -> (索引顺序, 内旋?, 单位换算)
#   值 = (roll_idx, pitch_idx, yaw_idx) 在该帧字段里的位置
EULER_CONVENTIONS = {
    # 常见：roll, pitch, yaw 顺序排列
    "rpy_zyx_intrinsic": (0, 1, 2),   # 等价于 legged_gym/Isaac 常用的 ZYX 内旋
    "ypr_zyx_intrinsic": (2, 1, 0),   # 下位机常按 yaw,pitch,roll 发
}


@dataclass
class ImuSample:
    """一次 IMU 采样。字段为 None 表示下位机没提供该项。"""
    gyro: Optional[np.ndarray] = None       # 机体系角速度 [rad/s]  ← 策略必需
    quat_wxyz: Optional[np.ndarray] = None  # 机身姿态（w,x,y,z）  ← 投影重力用
    accel: Optional[np.ndarray] = None      # 机体系加速度 [m/s^2]（若下位机给）
    source_tag: str = ""
    recv_time: float = field(default_factory=time.monotonic)

    def age_ms(self) -> float:
        return (time.monotonic() - self.recv_time) * 1000.0

    def has_gyro(self) -> bool:
        return self.gyro is not None

    def has_attitude(self) -> bool:
        return self.quat_wxyz is not None or self.accel is not None


class ImuSource:
    """IMU 数据源接口。"""

    def start(self, timeout: float = 8.0):
        raise NotImplementedError

    def get_latest(self) -> ImuSample:
        raise NotImplementedError

    def stop(self):
        pass

    # 供 USB 链路层把非 CAN 帧喂进来（共享串口的实现覆盖它）
    def feed_usb_frame(self, frame: bytes) -> bool:
        return False


# ============================================================================
def euler_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX 内旋（先 yaw、再 pitch、最后 roll）→ 四元数 wxyz。

    与 legged_gym / Isaac 的约定一致：机身姿态由世界系到机体系用同一个 R。
    若下位机用的是外旋或别的排列，用 EULER_CONVENTIONS 换位，或在这里改公式。
    """
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


class UsbEulerImuSource(ImuSource):
    """解析下位机经 Type-C 回传的欧拉角帧。

    :param frame_id      : 帧类型/标识字节（★ 需实测，见 link_sniffer.py）
    :param euler_offset  : 欧拉角数据在帧里的字节偏移
    :param euler_dtype   : "f4" / "i2" / "i4"
    :param euler_scale   : 逐字段比例尺（i16 时必需，例如 deg->rad 用 0.01*pi/180）
    :param convention    : EULER_CONVENTIONS 的键
    :param gyro_offset   : 角速度字段的偏移；None 表示这帧没有角速度
    :param gyro_dtype/gyro_scale : 同上
    :param byteorder     : "<" 小端 / ">" 大端
    """

    def __init__(self, frame_id=None, euler_offset: int = 0, euler_dtype: str = "f4",
                 euler_scale=None, convention: str = "rpy_zyx_intrinsic",
                 gyro_offset=None, gyro_dtype: str = "f4", gyro_scale=None,
                 byteorder: str = "<", max_age_ms: float = 200.0):
        if convention not in EULER_CONVENTIONS:
            raise ValueError("未知欧拉角约定 %r，可选 %s"
                             % (convention, sorted(EULER_CONVENTIONS)))
        self.frame_id = frame_id
        self.euler_offset = int(euler_offset)
        self.euler_dtype = euler_dtype
        self.euler_scale = (np.ones(3) if euler_scale is None
                            else np.asarray(euler_scale, dtype=np.float64))
        self.convention = convention
        self.gyro_offset = gyro_offset
        self.gyro_dtype = gyro_dtype
        self.gyro_scale = (np.ones(3) if gyro_scale is None
                           else np.asarray(gyro_scale, dtype=np.float64))
        self.byteorder = byteorder
        self.max_age_ms = float(max_age_ms)

        self._latest = ImuSample(source_tag="usb_euler")
        self.frames_seen = 0
        self.frames_bad = 0
        self.has_gyro_field = gyro_offset is not None

    # ---------------------------------------------------------------- 解析
    def _read_vec(self, frame: bytes, offset, dtype, scale) -> np.ndarray:
        n = 3
        nbytes = n * np.dtype(dtype).itemsize
        seg = bytes(frame[offset:offset + nbytes])
        if len(seg) < nbytes:
            raise ValueError("帧太短: 需要 %d 字节 @%d，实际 %d" % (nbytes, offset, len(seg)))
        raw = np.frombuffer(seg, dtype=np.dtype(self.byteorder + dtype)).astype(np.float64)
        return raw * scale

    def feed_usb_frame(self, frame: bytes) -> bool:
        """喂一整帧（不含帧头/帧尾的载荷部分，或含也行，用 offset 定位）。"""
        if self.frame_id is not None and len(frame) > 0 and frame[0] != self.frame_id:
            return False
        self.frames_seen += 1
        try:
            e = self._read_vec(frame, self.euler_offset, self.euler_dtype, self.euler_scale)
        except Exception:
            self.frames_bad += 1
            return True

        idx = EULER_CONVENTIONS[self.convention]
        roll, pitch, yaw = e[idx[0]], e[idx[1]], e[idx[2]]
        sample = ImuSample(quat_wxyz=euler_to_quat_wxyz(roll, pitch, yaw),
                           source_tag="usb_euler", recv_time=time.monotonic())
        if self.has_gyro_field:
            try:
                sample.gyro = self._read_vec(frame, self.gyro_offset,
                                             self.gyro_dtype, self.gyro_scale)
            except Exception:
                sample.gyro = None
        self._latest = sample
        return True

    # ---------------------------------------------------------------- 接口
    def start(self, timeout: float = 8.0):
        print("[IMU] 等待下位机回传欧拉角帧 (frame_id=%s) ..." % self.frame_id)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._latest.has_attitude():
                print("[IMU] 收到首帧 ✓   角速度字段: %s"
                      % ("有" if self.has_gyro_field else "★ 没有（观测会缺一段！）"))
                return
            time.sleep(0.02)
        raise TimeoutError(
            "%.0fs 内没收到欧拉角帧。用 tools/link_sniffer.py 看 Type-C 链路上"
            "到底有哪些帧、帧头是什么。" % timeout)

    def get_latest(self) -> ImuSample:
        return self._latest

    def stop(self):
        pass


# ============================================================================
class DerivedGyroSource(ImuSource):
    """⚠️ 兜底：下位机只给欧拉角时，差分反推机体角速度。

    原理：由姿态四元数序列做有限差分得到机体角速度
        ω ≈ 2 * (q_k^{-1} ⊗ q_{k-1})_vec / Δt
    这是【近似】，有三个本质缺陷（决定了它只能做低速试验）：
        1. 有一拍延迟（要用到上一个样本）
        2. 有量化噪声，差分会放大
        3. Δt 取自本机接收时刻，串口抖动会直接进角速度
    训练时的 omega 是真陀螺仪信号，分布不同 → 动态动作会退化。
    ★ 正解还是让下位机把陀螺仪角速度一起回传。
    """

    def __init__(self, inner: ImuSource, alpha: float = 0.4, max_dt: float = 0.1):
        self.inner = inner
        self.alpha = float(alpha)          # 一阶低通，抑制差分噪声（0=不滤波）
        self.max_dt = float(max_dt)
        self._prev_q = None
        self._prev_t = None
        self._omega = np.zeros(3)
        self._latest = ImuSample(source_tag="derived")

    def start(self, timeout: float = 8.0):
        self.inner.start(timeout)

    def get_latest(self) -> ImuSample:
        s = self.inner.get_latest()
        q = s.quat_wxyz
        now = time.monotonic()
        if q is not None:
            if self._prev_q is not None:
                dt = now - self._prev_t
                if 1e-4 < dt < self.max_dt:
                    dq = _quat_mul(_quat_conj(self._prev_q), q)
                    w = 2.0 * dq[1:4] / dt
                    # 小角度假设：w ≈ 2*vec(dq)/dt
                    self._omega = (1 - self.alpha) * self._omega + self.alpha * w
            self._prev_q = q.copy()
            self._prev_t = now
        self._latest = ImuSample(gyro=self._omega.copy(), quat_wxyz=q,
                                 accel=s.accel, source_tag="derived", recv_time=now)
        return self._latest

    def stop(self):
        self.inner.stop()


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


# ============================================================================
class StubImuSource(ImuSource):
    """干跑/自检：完全水平、静止。"""

    def __init__(self, quat_wxyz=None, gyro=None):
        self._q = (np.array([1.0, 0.0, 0.0, 0.0]) if quat_wxyz is None
                   else np.asarray(quat_wxyz, float))
        self._w = np.zeros(3) if gyro is None else np.asarray(gyro, float)
        self._latest = ImuSample()

    def start(self, timeout: float = 8.0):
        self._latest = ImuSample(gyro=self._w.copy(), quat_wxyz=self._q.copy(),
                                 source_tag="stub")

    def get_latest(self) -> ImuSample:
        self._latest.recv_time = time.monotonic()
        return self._latest


def make_imu_source(cfg: dict) -> ImuSource:
    """按 config.yaml 的 imu 段构造 IMU 源（含角速度来源的选择与告警）。"""
    cfg = cfg or {}
    src = cfg.get("source", "usb_euler")
    if src == "stub":
        return StubImuSource()

    if src == "usb_euler":
        deg = bool(cfg.get("euler_unit_is_degree", True))
        scale = np.array([np.pi / 180.0] * 3) if deg else np.ones(3)
        if cfg.get("euler_scale") is not None:
            scale = np.asarray(cfg["euler_scale"], dtype=np.float64)
        base = UsbEulerImuSource(
            frame_id=cfg.get("frame_id"),
            euler_offset=int(cfg.get("euler_offset", 0)),
            euler_dtype=cfg.get("euler_dtype", "f4"),
            euler_scale=scale,
            convention=cfg.get("euler_convention", "rpy_zyx_intrinsic"),
            gyro_offset=cfg.get("gyro_offset"),
            gyro_dtype=cfg.get("gyro_dtype", "f4"),
            gyro_scale=(None if cfg.get("gyro_scale") is None
                        else np.asarray(cfg["gyro_scale"], dtype=np.float64)),
            byteorder=cfg.get("byteorder", "<"),
            max_age_ms=float(cfg.get("max_age_ms", 200.0)),
        )
        if base.has_gyro_field:
            return base
        mode = cfg.get("gyro_mode", "derived")
        if mode == "derived":
            print("[IMU] ⚠️  下位机帧里没有角速度字段 → 用欧拉角差分反推（兜底）。")
            print("       这会让 obs[3:6] 与训练时的真陀螺仪分布不同，"
                  "【只建议低速站立/微走】，动态动作会退化。")
            print("       正解：让下位机固件把陀螺仪角速度一起回传。")
            return DerivedGyroSource(base, alpha=float(cfg.get("derived_alpha", 0.4)))
        if mode == "zero":
            print("[IMU] ⚠️⚠️  角速度恒置 0 —— 观测缺一段，策略行为不可信！只用于确认链路通。")
            return _ZeroGyroWrapper(base)
        raise ValueError("gyro_mode 只能是 derived / zero（当前 %r）" % mode)

    raise ValueError("未知 imu.source: %r" % src)


class _ZeroGyroWrapper(ImuSource):
    def __init__(self, inner):
        self.inner = inner

    def start(self, timeout: float = 8.0):
        self.inner.start(timeout)

    def get_latest(self) -> ImuSample:
        s = self.inner.get_latest()
        return ImuSample(gyro=np.zeros(3), quat_wxyz=s.quat_wxyz, accel=s.accel,
                         source_tag="zero_gyro", recv_time=s.recv_time)

    def stop(self):
        self.inner.stop()


if __name__ == "__main__":
    print("欧拉角约定:", ", ".join(sorted(EULER_CONVENTIONS)))
    # 欧拉角 -> 四元数 -> 投影重力，验证符号方向
    from tools.math_utils import get_gravity_orientation
    for roll, pitch, name in ((0.0, 0.0, "水平"), (0.3, 0.0, "左倾 roll+0.3"),
                              (0.0, 0.3, "抬头 pitch+0.3")):
        q = euler_to_quat_wxyz(roll, pitch, 0.0)
        g = get_gravity_orientation(q)
        print("  %-14s roll=%+.2f pitch=%+.2f -> pg=[%+.3f %+.3f %+.3f]"
              % (name, roll, pitch, g[0], g[1], g[2]))

    # 帧解析：int16 欧拉角（单位 0.01 度）
    scale = np.array([0.01 * np.pi / 180.0] * 3)
    payload = bytes([0x21]) + (np.array([1000, 0, 0]) ).astype("<i2").tobytes()
    s = UsbEulerImuSource(frame_id=0x21, euler_offset=1, euler_dtype="i2",
                          euler_scale=scale)
    assert s.feed_usb_frame(payload) is True
    got = s.get_latest()
    print("  int16 欧拉角帧解码 ✓  quat=", np.round(got.quat_wxyz, 4),
          " gyro字段:", s.has_gyro_field)

    # 差分角速度：绕 z 以 1 rad/s 转
    class _Spin(ImuSource):
        def __init__(self): self.t0 = time.monotonic()
        def start(self, timeout=8.0): self.t0 = time.monotonic()
        def get_latest(self):
            yaw = (time.monotonic() - self.t0) * 1.0
            return ImuSample(quat_wxyz=euler_to_quat_wxyz(0, 0, yaw))
    d = DerivedGyroSource(_Spin(), alpha=0.2)
    d.start()
    for _ in range(30):
        time.sleep(0.005)
        d.get_latest()
    print("  差分角速度 (绕 z 真值 1.0 rad/s) ->", np.round(d.get_latest().gyro, 3))
    print("✅ imu 自测通过（★ 真实 frame_id / 字段偏移 / 旋序 仍需 tools/link_sniffer.py 实测）")
