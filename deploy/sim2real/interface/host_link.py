#!/usr/bin/env python3
"""上位机 ↔ 下位机 通信协议（Python 端）

★ 协议的唯一真源是固件侧的 `dm02/User_File/task/lgdog_link.h`。
  本文件是它的 Python 镜像，两边必须逐字节一致 —— 由
  `tools/test_link_protocol.py` 交叉验证：它用 gcc 单独编译 C 端编解码，
  与这里的实现比对字节流。改协议要同时改两处并让那个测试通过。

为什么需要这条链路：
  下位机固件原本既不上报（CDC_Transmit_HS 零调用）、也不收命令
  （CDC_Receive_HS 收到就丢）。而策略需要：
    上行 IMU（欧拉角 + ★陀螺仪角速度）+ 12 个电机状态
    下行 PC 算好的 MIT 指令（pos/kp/kd/tau）
  其中【陀螺仪角速度】是观测 obs[3:6] 的必需项，欧拉角给不出来。

帧格式：
    [SOF 1B][msg_id 1B][len 1B][payload len B][CRC8 1B]
    SOF: 0xA5 上位机→下位机 ; 0x5A 下位机→上位机
    CRC8: poly 0x07, init 0x00, 覆盖 msg_id, len, payload
    小端；float 为 IEEE-754 binary32
"""

import os
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import spec  # noqa: E402

# ======================== 协议常量（与 lgdog_link.h 逐项对应）========================
SOF_HOST2BOARD = 0xA5
SOF_BOARD2HOST = 0x5A
MAX_PAYLOAD = 200
CRC8_POLY = 0x07
CRC8_INIT = 0x00

NUM_JOINTS = 12

HEARTBEAT_TIMEOUT_MS = 100        # 下位机侧的超时，仅用于文档说明

MODE_BOARD_OWN = 0
MODE_PC_CONTROL = 1

MSG_HEARTBEAT = 0x01
MSG_SET_MODE = 0x02
MSG_SET_ENABLE = 0x03
MSG_MIT_CMD = 0x04
MSG_DAMPING_BRAKE = 0x05
MSG_SET_ZERO = 0x06

MSG_IMU = 0x81
MSG_MOTOR_STATE = 0x82
MSG_STATUS = 0x83
MSG_ACK = 0x84
MSG_MOTOR_STATE_HI = 0x85

MOTOR_CHUNK = 6

ACK_OK, ACK_BAD_MODE, ACK_REJECTED = 0, 1, 2

# 载荷长度（用于自检：与 C 端 sizeof 对齐）
LEN_MIT_CMD = 12 * 4 * 4          # 192
LEN_BRAKE = 12 * 4                # 48
LEN_IMU = 13 * 4 + 4 + 1          # 57
LEN_MOTOR_CHUNK = 4 + 6 * (4 * 4 + 1)   # 106
LEN_STATUS = 4 + 2 + 4 * 4 + 4    # 26（2026-09 加 motor_missing）


# ======================== CRC8 ========================
def crc8(data: bytes) -> int:
    """poly 0x07 / init 0x00，多项式除法逐位实现 —— 与 LgdLink_Crc8 逐位一致。"""
    crc = CRC8_INIT
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


# ======================== 打包 / 解包 ========================
def encode(sof: int, msg_id: int, payload: bytes = b"") -> bytes:
    """打包成完整帧。与 LgdLink_Encode 行为一致（含参数合法性检查）。"""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("payload %d 字节超过上限 %d" % (len(payload), MAX_PAYLOAD))
    body = struct.pack("<BB", msg_id, len(payload)) + payload
    return struct.pack("<B", sof) + body + struct.pack("<B", crc8(body))


@dataclass
class LinkMessage:
    msg_id: int
    payload: bytes


class LinkParser:
    """流式解帧，与 LgdLink_Parse 同算法（半包 / 粘包 / 假 SOF / CRC 错都处理）。"""

    def __init__(self, sof: int):
        self.sof = sof
        self.buf = bytearray()
        self.crc_errors = 0
        self.len_errors = 0
        self.sof_drops = 0
        self.total_frames = 0

    def parse(self, data: bytes) -> List[LinkMessage]:
        self.buf.extend(data)
        out: List[LinkMessage] = []
        while True:
            # 对齐 SOF
            if not self.buf:
                break
            if self.buf[0] != self.sof:
                del self.buf[0]
                self.sof_drops += 1
                continue
            if len(self.buf) < 3:
                break                      # 还不够读出 len
            length = self.buf[2]
            if length > MAX_PAYLOAD:
                del self.buf[0]
                self.len_errors += 1
                continue
            frame_len = length + 4
            if len(self.buf) < frame_len:
                break                      # 半包
            body = bytes(self.buf[1:3 + length])
            if crc8(body) != self.buf[frame_len - 1]:
                del self.buf[0]             # CRC 错 → 丢当前 SOF 重找
                self.crc_errors += 1
                continue
            out.append(LinkMessage(self.buf[1], bytes(self.buf[3:3 + length])))
            self.total_frames += 1
            del self.buf[:frame_len]
        return out


# ======================== 载荷编解码 ========================
def pack_heartbeat(counter: int) -> bytes:
    return struct.pack("<H", counter & 0xFFFF)


def pack_set_mode(mode: int) -> bytes:
    return struct.pack("<B", mode)


def pack_set_enable(enable: bool) -> bytes:
    return struct.pack("<B", 1 if enable else 0)


def pack_mit_cmd(pos, kp, kd, tau) -> bytes:
    """12 组 MIT 参数。顺序 = 策略关节序（spec.JOINT_LABELS）。

    注意 pos/tau 是【电机侧】值 —— PC 已做过 sign/gear_ratio/zero_offset 换算。
    """
    out = b""
    for arr in (pos, kp, kd, tau):
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        assert a.size == NUM_JOINTS, "MIT 指令每项必须 12 个值"
        out += a.tobytes()
    assert len(out) == LEN_MIT_CMD
    return out


def pack_damping_brake(kd) -> bytes:
    a = np.asarray(kd, dtype=np.float32).reshape(-1)
    assert a.size == NUM_JOINTS
    return a.tobytes()


@dataclass
class ImuPayload:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    tick_ms: int = 0
    flags: int = 0

    def ready(self) -> bool:
        return bool(self.flags & 0x01)


def unpack_imu(payload: bytes) -> ImuPayload:
    assert len(payload) == LEN_IMU, "IMU 载荷 %d 字节，期望 %d" % (len(payload), LEN_IMU)
    v = struct.unpack("<13fIB", payload)
    return ImuPayload(roll=v[0], pitch=v[1], yaw=v[2],
                      gyro=np.array(v[3:6]), accel=np.array(v[6:9]),
                      quat=np.array(v[9:13]), tick_ms=v[13], flags=v[14])


@dataclass
class MotorChunk:
    tick_ms: int = 0
    pos: np.ndarray = field(default_factory=lambda: np.zeros(MOTOR_CHUNK))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(MOTOR_CHUNK))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(MOTOR_CHUNK))
    temp: np.ndarray = field(default_factory=lambda: np.zeros(MOTOR_CHUNK))
    err: np.ndarray = field(default_factory=lambda: np.zeros(MOTOR_CHUNK, dtype=np.uint8))


def unpack_motor_chunk(payload: bytes) -> MotorChunk:
    assert len(payload) == LEN_MOTOR_CHUNK
    tick = struct.unpack_from("<I", payload, 0)[0]
    off, out = 4, MotorChunk(tick_ms=tick)
    for name in ("pos", "vel", "tau", "temp"):
        setattr(out, name, np.array(struct.unpack_from("<%df" % MOTOR_CHUNK, payload, off)))
        off += MOTOR_CHUNK * 4
    out.err = np.frombuffer(payload[off:off + MOTOR_CHUNK], dtype=np.uint8).copy()
    return out


@dataclass
class StatusPayload:
    mode: int = 0
    enabled: int = 0
    pc_alive: int = 0
    err_flags: int = 0
    loop_us: int = 0
    rx_frames: int = 0
    crc_errors: int = 0
    tx_drops: int = 0
    tick_ms: int = 0
    motor_missing: int = 0     # ★ 启动时缺反馈的电机掩码（bit0..5=CAN1, bit6..11=CAN2）


def unpack_status(payload: bytes) -> StatusPayload:
    assert len(payload) == LEN_STATUS
    v = struct.unpack("<4BH5I", payload)
    # 字段顺序：mode, enabled, pc_alive, err_flags, motor_missing, loop_us,
    #           rx_frames, crc_errors, tx_drops, tick_ms
    return StatusPayload(mode=v[0], enabled=v[1], pc_alive=v[2], err_flags=v[3],
                         motor_missing=v[4], loop_us=v[5], rx_frames=v[6],
                         crc_errors=v[7], tx_drops=v[8], tick_ms=v[9])


def unpack_ack(payload: bytes):
    return struct.unpack("<BB", payload)


# ============================================================================
class HostLinkBus:
    """经下位机中转到 12 个电机的总线（与 interface/motor_driver.MotorBus 同接口）。

    上位机职责（★ 速率必须与控制环同频，否则观测历史的时间尺度就错了）：
      1) 按 spec.CONTROL_FREQ（50Hz，周期 20ms）下发 MIT 指令（PC 算好的 target_q）。
         板子收到后存在 s_tgt_pos 里，由 TIM8 的 2ms 任务写进电机（500Hz 零阶保持）——
         于是"一个控制周期内目标保持不变"这件事与训练侧的语义一致
         （Isaac: 目标保持 decimation=4 × SIM_DT=5ms = 20ms）。这条是【免费对齐】的，
         代价是要求上位机真的按 50Hz 发（发太快没意义，发太慢等于拉长控制周期）。
      2) 同样 50Hz 发心跳（下位机 100ms 收不到就自动阻尼刹车）
      3) 后台线程收上行帧 → 维护最新的 IMU 与电机状态

    上行速率（固件侧定时器决定，见 host_link.h）：
        IMU 200Hz(5ms) / 电机状态 100Hz(10ms) / 状态帧 10Hz(100ms)
      比控制环【快】是好事（每周期取最新值），但"最新值有多旧"就是链路延迟，
      用 imu_frames/motor_frames 计数 + --check --live 的能量化测出来。
    """

    def __init__(self, port: str, mapping, baudrate: int = 921600,
                 verbose: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.mapping = mapping
        self.verbose = verbose

        self.ser = None
        self.parser = LinkParser(SOF_BOARD2HOST)
        self._rx_thread = None
        self._rx_stop = threading.Event()
        self._lock = threading.Lock()

        # 最新上行状态
        self.imu = ImuPayload()
        self.motors = np.zeros(spec.NUM_JOINTS * 4 + spec.NUM_JOINTS)  # 占位，见 _apply_chunk
        self._m_pos = np.zeros(spec.NUM_JOINTS)
        self._m_vel = np.zeros(spec.NUM_JOINTS)
        self._m_tau = np.zeros(spec.NUM_JOINTS)
        self._m_temp = np.zeros(spec.NUM_JOINTS)
        self._m_err = np.zeros(spec.NUM_JOINTS, dtype=np.uint8)
        self._m_tick = 0
        self._got_lo = False
        self._got_hi = False
        self.status = StatusPayload()
        self.last_imu_time = 0.0
        self.last_motor_time = 0.0

        self._hb_counter = 0
        self.tx_frames = 0
        # ★ 上行帧计数：用来在真机上量出"链路实际吞吐/是否丢帧"。
        #   固件按 200Hz 发 IMU，若这里量到只有 60Hz，说明 USB CDC 或读线程跟不上 ——
        #   那样即使控制环是 50Hz，观测里的角速度也是"早就不新鲜"的（见 --check --live）。
        self.imu_frames = 0
        self.motor_frames = 0
        self.status_frames = 0

    # ---------------------------------------------------------------- 连接
    def connect(self):
        import serial
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01)
        self._rx_stop.clear()
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()
        if self.verbose:
            print("[HostLink] 已打开 %s @ %d" % (self.port, self.baudrate))

    def disconnect(self):
        try:
            self.damping_brake()
        except Exception:
            pass
        self._rx_stop.set()
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    # ---------------------------------------------------------------- 收
    def _rx_loop(self):
        while not self._rx_stop.is_set():
            try:
                n = self.ser.in_waiting
                data = self.ser.read(n if n else 1)
            except Exception:
                time.sleep(0.01)
                continue
            if not data:
                continue
            for msg in self.parser.parse(data):
                self._on_message(msg)

    def _on_message(self, msg: LinkMessage):
        now = time.monotonic()
        if msg.msg_id == MSG_IMU:
            with self._lock:
                self.imu = unpack_imu(msg.payload)
                self.last_imu_time = now
                self.imu_frames += 1
        elif msg.msg_id in (MSG_MOTOR_STATE, MSG_MOTOR_STATE_HI):
            c = unpack_motor_chunk(msg.payload)
            base = 0 if msg.msg_id == MSG_MOTOR_STATE else MOTOR_CHUNK
            with self._lock:
                self.motor_frames += 1
                self._m_pos[base:base + MOTOR_CHUNK] = c.pos
                self._m_vel[base:base + MOTOR_CHUNK] = c.vel
                self._m_tau[base:base + MOTOR_CHUNK] = c.tau
                self._m_temp[base:base + MOTOR_CHUNK] = c.temp
                self._m_err[base:base + MOTOR_CHUNK] = c.err
                self._m_tick = c.tick_ms
                if base == 0:
                    self._got_lo = True
                else:
                    self._got_hi = True
                self.last_motor_time = now
        elif msg.msg_id == MSG_STATUS:
            with self._lock:
                self.status = unpack_status(msg.payload)
                self.status_frames += 1
        elif msg.msg_id == MSG_ACK:
            acked, result = unpack_ack(msg.payload)
            if self.verbose:
                print("[HostLink] ACK msg=0x%02X result=%d" % (acked, result))

    # ---------------------------------------------------------------- 发
    def _send(self, msg_id: int, payload: bytes = b""):
        if self.ser is None:
            return
        try:
            self.ser.write(encode(SOF_HOST2BOARD, msg_id, payload))
            self.tx_frames += 1
        except Exception as e:
            if self.verbose:
                print("[HostLink] 发送失败: %s" % e)

    def send_heartbeat(self):
        self._hb_counter = (self._hb_counter + 1) & 0xFFFF
        self._send(MSG_HEARTBEAT, pack_heartbeat(self._hb_counter))

    def set_mode(self, mode: int):
        self._send(MSG_SET_MODE, pack_set_mode(mode))

    def set_enable(self, enable: bool):
        self._send(MSG_SET_ENABLE, pack_set_enable(enable))

    def set_zero(self):
        self._send(MSG_SET_ZERO)

    # ---------------------------------------------------------------- 电机接口
    def enable_all(self):
        self.set_enable(True)

    def disable_all(self):
        self.set_enable(False)

    def wait_feedback_ready(self, timeout: float = 5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._got_lo and self._got_hi and self.imu.tick_ms:
                    return True
            self.send_heartbeat()
            time.sleep(0.02)
        raise TimeoutError(
            "%.1fs 内没收齐上行帧（IMU + 两组电机状态）。检查：下位机是否已烧入 "
            "host_link、串口号对不对、波特率是否与下位机一致。" % timeout)

    def read(self):
        """读 12 关节状态（★ 关节空间、策略顺序）。

        下位机上报的是【电机侧】角度，这里按映射表换算回关节空间。
        """
        with self._lock:
            pos = self._m_pos.copy()
            vel = self._m_vel.copy()
            tau = self._m_tau.copy()
            temp = self._m_temp.copy()
            age_ms = (time.monotonic() - self.last_motor_time) * 1000.0 if self.last_motor_time else 1e9
            imu_age_ms = (time.monotonic() - self.last_imu_time) * 1000.0 if self.last_imu_time else 1e9

        q = self.mapping.real_to_sim(pos)
        dq = self.mapping.real_vel_to_sim(vel)
        # 力矩：电机侧 -> 关节侧（乘减速比）
        tau_j = tau * self.mapping.gear_ratio
        fresh = np.full(spec.NUM_JOINTS, age_ms < 100.0, dtype=bool)
        return q, dq, tau_j, temp, fresh, imu_age_ms

    def write_targets(self, q_target_sim, kp, kd, tau_ff=None):
        """下发关节目标。换算到电机侧后打包成 MIT 指令。"""
        q_target_sim = np.asarray(q_target_sim, dtype=np.float64)
        kp = np.broadcast_to(np.asarray(kp, dtype=np.float64), (spec.NUM_JOINTS,))
        kd = np.broadcast_to(np.asarray(kd, dtype=np.float64), (spec.NUM_JOINTS,))
        tau_sim = (np.zeros(spec.NUM_JOINTS) if tau_ff is None
                   else np.asarray(tau_ff, dtype=np.float64))
        # 增益裁到 RS-02 MIT 合法范围（下位机也会裁，这里先裁避免发无效值）
        kp, kd = self.mapping.clip_kp_kd(kp, kd)

        pos_m = self.mapping.sim_to_real(q_target_sim)
        tau_m = self.mapping.sim_tau_to_real(tau_sim)
        self._send(MSG_MIT_CMD, pack_mit_cmd(pos_m, kp, kd, tau_m))

    def damping_brake(self, kd=None):
        kd = (np.full(spec.NUM_JOINTS, 2.5) if kd is None
              else np.broadcast_to(np.asarray(kd, dtype=np.float64), (spec.NUM_JOINTS,)))
        kd, _ = self.mapping.clip_kp_kd(np.zeros(spec.NUM_JOINTS), kd)
        self._send(MSG_DAMPING_BRAKE, pack_damping_brake(kd))

    def poll(self):
        """协议里上行是下位机主动推的，这里只维护心跳。"""
        self.send_heartbeat()


def _selftest():
    """编解码自测：往返 + 半包/粘包/CRC 错。"""
    print("载荷长度自检（应与 C 端 sizeof 一致）:")
    for name, got, exp in (("MIT_CMD", LEN_MIT_CMD, 192), ("BRAKE", LEN_BRAKE, 48),
                           ("IMU", LEN_IMU, 57), ("MOTOR_CHUNK", LEN_MOTOR_CHUNK, 106),
                           ("STATUS", LEN_STATUS, 26)):
        print("  %-12s %3d 字节 (期望 %3d) %s" % (name, got, exp, "✓" if got == exp else "✗"))
        assert got == exp

    # CRC 已知值
    assert crc8(b"123456789") == 0xF4, "CRC8 标准检验向量不匹配: 0x%02X" % crc8(b"123456789")
    print("  CRC8(\"123456789\") = 0xF4 ✓ (CRC-8/SMBUS 标准检验向量)")

    # 往返
    payload = pack_mit_cmd(np.arange(12), np.full(12, 30.0), np.full(12, 0.8), np.zeros(12))
    frame = encode(SOF_HOST2BOARD, MSG_MIT_CMD, payload)
    p = LinkParser(SOF_HOST2BOARD)
    msgs = p.parse(frame)
    assert len(msgs) == 1 and msgs[0].msg_id == MSG_MIT_CMD and msgs[0].payload == payload
    print("  往返 ✓  (%d 字节帧)" % len(frame))

    # 半包
    p = LinkParser(SOF_HOST2BOARD)
    assert len(p.parse(frame[:5])) == 0
    assert len(p.parse(frame[5:])) == 1
    print("  半包 ✓")

    # 粘包
    p = LinkParser(SOF_HOST2BOARD)
    assert len(p.parse(frame + frame + frame)) == 3
    print("  粘包 ✓")

    # 垃圾字节 + CRC 错
    p = LinkParser(SOF_HOST2BOARD)
    bad = bytearray(frame); bad[-1] ^= 0xFF
    assert len(p.parse(b"\x11\x22\x33" + frame + bytes(bad))) == 1
    assert p.crc_errors == 1
    print("  垃圾前缀 + CRC 错 ✓ (crc_errors=%d, sof_drops=%d)" % (p.crc_errors, p.sof_drops))

    # IMU 往返
    imu = struct.pack("<13fIB", 0.1, 0.2, 0.3, 1, 2, 3, 0, 0, 9.81, 1, 0, 0, 0, 12345, 1)
    u = unpack_imu(imu)
    assert abs(u.roll - 0.1) < 1e-6 and np.allclose(u.gyro, [1, 2, 3]) and u.tick_ms == 12345
    print("  IMU 载荷 ✓")

    # 电机块往返
    mc = struct.pack("<I", 999) + struct.pack("<6f", *range(6)) + struct.pack("<6f", *range(6)) \
         + struct.pack("<6f", *range(6)) + struct.pack("<6f", *range(6)) + bytes(range(6))
    c = unpack_motor_chunk(mc)
    assert c.tick_ms == 999 and np.allclose(c.pos, range(6)) and c.err[5] == 5
    print("  电机块载荷 ✓")

    print("✅ host_link 编解码自测通过")


if __name__ == "__main__":
    _selftest()
