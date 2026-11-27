#!/usr/bin/env python3
"""LgDog1 电机层：12 个灵足 RS-02，经达妙 DM-MC02（USB→CAN）批量读写

复用 RC_WheelLeg 的 vendored 驱动（`vendored/drivers/`）：
    DmUsbAdapter      —— DM-MC02 的 USB 串口帧协议
    RobStrideDriver   —— RS-02 的 CAN 帧 + MIT 模式编解码

本文件只做三件事：
  1) 按 spec.JOINT_LABELS 建 12 个电机，把映射表（can_id / sign / gear_ratio /
     zero_offset）接进来，对外一律用【策略关节空间】(rad) 说话，不暴露电机空间。
  2) 反馈新鲜度管理：RobStride 在 MIT 模式下"有指令才有反馈"，所以每周期
     发完指令后 process_messages 收回复；没收到就沿用上一帧并计数（holdover）。
  3) 阻尼刹车：kp=0 只留 kd —— 这是所有异常路径的兜底动作。
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "vendored"))    # 让 `drivers.*` 可导入

import spec  # noqa: E402
from interface.motor_mapping import MotorMapping  # noqa: E402


class MotorBus:
    """12 关节电机总线。对外接口全部是【策略关节空间】。"""

    def __init__(self, mapping: MotorMapping, port: str, debug: bool = False,
                 stale_frames_to_holdover: int = 2):
        from drivers.motor_driver import RobStrideDriver

        self.mapping = mapping
        self.port = port
        self.debug = debug
        self.stale_frames_to_holdover = stale_frames_to_holdover

        self.driver = RobStrideDriver(port, debug=debug)
        self._names = list(spec.JOINT_LABELS)
        for i, name in enumerate(self._names):
            self.driver.add_motor(name, int(mapping.can_id[i]), mapping.motor_model)

        self._connected = False
        self._enabled = False
        self._last_update = np.zeros(spec.NUM_JOINTS, dtype=np.int64)
        self._stale = np.zeros(spec.NUM_JOINTS, dtype=np.int64)
        self._q = np.zeros(spec.NUM_JOINTS, dtype=np.float64)     # 关节空间
        self._dq = np.zeros(spec.NUM_JOINTS, dtype=np.float64)
        self._tau = np.zeros(spec.NUM_JOINTS, dtype=np.float64)
        self._temp = np.zeros(spec.NUM_JOINTS, dtype=np.float64)
        self.holdover_count = 0

    # ------------------------------------------------------------------ 连接
    def connect(self):
        if self._connected:
            return
        self.driver.connect()
        self._connected = True
        # 记录初值，之后靠 update_count 变化判断反馈是否新鲜
        self._snapshot_update_counts(initial=True)

    def disconnect(self):
        if not self._connected:
            return
        try:
            if self._enabled:
                self.damping_brake()
                time.sleep(0.2)
                self.disable_all()
        finally:
            try:
                self.driver.disconnect()
            finally:
                self._connected = False
                self._enabled = False

    # ------------------------------------------------------------------ 使能
    def clear_warnings_all(self):
        for n in self._names:
            try:
                self.driver.clear_warnings(n)
            except Exception:
                pass

    def enable_all(self):
        for n in self._names:
            self.driver.enable(n)
        self._enabled = True

    def disable_all(self):
        for n in self._names:
            try:
                self.driver.disable(n)
            except Exception:
                pass
        self._enabled = False

    def set_zero_position_all(self):
        """把当前姿态记为电机零位（⚠️ 只在机器人摆成 URDF 零位时用）。"""
        for n in self._names:
            self.driver.set_zero_position(n)

    # ------------------------------------------------------------------ 反馈
    def _snapshot_update_counts(self, initial=False):
        for i, n in enumerate(self._names):
            self._last_update[i] = self.driver.motors[n].state.update_count
        if initial:
            self._stale[:] = 0

    def poll(self, max_messages: int = 60):
        """收一轮 CAN 回复并更新内部状态（发完指令后调用）。"""
        self.driver.process_messages(max_messages=max_messages)

    def read(self):
        """读取 12 关节状态（关节空间）。

        :return: (q, dq, tau, temp, fresh)  —— fresh[i]=False 表示该电机这帧没刷新，
                 其 q/dq 是沿用上一帧的（holdover）。
        """
        fresh = np.zeros(spec.NUM_JOINTS, dtype=bool)
        for i, n in enumerate(self._names):
            st = self.driver.motors[n].state
            if st.update_count != self._last_update[i]:
                self._last_update[i] = st.update_count
                self._stale[i] = 0
                # 电机空间 -> 关节空间
                self._q[i] = self.mapping.real_to_sim(st.position)[i]
                self._dq[i] = self.mapping.real_vel_to_sim(st.velocity)[i]
                # 电机侧力矩 -> 关节侧（RS-02 的 torque 是电机侧估计值）
                self._tau[i] = st.torque * self.mapping.gear_ratio[i]
                self._temp[i] = st.temperature
                fresh[i] = True
            else:
                self._stale[i] += 1
        n_stale = int(np.sum(self._stale >= self.stale_frames_to_holdover))
        if n_stale:
            self.holdover_count += 1
        return (self._q.copy(), self._dq.copy(), self._tau.copy(),
                self._temp.copy(), fresh)

    def wait_feedback_ready(self, timeout: float = 5.0, use_passive_poll: bool = True):
        """等 12 个电机都有反馈。

        MIT 模式必须有指令才有回复，所以用【全零 MIT 帧】被动索要（passive poll）。
        """
        t0 = time.time()
        seen = np.zeros(spec.NUM_JOINTS, dtype=bool)
        while time.time() - t0 < timeout:
            self._snapshot_update_counts()
            if use_passive_poll:
                for n in self._names:
                    # kp=kd=0 → 不产生力矩，纯索要反馈
                    self.driver.control_mit(n, 0.0, 0.0, 0.0, 0.0, 0.0)
            self.poll()
            for i, n in enumerate(self._names):
                if self.driver.motors[n].state.update_count != self._last_update[i]:
                    seen[i] = True
            if seen.all():
                return True
            time.sleep(0.02)
        missing = [spec.JOINT_LABELS[i] for i in range(spec.NUM_JOINTS) if not seen[i]]
        raise TimeoutError(
            "等 %.1fs 仍有 %d 个电机没有反馈: %s（查 CAN 接线/电机 ID/供电）"
            % (timeout, len(missing), ", ".join(missing)))

    # ------------------------------------------------------------------ 指令
    def write_targets(self, q_target_sim, kp, kd, tau_ff=None):
        """下发位置目标（关节空间）。

        :param q_target_sim: 12 关节目标角 [rad]（策略顺序）
        :param kp: 12 个位置增益（关节侧单位）
        :param kd: 12 个阻尼增益
        :param tau_ff: 可选前馈力矩（关节侧），会换算到电机侧
        """
        q_target_sim = np.asarray(q_target_sim, dtype=np.float64)
        assert q_target_sim.shape == (spec.NUM_JOINTS,)
        kp = np.broadcast_to(np.asarray(kp, dtype=np.float64), (spec.NUM_JOINTS,))
        kd = np.broadcast_to(np.asarray(kd, dtype=np.float64), (spec.NUM_JOINTS,))
        tau_sim = (np.zeros(spec.NUM_JOINTS) if tau_ff is None
                   else np.asarray(tau_ff, dtype=np.float64))

        # 关节目标 -> 电机目标
        q_motor = self.mapping.sim_to_real(q_target_sim)
        tau_motor = self.mapping.sim_tau_to_real(tau_sim)
        for i, n in enumerate(self._names):
            self.driver.control_mit(n, float(q_motor[i]), 0.0,
                                    float(kp[i]), float(kd[i]), float(tau_motor[i]))

    def damping_brake(self, kd=None):
        """阻尼刹车：kp=0、只留 kd，卸掉刚度保留阻尼。所有异常路径的兜底。"""
        if kd is None:
            kd = np.full(spec.NUM_JOINTS, spec.CONTROL_FREQ and _DEFAULT_BRAKE_KD)
        kd = np.broadcast_to(np.asarray(kd, dtype=np.float64), (spec.NUM_JOINTS,))
        for i, n in enumerate(self._names):
            self.driver.control_mit(n, 0.0, 0.0, 0.0, float(kd[i]), 0.0)

    def hold_current_target(self, q_target_sim, kp, kd, seconds: float, dt: float,
                            guard=None):
        """按给定周期重复下发同一目标（用于软保持/过渡阶段）。"""
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.write_targets(q_target_sim, kp, kd)
            self.poll()
            if guard is not None and guard.should_stop():
                return False
            time.sleep(dt)
        return True


_DEFAULT_BRAKE_KD = 2.5     # 阻尼刹车的 kd（RS-02 MIT 范围 0~5 内的温和值）


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="电机总线自检（不发大力，仅扫 ID 与读反馈）")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--map", default=None)
    ap.add_argument("--scan", action="store_true", help="只扫总线上的电机 ID，不使能")
    args = ap.parse_args()

    m = MotorMapping(args.map) if args.map else MotorMapping()
    print(m.describe())
    print()

    if args.scan:
        from drivers.motor_driver import RobStrideDriver
        d = RobStrideDriver(args.port)
        d.connect()
        try:
            ids = d.scan_motors(timeout=0.5)
            print("总线上扫到 %d 个电机 ID: %s" % (len(ids), sorted(ids)))
            expected = sorted(int(x) for x in m.can_id)
            print("映射表期望的 ID   : %s" % expected)
            missing = [i for i in expected if i not in ids]
            if missing:
                print("⚠️  这些 ID 没扫到: %s（查接线/供电/ID 拨码）" % missing)
            else:
                print("✅ 期望的 12 个 ID 全部在线")
        finally:
            d.disconnect()
        sys.exit(0)

    bus = MotorBus(m, args.port)
    bus.connect()
    try:
        print("等待 12 个电机反馈 ...")
        bus.wait_feedback_ready(timeout=5.0)
        q, dq, tau, temp, fresh = bus.read()
        print("关节角 [rad]:", np.round(q, 4))
        print("温度   [℃ ]:", np.round(temp, 1))
        print("反馈新鲜:", fresh.all())
    finally:
        bus.disconnect()
    print("✅ 电机总线自检结束（未使能、未发力矩）")
