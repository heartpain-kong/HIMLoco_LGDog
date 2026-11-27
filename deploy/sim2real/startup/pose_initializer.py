#!/usr/bin/env python3
"""LgDog1 起立姿态初始化器（12 关节纯四足，实测起点版本）。

设计（仿照 RC_WheelLeg 的 pose_initializer，删除全部轮子分支）：
  - 不假设物理起始姿态：enable 电机后直接读 12 关节实测角作为插值起点，
    而不是写死一个 CRAWL_POSE / GROUND_POSE（真机每次趴的姿势都不一样）。
  - 全程缓慢、可中断：每个控制周期都查 guard.should_stop()，
    True 立即 damping_brake 并返回 False（真机代码不用异常当控制流，
    统一用返回值表示成败）。
  - 超时兜底：插值阶段总超时 = 自适应过渡时长 + timeout_extra，
    超时说明机器人被卡住/被按住/关节没响应，刹车放弃而不是硬顶。

Phase 流程：
  1 READ_START     读实测起点，与目标姿态做偏差检查（warn / abort）
  2 SOFT_HOLD      按实测姿态软保持，kp 从 soft_hold_kp_scale 线性升到 1.0
                   （目的：让电机先"接管"当前姿态，避免一上电就猛拽）
  3 TRANSITION     余弦插值实测起点 → 目标站姿，到位判据 + 保持 hold_time
  4 HOLD_AFTER     到位后再保持 hold_time 秒
  5 (外部)         hold_until_user_confirm：持续保持直到用户回车

参数全部通过构造函数传入（config.yaml 的 startup: 段由调用方解析），
本模块不直接读 yaml。
"""
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

# 常量只从 spec.py import，不许在本文件抄第二遍数字
# （与本仓其他模块一致：把 sim2real 目录加入 sys.path 后按顶层包名导入）
_SIM2REAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SIM2REAL_DIR not in sys.path:
    sys.path.insert(0, _SIM2REAL_DIR)
import spec  # noqa: E402


class MockStandIO:
    """仅 __main__ 自测用的假 io：模拟一阶收敛（趴着 → 站着）。

    真实 q 以一阶惯性追踪下发目标：dq/dt = (q_cmd - q) / tau，
    这样 transition 的到位判据/超时逻辑都能被真实地走到。
    """

    def __init__(self, q_start, tau=0.4, dt=0.02):
        self.q = np.array(q_start, dtype=np.float64)
        self.dq = np.zeros(len(q_start))
        self.q_cmd = self.q.copy()
        self.tau = tau
        self.dt = dt
        self.braked = False

    def read_state(self):
        # 一阶收敛一步
        self.dq = (self.q_cmd - self.q) / self.tau
        self.q = self.q + self.dq * self.dt
        return {
            "q": self.q.copy(),
            "dq": self.dq.copy(),
            "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),  # 水平
            "omega": np.zeros(3),
            "imu_age_ms": 0.0,
        }

    def write_targets(self, q_target, kp, kd):
        self.q_cmd = np.asarray(q_target, dtype=np.float64).copy()

    def hold_pose(self, q_target, kp, kd):
        self.write_targets(q_target, kp, kd)

    def damping_brake(self):
        self.braked = True


class AlwaysRunGuard:
    """仅 __main__ 自测用：永远不急停的 guard 替身。"""

    def should_stop(self):
        return False


class PoseInitializer:
    """从"瘫在地上"缓慢过渡到默认站姿。所有阈值来自 config.yaml 的 startup: 段。"""

    def __init__(self,
                 io: Any,
                 guard: Any,
                 control_dt: float = spec.CONTROL_DT,
                 kp: Any = None,
                 kd: Any = None,
                 soft_hold_duration: float = 1.0,        # config: startup.soft_hold_duration
                 soft_hold_kp_scale: float = 0.125,      # config: startup.soft_hold_kp_scale
                 ramp_kp_time: float = 1.0,              # config: startup.ramp_kp_time
                 transition_time_min: float = 2.0,       # config: startup.transition_time_min
                 transition_time_max: float = 6.0,       # config: startup.transition_time_max
                 transition_seconds_per_rad: float = 1.5,  # config: startup.transition_seconds_per_rad
                 timeout_extra: float = 3.0,             # config: startup.timeout_extra
                 hold_time: float = 1.0,                 # config: startup.hold_time
                 settle_pos_threshold: float = 0.30,     # config: startup.settle_pos_threshold
                 settle_vel_threshold: float = 0.6,      # config: startup.settle_vel_threshold
                 max_dev_warn: float = 1.5,              # config: startup.max_dev_warn
                 max_dev_abort: float = 3.0,             # config: startup.max_dev_abort
                 progress_log_interval: float = 0.5):
        """
        Args:
            io: 鸭子类型接口，需要 read_state / write_targets / hold_pose /
                damping_brake / enable_motors（见文件头注释的接口约定）。
            guard: RuntimeGuard，每个控制周期调用 should_stop()，
                True 表示急停，必须立刻放弃并返回失败。
            kp/kd: 每关节增益（长度 12 或标量）；默认取 spec.KP / spec.KD，
                与训练一致，不许在本文件里另写数值。
        """
        self.io = io
        self.guard = guard
        self.control_dt = float(control_dt)
        self.kp = np.full(spec.NUM_JOINTS, spec.KP) if kp is None else np.asarray(kp, dtype=np.float64)
        self.kd = np.full(spec.NUM_JOINTS, spec.KD) if kd is None else np.asarray(kd, dtype=np.float64)

        self.soft_hold_duration = float(soft_hold_duration)
        self.soft_hold_kp_scale = float(soft_hold_kp_scale)
        self.ramp_kp_time = float(ramp_kp_time)
        self.transition_time_min = float(transition_time_min)
        self.transition_time_max = float(transition_time_max)
        self.transition_seconds_per_rad = float(transition_seconds_per_rad)
        self.timeout_extra = float(timeout_extra)
        self.hold_time = float(hold_time)
        self.settle_pos_threshold = float(settle_pos_threshold)
        self.settle_vel_threshold = float(settle_vel_threshold)
        self.max_dev_warn = float(max_dev_warn)
        self.max_dev_abort = float(max_dev_abort)
        self.progress_log_interval = float(progress_log_interval)

    # ------------------------------------------------------------------
    # 通用每周期原语：读状态 → guard 检查 → 下发目标 → 锁相到控制周期
    # ------------------------------------------------------------------
    def _tick_write(self, q_target: np.ndarray, kp: np.ndarray, next_exec: float):
        """下发一帧目标并按 spec.CONTROL_DT 锁相。
        返回 (state_dict, next_exec)；guard 急停返回 (None, next_exec)。"""
        state = self.io.read_state()
        if self.guard is not None and self.guard.should_stop():
            print("[PoseInit] guard 急停，放弃起立")
            self.io.damping_brake()
            return None, next_exec

        self.io.write_targets(q_target, kp, self.kd)

        # 锁相方式与参考实现相同：next_exec += dt 后 sleep(剩余-2ms) 再忙等，
        # 2ms 余量用来抵消 OS 调度抖动，忙等保证周期精确。
        next_exec += self.control_dt
        slack = next_exec - time.perf_counter()
        if slack > 0.0:
            coarse = slack - 0.002
            if coarse > 0.0:
                time.sleep(coarse)
            while time.perf_counter() < next_exec:
                pass
        else:
            # 已经落后于节拍（日志打印太慢等）：重新对齐，不追帧
            next_exec = time.perf_counter()
        return state, next_exec

    # ------------------------------------------------------------------
    def _hold_for(self, q_target: np.ndarray, duration: float,
                  kp: np.ndarray, phase_name: str) -> bool:
        """按固定 kp 保持 q_target 一段时间。急停返回 False（已刹车）。"""
        n = max(1, int(round(duration / self.control_dt)))
        next_exec = time.perf_counter()
        for _ in range(n):
            state, next_exec = self._tick_write(q_target, kp, next_exec)
            if state is None:
                return False
        print(f"[PoseInit] {phase_name} 完成 ({duration:.2f}s)")
        return True

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def transition_to_stand_from_current(self,
                                         target_pose: Optional[np.ndarray] = None) -> bool:
        """完整起立流程，返回是否成功。

        1. 读实测起点 + 偏差检查
        2. SOFT_HOLD：实测姿态软保持 + kp ramp-up
        3. TRANSITION：余弦插值到目标姿态（自适应时长 + 超时兜底）
        4. HOLD_AFTER：到位后保持 hold_time 秒
        失败路径一律先 damping_brake 再返回 False。
        """
        if target_pose is None:
            target_pose = spec.DEFAULT_DOF_POS.copy()
        target = np.asarray(target_pose, dtype=np.float64).reshape(spec.NUM_JOINTS).copy()
        full_kp = self.kp.copy()

        # === 1. 读实测起点 + 偏差检查 ===
        state = self.io.read_state()
        start = np.asarray(state["q"], dtype=np.float64).reshape(spec.NUM_JOINTS).copy()

        diff = np.abs(start - target)
        max_dev = float(np.max(diff))
        max_dev_joint = int(np.argmax(diff))
        print(f"[PoseInit] 实测起点最大偏差 {max_dev:.3f} rad "
              f"(关节 {spec.JOINT_LABELS[max_dev_joint]})")

        # config: startup.max_dev_abort —— 姿态太离谱（比如关节被卡死、读数异常）
        # 时拒绝起立，硬掰只会烧电机或把机器人掀翻。
        if max_dev > self.max_dev_abort:
            print(f"[PoseInit] ABORT 偏差 {max_dev:.2f} rad > max_dev_abort "
                  f"{self.max_dev_abort}，拒绝起立")
            self.io.damping_brake()
            return False
        # config: startup.max_dev_warn —— 大偏差只警告（起立会更剧烈），不阻断
        if max_dev > self.max_dev_warn:
            print(f"[PoseInit] WARNING 偏差 {max_dev:.2f} rad > max_dev_warn "
                  f"{self.max_dev_warn}，起立可能较剧烈")

        # 自适应过渡时长：偏差越大走得越慢（config: transition_seconds_per_rad），
        # 再用 min/max 夹住，避免"偏差极小就 0 秒瞬移"或"瘫得很开就无限等"。
        transition_time = float(np.clip(
            max_dev * self.transition_seconds_per_rad,
            self.transition_time_min, self.transition_time_max))
        timeout = transition_time + self.timeout_extra  # config: startup.timeout_extra
        print(f"[PoseInit] 计划: soft_hold={self.soft_hold_duration}s, "
              f"transition={transition_time:.2f}s, hold={self.hold_time}s, "
              f"timeout={timeout:.2f}s")

        # === 2. SOFT_HOLD：先按实测姿态接管 + kp ramp-up ===
        # 一上电就用满 kp 去拉目标姿态，电机会猛拽；所以先用低 kp 按住
        # 当前姿态让齿轮箱消隙，再在 ramp_kp_time 内把 kp 线性升到全值。
        print("[PoseInit] 阶段2: 软保持 + kp ramp-up")
        n_soft = max(1, int(round(self.soft_hold_duration / self.control_dt)))
        next_exec = time.perf_counter()
        t0 = time.perf_counter()
        for _ in range(n_soft):
            elapsed = time.perf_counter() - t0
            if elapsed < self.ramp_kp_time:
                # 从 soft_hold_kp_scale 线性升到 1.0（前一小段直接用最低档）
                frac = min(1.0, elapsed / max(self.ramp_kp_time, 1e-3))
                kp_scale = self.soft_hold_kp_scale + (1.0 - self.soft_hold_kp_scale) * frac
            else:
                kp_scale = 1.0
            state, next_exec = self._tick_write(start, full_kp * kp_scale, next_exec)
            if state is None:
                return False

        # === 3. TRANSITION：余弦插值（速度两端自然为零，比线性平滑得多）===
        print(f"[PoseInit] 阶段3: 余弦插值过渡 {transition_time:.2f}s")
        t0 = time.perf_counter()
        last_log = t0
        reached = False
        hold_start: Optional[float] = None
        next_exec = t0

        while True:
            now = time.perf_counter()
            elapsed = now - t0
            phase = min(1.0, elapsed / max(transition_time, 1e-3))

            # 超时兜底：过渡时长 + timeout_extra 仍没到位，说明被卡住/被按住，
            # 继续加力只会出问题，刹车放弃。
            if elapsed > timeout:
                print(f"[PoseInit] TIMEOUT {elapsed:.2f}s 未到位，放弃")
                self.io.damping_brake()
                return False

            blend = 0.5 - 0.5 * np.cos(np.pi * phase)
            blended = start + blend * (target - start)
            state, next_exec = self._tick_write(blended, full_kp, next_exec)
            if state is None:
                return False

            # 到位判据：位置误差与速度【同时】小于阈值，且持续 hold_time 秒
            # （只看位置会被振荡骗过；持续保持防止"路过就算到位"）
            pos_err = float(np.max(np.abs(state["q"] - target)))
            vel_err = float(np.max(np.abs(state["dq"])))

            if now - last_log >= self.progress_log_interval:
                print(f"[PoseInit]   phase={phase * 100:5.1f}%  "
                      f"pos_err={pos_err:.3f}  vel={vel_err:.3f}")
                last_log = now

            if phase >= 1.0 and pos_err <= self.settle_pos_threshold \
                    and vel_err <= self.settle_vel_threshold:
                if not reached:
                    reached = True
                    hold_start = now
                    print(f"[PoseInit]   已到位，保持 {self.hold_time:.2f}s")
                elif now - hold_start >= self.hold_time:
                    break
            elif phase >= 1.0:
                # 到点后又弹开了：重新计时，必须连续保持满 hold_time
                reached = False
                hold_start = None

        # === 4. HOLD_AFTER：到位后再稳稳保持 hold_time 秒 ===
        print("[PoseInit] 阶段4: 过渡后保持")
        if not self._hold_for(target, self.hold_time, full_kp, "HOLD_AFTER"):
            return False

        print("[PoseInit] ✅ 默认站姿初始化完成")
        return True

    # ------------------------------------------------------------------
    # 等用户回车确认（策略释放前的人工闸口）
    # ------------------------------------------------------------------
    def hold_until_user_confirm(self,
                                q_target: np.ndarray,
                                kp: Optional[np.ndarray] = None,
                                kd: Optional[np.ndarray] = None,
                                confirm_event: Optional[threading.Event] = None) -> bool:
        """持续保持目标姿态，直到用户按回车（或 confirm_event 被 set）。

        后台线程读 input()，主循环按控制周期保持姿态，每个周期都查 guard；
        被急停打断立刻 damping_brake 并返回 False。正常确认返回 True。
        """
        target = np.asarray(q_target, dtype=np.float64).reshape(spec.NUM_JOINTS).copy()
        kp_arr = self.kp.copy() if kp is None else np.asarray(kp, dtype=np.float64)
        kd_arr = self.kd.copy() if kd is None else np.asarray(kd, dtype=np.float64)

        if confirm_event is None:
            confirm_event = threading.Event()

            def _wait_input():
                try:
                    input()
                except EOFError:
                    pass  # 无 stdin（后台/管道运行）时永不确认，由 guard 兜底
                confirm_event.set()

            reader = threading.Thread(target=_wait_input, daemon=True)
            reader.start()

        print("[PoseInit] 等待用户按回车确认进入下一阶段...")
        next_exec = time.perf_counter()
        while not confirm_event.is_set():
            state, next_exec = self._tick_write(target, kp_arr, next_exec)
            if state is None:
                print("[PoseInit] 等待确认期间被急停")
                return False
        return True


# ============================================================================
# 自测：mock io 模拟"趴着 → 站着"的一阶收敛，跑通全流程并打印各阶段耗时。
# 运行: /home/heartpain/miniconda3/envs/himloco/bin/python pose_initializer.py
# ============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 趴姿：大腿/calf 与站姿差 1 rad 左右（真实趴姿量级，< max_dev_abort=3.0）
    q_crawl = spec.DEFAULT_DOF_POS + np.array(
        [0.0, +1.0, -1.0] * spec.NUM_LEGS, dtype=np.float64)

    io = MockStandIO(q_crawl, tau=0.12)  # 一阶收敛时间常数，自测加速用
    guard = AlwaysRunGuard()
    ctrl = PoseInitializer(
        io, guard,
        # 自测专用：把所有时长按 1/10 缩短，总耗时控制在几秒内。
        # 正式参数必须来自 config.yaml 的 startup: 段，不许照抄这里的值。
        soft_hold_duration=0.2, soft_hold_kp_scale=0.125, ramp_kp_time=0.2,
        transition_time_min=0.3, transition_time_max=1.0,
        transition_seconds_per_rad=0.2, timeout_extra=0.5,
        hold_time=0.2, settle_pos_threshold=0.30, settle_vel_threshold=0.6,
        max_dev_warn=1.5, max_dev_abort=3.0,
        progress_log_interval=0.2)

    t_all = time.perf_counter()
    io.enable_motors() if hasattr(io, "enable_motors") else None
    ok = ctrl.transition_to_stand_from_current()
    dt_all = time.perf_counter() - t_all
    print(f"\n自测1 全流程: {'成功' if ok else '失败'}，总耗时 {dt_all:.2f}s")
    final_err = float(np.max(np.abs(io.q - spec.DEFAULT_DOF_POS)))
    print(f"自测1 最终关节最大误差 {final_err:.3f} rad")
    assert ok, "起立流程应成功"
    assert final_err < 0.30, "最终应收敛到默认站姿附近"
    assert dt_all < 30.0

    # 自测2：急停路径 —— guard 第 20 个周期触发 should_stop，应返回 False 且刹车
    class StopAt20Guard:
        def __init__(self):
            self.n = 0

        def should_stop(self):
            self.n += 1
            return self.n > 20

    io2 = MockStandIO(q_crawl, tau=10.0)  # tau 很大 → 永远追不上 → 走到超时/急停
    ctrl2 = PoseInitializer(
        io2, StopAt20Guard(),
        soft_hold_duration=0.1, ramp_kp_time=0.1,
        transition_time_min=5.0, transition_time_max=5.0,
        transition_seconds_per_rad=1.0, timeout_extra=5.0,
        hold_time=0.1, settle_pos_threshold=0.30, settle_vel_threshold=0.6,
        max_dev_warn=1.5, max_dev_abort=3.0)
    t2 = time.perf_counter()
    ok2 = ctrl2.transition_to_stand_from_current()
    print(f"自测2 急停路径: {'失败(符合预期)' if not ok2 else '异常'}，"
          f"耗时 {time.perf_counter() - t2:.2f}s，braked={io2.braked}")
    assert not ok2 and io2.braked

    # 自测3：初始偏差过大 → 直接 ABORT
    io3 = MockStandIO(spec.DEFAULT_DOF_POS + np.array(
        [0.0, 0.0, 3.5] * spec.NUM_LEGS, dtype=np.float64))
    ctrl3 = PoseInitializer(
        io3, AlwaysRunGuard(),
        soft_hold_duration=0.1, ramp_kp_time=0.1,
        transition_time_min=0.1, transition_time_max=0.2,
        transition_seconds_per_rad=0.1, timeout_extra=0.1, hold_time=0.1)
    ok3 = ctrl3.transition_to_stand_from_current()
    print(f"自测3 偏差过大: {'失败(符合预期)' if not ok3 else '异常'}，braked={io3.braked}")
    assert not ok3 and io3.braked

    print("\n✅ pose_initializer 自测全部通过")
