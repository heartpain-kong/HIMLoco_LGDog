#!/usr/bin/env python3
"""SafetyMonitor —— 策略输出侧的三级安全监控（NORMAL / CLIP / BRAKE / ESTOP）。

与 RuntimeGuard 的分工：
    guard 看【测量到的世界】（IMU/关节反馈）——坏了说明感知或本体出错；
    monitor 看【策略想干什么】（q_target）——异常说明策略 OOD 或观测对不上。
两者都要过，主循环顺序：guard.check(state) → monitor.filter_targets(q_target)
→ 若任一 STOP/BRAKE → io.damping_brake()。

等级语义与处置动作：
    NORMAL  继续跑策略
    CLIP    策略目标仍被下发，但被限幅到安全框内（先观察）
    BRAKE   停止跟随策略目标，上层应改调 io.damping_brake()（kp=0 只留 kd）
    ESTOP   用户急停，上层应断电/抱起

阈值全部由 main.py 从 spec.py / config.yaml 读好以形参传入，本模块不读 yaml。
"""
import os
import sys
from enum import Enum

import numpy as np

# 与 runtime_guard.py 相同的路径处理：可包导入，也可单独 `python xxx.py` 自测
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import spec                                    # noqa: E402  常量唯一来源


class SafetyLevel(Enum):
    NORMAL = 0
    CLIP = 1     # 单帧目标偏移超限，已截断
    BRAKE = 2    # 倾倒 / 角速度爆表 / 连续截断 —— 松刚度只留阻尼
    ESTOP = 3    # 用户急停

    def __str__(self):
        return self.name


class SafetyMonitor:
    """按 50Hz 控制频率调用，持有内部状态（连续 CLIP 计数），所以【每个
    机器人实例一个】，reset() 只在重新进入策略模式时调用。

    :param default_dof_pos: 默认站姿（spec.DEFAULT_DOF_POS，12 维，Isaac 顺序）。
                            目标偏移以它为参考而不是以绝对零位：策略输出本来就是
                            DEFAULT + action*ACTION_SCALE 的形式，看"离站姿多远"
                            才是物理上有意义的量（calf 限位 ±0.7 rad，偏移框必须
                            小于限位到站姿的剩余行程才有保护意义）。
    :param max_target_offset: 单关节目标相对站姿的偏移上限 [rad]
                              （config.safety.max_target_offset=0.6）。截断而不是
                              停机：单帧尖峰可能是观测毛刺，直接刹车代价大；但
                              限幅保证即使策略发疯，PD 力矩也被限制在
                              kp*0.6 ≈ 18 Nm 量级，不致甩断腿。
    :param clip_to_brake:   连续 CLIP 多少帧升级 BRAKE（config=3）。策略【持续】
                            给超框目标说明它确实 OOD（观测错位/域差），限幅后的
                            目标也不再可信，必须退出策略环路。
    :param max_ang_vel / max_tilt_z: 同 RuntimeGuard，与 config.safety 一致。
    :param tau_ratio_warn:  check_torque 中力矩被裁剪的比例阈值（0~1）。
    """

    def __init__(self,
                 default_dof_pos=None,
                 max_target_offset: float = 0.6,
                 clip_to_brake: int = 3,
                 max_ang_vel: float = 10.0,
                 max_tilt_z: float = -0.3,
                 tau_ratio_warn: float = 0.95,
                 clamp_joint_limits: bool = True,
                 joint_limit_margin_rad: float = 0.0):
        self.default_dof_pos = (np.asarray(default_dof_pos, dtype=np.float64)
                                if default_dof_pos is not None
                                else spec.DEFAULT_DOF_POS.copy())
        assert self.default_dof_pos.shape == (spec.NUM_JOINTS,), \
            "default_dof_pos 必须是 %d 维" % spec.NUM_JOINTS
        self.max_target_offset = float(max_target_offset)
        self.clip_to_brake = int(clip_to_brake)
        self.max_ang_vel = float(max_ang_vel)
        self.max_tilt_z = float(max_tilt_z)
        # 力矩裁剪判据：PD 算出的原始力矩落在 [±TAU_LIMIT*warn] 区间外说明
        # 策略想要极限力矩（被电机固件再裁一次后实际输出与策略预期不符）
        self.tau_ratio_warn = float(tau_ratio_warn)
        # ★★ 关节限位夹取（2026-09-27 起这是【唯一】的行程保护）★★
        # 背景：下位机固件原来对每个电机有自己的 Pos_min/Pos_max 软限位，桥接模式下
        # 会把上位机目标夹住 —— 那道板侧护栏【已被删除】（它只会静默夹住目标、并在
        # 越界时静默把前馈力矩置 0，制造"PC 发的角度和电机到的角度对不上"的假象）。
        # 于是行程保护全部落到上位机，也就是这里。
        # 为什么必须有这一层、而且是【无条件】的：
        #   目标偏移框是"相对默认站姿 ±max_target_offset"，它本身就可能越过关节限位 ——
        #   calf 默认 +0.5、框到 +1.1，而 URDF 限位只到 +0.7（超 0.4 rad）。
        #   在仿真里这一越界由 PhysX 的关节限位兜住；真机现在没人兜了。
        # 为什么不升级为 BRAKE：策略顶着限位推在训练里是正常现象（PhysX 也是顶住而已），
        #   静默夹到限位正是仿真的等价行为。但会计数，供日志观察是不是长期贴着限位跑。
        self.clamp_joint_limits = bool(clamp_joint_limits)
        self.joint_limit_margin_rad = float(joint_limit_margin_rad)
        self.total_joint_clamps = 0
        self.last_joint_clamps = []

        # 对外状态
        self.level = SafetyLevel.NORMAL
        self.reason = ""
        # 内部状态
        self.consecutive_clips = 0
        self._estop = False
        self._last_filtered = self.default_dof_pos.copy()
        # 运行统计（进日志用）
        self.total_clips = 0
        self.total_brakes = 0

    # ------------------------------------------------------------ 用户急停
    def request_estop(self, reason: str = "") -> None:
        """急停 latch：一旦触发，即使后续目标完全正常也保持 ESTOP，
        必须显式 reset()/clear_estop() 人工确认后才能恢复。"""
        self._estop = True
        self._set(SafetyLevel.ESTOP, reason or "user E-stop")

    def clear_estop(self) -> None:
        self._estop = False

    # ------------------------------------------------------------ 主入口
    def filter_targets(self, q_target, state=None) -> np.ndarray:
        """每控制周期调用：检查 q_target 并返回（可能的）截断版本。

        q_target: 12 维 Isaac 顺序关节目标 [rad]
        state:    可选，io.read_state() 的 dict（用 quat_wxyz/omega 做 BRAKE 判定；
                  传 None 则只做目标限幅，倾倒/角速度交给 RuntimeGuard）

        ★ 无条件返回 12 维数组：即使 BRAKE/ESTOP 也返回【站姿】而不是透传
        策略目标——这样主循环哪怕忘了检查 level、继续把返回值写给电机，
        写出去的也是安全的默认角（fail-safe 设计：忘记检查的后果被兜住）。
        """
        q_target = np.asarray(q_target, dtype=np.float64).flatten()
        if q_target.size != spec.NUM_JOINTS:
            # 维度不对说明上游代码写错了，这是 bug 不是瞬态故障，
            # 不能默默截断糊弄过去——抱异常让开发者当场发现
            raise ValueError("q_target 应为 %d 维，实际 %d" % (spec.NUM_JOINTS, q_target.size))
        # 目标本身含 NaN 时同样返回站姿：NaN 目标经 PD 会产生 NaN 力矩
        if not np.all(np.isfinite(q_target)):
            self._set(SafetyLevel.BRAKE, "q_target 含 NaN/Inf")
            self.total_brakes += 1
            self._last_filtered = self.default_dof_pos.copy()
            return self._clamp_limits(self._last_filtered)

        # ---- 1) 急停最高优先级（latch）
        if self._estop:
            self._set(SafetyLevel.ESTOP, self.reason or "user E-stop")
            self._last_filtered = self.default_dof_pos.copy()
            return self._clamp_limits(self._last_filtered)

        # ---- 2) 倾倒 / 角速度爆表 → BRAKE（机身都不正了，跟随目标没有意义）
        if state is not None:
            quat = state.get("quat_wxyz")
            if quat is not None:
                quat = np.asarray(quat, dtype=np.float64)
                n = float(np.linalg.norm(quat))
                if n > 1e-6 and np.all(np.isfinite(quat)):
                    from tools.math_utils import get_gravity_orientation
                    # 水平时 pg.z=-1，倾角越大 pg.z 越向 0 靠拢 → 倾倒是 pg.z 上穿阈值
                    gz = float(get_gravity_orientation(quat / n)[2])
                    if gz > self.max_tilt_z:
                        self._set(SafetyLevel.BRAKE, "tilt: pg.z=%.3f > %.3f" % (gz, self.max_tilt_z))
                        self.total_brakes += 1
                        self._last_filtered = self.default_dof_pos.copy()
                        return self._clamp_limits(self._last_filtered)
            omega = state.get("omega")
            if omega is not None:
                w = np.asarray(omega, dtype=np.float64)
                if np.all(np.isfinite(w)):
                    w_norm = float(np.linalg.norm(w))
                    if w_norm > self.max_ang_vel:
                        self._set(SafetyLevel.BRAKE, "ang_vel overflow: |w|=%.2f" % w_norm)
                        self.total_brakes += 1
                        self._last_filtered = self.default_dof_pos.copy()
                        return self._clamp_limits(self._last_filtered)

        # ---- 3) 目标偏移限幅：策略动作经 ACTION_SCALE 后仍可能超出安全框
        # （观测含毛刺时 action 可瞬间饱和）。逐关节截断到 [default-off, default+off]。
        offset = q_target - self.default_dof_pos
        limit = self.max_target_offset
        over = np.abs(offset) > limit
        if np.any(over):
            self.consecutive_clips += 1
            self.total_clips += 1
            filtered = self.default_dof_pos + np.clip(offset, -limit, limit)
            bad = np.where(over)[0]
            detail = ", ".join("%s:%+.3f" % (spec.JOINT_LABELS[i], offset[i]) for i in bad[:4])
            # ---- 4) 连续 CLIP 达标 → 升级 BRAKE：策略持续异常，退出环路。
            # 用【连续】计数而不是累计：偶发单帧尖峰（CAN 毛刺）不升级，
            # 持续 3 帧（60ms）以上才认定是策略本身的问题。
            if self.consecutive_clips >= self.clip_to_brake:
                self._set(SafetyLevel.BRAKE,
                          "连续 %d 帧目标超限: %s" % (self.consecutive_clips, detail))
                self.total_brakes += 1
            else:
                self._set(SafetyLevel.CLIP,
                          "目标超限截断(%d/%d): %s" % (self.consecutive_clips, self.clip_to_brake, detail))
            self._last_filtered = filtered
            return self._clamp_limits(filtered)

        # ---- 正常
        self.consecutive_clips = 0     # 只有连续计数，一帧正常就清零
        self._set(SafetyLevel.NORMAL, "")
        self._last_filtered = q_target.copy()
        # ★ 这条路径也必须过限位夹取！它是最常走的一条 —— 偏移框（站姿 ±max_target_offset）
        #   不越限【不代表】关节限位不越限：两者原点不同（偏移框相对站姿，限位是绝对角度）。
        return self._clamp_limits(self._last_filtered)

    # ------------------------------------------------------------ 关节限位夹取
    def _clamp_limits(self, t):
        """把最终目标夹进 URDF 关节限位内。这是最后一层，也是现在唯一一层行程保护。

        与"目标偏移框"的区别：偏移框管的是"策略离站姿有多远"（OOD 判据，超了要升级
        BRAKE）；这里管的是"绝对不许越过关节限位"（物理保护，不升级，只计数）。
        两者作用在不同坐标系原点，不能互相替代 —— 偏移框允许的区间会越出限位。
        """
        if not self.clamp_joint_limits:
            self._last_filtered = t
            return t
        m = self.joint_limit_margin_rad
        lo, hi = spec.JOINT_LOWER + m, spec.JOINT_UPPER - m
        c = np.clip(np.asarray(t, dtype=np.float64), lo, hi)
        hit = np.where(np.abs(c - np.asarray(t, dtype=np.float64)) > 1e-9)[0]
        if hit.size:
            self.total_joint_clamps += 1
            self.last_joint_clamps = [spec.JOINT_LABELS[int(i)] for i in hit]
        self._last_filtered = c
        return c

    # ------------------------------------------------------------ 力矩检查（可选）
    def check_torque(self, tau) -> tuple:
        """检查 PD 计算出的原始力矩是否顶到 TAU_LIMIT。

        返回 (bool_clipped, tau_clipped)。判【裁剪】而不是【超限就停】：
        力矩顶限在腿式机器人里正常发生（快速摆腿、抗扰），停机反而危险
        （站立中突然卸力会摔）。真正危险的是【持续】顶限，那由上层统计
        本方法的返回值来处理。
        """
        tau = np.asarray(tau, dtype=np.float64).flatten()
        if tau.size != spec.NUM_JOINTS:
            raise ValueError("tau 应为 %d 维，实际 %d" % (spec.NUM_JOINTS, tau.size))
        if not np.all(np.isfinite(tau)):
            self._set(SafetyLevel.BRAKE, "tau 含 NaN/Inf")
            self.total_brakes += 1
            return True, np.zeros(spec.NUM_JOINTS)
        clipped = np.clip(tau, -spec.TAU_LIMIT, spec.TAU_LIMIT)
        hard = np.abs(tau) > self.tau_ratio_warn * spec.TAU_LIMIT
        if np.any(hard):
            # 只记原因，不升 BRAKE（见上）
            bad = np.where(hard)[0]
            self.reason = ("力矩接近上限(%d/%d 关节): %s"
                           % (bad.size, spec.NUM_JOINTS,
                              ", ".join(spec.JOINT_LABELS[i] for i in bad[:4])))
            if self.level is SafetyLevel.NORMAL:
                # 不改变 level，但留下 reason 供日志
                pass
        return bool(np.any(np.abs(tau) > spec.TAU_LIMIT)), clipped

    # ------------------------------------------------------------ 其他
    def reset(self) -> None:
        """重新进入策略模式时调用：清计数与等级（不清用户急停 latch，
        那个必须走 clear_estop 人工确认）。"""
        self.level = SafetyLevel.NORMAL
        self.reason = ""
        self.consecutive_clips = 0
        self._last_filtered = self.default_dof_pos.copy()

    @property
    def should_brake(self) -> bool:
        return self.level in (SafetyLevel.BRAKE, SafetyLevel.ESTOP)

    def status_line(self) -> str:
        return ("monitor[%s] clip=%d/%d %s"
                % (self.level.name, self.consecutive_clips, self.clip_to_brake,
                   self.reason or "-"))

    # ------------------------------------------------------------ 内部
    def _set(self, level: SafetyLevel, reason: str) -> None:
        # 等级只升不降（除 NORMAL 外）；BRAKE 后没有显式 reset 就不会自己回 CLIP
        if level.value >= self.level.value or level is SafetyLevel.NORMAL:
            self.level = level
            self.reason = reason


# ============================================================================
if __name__ == "__main__":
    """自测：正常 / 限幅 / 连续限幅升级 / NaN 目标 / 倾倒 / 角速度 / 急停 / 力矩。"""
    m = SafetyMonitor(max_target_offset=0.6, clip_to_brake=3)

    def ok_target(**kw):
        t = spec.DEFAULT_DOF_POS.copy()
        for i, v in kw.items():
            t[int(i)] = v
        return t

    def flat_state(quat=None, omega=None):
        return {"quat_wxyz": quat, "omega": omega}

    # 1) 正常目标 → NORMAL，原样透传
    out = m.filter_targets(ok_target())
    print("正常目标      ->", m.level, "| 透传:", np.allclose(out, spec.DEFAULT_DOF_POS))
    assert m.level is SafetyLevel.NORMAL and np.allclose(out, spec.DEFAULT_DOF_POS)

    # 1b) 目标偏离站姿 0.5 rad（< 0.6）→ 仍 NORMAL
    t = ok_target(); t[1] = spec.DEFAULT_DOF_POS[1] + 0.5
    out = m.filter_targets(t)
    print("偏移 0.5      ->", m.level)
    assert m.level is SafetyLevel.NORMAL

    # 2) 单帧超限 → CLIP，且被截断到边界
    t = ok_target(); t[1] = spec.DEFAULT_DOF_POS[1] + 1.2   # 超出 0.6 一倍
    out = m.filter_targets(t)
    print("偏移 1.2      ->", m.level, "| 截断后偏移 = %+.3f" % (out[1] - spec.DEFAULT_DOF_POS[1]))
    assert m.level is SafetyLevel.CLIP
    assert abs(out[1] - spec.DEFAULT_DOF_POS[1] - 0.6) < 1e-9
    # 下一帧正常 → 计数清零
    m.filter_targets(ok_target())
    assert m.level is SafetyLevel.NORMAL and m.consecutive_clips == 0

    # 3) 连续 3 帧超限 → 第 3 帧升级 BRAKE
    t = ok_target(); t[4] = spec.DEFAULT_DOF_POS[4] - 0.9
    m.filter_targets(t); print("连续超限 1/3  ->", m.level)
    assert m.level is SafetyLevel.CLIP
    m.filter_targets(t); print("连续超限 2/3  ->", m.level)
    assert m.level is SafetyLevel.CLIP
    out = m.filter_targets(t); print("连续超限 3/3  ->", m.level, m.reason)
    assert m.level is SafetyLevel.BRAKE and m.should_brake
    # BRAKE 后仍返回【截断后的安全目标】，不是透传
    assert abs(out[4] - spec.DEFAULT_DOF_POS[4] + 0.6) < 1e-9

    # 4) NaN 目标 → BRAKE + 返回站姿（fail-safe）
    m.reset()
    t = ok_target(); t[7] = np.nan
    out = m.filter_targets(t)
    print("NaN 目标      ->", m.level, "| 返回站姿:", np.allclose(out, spec.DEFAULT_DOF_POS))
    assert m.level is SafetyLevel.BRAKE and np.allclose(out, spec.DEFAULT_DOF_POS)

    # 5) 倾倒状态 → BRAKE（roll 90° → pg.z ≈ 0）
    m.reset()
    from interface.imu import euler_to_quat_wxyz
    out = m.filter_targets(ok_target(), flat_state(quat=euler_to_quat_wxyz(np.pi / 2, 0, 0)))
    print("倾倒          ->", m.level, m.reason)
    assert m.level is SafetyLevel.BRAKE and np.allclose(out, spec.DEFAULT_DOF_POS)

    # 6) 角速度爆表 → BRAKE
    m.reset()
    m.filter_targets(ok_target(), flat_state(omega=np.array([0, 0, 12.0])))
    print("|w|=12        ->", m.level, m.reason)
    assert m.level is SafetyLevel.BRAKE

    # 7) 急停 latch：触发后即使恢复正常状态也保持 ESTOP
    m.reset()
    m.request_estop("自测急停")
    m.filter_targets(ok_target(), flat_state())
    print("急停中        ->", m.level)
    assert m.level is SafetyLevel.ESTOP
    m.clear_estop()
    m.filter_targets(ok_target())
    print("急停解除      ->", m.level)
    assert m.level is SafetyLevel.NORMAL

    # 8) 力矩检查：hip 限 10 Nm，给 15 → 被裁到 10 并报告
    tau = np.zeros(spec.NUM_JOINTS); tau[0] = 15.0; tau[2] = -25.0   # calf 限 20
    hit, tau_c = m.check_torque(tau)
    print("力矩裁剪      -> hit=%s tau[0]=%.1f tau[2]=%.1f | %s"
          % (hit, tau_c[0], tau_c[2], m.reason))
    assert hit and tau_c[0] == 10.0 and tau_c[2] == -20.0

    # 8b) ★ 关节限位夹取：把偏移框放大到"会越过关节限位"，验证它确实兜住
    #     ⚠️ 为什么不能用默认的 ±0.6 去测：2026-10-02 起默认站姿变成 (0,-0.20,+0.10)、
    #     calf 限位变成 ±0.85，于是 ±0.6 的偏移框（calf 最大 0.10+0.6=0.70）【不再越限】——
    #     这一层在当前配置下不会触发，它是"参数被改大时"的兜底。所以要显式把框放大来测。
    m2 = SafetyMonitor(max_target_offset=1.0, clip_to_brake=3)
    t = ok_target(); t[2] = spec.DEFAULT_DOF_POS[2] + 1.0     # 0.10+1.0=1.10 > 0.85
    out = m2.filter_targets(t)
    print("calf 框内越限  -> 目标 %+.3f → 偏移框先截到 %+.3f → 限位夹到 %+.3f（上限 %+.3f）"
          % (spec.DEFAULT_DOF_POS[2] + 1.0,
             spec.DEFAULT_DOF_POS[2] + m2.max_target_offset, out[2], spec.JOINT_UPPER[2]))
    assert abs(out[2] - spec.JOINT_UPPER[2]) < 1e-9, "越出关节限位的目标没有被夹住"
    assert m2.total_joint_clamps >= 1
    print("   （当前默认参数下偏移框不会越限；这一层是参数被改大时的兜底，")
    print("     也是板侧固件软限位被删除后唯一的【绝对】行程保护）")
    m.reset()

    # 9) 维度错误必须抛异常（bug 不能被静默吞掉）
    try:
        m.filter_targets(np.zeros(11))
        raise AssertionError("应当抛 ValueError")
    except ValueError as e:
        print("维度错误      -> ValueError ✓")

    print("统计: clips=%d brakes=%d | %s" % (m.total_clips, m.total_brakes, m.status_line()))
    print("✅ safety_monitor 自测通过")
