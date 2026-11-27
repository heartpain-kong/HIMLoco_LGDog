#!/usr/bin/env python3
"""LgDog1 站立水平保持控制器（策略释放之前的"稳稳站着"阶段）。

与 RC_WheelLeg 原实现的差异：
  - 删掉轮子分支（本机 12 关节纯四足）。
  - 删掉 profile_h / profile_hip / profile_knee 高度查表——那是那台轮足
    的几何标定值，对 LgDog1 无意义且我们未标定。基础姿态直接用
    spec.DEFAULT_DOF_POS（训练侧默认站姿，几何站高 0.3540 m）。
  - 水平修正：按 IMU 投影重力的 roll/pitch 与角速度做 PD，修正量只叠在
    hip（横滚方向）和 thigh（俯仰方向）上。config 里四个增益默认全 0
    （未标定），此时本控制器必须原样返回默认站姿，不引入任何扰动。

姿态角公式与符号约定（机体坐标系，机体系下的重力投影 pg = get_gravity_orientation）：
  roll  = atan2(-pg_y, -pg_z)    机身向右侧倾（右低左高）→ roll > 0
  pitch = atan2(pg_x, sqrt(pg_y² + pg_z²))   机身抬头（前高后低）→ pitch > 0
  角速度取 imu 角速度体坐标分量：roll_rate ≈ omega[0]，pitch_rate ≈ omega[1]。
  修正为负反馈：corr = -kp*角度 - kd*角速度，再叠加到对应关节并夹紧。

参数全部通过构造函数传入（config.yaml 的 stand_balance: 段由调用方解析），
本模块不直接读 yaml。不依赖 torch / mujoco。
"""
import os
import sys
from typing import Any, Dict

import numpy as np

# 与本仓其他模块一致：把 sim2real 目录加入 sys.path 后按顶层包名导入
_SIM2REAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SIM2REAL_DIR not in sys.path:
    sys.path.insert(0, _SIM2REAL_DIR)
import spec  # noqa: E402  常量唯一来源
from tools.math_utils import get_gravity_orientation  # noqa: E402


class StandBalanceController:
    """默认站姿 PD 保持 + 可选水平姿态修正（增益 0 = 关闭）。"""

    def __init__(self,
                 control_dt: float = spec.CONTROL_DT,
                 kp_roll: float = 0.0,          # config: stand_balance.kp_roll（0 = 关闭）
                 kp_pitch: float = 0.0,         # config: stand_balance.kp_pitch
                 kd_roll_rate: float = 0.0,     # config: stand_balance.kd_roll_rate
                 kd_pitch_rate: float = 0.0,    # config: stand_balance.kd_pitch_rate
                 max_correction: float = 0.15,  # config: stand_balance.max_correction [rad]
                 stable_roll_deg: float = 6.0,      # config: stand_balance.stable_roll_deg
                 stable_pitch_deg: float = 8.0,     # config: stand_balance.stable_pitch_deg
                 stable_gyro_deg_s: float = 45.0,   # config: stand_balance.stable_gyro_deg_s
                 enter_hold_s: float = 1.0,         # config: stand_balance.enter_hold_s
                 kp: Any = None,
                 kd: Any = None):
        """
        Args:
            control_dt: 控制周期（spec.CONTROL_DT = 0.02s），用于稳定计时累计。
            max_correction: 单个关节修正量的绝对上限 [rad]，防止修正把关节
                推出合理范围（比如 calf 限位只有 ±0.7，见 spec.py 注释）。
            enter_hold_s: 判"稳"要求条件连续满足的时长。
            kp/kd: 每关节 PD 增益（保持用），默认 spec.KP / spec.KD。
        """
        self.control_dt = float(control_dt)
        self.kp_roll = float(kp_roll)
        self.kp_pitch = float(kp_pitch)
        self.kd_roll_rate = float(kd_roll_rate)
        self.kd_pitch_rate = float(kd_pitch_rate)
        self.max_correction = float(max_correction)
        self.stable_roll_deg = float(stable_roll_deg)
        self.stable_pitch_deg = float(stable_pitch_deg)
        self.stable_gyro_deg_s = float(stable_gyro_deg_s)
        self.enter_hold_s = float(enter_hold_s)
        self.kp = np.full(spec.NUM_JOINTS, spec.KP) if kp is None else np.asarray(kp, dtype=np.float64)
        self.kd = np.full(spec.NUM_JOINTS, spec.KD) if kd is None else np.asarray(kd, dtype=np.float64)

        self._stable_time = 0.0   # 连续满足稳定条件的秒数
        self._last_roll = 0.0
        self._last_pitch = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_roll_pitch(projected_gravity: np.ndarray):
        """从投影重力反算 roll / pitch（公式见文件头注释的符号约定）。"""
        gx, gy, gz = (float(v) for v in projected_gravity)
        # 分母加 1e-6 防止机身接近竖直时除零（此时姿态本就无意义，给出饱和值即可）
        roll = float(np.arctan2(-gy, max(1e-6, -gz)))
        pitch = float(np.arctan2(gx, np.sqrt(max(1e-6, gy * gy + gz * gz))))
        return roll, pitch

    # ------------------------------------------------------------------
    def compute_target(self, state: Dict[str, Any]) -> np.ndarray:
        """由当前 IMU 状态计算 12 维关节目标。

        state 需含 'quat_wxyz'(4,) 与 'omega'(3,)（体坐标角速度）。
        增益全 0 时逐位返回 spec.DEFAULT_DOF_POS。
        """
        quat = np.asarray(state["quat_wxyz"], dtype=np.float64)
        omega = np.asarray(state["omega"], dtype=np.float64)
        pg = get_gravity_orientation(quat)
        roll, pitch = self.estimate_roll_pitch(pg)
        self._last_roll, self._last_pitch = roll, pitch

        # 负反馈 PD：往倾斜的反方向修；角速度项提供阻尼，防止修过头振荡
        roll_corr = -self.kp_roll * roll - self.kd_roll_rate * float(omega[0])
        pitch_corr = -self.kp_pitch * pitch - self.kd_pitch_rate * float(omega[1])
        # config: stand_balance.max_correction —— 修正量夹紧，修正是"微调"不是"翻回去"
        roll_corr = float(np.clip(roll_corr, -self.max_correction, self.max_correction))
        pitch_corr = float(np.clip(pitch_corr, -self.max_correction, self.max_correction))

        target = spec.DEFAULT_DOF_POS.copy()
        if self.kp_roll != 0.0 or self.kd_roll_rate != 0.0:
            # hip 是横滚关节：roll 修正对称叠加到四条腿的 hip
            target[0::3] += roll_corr
        if self.kp_pitch != 0.0 or self.kd_pitch_rate != 0.0:
            # thigh 是俯仰关节：pitch 修正对称叠加到四条腿的 thigh
            target[1::3] += pitch_corr
        # calf 不动：它的限位只有 ±0.7（spec.JOINT_LOWER/UPPER），且加在 calf 上
        # 会改变站高，与"只调水平、不改姿态"的设计意图冲突。

        # 夹到关节限位内（保险：即使修正叠加也不许越限）
        target = np.clip(target, spec.JOINT_LOWER, spec.JOINT_UPPER)
        return target

    # ------------------------------------------------------------------
    def is_stable(self, state: Dict[str, Any]) -> bool:
        """当前是否"稳"：roll/pitch/|omega| 都在阈值内，且连续保持 enter_hold_s 秒。

        需要连续满足才计时（一旦超阈值清零重计），防止瞬时晃动被误判为稳。
        """
        quat = np.asarray(state["quat_wxyz"], dtype=np.float64)
        omega = np.asarray(state["omega"], dtype=np.float64)
        roll, pitch = self.estimate_roll_pitch(get_gravity_orientation(quat))
        gyro_deg = float(np.linalg.norm(omega)) * 180.0 / np.pi

        ok = (abs(np.degrees(roll)) <= self.stable_roll_deg
              and abs(np.degrees(pitch)) <= self.stable_pitch_deg
              and gyro_deg <= self.stable_gyro_deg_s)
        self._stable_time = self._stable_time + self.control_dt if ok else 0.0
        return self._stable_time >= self.enter_hold_s

    # ------------------------------------------------------------------
    def status_line(self) -> str:
        """一行日志：当前 roll/pitch 与稳定计时。"""
        return (f"[StandBalance] roll={np.degrees(self._last_roll):+6.2f}° "
                f"pitch={np.degrees(self._last_pitch):+6.2f}° "
                f"stable_t={self._stable_time:.2f}/{self.enter_hold_s:.2f}s "
                f"(阈值 roll≤{self.stable_roll_deg}° pitch≤{self.stable_pitch_deg}° "
                f"gyro≤{self.stable_gyro_deg_s}°/s)")


# ============================================================================
# 自测：
#   1) 增益全 0 时 compute_target 必须逐位等于 spec.DEFAULT_DOF_POS；
#   2) 增益非 0 且姿态倾斜很大时，修正量被 max_correction 夹住；
#   3) is_stable 计时逻辑：连续满足才累积，超阈值清零。
# 运行: /home/heartpain/miniconda3/envs/himloco/bin/python stand_balance.py
# ============================================================================
if __name__ == "__main__":
    def quat_from_rpy(roll, pitch, yaw=0.0):
        """ZYX 内旋欧拉角 → 四元数(wxyz)，用于构造测试姿态。"""
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ])

    # --- 自测1：增益全 0 → 原样返回默认站姿 ---
    ctrl0 = StandBalanceController()  # 全部默认（config 里增益就是 0）
    tilted = {"quat_wxyz": quat_from_rpy(0.3, 0.4), "omega": np.array([1.0, -1.0, 0.0])}
    t0 = ctrl0.compute_target(tilted)
    print("自测1 增益0: max|target-default| =",
          float(np.max(np.abs(t0 - spec.DEFAULT_DOF_POS))))
    assert np.array_equal(t0, spec.DEFAULT_DOF_POS), "增益 0 时必须逐位等于默认站姿"

    # --- 自测2：增益非 0 且倾斜巨大 → 修正被 max_correction 夹住 ---
    mc = 0.15
    ctrl1 = StandBalanceController(kp_roll=2.0, kp_pitch=2.0,
                                   kd_roll_rate=0.05, kd_pitch_rate=0.05,
                                   max_correction=mc)
    big = {"quat_wxyz": quat_from_rpy(1.0, 1.0), "omega": np.array([5.0, -5.0, 0.0])}
    t1 = ctrl1.compute_target(big)
    roll_corr = float(np.max(np.abs(t1[0::3] - spec.DEFAULT_DOF_POS[0::3])))
    pitch_corr = float(np.max(np.abs(t1[1::3] - spec.DEFAULT_DOF_POS[1::3])))
    calf_corr = float(np.max(np.abs(t1[2::3] - spec.DEFAULT_DOF_POS[2::3])))
    print(f"自测2 增益非0: hip 修正 {roll_corr:.4f} rad, thigh 修正 {pitch_corr:.4f} rad, "
          f"calf 不动 ({calf_corr:.4f})")
    assert roll_corr <= mc + 1e-12 and pitch_corr <= mc + 1e-12, "修正必须被 max_correction 夹住"
    assert np.isclose(roll_corr, mc) and np.isclose(pitch_corr, mc), "大倾斜下应饱和到上限"
    assert calf_corr == 0.0, "calf 不应被修改"

    # 小倾斜 + 小增益 → 修正远小于上限（说明不是无脑饱和）
    small = {"quat_wxyz": quat_from_rpy(0.02, 0.0), "omega": np.zeros(3)}
    t2 = ctrl1.compute_target(small)
    assert abs(t2[0] - spec.DEFAULT_DOF_POS[0]) < 0.05

    # --- 自测3：is_stable 连续计时 ---
    ctrl2 = StandBalanceController(control_dt=spec.CONTROL_DT,
                                   stable_roll_deg=6.0, stable_pitch_deg=8.0,
                                   stable_gyro_deg_s=45.0, enter_hold_s=0.5)
    level = {"quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]), "omega": np.zeros(3)}
    hit = False
    n = 0
    while not ctrl2.is_stable(level):
        n += 1
        assert n < 100, "水平静止应在 enter_hold_s 内判稳"
    print(f"自测3 水平静止 {n} 个周期后判稳（≈{n * spec.CONTROL_DT:.2f}s）")
    assert abs(n * spec.CONTROL_DT - 0.5) <= spec.CONTROL_DT + 1e-9

    # 中途插一次大 roll → 计时清零，需要重新累积
    ctrl2.is_stable(level)
    big_roll = {"quat_wxyz": quat_from_rpy(0.5, 0.0), "omega": np.zeros(3)}
    assert not ctrl2.is_stable(big_roll)
    assert ctrl2._stable_time == 0.0, "超阈值必须清零重计"
    n2 = 0
    while not ctrl2.is_stable(level):
        n2 += 1
        assert n2 < 100
    print(f"自测3 清零后重新累积 {n2} 个周期判稳")
    print("status_line:", ctrl2.status_line())
    print("\n✅ stand_balance 自测全部通过")
