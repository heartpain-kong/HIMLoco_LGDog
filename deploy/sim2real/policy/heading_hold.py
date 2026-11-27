#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""heading_hold.py —— 部署侧【朝向保持层】（2026-10-06 新增）

为什么需要它（用户 MuJoCo 反馈『为什么会在原地旋转』）
----------------------------------------------------
策略的观测里**没有自己的绝对朝向**: 46 维单步观测只有 cmd(3) + 身高指令(1) +
体坐标系角速度(3) + 重力投影(3) + 关节量，重力向量只能给出 roll/pitch，角速度只有 yaw 角速度。
训练时 yaw 指令【由环境的朝向控制器生成】:
    cmd_yaw = 0.5 × wrap(朝向目标 − 当前朝向)，并 clip 到 ±2 rad/s
（legged_robot.py::_post_physics_step_callback）——也就是说**训练时是环境在替它闭合这个环**。

因此若部署侧直接把操作杆的 wz 下发（开环，站住时 wz=0），就再没有任何东西纠正偏航:
策略自身那点偏航偏置会一直被积起来。实测（Isaac，V38@8000）:

    yaw 指令=0 时的累积朝向漂移（直行 40s）
      开环（直接下发 wz=0）      : 净转 −75.7°  (偏置 ω̄ = −0.034 rad/s)
      本层（朝向保持）           : 净转  −8.4°  (偏置 ω̄ = −0.003 rad/s)   ⇒ 漂移降 9 倍
    站立时（V38 该轮尚未修好）   : 开环 |ω_z| ≈ 3.0 rad/s ⇒ 本层可把"可纠正的偏置"压到
      ±wz_cmd_max / 跟踪增益 以内（约 1.8 rad/s），超出部分需要策略侧修（见 tuning_log V39）。

做法（与训练完全一致，不需要重训）
----------------------------------
    操作杆/遥控的 yaw 输入不再直接下发，而是【积分成朝向目标】:
        target ← wrap(target + wz_stick × dt)
    下发给策略的还是同一个公式:
        wz_cmd = clip(heading_p × wrap(target − yaw_now), ±wz_cmd_max)
⇒ 行为语义 = "按住转向键 = 以该速率转动目标朝向；松开 = 目标冻结、机器人把朝向拉回来并保持"。
   直行不画圈、站立不旋转；而真·转弯（按住键）完全不受影响。

yaw 从哪来
----------
用接口已经提供的姿态四元数（`st["quat_wxyz"]`，策略算重力投影用的同一个量）解出 yaw:
    yaw = atan2(2(w·z + x·y), 1 − 2(y² + z²))
无磁力计时它是"陀螺积分的相对偏航"，对【短时保持朝向】完全够用（漂移是慢变量，
而本层要压的是策略的偏置）；若硬件日后提供绝对航向（磁/视觉），把 quat 换成它即可。

用法
----
    from policy.heading_hold import HeadingHold
    hh = HeadingHold(p=0.5, wz_max=2.0)
    hh.reset(quat_wxyz)                      # 启用/复位/急停后调用
    wz_cmd = hh.update(quat_wxyz, wz_stick, dt)   # 每控制周期调一次
    action, ... = policy.step(..., np.array([vx, vy, wz_cmd]), ...)
"""
from __future__ import annotations

import math

__all__ = ["HeadingHold", "yaw_from_quat_wxyz", "wrap_pi"]


def wrap_pi(a: float) -> float:
    """把角度归一化到 (−π, π]（与训练 wrap_to_pi 同语义）。"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat_wxyz(quat) -> float:
    """四元数(w,x,y,z) → 世界系 yaw。"""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class HeadingHold:
    """朝向保持层（见模块 docstring）。

    参数
    ----
    p       : 比例增益，默认 0.5（与训练 _post_physics_step_callback 一致，别乱改）
    wz_max  : 下发 yaw 指令的限幅 [rad/s]，默认 2.0（训练里 heading_command 的 clip 值；
              超过它就超出策略的训练分布）
    enabled : 关掉即退回"操作杆 wz 直接下发"的开环行为
    """

    def __init__(self, p: float = 0.5, wz_max: float = 2.0, enabled: bool = True,
                 stick_eps: float = 1e-3):
        self.p = float(p)
        self.wz_max = float(wz_max)
        self.enabled = bool(enabled)
        self.stick_eps = float(stick_eps)   # 判定"操作杆在转向"的阈值 [rad/s]
        self.target = None          # type: float | None
        self.last_err = 0.0
        self.last_wz = 0.0
        self.last_yaw = 0.0

    def reset(self, quat_wxyz=None):
        """复位: 下一次 update 以当前朝向为目标（启用/急停/重新站立时调用）。"""
        self.target = None if quat_wxyz is None else yaw_from_quat_wxyz(quat_wxyz)
        self.last_err = 0.0
        self.last_wz = 0.0
        self.last_yaw = 0.0 if quat_wxyz is None else yaw_from_quat_wxyz(quat_wxyz)

    def update(self, quat_wxyz, wz_stick: float, dt: float) -> float:
        """返回本周期应下发给策略的 wz [rad/s]。

        ⚠️【2026-10-06 修正·转弯失效】第一版把操作杆速率【积分进朝向目标】再让 P 控制器去追,
        结果目标以 r 前进、误差稳定在 r/p = 4 rad > π ⇒ wrap 把误差翻到反方向 ⇒ 控制器反复反向,
        实测按住转向键 3 秒只转了 0.25 rad（应 6 rad）、MuJoCo 里 4/6 键"用不了"。
        正确做法 = 【两种模式分开】:
          转向模式: |wz_stick| > eps 时【直接下发该速率】(与原行为完全一致)，并把朝向目标同步到
                    当前朝向 ⇒ 松手那一刻自然从"新朝向"开始保持;
          保持模式: 无转向输入时下发 clamp(p×朝向误差, ±wz_max)。
        这样转弯手感与改动前逐位一致，只有"没让转时"才由本层接管。

        quat_wxyz : 当前姿态四元数（接口已有，策略算重力投影用的同一个）
        wz_stick  : 操作杆/遥控给出的【期望转向速率】[rad/s]（松开=0）
        dt        : 本控制周期时长 [s]（保持模式不用，转向模式不用；保留参数以便日后加滤波）
        """
        if not self.enabled:
            return float(wz_stick)
        yaw = yaw_from_quat_wxyz(quat_wxyz)
        self.last_yaw = yaw
        if abs(float(wz_stick)) > self.stick_eps:      # ---- 转向模式: 与改动前完全一致 ----
            self.target = yaw                          # 目标跟随当前朝向（松手即以此为保持点）
            self.last_err = 0.0
            self.last_wz = max(-self.wz_max, min(self.wz_max, float(wz_stick)))
            return self.last_wz
        if self.target is None:                        # ---- 保持模式 ----
            self.target = yaw
        self.last_err = wrap_pi(self.target - yaw)
        self.last_wz = max(-self.wz_max, min(self.wz_max, self.p * self.last_err))
        return self.last_wz

    def __repr__(self):
        return ("HeadingHold(p=%.2f, wz_max=%.2f, enabled=%s, target=%s, err=%+.3f, wz=%+.3f)"
                % (self.p, self.wz_max, self.enabled,
                   "None" if self.target is None else "%.3f" % self.target,
                   self.last_err, self.last_wz))
