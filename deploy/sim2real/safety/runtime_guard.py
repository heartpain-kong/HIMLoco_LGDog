#!/usr/bin/env python3
"""RuntimeGuard —— 运行期守护（每个控制周期调用一次，无副作用）。

与 RC_WheelLeg 的 RuntimeGuard 同构，但按 LgDog1（12 关节四足）重写：

设计原则（为什么这么写）：
- 守护只【判定】不【动作】：不下发电流、不打印刷屏（除非 verbose=True 调试），
  返回 (level, reason) 由主循环决定。把"判定"和"动作"分开，才能对每条检查
  单独写单元测试，也避免检查代码里隐藏 I/O 导致 50Hz 周期超时。
- 任何一项 STOP 就必须停：真机代码"宁可误停，不可漏停"。误停的代价是
  一次抱起机器人复位（几分钟）；漏停的代价是摔机/伤人（不可逆）。
- 阈值全部由 main.py 从 spec.py / config.yaml 读好后以形参传入，
  本模块不读 yaml，也不抄任何限位数字（spec.py 是唯一来源）。

检查顺序即优先级：急停 > NaN/Inf > IMU 陈旧 > 倾倒 > 角速度 > 关节越限。
靠前的项是"数据本身不可信"，一旦命中后面的检查没有意义，直接短路返回。
"""
import os
import sys
import time
from collections import deque
from enum import Enum

import numpy as np

# 与 interface/imu.py 相同的路径处理：保证本文件既能被包导入，也能单独 `python xxx.py` 自测
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import spec                                    # noqa: E402  常量唯一来源
from tools.math_utils import get_gravity_orientation   # noqa: E402


class GuardLevel(Enum):
    """守护等级。OK=继续，WARN=记录但不停，STOP=主循环应立刻 io.damping_brake() 并退出当前阶段。"""
    OK = 0
    WARN = 1
    STOP = 2

    def __str__(self):
        return self.name


class RuntimeGuard:
    """12 关节四足的运行期状态守护。

    覆盖：用户急停 / NaN-Inf / IMU 陈旧 / 倾倒 / 角速度爆表 / 关节越限。
    不监控目标偏移（那是 SafetyMonitor.filter_targets 的职责），两边职责
    分开的原因：guard 检查的是【测量到的世界】，monitor 检查的是【策略想
    干什么】——前者坏了说明感知或本体出错，后者坏了说明策略 OOD，处置
    动作不同（前者直接刹车，后者先限幅观察）。

    :param max_ang_vel:     |omega| 上限 [rad/s]（config.safety.max_ang_vel=10）。
                            来源：机器人摔倒翻滚时角速度可达 10~20 rad/s，
                            正常步态 <3 rad/s；取 10 是"摔倒已发生或正在发生"的判据。
    :param max_tilt_z:      projected_gravity.z 阈值（config.safety.max_tilt_z=-0.3）。
                            本实现中 pg.z ∈ [-1,0]，水平站立时 ≈ -1，倾角 θ 越大
                            pg.z = -cosθ 越接近 0；pg.z 上穿 -0.3 意味着机身倾斜
                            超过 ~72°，已无法靠腿收回，继续跑策略只会输出无意义的
                            挣扎动作。（注意：判据是 pg.z【大于】阈值，见 check 内注释）
    :param imu_age_warn_ms: IMU 数据龄超过此值告警（60ms > 1 个控制周期 20ms，
                            说明至少丢了一帧，观测开始用旧数据，先警告观察）。
    :param imu_age_stop_ms: 超过此值直接停（200ms = 10 个控制周期）。此时
                            projected_gravity/omega 早已不代表当前姿态，策略输入
                            全是幻觉，必须停；不停会在看不见的地方摔。
    :param joint_margin:    关节角允许超出 URDF 限位的余量 [rad]。给余量而不是
                            拿限位当硬边界：正常触地/越障时软限位附近有少量压缩，
                            直接用限位会频繁误停；但超限 0.15 rad 以上说明
                            关节映射顺序或零位标定错了（策略不知道自己在顶限位），
                            这种系统性错误必须停。
    :param event_history:   保留最近 N 条事件供日志（默认 200 ≈ 50Hz 下 4 秒）。
    """

    def __init__(self,
                 max_ang_vel: float = 10.0,
                 max_tilt_z: float = -0.3,
                 imu_age_warn_ms: float = 60.0,
                 imu_age_stop_ms: float = 200.0,
                 joint_margin: float = 0.15,
                 event_history: int = 200,
                 verbose: bool = False):
        self.max_ang_vel = float(max_ang_vel)
        self.max_tilt_z = float(max_tilt_z)
        self.imu_age_warn_ms = float(imu_age_warn_ms)
        self.imu_age_stop_ms = float(imu_age_stop_ms)
        # 越限余量取半程的保守值：LgDog1 最窄的限位是 calf ±0.7 rad，
        # 0.15 rad ≈ 21% 量程，足以区分"顶到软限位"和"标定/映射错误"
        self.joint_margin = float(joint_margin)
        self.verbose = bool(verbose)

        self.last_level = GuardLevel.OK
        self.last_reason = ""
        self._estop = False
        self._estop_reason = ""
        self.events = deque(maxlen=int(event_history))

    # ------------------------------------------------------------ 用户急停
    def request_estop(self, reason: str = "") -> None:
        """人为急停（键盘/遥控器/上位机触发）。 latch 住：一旦按下，必须显式 clear_estop()
        才能恢复——急停是安全回路，不能因为下一帧数据正常就自动解除。"""
        self._estop = True
        self._estop_reason = reason or "user E-stop"
        self._log(GuardLevel.STOP, self._estop_reason)

    def clear_estop(self) -> None:
        """清除急停。只应在人工确认现场安全后调用。"""
        self._estop = False
        self._estop_reason = ""

    @property
    def estop_active(self) -> bool:
        return self._estop

    # ------------------------------------------------------------ 主检查
    def check(self, state) -> tuple:
        """每个控制周期调用一次。state 可以是 io.read_state() 返回的 dict，
        也可以是带 .gyro/.quat_wxyz/.age_ms() 的 ImuSample（鸭子类型）。

        返回 (GuardLevel, reason)。STOP 时 reason 是给人看的一句话，
        会原样进日志并写入 self.last_reason。
        """
        level, reason = self._check_impl(state)
        self.last_level = level
        self.last_reason = reason
        if level is not GuardLevel.OK:
            self._log(level, reason)
        if self.verbose and level is not GuardLevel.OK:
            print("[guard] %s %s" % (level.name, reason))
        return level, reason

    def _check_impl(self, state) -> tuple:
        # ---- 1) 用户急停：最高优先级，其余检查都跳过（人都喊停了，数据对错无关）
        if self._estop:
            return GuardLevel.STOP, self._estop_reason or "user E-stop"

        # 兼容两种输入：dict（io.read_state）或 ImuSample
        q, dq, quat, omega, imu_age_ms = self._extract(state)

        # ---- 2) NaN/Inf：数据链路坏（CAN 抖动、解析错位、驱动 bug）。
        # NaN 进观测会让策略输出任意动作，且 pd 控制里 NaN 会传到电流指令——
        # 这是唯一一个"下一周期也不会自己变好"的故障，必须立即停。
        for name, arr in (("q", q), ("dq", dq), ("quat_wxyz", quat), ("omega", omega)):
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float64)
            if arr.size and (not np.all(np.isfinite(arr))):
                return GuardLevel.STOP, "NaN/Inf in %s: %s" % (name, np.array2string(arr, precision=3))

        # ---- 2.5) 与下位机的链路：反馈超时说明"感知与本体都断了"。
        # 由后端在 state["link_alive"] 里给出（host_link: 电机反馈帧龄 > 200ms）。
        # ★ 加这一条的原因：链路一断，q/dq 就冻结在最后一帧上，观测【看起来正常】
        #   但全是旧数据 —— 策略会照着一个幻影机器人输出动作。主循环当然也会刹车，
        #   但起立流程只认 guard.should_stop()，所以判据必须放在 guard 里。
        # 兼容性：state 里没有这个键（老自测/自造后端）时跳过，行为与以前一致。
        if isinstance(state, dict) and state.get("link_alive") is False:
            return GuardLevel.STOP, "与下位机链路中断（电机反馈超时）"

        # ---- 3) IMU 陈旧：角度见 __init__ 注释
        if imu_age_ms is not None:
            if imu_age_ms > self.imu_age_stop_ms:
                return GuardLevel.STOP, "IMU stale %.0f ms (limit %.0f)" % (imu_age_ms, self.imu_age_stop_ms)
            # 超 warn 阈值：至少丢了一帧（20ms/帧），观测开始用旧姿态数据，
            # 先警告不停——短暂抖动恢复后策略能跟上，持续陈旧会升级成 STOP
            stale_warn = imu_age_ms > self.imu_age_warn_ms
            stale_msg = "IMU age %.0f ms (warn %.0f)" % (imu_age_ms, self.imu_age_warn_ms)
        else:
            # 拿不到数据龄（比如自测喂的裸 dict）视作最坏情况？不：主循环保证
            # 这个字段存在；缺席只在干跑/单测中出现，这里放行但记一条 WARN。
            stale_warn = True
            stale_msg = "IMU age unavailable（干跑/自测模式？）"

        # ---- 4) 倾倒：pg.z 由四元数算，站立 ≈ -1。先算重力投影再查角速度，
        # 因为倾倒（pg.z 失效）比角速度大更根本——慢速侧翻角速度并不大但同样致命。
        projected_gravity = None
        if quat is not None:
            quat = np.asarray(quat, dtype=np.float64)
            if np.all(np.isfinite(quat)):
                # 四元数归一化：下位机回传的四元数经常带数值误差甚至未归一，
                # get_gravity_orientation 假设单位四元数，不归一会算出错误的倾角
                n = float(np.linalg.norm(quat))
                if n < 1e-6:
                    return GuardLevel.STOP, "quat norm ~ 0, IMU data invalid"
                projected_gravity = get_gravity_orientation(quat / n)
                # 判定方向说明：get_gravity_orientation 在水平时 pg.z = -1（重力
                # 指向下），倾斜 θ 时 pg.z = -cosθ 随倾角【增大】向 0 靠拢。
                # 所以"倾倒"是 pg.z 从 -1 向上越过阈值，即 pg.z > max_tilt_z
                # （与 RC_WheelLeg 参考实现一致）。若反过来写 pg.z < 阈值，
                # 机器人一开机水平站立就会被判倾倒误停。
                if float(projected_gravity[2]) > self.max_tilt_z:
                    return GuardLevel.STOP, ("tilt: pg.z=%.3f > %.3f (机身倾倒)"
                                             % (projected_gravity[2], self.max_tilt_z))

        # ---- 5) 角速度爆表：刚体被抛起/翻滚的特征。正常 trot 步态 |w| < 3 rad/s，
        # 训练域随机化上限也在 10 以内，超了说明机器人已不受控。
        if omega is not None:
            omega = np.asarray(omega, dtype=np.float64)
            if np.all(np.isfinite(omega)):
                w_norm = float(np.linalg.norm(omega))
                if w_norm > self.max_ang_vel:
                    return GuardLevel.STOP, "ang_vel overflow: |w|=%.2f > %.2f" % (w_norm, self.max_ang_vel)

        # ---- 6) 关节角越限（带余量）：正常控制不应该越过 URDF 限位 + margin。
        # 这里判的是【测量值】越限——电机真的顶死在机械限位上。若策略目标正常
        # 而测量持续越限，几乎一定是关节顺序映射错或零位标定错（见 spec.py
        # 头注释：go1 站姿套在 calf ±0.7 限位上会直接顶死）。继续通电会烧电机。
        if q is not None and dq is not None:
            q = np.asarray(q, dtype=np.float64)
            if q.size == spec.NUM_JOINTS:
                over_lo = spec.JOINT_LOWER - self.joint_margin - q
                over_hi = q - (spec.JOINT_UPPER + self.joint_margin)
                bad = np.where((over_lo > 0) | (over_hi > 0))[0]
                if bad.size > 0:
                    i = int(bad[0])
                    return GuardLevel.STOP, (
                        "joint %s=%.3f 越出限位[%.2f, %.2f]+%.2f（映射或标定错误？）"
                        % (spec.JOINT_LABELS[i], q[i],
                           spec.JOINT_LOWER[i], spec.JOINT_UPPER[i], self.joint_margin))

        # ---- 全部通过
        if stale_warn:
            return GuardLevel.WARN, stale_msg
        return GuardLevel.OK, ""

    # ------------------------------------------------------------ 对外查询
    def should_stop(self) -> bool:
        """主循环每周期问一次：True 就调 io.damping_brake() 并退出策略循环。"""
        return self.last_level is GuardLevel.STOP

    def status_line(self) -> str:
        """一行状态，给终端/日志用。"""
        return "guard[%s] %s" % (self.last_level.name, self.last_reason or "-")

    # ------------------------------------------------------------ 内部
    @staticmethod
    def _extract(state):
        """从 dict 或 ImuSample 里抠出 (q, dq, quat, omega, imu_age_ms)，缺项为 None。"""
        if isinstance(state, dict):
            q = state.get("q")
            dq = state.get("dq")
            quat = state.get("quat_wxyz")
            omega = state.get("omega")
            age = state.get("imu_age_ms")
            return q, dq, quat, omega, (None if age is None else float(age))
        # ImuSample（鸭子类型：有 gyro / quat_wxyz / age_ms）
        quat = getattr(state, "quat_wxyz", None)
        omega = getattr(state, "gyro", None)
        age = None
        if hasattr(state, "age_ms"):
            try:
                age = float(state.age_ms())
            except Exception:
                age = None
        return None, None, quat, omega, age

    def _log(self, level: GuardLevel, reason: str) -> None:
        self.events.append((time.monotonic(), level.name, reason))


# ============================================================================
if __name__ == "__main__":
    """自测：正常 / NaN / 陈旧 / 倾倒 / 角速度 / 关节越限 / 急停，逐项断言。"""
    g = RuntimeGuard(max_ang_vel=10.0, max_tilt_z=-0.3,
                     imu_age_warn_ms=60.0, imu_age_stop_ms=200.0)

    def ok_state(**kw):
        """构造一个"一切正常"的状态，再按需覆盖字段。"""
        s = {
            "q": spec.DEFAULT_DOF_POS.copy(),
            "dq": np.zeros(spec.NUM_JOINTS),
            "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),   # 水平
            "omega": np.zeros(3),
            "imu_age_ms": 5.0,                             # 刚到的数据
        }
        s.update(kw)
        return s

    # 1) 正常 → OK
    lv, r = g.check(ok_state())
    print("正常          ->", lv, r)
    assert lv is GuardLevel.OK and not g.should_stop()

    # 2) IMU 轻微陈旧 → WARN（记录但不停）
    lv, r = g.check(ok_state(imu_age_ms=100.0))
    print("IMU 100ms     ->", lv, r)
    assert lv is GuardLevel.WARN

    # 3) IMU 严重陈旧 → STOP
    lv, r = g.check(ok_state(imu_age_ms=250.0))
    print("IMU 250ms     ->", lv, r)
    assert lv is GuardLevel.STOP and g.should_stop()

    # 4) NaN 关节角 → STOP
    q_bad = spec.DEFAULT_DOF_POS.copy(); q_bad[7] = np.nan
    lv, r = g.check(ok_state(q=q_bad))
    print("q 含 NaN      ->", lv, r)
    assert lv is GuardLevel.STOP and "NaN" in r

    # 5) dq 含 Inf → STOP
    dq_bad = np.zeros(spec.NUM_JOINTS); dq_bad[3] = np.inf
    lv, r = g.check(ok_state(dq=dq_bad))
    print("dq 含 Inf     ->", lv, r)
    assert lv is GuardLevel.STOP

    # 6) 倾倒：roll=90° 时 pg.z ≈ 0 > -0.3 → STOP
    sys.path.insert(0, os.path.join(_ROOT))  # 复用 imu.py 的欧拉角工具
    from interface.imu import euler_to_quat_wxyz
    lv, r = g.check(ok_state(quat_wxyz=euler_to_quat_wxyz(np.pi / 2, 0.0, 0.0)))
    print("侧倾 90°      ->", lv, r)
    assert lv is GuardLevel.STOP and "tilt" in r

    # 6b) 轻微倾斜 30°：pg.z = -cos30 ≈ -0.87，仍在安全范围 → OK（验证不会误停）
    lv, r = g.check(ok_state(quat_wxyz=euler_to_quat_wxyz(np.pi / 6, 0.0, 0.0)))
    print("侧倾 30°      ->", lv, r)
    assert lv is GuardLevel.OK

    # 7) 角速度爆表 → STOP
    lv, r = g.check(ok_state(omega=np.array([0.0, 0.0, 11.0])))
    print("|w|=11        ->", lv, r)
    assert lv is GuardLevel.STOP and "ang_vel" in r

    # 8) 关节越限：FL_calf 顶出「限位 + 余量」→ STOP，且原因里带关节名。
    #    ★ 判据全部由 spec.JOINT_LOWER/UPPER 推出来，不写死数字 ——
    #      2026-10-02 calf 限位从 ±0.7 改成 ±0.85（对齐模型文件）时，
    #      写死 -1.0 的旧写法就不再生效（-1.0 恰好落在 ±0.85+0.15 的边界上），
    #      自测当场变红 → 这类"改了模型没改测试"的漂移让它自己暴露出来。
    calf_lo = float(spec.JOINT_LOWER[2])
    margin = float(g.joint_margin)
    q_over = spec.DEFAULT_DOF_POS.copy()
    q_over[2] = calf_lo - margin - 0.05           # 明确越过「限位 + 余量」
    lv, r = g.check(ok_state(q=q_over))
    print("calf 越限     ->", lv, r)
    assert lv is GuardLevel.STOP and "FL_calf" in r

    # 8b) 关节在限位 + 余量之内 → 不误停
    q_edge = spec.DEFAULT_DOF_POS.copy()
    q_edge[2] = calf_lo + 0.02                    # 限位内侧一点
    lv, r = g.check(ok_state(q=q_edge))
    print("calf 限位内侧 ->", lv, r)
    assert lv is GuardLevel.OK

    # 8c) 关节正好在「限位 + 余量」边界内 → 不该误停（余量的意义）
    q_marg = spec.DEFAULT_DOF_POS.copy()
    q_marg[2] = calf_lo - margin + 0.01
    lv, r = g.check(ok_state(q=q_marg))
    print("calf 余量边界 ->", lv, r)
    assert lv is GuardLevel.OK

    # 9) 用户急停：请求后即使数据全对也 STOP，clear 后恢复
    g.request_estop("手动测试急停")
    lv, r = g.check(ok_state())
    print("急停中        ->", lv, r)
    assert lv is GuardLevel.STOP
    g.clear_estop()
    lv, _ = g.check(ok_state())
    print("急停解除      ->", lv)
    assert lv is GuardLevel.OK

    # 10) ImuSample 鸭子类型输入也能检查
    from interface.imu import ImuSample
    lv, _ = g.check(ImuSample(gyro=np.zeros(3),
                              quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0])))
    print("ImuSample输入 ->", lv)
    assert lv is GuardLevel.OK

    print("事件日志(%d 条):" % len(g.events))
    for t, lvn, rs in list(g.events)[-5:]:
        print("  %s %s %s" % (lvn, rs, ""))
    print("status_line:", g.status_line())
    print("✅ runtime_guard 自测通过")
