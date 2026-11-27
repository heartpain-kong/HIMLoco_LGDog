#!/usr/bin/env python3
"""LgDog1 sim2real 主入口 —— 状态机 + 安全链 + 自动日志

    PC(N100) ──Type-C──> 下位机(dm02/STM32H7) ──FDCAN1/2──> 12×灵足 RS-02
    人手输入 = USB 手柄直接插上位机（下位机只做电机中继 + IMU 回传）

============================================================================
状态机
============================================================================
    INIT ──打开链路/等反馈/等 IMU──> CONNECTED ──使能电机──> ENABLED
      └─ 起立流程（PoseInitializer：软保持 → kp ramp → 余弦插值 → 保持）
                                                              │
                                        ┌─────────────────────┘
                                        ▼
                                      HOLD ──按住 run 键（死人开关）──> RUNNING
                                        ▲                                 │
                                        └──────松开 run 键（回到站立）─────┘

    任何时刻，下列任一条件成立 → DAMPED（阻尼刹车、电机不再跟随目标）：
      · 手柄 estop 键闩锁            · 手柄失联（stop_on_pad_lost）
      · RuntimeGuard 判 STOP         · 与下位机链路中断
      · SafetyMonitor 升级 BRAKE     · 手动 Ctrl-C
    DAMPED 下必须人工按回车才能重新起立（急停是安全回路，不自动恢复）。

============================================================================
在 N100 上的三条命令
============================================================================
    ./install.sh                 # 一次性装环境（venv + 依赖 + udev + 权限）
    python main.py --check       # 静态预检：常量/配置/策略/设备权限，不碰电机
    python main.py --dry-run     # 干跑：用假机器人跑通整条链路（无需硬件）
    python main.py               # 真机：起立 → 等回车 → 按住 run 键走

真机第一次上电前请【务必】先看 PORTING.md 的"上电顺序"一节。
============================================================================
"""

import argparse
import os
import signal
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import spec  # noqa: E402
from backends import find_serial_ports, list_devices, make_robot_io  # noqa: E402
from interface.motor_mapping import MotorMapping, Sim2RealMapNotVerified  # noqa: E402
from interface.teleop import GamepadReader, TeleopCommander  # noqa: E402
from policy.policy_runner import PolicyRunner  # noqa: E402
from policy.heading_hold import HeadingHold
from safety.runtime_guard import GuardLevel, RuntimeGuard  # noqa: E402
from safety.safety_monitor import SafetyLevel, SafetyMonitor  # noqa: E402
from startup.pose_initializer import PoseInitializer  # noqa: E402
from startup.stand_balance import StandBalanceController  # noqa: E402
from tools.math_utils import get_gravity_orientation  # noqa: E402
from tools.logger import LogSession  # noqa: E402


# ============================================================================
class State:
    INIT = "INIT"
    CONNECTED = "CONNECTED"
    ENABLED = "ENABLED"
    HOLD = "HOLD"
    RUNNING = "RUNNING"
    DAMPED = "DAMPED"
    DONE = "DONE"


_STATE_ORDER = (State.INIT, State.CONNECTED, State.ENABLED,
                State.HOLD, State.RUNNING, State.DAMPED, State.DONE)


# ============================================================================
class StdinWatcher:
    """后台线程读回车。用来做"人工闸口"（进策略前确认 / 急停后重新起立）。

    ★ 为什么用线程：主循环是 50Hz 硬节拍，不能阻塞在 input() 上。
      无 stdin（systemd/管道/后台）时 input() 立刻 EOF，此时它永不触发 ——
      调用方必须靠别的条件兜底（或显式 --yes）。
    """

    def __init__(self):
        self.event = threading.Event()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:            # EOF（无终端）
                return
            self.event.set()

    def wait(self, timeout=None) -> bool:
        return self.event.wait(timeout)

    def take(self) -> bool:
        """取走一次回车（消费掉，避免一次回车被两个阶段都看到）。"""
        if self.event.is_set():
            self.event.clear()
            return True
        return False

    def stop(self):
        self._stop.set()


# ============================================================================
class StageTimer:
    """逐阶段耗时统计 —— 用来回答"20ms 的预算被谁吃掉了"。

    ★ 为什么这个比"总周期"更有用：控制环跑不进 20ms 时，想知道是
      torch 推理慢、串口收包慢、还是日志刷盘慢。分开量才有可操作性
      （N100 是 4 核小机器，OMP 线程数/CPU 亲和/日志抽帧是三个真实的旋钮）。
    """

    STAGES = ("read", "policy", "safety", "write", "log")

    def __init__(self, cap: int = 600):
        self.d = {n: deque(maxlen=cap) for n in self.STAGES}
        self.step_dt = deque(maxlen=cap)      # 相邻两次策略推理的真实墙钟间隔

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.d.setdefault(name, deque(maxlen=600)).append(time.perf_counter() - t0)

    def mark_step(self, now: float, last: float):
        if last is not None:
            self.step_dt.append(now - last)

    @staticmethod
    def _p(a, q):
        return float(np.percentile(np.asarray(a) * 1000.0, q)) if len(a) else float("nan")

    def stats(self, name: str):
        a = self.d.get(name, ())
        if not len(a):
            return None
        return {"n": len(a), "p50": self._p(a, 50), "p99": self._p(a, 99),
                "max": max(a) * 1000.0}

    def report(self, target_ms: float) -> str:
        """一行行打印各阶段 p50/p99/max，并给出"计算占用预算"的比例。"""
        lines = []
        total = 0.0
        for n in self.STAGES:
            st = self.stats(n)
            if st is None:
                continue
            total += st["p50"]
            lines.append("    %-8s p50 %6.2f ms  p99 %6.2f ms  max %7.2f ms"
                         % (n, st["p50"], st["p99"], st["max"]))
        if self.step_dt:
            span = (len(self.step_dt) - 1)
            lines.append("    推理间隔 p50 %6.2f ms  p99 %6.2f ms  max %7.2f ms"
                         % (self._p(self.step_dt, 50), self._p(self.step_dt, 99),
                            max(self.step_dt) * 1000.0))
            lines.append("    观测历史时间跨度（%d 帧 × %d 个间隔）= p50 %.1f ms  p99 %.1f ms"
                         "   ← 当前契约应当恒为 %.1f ms"
                         % (spec.HISTORY_LEN, spec.HISTORY_LEN - 1,
                            5 * self._p(self.step_dt, 50), 5 * self._p(self.step_dt, 99),
                            spec.obs_history_span_ms()))
        if total > 0:
            lines.append("    计算占用：p50 %.2f ms / 预算 %.1f ms = %.0f%%（余量 %.1fx）"
                         % (total, target_ms, 100.0 * total / target_ms,
                            target_ms / max(total, 1e-6)))
        return "\n".join(lines)


# ============================================================================
class GuardedIO:
    """给任意后端套一层"读状态时顺便喂守护"的代理。

    ★★ 为什么需要这一层（这是本次移植发现的一个真实安全漏洞）★★
      PoseInitializer / StandBalanceController 判断"要不要放弃"的唯一信号是
      guard.should_stop()，而 should_stop() 读的是【上一次 check() 的结果】。
      原设计里 check() 只在主循环里调 —— 于是【起立流程期间守护完全没有被喂过】：
      IMU 在起立中途挂掉、关节读出 NaN、机身已经倒了，起立照样会一路把机器人
      往站姿掰，直到自己的超时才放弃（真机上这几秒就是硬掰）。

      修法：把"读状态"和"过守护"绑死在一处 —— 谁读状态谁就喂守护，不可能漏。
      主循环因此不再需要自己调 check()，直接看 guard.last_level 即可。
    """

    def __init__(self, io, guard, ignore_backend_fault: bool = False):
        self._io = io
        self._guard = guard
        self.reads = 0
        self.ignore_backend_fault = bool(ignore_backend_fault)

    # 转发（显式列出，保证 PoseInitializer 的鸭子类型契约完整）
    def start(self, *a, **k):
        return self._io.start(*a, **k)

    def wait_feedback_ready(self, *a, **k):
        return self._io.wait_feedback_ready(*a, **k)

    def describe(self, *a, **k):
        return self._io.describe(*a, **k)

    def enable_motors(self, *a, **k):
        return self._io.enable_motors(*a, **k)

    def write_targets(self, *a, **k):
        return self._io.write_targets(*a, **k)

    def hold_pose(self, *a, **k):
        return self._io.hold_pose(*a, **k)

    def damping_brake(self, *a, **k):
        return self._io.damping_brake(*a, **k)

    def poll(self, *a, **k):
        return self._io.poll(*a, **k)

    def close(self, *a, **k):
        return self._io.close(*a, **k)

    def __getattr__(self, name):
        # 其它属性（imu_core / imu / bus / 后端特有诊断…）原样透传
        return getattr(self._io, name)

    def read_state(self):
        st = self._io.read_state()
        self.reads += 1
        # ★ 硬件自己报的故障先上升为守护急停：这类故障是"我们与固件的约定不一致"
        #   （零点/符号/减速比/心跳），靠观测数值的守护判据看不出来，只有硬件知道。
        fault = "" if self.ignore_backend_fault else str(st.get("backend_fault") or "")
        if fault:
            self._guard.request_estop("下位机报告故障: %s" % fault)
        try:
            self._guard.check(st)
        except Exception as e:            # 守护自己崩了也不能让控制环崩
            print("[Guard] ⚠️ 状态守护异常: %r（本周期跳过）" % e)
        return st


# ============================================================================
class Pacer:
    """控制周期锁相：sleep 到 (周期 - 2ms) 再忙等。与 PoseInitializer 同法。

    ★ 为什么不用 time.sleep(dt)：Linux 的 sleep 精度约 1ms 且有调度抖动，
      50Hz 下累积起来会让"实际控制频率"偏离 50Hz —— 而观测历史是按 50Hz
      构造的，节拍不准等于给策略喂了错误的时间尺度。
    """

    def __init__(self, dt: float = spec.CONTROL_DT, busy_margin: float = 0.002):
        self.dt = float(dt)
        self.margin = float(busy_margin)
        self.next_exec = time.perf_counter()
        self.overruns = 0
        self.max_lag = 0.0
        self.n = 0
        self._last = None
        self._intervals = []          # 最近若干周期的实际时长，用于 p99

    def start(self):
        self.next_exec = time.perf_counter()
        self._last = self.next_exec
        self.n = 0
        self.overruns = 0
        self.max_lag = 0.0
        self._intervals = []

    def sleep(self):
        now = time.perf_counter()
        if self._last is not None:
            self.n += 1
            self._intervals.append(now - self._last)
            if len(self._intervals) > 4000:       # 80s @50Hz，够看分布了
                del self._intervals[:2000]
        self._last = now

        self.next_exec += self.dt
        slack = self.next_exec - time.perf_counter()
        if slack > 0.0:
            coarse = slack - self.margin
            if coarse > 0.0:
                time.sleep(coarse)
            while time.perf_counter() < self.next_exec:
                pass
        else:
            # 这一周期超时了（torch 推理被抢 CPU、串口卡顿、日志刷盘…）：
            # 重新对齐、不追帧 —— 追帧会让后面的周期连着跑，等于把时间尺度搞乱
            self.overruns += 1
            self.max_lag = max(self.max_lag, -slack)
            self.next_exec = time.perf_counter()

    @property
    def actual_freq(self) -> float:
        return 1.0 / self.dt

    def recent_stats(self, n: int = 200):
        """最近 n 个周期的统计（用于"时序资格"判定：要看现在的，不是全程平均）。"""
        a = np.asarray(self._intervals[-n:]) * 1000.0 if self._intervals else np.zeros(0)
        if not a.size:
            return None
        return {"n": int(a.size), "p50": float(np.median(a)),
                "p99": float(np.percentile(a, 99)), "max": float(a.max())}

    def report(self) -> str:
        """真实节拍统计。★ 上真机后最该看的数字：p99 是否明显超过控制周期。
        超过说明有东西在偷时间（torch 推理、串口阻塞、日志落盘），
        观测历史是按固定 20ms 构造的，节拍抖动直接等于给策略喂错的时间尺度。"""
        if not self._intervals:
            return "（无周期样本）"
        a = np.asarray(self._intervals) * 1000.0
        return ("周期 n=%d  中位 %.2f ms  p99 %.2f ms  最大 %.2f ms  "
                "超期 %d 次（最大滞后 %.1f ms）  目标 %.1f ms"
                % (self.n, float(np.median(a)), float(np.percentile(a, 99)),
                   float(a.max()), self.overruns, self.max_lag * 1000.0, self.dt * 1000.0))


# ============================================================================
class RobotApp:
    """把各层拼起来的主程序。"""

    def __init__(self, cfg: dict, args):
        self.cfg = cfg
        self.args = args
        self.cfg_path = args.config

        self.state = State.INIT
        self._prev_state = None
        self.quit = False
        self.exit_code = 0
        self.rearm = False
        self.start_time = time.time()

        # 各层（在后面按需构造，方便 --check 只做静态检查）
        self.io = None
        self.log = None
        self.policy = None
        # 【朝向保持层 2026-10-06】见 policy/heading_hold.py: 策略观测里没有绝对朝向，
        #   训练时由环境的朝向控制器闭合 yaw 环；部署若直接下发 wz=0 会累积偏航漂移（实测开环 40s 漂 76°）。
        _tcfg_hh = (self.cfg.get("teleop", {}) or {})
        self.heading_hold = HeadingHold(
            p=float(_tcfg_hh.get("heading_p", 0.5)),
            wz_max=float(_tcfg_hh.get("heading_wz_max", 2.0)),
            enabled=bool(_tcfg_hh.get("heading_hold", True)))
        self._hh_dt = float(_tcfg_hh.get("heading_hold_dt", 0.02))
        self.teleop = None
        self.pad = None
        self.guard = None
        self.monitor = None
        self.pose_init = None
        self.stand = None
        self.stdin = None
        self.pacer = Pacer(spec.CONTROL_DT)
        self.stage = StageTimer()
        self._last_step_t = None
        self._gate_warned = False
        self._policy_ok = None        # 策略指纹核对结果缓存

        # 运行统计
        self.stats = {
            "steps": 0, "hold_steps": 0, "run_steps": 0,
            "brakes": 0, "estops": 0, "releases": 0, "suppressed_steps": 0,
        }
        self._last_brake_reason = ""

    # ==================================================================== 工具
    def say(self, msg: str):
        print("[%7.3fs] %s" % (time.time() - self.start_time, msg), flush=True)

    def set_state(self, new: str, reason: str = ""):
        if new == self.state:
            return
        self._prev_state = self.state
        self.say("状态 %s → %s%s" % (self.state, new, ("（%s）" % reason) if reason else ""))
        self.state = new
        if self.log is not None:
            self.log.event("STATE", reason, frm=self._prev_state, to=new)

    # ==================================================================== 日志
    def _build_logger(self):
        log_cfg = dict(self.cfg.get("log", {}) or {})
        if self.args.no_log:
            log_cfg["enabled"] = False
        if self.args.log_dir:
            log_cfg["dir"] = self.args.log_dir
        tag = self.args.tag or ("dryrun" if self._is_dry_run() else "real")
        meta = {
            "mode": self.args.mode,
            "backend": self.args.backend,
            "argv": " ".join(sys.argv),
            "config_path": os.path.abspath(self.cfg_path) if self.cfg_path else None,
            "policy_path": getattr(self.policy, "policy_path", None),
            "policy_md5": _md5(getattr(self.policy, "policy_path", None)),
            "imu": self.cfg.get("imu", {}),
            "hostname_note": "N100 目标机 / 开发机均可",
        }
        if self.io is not None:
            meta["io"] = self.io.describe()
        if self.guard is not None:
            meta["guard"] = {
                "max_ang_vel": self.guard.max_ang_vel,
                "max_tilt_z": self.guard.max_tilt_z,
                "imu_age_warn_ms": self.guard.imu_age_warn_ms,
                "imu_age_stop_ms": self.guard.imu_age_stop_ms,
                "joint_margin": self.guard.joint_margin,
            }
        self.log = LogSession(log_cfg, meta=meta, tag=tag)

    # ==================================================================== 构造
    def _is_dry_run(self) -> bool:
        return (self.args.backend or "").lower() in ("stub", "dry", "dryrun", "dry-run")

    def _need_mapping(self) -> bool:
        """干跑后端不需要真机映射表（也不该被 verified 闸门挡住）。"""
        return not self._is_dry_run()

    def build_policy(self):
        path = resolve_policy_path(self.args)
        self.policy = PolicyRunner(path, action_scale=self.cfg.get("policy", {})
                                   .get("action_scale"), device="cpu")
        self.say("策略已加载: %s（输入 %d 维 / 输出 %d 维）"
                 % (path, self.policy.policy_in_dim, self.policy.policy_out_dim))
        self.say("  权重指纹 = %s   ← 判断『跑的是哪一版策略』只看这个（文件 md5 不可靠）"
                 % self.policy.weight_fp)
        self._check_policy_sidecar()

    def _check_policy_sidecar(self) -> bool:
        """与 models/policy.json 里登记的指纹比对。结果缓存（只报一次）。

        ★ 为什么需要：训练过程中 exported/policies/policy.pt 会被【反复覆盖】
          （本仓库实测：13:17 还是 V19 的 fdf3053d8b29，20:05 就被正在跑的 V20
          训练轮覆盖成了中间快照 28599e9bf8b1，那份从没在 sim2sim 里验过）。
          忘带 --policy 就可能拿一个没验过的权重上真机 —— 这比不跑更危险。

        ★ 比对基准固定是 <sim2real>/models/policy.json（"这台机器该跑的那份"），
          而不是"所加载文件旁边的 json"。所以换任何别的 .pt 都会报警告，
          要做 A/B 就显式加 --allow-policy-mismatch。
        """
        if self._policy_ok is not None:
            return self._policy_ok
        self._policy_ok = True

        # 身份登记表按【所加载策略旁边的那份】找（每个策略目录自带一份 policy.json）：
        #   models/policy.pt → models/policy.json
        #   V6_h/policy.pt   → V6_h/policy.json        ← 这样不同"方案"各自登记，互不干扰
        #   V43/policy_X.pt  → V43/policy_X.json       ← 一个目录放多份策略时各自登记
        # 找不到时退回 models/policy.json 并把两边指纹摆出来 —— 于是"随手指向训练仓库里
        # 某个没登记的 .pt"仍会被拦住（这正是当初要这道闸门的原因）。
        side = _sidecar_for(self.policy.policy_path)
        if side is None:
            side = os.path.join(_HERE, "models", "policy.json")
            if not os.path.isfile(side):
                self.say("  ⚠️  既没有 <策略目录>/policy.json 也没有 models/policy.json，"
                         "无法核对『跑的是哪一版』")
                return True
        try:
            import json
            with open(side, encoding="utf-8") as f:
                want = (json.load(f) or {}).get("weight_fingerprint")
        except Exception as e:
            self.say("  ⚠️  models/policy.json 读不了: %r" % e)
            return True
        if not want:
            return True
        side_rel = os.path.relpath(side, _HERE)
        if want == self.policy.weight_fp:
            self.say("  ✓ 权重指纹与 %s 登记的一致（%s）" % (side_rel, want))
            return True
        self.say("  ❌ 权重指纹不符！登记(%s) = %s" % (side_rel, want))
        self.say("                     实际加载的权重 = %s  (%s)"
                 % (self.policy.weight_fp, self.policy.policy_path))
        self.say("     说明这份权重不是当初在 sim2sim 里验过的那一份（被覆盖了？拷错了？）")
        self._policy_ok = False
        return False

    def _make_mapping(self):
        """按 config 的 hardware_map 建映射表（没配就用默认路径）。"""
        return MotorMapping(self.cfg.get("hardware_map") or None)

    def _default_policy_path(self):
        """默认策略查找顺序（便于移植：随包带一份就不用管训练仓库在哪）：
            1) --policy
            2) <sim2real>/models/policy.pt      ← 推荐：和代码一起拷到 N100
            3) <LgRobotDog>/legged_gym/logs/flat_LgDog1/exported/policies/policy.pt
        """
        local = os.path.join(_HERE, "models", "policy.pt")
        if os.path.isfile(local):
            return local
        repo = os.path.dirname(os.path.dirname(_HERE))       # -> <LgRobotDog>
        return os.path.join(repo, "legged_gym", "logs", "flat_LgDog1",
                            "exported", "policies", "policy.pt")

    def build_io(self):
        mapping = None
        if self._need_mapping():
            map_path = self.cfg.get("hardware_map") or None
            mapping = MotorMapping(map_path) if map_path else MotorMapping()
        self.io = make_robot_io(self.cfg, self.args, mapping=mapping)

    def build_safety(self):
        s = self.cfg.get("safety", {}) or {}
        self.guard = RuntimeGuard(
            max_ang_vel=float(s.get("max_ang_vel", 10.0)),
            max_tilt_z=float(s.get("max_tilt_z", -0.3)),
            imu_age_warn_ms=float(s.get("imu_age_warn_ms", 60.0)),
            imu_age_stop_ms=float(s.get("imu_age_stop_ms", 200.0)),
            verbose=False)
        self.monitor = SafetyMonitor(
            max_target_offset=float(s.get("max_target_offset", 0.6)),
            clip_to_brake=int(s.get("clip_to_brake", 3)),
            max_ang_vel=float(s.get("max_ang_vel", 10.0)),
            max_tilt_z=float(s.get("max_tilt_z", -0.3)),
            clamp_joint_limits=bool(s.get("clamp_to_joint_limits", True)),
            joint_limit_margin_rad=float(s.get("joint_limit_margin_rad", 0.0)))

    def build_teleop(self):
        tcfg = dict(self.cfg.get("teleop", {}) or {})
        if self.args.max_vx is not None:
            tcfg["max_vx"] = float(self.args.max_vx)
        if self.args.max_vy is not None:
            tcfg["max_vy"] = float(self.args.max_vy)
        if self.args.max_yaw is not None:
            tcfg["max_yaw"] = float(self.args.max_yaw)
        self.teleop = TeleopCommander(tcfg)
        if self.args.gamepad_off or not tcfg.get("enabled", True):
            backend = "stub"
        else:
            backend = tcfg.get("backend", "auto")
        kb = tcfg.get("keyboard", {}) or {}
        self.pad = GamepadReader(device=tcfg.get("device", "auto"),
                                 backend=backend,
                                 watchdog_ms=float(tcfg.get("watchdog_ms", 500.0)),
                                 verbose=self.args.verbose,
                                 keyboard_cfg=kb,
                                 keyboard_enabled=bool(kb.get("enabled", True)),
                                 keyboard_hold_ms=float(kb.get("hold_timeout_ms", 400.0)))
        self.pad.start()
        # ★ 让键盘后端用与 TeleopCommander 相同的轴号/键号（否则键盘兜底会错位）
        self.pad.bind_keyboard_map(self.teleop.ax_vx, self.teleop.ax_vy,
                                   self.teleop.ax_wz, self.teleop.btn_run,
                                   self.teleop.btn_estop)
        self.say("人手输入: %s%s"
                 % ("已关闭（--gamepad-off）" if backend == "stub"
                    else "手柄 backend=%s" % backend,
                    "" if backend == "stub" else "，死人开关 = 按住 run 键才放行策略"))

    def build_startup(self):
        st = self.cfg.get("startup", {}) or {}
        kp = self._kp(); kd = self._kd()
        self.pose_init = PoseInitializer(
            io=self.io, guard=self.guard, control_dt=spec.CONTROL_DT, kp=kp, kd=kd,
            soft_hold_duration=float(st.get("soft_hold_duration", 1.0)),
            soft_hold_kp_scale=float(st.get("soft_hold_kp_scale", 0.125)),
            ramp_kp_time=float(st.get("ramp_kp_time", 1.0)),
            transition_time_min=float(st.get("transition_time_min", 2.0)),
            transition_time_max=float(st.get("transition_time_max", 6.0)),
            transition_seconds_per_rad=float(st.get("transition_seconds_per_rad", 1.5)),
            timeout_extra=float(st.get("timeout_extra", 3.0)),
            hold_time=float(st.get("hold_time", 1.0)),
            settle_pos_threshold=float(st.get("settle_pos_threshold", 0.30)),
            settle_vel_threshold=float(st.get("settle_vel_threshold", 0.6)),
            max_dev_warn=float(st.get("max_dev_warn", 1.5)),
            max_dev_abort=float(st.get("max_dev_abort", 3.0)))
        sb = self.cfg.get("stand_balance", {}) or {}
        self.stand = StandBalanceController(
            control_dt=spec.CONTROL_DT,
            kp_roll=float(sb.get("kp_roll", 0.0)),
            kp_pitch=float(sb.get("kp_pitch", 0.0)),
            kd_roll_rate=float(sb.get("kd_roll_rate", 0.0)),
            kd_pitch_rate=float(sb.get("kd_pitch_rate", 0.0)),
            max_correction=float(sb.get("max_correction", 0.15)),
            stable_roll_deg=float(sb.get("stable_roll_deg", 6.0)),
            stable_pitch_deg=float(sb.get("stable_pitch_deg", 8.0)),
            stable_gyro_deg_s=float(sb.get("stable_gyro_dg_s", sb.get("stable_gyro_deg_s", 45.0))),
            enter_hold_s=float(sb.get("enter_hold_s", 1.0)),
            kp=self._kp(), kd=self._kd())

    # ---- 增益：config 里是每腿 3 个，展开成 12 个（hip, thigh, calf 循环）----
    def _kp(self):
        return _expand_gain(self.cfg.get("kp"), spec.KP)

    def _kd(self):
        return _expand_gain(self.cfg.get("kd"), spec.KD)

    # ==================================================================== 预检
    def run_check(self) -> int:
        """--check：静态 + 可选真机只读预检。★ 不发任何电机指令（只发心跳）。"""
        print("=" * 74)
        print("LgDog1 sim2real 预检（不使能电机、不下发目标）")
        print("=" * 74)
        ok = True

        # ---- 1) spec 常量 ----
        print("\n[1/9] 训练接口常量（spec.py，唯一来源）")
        print("      控制频率 %.1f Hz（dt=%.4f s，decimation=%d）"
              % (spec.CONTROL_FREQ, spec.CONTROL_DT, spec.DECIMATION))
        print("      观测 %d 维 = 单帧 %d × 历史 %d，帧序 %s"
              % (spec.NUM_OBS, spec.NUM_ONE_STEP_OBS, spec.HISTORY_LEN, spec.FRAME_ORDER))
        print("      PD: kp=%s kd=%s（config: %s / %s）"
              % (_fmt_gain(spec.KP), _fmt_gain(spec.KD),
                 _fmt_gain(self._kp()), _fmt_gain(self._kd())))
        _as_eff = (self.cfg.get("policy", {}) or {}).get("action_scale")
        _as_eff = spec.ACTION_SCALE if _as_eff is None else float(_as_eff)
        print("      action_scale=%.3f%s  默认站姿=%s"
              % (_as_eff,
                 ("（spec 默认 %.3f，被 config/--snapshot 覆盖）" % spec.ACTION_SCALE
                  if abs(_as_eff - spec.ACTION_SCALE) > 1e-9 else ""),
                 np.round(spec.DEFAULT_DOF_POS[:3], 3)))
        layout_ok = spec.NUM_ONE_STEP_OBS * spec.HISTORY_LEN == spec.NUM_OBS
        print("      观测分段: %s" % ("✓" if layout_ok else "✗"))
        ok &= layout_ok

        # ---- 2) config 一致性 ----
        print("\n[2/9] config.yaml 与 spec 的一致性")
        print("      当前生效: decimation=%d → %.1f Hz（周期 %.1f ms，历史跨度 %.0f ms）%s"
              % (spec.DECIMATION, spec.CONTROL_FREQ, spec.CONTROL_DT * 1000.0,
                 spec.obs_history_span_ms(),
                 "（被 --decimation 覆盖过）" if self.args.decimation else ""))
        ctrl = float(self.cfg.get("control_freq", 0) or 0)
        if abs(ctrl - spec.CONTROL_FREQ) > 1e-6:
            print("      ✗ control_freq=%s，但 spec.CONTROL_FREQ=%.1f —— 控制频率不一致会让"
                  "观测历史的实际时间尺度错掉" % (ctrl, spec.CONTROL_FREQ))
            ok = False
        else:
            print("      control_freq=%.1f ✓" % ctrl)
        kp, kd = self._kp(), self._kd()
        if not np.allclose(kp, spec.KP):
            print("      ⚠️  生效 kp=%s 与 spec 默认 %.1f 不同（来源：策略契约/--snapshot/config）"
                  % (_fmt_gain(kp), spec.KP))
        if not np.allclose(kd, spec.KD):
            print("      ⚠️  生效 kd=%s 与 spec 默认 %.1f 不同（同上；若这份策略训练时就是它，则为正常）"
                  % (_fmt_gain(kd), spec.KD))
        if np.any(kp > 500.0) or np.any(kd > 5.0):
            print("      ✗ kp/kd 超出 RS-02 MIT 模式范围（kp≤500, kd≤5）")
            ok = False
        imu_cfg = self.cfg.get("imu", {}) or {}
        if imu_cfg.get("euler_unit_is_degree"):
            print("      ⚠️  imu.euler_unit_is_degree=true，但固件 imu_task.c 用的是【弧度】"
                  "（回绕判据 3.1415926f）—— 除非你已实测确认，否则应改成 false")
        if not imu_cfg.get("apply_mount_transform", True):
            print("      ⚠️  imu.apply_mount_transform=false：IMU 装反（绕 Z 180°）的换算"
                  "被关掉了，观测里的角速度/重力会是 IMU 系而非机体系")
        print("      姿态源 = %s，安装变换 = %s"
              % (imu_cfg.get("attitude_source", "euler"),
                 "开" if imu_cfg.get("apply_mount_transform", True) else "关"))

        # ---- 3) 策略 ----
        print("\n[3/9] 策略")
        try:
            self.build_policy()
            print("      ✓ %d 维输入（%d 单步 × %d 历史） / %d 维输出，权重指纹 %s"
                  % (self.policy.policy_in_dim, spec.NUM_ONE_STEP_OBS, spec.HISTORY_LEN,
                     self.policy.policy_out_dim, self.policy.weight_fp))
        except Exception as e:
            print("      ✗ %s" % e)
            ok = False

        # ---- 4) 依赖 ----
        print("\n[4/9] 依赖")
        missing_required = []
        for mod, why, need in (("numpy", "数值", True), ("torch", "TorchScript 策略", True),
                               ("yaml", "配置解析", True),
                               ("serial", "下位机串口", not self._is_dry_run())):
            try:
                m = __import__(mod)
                print("      %-8s %-10s ✓  %s" % (mod, getattr(m, "__version__", ""), why))
            except ImportError:
                print("      %-8s %-10s %s  %s" % (mod, "", "✗" if need else "（可选）", why))
                if need:
                    ok = False
                    missing_required.append(mod)
        if missing_required:
            print("      → 在目标机(N100)上一条命令装齐：./install.sh")
            print("        （或手动：pip install -r requirements-deploy.txt "
                  "＋ torch CPU 版见该文件注释）")

        # ---- 5) 设备 ----
        print("\n[5/9] 设备")
        ports = find_serial_ports()
        if not ports:
            print("      串口: 没有发现 /dev/ttyACM* / ttyUSB* / serial/by-id/*")
            if not self._is_dry_run():
                print("            ★ 真机跑必须有下位机的 USB CDC；只做干跑请加 --dry-run")
        for d in ports:
            print("      串口: %-38s %s" % (d["path"],
                                         ("[%04x:%04x] %s" % (d["vid"], d["pid"] or 0, d["desc"])
                                          if d["vid"] is not None else "")))
        try:
            from interface.teleop import EvdevBackend
            devs = [x for x in EvdevBackend.list_devices() if x[2]]
            print("      手柄: %s" % (", ".join("%s(%s)" % (p, n) for p, n, _ in devs)
                                      if devs else "没找到像手柄的 evdev 设备（KeyboardBackend 会自动兜底）"))
        except Exception as e:
            print("      手柄: 枚举失败 %s" % e)

        # ---- 6) 真机映射表 ----
        print("\n[6/9] 真机硬件映射表")
        try:
            m = self._make_mapping()
            print("      %s" % m.describe().replace("\n", "\n      "))
            if not m.verified:
                print("      ⚠️  verified=false：策略控制会被拦住（这是有意的）。"
                      "先按 hardware_map.yaml 里的 7 步标定，确认后再改 true。")
        except Exception as e:
            print("      ✗ %s" % e)
            ok = False

        # ---- 7) ★ 关节行程 vs 下位机软限位（零点/符号/减速比的交叉检查）----
        print("\n[7/9] 关节行程 ↔ 下位机软限位（独立参考系交叉检查）")
        print("      （2026-09-27 起固件已删除软限位 → 这一段通常只剩提示，见下）")
        try:
            m2 = self._make_mapping()
            wok, wlines = m2.window_fit_report()
            for ln in wlines:
                print(ln)
            if not wok and not self._need_mapping():
                print("      （注：当前是干跑/stub 口径，不经过真机映射表，因此这条不算失败；"
                      "真机模式下它会拦住启动）")
            elif not wok:
                ok = False
                print("      ★ 这条不是『配置不好看』，而是【现在通电就起不来】：")
                print("        固件会在它自己那一侧把目标夹住（并置 err_flags bit4），")
                print("        关节到不了指令位置 —— 典型现象是某条腿的膝盖根本伸不开。")
                print("        两个根因，二选一（详见 hardware_map.yaml 的 board_limits_deg 注释）：")
                print("          (a) 电机的【零位】不是关节零位 → 用固件的 SET_ZERO 在")
                print("              机械零位重新清零（见 PORTING.md 第 5 节），或按上面")
                print("              候选值填 zero_offset（但要先实测符号！）")
                print("          (b) 固件里那 12 个 Pos_min/Pos_max 是下位机自跑步态时代的")
                print("              遗留值 → 下位机现在只是桥，把它改成机械真实行程更合理")
        except Exception as e:
            print("      ✗ 窗口检查失败: %s" % e)
            ok = False

        # ---- PD 增益换算（关节侧 → 电机侧 MIT）----
        print("\n[8/9] PD 增益换算（关节侧 → 电机侧 MIT）")
        try:
            _m = self._make_mapping()
            _kp, _kd = self._kp(), self._kd()
            print("      关节侧（策略训练时用的）: kp=%s  kd=%s"
                  % (_fmt_gain(_kp), _fmt_gain(_kd)))
            if not _m.gain_correction:
                # ★ 2026-10-08：默认改成"不换算"（依据现场树 verified:true 的实机标定表 +
                #   用户决定），所以这里不再是错误项，只把事实与 A/B 方法说清。
                print("      · gain_correction=false：按下发【关节侧原值】(hip/thigh/calf 都是 "
                      "kp=%.1f kd=%.1f)" % (spec.KP, spec.KD))
                print("        依据：现场树 ~/rl/LgRobot/sim2real 的实机标定表全程如此，机器正常。")
                print("        推导上 calf(gear=2) 若按电机侧口径应 ÷4（30→7.5）；要 A/B 就把")
                print("        hardware_map.yaml 的 gain_correction 改成 true 再看本节打印值。")
                print("        台架判定法见 rl/dog/小腿减速比_kp_kd_计算.md（稳住站姿读 calf 静态误差）。")
            else:
                print("      电机侧（实际写进 MIT 指令的）—— Kp/Kd 除以 gear_ratio²:")
                seen = set()
                for name, kpj, kpm, kdm, gear in _m.gain_report(_kp, _kd):
                    key = (round(kpm, 4), round(kdm, 4), gear)
                    if key in seen:
                        continue
                    seen.add(key)
                    print("        %-10s gear=%.1f  Kp %6.2f → %6.3f    Kd %.3f → %.4f"
                          % (name, gear, kpj, kpm,
                             float(np.broadcast_to(_kd, (spec.NUM_JOINTS,))[0]), kdm))
                print("      ★ 为什么除 gear²：MIT 的 PD 在电机编码器坐标系里算，")
                print("        角度要乘 n、力矩要除 n ⇒ 关节等效刚度 = n²·Kp_电机。")
                print("        小腿 n=2 → 除以 4（30→7.5, 0.8→0.2）；hip/thigh n=1 → 不变。")
                print("        推导与台架核实方法：rl/dog/小腿减速比_kp_kd_计算.md")
        except Exception as e:
            print("      ✗ 增益换算检查失败: %s" % e)
            ok = False

        # ---- 行程保护现状：固件删除软限位之后，这一层成了唯一护栏 ----
        print("\n[9/9] 行程保护现状（固件软限位已删除）")
        sm = self.cfg.get("safety", {}) or {}
        if not sm.get("clamp_to_joint_limits", True):
            print("      ✗ config.safety.clamp_to_joint_limits=false，而固件软限位已删除")
            print("        → 现在【没有任何】绝对行程保护（只剩电机电流限与机械限位）")
            ok = False
        else:
            print("      ✓ safety 关节限位夹取 = 开（margin %.2f rad）—— 当前唯一的绝对行程保护"
                  % float(sm.get("joint_limit_margin_rad", 0.0)))
            print("        它把最终目标无条件夹进 URDF 限位。与 max_target_offset 不同源：")
            print("        偏移框管『离站姿多远』，calf 允许到 +1.1 rad，而限位只到 +0.7 ——")
            print("        所以这一层不是冗余，是必需的。")
            print("        ⚠️ 它【零点依赖】：零点偏 δ，保护边界也偏 δ（PORTING.md 第 6 节）")

        # ---- 可选：真机只读预检 ----
        if self.args.live:
            print("\n[附加] 真机只读预检（发心跳、收反馈，不发电机指令）")
            ok &= self._check_live()

        print()
        print("=" * 74)
        print("预检结果: %s" % ("✅ 通过" if ok else "❌ 有问题（见上面 ✗）"))
        if not ok:
            print("静态项没过就先别上电。只差真机项（映射表 verified 等）时，"
                  "仍可用 --dry-run 先把链路跑通。")
        print("=" * 74)
        return 0 if ok else 1

    def _check_live(self) -> bool:
        """打开真机链路，只读地报告 IMU 与 12 个电机状态。"""
        try:
            self.build_io()
            self.build_safety()
            self.io.start(timeout=float(self.cfg.get("startup", {}).get("feedback_timeout_s", 5.0)))
        except Exception as e:
            print("      ✗ 打开链路失败: %s" % e)
            return False
        try:
            print("      %s" % self.io.describe().replace("\n", "\n      "))
            st = None
            for _ in range(30):
                self.io.poll()
                st = self.io.read_state()
                time.sleep(0.02)
            q = st["q"]
            print("      关节回读（策略顺序 FL,FR,RL,RR × hip,thigh,calf）:")
            for leg in range(spec.NUM_LEGS):
                row = ["%s=%+.3f" % (spec.JOINT_LABELS[leg * 3 + j], q[leg * 3 + j])
                       for j in range(3)]
                print("        %s" % "  ".join(row))
            print("      温度 max=%.1f°C   |tau|max=%.2f Nm   missing=0x%03X   rx=%d   drop=%d"
                  % (st["temp"].max(), np.abs(st["tau"]).max(), st["motor_missing"],
                     st["rx_frames"], st["tx_drops"]))
            err = st.get("motor_err")
            if err is not None and np.any(err):
                print("      ⚠️  电机报错: %s"
                      % ", ".join("%s=0x%02X" % (spec.JOINT_LABELS[i], err[i])
                                  for i in range(spec.NUM_JOINTS) if err[i]))
            print("      IMU（机体系，已做安装变换）: quat=%s omega=%s age=%.1fms"
                  % (np.round(st["quat_wxyz"], 4), np.round(st["omega"], 4),
                     st["imu_age_ms"]))
            pg = get_gravity_orientation(st["quat_wxyz"])
            tilt = np.degrees(np.arccos(np.clip(-float(pg[2]), -1.0, 1.0)))
            print("      投影重力=%s  → 静止倾角 %.1f°（机器人摆平后应 < 5°）"
                  % (np.round(pg, 4), tilt))
            # 原始（未变换）IMU 诊断：单位、姿态源、符号
            if hasattr(self.io, "imu_core"):
                rep = self.io.imu_core.sanity_report(20)
                print("      IMU 原始诊断（IMU 系）: |gyro|=%.4f rad/s  欧拉角vs四元数夹差=%.2f° "
                      "（与共轭的夹差=%.2f°）"
                      % (rep.get("gyro_norm", float("nan")), rep.get("crosscheck_deg", float("nan")),
                         rep.get("crosscheck_quatT_deg", float("nan"))))
                for h in rep.get("hints", []):
                    print("      ⚠️  %s" % h)
            # ---- 上行速率实测（固件标称 vs 实际到达）----
            if hasattr(self.io, "link_rate_report"):
                r = self.io.link_rate_report(2.0)
                exp = r["expected"]
                print("      上行速率实测（%.1fs）:" % r["seconds"])
                for k, name in (("imu_hz", "IMU"), ("motor_hz", "电机状态帧*"),
                                ("status_hz", "状态帧")):
                    got, want = r[k], exp[k]
                    flag = "✓" if got >= 0.9 * want else "✗"
                    print("        %-8s %6.1f Hz（固件标称 %.0f Hz）%s" % (name, got, want, flag))
                print("        * 两条总线各一帧 → 帧率 200Hz；每个电机实际 100Hz 更新")
                print("        最新值最大龄: IMU %.1f ms  电机 %.1f ms"
                      "   ← 观测里角速度/位置的【最坏新鲜度】"
                      % (r["imu_max_age_ms"], r["motor_max_age_ms"]))
                print("        下位机时钟/上位机时钟 = %.4f（应≈1.0000）" % r["clock_ratio"])
                if r["imu_hz"] < 0.9 * 200.0:
                    print("      ✗ IMU 实际到得比 200Hz 慢很多 —— USB CDC 或 PC 读线程跟不上。")
                    print("        后果：obs[3:6] 的角速度可能已经老了好几毫秒。")
                    print("        排查：换更短的线/直连 USB3 口、确认没有别的程序占用串口、"
                          "看是否只在打印/刷日志时掉")
                if abs(r["clock_ratio"] - 1.0) > 0.02:
                    print("      ⚠️  下位机 tick 与上位机时钟不同步（%.4f）——"
                          "板子时钟树配置与固件注释不一致？" % r["clock_ratio"])
            if st["imu_age_ms"] > 200.0:
                print("      ✗ IMU 数据龄 %.0f ms > 200：链路在丢帧" % st["imu_age_ms"])
                return False
            return True
        finally:
            try:
                self.io.close()
            except Exception:
                pass
            self.io = None

    # ==================================================================== 主流程
    def run(self) -> int:
        # 手柄 / 安全 / 策略先建好：它们不碰硬件，失败要早失败
        self.build_safety()
        self.build_policy()
        self.build_teleop()
        self.stdin = StdinWatcher()

        # ---- 映射表闸门：真机进策略前必须 verified ----
        if self._need_mapping() and self.args.mode == "policy":
            m = self._make_mapping()
            if not m.verified and not self.args.allow_unverified:
                print()
                print(m.describe())
                print()
                try:
                    m.require_verified()
                except Sim2RealMapNotVerified as e:
                    print("❌ %s" % e)
                    print()
                    print("  · 只想先把链路和控制环跑通（不起策略、不指望它走）："
                          "python main.py --mode init")
                    print("  · 确认过映射、愿意承担风险：python main.py --allow-unverified")
                    print("  · 完全没硬件、只想验证整条链路：python main.py --dry-run")
                    return 2

        # ---- 软限位一致性闸门：装不进固件窗口就别通电（策略模式硬拦，init 模式只警告）----
        try:
            wm = self._make_mapping()
            wok, _ = wm.window_fit_report()
            if not wok and not self._need_mapping():
                wok = True          # stub 不经过真机映射表
            if not wok and not self.args.allow_window_mismatch:
                if self.args.mode == "policy":
                    print()
                    print("❌ 关节行程/默认姿态装不进下位机软限位 —— 通电也起不来，拒绝启动。")
                    print("   看诊断：python main.py --check   （[7/7] 段）")
                    print("   只想上真机观察现象：--mode init；确实要跑策略：--allow-window-mismatch")
                    return 4
                print("⚠️  --mode init：软限位装不下（见 --check 的 [7/7]），"
                      "起立时会有关节被板子夹住 —— 这正是要观察的现象")
        except Exception:
            pass

        self.build_io()
        # ★ 把"读状态"与"过守护"绑死：起立流程/站立保持也走这条读路径，
        #   于是它们在 IMU 挂掉/NaN/倾倒时会当场被 should_stop() 打断
        self.io = GuardedIO(self.io, self.guard,
                            ignore_backend_fault=self.args.ignore_backend_fault)
        self.build_startup()          # 依赖 io（read_state/write_targets）与 guard，必须最后建
        self._build_logger()

        # 策略指纹不符：默认拦住（拿没验过的权重上真机，比不跑更危险）
        if self.policy is not None and not self._check_policy_sidecar() \
                and not self.args.allow_policy_mismatch:
            self.say("❌ 策略指纹不符，拒绝启动。确认要用它请加 --allow-policy-mismatch")
            if self.log:
                self.log.event("POLICY_MISMATCH", self.policy.policy_path,
                               fingerprint=self.policy.weight_fp)
            return 3

        try:
            # ---------------- CONNECTED ----------------
            self.set_state(State.CONNECTED, "打开链路")
            fb_timeout = float(self.cfg.get("startup", {}).get("feedback_timeout_s", 5.0))
            self.io.start(timeout=fb_timeout)
            self.say(self.io.describe().splitlines()[0])
            if self.log:
                self.log.event("IO_READY", self.io.describe().splitlines()[0])

            # 静止自检（真机才有）：单位/符号/姿态源三件套
            if not self._imu_sanity():
                self._brake("IMU 自检未通过")
                return self._finish()

            # ---------------- ENABLED ----------------
            self.set_state(State.ENABLED, "使能电机")
            self.io.enable_motors(True)
            time.sleep(0.1)

            # ---------------- 起立 ----------------
            if not self.pose_init.transition_to_stand_from_current():
                self.say("起立失败，进入 DAMPED")
                self._brake("起立失败")
                return self._finish()

            # ---- 启动时缺反馈的电机：下位机在开机时上报的掩码，必须在跑策略前说清楚 ----
            st0 = self.io.read_state()
            if int(st0.get("motor_missing", 0)):
                miss = [spec.JOINT_LABELS[i] for i in range(spec.NUM_JOINTS)
                        if int(st0["motor_missing"]) & (1 << i)]
                self.say("⚠️  下位机报告启动时缺反馈的电机（%d 个）: %s"
                         % (len(miss), ", ".join(miss)))
                self.say("    这些关节的读数不可信，跑策略会直接把它们顶到限位。")
                if self.log:
                    self.log.event("MOTOR_MISSING", ", ".join(miss),
                                   mask=int(st0["motor_missing"]))
                if self.args.mode == "policy" and not self.args.allow_motor_missing:
                    self._brake("启动时缺电机反馈")
                    self.say("    确认只是上电顺序问题（重新上电下位机）后重启；"
                             "坚持要继续请加 --allow-motor-missing")
                    return self._finish()

            # ---- 时序资格自检（满负载、只写站姿）----
            if not self._timing_qualification():
                if not self.args.allow_slow_loop:
                    self.say("时序自检未通过 → 直接进入 DAMPED（按回车可重试）")
                    self._brake("时序自检未通过")
                    return self._finish()

            # ---------------- HOLD ----------------
            self.set_state(State.HOLD, "站姿保持")
            if self.cfg.get("policy", {}).get("require_user_confirm", True) and not self.args.yes:
                self.say("=" * 66)
                self.say("机器人已站好。确认周围安全后【按回车】进入保持+待命状态。")
                self.say("（保持态：只维持站姿；按住手柄 run 键才会放行策略）")
                self.say("=" * 66)
                if not self.pose_init.hold_until_user_confirm(spec.DEFAULT_DOF_POS.copy()):
                    self.say("等待确认期间被急停")
                    return self._finish()

            self._main_loop()
            return self._finish()
        except KeyboardInterrupt:
            self.say("收到 Ctrl-C")
            self._brake("用户 Ctrl-C")
            return self._finish()
        except Exception as e:
            self.say("❌ 运行期异常: %r" % e)
            import traceback
            traceback.print_exc()
            try:
                self._brake("异常")
            except Exception:
                pass
            self.exit_code = 1
            return self._finish()

    # ------------------------------------------------------------------ 静止自检
    def _imu_sanity(self) -> bool:
        """静止时给 IMU 做体检。返回是否"看起来可信"。

        真机才有这一步（干跑后端的 IMU 是理想值，检不出东西）。
        """
        if not hasattr(self.io, "imu_core"):
            return True
        self.say("IMU 静止自检（~1.5s）… 机器人请保持不动")
        rep = self.io.imu_core.sanity_report(50)
        self.say("  |gyro|=%.4f rad/s  静止倾角=%.1f°  欧拉角vs四元数=%.2f°（与共轭=%.2f°）"
                 % (rep.get("gyro_norm", float("nan")), rep.get("tilt_deg", float("nan")),
                    rep.get("crosscheck_deg", float("nan")),
                    rep.get("crosscheck_quatT_deg", float("nan"))))
        if self.log:
            self.log.event("IMU_SANITY", "", **{k: v for k, v in rep.items()
                                                if k in ("gyro_norm", "tilt_deg",
                                                         "crosscheck_deg",
                                                         "crosscheck_quatT_deg",
                                                         "frames_seen", "frames_not_ready")})
        hints = rep.get("hints", [])
        for h in hints:
            self.say("  ⚠️  %s" % h)
            if self.log:
                self.log.event("IMU_SANITY_WARN", h)
        if not hints:
            return True
        if self.args.allow_bad_imu or self.args.mode != "policy":
            self.say("  ⚠️  IMU 自检有疑点但继续（--mode %s%s）"
                     % (self.args.mode, " / --allow-bad-imu" if self.args.allow_bad_imu else ""))
            return True
        self.say("  ❌ IMU 自检没过，拒绝进入策略控制 —— 观测里的角速度/重力会整段错位。")
        self.say("     建议：python tools/imu_check.py --port <串口>  用加计当地面真值定标；")
        self.say("     确实要继续（自担风险）：加 --allow-bad-imu")
        return False

    # ------------------------------------------------------------------ 时序
    def _frame_gap_stats(self):
        """相邻两次推理的间隔统计（含它蕴含的观测历史跨度）。"""
        a = self.stage.step_dt
        if len(a) < 2:
            return None
        arr = np.asarray(a) * 1000.0
        return {"n": int(arr.size),
                "p50": float(np.median(arr)),
                "p99": float(np.percentile(arr, 99)),
                "max": float(arr.max()),
                "span_p50": float(5.0 * np.median(arr)),
                "span_p99": float(5.0 * np.percentile(arr, 99))}

    def _timing_qualification(self) -> bool:
        """★ 时序资格自检：满负载跑一段，但【只写站姿目标】。

        为什么放在这里、这么做：
          · 控制环跑不进 20ms 是"能不能跑"的问题，但它只在【真的在推理】时才暴露；
          · 起立刚完成时机器人稳稳站着，此时"照常推理 + 只写站姿"＝计算负载与跑策略
            完全一致、而机器人不会因为时序不准而乱动 —— 这是唯一能安全测量的窗口；
          · 测得的结果直接决定要不要放行策略（RUNNING 分支还有一道同样的闸门兜底）。

        为什么必须测而不能"相信代码是快的"：
          N100 是 4 核小机器，torch CPU 推理、串口收发、日志刷盘互相抢时间片。
          观测历史是按固定 20ms 构造的：周期抖动 = 给策略喂错的时间尺度（HIM 的
          估计器就是从这 6 帧里推速度的）。
        """
        if self.args.mode != "policy":
            return True
        secs = float(self.args.timing_qual_seconds or 0.0)
        if secs <= 0:
            return True
        n = max(1, int(secs / spec.CONTROL_DT))
        self.say("⏱  时序资格自检 %.1fs：每周期照常推理，但只写站姿目标"
                 "（机器人保持站立，安全；这一步决定能不能放行策略）" % secs)
        hold = spec.DEFAULT_DOF_POS.copy()
        kp, kd = self._kp(), self._kd()
        pacer = Pacer(spec.CONTROL_DT)
        pacer.start()
        zero_cmd = np.zeros(3)
        aborted = False
        for _ in range(n):
            with self.stage("read"):
                st = self.io.read_state()
            if self.guard.last_level is GuardLevel.STOP:
                self._brake("时序自检期间 guard 急停: %s" % self.guard.last_reason)
                aborted = True
                break
            t_now = time.perf_counter()
            self.stage.mark_step(t_now, self._last_step_t)
            self._last_step_t = t_now
            with self.stage("policy"):
                self.policy.step(st["q"], st["dq"], st["quat_wxyz"], st["omega"], zero_cmd,
                                 height_cmd=self.args.height)
            with self.stage("write"):
                self.io.write_targets(hold, kp, kd)
            self.io.poll()
            pacer.sleep()
        if aborted:
            return False
        fg = self._frame_gap_stats()
        limit = (self.args.timing_max_period_ms if self.args.timing_max_period_ms is not None
                 else 1.3 * spec.CONTROL_DT * 1000.0)
        self.say("  推理间隔 p50 %.2f ms / p99 %.2f ms（阈值 %.1f ms）"
                 % (fg["p50"], fg["p99"], limit))
        self.say("  观测历史跨度 p50 %.1f ms / p99 %.1f ms（当前契约恒 %.1f ms = %d 帧 × %.1f ms）"
                 % (fg["span_p50"], fg["span_p99"], spec.obs_history_span_ms(),
                    spec.HISTORY_LEN, spec.CONTROL_DT * 1000.0))
        self.say("  阶段耗时分解（预算 %.1f ms）:" % (spec.CONTROL_DT * 1000.0))
        for ln in self.stage.report(spec.CONTROL_DT * 1000.0).splitlines():
            self.say(ln)
        if self.log:
            self.log.event("TIMING_QUAL", "时序资格自检",
                           gap_p50_ms=fg["p50"], gap_p99_ms=fg["p99"],
                           span_p99_ms=fg["span_p99"], limit_ms=limit)
        ok = fg["p99"] <= limit
        if not ok:
            self.say("  ❌ 时序未达标：推理间隔 p99 %.1f ms > %.1f ms" % (fg["p99"], limit))
            self.say("     缓解手段：LGDOG_CPU=2 钉核 / OMP_NUM_THREADS=2 / "
                     "config 里 log.every 调大 / --no-log 先排除日志")
            if self.args.allow_slow_loop:
                self.say("     --allow-slow-loop 已给 → 继续，但观测历史的时间尺度是失真的")
            else:
                self.say("     策略不会被放行（抑制期会一直保持站姿）。"
                         "确认要硬跑：--allow-slow-loop")
                if self.log:
                    self.log.event("TIMING_QUAL_FAIL", "时序不达标", p99_ms=fg["p99"])
        else:
            self.say("  ✓ 时序达标（余量 %.1fx）" % (limit / max(fg["p99"], 1e-6)))
        return ok

    # ------------------------------------------------------------------ 主循环
    def _main_loop(self):
        keep_pose = spec.DEFAULT_DOF_POS.copy()
        kp, kd = self._kp(), self._kd()
        pcfg = self.cfg.get("policy", {}) or {}
        blend_time = max(1e-3, float(pcfg.get("release_target_blend_s", 0.30)))
        suppress_on = bool(pcfg.get("enable_zero_cmd_suppression", True))
        hold_zero_pose = bool(pcfg.get("hold_zero_command_pose", True))
        if self.args.no_zero_cmd_suppression:
            suppress_on = False
        cmd_hold_s = float(pcfg.get("release_command_hold_s", 0.12))
        posture_max_err = float(pcfg.get("release_posture_max_err", 0.35))

        blend = 0.0                 # 0 = 保持站姿，1 = 完全跟随策略
        cmd_active_s = 0.0
        last_status = 0.0
        fixed_cmd = self.args.cmd
        auto_run = bool(self.args.auto_run) or self._is_dry_run()
        auto_stop_at = (time.time() + float(self.args.auto_run_seconds)
                        if self.args.auto_run_seconds else None)
        total_deadline = (time.time() + float(self.args.duration)
                          if self.args.duration else None)

        self.pacer.start()
        loop_t0 = time.perf_counter()
        loop_steps = 0
        suppressed = False
        while not self.quit:
            loop_steps += 1
            target_written = None          # 本周期真正下发的目标（给日志用）
            # ============ 读状态 ============
            with self.stage("read"):
                st = self.io.read_state()
            self.stats["steps"] += 1

            # ============ 人工闸口：DAMPED 下按回车重新起立 ============
            if self.state == State.DAMPED:
                self.io.damping_brake()
                self.io.poll()
                self._log_state(st, cmd=None, blend=0.0, suppressed=False)
                if self.stdin is not None and self.stdin.take():
                    if self._rearm(st):
                        keep_pose = spec.DEFAULT_DOF_POS.copy()
                        blend = 0.0
                    else:
                        self.say("重新起立失败，仍停在 DAMPED")
                if total_deadline and time.time() > total_deadline:
                    self.quit = True
                self.pacer.sleep()
                continue

            # ============ 安全守护 ============
            # ★ check() 已经在 GuardedIO.read_state() 里做过了（起立流程也走同一条路），
            #   这里只读结果，避免同一个状态被检查两遍、事件重记。
            if self.guard.last_level is GuardLevel.STOP:
                self._brake("guard: %s" % self.guard.last_reason)
                self.pacer.sleep()
                continue
            # 兜底：guard 不认识的兜底后端（例如自造后端没有 link_alive 字段）也不放过
            if st.get("link_alive") is False:
                self._brake("与下位机链路中断（电机反馈超时）")
                self.pacer.sleep()
                continue

            # ============ 人手输入 ============
            pad_st = self.pad.get()
            pad_alive = self.pad.is_alive()
            cmd = self.teleop.update(pad_st, alive=pad_alive)
            if self.args.gamepad_off and self._is_dry_run():
                # 干跑且关了输入：用固定指令（否则永远没有速度指令，跑不到策略分支）
                cmd.vx, cmd.vy, cmd.wz = fixed_cmd
                cmd.run_policy = True
            if auto_run:
                cmd.run_policy = True
                if fixed_cmd != (0.0, 0.0, 0.0):
                    cmd.vx, cmd.vy, cmd.wz = fixed_cmd
            if self.args.cmd_is_fixed and not auto_run:
                cmd.vx, cmd.vy, cmd.wz = fixed_cmd

            if cmd.estop:
                self._estop("手柄急停（闩锁）")
                self.pacer.sleep()
                continue
            if cmd.pad_lost and self.cfg.get("teleop", {}).get("stop_on_pad_lost", True) \
                    and not self.args.gamepad_off:
                self._estop("手柄失联")
                self.pacer.sleep()
                continue

            # ============ 状态分支 ============
            # 保持态的目标：默认站姿，可选叠一点点水平修正（stand_balance 增益默认 0）
            # ★ HOLD 与 RUNNING 里"混合的锚点"必须是同一个目标，否则从 HOLD 切进
            #   RUNNING 的瞬间目标会跳一下（增益非 0 时就能看出来）。
            hold_target = (self.stand.compute_target(st)
                           if self.cfg.get("stand_balance", {}).get("enabled", True)
                           else keep_pose)

            if self.state == State.HOLD:
                self.stats["hold_steps"] += 1
                blend = max(0.0, blend - spec.CONTROL_DT / blend_time)
                if cmd.run_policy and self.args.mode == "policy":
                    self.set_state(State.RUNNING, "死人开关按下 → 放行策略")
                    self.monitor.reset()
                    self.say("策略已放行。松开 run 键即回到站立保持。")
                self.io.write_targets(hold_target, kp, kd)
                target_written = hold_target

            elif self.state == State.RUNNING:
                self.stats["run_steps"] += 1
                if not cmd.run_policy:
                    self.stats["releases"] += 1
                    self.set_state(State.HOLD, "松开 run 键 → 回到站立保持")
                    self.io.write_targets(hold_target, kp, kd)
                    self.pacer.sleep()
                    continue

                # ---- 策略推理：★ 无论是否抑制目标，都照常喂历史 ----
                # 为什么必须照常喂：观测里 [33:45] 是 last_action、[9:33] 是当前姿态，
                # 都是【真实发生过的】量。若在保持期跳过推理，历史就会出现时间空洞，
                # 松开死人开关那一刻策略拿到的是陈旧历史，等于让它蒙着眼睛起步。
                # 记录相邻两次推理的真实墙钟间隔 —— 这就是"观测历史帧距"的实测值
                _t_now = time.perf_counter()
                self.stage.mark_step(_t_now, self._last_step_t)
                self._last_step_t = _t_now
                with self.stage("policy"):
                    # 【朝向保持层】操作杆 wz → 朝向目标积分 → 下发 0.5×朝向误差(clip ±2)
                    _wz_cmd = self.heading_hold.update(st["quat_wxyz"], cmd.wz, self._hh_dt)
                    action, one_step, obs_hist = self.policy.step(
                        st["q"], st["dq"], st["quat_wxyz"], st["omega"],
                        np.array([cmd.vx, cmd.vy, _wz_cmd]),
                        height_cmd=self.args.height)
                    policy_target = self.policy.target_from_action(action)

                # ---- 安全监视：过滤"策略想写的目标" ----
                with self.stage("safety"):
                    filtered = self.monitor.filter_targets(policy_target, st)
                if self.monitor.should_brake:      # ★ 是 property，不是方法
                    self._brake("safety: %s" % self.monitor.reason)
                    self.pacer.sleep()
                    continue

                # ---- 零指令抑制 / 释放混合 ----
                # 判定"有指令"用的是【指令是否真的非零】而不是"键按着没"：
                # 按着 run 键但摇杆居中时，策略在零指令下的输出是无意义的抖腿，
                # 与其让它划水，不如稳稳保持站姿。姿态误差过大时不抑制（说明
                # 机器人已经不在站姿附近，此时把目标拉回站姿反而危险，交给策略）。
                motion = float(np.hypot(cmd.vx, cmd.vy) + abs(cmd.wz)) > 1e-6
                posture_err = float(np.max(np.abs(st["q"] - hold_target)))
                if suppress_on and (not motion) and posture_err < posture_max_err:
                    cmd_active_s = 0.0
                else:
                    cmd_active_s += spec.CONTROL_DT
                suppressed = (suppress_on and cmd_active_s < cmd_hold_s)

                # ★★ 时序资格闸门 ★★
                # 抑制期里"策略照常推理、但写出去的是站姿"，也就是【计算负载已经和跑策略
                # 完全一样了，机器人却还稳稳站着】—— 这是唯一一个可以安全测时序的时刻。
                # 所以在这里判：最近 200 个周期的 p99 有没有超预算；超了就先不放行
                # （否则观测历史的时间尺度是错的，等于给策略喂错的时间轴）。
                # ★ 只在"本来就要放行"时判一次，不阻塞抑制期的其它逻辑。
                if (not suppressed) and (not self.args.allow_slow_loop):
                    # ★ 判据用【相邻两次推理的间隔】而不是"所有循环周期"：
                    #   定义观测历史时间尺度的是前者。把所有周期都算进去会把
                    #   DAMPED/HOLD 那些不喂历史的周期（有刹车重发、有打印）也算进来，
                    #   结果虚高、误拦。（实测：全体周期 p99=25.4ms，而推理间隔 p99=20.1ms。）
                    limit = (self.args.timing_max_period_ms
                             if self.args.timing_max_period_ms is not None
                             else 1.3 * spec.CONTROL_DT * 1000.0)
                    rs = self._frame_gap_stats()
                    if rs and rs["n"] >= 50 and rs["p99"] > limit:
                        suppressed = True
                        cmd_active_s = 0.0
                        if not self._gate_warned:
                            self._gate_warned = True
                            self.say("⏳ 时序未达标，暂不放行策略：推理间隔 p99 = %.1f ms > 阈值 %.1f ms"
                                     % (rs["p99"], limit))
                            self.say("   观测历史按固定 20ms 构造，周期抖动 = 给策略喂错的时间尺度。")
                            self.say("   先看阶段耗时报告（收尾时会打印）找是谁在吃预算："
                                     "推理？串口？日志？"
                                     "缓解：LGDOG_CPU 钉核 / OMP_NUM_THREADS=2 / log.every 调大")
                            self.say("   确认要硬跑：加 --allow-slow-loop")
                            if self.log:
                                self.log.event("TIMING_GATE", "推理间隔 p99 超预算，拒绝放行策略",
                                               p99_ms=rs["p99"], limit_ms=limit,
                                               history_span_p99_ms=rs["span_p99"])
                        self.stats["gate_blocks"] = self.stats.get("gate_blocks", 0) + 1

                if suppressed:
                    self.stats["suppressed_steps"] += 1

                if suppressed and hold_zero_pose:
                    blend = max(0.0, blend - spec.CONTROL_DT / blend_time)
                    target = hold_target
                else:
                    blend = min(1.0, blend + spec.CONTROL_DT / blend_time)
                    target = hold_target + blend * (filtered - hold_target)

                with self.stage("write"):
                    self.io.write_targets(target, kp, kd)
                target_written = target

                if self.log:
                    self.log.add_array("obs", obs_hist)
                    self.log.add_array("action", action)
                    self.log.add_array("target_q", target)
                    self.log.add_array("q", st["q"])
                    self.log.add_array("dq", st["dq"])

            else:
                # 其它状态（例如 --mode init）：只保持站姿
                self.stats["hold_steps"] += 1
                self.io.write_targets(keep_pose, kp, kd)
                target_written = keep_pose

            # ============ 心跳 / 日志 / 节拍 ============
            self.io.poll()
            with self.stage("log"):
                self._log_state(st, cmd=cmd, blend=blend, suppressed=suppressed,
                                target=target_written)
            if self.log and self.state == State.RUNNING:
                pass

            now = time.time()
            if now - last_status >= 2.0:
                last_status = now
                self._status_line(st, cmd, blend, suppressed if self.state == State.RUNNING else False)

            if auto_stop_at and time.time() > auto_stop_at:
                self.say("--auto-run-seconds 到点，自动刹车退出")
                self._brake("auto-run 结束")
            if total_deadline and time.time() > total_deadline:
                self.say("--duration 到点，退出")
                self.quit = True
            if self.stdin is not None and self.stdin.take() and self.state == State.HOLD \
                    and self.args.enter_quits:
                self.say("收到回车（--enter-quits），退出")
                self.quit = True

            self.pacer.sleep()

        self._loop_elapsed = time.perf_counter() - loop_t0
        self._loop_steps = loop_steps
        self._brake("主循环结束")

    # ------------------------------------------------------------------ 辅助
    def _rearm(self, st) -> bool:
        """DAMPED 下的人工重新起立。

        三道门槛，全过才动：
          1) 手柄急停没闩锁（人得先松开那个键）
          2) 清掉 guard/monitor 的急停闩锁
          3) ★ 用【当前实测状态】重新过一遍 guard —— 因为 guard.should_stop()
             是 PoseInitializer 唯一的打断信号，若这一帧数据依然有问题
             （NaN、IMU 陈旧、机身还倒着），起立流程会立刻被它打断，
             这里先判一次可以给出更清楚的提示，也避免"清了闩锁就硬掰"。
        """
        pad_st = self.pad.get()
        cmd = self.teleop.update(pad_st, alive=self.pad.is_alive())
        if cmd.estop:
            self.say("手柄急停仍闩锁（再按一下 estop 键解除），不重新起立")
            return False

        self.guard.clear_estop()
        self.monitor.clear_estop()
        lvl, why = self.guard.check(st)
        if lvl is GuardLevel.STOP:
            self.say("当前状态仍不合格（%s），不重新起立" % why)
            return False

        self.monitor.reset()
        # 【朝向保持层】重新起立时把目标朝向重锁到当前朝向（否则会去追起立前的旧朝向）
        self.heading_hold.reset(st["quat_wxyz"])
        self.say("重新起立…")
        self.io.enable_motors(True)
        ok = self.pose_init.transition_to_stand_from_current()
        if ok:
            self.set_state(State.HOLD, "人工确认后重新起立")
        else:
            self.say("重新起立失败（见上面 PoseInit 的输出）")
        return ok

    def _estop(self, reason: str):
        """急停：同时闩锁 guard 与 monitor，再刹车。

        ★ 为什么要惊动 guard：PoseInitializer 打断自己靠的是 guard.should_stop()。
          只刹车不闩锁的话，正好卡在"起立流程中"按急停时，起立会继续把机器人
          往站姿掰 —— 而急停的语义是"立刻停止一切动作"。
        """
        self.stats["estops"] += 1
        try:
            self.guard.request_estop(reason)
            self.monitor.request_estop(reason)
        except Exception:
            pass
        self._brake(reason)

    def _brake(self, reason: str):
        # 幂等：DAMPED 下每个周期都会被再次要求刹车，重复计数会让统计数字失去意义
        if self.state == State.DAMPED and reason == self._last_brake_reason:
            try:
                self.io.damping_brake()      # 但刹车指令要继续发（电机要一直保持阻尼）
            except Exception:
                pass
            return
        self.stats["brakes"] += 1
        self._last_brake_reason = reason
        try:
            self.io.damping_brake()
        except Exception as e:
            self.say("⚠️ 刹车指令发送失败: %r" % e)
        if self.log:
            self.log.event("BRAKE", reason, steps=self.stats["steps"])
        if self.state != State.DAMPED:
            self.set_state(State.DAMPED, reason)
        self.say("🛑 阻尼刹车: %s（按回车可重新起立，Ctrl-C 退出）" % reason)

    def _status_line(self, st, cmd, blend, suppressed):
        pg = get_gravity_orientation(st["quat_wxyz"])
        tilt = np.degrees(np.arccos(np.clip(-float(pg[2]), -1.0, 1.0)))
        flt = ""
        if getattr(self.io, "_io", self.io) is not None:
            try:
                bf = self.io._io.last_clip_joints if hasattr(self.io._io, "last_clip_joints") else []
                if bf:
                    flt = " 目标越界:%s" % ",".join(bf[:2])
            except Exception:
                pass
        rs = self.pacer.recent_stats(200)
        per = ("%.1f/%.1fms" % (rs["p50"], rs["p99"])) if rs else "?/?"
        sd = self.stage.stats("step") or {}
        sdt = self.stage.step_dt
        step_txt = ("%.1fms" % (float(np.median(sdt)) * 1000.0)) if sdt else "?"
        self.say("%-8s cmd=(%+.2f,%+.2f,%+.2f) blend=%.2f%s | q_err=%+.3f |ω|=%.2f "
                 "倾角=%.1f° imu=%.0fms 链路=%s 刹车=%d%s | 周期 %s 帧距 %s"
                 % (self.state, cmd.vx if cmd else 0.0, cmd.vy if cmd else 0.0,
                    cmd.wz if cmd else 0.0, blend,
                    " 抑制" if suppressed else "",
                    float(np.max(np.abs(st["q"] - spec.DEFAULT_DOF_POS))),
                    float(np.linalg.norm(st["omega"])), tilt, st["imu_age_ms"],
                    "✓" if st.get("link_alive", True) else "✗", self.stats["brakes"], flt,
                    per, step_txt))

    def _log_state(self, st, cmd, blend=0.0, suppressed=False, target=None):
        if self.log is None:
            return
        pg = get_gravity_orientation(st["quat_wxyz"])
        offset = st["q"] - spec.DEFAULT_DOF_POS
        self.log.state(
            t=time.time() - self.start_time,
            step=self.stats["steps"],
            level=self.state,
            cmd_vx=(cmd.vx if cmd else 0.0), cmd_vy=(cmd.vy if cmd else 0.0),
            cmd_wz=(cmd.wz if cmd else 0.0),
            run_policy=int(bool(cmd.run_policy) if cmd else 0),
            estop=int(bool(cmd.estop) if cmd else 0),
            blend=float(blend),
            suppressed=int(bool(suppressed)),
            act_min=float(np.min(self.policy.last_action)) if self.policy else 0.0,
            act_max=float(np.max(self.policy.last_action)) if self.policy else 0.0,
            act_mean=float(np.mean(self.policy.last_action)) if self.policy else 0.0,
            tgt_err_max=float(np.max(np.abs(offset))),
            tgt_sat_n=int(np.sum(np.abs(offset) > self.monitor.max_target_offset))
            if self.monitor else 0,
            # ★ 真正下发的目标相对站姿的最大偏移（含关节限位夹取之后的结果）
            tgt_off_max=(float(np.max(np.abs(np.asarray(target, dtype=np.float64)
                                             - spec.DEFAULT_DOF_POS)))
                         if target is not None else float("nan")),
            q_min=float(np.min(st["q"])), q_max=float(np.max(st["q"])),
            dq_max=float(np.max(np.abs(st["dq"]))),
            tau_max=float(np.max(np.abs(st["tau"]))),
            temp_max=float(np.max(st["temp"])),
            roll=0.0, pitch=0.0, yaw=0.0,     # 欧拉角只对日志友好，这里留 0（用 pg 更直接）
            gyro_x=float(st["omega"][0]), gyro_y=float(st["omega"][1]),
            gyro_z=float(st["omega"][2]),
            pg_x=float(pg[0]), pg_y=float(pg[1]), pg_z=float(pg[2]),
            imu_age_ms=float(st["imu_age_ms"]),
            pad_alive=int(bool(self.pad.is_alive())),
            link_alive=int(bool(st.get("link_alive", True))),
            rx_frames=int(st.get("rx_frames", 0)),
            tx_drops=int(st.get("tx_drops", 0)),
            motor_missing=int(st.get("motor_missing", 0)),
        )

    def _finish(self) -> int:
        self.set_state(State.DONE, "收尾")
        try:
            self.io.damping_brake()
        except Exception:
            pass
        time.sleep(0.05)
        for obj, name in ((self.pad, "手柄"), (self.io, "链路"), (self.stdin, "stdin")):
            try:
                if obj is not None:
                    obj.stop() if name == "手柄" else (obj.close() if name == "链路" else obj.stop())
            except Exception as e:
                print("[收尾] %s 关闭失败: %r" % (name, e))
        if self.log:
            self.log.event("RUN_SUMMARY", "",
                           elapsed_s=round(time.time() - self.start_time, 2),
                           **self.stats)
            self.log.close()
        self.say("运行统计: %s" % self.stats)
        el = float(getattr(self, "_loop_elapsed", 0.0))
        n = int(getattr(self, "_loop_steps", 0))
        self.say("控制环: 目标 %.1f Hz，实际 %d 步 / %.1f s = %.2f Hz"
                 % (self.pacer.actual_freq, n, el, (n / el) if el > 0 else 0.0))
        self.say("节拍质量: %s" % self.pacer.report())
        self.say("阶段耗时分解（控制周期预算 %.1f ms）:" % (spec.CONTROL_DT * 1000.0))
        rep = self.stage.report(spec.CONTROL_DT * 1000.0)
        for ln in rep.splitlines():
            self.say(ln)
        if self.log:
            self.log.event("TIMING_STAGES", "", report=rep,
                           pacer=self.pacer.report())
        if self.log:
            self.log.event("LOOP_TIMING", self.pacer.report(),
                           loop_steps=n, loop_seconds=round(el, 3))
        return self.exit_code


# ============================================================================
def _expand_gain(val, default: float) -> np.ndarray:
    """config 里的 kp/kd：(3,) 或标量 → 12 维（每腿 hip,thigh,calf 循环）。"""
    if val is None:
        return np.full(spec.NUM_JOINTS, float(default))
    a = np.asarray(val, dtype=np.float64).flatten()
    if a.size == 1:
        return np.full(spec.NUM_JOINTS, float(a[0]))
    if a.size == 3:
        return np.tile(a, spec.NUM_LEGS)
    if a.size == spec.NUM_JOINTS:
        return a.copy()
    raise ValueError("kp/kd 必须是标量 / 3 个（每腿）/ 12 个，实际 %d 个" % a.size)


def _fmt_gain(a) -> str:
    a = np.asarray(a, dtype=np.float64).flatten()
    if np.allclose(a, a[0]):
        return "%.3g" % a[0]
    return "[" + ",".join("%.3g" % v for v in a[:3]) + "…]"


def resolve_policy_path(args) -> str:
    """确定要用哪份策略（提到模块级：main() 要在构造任何对象之前读它的契约）。"""
    if args.policy:
        return args.policy
    local = os.path.join(_HERE, "models", "policy.pt")
    if os.path.isfile(local):
        return local
    repo = os.path.dirname(os.path.dirname(_HERE))
    return os.path.join(repo, "legged_gym", "logs", "flat_LgDog1",
                        "exported", "policies", "policy.pt")


def _sidecar_for(policy_path: str):
    """找策略的【身份登记文件】（sidecar）。

    查找顺序（先具体、后通用）：
      1) 与策略同名的 json：policy_X.pt     → policy_X.json      ← 一个目录里放多份策略时用它
      2) 目录级的 policy.json                                    ← 常规情况
    ★ 为什么要第 1 条：`dog/V43/` 这类目录会同时放"主策略 + 备选检查点"，
      两者指纹不同，却只有一份 policy.json ⇒ 备选那份一定会被【指纹闸门】拦住，
      而绕过闸门（--allow-policy-mismatch）恰恰是最不该做的事。
      给它配一份同名 sidecar 就能各自声明自己的身份与契约。
    """
    d = os.path.dirname(os.path.abspath(policy_path))
    base = os.path.splitext(os.path.basename(policy_path))[0]
    for cand in (os.path.join(d, base + ".json"), os.path.join(d, "policy.json")):
        if os.path.isfile(cand):
            return cand
    return None


def load_policy_contract(policy_path: str):
    """读策略旁边的 policy.json 里登记的【控制契约】。

    ★ 为什么让策略自带契约：观测/控制的契约是【每个 run 各自的】——
      老 run 的 action_scale=0.25、kd=0.8、50Hz；V28 那类新 run 是 0.3/1.2、50Hz；
      V6_h 是 100Hz。用一份全局常量去套所有策略，必然有一批是错的，
      而"错了不报错、只是行为不可信"（本仓库 action_scale 错配作废过 60000 步的测试）。
      所以契约跟着策略走，spec.py 只提供"没有声明时"的默认值。
    返回 dict（可能为空）：{decimation, kp, kd, action_scale, ...}
    """
    side = _sidecar_for(policy_path)
    if side is None:
        return {}
    try:
        import json as _json
        d = _json.load(open(side, encoding="utf-8")) or {}
    except Exception:
        return {}
    cc = d.get("control_contract") or {}
    cc = dict(cc)
    cc["_sidecar"] = os.path.relpath(side, _HERE)
    cc["_fingerprint"] = d.get("weight_fingerprint")

    # ★ policy.json 是人手写的，default_dof_pos 很容易被写成一句说明
    #   （例如 "每腿 (hip, thigh, calf) = (0, -0.20, +0.10)"）—— 那种值喂给
    #   np.asarray 会抛 IndexError / ValueError，在"打印契约"这一步就把整个程序带崩，
    #   而它明明只是资料，不该有这种杀伤力。所以这里只认"能解析成 NUM_JOINTS 个浮点"
    #   的值，其余一律丢掉并警告（行为不变，只是退回 spec 的默认站姿）。
    if cc.get("default_dof_pos") is not None:
        _n_j = int(getattr(spec, "NUM_JOINTS", 12))
        _ok, _why = True, ""
        try:
            import numpy as _np
            _arr = _np.asarray(cc["default_dof_pos"], dtype=_np.float64)
            if _arr.ndim != 1 or _arr.size != _n_j:
                _ok = False
                _why = "需要 %d 个数，实际 %d 个（%r）" % (
                    _n_j, int(_arr.size), cc["default_dof_pos"])
        except Exception as _e:
            _ok = False
            _why = "%s: %r" % (type(_e).__name__, cc["default_dof_pos"])
        if _ok:
            cc["default_dof_pos"] = _arr
        else:
            print("⚠️  %s 的 control_contract.default_dof_pos 不是 %d 个关节角：%s"
                  % (os.path.relpath(side, _HERE), _n_j, _why))
            print("    → 已忽略该项，本次用 spec 默认站姿 %s；"
                  "若确实要覆盖，请写成 [%s]"
                  % (["%.2f" % x for x in spec.DEFAULT_DOF_POS],
                     ", ".join("%.2f" % x for x in spec.DEFAULT_DOF_POS)))
            cc.pop("default_dof_pos", None)

    # ★ height_nominal 同理：必须是 0.10~0.60 之间的数。手写 policy.json 里
    #   很容易写成 "0.30 m" 或 "见训练配置" 这类字符串 —— 与其静默按字符串走，
    #   不如丢掉并说清楚（丢掉后退回 spec 默认，只在传 --height 时才有影响）。
    if cc.get("height_nominal") is not None:
        try:
            _hn = float(cc["height_nominal"])
            if not (0.10 <= _hn <= 0.60):
                raise ValueError("超出合理范围 [0.10, 0.60] m")
            cc["height_nominal"] = _hn
        except Exception as _e:
            print("⚠️  %s 的 control_contract.height_nominal = %r 不可用（%s）"
                  % (os.path.relpath(side, _HERE), cc["height_nominal"], _e))
            print("    → 已忽略该项，本次按 spec 默认 %.3f m 编码身高指令"
                  "（只影响 --height；不传时 obs[3] 恒 0）" % spec.HEIGHT_NOMINAL)
            cc.pop("height_nominal", None)
    return cc


def _md5(path):
    if not path or not os.path.isfile(path):
        return None
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def do_set_zero(cfg: dict, args) -> int:
    """把电机【当前的物理位置】记为电机零位（固件的 LGD_MSG_SET_ZERO → 电机自己的 zero()）。

    ★ 这是固件提供的、唯一能真正"设定绝对零点"的手段（写进电机自己的存储，
      断电不丢）。它必须【人已经把机器人摆到零位姿态】之后才能用 —— 所以这个
      函数把"我以为的当前姿态"和"我打算做的操作"都打印出来，逼人确认一遍。

    为什么不能靠软办法代替：
      上位机上报的 q = (电机角度 − zero_offset)/sign/gear，是一条【闭环自证】的链。
      零点错一个常数，起立流程照样报"✅ 已到位"，而机器人实际姿态是错的
      （PoseInitializer 的收敛判据就是"上报值到目标"）。所以零点必须由一个
      【机械参考】来锚定，set_zero 就是把这个锚定动作写进电机。

    用法（缺一不可）：
      1) 断电/阻尼状态下，用夹具或水平仪把机器人【物理上】摆到关节零位姿态
         （URDF 里 12 个关节 q=0 的构型）
      2) python main.py --set-zero          ← 会打印当前读数并要你确认
      3) python main.py --check --live      ← 复核：零位姿态下 12 个关节应都≈0
      4) 若报"rejected"，说明下位机处于使能态 —— 先断电重启下位机再试
    """
    print("=" * 74)
    print("SET_ZERO —— 把电机当前位置记为电机零位（写进电机，断电不丢）")
    print("=" * 74)
    print("""
★ 执行前必须确认（固件只在【未使能】时接受这条指令）：
   [ ] 机器人已断电或处于阻尼状态（本程序不会使能电机）
   [ ] 机器人【物理上】已经摆到关节零位姿态：URDF 里 12 个关节 q=0 的构型
   [ ] 你是【用手/夹具/水平仪】摆的，不是用本程序命令过去的
       —— 因为零点错的时候，命令过去到达的位置本身就不对
   [ ] 四条腿姿态一致、没有别着劲（否则记下来的"零位"是四条腿各不相同的）
""")
    mapping = MotorMapping(cfg.get("hardware_map") or None)
    io = make_robot_io(cfg, args, mapping=mapping)
    if not hasattr(io, "set_zero"):
        print("❌ 这个后端不支持 set_zero（干跑后端没有电机概念）。")
        try:
            io.close()
        except Exception:
            pass
        return 2
    try:
        io.start(timeout=float(cfg.get("startup", {}).get("feedback_timeout_s", 5.0)))
        for _ in range(20):
            io.poll()
            st = io.read_state()
            time.sleep(0.02)
        print("当前读数（策略顺序 FL,FR,RL,RR × hip,thigh,calf）:")
        for leg in range(spec.NUM_LEGS):
            print("  %s   %s" % (
                ("FL", "FR", "RL", "RR")[leg],
                "  ".join("%s=%+.4f" % (spec.JOINT_LABELS[leg * 3 + j], st["q"][leg * 3 + j])
                          for j in range(3))))
        print("  温度 max=%.1f°C  缺反馈=0x%03X  下位机 err_flags=0x%02X"
              % (st["temp"].max(), st.get("motor_missing", 0), st.get("board_err_flags", 0)))
        print()
        print("  ⚠️ 上面这些数字【现在看着正常也没意义】—— 它们正是由待验证的零点算出来的。")
        print("     判断『摆得对不对』只能靠尺子/水平仪/夹具，不能看这些数。")
        if not args.yes:
            print()
            print("确认要继续请输入大写 YES 回车，其它任何输入都会放弃：")
            try:
                if input().strip() != "YES":
                    print("已放弃，什么都没改。")
                    return 0
            except EOFError:
                print("没有 stdin，无法确认 —— 已放弃（脚本化时加 --yes）。")
                return 0
        io.set_zero()
        time.sleep(0.5)
        print("已下发 SET_ZERO。重新读一遍确认：")
        for _ in range(20):
            io.poll()
            st = io.read_state()
            time.sleep(0.02)
        print("  新读数 = %s" % np.round(st["q"], 4))
        print("  若与刚才不同 → 电机零位已改写；若完全一样 → 可能被拒（下位机处于使能态？）")
        print("  下一步：python main.py --check --live  （零位姿态下 12 个关节应都≈0）")
        return 0
    finally:
        try:
            io.close()
        except Exception:
            pass


def load_config(path: str, args) -> dict:
    import yaml
    cfg = {}
    if path:
        if not os.path.isfile(path):
            raise SystemExit("找不到配置文件: %s" % path)
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    # 命令行覆盖
    if args.port:
        cfg["serial_port"] = args.port
    if args.backend:
        cfg["backend"] = args.backend
    if getattr(args, "imu_attitude", None):
        cfg.setdefault("imu", {})["attitude_source"] = args.imu_attitude
    if getattr(args, "imu_degrees", False):
        cfg.setdefault("imu", {})["euler_unit_is_degree"] = True
    if getattr(args, "imu_radians", False):
        cfg.setdefault("imu", {})["euler_unit_is_degree"] = False
    if getattr(args, "no_mount_transform", False):
        cfg.setdefault("imu", {})["apply_mount_transform"] = False
    return cfg


def build_argparser():
    p = argparse.ArgumentParser(
        description="LgDog1 sim2real 主程序（N100 上位机 + dm02 下位机 + 12×RS-02）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="常用组合：\n"
               "  python main.py --check                预检（不碰电机）\n"
               "  python main.py --check --live          预检 + 真机只读检查\n"
               "  python main.py --dry-run               干跑整条链路（无需硬件）\n"
               "  python main.py --mode init             真机只起立保持，不跑策略\n"
               "  python main.py                         真机跑策略\n")
    p.add_argument("--config", default=os.path.join(_HERE, "config.yaml"))
    p.add_argument("--height", type=float, default=None, metavar="M",
                   help="身高指令 [m]（46 维观测布局才有这一维）。不给 = 该策略的名义站高 "
                        "→ 该维恒 0；给了就按【该策略 policy.json 里的 height_nominal】编码，"
                        "所以换策略不用改这个参数")
    p.add_argument("--kp", type=float, default=None, help="显式覆盖 kp（★ 只在 A/B 时用）")
    p.add_argument("--kd", type=float, default=None, help="显式覆盖 kd（★ 只在 A/B 时用）")
    p.add_argument("--action-scale", type=float, default=None,
                   help="显式覆盖 action_scale（★ 只在 A/B 时用）")
    p.add_argument("--snapshot", default=None, metavar="PATH",
                   help="训练 run 的 config_snapshot.json：★ 把 kp/kd/action_scale/decimation/"
                        "观测缩放/裁剪/默认站姿一次对齐到策略真正被训练时的值，并逐项打印对照。"
                        "换用别的 run 的策略时强烈建议带上（例：跑 V28 就指它的 snapshot）")
    p.add_argument("--allow-contract-mismatch", action="store_true",
                   help="观测维数/动作维数与快照不符时也继续（★ 通常意味着这份策略不是给 "
                        "LgDog1 12 关节用的，别硬跑）")
    p.add_argument("--policy", default=None,
                   help="策略文件（默认 models/policy.pt，其次训练仓库的 exported/policies/policy.pt）")
    p.add_argument("--backend", default=None, choices=["auto", "host_link", "stub"],
                   help="auto=只选真机（找不到串口就报错）；stub=干跑假机器人")
    p.add_argument("--port", default=None, help="下位机串口，如 /dev/ttyACM0（默认自动挑）")
    p.add_argument("--mode", default="policy", choices=["policy", "init"],
                   help="policy=完整跑策略；init=只起立并保持（标定/台架用，不需要 verified 映射表）")
    p.add_argument("--tag", default=None, help="日志目录前缀，默认 real / dryrun")
    p.add_argument("--yes", action="store_true", help="跳过'按回车进入'的人工闸口")
    p.add_argument("--dry-run", action="store_true",
                   help="等价于 --backend stub --yes --gamepad-off --auto-run（无硬件验证整条链路）")
    p.add_argument("--check", action="store_true", help="只做预检然后退出")
    p.add_argument("--live", action="store_true", help="配合 --check：加做真机只读预检（发心跳、收反馈）")
    p.add_argument("--list-devices", action="store_true", help="列出串口与手柄")
    p.add_argument("--set-zero", action="store_true",
                   help="★ 把电机当前位置记为电机零位（人先把机器人摆到关节零位姿态！）")
    p.add_argument("--gamepad-off", action="store_true", help="关掉手柄（键盘/固定指令）")
    p.add_argument("--auto-run", action="store_true", help="忽略死人开关，直接放行策略（台架/自测）")
    p.add_argument("--cmd", default=None, metavar="VX,VY,WZ",
                   help="固定速度指令（如 0.3,0,0）；台架测试与干跑用")
    p.add_argument("--cmd-is-fixed", action="store_true", help="把手柄指令整个替换成 --cmd（忽略摇杆）")
    p.add_argument("--auto-run-seconds", type=float, default=None,
                   help="跑策略多少秒后自动刹车退出")
    p.add_argument("--duration", type=float, default=None, help="整个进程最长运行秒数")
    p.add_argument("--max-vx", type=float, default=None)
    p.add_argument("--max-vy", type=float, default=None)
    p.add_argument("--max-yaw", type=float, default=None)
    p.add_argument("--decimation", type=int, default=None, choices=[1, 2, 3, 4, 5],
                   help="覆盖控制频率（decimation）：★ 换按别的频率训出来的策略时必须给。"
                        "4=50Hz(默认,历史跨度100ms) / 2=100Hz(跨度50ms)。"
                        "例：V6_h 的策略是 decimation=2 训的，跑它就要 --decimation 2")
    p.add_argument("--timing-qual-seconds", type=float, default=2.0,
                   help="起立后、放行策略前的时序资格自检时长 [s]（满负载只写站姿；0=跳过）")
    p.add_argument("--timing-max-period-ms", type=float, default=None,
                   help="放行策略前允许的最大控制周期 p99 [ms]（默认 1.3×周期=26ms）。"
                        "超了会拒绝放行策略，避免用错的时间尺度喂观测历史")
    p.add_argument("--allow-slow-loop", action="store_true",
                   help="控制环跑不进预算时也放行策略（★ 观测历史的时间尺度会失真）")
    p.add_argument("--ignore-backend-fault", action="store_true",
                   help="忽略下位机自报的故障（err_flags 的致命位 / 目标越界）。★ 只用于台架排查")
    p.add_argument("--allow-window-mismatch", action="store_true",
                   help="关节行程/默认姿态装不进下位机软限位时也继续（★ 危险，先看 --check 的 [7/7]）")
    p.add_argument("--allow-motor-missing", action="store_true",
                   help="下位机报告有电机缺反馈时也继续跑策略（★ 危险）")
    p.add_argument("--allow-policy-mismatch", action="store_true",
                   help="models/policy.json 记的权重指纹与实际不符时也继续（★ 说明权重被换过）")
    p.add_argument("--allow-unverified", action="store_true",
                   help="映射表 verified=false 也允许跑策略（★ 危险，仅台架）")
    p.add_argument("--allow-bad-imu", action="store_true", help="IMU 自检有疑点也继续")
    p.add_argument("--imu-attitude", default=None, choices=["euler", "quat"],
                   help="姿态用欧拉角重建（默认）还是直接用下位机四元数")
    p.add_argument("--imu-degrees", action="store_true", help="下位机欧拉角单位是【度】")
    p.add_argument("--imu-radians", action="store_true", help="下位机欧拉角单位是【弧度】（固件实测值）")
    p.add_argument("--no-mount-transform", action="store_true",
                   help="关掉 IMU 安装变换（★ 只在怀疑变换写错时用于对比）")
    p.add_argument("--no-zero-cmd-suppression", action="store_true",
                   help="关掉零指令抑制（零指令时也让策略直接控制电机）")
    p.add_argument("--enter-quits", action="store_true", help="保持态按回车即退出")
    p.add_argument("--stub-sag", type=float, default=0.02, help="干跑：重力下垂 [rad]")
    p.add_argument("--stub-tilt", default=None, metavar="ROLL,PITCH",
                   help="干跑：给一个固定倾角 [deg]，用来验证水平保持")
    p.add_argument("--stub-fault-nan", type=float, default=None, metavar="SEC",
                   help="干跑故障注入：第 SEC 秒起关节角变 NaN")
    p.add_argument("--stub-fault-imu-freeze", type=float, default=None, metavar="SEC",
                   help="干跑故障注入：第 SEC 秒起 IMU 停更")
    p.add_argument("--stub-fault-motor-missing", type=float, default=None, metavar="SEC")
    p.add_argument("--stub-fault-drop-link", type=float, default=None, metavar="SEC")
    p.add_argument("--stub-fault-backend", type=float, default=None, metavar="SEC",
                   help="干跑故障注入：第 SEC 秒起后端自报故障（测『硬件报故障→守护急停』这条链）")
    p.add_argument("--no-log", action="store_true", help="关掉日志")
    p.add_argument("--log-dir", default=None, help="日志根目录（默认 <sim2real>/logs）")
    p.add_argument("--verbose", action="store_true", help="打印链路细节")
    p.add_argument("--quiet-check", action="store_true", help=argparse.SUPPRESS)
    return p


# ============================================================================
def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    # --dry-run 是一组开关的糖
    if args.dry_run:
        args.backend = "stub"
        args.yes = True
        args.gamepad_off = True
        if not args.auto_run_seconds:
            args.auto_run_seconds = 8.0
        # 干跑不需要人守着：跑够时间 → 自动刹车 → 再过几秒退出进程
        # （--duration 是"整个进程最长运行秒数"，DAMPED 下到点就 quit）
        if not args.duration:
            args.duration = args.auto_run_seconds + 3.0
        args.tag = args.tag or "dryrun"
        print("[dry-run] 干跑模式：假机器人 + 跳过人工闸口 + 关手柄 + 自动放行策略")

    args.verbose = args.verbose or os.environ.get("LGDOG_VERBOSE") == "1"

    # ★★ 策略自带契约（policy.json 的 control_contract）：优先于 spec 默认值。
    #    优先级：--snapshot（训练配置原件） > 策略的 policy.json > spec.py 默认值
    #    > config.yaml 里显式写的值（config 是"人工覆盖"，仍然最高，见下面的说明）
    pol_path = resolve_policy_path(args)
    pol_contract = load_policy_contract(pol_path)
    if pol_contract:
        print("=" * 74)
        print("策略自带契约（%s）" % pol_contract.get("_sidecar"))
        for k in ("decimation", "kp", "kd", "action_scale"):
            if pol_contract.get(k) is not None:
                print("    %-14s = %s" % (k, pol_contract[k]))
        if pol_contract.get("default_dof_pos") is not None:
            print("    %-14s = %s" % ("默认站姿(每腿)",
                                      np.round(np.asarray(pol_contract["default_dof_pos"])[:3], 3)))
        if pol_contract.get("height_nominal") is not None and spec.HEIGHT_CMD_ENABLED:
            print("    %-14s = %.3f m（obs[3] 的编码基准；不给 --height 时该维恒 0）"
                  % ("height_nominal", float(pol_contract["height_nominal"])))
        print("    （这是这份策略被训练时的值；spec.py 只负责『没声明时』的默认）")
        print("=" * 74)

    # ★★ 训练快照对齐（--snapshot）：把观测/控制契约一次对齐到策略被训练时的值。
    #    必须在 load_config/RobotApp 之前做（Pacer/起立/保持都按 control_dt 构造）。
    snap_contract = {}
    if args.snapshot:
        if not os.path.isfile(args.snapshot):
            raise SystemExit("找不到快照: %s" % args.snapshot)
        import json as _json
        _snap = _json.load(open(args.snapshot, encoding="utf-8"))
        _e = _snap.get("env_cfg", _snap)
        _ctrl = _e.get("control", {}) or {}
        _norm = _e.get("normalization", {}) or {}
        _env = _e.get("env", {}) or {}
        _init = _e.get("init_state", {}) or {}
        snap_contract = {
            "decimation": _ctrl.get("decimation"),
            "kp": (_ctrl.get("stiffness") or {}).get("joint"),
            "kd": (_ctrl.get("damping") or {}).get("joint"),
            "action_scale": _ctrl.get("action_scale"),
            "obs_scales": _norm.get("obs_scales"),
            "clip": _norm.get("clip_observations"),
            "num_obs": _env.get("num_observations"),
            "num_one_step": _env.get("num_one_step_observations"),
            "num_actions": _env.get("num_actions"),
            "default_joint_angles": _init.get("default_joint_angles"),
            # ★ 身高指令的编码基准（commands.height_nominal）：obs[3] = (身高 − 它) × HC。
            #   是【每个 run 各自的】（V41 系 0.30 / V32 是 0.28），写错只在传 --height 时才显形。
            "height_nominal": (_e.get("commands", {}) or {}).get("height_nominal"),
        }
        # 默认站姿：快照是 dict(关节名)，按 spec.JOINT_LABELS 的顺序取
        dja = snap_contract["default_joint_angles"]
        if isinstance(dja, dict):
            # 快照里的键是 'FL_hip_joint'，spec.JOINT_LABELS 是 'FL_hip' —— 两种都认
            def _get(name):
                for k in (name, name + "_joint"):
                    if k in dja:
                        return float(dja[k])
                raise KeyError("快照里找不到关节 %s（键: %s…）"
                               % (name, ", ".join(list(dja)[:3])))
            snap_contract["default_dof_pos"] = np.array(
                [_get(n) for n in spec.JOINT_LABELS], dtype=np.float64)

        print("=" * 74)
        print("训练快照对齐：%s" % args.snapshot)
        print("=" * 74)
        hard = []
        def _row(name, snap_v, now_v, hard_check=False):
            ok = (snap_v is None) or (abs(float(snap_v) - float(now_v)) < 1e-9
                                      if isinstance(snap_v, (int, float)) else snap_v == now_v)
            print("    %s %-24s 当前=%s  训练=%s"
                  % ("✅" if ok else ("❌" if hard_check else "⚠️ "),
                     name, now_v, snap_v))
            if not ok and hard_check:
                hard.append("%s（当前 %s，训练 %s）" % (name, now_v, snap_v))
        _row("sim.dt", (_e.get("sim") or {}).get("dt"), spec.SIM_DT)
        _row("decimation", snap_contract["decimation"], spec.DECIMATION)
        _row("kp(stiffness)", snap_contract["kp"], spec.KP)
        _row("kd(damping)", snap_contract["kd"], spec.KD)
        _row("action_scale", snap_contract["action_scale"], spec.ACTION_SCALE)
        for k, v in (snap_contract["obs_scales"] or {}).items():
            if k == "height_measurements":
                continue
            cur = {"lin_vel": spec.LV, "ang_vel": spec.AV,
                   "dof_pos": spec.DP, "dof_vel": spec.DV}.get(k)
            if cur is not None:
                _row("obs_scales.%s" % k, v, cur, hard_check=True)
        _row("clip_observations", snap_contract["clip"], spec.CLIP, hard_check=True)
        _row("obs 维数", snap_contract["num_obs"], spec.NUM_OBS, hard_check=True)
        # 布局可以切（45/46），所以这里先按快照切、再核对；只在"两者都不是"时才硬拒
        try:
            if snap_contract["num_one_step"] in (45, 46) and \
                    int(snap_contract["num_one_step"]) != spec.NUM_ONE_STEP_OBS:
                spec.set_obs_layout(int(snap_contract["num_one_step"]))
                print("    → 按快照切到 %d 维单帧观测（总 %d 维）"
                      % (spec.NUM_ONE_STEP_OBS, spec.NUM_OBS))
        except Exception as e:
            print("    ❌ 布局切换失败: %s" % e)
        _row("单步 obs 维数", snap_contract["num_one_step"], spec.NUM_ONE_STEP_OBS, hard_check=True)
        _row("动作维数", snap_contract["num_actions"], spec.NUM_JOINTS, hard_check=True)
        _dd = snap_contract.get("default_dof_pos")
        if _dd is not None:
            same = bool(np.allclose(_dd, spec.DEFAULT_DOF_POS))
            print("    %s %-24s %s" % ("✅" if same else "⚠️ ", "默认站姿",
                                       "与 spec 一致" if same else
                                       "快照=%s  当前=%s" % (np.round(_dd[:3], 3),
                                                          np.round(spec.DEFAULT_DOF_POS[:3], 3))))
        # 身高指令基准：只在 46 维布局下有意义。
        # ★ 与默认站姿同类：它是"可以按快照对齐"的项（下面 apply_contract 会改），
        #   所以只能 ⚠️ 不能当硬项 —— 否则换 V32（0.28）这种快照时会被误判成契约不符而拒启动。
        _hn = snap_contract.get("height_nominal")
        if spec.HEIGHT_CMD_ENABLED and _hn is not None:
            _row("height_nominal(m)", _hn, spec.HEIGHT_NOMINAL)
        elif _hn is not None:
            print("    ·  height_nominal(m)          当前 %s  训练=%s"
                  "（45 维布局无身高通道，不影响）" % (spec.HEIGHT_NOMINAL, _hn))
        print()
        if hard and not args.allow_contract_mismatch:
            print("    ❌ 观测/动作契约与快照不符 —— 这不是靠参数能补的，说明这份策略")
            print("       不是按当前接口训的（或快照取错了）。拒绝启动。")
            print("       确实要硬跑：--allow-contract-mismatch")
            raise SystemExit(3)
        # 可修的项一次性应用
        ch = spec.apply_contract(obs_scales=snap_contract["obs_scales"],
                                 clip=snap_contract["clip"],
                                 default_dof_pos=_dd,
                                 decimation=snap_contract["decimation"],
                                 height_nominal=snap_contract.get("height_nominal"))
        for name, old_v, new_v in ch:
            print("    → 已对齐 %-20s %s → %s" % (name, np.round(np.atleast_1d(old_v), 4)[:4]
                                                if not isinstance(old_v, (int, float)) else old_v,
                                                np.round(np.atleast_1d(new_v), 4)[:4]
                                                if not isinstance(new_v, (int, float)) else new_v))
        if not ch:
            print("    → 契约项无需改动（spec 已经与训练配置一致）")
        print("=" * 74)

    # ★★ 控制频率覆盖：必须在 load_config/RobotApp 之前做（否则 Pacer/起立/保持
    #    会按旧周期构造，节拍就错了）。spec 仍是唯一来源，这里只是让它重算一次。
    # ★ 观测布局：也属于契约（45 维无身高通道 / 46 维有）。按策略声明切换。
    _lay = pol_contract.get("obs_one_step")
    if _lay and not args.snapshot:
        if spec.set_obs_layout(int(_lay)):
            print("→ 按策略自带契约切到 %d 维单帧观测（总 %d 维）"
                  % (spec.NUM_ONE_STEP_OBS, spec.NUM_OBS))

    if not args.decimation and not args.snapshot and pol_contract.get("decimation"):
        # 策略自带契约里的频率（例：V6_h 是 2 → 100Hz）自动生效，不必手打 --decimation
        _o, _n = spec.apply_decimation(int(pol_contract["decimation"]))
        if _o[0] != _n[0]:
            print("→ 按策略自带契约设 decimation=%d（控制周期 %.1f ms，历史跨度 %.0f ms）"
                  % (_n[0], _n[2] * 1000, (spec.HISTORY_LEN - 1) * _n[2] * 1000))
    if args.decimation:
        _old, _new = spec.apply_decimation(args.decimation)
        print("=" * 74)
        print("⚠️  控制频率被覆盖：decimation %d → %d" % (_old[0], _new[0]))
        print("    控制周期   %.1f ms → %.1f ms（控制频率 %.0f → %.0f Hz）"
              % (_old[2] * 1000, _new[2] * 1000, _old[1], _new[1]))
        print("    观测历史跨度 %.0f ms → %.0f ms（6 帧；这个值是【策略契约的一部分】）"
              % ((spec.HISTORY_LEN - 1) * _old[2] * 1000,
                 (spec.HISTORY_LEN - 1) * _new[2] * 1000))
        print("    ⇒ 只有在【策略本身就是按这个频率训的】时候才允许这么改。")
        print("    ⇒ 每个控制周期的推理预算也变成 %.1f ms（原来 %.1f ms）——"
              "时序资格自检会重新判。" % (_new[2] * 1000, _old[2] * 1000))
        print("=" * 74)

    cfg = load_config(args.config, args)
    if args.decimation or args.snapshot or pol_contract.get("decimation"):
        # config.yaml 里的 control_freq 也要跟着走，否则 --check 会报不一致
        cfg["control_freq"] = spec.CONTROL_FREQ
        cfg["decimation"] = spec.DECIMATION
    # ★ 优先级（高→低）：--kp/--kd/--action-scale(CLI) > --snapshot > 策略自带契约 >
    #   config.yaml > spec.py 默认值。
    #   为什么"策略自带契约"要压过 config.yaml：config.yaml 里那份 kp/kd 本来就是
    #   【训练值的副本】（文件头也写了"不在这里定义第二遍"），而契约是这份策略的
    #   真值。实测过：config.yaml 里写死的 kd=0.8 会把 V28 契约里的 1.2 挡住，
    #   于是静默按 0.8 跑 —— 正是我们要消灭的那类错配。
    def _apply_contract(kp=None, kd=None, action_scale=None, source="", default_dof_pos=None,
                        height_nominal=None):
        if kp is not None:
            old_v = cfg.get("kp")
            cfg["kp"] = [float(kp)] * 3
            print("→ 按%s设 kp = %s%s" % (source, kp,
                  ("（config 原值 %s）" % old_v) if old_v else ""))
        if kd is not None:
            old_v = cfg.get("kd")
            cfg["kd"] = [float(kd)] * 3
            print("→ 按%s设 kd = %s%s" % (source, kd,
                  ("（config 原值 %s）" % old_v) if old_v else ""))
        if action_scale is not None:
            old_v = (cfg.get("policy") or {}).get("action_scale")
            cfg.setdefault("policy", {})["action_scale"] = float(action_scale)
            print("→ 按%s设 action_scale = %s%s" % (source, action_scale,
                  ("（config 原值 %s）" % old_v) if old_v is not None else ""))
        if default_dof_pos is not None:
            _new = np.asarray(default_dof_pos, dtype=np.float64).reshape(-1)
            if _new.size == spec.NUM_JOINTS and not np.allclose(_new, spec.DEFAULT_DOF_POS):
                print("→ 按%s设默认站姿 %s（原 %s）"
                      % (source, np.round(_new[:3], 3), np.round(spec.DEFAULT_DOF_POS[:3], 3)))
                spec.apply_contract(default_dof_pos=_new)
        # ★ 身高指令基准：obs[3] = (目标身高 − height_nominal) × HC。只在 46 维布局下有效。
        if height_nominal is not None and spec.HEIGHT_CMD_ENABLED:
            try:
                _hn = float(height_nominal)
            except Exception:
                print("⚠️  %s的 height_nominal 不是数字（%r）→ 忽略" % (source, height_nominal))
                _hn = None
            if _hn is not None:
                if abs(_hn - spec.HEIGHT_NOMINAL) > 1e-9:
                    print("→ 按%s设 height_nominal = %.3f m（原 %.3f）—— "
                          "obs[3] 的编码基准" % (source, _hn, spec.HEIGHT_NOMINAL))
                    try:
                        spec.apply_contract(height_nominal=_hn)
                    except ValueError as e:
                        print("⚠️  %s" % e)
                else:
                    print("→ %s的 height_nominal = %.3f m，与当前一致" % (source, _hn))

    if pol_contract and not args.snapshot:
        _apply_contract(pol_contract.get("kp"), pol_contract.get("kd"),
                        pol_contract.get("action_scale"), "策略自带契约",
                        default_dof_pos=pol_contract.get("default_dof_pos"),
                        height_nominal=pol_contract.get("height_nominal"))
    # CLI 显式覆盖（最高优先级，仅 A/B 用）
    if args.kp is not None or args.kd is not None or args.action_scale is not None:
        print("⚠️  命令行显式覆盖 PD/动作缩放 —— 会与训练配置不一致，只在 A/B 时用")
        _apply_contract(args.kp, args.kd, args.action_scale, "命令行")

    if args.snapshot and snap_contract:
        # 快照优先于策略自带契约（它是同一份信息的原件）
        _apply_contract(snap_contract.get("kp"), snap_contract.get("kd"),
                        snap_contract.get("action_scale"), "训练快照")

    if args.imu_degrees and args.imu_radians:
        raise SystemExit("--imu-degrees 与 --imu-radians 不能同时给")

    # --cmd 解析
    fixed = (0.0, 0.0, 0.0)
    if args.cmd:
        try:
            parts = [float(x) for x in args.cmd.replace("，", ",").split(",")]
            if len(parts) != 3:
                raise ValueError
            fixed = tuple(parts)
        except ValueError:
            raise SystemExit("--cmd 格式应为 VX,VY,WZ，例如 --cmd 0.3,0,0")
    args.cmd = fixed

    # --stub-tilt 解析
    if args.stub_tilt:
        try:
            args.stub_tilt = tuple(float(x) for x in args.stub_tilt.split(","))
            if len(args.stub_tilt) != 2:
                raise ValueError
        except ValueError:
            raise SystemExit("--stub-tilt 格式应为 ROLL,PITCH，例如 --stub-tilt 5,0")

    if args.set_zero:
        return do_set_zero(cfg, args)

    app = RobotApp(cfg, args)

    # Ctrl-C / SIGTERM → 走正常收尾（刹车 + 存日志）
    def _sig(signum, frame):
        app.quit = True
        app.say("收到信号 %d，准备收尾" % signum)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    if args.check:
        return app.run_check()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
