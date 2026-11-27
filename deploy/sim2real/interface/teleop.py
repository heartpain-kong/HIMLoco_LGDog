#!/usr/bin/env python3
"""手柄输入（PC 侧遥操作）—— 手柄直接插上位机

拓扑（2026-09）：人手输入 = **USB 手柄**，直接接上位机（原来的遥控接收机 → 下位机
UART5 那套已整体删除）。所以上位机同时握着三种输入：
    手柄（人） / 策略（RL） / 以及两者的仲裁

★ 不引入新依赖：直接读 Linux 输入设备 + ioctl，只用标准库。
   evdev（/dev/input/event*）为主 —— 所有现代内核都有；
   joydev（/dev/input/js*）为兜底 —— 老接口，部分发行版已移除。

★ 手柄的轴号/按键号**因手柄型号而异**（Xbox / PS / 北通 / 杂牌都不一样），
   所以不要靠猜。先跑 `tools/gamepad_probe.py` 把轴和按键实测出来，
   再填 config.yaml 的 teleop 段。

═══════════════════════════════════════════════════════════════════════════
安全设计：**死人开关（dead-man switch）**
═══════════════════════════════════════════════════════════════════════════
    默认（不按任何安全键） = 保持：只站立，不放行策略
    按住「运行键」         = 运行：放行 RL 策略
    按一下「急停键」       = 急停：闩锁，必须重新按住运行键才解除

为什么用"按住才跑"而不是"按一下切换"：四足一旦失控就是真摔。死人开关让
"人松手"这个最自然的动作直接等价于停机，不需要人在慌张时还记得按哪个键。
"""

import fcntl
import os
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

# ---- Linux input 常量 ----
EV_SYN, EV_KEY, EV_ABS = 0x00, 0x01, 0x03

ABS_CODES = {
    "ABS_X": 0x00, "ABS_Y": 0x01, "ABS_Z": 0x02,
    "ABS_RX": 0x03, "ABS_RY": 0x04, "ABS_RZ": 0x05,
    "ABS_THROTTLE": 0x06, "ABS_RUDDER": 0x07,
    "ABS_HAT0X": 0x10, "ABS_HAT0Y": 0x11,
}
ABS_NAMES = {v: k for k, v in ABS_CODES.items()}

BTN_CODES = {
    "BTN_SOUTH": 0x130, "BTN_A": 0x130, "BTN_EAST": 0x131, "BTN_B": 0x131,
    "BTN_C": 0x132, "BTN_NORTH": 0x133, "BTN_X": 0x133, "BTN_WEST": 0x134,
    "BTN_Y": 0x134, "BTN_Z": 0x135,
    "BTN_TL": 0x136, "BTN_TR": 0x137, "BTN_TL2": 0x138, "BTN_TR2": 0x139,
    "BTN_SELECT": 0x13A, "BTN_START": 0x13B, "BTN_MODE": 0x13C,
    "BTN_THUMBL": 0x13D, "BTN_THUMBR": 0x13E,
}
BTN_NAMES = {}
for k, v in BTN_CODES.items():
    BTN_NAMES.setdefault(v, k)

# ioctl 号构造（Linux _IOC 宏）
_IOC_READ = 2


def _IOC(direction, typ, nr, size):
    return (direction << 30) | (size << 16) | (ord(typ) << 8) | nr


def _EVIOCGNAME(length):
    return _IOC(_IOC_READ, "E", 0x06, length)


def _EVIOCGBIT(ev, length):
    return _IOC(_IOC_READ, "E", 0x20 + ev, length)


def _EVIOCGABS(abs_code):
    return _IOC(_IOC_READ, "E", 0x40 + abs_code, 6 * 4)   # struct input_absinfo = 6×s32


@dataclass
class GamepadState:
    axes: Dict[int, float] = field(default_factory=dict)     # 轴号 -> 归一化值
    buttons: Dict[int, bool] = field(default_factory=dict)   # 按键号 -> 按下
    # ★ 用"最后一次收到事件"计时而不是"最后一次读数"：
    #   手柄静止时不发事件，但那不代表失联。所以失联判定要另用看门狗 ——
    #   见 GamepadReader.watchdog_ms（靠 EV_SYN 心跳）。
    last_event_t: float = 0.0
    last_syn_t: float = 0.0
    events: int = 0

    def age_ms(self) -> float:
        return (time.monotonic() - self.last_event_t) * 1000.0 if self.last_event_t else 1e9

    def syn_age_ms(self) -> float:
        return (time.monotonic() - self.last_syn_t) * 1000.0 if self.last_syn_t else 1e9


# ============================================================================
# 后端 1：evdev（/dev/input/event*）
# ============================================================================
class EvdevBackend:
    """读 /dev/input/event*。struct input_event 在 64 位 Linux 上是 24 字节：
           struct timeval { long tv_sec; long tv_usec; }  = 16
           __u16 type; __u16 code; __s32 value;           =  8
    （32 位系统上是 16 字节，下面按平台自适应）
    """

    EVENT_FMT = ("llHHi" if struct.calcsize("l") == 8 else "llHHi")
    EVENT_SIZE = struct.calcsize(EVENT_FMT)

    def __init__(self, path: str):
        self.path = path
        self.fd = None
        self.name = ""
        self.axis_info = {}      # code -> (min, max, center)
        self.button_codes = []

    # ------------------------------------------------------------------ 发现
    @staticmethod
    def list_devices() -> List[Tuple[str, str, bool]]:
        """返回 [(路径, 名字, 是否像手柄)]。"""
        out = []
        for i in range(64):
            p = "/dev/input/event%d" % i
            if not os.path.exists(p):
                continue
            try:
                fd = os.open(p, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                continue
            try:
                name = EvdevBackend._name(fd)
                is_pad = EvdevBackend._looks_like_gamepad(fd)
            finally:
                os.close(fd)
            out.append((p, name, is_pad))
        return out

    @staticmethod
    def _name(fd) -> str:
        buf = bytearray(256)
        try:
            fcntl.ioctl(fd, _EVIOCGNAME(len(buf)), buf, True)
        except OSError:
            return ""
        return buf.split(b"\x00")[0].decode("utf-8", "replace")

    @staticmethod
    def _looks_like_gamepad(fd) -> bool:
        """有手柄类按键（BTN_GAMEPAD 0x130..0x13F）或有 ABS_X/ABS_Y 就当作手柄。"""
        try:
            buf = bytearray(8)
            fcntl.ioctl(fd, _EVIOCGBIT(EV_KEY, len(buf)), buf, True)
            for code in range(0x130, 0x140):
                if buf[code // 8] & (1 << (code % 8)):
                    return True
            buf2 = bytearray(8)
            fcntl.ioctl(fd, _EVIOCGBIT(EV_ABS, len(buf2)), buf2, True)
            return bool(buf2[0] & 0x03)      # ABS_X | ABS_Y
        except OSError:
            return False

    # ------------------------------------------------------------------ 打开
    def open(self):
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        self.name = self._name(self.fd)
        # 轴量程
        for code, nm in ABS_NAMES.items():
            info = bytearray(6 * 4)
            try:
                fcntl.ioctl(self.fd, _EVIOCGABS(code), info, True)
            except OSError:
                continue
            _val, mn, mx, _fz, _fl, _res = struct.unpack("6i", bytes(info))
            if mx > mn:
                self.axis_info[code] = (mn, mx, (mn + mx) / 2.0)
        # 按键
        buf = bytearray(16)
        try:
            fcntl.ioctl(self.fd, _EVIOCGBIT(EV_KEY, len(buf)), buf, True)
            for code in range(0x120, 0x150):
                if buf[code // 8] & (1 << (code % 8)):
                    self.button_codes.append(code)
        except OSError:
            pass
        return self

    def close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            finally:
                self.fd = None

    # ------------------------------------------------------------------ 读
    def read(self, state: GamepadState) -> int:
        """把可用事件读干净，更新 state。返回处理的事件数。"""
        n = 0
        while True:
            try:
                data = os.read(self.fd, self.EVENT_SIZE * 64)
            except BlockingIOError:
                break
            except OSError:
                break
            if not data:
                break
            for off in range(0, len(data) - self.EVENT_SIZE + 1, self.EVENT_SIZE):
                _s, _us, typ, code, val = struct.unpack(
                    self.EVENT_FMT, data[off:off + self.EVENT_SIZE])
                n += 1
                now = time.monotonic()
                if typ == EV_SYN:
                    state.last_syn_t = now
                elif typ == EV_ABS:
                    info = self.axis_info.get(code)
                    if info:
                        mn, mx, ctr = info
                        # 居中轴用 (v-ctr)/半程；单向轴（扳机，mn=0）用 v/mx
                        if mn < -1 and mx > 1:
                            state.axes[code] = float(np.clip((val - ctr) / max(ctr - mn, 1), -1, 1))
                        else:
                            state.axes[code] = float(np.clip(val / max(mx, 1), 0, 1))
                    else:
                        state.axes[code] = float(val)
                    state.last_event_t = now
                elif typ == EV_KEY:
                    state.buttons[code] = bool(val)
                    state.last_event_t = now
            if len(data) < self.EVENT_SIZE * 64:
                break
        state.events += n
        return n


# ============================================================================
# 后端 2：joydev（/dev/input/js*）—— 老接口，部分内核已移除
# ============================================================================
class JoydevBackend:
    """struct js_event { __u32 time; __s16 value; __u8 type; __u8 number; } = 8 字节。"""

    FMT = "IhBB"
    SIZE = struct.calcsize(FMT)
    JS_EVENT_BUTTON, JS_EVENT_AXIS, JS_EVENT_INIT = 0x01, 0x02, 0x80

    def __init__(self, path: str):
        self.path = path
        self.fd = None
        self.name = os.path.basename(path)

    @staticmethod
    def list_devices() -> List[str]:
        return ["/dev/input/js%d" % i for i in range(8)
                if os.path.exists("/dev/input/js%d" % i)]

    def open(self):
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        return self

    def close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            finally:
                self.fd = None

    def read(self, state: GamepadState) -> int:
        n = 0
        while True:
            try:
                data = os.read(self.fd, self.SIZE * 64)
            except (BlockingIOError, OSError):
                break
            if not data:
                break
            for off in range(0, len(data) - self.SIZE + 1, self.SIZE):
                _t, val, typ, num = struct.unpack(self.FMT, data[off:off + self.SIZE])
                n += 1
                now = time.monotonic()
                base = typ & ~self.JS_EVENT_INIT
                if base == self.JS_EVENT_AXIS:
                    state.axes[num] = float(np.clip(val / 32767.0, -1, 1))
                    state.last_event_t = now
                elif base == self.JS_EVENT_BUTTON:
                    state.buttons[num] = bool(val)
                    state.last_event_t = now
            state.last_syn_t = time.monotonic()   # jsdev 没有 SYN，用读到的时刻
            if len(data) < self.SIZE * 64:
                break
        state.events += n
        return n


# ============================================================================
# 后端 3：键盘（手柄不在时的兜底）
# ============================================================================
class KeyboardBackend:
    """用终端按键操控，作为手柄缺失时的退化方案。

    ★★ 为什么用"按键自动重复"实现死人开关
        终端只给按键事件，**不给松开事件** —— 没法直接问"这个键还按着没"。
        但终端在按住时会按 autorepeat 速率反复发同一个字符，于是
        "最近 hold_timeout_ms 内见过这个键" 就等价于"键还按着"。
        ⚠️ 前提是系统开着键盘自动重复（Linux 终端默认开）。
           若被关掉了（xset -r），按住只会来一次事件，死人开关会失效 ——
           那种情况下请把 config 的 keyboard.run_is_toggle 设为 true，
           改成"按一下切换运行/保持"。

    ★ 终端状态必须恢复：进入 cbreak 模式后若异常退出，终端会留在无回显状态，
      用户会以为系统坏了。这里 close() + atexit 双重恢复。

    支持的键（可在 config 改）：
        W/S 前后   A/D 左右   Q/E 转向
        空格 = 运行（按住）   X = 急停    Esc/Ctrl-C = 退出
        也支持方向键。
    """

    # 终端转义序列 → 虚拟键名
    ESC_SEQ = {
        b"\x1b[A": "up", b"\x1b[B": "down", b"\x1b[C": "right", b"\x1b[D": "left",
        b"\x1b[1;2A": "shift_up", b"\x1b[1;2B": "shift_down",
    }

    def __init__(self, cfg: dict, hold_timeout_ms: float = 400.0):
        self.cfg = cfg or {}
        self.hold_timeout = hold_timeout_ms / 1000.0
        self.fd = None
        self.saved = None
        self.name = "keyboard"
        self._last_seen = {}          # 虚拟键名 -> 最后见到的时间
        self._warned_repeat = False
        self._atexit_registered = False
        # 按键 → 虚拟键名
        self._bind = {}
        for name, keys in self.cfg.items():
            if not name.startswith("key_"):
                continue
            if isinstance(keys, str):
                keys = [keys]
            for k in keys:
                self._bind[str(k).lower()] = name[4:]

    # ------------------------------------------------------------------
    def open(self):
        import termios, tty
        self.fd = sys.stdin.fileno()
        if not os.isatty(self.fd):
            raise RuntimeError("stdin 不是终端（被重定向了？）—— 键盘兜底需要交互终端")
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)         # 用 cbreak 而不是 raw：保留 Ctrl-C 能中断
        if not self._atexit_registered:
            import atexit
            atexit.register(self.close)
            self._atexit_registered = True
        # 设成非阻塞
        import fcntl as _fcntl
        fl = _fcntl.fcntl(self.fd, _fcntl.F_GETFL)
        _fcntl.fcntl(self.fd, _fcntl.F_SETFL, fl | os.O_NONBLOCK)
        return self

    def close(self):
        if self.saved is not None and self.fd is not None:
            try:
                import termios
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            except Exception:
                pass
            self.saved = None

    # ------------------------------------------------------------------
    def _key_name(self, seq: bytes):
        if seq in self.ESC_SEQ:
            return self.ESC_SEQ[seq]
        try:
            ch = seq.decode("utf-8", "replace").lower()
        except Exception:
            return None
        if ch == "\x1b":
            return "escape"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        return ch

    def read(self, state: GamepadState) -> int:
        """把终端里可读的字节读干净，更新 state 的轴/按键（用 config 的号）。"""
        n = 0
        now = time.monotonic()
        try:
            data = os.read(self.fd, 64)
        except (BlockingIOError, OSError):
            data = b""
        i = 0
        while i < len(data):
            # 先试转义序列（最长 6 字节）
            name = None
            for ln in (6, 5, 4, 3):
                if data[i:i + ln] in self.ESC_SEQ:
                    name = self.ESC_SEQ[data[i:i + ln]]
                    i += ln
                    break
            if name is None:
                name = self._key_name(data[i:i + 1])
                i += 1
            if name:
                self._last_seen[name] = now
                state.last_event_t = now
                n += 1
        state.events += n
        state.last_syn_t = now          # 键盘没有内核心跳，用读取时刻当心跳
        return n

    def axes_from_keys(self, ax_vx, ax_vy, ax_wz, btn_run, btn_estop, now=None):
        """把当前按键状态写成 GamepadState 需要的三个轴 + 两个键。

        独立成方法是因为它需要 config 里的轴号/键号（那些属于 TeleopCommander），
        由 GamepadReader 在两者都建好后调用。
        """
        now = time.monotonic() if now is None else now

        def held(kind):
            t = self._last_seen.get(kind)
            return (t is not None) and ((now - t) < self.hold_timeout)

        vx = (1.0 if held("w") else 0.0) - (1.0 if held("s") else 0.0)
        vy = (1.0 if held("a") else 0.0) - (1.0 if held("d") else 0.0)
        wz = (1.0 if held("q") else 0.0) - (1.0 if held("e") else 0.0)
        vx += (1.0 if held("up") else 0.0) - (1.0 if held("down") else 0.0)
        vy += (1.0 if held("left") else 0.0) - (1.0 if held("right") else 0.0)
        return {ax_vx: float(np.clip(vx, -1, 1)),
                ax_vy: float(np.clip(vy, -1, 1)),
                ax_wz: float(np.clip(wz, -1, 1))}, {
                btn_run: bool(held("space")),
                btn_estop: bool(held("x"))}, held("escape")


# ============================================================================
# 读取线程
# ============================================================================
class GamepadReader:
    def __init__(self, device: str = "auto", backend: str = "auto",
                 watchdog_ms: float = 500.0, verbose: bool = False,
                 keyboard_cfg: dict = None, keyboard_enabled: bool = True,
                 keyboard_hold_ms: float = 400.0):
        self.device = device
        self.backend_name = backend
        self.watchdog_ms = float(watchdog_ms)
        self.verbose = verbose
        self.kb_cfg = keyboard_cfg or {}
        self.kb_enabled = bool(keyboard_enabled)
        self.kb_hold_ms = float(keyboard_hold_ms)
        self.is_keyboard = False
        self._kb_map = None
        self.backend = None
        self.state = GamepadState()
        self._thread = None
        self._stop = threading.Event()
        self._fails = 0

    # ------------------------------------------------------------------
    def _pick_backend(self):
        if self.backend_name == "stub":
            return None
        if self.backend_name == "keyboard":
            return KeyboardBackend(self.kb_cfg, self.kb_hold_ms)
        if self.device != "auto":
            if "js" in os.path.basename(self.device):
                return JoydevBackend(self.device)
            return EvdevBackend(self.device)
        # 自动：优先 evdev，按名字猜手柄
        if self.backend_name in ("auto", "evdev"):
            cands = EvdevBackend.list_devices()
            pads = [(p, n) for p, n, is_pad in cands if is_pad]
            if pads:
                if self.verbose:
                    print("[PAD] 自动选中 evdev 设备: %s (%s)" % pads[0])
                return EvdevBackend(pads[0][0])
        if self.backend_name in ("auto", "joydev"):
            js = JoydevBackend.list_devices()
            if js:
                if self.verbose:
                    print("[PAD] 回退到 joydev 设备: %s" % js[0])
                return JoydevBackend(js[0])
        # ★ 手柄不在 → 自动退化为键盘（--no-keyboard-fallback 可关掉）
        if self.kb_enabled:
            try:
                kb = KeyboardBackend(self.kb_cfg, self.kb_hold_ms).open()
                print("[PAD] ⚠️ 没找到手柄 → 自动退化为【键盘】控制")
                print("      W/S 前后  A/D 左右  Q/E 转向  空格=按住运行  X=急停  Esc=退出")
                print("      （终端只给按键事件不给松开事件，所以用'按键自动重复'判按住；"
                      "若关掉了系统自动重复，请设 keyboard.run_is_toggle）")
                return kb
            except Exception as e:
                if self.verbose:
                    print("[PAD] 键盘兜底不可用: %s" % e)
        return None

    def start(self):
        self.backend = self._pick_backend()
        if self.backend is None:
            # stub：给一个可用的中性状态，便于干跑
            self.state.axes = {0: 0.0, 1: 0.0, 3: 0.0}
            self.state.last_event_t = time.monotonic()
            self.state.last_syn_t = time.monotonic()
            if self.verbose:
                print("[PAD] 未找到手柄，使用 stub（干跑用）")
            return
        self.backend.open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.backend is not None:
            self.backend.close()

    def _loop(self):
        kb_map = None
        while not self._stop.is_set():
            try:
                if isinstance(self.backend, KeyboardBackend):
                    self.backend.read(self.state)
                    if kb_map is None:
                        kb_map = self._kb_map
                    if kb_map:
                        ax, bt, _quit = self.backend.axes_from_keys(*kb_map)
                        self.state.axes.update(ax)
                        self.state.buttons.update(bt)
                else:
                    self.backend.read(self.state)
            except Exception:
                self._fails += 1
                time.sleep(0.01)
            time.sleep(0.002)

    def bind_keyboard_map(self, ax_vx, ax_vy, ax_wz, btn_run, btn_estop):
        """告诉键盘后端该用哪些轴号/键号（与 TeleopCommander 保持一致）。"""
        self._kb_map = (ax_vx, ax_vy, ax_wz, btn_run, btn_estop)

    def get(self) -> GamepadState:
        return self.state

    def is_alive(self) -> bool:
        """★ 失联判定用"最后一次 EV_SYN"，不是"最后一次事件" ——
        手柄静止时不发轴/键事件，但内核仍会周期性送 SYN 心跳。
        ⚠️ 键盘后端没有内核心跳，只能靠"控制环每周期都会来读一次"刷新；
           所以键盘模式下这里恒为 True（失联保护交给 PC 与板子的心跳看门狗）。"""
        if self.backend is None or isinstance(self.backend, KeyboardBackend):
            return True
        return self.state.syn_age_ms() < self.watchdog_ms


# ============================================================================
# 映射到手柄 → 指令
# ============================================================================
@dataclass
class TeleopCommand:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    run_policy: bool = False
    estop: bool = False
    pad_lost: bool = False

    def is_motion(self) -> bool:
        return abs(self.vx) + abs(self.vy) + abs(self.wz) > 1e-6


class TeleopCommander:
    """手柄 → 速度指令 + 安全状态。轴号/按键号由 config 指定（用 probe 实测）。"""

    def __init__(self, cfg: dict):
        ax = cfg.get("axes", {}) or {}
        self.ax_vx = int(ax.get("vx", 1))
        self.ax_vy = int(ax.get("vy", 0))
        self.ax_wz = int(ax.get("wz", 3))
        inv = cfg.get("invert", {}) or {}
        self.inv = (bool(inv.get("vx", True)),      # 默认推杆向前通常给出负值
                    bool(inv.get("vy", False)),
                    bool(inv.get("wz", False)))
        bt = cfg.get("buttons", {}) or {}
        self.btn_run = int(bt.get("run", 0))        # 按住才跑（死人开关）
        self.btn_estop = int(bt.get("estop", 1))    # 按一下急停（闩锁）
        self.btn_aux = int(bt.get("aux", 2))

        self.deadband = float(cfg.get("deadband", 0.08))
        self.max_vx = float(cfg.get("max_vx", 0.6))
        self.max_vy = float(cfg.get("max_vy", 0.2))
        self.max_wz = float(cfg.get("max_yaw", 0.4))
        self.alpha = float(cfg.get("lpf_alpha", 0.25))
        self.stop_on_pad_lost = bool(cfg.get("stop_on_pad_lost", True))

        self._vx = self._vy = self._wz = 0.0
        self.estop_latched = False
        self._prev_estop_btn = False

    def _axis(self, st: GamepadState, idx: int) -> float:
        v = float(st.axes.get(idx, 0.0))
        if abs(v) < self.deadband:
            return 0.0
        # 死区外重新拉伸到 [0,1]，避免死区边缘跳变
        s = np.sign(v)
        return float(s * min(1.0, (abs(v) - self.deadband) / max(1e-6, 1.0 - self.deadband)))

    def clear_estop(self):
        self.estop_latched = False

    def update(self, st: GamepadState, alive: bool = True) -> TeleopCommand:
        cmd = TeleopCommand(pad_lost=not alive)

        # ---- 急停按键：上升沿闩锁，再按一下解除 ----
        pressed = bool(st.buttons.get(self.btn_estop, False))
        if pressed and not self._prev_estop_btn:
            self.estop_latched = not self.estop_latched
        self._prev_estop_btn = pressed
        cmd.estop = self.estop_latched

        # ---- 死人开关：必须【按住】运行键 ----
        run_held = bool(st.buttons.get(self.btn_run, False))
        cmd.run_policy = run_held and (not self.estop_latched) and alive

        # ---- 摇杆 ----
        t = [self._axis(st, self.ax_vx) * self.max_vx * (-1 if self.inv[0] else 1),
             self._axis(st, self.ax_vy) * self.max_vy * (-1 if self.inv[1] else 1),
             self._axis(st, self.ax_wz) * self.max_wz * (-1 if self.inv[2] else 1)]

        for i, name in enumerate(("_vx", "_vy", "_wz")):
            setattr(self, name, self.alpha * t[i] + (1.0 - self.alpha) * getattr(self, name))

        if cmd.estop or (cmd.pad_lost and self.stop_on_pad_lost) or not run_held:
            # 不按运行键时也把速度清零（保持态不该残留上次的摇杆量）
            self._vx = self._vy = self._wz = 0.0

        cmd.vx, cmd.vy, cmd.wz = self._vx, self._vy, self._wz
        return cmd

    def status_line(self, st: GamepadState, cmd: TeleopCommand) -> str:
        return ("PAD ax=%s btn=%s -> vx=%+.2f vy=%+.2f wz=%+.2f %s%s%s"
                % ({k: round(v, 2) for k, v in sorted(st.axes.items())},
                   sorted(k for k, v in st.buttons.items() if v),
                   cmd.vx, cmd.vy, cmd.wz,
                   "RUN" if cmd.run_policy else "HOLD",
                   " ESTOP" if cmd.estop else "",
                   " LOST" if cmd.pad_lost else ""))


# ============================================================================
def _selftest():
    print("手柄后端自测（无手柄，用 stub + 合成状态）")
    print("  可见 evdev 设备:")
    for p, n, is_pad in EvdevBackend.list_devices():
        print("    %-20s %-32s %s" % (p, n or "(无名)", "★像手柄" if is_pad else ""))

    print()
    print("映射自测")
    cfg = {"axes": {"vx": 1, "vy": 0, "wz": 3}, "invert": {"vx": True},
           "buttons": {"run": 0, "estop": 1}, "deadband": 0.08,
           "max_vx": 0.6, "max_vy": 0.2, "max_yaw": 0.4, "lpf_alpha": 1.0}
    cmdr = TeleopCommander(cfg)

    st = GamepadState(axes={0: 0.0, 1: 0.0, 3: 0.0}, buttons={0: False, 1: False})
    c = cmdr.update(st)
    assert not c.run_policy and not c.estop and not c.is_motion()
    print("  不按键 -> HOLD、静止 ✓")

    # 按住运行键 + 推杆前（轴 1 推到 -1，因 invert_vx=True 应得 +0.6）
    st.buttons[0] = True
    st.axes[1] = -1.0
    c = cmdr.update(st)
    print("  按住运行 + 轴1=-1.0 (invert_vx) -> vx=%+.2f run=%s" % (c.vx, c.run_policy))
    assert c.run_policy and abs(c.vx - 0.6) < 1e-6

    # 松开运行键 → 立刻停（死人开关）
    st.buttons[0] = False
    c = cmdr.update(st)
    print("  松开运行键 -> run=%s vx=%+.2f（死人开关生效）" % (c.run_policy, c.vx))
    assert not c.run_policy and c.vx == 0.0

    # 急停：上升沿闩锁、再按解除
    st.buttons[0] = True
    cmdr.update(st)
    st.buttons[1] = True
    c = cmdr.update(st)
    print("  按急停 -> estop=%s run=%s（闩锁）" % (c.estop, c.run_policy))
    assert c.estop and not c.run_policy
    st.buttons[1] = False
    cmdr.update(st)
    st.buttons[1] = True
    c = cmdr.update(st)
    print("  再按一次急停 -> estop=%s（解除）" % c.estop)
    assert not c.estop

    # 死区
    st.buttons[1] = False; cmdr.update(st)
    st.axes[1] = -0.05      # 小于死区 0.08
    c = cmdr.update(st)
    print("  轴 1 = -0.05（死区内）-> vx=%+.2f ✓" % c.vx)
    assert c.vx == 0.0

    # 失联
    st.axes[1] = -1.0
    c = cmdr.update(st, alive=False)
    print("  失联(alive=False) -> run=%s vx=%+.2f lost=%s" % (c.run_policy, c.vx, c.pad_lost))
    assert not c.run_policy and c.vx == 0.0 and c.pad_lost

    # 读取器 stub
    print()
    r = GamepadReader(backend="stub")
    r.start()
    print("  GamepadReader(stub) alive=%s axes=%s" % (r.is_alive(), r.get().axes))
    r.stop()

    # ---- 键盘后端（不碰真终端，直接驱动它的内部状态）----
    print()
    print("键盘兜底自测")
    kb = KeyboardBackend({}, hold_timeout_ms=400.0)
    now = time.monotonic()
    # 模拟"刚刚收到空格与 W 的自动重复"
    kb._last_seen = {"space": now, "w": now}
    ax, bt, quit_ = kb.axes_from_keys(1, 0, 3, 0, 1, now=now)
    print("  按住 W+空格 -> 轴=%s 键=%s" % ({k: round(v, 2) for k, v in ax.items()}, bt))
    assert ax[1] == 1.0 and bt[0] is True

    # 自动重复停了 → 等价于松开（死人开关生效）
    ax2, bt2, _ = kb.axes_from_keys(1, 0, 3, 0, 1, now=now + 0.5)
    print("  停止重复 500ms(>400ms) -> 轴=%s 键=%s（视为松开）"
          % ({k: round(v, 2) for k, v in ax2.items()}, bt2))
    assert ax2[1] == 0.0 and bt2[0] is False

    # 方向键与 S/D/Q/E
    kb._last_seen = {"down": now, "left": now, "q": now, "x": now, "escape": now}
    ax3, bt3, q3 = kb.axes_from_keys(1, 0, 3, 0, 1, now=now)
    print("  下键/左键/Q/X/Esc -> vx=%+.0f vy=%+.0f wz=%+.0f estop=%s quit=%s"
          % (ax3[1], ax3[0], ax3[3], bt3[1], q3))
    assert ax3[1] == -1.0 and ax3[0] == 1.0 and ax3[3] == 1.0 and bt3[1] and q3

    print("  （说明：终端只给按键事件不给松开事件，所以用'最近 hold_timeout 内")
    print("    见过该键'来等价于'还按着'；前提是系统开着键盘自动重复）")

    print()
    print("✅ gamepad/teleop 自测通过")
    print("   ★ 真实轴号/按键号请用 tools/gamepad_probe.py 实测后再填 config.yaml")


if __name__ == "__main__":
    _selftest()
