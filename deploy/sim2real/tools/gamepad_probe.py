#!/usr/bin/env python3
"""手柄探针 —— 实测轴号/按键号，用来填 config.yaml 的 teleop 段

★ 为什么必须有这个工具：手柄的轴号和按键号**因型号而异**（Xbox / PS / 北通 /
  杂牌全不一样），同一个物理摇杆在不同手柄上可能是轴 0、1、3 或 4。靠猜必然错，
  而猜错的后果是"推前进结果机器人在原地转"。所以：先跑这个，看着屏幕动一遍，
  把号抄进 config。

用法：
    python tools/gamepad_probe.py                 # 列出设备 + 实时监视
    python tools/gamepad_probe.py --list          # 只列设备
    python tools/gamepad_probe.py --device /dev/input/event5
    python tools/gamepad_probe.py --suggest       # 监视若干秒后自动给出配置片段

★ 权限：读 /dev/input/event* 需要 root 或在 input 组。
     sudo usermod -aG input $USER      # 然后【注销重新登录】才生效
  本工具会先自检权限并给出提示，不会让你对着"没有设备"发呆。
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from interface.teleop import (EvdevBackend, JoydevBackend, GamepadReader,
                              GamepadState, ABS_NAMES, BTN_NAMES)


def check_permission():
    """返回 (是否可读, 提示文本)。"""
    paths = ["/dev/input/event%d" % i for i in range(8)]
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            fd = os.open(p, os.O_RDONLY | os.O_NONBLOCK)
            os.close(fd)
            return True, ""
        except PermissionError:
            user = os.environ.get("USER", "?")
            return False, (
                "★ 读不了 %s —— 权限不足。\n"
                "  /dev/input/event* 属于 root:input，当前用户不在 input 组。\n"
                "  修法（二选一，都要重新登录）：\n"
                "     sudo usermod -aG input %s\n"
                "     或加 udev 规则 /etc/udev/rules.d/99-gamepad.rules：\n"
                "       KERNEL==\"event*\", SUBSYSTEM==\"input\", TAG+=\"uaccess\"\n"
                "     然后 sudo udevadm control --reload && 重插手柄\n"
                "  （当前用户组：%s）" % (p, user, _groups()))
        except OSError:
            continue
    return False, "★ 没找到任何 /dev/input/event* 设备"


def _groups():
    try:
        import grp
        return ",".join(grp.getgrgid(g).gr_name for g in os.getgroups())
    except Exception:
        return "?"


def cmd_list():
    print("=" * 78)
    print("可用输入设备")
    print("=" * 78)
    ok, hint = check_permission()
    if not ok and hint:
        print(hint)
        print()
    devs = EvdevBackend.list_devices()
    if not devs:
        print("  （没有可读的 evdev 设备）")
    for p, n, is_pad in devs:
        print("  %-22s %-34s %s" % (p, n or "(无名)", "★ 像手柄" if is_pad else ""))
    js = JoydevBackend.list_devices()
    print()
    print("  joydev(老接口): %s" % (js if js else "无"))
    print()
    print("  /proc/bus/input/devices 里带 js 的条目：")
    try:
        txt = open("/proc/bus/input/devices").read()
        found = False
        for blk in txt.split("\n\n"):
            if "Handlers=" in blk and ("js" in blk.split("Handlers=")[1].split("\n")[0]):
                nm = [l for l in blk.split("\n") if l.startswith("N:")]
                print("   ", nm[0] if nm else blk[:60])
                found = True
        if not found:
            print("    （无 —— 手柄当前没插，或内核没加载 joydev）")
    except Exception:
        pass


def cmd_monitor(device, seconds, suggest):
    ok, hint = check_permission()
    if not ok:
        print(hint)
        if device == "auto":
            return 1
    r = GamepadReader(device=device, backend="auto", watchdog_ms=500, verbose=True)
    r.start()
    if r.backend is None:
        print("❌ 没有可用手柄。先插上手柄，再跑 --list 确认它出现在列表里。")
        return 1
    print("已打开: %s (%s)" % (getattr(r.backend, 'path', '?'),
                              getattr(r.backend, 'name', '')))
    print()
    print("把两个摇杆各推到底、把所有按键按一遍。按 Ctrl-C 结束。")
    print("★ 想看的是：哪个轴号跟着「左摇杆上下」变、哪个按键号是你想用的运行/急停键")
    print("-" * 78)

    st = r.get()
    t0 = time.time()
    seen_axes = {}
    seen_btn = set()
    last_print = 0.0
    try:
        while True:
            if seconds and (time.time() - t0) > seconds:
                break
            now = time.time()
            if now - last_print > 0.1:
                last_print = now
                axes = " ".join("%d=%+.2f" % (k, v) for k, v in sorted(st.axes.items()))
                btns = " ".join(str(k) for k in sorted(k for k, v in st.buttons.items() if v))
                print("\r  轴[%-46s] 按下[%-18s]" % (axes[:46], btns[:18]), end="")
            for k, v in st.axes.items():
                if abs(v) > 0.3:
                    seen_axes.setdefault(k, []).append(v)
            for k, v in st.buttons.items():
                if v:
                    seen_btn.add(k)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        r.stop()
    print()
    print("-" * 78)
    print("活动过的轴: %s" % {k: "min %+.2f max %+.2f" % (min(v), max(v))
                              for k, v in sorted(seen_axes.items())})
    print("按过的按键: %s" % sorted(seen_btn))
    if suggest:
        print()
        print("=" * 78)
        print("建议的 config.yaml teleop 段（请你按实际推的方向确认 invert）")
        print("=" * 78)
        ax = sorted(seen_axes)
        print("teleop:")
        print("  enabled: true")
        print("  device: \"auto\"")
        print("  axes:")
        print("    # 按你推的方向选：推左摇杆【上下】的那个号填 vx")
        print("    vx: %d" % (ax[1] if len(ax) > 1 else 1))
        print("    vy: %d" % (ax[0] if ax else 0))
        print("    wz: %d" % (ax[2] if len(ax) > 2 else 3))
        print("  buttons:")
        print("    run: %s     # ← 改成你实测出来的运行键（要按住才跑）"
              % (sorted(seen_btn)[0] if seen_btn else 0))
        print("    estop: %s   # ← 急停键" % (sorted(seen_btn)[1] if len(seen_btn) > 1 else 1))
    return 0


def main():
    ap = argparse.ArgumentParser(description="手柄探针：列设备 / 实测轴号按键号")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seconds", type=float, default=0, help="监视时长（0=直到 Ctrl-C）")
    ap.add_argument("--list", action="store_true", help="只列设备")
    ap.add_argument("--suggest", action="store_true", help="结束后给出 config 片段")
    args = ap.parse_args()
    if args.list:
        cmd_list()
        return 0
    return cmd_monitor(args.device, args.seconds, args.suggest)


if __name__ == "__main__":
    sys.exit(main())
