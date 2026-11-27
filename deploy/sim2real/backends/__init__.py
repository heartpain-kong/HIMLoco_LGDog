#!/usr/bin/env python3
"""后端工厂 —— 把「真机」和「干跑」两套实现统一成同一个 RobotIO 接口

    make_robot_io(cfg, args, mapping) -> HostLinkRobotIO | StubRobotIO

★ 关于 --backend auto 的选择规则（安全优先）：
    auto 只会选【真机】。找不到串口就报错退出，并提示你显式加 --dry-run。
    绝不"找不到硬件就悄悄用假机器人跑"—— 那会让人以为策略在真机上跑通了。

找串口不写死 /dev/ttyACM0：按 /sys 里的 USB idVendor 挑（STM32 虚拟串口是 0x0483），
多设备时把候选都列出来让人选，避免插了两个设备时挑错。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

#: 已知会以 USB CDC 虚拟串口出现的下位机/适配器（vid → 说明）
KNOWN_VIDS = {
    0x0483: "STMicroelectronics（STM32H7 下位机的 USB CDC 就是它）",
    0x1A86: "QinHeng（CH340，常见 USB-CAN 适配器）",
    0x2E88: "DAMIAO / 达妙（DM-MC02）",
    0x34B7: "LingZu / 灵足",
}

#: 干跑故障注入：config 键 → main.py 的 argparse 属性名
FAULT_ARGS = {
    "nan_after_s":          "stub_fault_nan",
    "imu_freeze_after_s":   "stub_fault_imu_freeze",
    "motor_missing_after_s": "stub_fault_motor_missing",
    "drop_link_after_s":    "stub_fault_drop_link",
    "backend_fault_after_s": "stub_fault_backend",
}


# ============================================================================
def _usb_vid_pid(tty_name: str):
    """从 sysfs 读 ttyACM0 对应的 USB vid/pid（读不到返回 (None, None)）。"""
    base = "/sys/class/tty/%s/device" % tty_name
    for _ in range(6):                       # 往上找几层，直到出现 idVendor
        vid_p = os.path.join(base, "idVendor")
        if os.path.isfile(vid_p):
            try:
                with open(vid_p) as f:
                    vid = int(f.read().strip(), 16)
                pid = None
                pid_p = os.path.join(base, "idProduct")
                if os.path.isfile(pid_p):
                    with open(pid_p) as f:
                        pid = int(f.read().strip(), 16)
                return vid, pid
            except Exception:
                return None, None
        parent = os.path.dirname(base)
        if parent == base:
            break
        base = parent
    return None, None


def find_serial_ports():
    """扫出候选串口。返回 [{'path','vid','pid','desc','likely'}]，likely 的排前面。"""
    import glob
    out = []
    for p in sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*")) \
            + sorted(glob.glob("/dev/serial/by-id/*")):
        real = os.path.realpath(p)
        name = os.path.basename(real)
        vid, pid = _usb_vid_pid(name)
        desc = KNOWN_VIDS.get(vid, "")
        if not desc and "by-id" in p:
            desc = os.path.basename(p)
        out.append({"path": p, "real": real, "vid": vid, "pid": pid,
                    "desc": desc, "likely": vid in KNOWN_VIDS})
    # 去重：by-id 与 ttyACM 指向同一设备时保留 by-id（名字稳定，重插也不变）
    seen = {}
    for d in out:
        key = (d["real"],)
        if key not in seen or "by-id" in d["path"]:
            seen[key] = d
    res = list(seen.values())
    res.sort(key=lambda d: (not d["likely"], "by-id" not in d["path"], d["path"]))
    return res


def _print_ports(ports):
    if not ports:
        print("        （一个都没有。下位机插好了吗？`lsusb` 能看到吗？）")
        return
    for d in ports:
        extra = ""
        if d["vid"] is not None:
            extra = "  [%04x:%04x]%s" % (d["vid"], d["pid"] or 0,
                                         ("  ★ " + d["desc"]) if d["desc"] else "")
        print("        %-40s%s" % (d["path"], extra))


def pick_serial_port(prefer: str = None, verbose: bool = True):
    """挑一个串口。prefer 给了就优先用它（但会检查存在性并给友好提示）。"""
    ports = find_serial_ports()
    if prefer:
        if os.path.exists(prefer):
            return prefer
        # 允许写 ttyACM0 这种短名
        cand = [p["path"] for p in ports if os.path.basename(p["path"]) == prefer]
        if cand:
            return cand[0]
        if verbose:
            print("[PORT] 指定的 %s 不存在。当前候选：" % prefer)
            _print_ports(ports)
        return None
    if not ports:
        return None
    if verbose and len(ports) > 1:
        print("[PORT] 发现多个串口，选第一个（要换用 --port 指定）：")
        _print_ports(ports)
    return ports[0]["path"]


# ============================================================================
def make_robot_io(cfg: dict, args, mapping=None):
    """按配置/命令行造后端。返回对象满足 RobotIO 契约（见 stub_backend.py 头注释）。"""
    backend = (getattr(args, "backend", None) or cfg.get("backend") or "auto").lower()
    verbose = bool(cfg.get("debug", False)) or bool(getattr(args, "verbose", False))

    # ---------------------------------------------------------------- 干跑
    if backend in ("stub", "dry", "dryrun", "dry-run"):
        from backends.stub_backend import StubRobotIO
        faults = {}
        for key, attr in FAULT_ARGS.items():
            v = getattr(args, attr, None)
            if v:
                faults[key] = float(v)
        return StubRobotIO(faults=faults,
                           tilt_deg=tuple(getattr(args, "stub_tilt", None) or (0.0, 0.0)),
                           sag_rad=float(getattr(args, "stub_sag", 0.02) or 0.0),
                           verbose=verbose)

    # ---------------------------------------------------------------- 真机
    if backend in ("host_link", "host", "real", "auto"):
        prefer = getattr(args, "port", None) or cfg.get("serial_port")
        port = pick_serial_port(prefer, verbose=True)
        if port is None:
            raise SystemExit(
                "找不到下位机串口（也没有有效的 --port）。\n"
                "  · 想在没有硬件的机器上验证整条链路：加 --dry-run\n"
                "  · 想看看有什么串口：加 --list-devices\n"
                "  · 检查权限：读 /dev/ttyACM* 需要 dialout 组（install.sh 会加）")
        if mapping is None:
            from interface.motor_mapping import MotorMapping
            mapping = MotorMapping(cfg.get("hardware_map") or None) \
                if cfg.get("hardware_map") else MotorMapping()
        from backends.host_link_backend import HostLinkRobotIO
        return HostLinkRobotIO(port, mapping,
                               imu_cfg=cfg.get("imu", {}) or {},
                               baudrate=int(cfg.get("serial_baud", 921600)),
                               verbose=verbose)

    raise SystemExit("未知 backend: %r（可选 auto / host_link / stub）" % backend)


def list_devices():
    """给 --list-devices 用：串口 + 手柄一次列全。"""
    print("=" * 72)
    print("串口（下位机走 Type-C 虚拟串口）")
    print("=" * 72)
    _print_ports(find_serial_ports())
    print()
    print("=" * 72)
    print("手柄（/dev/input/event*，策略的人机输入）")
    print("=" * 72)
    try:
        from interface.teleop import EvdevBackend
        devs = EvdevBackend.list_devices()
        if not devs:
            print("  没有可见的 /dev/input/event*（查看权限：不在 input 组？）")
        for path, name, is_pad in devs:
            print("  %-24s %-32s %s" % (path, name or "(无名)", "★像手柄" if is_pad else ""))
    except Exception as e:
        print("  （枚举失败：%s）" % e)
    print()
    print("  标定轴号/按键号： python tools/gamepad_probe.py --suggest")
