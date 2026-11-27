#!/usr/bin/env python3
"""Type-C 链路嗅探 —— 用来搞清下位机到底发了什么帧

背景：下位机（DM-MC02）在同一条 Type-C 链路上既跑电机的 CAN 透传，
又回传【欧拉角】。`vendored/drivers/usb_can_adapter.py` 只会解析
「0xAA 开头、0x55 结尾、共 16 字节」的 CAN 帧，其余字节会被它当垃圾跳过
—— 所以欧拉角帧的格式必须单独抓出来。

本工具的三种用法：

  1) 看总线上有哪些 CAN ID（确认 12 个电机都在）
       python tools/link_sniffer.py --port /dev/ttyUSB0 --seconds 3 --can

  2) 找欧拉角帧（重点）
       python tools/link_sniffer.py --port /dev/ttyUSB0 --seconds 5 --unknown
     把**不符合 CAN 帧结构**的字节流按 帧头/长度 聚类打印出来。
     然后慢慢转动/倾斜机器人，看哪一段数值跟着变 —— 那就是欧拉角。

  3) 原始字节倾泻（聚类也搞不定时的最后手段）
       python tools/link_sniffer.py --port /dev/ttyUSB0 --seconds 3 --raw

找到候选帧后，把 `frame_id / frame_len / euler_offset / euler_dtype /
euler_unit_is_degree / euler_convention` 填进 config.yaml 的 imu 段。
"""

import argparse
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "vendored"))

CAN_HDR = 0xAA      # 接收帧头（与 usb_can_adapter.RECV_HEADER 一致）
CAN_TAIL = 0x55
CAN_FRAME_LEN = 16  # 与 usb_can_adapter.RECV_FRAME_LEN 一致


def run_sniffer(port, baud, seconds, mode, can_baud, poll):
    import serial
    from drivers.motor_driver import RobStrideDriver

    ser = serial.Serial(port, baudrate=baud, timeout=0.01)
    print("已打开 %s @ %d，嗅探 %.1fs（模式: %s）" % (port, baud, seconds, mode))
    print("-" * 74)

    # 可选：先设 CAN 波特率 + 发被动轮询，把下位机的流量"叫醒"
    if poll:
        try:
            drv = RobStrideDriver(port)
            drv.connect()
            try:
                if can_baud:
                    drv.adapter.set_can_baudrate(0)
            finally:
                pass
            print("[已发送 CAN 波特率设置指令，开始被动轮询]")
        except Exception as e:
            print("⚠️ 轮询初始化失败（不影响继续嗅探）: %s" % e)
            ser = serial.Serial(port, baudrate=baud, timeout=0.01)

    buf = bytearray()
    t0 = time.time()
    total = 0
    can_ids = Counter()
    can_frames = 0
    other_bytes = 0

    # 未知帧聚类：用 (首字节, 该字节之后到下一个 0xAA/0x55 的长度) 做键
    clusters = defaultdict(list)

    last_poll = 0.0
    while time.time() - t0 < seconds:
        if poll and (time.time() - last_poll) > 0.02:
            # 用全零 MIT 帧被动索要反馈（kp=kd=0，不会产生力矩）
            try:
                for mid in range(1, 13):
                    drv.driver.control_mit(
                        list(drv.driver.motors.keys())[mid - 1], 0., 0., 0., 0., 0.)
            except Exception:
                pass
            last_poll = time.time()

        n = ser.in_waiting
        if n:
            data = ser.read(n)
            total += len(data)
            buf.extend(data)
            if mode == "raw":
                print("%8.3f  %s" % (time.time() - t0, data.hex(" ")))

        # --- 解析 CAN 帧 ---
        while len(buf) >= CAN_FRAME_LEN:
            idx = buf.find(CAN_HDR)
            if idx < 0:
                other_bytes += len(buf)
                _cluster(clusters, bytes(buf))
                buf.clear()
                break
            if len(buf) - idx < CAN_FRAME_LEN:
                if idx > 0:
                    other_bytes += idx
                    _cluster(clusters, bytes(buf[:idx]))
                    del buf[:idx]
                break
            if buf[idx + CAN_FRAME_LEN - 1] != CAN_TAIL:
                # 不是完整 CAN 帧 → 可能是别的帧类型
                other_bytes += 1
                _cluster(clusters, bytes(buf[idx:idx + 1]))
                del buf[:idx + 1]
                continue
            if idx > 0:
                other_bytes += idx
                _cluster(clusters, bytes(buf[:idx]))
                del buf[:idx]
            frame = bytes(buf[:CAN_FRAME_LEN])
            del buf[:CAN_FRAME_LEN]
            can_frames += 1
            can_id = int.from_bytes(frame[3:7], "little")
            can_ids[can_id] += 1
            if mode == "unknown" and can_frames <= 3:
                print("  [CAN 示例] id=0x%08X data=%s" % (can_id, frame[7:15].hex(" ")))

        time.sleep(0.002)

    while buf:
        idx = buf.find(CAN_HDR)
        if idx < 0:
            _cluster(clusters, bytes(buf)); other_bytes += len(buf); buf.clear()
        elif idx == 0:
            break
        else:
            _cluster(clusters, bytes(buf[:idx])); other_bytes += idx; del buf[:idx]

    try:
        ser.close()
    except Exception:
        pass

    print("-" * 74)
    print("收到 %d 字节；识别为 CAN 帧 %d 帧（%d 字节），其余 %d 字节"
          % (total, can_frames, can_frames * CAN_FRAME_LEN, other_bytes))

    if mode == "can":
        print()
        print("CAN ID 统计（出现次数）:")
        for cid, c in can_ids.most_common():
            print("  0x%08X  %6d 次" % (cid, c))
        ids = set(can_ids)
        print()
        print("对照 hardware_map.yaml 期望的 12 个电机 ID: 1..12")
        missing = [i for i in range(1, 13) if i not in ids]
        if missing:
            print("  ⚠️ 没出现: %s  ——  可能电机没上电 / ID 拨码不对 / 没发轮询" % missing)
        else:
            print("  ✅ 1..12 全部出现")
        extra = sorted(i for i in ids if i not in range(1, 13))
        if extra:
            print("  ★ 非电机的 ID（可能就是下位机自己在广播，含 IMU 的那个）: %s"
                  % ["0x%X" % e for e in extra])

    if mode in ("unknown", "raw") and clusters:
        print()
        print("非 CAN 字节流聚类（按 [首字节][长度] 归并，取最多的前 12 类）:")
        print("  %-8s %-8s %-8s  %s" % ("首字节", "长度", "次数", "样例"))
        ranked = sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:12]
        for (first, length), samples in ranked:
            ex = samples[0]
            print("  0x%02X     %-8d %-8d  %s"
                  % (first, length, len(samples), ex[:24].hex(" ")))
        print()
        print("★ 下一步：慢慢倾斜/转动机器人，看哪一类的字节在变 → 那就是欧拉角帧。")
        print("  找到后把 frame_id / frame_len / euler_offset / euler_dtype 填进 config.yaml。")


def _cluster(clusters, chunk: bytes, max_len: int = 64):
    """把一段非 CAN 字节按 (首字节, 长度) 归类，长度上限避免噪声撑爆内存。"""
    if not chunk:
        return
    if len(chunk) <= max_len:
        clusters[(chunk[0], len(chunk))].append(chunk)
    else:
        for i in range(0, min(len(chunk), max_len * 8), max_len):
            sub = chunk[i:i + max_len]
            if sub:
                clusters[(sub[0], len(sub))].append(sub)
    # 防止长跑内存增长
    for k in list(clusters):
        if len(clusters[k]) > 200:
            clusters[k] = clusters[k][-200:]


def main():
    ap = argparse.ArgumentParser(description="Type-C 链路嗅探（找欧拉角帧 / 查电机 ID）")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--can-baud", type=int, default=1000000)
    ap.add_argument("--no-poll", action="store_true",
                    help="不发被动轮询（只被动听。但 RS-02 是有指令才有反馈）")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--can", action="store_true", help="只看 CAN（电机 ID 统计）")
    g.add_argument("--unknown", action="store_true", help="重点看非 CAN 字节流（找 IMU 帧）")
    g.add_argument("--raw", action="store_true", help="原始字节倾泻")
    args = ap.parse_args()

    mode = "can" if args.can else ("raw" if args.raw else "unknown")
    try:
        run_sniffer(args.port, args.baud, args.seconds, mode, args.can_baud,
                    poll=not args.no_poll)
    except ImportError as e:
        print("❌ 缺 pyserial: pip install pyserial   (%s)" % e)
        return 1
    except Exception as e:
        print("❌ 打开 %s 失败: %s" % (args.port, e))
        print("   确认：设备插好了吗（ls /dev/ttyUSB* / ttyACM*）、"
              "当前用户在 dialout 组吗（sudo usermod -aG dialout $USER 后重登）")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
