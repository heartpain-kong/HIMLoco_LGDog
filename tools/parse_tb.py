#!/usr/bin/env python3
"""parse_tb.py — 零依赖解析 TensorBoard events 文件（纯 Python，无 tensorboard/numpy）

直接解析 TFRecord + protobuf wire format，提取标量指标：
  Train/mean_episode_length, Train/mean_reward, Episode/rew_*, Loss/*, Policy/mean_noise_std

用法：
  python parse_tb.py <events文件> [--tags tag1,tag2] [--samples 5]
  python parse_tb.py <run目录>                # 自动找 events 文件
示例：
  python parse_tb.py logs/rough_a1/Aug24_03-30-55_/
  python parse_tb.py logs/rough_a1/runV1/ --samples 8
"""

import struct
import sys
import os
import zlib


# ---------- TFRecord ----------
def read_records(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    n = len(data)
    while pos + 8 <= n:
        (length,) = struct.unpack_from("<Q", data, pos)      # uint64 length
        pos += 8 + 4                                          # + crc32c(length)
        if pos + length + 4 > n:
            break
        yield data[pos:pos + length]                          # Event 消息
        pos += length + 4                                     # + crc32c(data)


# ---------- protobuf wire ----------
def _varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _fields(buf):
    """解析 protobuf 消息的字段：返回 {field_num: [(wire_type, value)], ...}"""
    out = {}
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _varint(buf, pos)
        field = tag >> 3
        wt = tag & 7
        if wt == 0:      # varint
            v, pos = _varint(buf, pos)
        elif wt == 1:    # 64-bit
            v = buf[pos:pos + 8]; pos += 8
        elif wt == 2:    # length-delimited
            ln, pos = _varint(buf, pos)
            v = buf[pos:pos + ln]; pos += ln
        elif wt == 5:    # 32-bit
            v = buf[pos:pos + 4]; pos += 4
        else:
            break
        out.setdefault(field, []).append((wt, v))
    return out


def _f64(v):
    return struct.unpack("<d", v)[0]


def _f32(v):
    return struct.unpack("<f", v)[0]


def parse_event(event_bytes):
    """返回 (step, {tag: simple_value})"""
    f = _fields(event_bytes)
    step = 0
    tags = {}
    for wt, v in f.get(2, []):                       # step int64
        step = v if isinstance(v, int) else 0
    for wt, summary in f.get(5, []):                 # summary message（字段 5，非 9）
        sf = _fields(summary)
        for _, val in sf.get(1, []):                 # summary.value (repeated)
            vf = _fields(val)
            tag = None
            simple = None
            for _, t in vf.get(1, []):               # value.tag (string)
                tag = t.decode("utf-8", "ignore")
            for _, sv in vf.get(2, []):              # value.simple_value (float)
                simple = _f32(sv)
            if tag and simple is not None:
                tags[tag] = simple
    return step, tags


# ---------- 主逻辑 ----------
def find_events(path):
    if os.path.isfile(path):
        return [path]
    cands = [os.path.join(path, f) for f in os.listdir(path) if f.startswith("events.out.tfevents")]
    return cands


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    target = sys.argv[1]
    tags_filter = None
    samples = 5
    for i, a in enumerate(sys.argv[2:]):
        if a == "--tags" and i + 2 < len(sys.argv):
            tags_filter = sys.argv[3 + i].split(",")
        if a == "--samples" and i + 2 < len(sys.argv):
            samples = int(sys.argv[3 + i])

    events_files = find_events(target)
    if not events_files:
        print(f"❌ 找不到 events 文件: {target}")
        return

    for ef in events_files:
        print(f"\n===== {os.path.basename(os.path.dirname(ef))} =====")
        series = {}          # tag -> [(step, value)]
        for rec in read_records(ef):
            try:
                step, tags = parse_event(rec)
            except Exception:
                continue
            for t, v in tags.items():
                series.setdefault(t, []).append((step, v))

        if not series:
            print("  (无标量数据)")
            continue

        # 选择要显示的指标
        keys = list(series.keys())
        print(f"  共 {len(keys)} 个指标。step 范围: {series[min(series, key=lambda k: series[k][0][0])][0][0]} ~ "
              f"{series[max(series, key=lambda k: series[k][-1][0])][-1][0]}")
        print(f"  {'指标':<34} {'min':>10} {'max':>10} {'末值':>10}  首→末")
        print("-" * 100)
        for k in sorted(series.keys()):
            if tags_filter and not any(f in k for f in tags_filter):
                continue
            pts = series[k]
            vals = [v for _, v in pts]
            if len(vals) < 2:
                continue
            first = vals[0]
            last = vals[-1]
            print(f"  {k:<34} {min(vals):>10.4f} {max(vals):>10.4f} {last:>10.4f}  {first:.3f}→{last:.3f}")

        # 关键指标采样曲线
        important = ["Train/mean_episode_length", "Train/mean_reward",
                     "Episode/rew_base_height", "Episode/rew_orientation",
                     "Episode/rew_ang_vel_xy", "Episode/rew_collision",
                     "Episode/rew_lin_vel_z", "Episode/rew_tracking_lin_vel",
                     "Loss/Estimation Loss", "Loss/Swap Loss",
                     "Policy/mean_noise_std", "Loss/learning_rate"]
        print(f"\n  —— 关键指标采样曲线（{samples} 个点）——")
        for k in important:
            if k not in series:
                continue
            pts = series[k]
            idx = [int(i * (len(pts) - 1) / (samples - 1)) for i in range(samples)]
            row = "  " + k.ljust(34) + " " + " ".join(f"{pts[i][1]:9.3f}" for i in idx)
            print(row)


if __name__ == "__main__":
    main()
