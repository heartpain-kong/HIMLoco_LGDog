#!/usr/bin/env python3
"""report.py — LgRobotDog 训练报告：配置改动 vs 指标（三层记录体系第三层）

遍历 logs 下所有带 config_snapshot.json 的 run：
  1. 相邻 run 配置差异（哪些参数改了）
  2. 每个 run 的关键指标（parse_tb 解析 events）
  3. 输出总表 → 直接回答"哪个配置改动导致什么结果"

用法（Linux，himloco 环境）：
  python tools/report.py                          # 全部 run 总表
  python tools/report.py --exp rough_a1           # 只看某实验
  python tools/report.py --tags mean_episode_length,rew_ang_vel_xy   # 只看指定指标
"""

import argparse
import os
import sys
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import parse_tb  # noqa: E402
from tuning_log import find_snapshots, load_snapshot, diff_snapshots  # noqa: E402

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LOGS_ROOT = os.path.join(PROJECT_ROOT, "legged_gym", "logs")

DEFAULT_TAGS = ["Train/mean_episode_length", "Train/mean_reward",
                "Episode/rew_ang_vel_xy", "Episode/rew_lin_vel_z",
                "Episode/rew_base_height", "Episode/rew_tracking_lin_vel",
                "Episode/rew_orientation", "Loss/Estimation Loss",
                "Policy/mean_noise_std"]


def extract_last(events_path, tags):
    """解析 events，返回 {tag: 末值}"""
    result = {}
    series = {}
    if not events_path:
        return result
    for rec in parse_tb.read_records(events_path):
        try:
            step, t = parse_tb.parse_event(rec)
        except Exception:
            continue
        for tag, v in t.items():
            series.setdefault(tag, []).append(v)
    for tag in tags:
        if tag in series and series[tag]:
            result[tag] = series[tag][-1]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default=None, help="只看某实验（如 rough_a1）")
    p.add_argument("--tags", default=None, help="指定指标，逗号分隔")
    p.add_argument("--max-diffs", type=int, default=6, help="每 run 最多显示配置差异条数")
    args = p.parse_args()

    tags = DEFAULT_TAGS if not args.tags else args.tags.split(",")
    snaps = find_snapshots()
    if args.exp:
        snaps = [s for s in snaps if os.path.basename(os.path.dirname(os.path.dirname(s["path"]))) == args.exp]

    if not snaps:
        print("❌ 没有找到 config_snapshot.json（先跑改过的 train.py 训练）")
        return

    print(f"共 {len(snaps)} 个 run 带配置快照\n")

    for i, s in enumerate(snaps):
        run_dir = os.path.dirname(s["path"])
        exp = os.path.basename(os.path.dirname(run_dir))
        run = os.path.basename(run_dir)
        print("=" * 100)
        print(f"▶ {exp}/{run}")

        # 配置差异（vs 前一个 run）
        if i > 0:
            prev = snaps[i - 1]
            diffs = diff_snapshots(load_snapshot(prev["path"]), load_snapshot(s["path"]))
            if diffs:
                print(f"  配置改动（vs {os.path.basename(prev['path']).replace('config_snapshot.json','')}{os.path.basename(os.path.dirname(prev['path']))}）:")
                for d in diffs[:args.max_diffs]:
                    print(f"    {d['key']:<40} {d['old']} -> {d['new']}")
                if len(diffs) > args.max_diffs:
                    print(f"    ... 共 {len(diffs)} 项")
            else:
                print("  配置：与上一 run 无差异")
        else:
            print("  配置：基线（首个带快照的 run）")

        # 指标
        events = glob.glob(os.path.join(run_dir, "events.out.tfevents*"))
        if events:
            vals = extract_last(events[0], tags)
            if vals:
                print("  指标（末值）:")
                for tag in tags:
                    if tag in vals:
                        print(f"    {tag:<38} {vals[tag]:.4f}")
        else:
            print("  ⚠️ 无 events 文件")

    print("\n" + "=" * 100)
    print("提示：结合 tuning_log.py show 查看调参记录，形成完整闭环")


if __name__ == "__main__":
    main()
