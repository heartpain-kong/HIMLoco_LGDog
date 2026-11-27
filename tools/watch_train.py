#!/usr/bin/env python3
"""watch_train.py — 训练哨兵：从 TensorBoard events 检测"抖腿 chattering 崩解"

为什么需要它（V4/V10/V11 的教训）：
  这套配置在多个轮次里反复出现同类崩解——策略退化成【高频抖腿】，
  episode 骤降、reward 转负、之后就再也爬不出来（V4/V10 是永久锁死，
  V11 谷底较浅才自行恢复）。已确认 5 次：
      第13轮 1200-1500 / V10 2250-2500 / 第8轮 3000-3100
      V11 4500-6500（唯一自愈） / V4 8300-8400
  共同签名（可靠指纹）：
      rew_smoothness 底层量飙到几十~几千倍 + episode 骤降 + est loss 上升
  而 V11 还显示崩解【有前兆】：抖动先从 6 缓慢升到 32 再到 99，约 1000 轮后才崩。
  所以能在崩之前预警，不必等到 episode 已经掉下去。

判定口径（关键）：
  rew_* 是【每秒均值】，且被各自权重缩放，跨轮不可直接比。
  本脚本把 rew_smoothness 除以 config_snapshot 里的权重，还原成
  【动作二阶差分的抖动量】（记作 jitter），这才是跨轮可比的物理量。
  历史参考值：健康期 4~10；V11 谷底 ~750；V10/V4 崩解 ~1700~2000；第13轮 858。

用法：
  # 一次性查看最新 run
  python tools/watch_train.py

  # 指定 run（run 目录名，如 V11 或 Sep18_10-42-30_LgDog1_himloco）
  python tools/watch_train.py --run Sep18_10-42-30_LgDog1_himloco

  # 持续盯（默认每 60s 刷新一次；发现预警/崩解会重复告警）
  python tools/watch_train.py --watch

  # 只在异常时输出（配合 cron / 训练脚本用，退出码非 0 表示有异常）
  python tools/watch_train.py --quiet

退出码：0 = 正常；2 = 预警（抖动超基线倍数）；3 = 崩解（抖动/episode 双双越线）
"""

import argparse
import glob
import json
import os
import signal
import statistics
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LOGS_ROOT = os.path.join(PROJECT_ROOT, "legged_gym", "logs")

# ── 判定阈值（基于历轮实测标定，见文件头）─────────────────────────────────
JITTER_WARN_ABS = 100.0    # 抖动绝对值超过它 = 已崩（健康期 4~10、崩解 750~2000）
JITTER_WARN_MULT = 3.0     # 抖动超过【基线】的倍数即预警（V11 崩前是 5x 再 16x）
EP_WARN_MULT = 0.6         # episode 低于【基线】的 0.6 倍即算骤降
EST_WARN_MULT = 1.5        # est loss 超过基线 1.5 倍算恶化
BASELINE_MIN_SAMPLES = 8   # 至少这么多轮才敢算基线
SUSTAIN = 3                # 需连续这么多个采样点越线才算真崩（滤掉初始随机阶段的毛刺）
# 躺平解（与抖腿崩解是【两种不同】的失败模式，V6/V7 是躺平、V4/V10 是抖腿）
TRACK_LOW = 0.35           # tracking 归一化(0-1) 低于它 = 基本不会走（健康 0.6~0.8）
EP_SATURATED = 0.85        # episode 达到满值的这个比例 = 反常"活得久"（不动就不摔）
# ★最早判定迭代数：低于它【不做 STANDSTILL 判定】。
#   原因（V13 实测踩到）：迭代 145 时 V13 = episode 3446 / tracking 0.194，
#   而【最健康】的 V12 在同一位置是 3234 / 0.224 —— 开局"还不会走 + 不动就不摔"
#   是所有轮次的共同形态，此时判躺平必然是假警报。
#   下面这张实测分界表说明何时才分得开（tracking 归一化首次连续 5 个采样 ≥0.35 的迭代）：
#     健康/成功轮: V9=147  V5=150  V12=183  V2=265   → 最慢 265
#     真躺平轮:    V6/V7 永远达不到 0.35            → 400 轮时仍只有 0.18~0.22
#   取 400 = 最慢健康轮 265 之上留 1.5x 余量，同时仍能在 400 轮就抓出 V6/V7 那类退化。
MIN_ITERS_TO_JUDGE = 400


def find_latest_run():
    runs = []
    for snap in glob.glob(os.path.join(LOGS_ROOT, "*", "*", "config_snapshot.json")):
        d = os.path.dirname(snap)
        runs.append((os.path.getmtime(snap), os.path.basename(d)))
    if not runs:
        return None
    return max(runs)[1]


def find_run_dir(run_name):
    hits = glob.glob(os.path.join(LOGS_ROOT, "*", run_name))
    if not hits:
        # 也允许只给前缀
        hits = [d for d in glob.glob(os.path.join(LOGS_ROOT, "*", "*"))
                if os.path.basename(d).startswith(run_name)]
    if not hits:
        return None
    return sorted(hits, key=os.path.getmtime)[-1]


def load_weights(run_dir):
    """从 config_snapshot.json 取需要的奖励权重 + 满回合步数（还原物理量/判躺平用）"""
    with open(os.path.join(run_dir, "config_snapshot.json"), encoding="utf-8") as f:
        snap = json.load(f)
    env = snap["env_cfg"]
    sc = env["rewards"]["scales"]
    dt = float(env["sim"]["dt"]) * float(env["control"]["decimation"])
    ep_s = env["env"].get("episode_length_s")
    return {
        "smoothness": abs(float(sc.get("smoothness") or 0)) or 1.0,
        "tracking_lin_vel": abs(float(sc.get("tracking_lin_vel") or 0)) or 1.0,
        "episode_length_s": ep_s,
        "max_episode_length": (ep_s / dt) if ep_s else None,
    }


def read_series(run_dir):
    """读 events 文件，返回 {tag: [(step, value), ...]}（只读一次，由 event_accumulator 缓存）"""
    from tensorboard.backend.event_processing import event_accumulator
    evs = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not evs:
        return None
    ea = event_accumulator.EventAccumulator(max(evs, key=os.path.getmtime),
                                           size_guidance={"scalars": 0})
    ea.Reload()
    return {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in ea.Tags()["scalars"]}


def analyse(series, w):
    """返回 (指标字典, 判定)"""
    def last(tag, default=None):
        s = series.get(tag)
        return s[-1][1] if s else default

    def seg(tag, lo, hi):
        s = [v for k, v in series.get(tag, []) if lo <= k <= hi]
        return statistics.mean(s) if s else None

    sm = series.get("Episode/rew_smoothness")
    if not sm:
        return None, ("NO_DATA", "events 里还没有 rew_smoothness")

    it_max = sm[-1][0]
    jitter_now = abs(last("Episode/rew_smoothness")) / w["smoothness"]
    ep_now = last("Train/mean_episode_length")
    est_now = last("Loss/Estimation Loss")
    trk_now = abs(last("Episode/rew_tracking_lin_vel")) / w["tracking_lin_vel"] \
        if last("Episode/rew_tracking_lin_vel") is not None else None
    cmd_now = last("Episode/max_command_x")

    # 基线：取【早期健康段】(前 30%，且至少 BASELINE_MIN_SAMPLES 轮) 的抖动中位数。
    # 用中位数是为了不被崩解段污染。若早期就崩（如 V6/V7 开局即躺平）则基线会偏高，
    # 此时多重告警里的 episode 项会独立报出来。
    early_hi = max(int(it_max * 0.3), 1)
    early = [abs(v) / w["smoothness"] for k, v in sm if k <= early_hi]
    jitter_base = statistics.median(early) if len(early) >= BASELINE_MIN_SAMPLES else None
    ep_base = seg("Train/mean_episode_length", 0, early_hi) if early else None

    m = {
        "iter": it_max, "jitter": jitter_now, "jitter_base": jitter_base,
        "episode": ep_now, "episode_base": ep_base, "est": est_now,
        "tracking": trk_now, "max_cmd": cmd_now,
        "jitter_mult": (jitter_now / jitter_base) if jitter_base else None,
        "ep_ratio": (ep_now / ep_base) if (ep_base and ep_now) else None,
    }

    # ── 判定 ──
    # 预热态：样本不足时【不做任何判定】——初始随机策略的抖动天然高、tracking 天然低，
    # 此时报 WARN 是假警报（V12 第 14 轮实测误报过一次）。
    if jitter_base is None or len(sm) < BASELINE_MIN_SAMPLES:
        return m, ("WARMUP", f"预热中（仅 {len(sm)} 轮；基线需 ≥{BASELINE_MIN_SAMPLES} 轮才建立，暂不判定）"), []
    # ★迭代数闸门：开局"不会走 + 不动就不摔"是每一轮的共同形态（V12 在 145 轮时
    #   也只有 tracking 0.22），此时任何"躺平"判定都是假警报，必须等到 400 轮以后。
    #   注意仍会走到这里看 jitter/episode 相对基线的【骤降】——那种崩解在早期同样真实。
    if it_max < MIN_ITERS_TO_JUDGE:
        return m, ("WARMUP",
                   f"早期训练中（{it_max} < {MIN_ITERS_TO_JUDGE} 轮）："
                   f"tracking {trk_now:.2f} / episode {ep_now:.0f} 属正常早期形态"
                   f"（健康轮 V12 在 183 轮才越过 tracking 0.35），暂不判定"), []

    # ★崩解历史：只看【当前采样】会漏报 —— V16 实测踩到: 该轮在 3987~4632 与 5539~5845
    #   崩解过两次（抖动峰值 900 / 2434）又都自愈，末态抖动 5.9，于是当时报的是 '✅ OK'。
    #   自愈后的策略即使能动，"经历过崩解"这件事本身也该让人知道（模型是从崩解里恢复出来的），
    #   所以这里统计全部历史崩解段，并把结论并入 WARN（不新造退出码）。
    episodes, cur = [], None
    for k, v in sm:
        if k <= max(int(it_max * 0.3), 1):      # 跳过基线窗口（初始随机策略抖动天然高）
            continue
        over = abs(v) / w["smoothness"] >= JITTER_WARN_ABS
        if over and cur is None:
            cur = k
        elif not over and cur is not None:
            episodes.append((cur, k)); cur = None
    if cur is not None:
        episodes.append((cur, it_max))
    # ⚠️ 只保留【连续 ≥SUSTAIN 个采样点】的段，与 last_healthy_checkpoint 的口径一致。
    #   否则单点毛刺会被算成崩解: V14 实测在 4351 只有一个采样点抖动 103（episode 2310 几乎没掉），
    #   不加这层过滤就会把一个健康的 8828 轮 run 误报成 WARN。
    _n = {k: i for i, (k, _) in enumerate(sm)}
    episodes = [e for e in episodes if _n.get(e[1], 0) - _n.get(e[0], 0) + 1 >= SUSTAIN]

    reasons = []
    if jitter_base and m["jitter_mult"] and m["jitter_mult"] >= JITTER_WARN_MULT:
        reasons.append(f"抖动是基线的 {m['jitter_mult']:.1f} 倍(基线 {jitter_base:.1f})")
    if m["ep_ratio"] and m["ep_ratio"] <= EP_WARN_MULT:
        reasons.append(f"episode 降到基线的 {m['ep_ratio']:.0%}(基线 {ep_base:.0f})")
    est_base = seg("Loss/Estimation Loss", 0, early_hi)
    if est_base and est_now is not None and est_now / est_base >= EST_WARN_MULT:
        reasons.append(f"est loss 是基线的 {est_now / est_base:.1f} 倍")

    # 躺平解：tracking 极低 且 episode 反常接近满值（不动就不摔）
    max_ep = w.get("max_episode_length")
    standstill = False
    if trk_now is not None and trk_now < TRACK_LOW:
        if max_ep and ep_now and ep_now >= EP_SATURATED * max_ep:
            standstill = True
            reasons.append(f"tracking 仅 {trk_now:.2f} 却 episode {ep_now:.0f}"
                           f"({ep_now / max_ep:.0%} 满值) → 躺平解")
        elif jitter_now < JITTER_WARN_ABS:
            reasons.append(f"tracking 仅 {trk_now:.2f}（健康 0.6~0.8）")

    collapsed = jitter_now >= JITTER_WARN_ABS and (
        m["ep_ratio"] is None or m["ep_ratio"] <= EP_WARN_MULT)
    if collapsed:
        return m, ("COLLAPSE", "；".join(reasons) or f"抖动 {jitter_now:.0f} 越线"), episodes
    if standstill:
        return m, ("STANDSTILL", "；".join(reasons)), episodes
    if episodes:
        reasons = reasons + [
            f"本轮历史上崩解过 {len(episodes)} 段（最近 {episodes[-1][0]}~{episodes[-1][1]}，"
            f"当前{'已恢复' if jitter_now < JITTER_WARN_ABS else '仍未恢复'}）"
        ]
    if reasons:
        return m, ("WARN", "；".join(reasons)), episodes
    return m, ("OK", "健康"), episodes


def last_healthy_checkpoint(run_dir, jitter_base, w, mult=JITTER_WARN_MULT):
    """找最后一个【抖动未持续越线】的 checkpoint，供回滚用

    ⚠️ 必须跳过最初 30% 的基线窗口：初始随机策略的抖动天然很高，
    否则会把'第一个点'误判成越线点、回滚建议退化成 model_0（V11 实测踩过）。
    并且要求【连续 SUSTAIN 个采样点】越线才算真崩，滤掉单点毛刺。
    """
    cks = []
    for f in os.listdir(run_dir):
        if f.startswith("model_") and f.endswith(".pt"):
            try:
                cks.append(int(f[len("model_"):-3]))
            except ValueError:
                pass
    if not cks or not jitter_base:
        return None
    series = read_series(run_dir)
    sm = sorted(series.get("Episode/rew_smoothness", []))
    if not sm:
        return None
    it_max = sm[-1][0]

    # 只在【基线窗口之后】找持续越线点
    floor = int(it_max * 0.3)
    thr = jitter_base * mult
    streak, bad = 0, None
    for k, v in sm:
        if k <= floor:
            continue
        if abs(v) / w["smoothness"] > thr:
            streak += 1
            if streak >= SUSTAIN:
                bad = k
                break
        else:
            streak = 0
    safe = [c for c in sorted(cks) if bad is None or c < bad]
    return safe[-1] if safe else None


def report(m, verdict, kind, run_name, w, run_dir):
    icon = {"OK": "✅", "WARN": "⚠️", "COLLAPSE": "❌", "STANDSTILL": "🛑", "WARMUP": "·", "NO_DATA": "·"}[kind]
    print(f"{icon} [{kind}] {run_name}  iter {m['iter'] if m else '?'}")
    if m:
        jb = f"{m['jitter_base']:.1f}" if m["jitter_base"] else "?"
        print(f"     抖动 {m['jitter']:.1f} (基线 {jb}"
              + (f"，{m['jitter_mult']:.1f}x" if m["jitter_mult"] else "") + ")")
        ep = f"{m['episode']:.0f}" if m["episode"] else "?"
        eb = f"{m['episode_base']:.0f}" if m["episode_base"] else "?"
        print(f"     episode {ep} (基线 {eb})   est {m['est']:.4f}"
              f"   tracking {m['tracking']:.4f}   max_cmd {m['max_cmd']:.3f}"
              + (f"   [episode_length_s={w['episode_length_s']}]" if w["episode_length_s"] else ""))
    if verdict:
        print(f"     判定: {verdict}")
    if kind in ("WARN", "COLLAPSE", "STANDSTILL"):
        ck = last_healthy_checkpoint(run_dir, m["jitter_base"] if m else None, w)
        if ck is not None:
            print(f"     ↳ 回滚建议: 用 model_{ck}.pt（最后一个抖动未越线的 checkpoint）")


def find_training_pids():
    """找出正在跑的 train.py 进程 PID。

    ★只按 PID 精确杀，【绝不用 pkill -f 模式匹配】—— 本项目踩过: `pkill -f play.py`/`pkill -f watch...`
    会把发起命令的【自己的 shell】一起匹配杀掉（命令行里含同样的字符串）。
    这里排除自身与全部祖先进程，并要求命令行里同时出现 train.py 与 python、且不含 watch_train。
    """
    me = os.getpid()
    ancestors = set()
    p = me
    while p and p != 1:
        ancestors.add(p)
        try:
            with open(f"/proc/{p}/stat", "r") as f:
                p = int(f.read().rsplit(")", 1)[1].split()[1])
        except Exception:
            break
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        pid = int(d)
        if pid in ancestors:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cl = f.read().decode(errors="replace").replace("\0", " ").strip()
        except Exception:
            continue
        if "train.py" in cl and "python" in cl and "watch_train" not in cl:
            out.append((pid, cl[:120]))
    return out


def main():
    ap = argparse.ArgumentParser(description="训练哨兵：检测抖腿 chattering 崩解")
    ap.add_argument("--run", default=None, help="run 目录名（默认取最新 run）")
    ap.add_argument("--watch", action="store_true", help="持续盯（默认 60s 一次）")
    ap.add_argument("--interval", type=float, default=60.0, help="--watch 的刷新间隔 [s]")
    ap.add_argument("--quiet", action="store_true", help="只在预警/崩解时输出")
    ap.add_argument("--stop-on-collapse", action="store_true",
                    help="★崩解自动早停: 连续判定为 COLLAPSE 时 SIGTERM 掉训练进程（保住 checkpoint，"
                         "不再白跑崩坏后的几千轮）")
    ap.add_argument("--stop-after", type=int, default=2,
                    help="--stop-on-collapse 需要连续几次 COLLAPSE 才动手（默认 2，滤掉瞬时抖动）")
    args = ap.parse_args()

    worst = 0
    collapse_streak = 0
    while True:
        run_name = args.run or find_latest_run()
        if not run_name:
            print("❌ 找不到任何 run（logs/*/*/config_snapshot.json）")
            return 1
        run_dir = find_run_dir(run_name)
        if not run_dir:
            print(f"❌ 找不到 run 目录: {run_name}")
            return 1
        w = load_weights(run_dir)
        series = read_series(run_dir)
        if series is None:
            print(f"❌ {run_name} 还没有 events 文件")
            return 1

        m, (kind, why), episodes = analyse(series, w)
        if not args.quiet or kind in ("WARN", "COLLAPSE"):
            report(m, why, kind, os.path.basename(run_dir), w, run_dir)

        worst = max(worst, {"OK": 0, "NO_DATA": 0, "WARMUP": 0, "WARN": 2, "STANDSTILL": 2, "COLLAPSE": 3}[kind])
        # ★崩解自动早停（2026-09-22 新增）: 连续 N 次 COLLAPSE 就停训练，保住 checkpoint
        if args.stop_on_collapse:
            if kind == "COLLAPSE":
                collapse_streak += 1
                if collapse_streak >= args.stop_after:
                    ck = last_healthy_checkpoint(run_dir, m["jitter_base"] if m else None, w)
                    pids = find_training_pids()
                    print(f"🛑 [AUTO-STOP] 连续 {collapse_streak} 次判定 COLLAPSE → 停止训练，"
                          f"保住最好 checkpoint = model_{ck}.pt" if ck is not None else
                          f"🛑 [AUTO-STOP] 连续 {collapse_streak} 次 COLLAPSE → 停止训练")
                    if not pids:
                        print("   ⚠ 没找到 train.py 进程（可能已自行结束）")
                    for pid, cl in pids:
                        try:
                            os.kill(pid, signal.SIGTERM)
                            print(f"   已 SIGTERM PID {pid}: {cl}")
                        except Exception as e:
                            print(f"   ⚠ 结束 PID {pid} 失败: {e}")
                    print("   ⇒ 可以用 play.py 导出该 checkpoint，并在 tuning_log 里结算（区分崩坏前后的轮次）")
                    return 3
            else:
                collapse_streak = 0
        if not args.watch:
            return worst
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
