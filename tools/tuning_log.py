#!/usr/bin/env python3
"""tuning_log.py — LgRobotDog 调参实验日志（增强版，借鉴 OpenDoge tuning_log.json）

三层记录体系第二层：调参记录（手动 + 自动 diff）。
每个 run 的完整配置由 train.py 自动存到 run 目录的 config_snapshot.json，
本工具负责"改了什么 / 为什么改 / 效果如何"。

子命令：
  run    为一次训练 run 建记录（★每次训练都必须做★，自动 diff 出配置改动）
  add    追加一条记录（改配置时用；--diff 自动对比最近两个 run 快照生成 changes）
  update 验证后回填 actual_effect（防重复无效修改）
  show   查看全部记录（先 show 再改，避免重复踩坑）+ 漏记提醒
  diff   直接对比两个 run 的配置快照，输出差异（辅助诊断）

铁律：**每一个训练 run 都必须在 tuning_log.json 里有且仅有一条记录**。

用法：
  # ★训练启动前：为本轮 run 建记录（自动 diff 出改动，无改动也会记一条）
  python tools/tuning_log.py run --run_dir Sep13_03-07-23_LgDog1_himloco \
      --diagnosis "小腿胶囊修复后首轮" --expected "rew_collision 非零且 episode 不降"

  # 改配置后手动记录（值里可以安全带逗号）：
  python tools/tuning_log.py add --diagnosis "出生就倒，回退 V1 权重" \
      --changes "rewards.scales.lin_vel_z: -5.2 -> -2.0, rewards.scales.orientation: -5.2 -> -2.4" \
      --expected "episode 回到 1000" --run_dir Sep13_03-07-23_LgDog1_himloco

  # 自动 diff 最近两个 run（改动自动生成 changes）：
  python tools/tuning_log.py add --diff --diagnosis "V2 惩罚过载" --expected "能站稳"

  # 验证后回填（index 用 show 查，-1 = 最后一条）：
  python tools/tuning_log.py show
  python tools/tuning_log.py update --index 3 --effect "FAILED: ep=17 仍倒"

  # 对比任意两个 run 的配置：
  python tools/tuning_log.py diff --from runV1 --to Aug24_03-30-55_
"""

import argparse
import json
import os
import re
import glob
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                 # tools/ -> LgRobotDog/
DEFAULT_LOG = os.path.join(PROJECT_ROOT, "tuning_log.json")
LOGS_ROOT = os.path.join(PROJECT_ROOT, "legged_gym", "logs")

# 改动项分隔符：';' 永远分隔；',' 仅当后面紧跟 'key:' 时才分隔
SEP_RE = re.compile(r"\s*[;；]\s*|\s*,\s*(?=[^\s,:：]+[:：])")


def run_timestamp(run_dir):
    """从 run 目录名（如 Sep13_03-07-23_LgDog1_himloco）解析出时间，失败则取当前时间"""
    m = re.match(r"([A-Z][a-z]{2}\d{1,2})_(\d{2}-\d{2}-\d{2})", run_dir or "")
    if m:
        try:
            dt = datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%b%d_%H-%M-%S")
            return dt.replace(year=datetime.now().year).isoformat(timespec="seconds")
        except ValueError:
            pass
    return datetime.now().isoformat(timespec="seconds")


def find_record(records, run_dir):
    """返回该 run_dir 已记录的索引列表"""
    return [i for i, r in enumerate(records) if (r.get("run_dir") or "") == run_dir]


# ---------- 基础 ----------
def load(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save(path, records):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_changes(s):
    """解析 'k: 旧 -> 新, k2: 旧 -> 新' 为 dict

    ⚠️ 分隔符规则（修 bug）：只有『后面紧跟另一个 key: 的逗号』才算分隔符，
    外加 ';' / '；' 永远算分隔符。这样值里可以安全写逗号，例如：
        "urdf.*_calf: 新增胶囊 (cylinder r=0.025 length=0.19, 轴向=膝轴)"
    旧实现按任意逗号切分，会把括号里的逗号当分隔符，尾巴掉进 param2 垃圾键。
    需要精确控制时用 add --changes-json '{"k": "v"}'。
    """
    changes = {}
    if not s:
        return changes
    for item in SEP_RE.split(s):
        item = item.strip().strip(",;，；")
        if not item:
            continue
        if ":" in item or "：" in item:
            k, v = re.split(r"[:：]", item, maxsplit=1)
            changes[k.strip()] = v.strip()
        else:
            changes[f"param{len(changes) + 1}"] = item
            print(f"⚠️ 改动项缺少 'key:' 冒号，已存为 {list(changes)[-1]}: {item!r}")
    return changes


# ---------- 配置快照 ----------
def find_snapshots(exp=None):
    """返回 [{run, exp, path, mtime}]，按时间排序。exp 给定时只取该实验。"""
    out = []
    for snap in glob.glob(os.path.join(LOGS_ROOT, "*", "*", "config_snapshot.json")):
        run_dir = os.path.dirname(snap)
        e = os.path.basename(os.path.dirname(run_dir))
        if exp and e != exp:
            continue
        out.append({"run": os.path.basename(run_dir), "exp": e, "path": snap,
                    "mtime": os.path.getmtime(snap)})
    return sorted(out, key=lambda x: x["mtime"])


def exp_of_run(run_dir):
    """由 run 目录名反查所属实验（logs/<exp>/<run>/）"""
    for s in find_snapshots():
        if s["run"] == run_dir:
            return s["exp"]
    return None


def load_snapshot(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten(d, prefix=""):
    """把嵌套 dict 扁平化为 {'a.b.c': value}"""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def snapshot_leaves(snap):
    """把一个 config_snapshot.json 摊平成 {点路径: 值}，四层全都要覆盖：

      env_cfg            -> 裸点路径（如 rewards.scales.lin_vel_z、terrain.mesh_type）
      train_cfg          -> 前缀 train.（如 train.algorithm.learning_rate）
      urdf_fingerprint   -> 前缀 urdf.（URDF 限位/碰撞几何，config 里看不到）
      source_fingerprint -> 前缀 src.（奖励函数/算法源码，config 里更看不到）
    """
    out = dict(flatten(snap.get("env_cfg", {})))
    out.update({f"train.{k}": v for k, v in flatten(snap.get("train_cfg", {})).items()})
    out.update({f"urdf.{k}": v for k, v in flatten(snap.get("urdf_fingerprint", {})).items()})
    out.update({f"src.{k}": v for k, v in flatten(snap.get("source_fingerprint", {})).items()})
    return out


def diff_snapshots(a, b, skip=("asset.file", "asset.name")):
    """对比两个快照，返回 [{key, old, new}]（只取变化的叶子）

    任一侧缺 urdf_fingerprint / source_fingerprint（加指纹之前产生的旧 run）时，
    对应前缀整段跳过并告警——否则会刷出几十条 'None -> x' 的假差异。
    """
    fa, fb = snapshot_leaves(a), snapshot_leaves(b)
    both_urdf = "urdf_fingerprint" in a and "urdf_fingerprint" in b
    both_src = "source_fingerprint" in a and "source_fingerprint" in b
    diffs, skipped = [], []
    for k in sorted(set(fa) | set(fb)):
        if any(s in k for s in skip):
            continue
        if k.startswith("urdf.") and not both_urdf:
            skipped.append("urdf")
            continue
        if k.startswith("src.") and not both_src:
            skipped.append("src")
            continue
        if fa.get(k) != fb.get(k):
            diffs.append({"key": k, "old": fa.get(k), "new": fb.get(k)})
    if skipped:
        print(f"⚠️ 有 run 缺 {'/'.join(sorted(set(skipped)))} 指纹"
              f"（加指纹之前产生的旧 run），该层改动无法自动对比，需人工确认")
    return diffs


# ---------- 子命令 ----------
def cmd_add(args):
    records = load(args.log)

    if args.changes_json:
        changes = json.loads(args.changes_json)
        if not isinstance(changes, dict):
            raise SystemExit("❌ --changes-json 必须是 JSON 对象，如 '{\"k\": \"旧 -> 新\"}'")
    else:
        changes = parse_changes(args.changes)

    if args.diff:
        ref_exp = exp_of_run(args.run_dir) if args.run_dir else None
        snaps = find_snapshots(ref_exp)
        if not snaps:
            snaps = find_snapshots()          # 该 run 还没快照时退回全局
        if len(snaps) >= 2:
            a, b = snaps[-2], snaps[-1]
            diffs = diff_snapshots(load_snapshot(a["path"]), load_snapshot(b["path"]))
            changes = {d["key"]: f"{d['old']} -> {d['new']}" for d in diffs}
            if not changes:
                print("⚠️ 最近两个 run 配置无差异（可能没重新训练或配置没变）")
            print(f"自动 diff[{b['exp']}]: {a['run']} → {b['run']}，共 {len(changes)} 项")
        else:
            print("⚠️ 该实验快照不足 2 个（先训练出带 config_snapshot.json 的 run）")

    if args.run_dir and not args.force:
        dup = find_record(records, args.run_dir)
        if dup:
            raise SystemExit(
                f"❌ run_dir '{args.run_dir}' 已在 #{dup} 记录过。"
                f"确认要重复记录请加 --force")

    rec = {
        "timestamp": args.timestamp or (run_timestamp(args.run_dir) if args.run_dir
                                        else datetime.now().isoformat(timespec="seconds")),
        "checkpoint_step": args.step,
        "run_dir": args.run_dir or "",
        "diagnosis": args.diagnosis,
        "changes": changes,
        "expected_effect": args.expected,
        "actual_effect": None,
    }
    records.append(rec)
    save(args.log, records)
    print(f"✅ 已追加记录 #{len(records) - 1} → {args.log}")
    print(json.dumps(rec, ensure_ascii=False, indent=2))


def cmd_run(args):
    """为一次训练 run 建一条记录：自动定位快照 + 自动 diff 出改动（只在同一实验内对比）。"""
    records = load(args.log)
    all_snaps = find_snapshots()
    if not all_snaps:
        raise SystemExit("❌ 找不到任何 config_snapshot.json（先跑一次训练）")

    if args.run_dir:
        _cands = [s for s in all_snaps if s["run"] == args.run_dir]
        # ★重名加固: run 目录名在不同实验间可能重名（实测 rough_go1 里也有 V6/V7）。
        #   若同名有多个，优先取【本日志所属实验】里的那个，否则可能记录到别的项目的 run。
        if len(_cands) > 1:
            _exp = _infer_log_experiment(records)
            _pref = [s for s in _cands if s["exp"] == _exp]
            if _pref:
                print(f"⚠️ run 名 '{args.run_dir}' 在多个实验里存在 "
                      f"{[s['exp'] for s in _cands]}；按本日志所属实验取 '{_exp}'")
                _cands = _pref
        cur = _cands[0] if _cands else None
        if cur is None:
            raise SystemExit(f"❌ 找不到 run: {args.run_dir}")
    else:
        cur = all_snaps[-1]

    if not args.force:
        dup = find_record(records, cur["run"])
        if dup:
            raise SystemExit(
                f"❌ run '{cur['run']}' 已在 #{dup} 记录过。确认要重复记录请加 --force")

    # 只与本实验的上一个 run 对比（别的实验的 run 不算"上一轮"）
    exp_snaps = find_snapshots(cur["exp"])
    idx = next(i for i, s in enumerate(exp_snaps) if s["path"] == cur["path"])
    changes, prev = {}, (exp_snaps[idx - 1] if idx > 0 else None)
    if prev:
        diffs = diff_snapshots(load_snapshot(prev["path"]), load_snapshot(cur["path"]))
        changes = {d["key"]: f"{d['old']} -> {d['new']}" for d in diffs}

    if args.changes:
        changes.update(parse_changes(args.changes))
    if args.changes_json:
        changes.update(json.loads(args.changes_json))

    if not changes:
        diag = (f"纯训练 run（相对 {prev['run']} 配置无改动）" if prev
                else f"首个 run（{cur['exp']} baseline）")
    else:
        diag = f"训练 run，自动 diff 出 {len(changes)} 项改动"
    diag = args.diagnosis or diag
    expected = args.expected or ("（无特定预期，观察 baseline）" if not changes else "")

    rec = {
        "timestamp": run_timestamp(cur["run"]),
        "checkpoint_step": args.step,
        "run_dir": cur["run"],
        "diagnosis": diag,
        "changes": changes,
        "expected_effect": expected,
        "actual_effect": None,
    }
    records.append(rec)
    save(args.log, records)

    print(f"✅ 已记录 run '{cur['run']}'"
          + (f"（对比 {prev['run']}，{len(changes)} 项改动）" if prev else "（首个 run）"))
    for k, v in changes.items():
        print(f"     {k}: {v}")
    print(f"   记录编号 #{len(records) - 1}；训练结束用 update --index {len(records) - 1} --effect ... 回填")


def cmd_update(args):
    records = load(args.log)
    idx = args.index + len(records) if args.index < 0 else args.index
    if idx < 0 or idx >= len(records):
        raise SystemExit(f"❌ index {args.index} 越界（共 {len(records)} 条）")
    records[idx]["actual_effect"] = args.effect
    save(args.log, records)
    print(f"✅ 已更新 #{idx} ({records[idx].get('run_dir') or '无 run'}) actual_effect = {args.effect}")


def _infer_log_experiment(records):
    """推断这份日志属于哪个实验：对每条记录的 run_dir 查其所在实验，取众数。

    为什么不能只看最后一条记录: run 目录名跨实验可能重名 —— 实测 rough_go1 里也有 V6/V7，
    于是 exp_of_run('V6') 会解析到 rough_go1，导致本日志误报"别的项目的 run 未记录"，
    甚至可能让人去记录别的项目的 run。
    为什么必须排除 _archive: 已删除的 run 其快照都归档在 logs/_archive/ 下，而历史记录
    大多指向那些已删 run —— 不排除的话众数会落到 _archive。
    """
    from collections import Counter
    c = Counter()
    for r in records:
        e = exp_of_run(r.get("run_dir")) if r.get("run_dir") else None
        if e and e != "_archive":
            c[e] += 1
    return c.most_common(1)[0][0] if c else None


def cmd_show(args):
    records = load(args.log)
    if not records:
        print("（空日志）用 add / run 子命令追加第一条记录")
        return
    # 漏记检查：只看【同一实验、且最后一条记录之后】新产生的 run
    logged = {r.get("run_dir") for r in records if r.get("run_dir")}
    last_logged = next((r.get("run_dir") for r in reversed(records) if r.get("run_dir")), None)
    # ★实验归属必须从【整份记录】推断，不能只看最后一条:
    #   run 目录名在不同实验之间可能重名 —— 实测 DSH 项目里 rough_go1 也有 V6/V7，
    #   于是 exp_of_run('V6') 会解析到 rough_go1，导致本日志误报"rough_go1 的 run 未记录"
    #   （而且会诱导人去记录别的项目的 run）。取众数即可稳定落到本日志真正所属的实验。
    cur_exp = _infer_log_experiment(records) or (exp_of_run(last_logged) if last_logged else None)
    order = [s["run"] for s in find_snapshots(cur_exp)]
    tail = order[order.index(last_logged) + 1:] if last_logged in order else order
    missing = [run for run in tail if run not in logged]

    print(f"{'#':>3} {'时间':<19} {'run':<22} {'诊断':<28} {'改动':>4}  状态")
    print("-" * 108)
    for i, r in enumerate(records):
        diag = (r.get("diagnosis") or "")[:28]
        n = len(r.get("changes") or {})
        n_s = str(n) if n else "无"
        status = "✅ " + (r.get("actual_effect") or "")[:28] if r.get("actual_effect") else "⏳ 待验证"
        ts = (r.get("timestamp") or "")[:19]
        run = (r.get("run_dir") or "")[:22]
        print(f"{i:>3} {ts:<19} {run:<22} {diag:<28} {n_s:>4}  {status}")
    print("-" * 108)
    print(f"共 {len(records)} 条。详情见 {args.log}")
    if missing:
        print(f"\n⚠️ 有 {len(missing)} 个 run 未记录（每次训练都必须记录）：")
        for m in missing:
            print(f"     python tools/tuning_log.py run --run_dir {m}")


def cmd_diff(args):
    all_snaps = find_snapshots()
    snaps = {s["run"]: s for s in all_snaps}
    if args.to not in snaps:
        print(f"❌ 找不到 run: {args.to}（可用的: {list(snaps.keys())}）")
        return
    b = load_snapshot(snaps[args.to]["path"])
    if args.frm:
        if args.frm not in snaps:
            print(f"❌ 找不到 run: {args.frm}")
            return
        a, label = load_snapshot(snaps[args.frm]["path"]), f"{args.frm} → {args.to}"
    else:
        # 只在同一实验内找"上一个 run"
        exp_snaps = find_snapshots(snaps[args.to]["exp"])
        idx = [i for i, s in enumerate(exp_snaps) if s["run"] == args.to][0]
        if idx == 0:
            print(f"（这是 {snaps[args.to]['exp']} 的第一个 run，无前置对比）")
            return
        a = load_snapshot(exp_snaps[idx - 1]["path"])
        label = f"{exp_snaps[idx - 1]['run']} → {args.to}"
    diffs = diff_snapshots(a, b)
    print(f"===== 配置差异: {label} =====")
    for d in diffs:
        print(f"  {d['key']:<44} {d['old']}  ->  {d['new']}")
    if not diffs:
        print("  （无差异）")


def main():
    p = argparse.ArgumentParser(description="LgRobotDog 调参实验日志（OpenDoge tuning_log 增强版）")
    p.add_argument("--log", default=DEFAULT_LOG, help=f"日志文件（默认 {DEFAULT_LOG}）")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="追加调参记录（改配置时用）")
    a.add_argument("--diagnosis", required=True, help="诊断（一句话）")
    a.add_argument("--changes", default="", help="改动列表 'k: 旧 -> 新, k2: ...'（值里的逗号安全）")
    a.add_argument("--changes-json", default="", help="改动字典的 JSON（逗号/冒号歧义时用这个）")
    a.add_argument("--expected", default="", help="预期效果（可留空）")
    a.add_argument("--run_dir", default="", help="对应 run 目录名")
    a.add_argument("--step", type=int, default=None, help="checkpoint 迭代数")
    a.add_argument("--timestamp", default="", help="覆盖时间戳（默认按 run_dir 解析）")
    a.add_argument("--diff", action="store_true", help="自动对比最近两个 run 快照生成 changes")
    a.add_argument("--force", action="store_true", help="允许同一 run_dir 重复记录")
    a.set_defaults(func=cmd_add)

    r = sub.add_parser("run", help="为一次训练 run 建记录（每次训练必做，自动 diff 改动）")
    r.add_argument("--run_dir", default="", help="run 目录名（默认取最新快照）")
    r.add_argument("--diagnosis", default="", help="诊断（默认按有无改动自动生成）")
    r.add_argument("--expected", default="", help="预期效果（可留空）")
    r.add_argument("--changes", default="", help="补充改动（与自动 diff 合并）")
    r.add_argument("--changes-json", default="", help="补充改动的 JSON")
    r.add_argument("--step", type=int, default=None, help="checkpoint 迭代数")
    r.add_argument("--force", action="store_true", help="允许同一 run 重复记录")
    r.set_defaults(func=cmd_run)

    u = sub.add_parser("update", help="回填 actual_effect")
    u.add_argument("--index", type=int, required=True, help="记录编号（-1 = 最后一条）")
    u.add_argument("--effect", required=True)
    u.set_defaults(func=cmd_update)

    s = sub.add_parser("show", help="查看全部记录")
    s.set_defaults(func=cmd_show)

    d = sub.add_parser("diff", help="对比两个 run 的配置快照")
    d.add_argument("--frm", default=None, help="起始 run（默认取 to 的前一个）")
    d.add_argument("--to", required=True, help="目标 run")
    d.set_defaults(func=cmd_diff)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
