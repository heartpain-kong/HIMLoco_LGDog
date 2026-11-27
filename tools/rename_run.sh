#!/usr/bin/env bash
# =============================================================================
# rename_run.sh —— 把 legged_gym 自动生成的时间戳 run 目录改成本仓库惯例的 V<编号>。
#
# 为什么需要它: 训练日志目录名是 legged_gym 按时间戳自动生成的
#   （如 Oct04_13-45-25_LgDog1_himloco），而本仓库的查阅/引用惯例是 V28/V32/V35…
#   手工改名容易漏（V36 就漏了一次），且改名要同时更新 tuning_log.json 与脚本里的引用。
#
# ⚠️【绝对不要给正在训练的 run 改名】runner 在启动时把 log_dir 记成绝对路径，
#    存 checkpoint 时用 os.path.join(log_dir, 'model_X.pt') ⇒ 目录被改名后 torch.save
#    会直接 No such file or directory 崩在训练中途。本脚本据此强制校验。
#
# 用法:
#   bash tools/rename_run.sh                     # 把【最新】run 改成 V<最大编号+1>
#   bash tools/rename_run.sh Oct04_23-59-13_LgDog1_himloco V37
#   bash tools/rename_run.sh --dry               # 只看会改成什么，不动手
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$ROOT/legged_gym/logs/flat_LgDog1"

DRY=0
if [ "${1:-}" = "--dry" ]; then DRY=1; shift; fi

# ---- 门: 训练在跑就不许改名（见文件头说明）；--dry 不动手，放行 ----
if [ "$DRY" = 0 ] && pgrep -f "train\.py" >/dev/null 2>&1; then
    echo "❌ 有 train.py 在跑，拒绝改名（改名会让它在存 checkpoint 时崩溃）。"
    echo "   等训练结束（或先停训）再跑本脚本。"
    exit 1
fi

SRC="${1:-}"
DST="${2:-}"

# 默认: 取最新 run 目录（按 mtime），目标 = 现有最大 V 编号 + 1
if [ -z "$SRC" ]; then
    SRC=$(ls -t "$LOGS" | grep -v '^exported$' | head -1)
fi
if [ -z "$DST" ]; then
    MAXV=$(ls "$LOGS" | sed -n 's/^V\([0-9]\+\)$/\1/p' | sort -n | tail -1)
    DST="V$(( ${MAXV:-0} + 1 ))"
fi

[ -d "$LOGS/$SRC" ] || { echo "❌ 找不到 run 目录: $LOGS/$SRC"; exit 1; }
[ -e "$LOGS/$DST" ] && { echo "❌ 目标已存在: $LOGS/$DST"; exit 1; }
case "$SRC" in V[0-9]*) echo "ℹ️ 源目录已经是 V 编号形式（$SRC），无需改名"; exit 0;; esac

# 交付物核对: 至少要有 config_snapshot.json（改名前确认目录是完整的 run）
[ -f "$LOGS/$SRC/config_snapshot.json" ] || echo "⚠️ $SRC 下没有 config_snapshot.json（可能不是完整 run）"

NKPT=$(ls "$LOGS/$SRC"/model_*.pt 2>/dev/null | wc -l)
echo "改名: $SRC  →  $DST    (checkpoint $NKPT 个)"
[ "$DRY" = 1 ] && { echo "(dry-run，未执行)"; exit 0; }

mv "$LOGS/$SRC" "$LOGS/$DST" || exit 1

# 同步 tuning_log.json 里对该 run 的引用（只改 run_dir 字段里的目录名）
python3 - "$ROOT" "$SRC" "$DST" <<'PY'
import json, os, sys
root, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
p = os.path.join(root, 'tuning_log.json')
if os.path.exists(p):
    recs = json.load(open(p, encoding='utf-8'))
    n = 0
    for r in recs:
        if src in str(r.get('run_dir', '')):
            r['run_dir'] = str(r['run_dir']).replace(src, dst); n += 1
    if n:
        json.dump(recs, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        open(p, 'a', encoding='utf-8').write('\n')
    print('  tuning_log.json: 更新 %d 条 run_dir 引用' % n)
PY

# 同步 logs/train_logs 下脚本里的字面引用（评测/收尾脚本常把 run 名写死）
# ⚠️ 2026-10-07 事故: 收尾脚本自己的输出（v41_finalize.out）里也有 run 名，
#   `sed -i` 是【换 inode 重写】⇒ 正在往里写的那个进程仍指向旧 inode，
#   之后的所有输出（含"全部完成"横幅）就写进了已被删除的文件 ⇒ 日志尾巴凭空消失。
#   修法: 跳过【调用者 stdout 指向的文件】（/proc/$PPID/fd/1）。
if [ -d "$ROOT/logs/train_logs" ]; then
    CALLER_OUT="$(readlink -f "/proc/$PPID/fd/1" 2>/dev/null || true)"
    HIT=$(grep -rl -- "$SRC" "$ROOT/logs/train_logs" 2>/dev/null | wc -l)
    if [ "$HIT" -gt 0 ]; then
        grep -rl -- "$SRC" "$ROOT/logs/train_logs" 2>/dev/null | while read -r f; do
            if [ -n "$CALLER_OUT" ] && [ "$(readlink -f "$f")" = "$CALLER_OUT" ]; then
                echo "  跳过引用更新（这是调用者的输出文件）: ${f#$ROOT/}"; continue
            fi
            sed -i "s/$SRC/$DST/g" "$f"; echo "  已更新引用: ${f#$ROOT/}"
        done
    fi
fi

echo "✅ 完成。现在可以:  python tools/eval_isaac.py --run $DST --checkpoint <step> ..."
