#!/usr/bin/env bash
# =============================================================================
# cleanup_runs.sh —— 清理训练目录里的【临时/冒烟 run】，绝不碰正式轮次与在跑的 run
#
# 为什么需要它：我（助手）做冒烟验证时会起 `--max_iterations 4` 的短跑，它们会生成
# 形如 `Oct08_15-31-33_LgDog1_himloco` 的目录；手工清理时靠"我记得把它改名成 _smoke_*了"，
# 结果 2026-10-08 就漏了一个（15:31 那次没改名 ⇒ 用户发现它还在）。
# 本脚本用【客观判据】而不是记忆来判定"该删"：
#   ① 目录名不是 V<数字> 形式（正式轮次一律已改名为 V*）
#   ② 目录里最大的 model_<step>.pt 步数 < THRESH（默认 200 ⇒ 只训了几轮的冒烟）
#   ③ 目录最后修改时间早于 -AGE 分钟（默认 60 ⇒ 绝不误删【正在跑】的 run）
#   ④ 显式排除当前在跑的 run（从 logs/train_logs/train.pid 反查进程的 run 目录）
#
# 用法:
#   bash tools/cleanup_runs.sh            # 只列出将删的目录（dry-run）
#   bash tools/cleanup_runs.sh --delete   # 真删
#   bash tools/cleanup_runs.sh --delete --thresh 2000 --age 180
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="$ROOT/legged_gym/logs/flat_LgDog1"
DELETE=0; THRESH=200; AGE=60
while [ $# -gt 0 ]; do
  case "$1" in
    --delete) DELETE=1;;
    --thresh) THRESH="${2:-200}"; shift;;
    --age)    AGE="${2:-60}"; shift;;
    *) echo "未知参数: $1"; exit 1;;
  esac
  shift
done

# 当前在跑的 run（若能从 PID 反查出来）
# ★ 2026-10-11 修：以前【无条件】把"最新 events 目录"当在跑的 run 保护起来。机器硬挂后，
#   最新 events 目录恰恰就是已经死掉的那个 run ⇒ 它被永久保护、永远删不掉
#   （实测：486 轮就被硬挂打断的 Oct10_23-05-25 被误判成"在跑"）。
#   现在先确认【真的还有 train.py 进程活着】才启用保护；没有进程就不保护任何目录。
ACTIVE=""
PID="$(cat "$ROOT/logs/train_logs/train.pid" 2>/dev/null || true)"
TRAIN_ALIVE=0
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null \
   && tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | grep -q 'train\.py'; then
  TRAIN_ALIVE=1
  ACTIVE="$(readlink -f "/proc/$PID/cwd" 2>/dev/null | sed 's#.*/scripts$##' || true)"
fi
ACTIVE_NAME=""
if [ "$TRAIN_ALIVE" = 1 ]; then
  ACTIVE_NAME="$(python3 - "$ROOT" <<'PY' 2>/dev/null
import glob, os, sys
root=sys.argv[1]
ev=glob.glob(os.path.join(root,'legged_gym/logs/flat_LgDog1/*/events.out.tfevents.*'))
if not ev: raise SystemExit
best=max(ev, key=os.path.getmtime)
print(os.path.basename(os.path.dirname(best)))
PY
)"
fi

echo "在跑的 run（保护，不删）: ${ACTIVE_NAME:-（没有训练在跑 ⇒ 不保护任何目录）}"
echo "判据: 非 V* 名 + 最大 step < $THRESH + 最后修改早于 $AGE 分钟前"
echo
TOTAL=0
for d in "$RUNS"/*/; do
  name="$(basename "$d")"
  case "$name" in V[0-9]*) continue;; exported) continue;; esac
  [ "$name" = "$ACTIVE_NAME" ] && { echo "  跳过(在跑): $name"; continue; }
  maxstep=$(ls "$d"model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
  [ -z "${maxstep:-}" ] && maxstep=0
  fresh=$(find "$d" -maxdepth 1 -newermt "-$AGE minutes" 2>/dev/null | head -1)
  sz=$(du -sh "$d" 2>/dev/null | cut -f1)
  if [ "$maxstep" -lt "$THRESH" ] && [ -z "$fresh" ]; then
    echo "  将删除: $name  (最大 step=$maxstep, 大小=$sz)"
    TOTAL=$((TOTAL+1))
    [ "$DELETE" = 1 ] && rm -rf "$d"
  else
    echo "  保留  : $name  (最大 step=$maxstep, 大小=$sz)"
  fi
done
echo
if [ "$DELETE" = 1 ]; then echo "已删除 $TOTAL 个临时 run"; else echo "dry-run：以上共 $TOTAL 个将被删除（加 --delete 才会真删）"; fi
