#!/usr/bin/env bash
# =============================================================================
# merge_resumed_segment.sh — 把【自动续训新建的那一段目录】并回主目录，只留一个目录。
#
# 背景（为什么需要它）:
#   rsl_rl 的 --resume 语义是【新建 log_dir、只接权重/优化器/迭代数】（V41→V41b 就是这么来的），
#   所以每次机器硬挂 + 自动续训，就会多出一个时间戳目录。一轮训练因此会散成 2~3 个目录：
#     Oct12_22-34-29_...（原始段，model_0 ~ 11800）
#     Oct13_18-15-44_...（续训段，model_12000 ~ ...）
#   用户 2026-10-13 明确表示「续训段那个目录不需要」⇒ 收尾时把它们并成一个。
#
# 做法（只搬文件，不改训练）:
#   把 SEG 段里的 model_*.pt 与 events.out.tfevents.* 移到 MAIN 段目录；
#   同名 checkpoint 冲突时保留 mtime 更新的那份并打警告；然后删除（已空的）SEG 目录。
#   TB 支持一个目录里多个 events 文件 ⇒ 续训段的曲线不会丢。
#
# ⚠️ 绝对不要在训练还在写 SEG 时运行（runner 把 log_dir 记成绝对路径，搬走/删掉会让
#   下一次 torch.save 直接崩）。所以它只应由收尾链在"确认没有训练进程"之后调用。
#
# 用法:
#   bash tools/merge_resumed_segment.sh <主目录名或路径> [续训段目录名或路径...]
#   bash tools/merge_resumed_segment.sh V51 Oct13_18-15-44_LgDog1_himloco   # 指定
#   bash tools/merge_resumed_segment.sh V51 --auto                          # 自动找同一轮的续训段
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="$ROOT/legged_gym/logs/flat_LgDog1"

MAIN="${1:?用法: merge_resumed_segment.sh <主目录> [续训段...|--auto]}"
shift || true
[ -d "$MAIN" ] || MAIN="$RUNS/$MAIN"
[ -d "$MAIN" ] || { echo "❌ 找不到主目录: $MAIN"; exit 1; }

# ---- 门: 有训练在跑就拒绝（搬文件会让它在存 checkpoint 时崩）----
if ps -eo cmd | awk '/train\.py/ && !/awk/' | grep -q .; then
  echo "❌ 有训练进程在跑，拒绝合并（会让它在存 checkpoint 时崩溃）。等训练结束后由收尾链调用。"
  exit 1
fi

# ---- 找续训段 ----
SEGS=()
if [ "${1:-}" = "--auto" ] || [ $# -eq 0 ]; then
  # 主目录里最新 checkpoint 的步数；比它更新的时间戳目录 = 续训段
  MAIN_STEP=$(ls "$MAIN"/model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
  for d in "$RUNS"/Oct*/; do
    d="${d%/}"
    [ "$(cd "$d" && pwd)" = "$(cd "$MAIN" && pwd)" ] && continue
    # 只看"像同一轮"的目录: 与主目录的 config_snapshot.json 内容一致
    if [ -f "$d/config_snapshot.json" ] && [ -f "$MAIN/config_snapshot.json" ] \
       && cmp -s "$d/config_snapshot.json" "$MAIN/config_snapshot.json"; then
      # 且它的 checkpoint（若有）都 ≥ 主目录的最大步数 ⇒ 是续训段
      MINSEG=$(ls "$d"/model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt//' | sort -n | head -1)
      if [ -z "$MINSEG" ] || [ -z "$MAIN_STEP" ] || [ "$MINSEG" -ge "$MAIN_STEP" ]; then
        SEGS+=("$d")
      fi
    fi
  done
else
  for s in "$@"; do
    [ -d "$s" ] || s="$RUNS/$s"
    [ -d "$s" ] && SEGS+=("$s")
  done
fi

if [ ${#SEGS[@]} -eq 0 ]; then
  echo "没有找到需要合并的续训段（主目录: $(basename "$MAIN")）"
  exit 0
fi

echo "主目录: $(basename "$MAIN")"
for s in "${SEGS[@]}"; do
  echo "续训段: $(basename "$s")"
  moved=0; skipped=0
  for f in "$s"/model_*.pt; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    if [ -e "$MAIN/$b" ]; then
      if [ "$f" -nt "$MAIN/$b" ]; then
        echo "  ⚠️ $b 已存在且续训段更新 ⇒ 用续训段的覆盖（旧的那份留在 .merged/）"
        mkdir -p "$MAIN/.merged" && mv "$MAIN/$b" "$MAIN/.merged/$b"
        mv "$f" "$MAIN/$b"; moved=$((moved+1))
      else
        echo "  ⚠️ $b 已存在且主目录更新 ⇒ 跳过（续训段那份丢弃）"
        skipped=$((skipped+1))
      fi
    else
      mv "$f" "$MAIN/$b"; moved=$((moved+1))
    fi
  done
  for f in "$s"/events.out.tfevents.*; do
    [ -e "$f" ] || continue
    mv "$f" "$MAIN/$(basename "$f")"
  done
  echo "  搬走 checkpoint $moved 个（跳过 $skipped 个），events 已并入 ⇒ 删除空目录"
  rmdir "$s" 2>/dev/null || { echo "  ⚠️ 目录非空（有其它文件），保留: $s"; ls -A "$s" | head -5 | sed 's/^/       /'; }
done

echo "✅ 合并完成。$(basename "$MAIN") 现在有 $(ls "$MAIN" | grep -c '^model_') 个 checkpoint。"
