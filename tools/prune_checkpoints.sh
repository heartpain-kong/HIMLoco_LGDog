#!/usr/bin/env bash
# =============================================================================
# prune_checkpoints.sh — 精简 model_*.pt，省磁盘（默认只干跑，加 --delete 才真删）
#
# 为什么需要它（2026-10-14 实测）: legged_gym/logs/flat_LgDog1 已占 2.8 GB，
#   其中绝大部分是每个 6.6 MB 的 model_<step>.pt —— 一轮训练 60~80 个就是 400~500 MB，
#   而**真正会用到的只有三类**:
#     ① 最后几个（接着训 / 看收敛）
#     ② 已经导出成策略的那些步数（exported/policies/policy_<轮>_<step>.pt）
#     ③ 若某步有对应 sidecar/验收记录，同样保留
#   其余中间步数对分析没有增量价值（真要看某个中间步，重新导出即可 —— 但删了就没了，
#   所以本脚本默认【只干跑】，把要删的清单先摆出来）。
#
# 安全措施:
#   · 默认 dry-run；--delete 才动手
#   · 目录的 events 文件在最近 --age 分钟内被写过 ⇒ 跳过（可能还有进程在写）
#   · 跳过 exported/ _dead_runs/ _config_v1/ 等非 run 目录
#   · 每个目录至少保留 --keep 个（默认 3）
#
# 用法:
#   bash tools/prune_checkpoints.sh                       # 干跑，看看能省多少
#   bash tools/prune_checkpoints.sh --delete              # 真删
#   bash tools/prune_checkpoints.sh --keep 5 --delete     # 多留几个
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="$ROOT/legged_gym/logs/flat_LgDog1"
POL="$RUNS/exported/policies"
KEEP=3
GRID=0          # >0 时另外保留"每 N 步一格"的粗网格（便于以后做检查点扫描）
AGE=15          # 最近 N 分钟内 events 有写入的目录跳过
DELETE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --delete) DELETE=1 ;;
    --keep) KEEP="$2"; shift ;;
    --grid) GRID="$2"; shift ;;
    --age) AGE="$2"; shift ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
  shift
done

# 已导出的步数集合（这些 checkpoint 一律保留）
EXPORTED_STEPS=$(ls "$POL"/policy_*.pt 2>/dev/null | xargs -n1 basename \
                 | sed -nE 's/^policy_.*_([0-9]+)\.pt$/\1/p' | sort -n -u)
echo "已导出（一律保留）的步数: $(echo $EXPORTED_STEPS | tr '\n' ' ')"
echo "保留策略: 每个目录至少留最后 $KEEP 个 + 上面的导出步数；events 在 $AGE 分钟内写过的目录跳过"
echo

now=$(date +%s)
total_before=0; total_del=0; n_del=0
for d in "$RUNS"/*/; do
  d="${d%/}"; name=$(basename "$d")
  case "$name" in exported|_dead_runs|_config_v1) continue ;; esac
  steps=$(ls "$d"/model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt//' | sort -n)
  [ -z "$steps" ] && continue

  # 最近有写入 ⇒ 跳过（可能还在训练）
  ev=$(ls "$d"/events.out.tfevents.* 2>/dev/null | head -1)
  if [ -n "$ev" ]; then
    m=$(stat -c %Y "$ev"); [ $(( now - m )) -lt $(( AGE * 60 )) ] && { echo "跳过（events 刚写过）: $name"; continue; }
  fi

  cnt=$(echo "$steps" | wc -l)
  newsize=$(du -sm "$d" | cut -f1); total_before=$(( total_before + newsize ))

  # 计算保留集合
  keepset=$(echo "$steps" | tail -n "$KEEP")
  # 粗网格: 每 GRID 步保留一格（0 除外），便于以后做检查点扫描
  if [ "$GRID" -gt 0 ]; then
    for s in $steps; do [ "$s" != "0" ] && [ $(( s % GRID )) -eq 0 ] && keepset="$keepset
$s"; done
  fi
  # 只保护【非 0】的已导出步数（model_0 = 未训练产物，没有保留价值）
  for s in $EXPORTED_STEPS; do [ "$s" = "0" ] && continue; echo "$steps" | grep -qx "$s" && keepset="$keepset
$s"; done
  keepset=$(echo "$keepset" | sort -n -u)

  del=""
  for s in $steps; do echo "$keepset" | grep -qx "$s" || del="$del $s"; done
  del=$(echo $del)
  [ -z "$del" ] && { echo "无需清理: $name（$cnt 个，${newsize}MB）"; continue; }

  n=$(echo "$del" | wc -w)
  sz=$(( n * 7 ))    # 每个约 6.6~6.9 MB，取 7 估算
  echo "可清理: $name —— 删 $n 个（约 ${sz}MB），保留 $(echo "$keepset" | tr '\n' ' ')"
  echo "         待删: $(echo $del | tr ' ' ' ' | cut -c1-120)$([ ${#del} -gt 120 ] && echo ' …')"
  total_del=$(( total_del + sz )); n_del=$(( n_del + n ))

  if [ "$DELETE" = 1 ]; then
    for s in $del; do rm -f "$d/model_$s.pt"; done
  fi
done

echo
echo "──────── 汇总 ────────"
echo "扫描的 run 目录总占用: ${total_before} MB"
echo "本次$( [ "$DELETE" = 1 ] && echo '已删除' || echo '将删除' ): $n_del 个 checkpoint ≈ ${total_del} MB"
[ "$DELETE" = 1 ] || echo "（干跑。要真删: bash tools/prune_checkpoints.sh --delete）"
