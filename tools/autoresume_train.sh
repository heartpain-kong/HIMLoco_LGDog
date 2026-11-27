#!/usr/bin/env bash
# =============================================================================
# autoresume_train.sh — 机器重启后【自动接着训】+ 重新挂监控/收尾链
#
# 为什么要它：2026-10-06 ~ 10-11 这台机器硬挂了 8 次（详见 ~/rl/dog/V45/README.md 与
#   修复记录），每次都要人工发现"训练没了"→ 重开。而一轮训练要 24 h，靠人工重开
#   根本跑不完一整轮。本脚本把"重启后继续"变成自动的。
#
# 触发方式：桌面登录时自动跑（~/.config/autostart/lgdog-autoresume.desktop）。
#
# ★ 安全的默认值（避免自作主张）：
#   ① 只有存在标记文件 logs/train_logs/autoresume.enabled 才动作 —— 想关掉就删它；
#   ② 已有 train 进程在跑 ⇒ 什么都不做；
#   ③ 最新 run 的 checkpoint 步数 < MIN_STEP（默认 600）⇒ 认为是刚开/垃圾 run，不续；
#   ④ 已经训到目标轮数 ⇒ 不续（交给收尾链）；
#   ⑤ 续训语义 = rl_rl 的 --resume：**新建**一个 run 目录、只接权重/优化器/迭代数
#      （所以续训段的名字会是新的时间戳目录，跟 V41→V41b 一个道理）。
#
# 用法:
#   bash tools/autoresume_train.sh            # 正常（autostart 调用）
#   bash tools/autoresume_train.sh --dry      # 只看会做什么
#   bash tools/autoresume_train.sh --enable   # 打开自动续训
#   bash tools/autoresume_train.sh --disable  # 关掉自动续训
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGD="$ROOT/logs/train_logs"
RUNS="$ROOT/legged_gym/logs/flat_LgDog1"
FLAG="$LOGD/autoresume.enabled"
OUT="$LOGD/autoresume.out"
MIN_STEP="${MIN_STEP:-600}"
DRY=0

case "${1:-}" in
  --enable)  touch "$FLAG"; echo "✅ 已打开自动续训（$FLAG）"; exit 0 ;;
  --disable) rm -f "$FLAG"; echo "已关闭自动续训（删掉 $FLAG）"; exit 0 ;;
  --dry)     DRY=1 ;;
esac

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT"; }

mkdir -p "$LOGD"
log "──── 开机检查（dry=$DRY）────"

if [ ! -f "$FLAG" ]; then
  log "未启用（没有 $FLAG）⇒ 什么都不做。要启用: bash tools/autoresume_train.sh --enable"
  exit 0
fi

# ① 已经有训练在跑？
if ps -eo cmd | awk '/train\.py/ && !/awk/' | grep -q .; then
  log "已有训练进程在跑 ⇒ 不干预"
  exit 0
fi

# ② 找最新 run（有 events 的目录里 mtime 最新）
RUN=$(ls -dt "$RUNS"/*/ 2>/dev/null | while read -r d; do
        ls "$d"events.out.tfevents.* >/dev/null 2>&1 && { echo "${d%/}"; break; }
      done)
[ -z "$RUN" ] && { log "找不到任何 run 目录 ⇒ 不续"; exit 0; }
NAME=$(basename "$RUN")

# ③ 最新 checkpoint
STEP=$(ls "$RUN"/model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
[ -z "$STEP" ] && { log "$NAME 下没有 checkpoint ⇒ 不续"; exit 0; }

# ④ 目标轮数（从该 run 的 config_snapshot 里取 runner.max_iterations；取不到就用 16000）
MAXIT=$(python3 - "$RUN/config_snapshot.json" <<'PY' 2>/dev/null || echo 16000
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
c=d.get('train_cfg',d)
r=(c.get('runner') or {})
print(int(r.get('max_iterations') or 16000))
PY
)

log "最新 run=$NAME  最新 checkpoint=$STEP  目标轮数=$MAXIT"

if [ "$STEP" -lt "$MIN_STEP" ]; then
  log "checkpoint 只有 $STEP 步（< $MIN_STEP）⇒ 判定为刚开/垃圾 run，不续（人工重开更干净）"
  exit 0
fi
if [ "$STEP" -ge "$MAXIT" ]; then
  log "已训到 $STEP/$MAXIT ⇒ 不需要续训（应交给收尾链）"
  exit 0
fi

log "⇒ 续训: --resume --load_run $NAME --checkpoint $STEP  （目标 $MAXIT 轮）"
if [ "$DRY" = 1 ]; then log "（--dry：不真的启动）"; exit 0; fi

# ⑤ 启动续训（launch_train.sh 自带 90s 启动自检）
bash "$ROOT/tools/launch_train.sh" "$MAXIT" --resume --load_run "$NAME" --checkpoint "$STEP" \
  >> "$LOGD/autoresume_launch.out" 2>&1
sleep 5

# ⑥ 重新挂监控链与收尾链（取最新的一套 v<NN>_watch / v<NN>_finalize 脚本）
WATCH=$(ls -t "$LOGD"/v*_watch.sh 2>/dev/null | head -1)
FINAL=$(ls -t "$LOGD"/v*_finalize.sh 2>/dev/null | head -1)
if [ -n "$WATCH" ]; then
  setsid nohup bash "$WATCH" >> "${WATCH%.sh}.out" 2>&1 < /dev/null &
  log "已重挂监控链: $(basename "$WATCH")"
fi
if [ -n "$FINAL" ]; then
  setsid nohup bash "$FINAL" > "${FINAL%.sh}.out" 2>&1 < /dev/null &
  log "已重挂收尾链: $(basename "$FINAL")"
fi
# ★ 2026-10-11 补：看门狗也要一起重挂（首版漏了 ⇒ 续训后没人盯"降速前兆"）
setsid nohup bash "$ROOT/tools/train_watchdog.sh" > /dev/null 2>&1 < /dev/null &
log "已重挂降速看门狗: train_watchdog.sh"
log "续训完成（PID 见 $LOGD/train.pid）"
