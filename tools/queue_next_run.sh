#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 排队接力：等指定的训练进程结束 -> 自动启动下一轮训练 -> 自动建 tuning_log 记录。
#
# 用法:
#   nohup bash tools/queue_next_run.sh <被等的PID> <新轮次的日志名> [诊断] [预期] &
#
# 为什么要脱会话跑：V13 到 10000 轮还有约 10.5 小时，普通前台/会话内任务活不到那时候。
# 为什么必须自动建记录：用户的硬规矩是「每一次训练都必须记录到 tuning_log」，
#   无人值守启动的 run 没人来记，就等于违反规矩 —— 所以启动后自动调用
#   tools/tuning_log.py run（它会自动 diff 上一个 run 的快照，把改动写进记录）。
#
# 注意：本脚本【不会】自动杀训练。崩解只写醒目告警，由人决定是否中止。
# ─────────────────────────────────────────────────────────────────────────────
set -u

ROOT=/home/heartpain/rl/LgRobotDog
WATCH_PID=${1:?需要被等待的 PID}
NEXT_LOG=${2:-/tmp/lgdog1_next.log}
DIAGNOSIS=${3:-"【V14】防蹦轮: rewards.scales.lin_vel_z -3.0 -> -6.0"}
EXPECTED=${4:-"E[v_z^2]=|rew_lin_vel_z|/6.0 明显下降(v_z rms 降 29% 以上)，且 mean_episode_length 不塌、tracking 不掉到 0.5 以下"}

cd "$ROOT" || exit 1
export PATH="$HOME/miniconda3/envs/himloco/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/himloco/lib:${LD_LIBRARY_PATH:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QLOG="$SCRIPT_DIR/_queue_next_run.log"
PY="$HOME/miniconda3/envs/himloco/bin/python"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }

log "=============================================================="
log "排队启动：等待 PID $WATCH_PID 结束；新轮次日志 -> $NEXT_LOG"
log "=============================================================="

# ── 1. 等被等的训练结束 ──────────────────────────────────────────────────────
while kill -0 "$WATCH_PID" 2>/dev/null; do sleep 60; done
log "PID $WATCH_PID 已结束。"

# ── 2. 记录被等的那个 run 的最终迭代数 ──────────────────────────────────────
PREV_RUN=$(ls -1dt legged_gym/logs/flat_LgDog1/*/ 2>/dev/null | head -1)
PREV_STEP=$("$PY" - "$PREV_RUN" <<'EOF' 2>/dev/null
import glob, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ev = glob.glob(sys.argv[1] + 'events.out.tfevents.*')
if ev:
    import os
    ea = EventAccumulator(max(ev, key=os.path.getsize), size_guidance={'scalars': 0}); ea.Reload()
    s = ea.Scalars('Train/mean_reward')
    print(s[-1].step)
EOF
)
log "上一个 run: $(basename "$PREV_RUN")  最终迭代 ${PREV_STEP:-未知}"

# ── 3. 等 GPU 释放（最多 30 分钟）─────────────────────────────────────────────
for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)
    if [ "${used:-9999}" -lt 1500 ]; then
        log "GPU 已释放（占用 ${used} MiB），可以启动。"
        break
    fi
    [ $((i % 5)) -eq 0 ] && log "GPU 仍占用 ${used} MiB，继续等待…"
    sleep 30
done

# ── 4. 落盘前校验：本轮要测的改动必须在文件里 ────────────────────────────────
CFG=legged_gym/legged_gym/envs/LgDog1/LgDog1_config.py
if grep -qE '^\s*lin_vel_z\s*=\s*-6\.0' "$CFG"; then
    log "✅ 配置校验通过: lin_vel_z = -6.0（防蹦轮的唯一变量）"
else
    log "❌ 配置校验失败: $CFG 里没有找到 'lin_vel_z = -6.0'！"
    log "   已中止启动，避免跑出一轮无法归因的训练。请人工确认后再启动。"
    exit 1
fi

# ── 5. 启动训练 ─────────────────────────────────────────────────────────────
: > "$NEXT_LOG"
cd legged_gym/legged_gym/scripts || exit 1
nohup "$PY" train.py --task LgDog1 --headless >> "$NEXT_LOG" 2>&1 &
NEW_PID=$!
log "已启动训练 PID=$NEW_PID（日志 $NEXT_LOG），防蹦轮 = V14"

# ── 6. 等新 run 的快照出现，然后自动建 tuning_log 记录 ──────────────────────
NEW_RUN=""
for i in $(seq 1 60); do
    sleep 20
    cand=$(ls -1dt "$ROOT"/legged_gym/logs/flat_LgDog1/*/ 2>/dev/null | head -1)
    if [ -f "${cand}config_snapshot.json" ] && [ "$cand" != "$PREV_RUN" ]; then
        NEW_RUN="$cand"; break
    fi
done

if [ -n "$NEW_RUN" ]; then
    log "新 run 快照已生成: $(basename "$NEW_RUN")"
    cd "$ROOT" || exit 1
    "$PY" tools/tuning_log.py run --run_dir "$(basename "$NEW_RUN")" \
        --diagnosis "$DIAGNOSIS" --expected "$EXPECTED" >> "$QLOG" 2>&1 \
        && log "✅ 已自动建立 tuning_log 记录" \
        || log "⚠️ tuning_log 自动记录失败，请人工补记（run_dir=$(basename "$NEW_RUN")）"
else
    log "⚠️ 20 分钟内没等到新 run 的快照，请人工确认启动是否成功并补记 tuning_log"
fi

# ── 7. 之后每 10 分钟跑一次哨兵，把判定写进日志（不自动杀）──────────────────
COLLAPSE_STREAK=0
while kill -0 "$NEW_PID" 2>/dev/null; do
    sleep 600
    LINE=$(timeout 180 "$PY" tools/watch_train.py --run "$(basename "$NEW_RUN")" 2>/dev/null | head -1)
    log "哨兵: ${LINE:-（无输出）}"
    case "$LINE" in
        *COLLAPSE*) COLLAPSE_STREAK=$((COLLAPSE_STREAK + 1)) ;;
        *)          COLLAPSE_STREAK=0 ;;
    esac
    if [ "$COLLAPSE_STREAK" -ge 2 ]; then
        log "🚨🚨 连续 ${COLLAPSE_STREAK} 次判为 COLLAPSE —— 需要人工介入决定是否中止（本脚本不自动杀）"
    fi
done
log "训练 PID $NEW_PID 已结束。本轮（$(basename "$NEW_RUN")）请回填 tuning_log 的 actual_effect。"
