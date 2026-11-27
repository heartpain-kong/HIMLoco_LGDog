#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# auto_sim2sim_check.sh —— 等训练到达指定 checkpoint，自动做一次量化姿态检测。
#
# 为什么需要它：训练要跑几小时，而"平地行走是否抬头"的判据必须用 sim2sim 实测
#   （MuJoCo 侧，57× 实时，10s 墙钟就能跑 600s 仿真）。挂上本脚本后无人值守即可拿到数据。
#
# 用法（必须脱离会话跑，否则会话结束就被杀）:
#   nohup bash tools/auto_sim2sim_check.sh <run目录名> <step> [x_vel] [日志] &
#   nohup bash tools/auto_sim2sim_check.sh <run目录名> 3000 1.0 /tmp/vXX_auto.log &
#
# 流程: 等 model_<step>.pt 出现 -> export_checkpoint.py 导出(含归一化统计量还原)
#       -> sim2sim 平地直行 -> tools/analyze_attitude.py 量化 pitch/roll/站高 -> 写入日志
# ─────────────────────────────────────────────────────────────────────────────
set -u
ROOT=/home/heartpain/rl/LgRobotDog
RUN_NAME=${1:?需要 run 目录名}
STEP=${2:?需要 checkpoint 步数}
XVEL=${3:-1.0}
LOG=${4:-/tmp/auto_sim2sim_check.log}

export PATH="$HOME/miniconda3/envs/himloco/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/himloco/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
PY="$HOME/miniconda3/envs/himloco/bin/python"
RUN="$ROOT/legged_gym/logs/flat_LgDog1/$RUN_NAME"
POL="$ROOT/legged_gym/logs/flat_LgDog1/exported/policies"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=============================================================="
log "自动检测启动: run=$RUN_NAME step=$STEP x_vel=$XVEL"
log "=============================================================="

# ── 1. 等 checkpoint 落盘（每 60s 查一次）──
while [ ! -f "$RUN/model_$STEP.pt" ]; do
  if ! pgrep -f "train.py --task LgDog1" > /dev/null; then
    # 训练进程不在了且 ckpt 还没出现 -> 取现有最新 ckpt 兜底(避免白等)
    LATEST=$(ls -v "$RUN"/model_*.pt 2>/dev/null | tail -1)
    if [ -n "$LATEST" ]; then
      STEP=$(basename "$LATEST" | sed 's/model_//; s/\.pt//')
      log "⚠️ 训练进程已不在且无 model_$STEP.pt -> 改用最新 ckpt: step=$STEP"
      break
    fi
    log "⚠️ 训练进程已不在且目录里没有任何 ckpt，退出"
    exit 1
  fi
  sleep 60
done
log "✅ checkpoint 就绪: model_$STEP.pt（等待 5s 确保写盘完成）"
sleep 5

# ── 2. 导出 JIT（含 obs 归一化统计量还原）──
log "--- 导出 ---"
# --no-policy-pt: 不覆盖用户可能在用的 policy.pt（本脚本用自己导出的那份跑 sim2sim）
EXPORTED="$POL/policy_auto_${STEP}.pt"
"$PY" "$ROOT/tools/export_checkpoint.py" "$RUN" "$STEP" "policy_auto_${STEP}.pt" --no-policy-pt 2>&1 | tee -a "$LOG" | tail -6
if [ ! -f "$EXPORTED" ]; then log "❌ 导出失败，退出"; exit 1; fi

# ── 3. sim2sim 平地直行 600s（无渲染，CPU，57× 实时）──
log "--- sim2sim 平地直行 600s @ ${XVEL} m/s ---"
cd "$ROOT/legged_gym/legged_gym/scripts" || exit 1
timeout 300 "$PY" sim2sim.py --snapshot "$RUN/config_snapshot.json" \
  --policy "$EXPORTED" --no-render --no-keyboard --gamepad off \
  --x_vel "$XVEL" --clamp-vel 2>&1 | grep -E "^\[ *(0|400|800|1200|1600|2000|2400)\]|对齐|归一化|性能|错误|Error" | tee -a "$LOG"

# ── 4. 量化姿态 ──
REC=$(ls -t "$ROOT/legged_gym/legged_gym/scripts/recorded_io"/*.npz 2>/dev/null | head -1)
if [ -n "$REC" ]; then
  log "--- 姿态量化 ($REC) ---"
  "$PY" "$ROOT/tools/analyze_attitude.py" "$REC" 2>&1 | tee -a "$LOG"
  log "【对照基准】V16 行走 pitch=+12.81°(z=0.267) | V17@3600 pitch=+7.56°(z=0.210) | 本轮判据: pitch≤±4° 且 z≥0.24"
else
  log "❌ 没找到录制文件"
fi
log "自动检测结束"
