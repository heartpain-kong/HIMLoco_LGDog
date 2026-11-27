#!/bin/bash
# =============================================================================
# launch_train.sh — 训练统一启动器
#   把「启动训练 + 记录 PID + 落盘日志 + 启动自检」固化成一个脚本，
#   杜绝手工 setsid nohup 时 PID 文件/日志路径写错的问题
#   （2026-10-02 事故：PID 写到 legged_gym/logs/train_logs/ 而守护脚本从
#     仓库根 logs/train_logs/ 读 ⇒ 收尾链提前触发，与训练抢 GPU ⇒ Xid 31 ⇒
#     GPU 进入"需重启恢复"状态）。
#
# 用法:
#   bash tools/launch_train.sh [max_iterations] [额外 train.py 参数...]
#   例: bash tools/launch_train.sh 16000
#       bash tools/launch_train.sh 16000 --resume --load_run V28 --checkpoint 2000
#
# 约定（所有脚本统一遵守，勿改路径）:
#   运行时目录 = <仓库根>/logs/train_logs/
#     train.pid     训练进程 PID（launch 写入，watch/finalize 读取）
#     console.log   训练控制台输出
#   ⚠️ legged_gym/logs/train_logs/ 是历史遗留位置，不要再往那里写。
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNLOG="$ROOT/logs/train_logs"
mkdir -p "$RUNLOG"

MAXIT="${1:-16000}"
shift 2>/dev/null || true

# 训练进程必须真正脱离会话（setsid），否则工具/终端退出会带走它
cd "$ROOT/legged_gym/legged_gym/scripts"
export PATH="$HOME/miniconda3/envs/himloco/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/himloco/lib:${LD_LIBRARY_PATH:-}"

setsid nohup python train.py --task LgDog1_flat --headless \
    --num_envs 3072 --max_iterations "$MAXIT" "$@" \
    > "$RUNLOG/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$RUNLOG/train.pid"

# 启动自检: 90s 内必须还活着并打出第一行迭代，否则报错退出（不留下"假启动"）
sleep 90
if ! kill -0 "$PID" 2>/dev/null; then
    echo "❌ 启动失败（进程已退出），日志尾部:"
    tail -20 "$RUNLOG/console.log"
    exit 1
fi
if ! grep -q "Learning iteration" "$RUNLOG/console.log"; then
    echo "❌ 90s 未见 Learning iteration（可能卡在环境创建），日志尾部:"
    tail -20 "$RUNLOG/console.log"
    exit 1
fi

RUN=$(ls -dt "$ROOT"/logs/flat_LgDog1/*/ 2>/dev/null | head -1)
echo "✅ 训练已启动"
echo "   PID      : $PID （记录在 $RUNLOG/train.pid）"
echo "   run 目录 : $RUN"
echo "   控制台   : $RUNLOG/console.log"
echo "   监控/收尾: logs/train_logs/vXX_watch.sh 与 vXX_finalize.sh（下一轮编号时复制改名即可）"
