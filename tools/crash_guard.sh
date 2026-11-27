#!/usr/bin/env bash
# =============================================================================
# crash_guard.sh — 硬挂「黑匣子 + 内核日志尾随」常驻器（2026-10-09 新增）
#
# 为什么要有它（血的教训）:
#   机器已经硬挂 4 次（10-06 17:31 / 10-07 21:26 / 10-08 09:05 / 10-09 09:53），
#   每次 journalctl 在崩溃前都【什么都没有】。更糟的是：之前以为黑匣子在跑，
#   实际上 logs/blackbox.log 根本不存在 ⇒ 第 4 次硬挂依然零采样数据。
#   本脚本保证「黑匣子确实在跑」这件事本身可核查（PID 文件 + 心跳行 + status 子命令）。
#
# 用法:
#   bash tools/crash_guard.sh start     # 启动（幂等：已在跑则不动）
#   bash tools/crash_guard.sh status    # 看是否活着 + 最后一条心跳
#   bash tools/crash_guard.sh stop
#
# 产物（都在 <仓库根>/logs/train_logs/）:
#   blackbox.log      每 2s 一条：AC/电池、温度、CPU 频率、GPU 温度/功耗/显存、loadavg、内存
#   kernel_tail.log   journalctl -kf 的落盘（崩溃前的内核消息，若有）
#   crash_guard.pids  bb=<pid> ktail=<pid>
#
# ★ 采样间隔取 2s（不是默认 5s）: 硬挂是"瞬间断电式"的，5s 粒度可能整段错过
#   （实测崩溃都发生在后台维护任务收尾的那 1 秒内）。
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/logs/train_logs"
BB_TOOL="$HOME/rl/LgRobot/sim2real/tools/crash_blackbox.py"
PIDF="$OUT/crash_guard.pids"
mkdir -p "$OUT"

alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

# ★ 2026-10-10 修：`kill -0 <pid>` 只证明"这个号有进程"，不证明"还是我们那个进程"。
#   10-10 02:14 第 5 次硬挂后机器重启，黑匣子进程早没了，但 status 报"黑匣子 pid=3982 活着"
#   —— 因为重启后 3982 被别的进程占用了。这种假活着最危险：下次硬挂又是零数据。
#   判定改进：pid 存在【且】/proc/<pid>/cmdline 里含 blackbox 才认为活着（cmdline 是权威）。
alive_tool() {                    # alive_tool <pid> <cmdline 关键字>
  local pid="${1:-}" key="${2:-}"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -- "$key"
}

start() {
  local bb kt
  bb=$(awk -F= '/^bb=/{print $2}' "$PIDF" 2>/dev/null)
  kt=$(awk -F= '/^ktail=/{print $2}' "$PIDF" 2>/dev/null)

  if alive_tool "$bb" blackbox; then
    echo "黑匣子已在跑: pid=$bb"
  else
    alive "$bb" && echo "⚠️ pid=$bb 存在但不是黑匣子（PID 被复用，多半是重启过）→ 重新启动" 
    [ -f "$BB_TOOL" ] || { echo "❌ 找不到黑匣子脚本: $BB_TOOL"; return 1; }
    setsid nohup python3 "$BB_TOOL" --interval 2 --out "$OUT/blackbox.log" \
        > "$OUT/blackbox_stdout.log" 2>&1 < /dev/null &
    bb=$!
    echo "黑匣子启动: pid=$bb → $OUT/blackbox.log"
  fi

  if alive_tool "$kt" journalctl; then
    echo "内核尾随已在跑: pid=$kt"
  else
    setsid nohup journalctl -kf > "$OUT/kernel_tail.log" 2>&1 < /dev/null &
    kt=$!
    echo "内核尾随启动: pid=$kt → $OUT/kernel_tail.log"
  fi

  printf 'bb=%s\nktail=%s\n' "$bb" "$kt" > "$PIDF"
}

status() {
  local bb kt
  bb=$(awk -F= '/^bb=/{print $2}' "$PIDF" 2>/dev/null)
  kt=$(awk -F= '/^ktail=/{print $2}' "$PIDF" 2>/dev/null)
  if alive_tool "$bb" blackbox; then
    echo "黑匣子 pid=$bb  $(ps -p "$bb" -o etime= 2>/dev/null | tr -d ' ')  活着"
  elif alive "$bb"; then
    echo "黑匣子 pid=$bb  ✗ PID 存在但不是黑匣子（被复用）——需重跑 start"
  else
    echo "黑匣子 pid=${bb:-（无）}  ✗ 死了 —— 需重跑 start"
  fi
  if alive_tool "$kt" journalctl; then
    echo "内核尾随 pid=$kt  活着"
  else
    echo "内核尾随 pid=${kt:-（无）}  ✗ 死了"
  fi
  if [ -f "$OUT/blackbox.log" ]; then
    echo "blackbox.log: $(wc -l < "$OUT/blackbox.log") 行，最后一条："
    tail -1 "$OUT/blackbox.log" | cut -c1-200
  else
    echo "❌ blackbox.log 不存在 —— 黑匣子没在写"
  fi
}

stop() {
  local bb kt
  bb=$(awk -F= '/^bb=/{print $2}' "$PIDF" 2>/dev/null)
  kt=$(awk -F= '/^ktail=/{print $2}' "$PIDF" 2>/dev/null)
  # ★ 不用 pkill/pgrep 匹配脚本名: 命令行里出现同样的字符串会把自己也杀掉（踩过两次）
  [ -n "$bb" ] && kill "$bb" 2>/dev/null && echo "已停黑匣子 $bb"
  [ -n "$kt" ] && kill "$kt" 2>/dev/null && echo "已停内核尾随 $kt"
  rm -f "$PIDF"
}

case "${1:-start}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  *) echo "用法: $0 {start|status|stop}"; exit 2 ;;
esac
