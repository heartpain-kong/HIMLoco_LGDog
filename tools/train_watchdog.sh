#!/usr/bin/env bash
# =============================================================================
# train_watchdog.sh — 训练"降速看门狗"（只预警，不干预）
#
# 为什么要它：2026-10-11 实测出一个【硬挂前兆】：迭代耗时会在挂掉前十几分钟里
#   从正常的 ~5.4 s 单调涨到 16 s（GPU 利用率却只有 20~50%、功耗掉到 33 W），
#   而 CPU/GPU 都没降频、内存也充足。三次硬挂（10-11 00:37 / 03:01 / 04:02）前
#   都是这个形状 ⇒ 把它做成指标，就能在挂之前知道"该保存/该收手了"。
#
# 行为（刻意做成只读，不自动杀训练）：
#   · 每 60 s 读一次 console.log 里最后 5 个 "Iteration time"
#   · 与基线比（默认 5.4 s，可用 BASE 覆盖）：
#       > 2.0× 记一条 WARN；> 2.5× 记一条 ALERT 并写标记文件 SLOWDOWN_ALERT
#   · 同时把黑匣子最后一条采样（温度/功耗/负载）附在日志里，便于事后归因
#   · 训练进程消失就退出
# 产物: logs/train_logs/watchdog.out（追加）、SLOWDOWN_ALERT（有降速时存在）
#
# 用法:  nohup bash tools/train_watchdog.sh &      （或由 autoresume 一起拉起）
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGD="$ROOT/logs/train_logs"
CONSOLE="$LOGD/console.log"
BB="$LOGD/blackbox.log"
OUT="$LOGD/watchdog.out"
ALERT="$LOGD/SLOWDOWN_ALERT"
BASE="${BASE:-5.4}"
WARN_X="${WARN_X:-2.0}"
ALERT_X="${ALERT_X:-2.5}"
PIDFILE="$LOGD/train.pid"

log() { echo "[$(date '+%F %T')] $*" >> "$OUT"; }
log "──── 看门狗启动（基线 ${BASE}s，WARN>${WARN_X}×，ALERT>${ALERT_X}×）────"
rm -f "$ALERT"

RUN_BASE=""
while :; do
  PID="$(cat "$PIDFILE" 2>/dev/null || echo '')"
  if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null \
     || ! tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | grep -q 'train\.py'; then
    log "训练进程已结束 ⇒ 看门狗退出"
    exit 0
  fi

  # ★ 自适应基线：用本 run【最早 5 个】迭代耗时的中位，而不是写死的常量。
  #   为什么：不同轮/不同负载下健康值本身会变（V45 是 5.4 s，本轮是 6.6 s），
  #   写死 5.4 会把正常情况误报成 WARN（首版就是这么误报的）。
  if [ -z "$RUN_BASE" ]; then
    FIRST=$(grep -a "Iteration time" "$CONSOLE" 2>/dev/null | head -5 \
            | sed 's/\x1b\[[0-9;]*m//g' | awk '{gsub(/s$/,"",$3); print $3+0}')
    if [ "$(echo "$FIRST" | grep -c .)" -ge 3 ]; then
      RUN_BASE=$(echo "$FIRST" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
    else
      RUN_BASE="$BASE"
    fi
    log "自适应基线 = ${RUN_BASE}s（本 run 前几个迭代的中位；固定兜底 ${BASE}s）"
  fi

  # 最后 5 个迭代耗时（去掉 ANSI、取秒数）
  LAST=$(grep -a "Iteration time" "$CONSOLE" 2>/dev/null | tail -5 \
         | sed 's/\x1b\[[0-9;]*m//g' | awk '{gsub(/s$/,"",$3); print $3+0}')
  N=$(echo "$LAST" | grep -c . || true)
  if [ "$N" -ge 3 ]; then
    MED=$(echo "$LAST" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
    RATIO=$(awk -v m="$MED" -v b="$RUN_BASE" 'BEGIN{ if (b+0 <= 0) print "0"; else printf "%.3f", m/b }')
    # ★ 判定写成显式 if/else（首版用嵌套三元 + 字符串比较，awk 一失败就返回空串，
    #   结果被当成"非 ok" ⇒ 每次采样都误报 WARN）
    HIGH=$(awk -v r="$RATIO" -v w="$WARN_X" -v a="$ALERT_X" \
           'BEGIN{ r+=0; w+=0; a+=0; if (r >= a) print "A"; else if (r >= w) print "W"; else print "ok" }')
    case "${HIGH:-ok}" in
      A)
        TAIL=$(tail -1 "$BB" 2>/dev/null | sed -E 's/.*(k10temp\/temp1=[0-9.]+C).*(gpu=[0-9]+, [0-9]+, [0-9]+, [0-9]+, [0-9.]+).*(load=[0-9.]+).*/\1 \2 \3/')
        log "ALERT 迭代耗时中位 ${MED}s = ${RATIO}× 基线（基线 ${RUN_BASE}s，最近: $(echo $LAST | tr '\n' ' ')）| $TAIL"
        echo "迭代耗时 ${MED}s = ${RATIO}× 基线（基线 ${RUN_BASE}s，$(date '+%F %T')）" > "$ALERT"
        ;;
      W)
        TAIL=$(tail -1 "$BB" 2>/dev/null | sed -E 's/.*(k10temp\/temp1=[0-9.]+C).*(gpu=[0-9]+, [0-9]+, [0-9]+, [0-9]+, [0-9.]+).*(load=[0-9.]+).*/\1 \2 \3/')
        log "WARN  迭代耗时中位 ${MED}s = ${RATIO}× 基线（基线 ${RUN_BASE}s）| $TAIL"
        ;;
    esac
  fi
  sleep 60
done
