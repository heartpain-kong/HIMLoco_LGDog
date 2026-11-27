#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# gpu_procs.sh —— 列出/清理【占着 GPU 做计算】的进程（不会碰桌面显示客户端）。
#
# 为什么需要它（这两个坑都实际踩过，代价是白跑一次测量 + 一次 CUDA OOM）:
#   ① 用 `ps -o comm` / `pgrep -f python` 过滤【不可靠】: Isaac Gym 的 play.py
#      进程名是 `pt_main_thread`（不是 python），按 comm 过滤永远匹配不到，
#      于是"我以为它已经死了"，它还占着 4.9GB，下一次 GPU 测量直接 CUDA OOM。
#   ② `pgrep -f play.py` / `pkill -f play.py` 会【连自己的 shell 一起杀】——
#      因为发起命令的那个 bash，其命令行里就含 "play.py" 这个字符串。
#
# 判定口径（分开看，绝不混起来杀）:
#   · compute  = nvidia-smi --query-compute-apps 报出来的，即真正分配了显存的 CUDA 进程。
#                ★只有这一类会被 --kill 杀掉。
#   · display  = /proc/<pid>/fd 里持有 /dev/nvidia* 但不在 compute 列表里的，
#                那是 gnome-shell / VS Code / 浏览器之类的显示客户端，杀它们会搞坏桌面。
#                这里只列出提示，永不杀。
#   同时排除本 shell 及其全部祖先进程。
#
# 用法:
#   bash tools/gpu_procs.sh          # 只列
#   bash tools/gpu_procs.sh --kill   # 对 compute 进程先 SIGTERM，5 秒后仍活着再 SIGKILL
# ─────────────────────────────────────────────────────────────────────────────
set -u

KILL=0
[ "${1:-}" = "--kill" ] && KILL=1

# 自己 + 所有祖先进程（避免误杀调用者）
self_chain=""
p=$$
while [ -n "$p" ] && [ "$p" != "0" ] && [ "$p" != "1" ]; do
    self_chain="$self_chain $p"
    p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null || echo "")
done

# 用普通字符串存 "pid|显存" 行，避免 set -u 下空关联数组报"未绑定变量"
compute_list=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -vE '^(pid|\[Not Supported\]|)$' || true)
compute_extra=""
_raw_pids=$(printf '%s\n' "$compute_list" | cut -d, -f1 | grep -E '^[0-9]+$' || true)
# ★关键修正: nvidia-smi 报出来的 PID 有时是【线程 ID】而不是进程 ID（Isaac Gym 的 play.py 实测如此:
#   报 76567，真身却是 76715，且 76567 在进程表里根本不存在）。若直接用这个号去杀，
#   杀不到真身、显存不释放，还会把真身误判成"显示客户端"而拒绝处理。
#   所以这里把每个号做一次 TID→所属进程 的映射。
compute_pids=""
for _p in $_raw_pids; do
    if [ -d "/proc/$_p" ]; then
        _owner=$_p
    else
        _owner=""
        for _d in /proc/[0-9]*; do
            if [ -d "$_d/task/$_p" ]; then _owner=$(basename "$_d"); break; fi
        done
    fi
    [ -n "$_owner" ] && compute_pids="$compute_pids $_owner"
done
compute_pids=$(printf '%s\n' $compute_pids | sort -un | tr '\n' ' ')

display_lines=""
compute_extra=""
for d in /proc/[0-9]*; do
    pid=$(basename "$d")
    case "$self_chain" in *" $pid "*) continue ;; esac
    printf '%s\n' "$compute_pids" | grep -qx "$pid" && continue
    if ls -l "$d/fd" 2>/dev/null | grep -q "nvidia"; then
        # ★兜底判据: nvidia-smi 报的 PID 有时是【已退出线程的 TID】（实测 play.py: 报 76567，
        #   真身 76715，TID 映射也找不到属主）。这种情况下按 cmdline 判:
        #   我们所有 GPU 作业都是 python（train/play/measure_stand/sim2sim），
        #   而桌面客户端不是 —— 故"cmdline 含 python 的 nvidia 句柄持有者"一律当 compute 处理，
        #   否则真身会被误判成"显示客户端"而拒绝清理（这正是本脚本第一版踩的坑）。
        _c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
        case "$_c" in
            *python*|*isaacgym*|*mujoco*)
                compute_extra="$compute_extra $pid"
                continue ;;      # 归入 compute 就不再列进"显示客户端"，避免两边重复显示
        esac
        display_lines="$display_lines  PID $pid    $(cat "$d/comm" 2>/dev/null || echo '?')\n"
    fi
done

compute_pids=$(printf '%s\n%s\n' "$compute_pids" "$compute_extra" | grep -E '^[0-9]+$' | sort -un | tr '\n' ' ')

echo "== 计算进程（CUDA 占显存；--kill 只杀这一类）=="
if [ -z "$compute_pids" ]; then
    echo "  （无）"
else
    while IFS=, read -r pid mem; do
        pid=$(echo "$pid" | tr -d ' ')
        case "$pid" in ''|pid) continue ;; esac
        [ -d "/proc/$pid" ] || pid=$(for _d in /proc/[0-9]*; do [ -d "$_d/task/$pid" ] && basename "$_d" && break; done)
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-88)
        printf '  PID %-7s %-12s %s\n' "$pid" "$(echo "$mem" | tr -d ' ')" "${cmd:-(cmdline 不可读)}"
    done <<< "$compute_list"
fi

echo
echo "== 显示客户端（持有 nvidia 句柄但不做计算；不会被杀）=="
if [ -z "$display_lines" ]; then
    echo "  （无）"
else
    printf "$display_lines"
fi

if [ "$KILL" = "1" ]; then
    echo
    if [ -z "$compute_pids" ]; then
        echo "没有计算进程需要清理。"
    else
        for pid in $compute_pids; do
            kill -TERM "$pid" 2>/dev/null && echo "  → SIGTERM PID $pid"
        done
        sleep 5
        for pid in $compute_pids; do
            if [ -d "/proc/$pid" ]; then
                kill -9 "$pid" 2>/dev/null && echo "  强杀 PID $pid"
            fi
        done
        sleep 3
        echo "清理后: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
        echo "残余计算进程: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
    fi
fi

echo
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader)"
