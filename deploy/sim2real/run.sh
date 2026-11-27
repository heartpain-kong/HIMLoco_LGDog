#!/usr/bin/env bash
# LgDog1 sim2real 启动包装：自动用随包的 venv，参数原样透传给 main.py
#
#   ./run.sh                 # 真机跑策略
#   ./run.sh --mode init     # 只起立保持
#   ./run.sh --dry-run       # 干跑（无硬件）
#   ./run.sh --check --live  # 只读预检
#
# 为什么要有这个包装：
#   1) 不用每次记 .venv/bin/python 的路径（移植到新机器最容易踩的坑之一）
#   2) 顺手做点"实时性小优化" —— N100 是 4 核小 CPU，默认调度下 torch 推理
#      可能被抢时间片，控制环的 20ms 节拍就抖了
#   3) venv 不存在时给出明确指引，而不是丢一堆 ImportError
#
# 环境变量：
#   LGDOG_VENV=/opt/lgdog/venv   指定别的虚拟环境
#   LGDOG_PYTHON=/usr/bin/python3 直接指定解释器（开发机上用 conda 环境时更方便）
#   LGDOG_CPU=2                  把控制环钉在 2 号核（留出核给串口线程/系统）
#   LGDOG_NICE=0                 关掉 nice（默认 -5，降优先级抢先，不需 sudo 也能"尽力"）
#   LGDOG_NO_TWEAK=1             完全不做任何调度调整
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${LGDOG_VENV:-$HERE/.venv}"
PY="${LGDOG_PYTHON:-$VENV/bin/python}"

if [[ ! -x "$PY" ]]; then
  echo "找不到解释器: $PY" >&2
  echo "  先跑：$HERE/install.sh" >&2
  echo "  或指定已有的：LGDOG_VENV=/path/to/venv $0 ..." >&2
  echo "                 LGDOG_PYTHON=/path/to/python $0 ..." >&2
  exit 1
fi

# ---- 解释器能力预检（★ 2026-10-08 加）------------------------------------
# "venv 存在" ≠ "它能跑策略"。本树里就出现过：.venv 只装了 mujoco/glfw（是给 MuJoCo
# 工具建的），没有 torch —— 那时 run.sh 会一路跑到 PolicyRunner 才炸，报出来只有一句
# "No module named 'torch'"，看不出是"用错解释器"这个原因（换台机器最容易踩）。
#   ★ 需要什么，取决于要跑什么：
#       --check      静态预检：只要能读配置（yaml/numpy），不加载策略、不开串口
#       --dry-run    假机器人（stub 后端）：要 torch，【不需要】pyserial
#       --mode init  只起立保持：不开策略，但要真串口
#       默认/--mode policy 真机跑策略：torch + pyserial 都要
_need_torch=0; _need_serial=1
_prev=""
for _a in "$@"; do
  case "$_a" in
    --check) _need_torch=0; _need_serial=0 ;;
    --dry-run) _need_torch=1; _need_serial=0 ;;
    --mode=init) _need_torch=0 ;;
    init) [[ "$_prev" == "--mode" ]] && _need_torch=0 ;;
  esac
  _prev="$_a"
done
[[ "$_need_torch" == "1" || "${_prev}" == "--mode" || $# -eq 0 ]] && _need_torch=1
_req="numpy yaml"
[[ "$_need_torch" == "1" ]] && _req="torch $_req"
[[ "$_need_serial" == "1" ]] && _req="$_req serial"
_have_missing=()
for _m in $_req; do
  "$PY" -c "import $_m" 2>/dev/null || _have_missing+=("$_m")
done
if [[ ${#_have_missing[@]} -gt 0 ]]; then
  echo "❌ 解释器缺这个命令需要的模块: ${_have_missing[*]}" >&2
  echo "    解释器: $PY" >&2
  echo "    需要的（本条命令）: $_req" >&2
  if [[ " ${_have_missing[*]} " == *" torch "* ]]; then
    echo "    ⚠️  没有 torch ⇒ 【策略加载不了】。常见原因：" >&2
    echo "        1) 用了给 MuJoCo 工具建的那个 venv（里面只有 mujoco/glfw）" >&2
    echo "        2) 这台机器还没执行过 ./install.sh" >&2
  fi
  if [[ " ${_have_missing[*]} " == *" serial "* ]]; then
    echo "    ⚠️  没有 pyserial ⇒ 连不了下位机（但 --check / --dry-run 不需要它）" >&2
  fi
  echo "    怎么办：" >&2
  echo "      · 目标机(N100)：先 ./install.sh" >&2
  echo "      · 有现成环境：LGDOG_VENV=/path/to/venv $0 ..." >&2
  echo "                    或 LGDOG_PYTHON=/path/to/python $0 ..." >&2
  exit 2
fi


# ---- 调度小优化（都会失败即忽略，不影响启动）----
PREFIX=()
if [[ "${LGDOG_NO_TWEAK:-0}" != "1" ]]; then
  # nice -n -5：不需要 sudo 时会打印一句权限提示后照常运行，无害。
  # ★ 故意不给 -20：控制环独占优先级会把串口收包线程饿死，反而更容易丢帧。
  if [[ "${LGDOG_NICE:-1}" != "0" ]] && command -v nice >/dev/null; then
    PREFIX+=(nice -n -5)
  fi
  if [[ -n "${LGDOG_CPU:-}" ]] && command -v taskset >/dev/null; then
    PREFIX+=(taskset -c "$LGDOG_CPU")
  fi
fi

# 别让 torch 把 N100 的 4 个核全占了（否则串口收包线程会被挤掉）
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

cd "$HERE"
if [[ ${#PREFIX[@]} -gt 0 ]]; then
  exec "${PREFIX[@]}" "$PY" main.py "$@"
else
  exec "$PY" main.py "$@"
fi
