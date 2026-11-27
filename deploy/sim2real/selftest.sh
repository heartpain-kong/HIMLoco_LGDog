#!/usr/bin/env bash
# =============================================================================
# 一键自测：把所有模块的自测串一遍 + 预检 + 干跑
#
#   ./selftest.sh            # 全部（含干跑 8 秒）
#   ./selftest.sh --quick    # 跳过时间长的（干跑/imu_check 数值段）
#
# ★ 移植到新机器后的第一件事就是跑它。全绿说明：
#     代码没缺件、numpy/torch/yaml 版本兼容、策略能加载、整条控制链能跑通、
#     安全链在故障注入下确实会跳闸。
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 解释器挑选顺序：LGDOG_VENV > LGDOG_PYTHON > ./.venv > python3
# ★ 开发机上通常没有 .venv（用的是 conda 环境），所以留 LGDOG_PYTHON 这个口子：
#     LGDOG_PYTHON=~/miniconda3/envs/himloco/bin/python ./selftest.sh
PY=""
for cand in "${LGDOG_VENV:-}/bin/python" "${LGDOG_PYTHON:-}" "$HERE/.venv/bin/python"; do
  [[ -n "$cand" && -x "$cand" ]] && { PY="$cand"; break; }
done
if [[ -z "$PY" ]]; then
  PY="$(command -v python3)"
  echo "⚠️  没找到 .venv（也没给 LGDOG_VENV/LGDOG_PYTHON），用 $PY"
  if ! "$PY" -c "import torch" 2>/dev/null; then
    echo "⚠️  这个解释器没有 torch —— 策略相关的自测会失败。"
    echo "    目标机(N100)：先跑 ./install.sh"
    echo "    开发机：LGDOG_PYTHON=<你的训练环境>/bin/python ./selftest.sh"
  fi
fi

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

PASS=0; FAIL=0; FAILED=()
run() {
  local label="$1"; shift
  printf '\n\033[1;36m──── %s\033[0m\n' "$label"
  if "$@" >/tmp/lgdog_selftest_out.$$ 2>&1; then
    tail -3 /tmp/lgdog_selftest_out.$$ | sed 's/^/    /'
    printf '    \033[1;32m✓ %s\033[0m\n' "$label"
    PASS=$((PASS+1))
  else
    sed 's/^/    /' /tmp/lgdog_selftest_out.$$ | tail -25
    printf '    \033[1;31m✗ %s\033[0m\n' "$label"
    FAIL=$((FAIL+1)); FAILED+=("$label")
  fi
  rm -f /tmp/lgdog_selftest_out.$$
}

cd "$HERE"
echo "==============================================================================="
echo "LgDog1 sim2real 自测   python=$PY"
"$PY" -c "import sys;print('  ', sys.version.split()[0], sys.executable)"
echo "==============================================================================="

# ---- 1) 常量与数学 ----
run "spec.py（训练接口常量）"            "$PY" spec.py
run "tools/math_utils.py（四元数/滤波）" "$PY" tools/math_utils.py

# ---- 2) 接口层 ----
run "interface/host_link.py（协议编解码）" "$PY" interface/host_link.py
run "interface/imu_mount.py（安装变换）"   "$PY" interface/imu_mount.py
run "interface/motor_mapping.py（映射表）" "$PY" interface/motor_mapping.py
run "interface/teleop.py（手柄/键盘后端）" "$PY" interface/teleop.py

# ---- 3) 安全与启动 ----
run "safety/runtime_guard.py"              "$PY" safety/runtime_guard.py
run "safety/safety_monitor.py"             "$PY" safety/safety_monitor.py
run "startup/pose_initializer.py（起立流程）" "$PY" startup/pose_initializer.py
run "startup/stand_balance.py"             "$PY" startup/stand_balance.py

# ---- 4) 策略与后端 ----
run "policy/policy_runner.py（策略推理）"   "$PY" policy/policy_runner.py
run "backends/stub_backend.py（干跑后端）"  "$PY" backends/stub_backend.py
run "backends/host_link_backend.py（真机后端）" "$PY" backends/host_link_backend.py

# ---- 5) 日志与协议往返 ----
run "tools/logger.py（日志会话）"           "$PY" tools/logger.py
# 这个自测会现场用 gcc 编译【固件的】lgdog_link.c 做逐字节对照。需要三样：
#   tools/link_test_harness.c（随包）+ gcc + 固件源码 lgdog_link.c/.h
# 固件源码不在这个目录里，用 LGDOG_FIRMWARE_DIR 指，或者把它俩拷进 tools/firmware/。
_fw="${LGDOG_FIRMWARE_DIR:-$HERE/tools/firmware}"
if [[ -f "$_fw/lgdog_link.c" ]] && command -v gcc >/dev/null && [[ -f tools/link_test_harness.c ]]; then
  run "tools/test_link_protocol.py（PC↔固件逐字节对齐）" \
      env LGDOG_FIRMWARE_DIR="$_fw" "$PY" tools/test_link_protocol.py
else
  printf '\n\033[1;33m──── 跳过 tools/test_link_protocol.py\033[0m\n'
  echo "    需要 gcc + tools/link_test_harness.c + 固件的 lgdog_link.c/.h"
  echo "    想让它也跑：把 lgdog_link.c 与 lgdog_link.h 拷到 tools/firmware/（两个小文件）"
  echo "      mkdir -p tools/firmware && cp <dm02>/User_File/task/lgdog_link.[ch] tools/firmware/"
fi
# check_alignment 是【开发机专用】：它把训练仓库的 sim2sim.py 当模块导入来逐项对拍，
# 需要 mujoco + 那份 sim2sim.py。目标机(N100)上这两样都没有 → 自动跳过而不是报失败。
# 判据要同时满足三样，缺一个就跳过（否则在只有本目录的机器上会假失败）：
#   check_alignment.py 存在 + 有 mujoco + 能找到训练仓库的 sim2sim.py
# sim2sim.py 的位置与 check_alignment.py 里的 _REPO/legged_gym/... 一致
_sim2sim="$HERE/../../legged_gym/legged_gym/scripts/sim2sim.py"
if [[ -f tools/check_alignment.py ]] && "$PY" -c "import mujoco" 2>/dev/null \
   && [[ -f "$_sim2sim" ]]; then
  run "tools/check_alignment.py（与 sim2sim 对齐，开发机专用）" \
      "$PY" tools/check_alignment.py --skip-numerics
else
  printf '\n\033[1;33m──── 跳过 tools/check_alignment.py（开发机专用）\033[0m\n'
  echo "    它要把训练仓库的 sim2sim.py 当模块导入来逐项对拍，需要：mujoco +"
  echo "    <LgRobotDog>/legged_gym/legged_gym/scripts/sim2sim.py"
  echo "    目标机(N100)两样都没有，跳过是正常的 —— 改过 spec.py 的常量时请在开发机上跑一次。"
fi

# ---- 6) 预检 ----
# 分两个口径，因为它们的"过"含义不同：
#   (a) 代码/配置口径（--dry-run）：与硬件无关，必须全绿 —— 这才是这个自测的判定项
#   (b) 真机口径：需要真机映射表已标定、软限位不矛盾。现在不过【很正常】
#       （verified 还是 false、board_limits_deg 的矛盾还没解决），所以只报不判，
#       否则一装上就"红"，会让人以为代码坏了。
run "main.py --check --dry-run（代码与配置口径，判定项）" "$PY" main.py --check --dry-run

if "$PY" -c "import serial" 2>/dev/null; then
  printf '\n\033[1;36m──── main.py --check（真机口径，信息性，不算失败）\033[0m\n'
  _chk_out="$(mktemp)"
  if "$PY" main.py --check >"$_chk_out" 2>&1; then
    echo "    ✓ 真机口径也全绿（静态项全过；注意 verified=false 只算警告，运行时才会拦）"
  else
    echo "    ⚠️  真机口径未通过 —— 这是【预期】的，等标定完再来看这里："
    grep -E "^      (✗|⚠️)|预检结果" "$_chk_out" | head -8 | sed 's/^/    /'
    echo "    （完整输出：$PY main.py --check）"
  fi
  rm -f "$_chk_out"
else
  printf '\n\033[1;33m──── 跳过真机口径预检（没装 pyserial）\033[0m\n'
fi

run "main.py --list-devices"                "$PY" main.py --list-devices

# ---- 7) 干跑整条链路 + 故障注入 ----
if [[ "$QUICK" == "0" ]]; then
  run "main.py --dry-run（整条链路 8s）"     "$PY" main.py --dry-run --duration 9
  run "故障注入：IMU 停更必须跳闸"           "$PY" main.py --dry-run --duration 8 \
        --stub-fault-imu-freeze 1.0
  run "故障注入：NaN 必须跳闸"               "$PY" main.py --dry-run --duration 8 \
        --stub-fault-nan 1.0
fi

echo
echo "==============================================================================="
printf '自测结果: 通过 \033[1;32m%d\033[0m  失败 \033[1;31m%d\033[0m\n' "$PASS" "$FAIL"
if [[ $FAIL -gt 0 ]]; then
  printf '失败项:\n'
  for f in "${FAILED[@]}"; do echo "  · $f"; done
  echo
  echo "注意：'serial 缺失' / '串口没发现' / 'verified=false' 这类属于【预期失败】——"
  echo "     没插下位机、没装 pyserial、映射表还没标定时它们本来就该报出来。"
  exit 1
fi
echo "全绿。下一步：main.py --check --live（插上下位机做只读检查）"
echo "==============================================================================="
