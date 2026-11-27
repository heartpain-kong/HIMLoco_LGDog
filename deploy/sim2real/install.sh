#!/usr/bin/env bash
# =============================================================================
# LgDog1 sim2real —— N100（或任意 x86_64 Linux 上位机）一键装环境
#
#   ./install.sh                    # 建 venv、装依赖、装 udev 规则、加用户组、预检
#   ./install.sh --wheels ./wheels  # 离线安装（先用 make_offline_bundle.sh 在有网机器上拉包）
#   ./install.sh --no-sudo          # 不动 udev/用户组（自己已经有权限时）
#   ./install.sh --venv /opt/lgdog/venv
#
# 设计原则：
#   · 幂等 —— 重复跑不会有副作用
#   · 不 sudo 也能跑完大部分（只有 udev/用户组那两步需要）
#   · 装完自动跑 main.py --check，把"还差什么"直接打出来
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
WHEELS=""
USE_SUDO=1
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
SKIP_CHECK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)       VENV="$2"; shift 2 ;;
    --wheels)     WHEELS="$2"; shift 2 ;;
    --no-sudo)    USE_SUDO=0; shift ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    --torch-index) TORCH_INDEX="$2"; shift 2 ;;
    -h|--help)    sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（-h 看用法）" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[x] %s\033[0m\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
say "0/6 环境自检"
command -v python3 >/dev/null || die "没有 python3"
PYV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
PYOK=$(python3 -c 'import sys;print(1 if sys.version_info>=(3,8) else 0)')
[[ "$PYOK" == "1" ]] || die "python 版本 $PYV 太老（需要 ≥3.8；N100 上 Ubuntu 22.04/24.04 自带就够）"
echo "    python3 = $(command -v python3)  ($PYV)"
echo "    架构    = $(uname -m)   $(uname -s)"
echo "    目录    = $HERE"
[[ "$(uname -s)" == "Linux" ]] || warn "目标机是 Linux；当前不是，装完只能做静态预检"

# -----------------------------------------------------------------------------
say "1/6 创建虚拟环境 $VENV"
VPY="$VENV/bin/python"
if [[ -x "$VPY" ]]; then
  echo "    已存在，复用"
else
  if python3 -m venv "$VENV" 2>/tmp/lgdog_venv_err.$$; then
    echo "    已创建"
  else
    # ---- 常见败因：Ubuntu 没装 python3-venv（ensurepip 缺失）----
    # 这时候不用装系统包也有救：建一个"不带 pip"的 venv，再用本地 wheel 把 pip 装进去
    # （--wheels 给了的话里面就有 pip-*.whl）。这样【完全离线】也能装好。
    cat /tmp/lgdog_venv_err.$$ | sed 's/^/      /'
    if [[ -n "$WHEELS" ]] && ls "$WHEELS"/pip-*.whl >/dev/null 2>&1; then
      warn "缺 python3-venv（ensurepip）。改用 --without-pip + 本地 pip wheel 引导"
      rm -rf "$VENV"
      python3 -m venv --without-pip "$VENV" || die "连 --without-pip 都失败了（python3 缺 venv 模块？）"
      PIPWHL=$(ls "$WHEELS"/pip-*.whl | head -1)
      echo "    用 $PIPWHL 引导 pip"
      # 官方引导法：pip 的 wheel 本身就是可执行的 zip
      "$VPY" "$PIPWHL/pip" install --no-index --find-links "$WHEELS" pip setuptools wheel \
        || die "pip 引导失败"
      echo "    pip 已装入 venv ✓"
    else
      rm -rf "$VENV"
      die "建 venv 失败：目标机缺 python3-venv。两条路任选：
        A) 联网装一下：sudo apt install python3-venv
        B) 完全离线：make_offline_bundle.sh 造的包里带了 pip 的 wheel，
           用 ./install.sh --wheels ../wheels 会自动走 --without-pip 引导"
    fi
  fi
  rm -f /tmp/lgdog_venv_err.$$
fi
# ★ 离线机器上这一步会失败（要联网），所以不能让它中断整个安装
"$VPY" -m pip install --quiet --upgrade pip setuptools wheel \
  || warn "升级 pip/setuptools 失败（离线机器正常现象），继续用自带的版本"

# -----------------------------------------------------------------------------
say "2/6 安装依赖"
if [[ -n "$WHEELS" ]]; then
  [[ -d "$WHEELS" ]] || die "--wheels 目录不存在: $WHEELS"
  echo "    离线模式：--no-index --find-links $WHEELS"
  if ! "$VPY" -m pip install --no-index --find-links "$WHEELS" -r "$HERE/requirements-deploy.txt"; then
    echo
    die "离线安装依赖失败。最常见的原因：
      · 打包时用了 --no-torch → 包里没有 torch。重打包（去掉 --no-torch），
        或在目标机上单独装：pip install torch --index-url https://download.pytorch.org/whl/cpu
      · 打包时没给 --for-python → wheel 是别的 python 版本的（比如 cp38 装不到 3.10 上）。
        看 $WHEELS 里的文件名：numpy-2.2.6-cp310-... 的 cp310 就是 python 版本，
        必须与目标机 python3 -V 一致。"
  fi
else
  echo "    numpy / PyYAML / pyserial"
  "$VPY" -m pip install --quiet -r "$HERE/requirements-deploy.txt"
  if "$VPY" -c "import torch" 2>/dev/null; then
    echo "    torch 已装，跳过"
  else
    echo "    torch（CPU 版，约 200MB；N100 不需要 CUDA）"
    echo "      源: $TORCH_INDEX"
    "$VPY" -m pip install torch --index-url "$TORCH_INDEX" \
      || { warn "CPU 版 torch 装失败，退回默认源重试"; "$VPY" -m pip install torch; }
  fi
fi
echo "    已装版本："
"$VPY" - <<'PY'
import importlib
for m in ("numpy", "yaml", "serial", "torch"):
    try:
        mod = importlib.import_module(m)
        print("      %-8s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("      %-8s ✗ %s" % (m, e))
PY

# -----------------------------------------------------------------------------
say "3/6 串口权限（/dev/ttyACM* 属于 dialout 组）"
NEED_GROUPS=()
id -nG "$USER" | tr ' ' '\n' | grep -qx dialout || NEED_GROUPS+=(dialout)
id -nG "$USER" | tr ' ' '\n' | grep -qx input   || NEED_GROUPS+=(input)     # 手柄
if [[ ${#NEED_GROUPS[@]} -eq 0 ]]; then
  echo "    已在 dialout / input 组 ✓"
elif [[ "$USE_SUDO" == "1" ]]; then
  SUDO=""
  [[ "$(id -u)" == "0" ]] || SUDO="sudo"
  echo "    把 $USER 加入: ${NEED_GROUPS[*]}"
  $SUDO usermod -aG "$(IFS=,; echo "${NEED_GROUPS[*]}")" "$USER" \
    && warn "★ 要【注销重新登录】才生效（或者用 newgrp dialout 开个新 shell）"
else
  warn "缺组: ${NEED_GROUPS[*]}（--no-sudo 模式没动）。手动：sudo usermod -aG $(IFS=,; echo "${NEED_GROUPS[*]}") $USER"
fi

# -----------------------------------------------------------------------------
say "4/6 udev 规则（免每次手动改权限 / 固定设备名）"
RULES=/etc/udev/rules.d/99-lgdog-sim2real.rules
RULES_BODY='# LgDog1 sim2real —— 下位机(STM32 USB CDC) 与手柄
# 下位机：STM32 虚拟串口，给出稳定别名 /dev/lgdog_board，并放开到 dialout 组
SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", SYMLINK+="lgdog_board", MODE="0660", GROUP="dialout"
# 常见 USB-CAN 适配器（绕过下位机直连电机时用）
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", MODE="0660", GROUP="dialout"
SUBSYSTEM=="tty", ATTRS{idVendor}=="2e88", MODE="0660", GROUP="dialout"
# 手柄：/dev/input/event* 默认 root:input 0660
KERNEL=="event*", SUBSYSTEM=="input", MODE="0660", GROUP="input"
'
if [[ "$USE_SUDO" == "1" ]]; then
  SUDO=""
  [[ "$(id -u)" == "0" ]] || SUDO="sudo"
  printf '%s' "$RULES_BODY" | $SUDO tee "$RULES" >/dev/null \
    && $SUDO udevadm control --reload-rules && $SUDO udevadm trigger \
    && echo "    已写 $RULES 并 reload ✓（插拔一次设备生效）"
  echo "    下位机的稳定设备名: /dev/lgdog_board（config 里也可以写它）"
else
  warn "跳过 udev（--no-sudo）。内容见 $HERE/systemd/99-lgdog-sim2real.rules"
fi

# -----------------------------------------------------------------------------
say "5/6 策略文件"
if [[ -f "$HERE/models/policy.pt" ]]; then
  echo "    models/policy.pt ✓ ($(du -h "$HERE/models/policy.pt" | cut -f1))"
else
  warn "没有 $HERE/models/policy.pt"
  echo "    从开发机拷一份过来（随包带一份就不依赖训练仓库了）："
  echo "      scp <开发机>:<LgRobotDog>/legged_gym/logs/flat_LgDog1/exported/policies/policy.pt \\"
  echo "          $HERE/models/"
fi

# -----------------------------------------------------------------------------
say "6/6 预检"
if [[ "$SKIP_CHECK" == "1" ]]; then
  echo "    --skip-check，跳过"
else
  set +e
  "$VPY" "$HERE/main.py" --check
  RC=$?
  set -e
  echo
  if [[ $RC -eq 0 ]]; then
    echo "✅ 预检通过。"
  else
    warn "预检有未通过项（见上）。常见就两项："
    echo "      · 没插下位机 → 属正常；先跑 $VPY main.py --dry-run 验证整条链路"
    echo "      · pyserial 没装上 → 回看第 2 步的输出"
  fi
fi

cat <<EOF

────────────────────────────────────────────────────────────────
装好了。接下来的顺序（★ 第一次上电务必按这个顺序）：

  # 1) 无硬件验证整条链路（策略/状态机/安全/日志）
  $VENV/bin/python main.py --dry-run

  # 2) 插上下位机，只读预检（发心跳、收反馈，不发电机指令）
  $VENV/bin/python main.py --check --live

  # 3) 真机标定（12 个关节的 id/方向/减速比/零位）
  $VENV/bin/python tools/calibrate.py --map hardware_map.yaml scan
  ... 按 hardware_map.yaml 头注释里的 7 步走，最后把 verified 改成 true

  # 4) 只起立保持，不跑策略（确认电机方向全对）
  $VENV/bin/python main.py --mode init

  # 5) 跑策略（按住手柄 run 键才动）
  $VENV/bin/python main.py

  日志自动落在 $HERE/logs/<tag>_<时间>/，latest 是软链接。
  日常启动可以直接用： ./run.sh  （内部就是 .venv/bin/python main.py "$@"）
────────────────────────────────────────────────────────────────
EOF
