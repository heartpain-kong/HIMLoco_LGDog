#!/usr/bin/env bash
# =============================================================================
# 给 N100 造一个【离线安装包】（N100 常常没网/网很差）
#
# 在【有网】的机器上跑：
#     ./make_offline_bundle.sh                     # → dist/lgdog-sim2real-offline-<日期>.tar.gz
#     ./make_offline_bundle.sh --for-python 3.10   # ★ 目标机 python 与打包机不同时必给
#     ./make_offline_bundle.sh --no-torch          # 目标机已装 torch / 先做个小包（18MB）
#
# 然后把 tar.gz 拷到 N100，解压后：
#     ./install.sh --wheels ./wheels
#
# 注意：pip download 默认按【当前解释器所在平台】拉 wheel。N100 是 linux/x86_64，
#       所以请在 x86_64 Linux 上跑这个脚本；跨平台请在目标架构机器/容器里跑。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/dist"
PKG="lgdog-sim2real-offline-$(date +%Y%m%d)"
STAGE="$OUT/$PKG"        # ★ stage 目录名就是包名：tar 直接打它，不用改名
WHEELS="$STAGE/wheels"
WITH_CUDA=0
NO_TORCH=0
FOR_PY=""
PY="${LGDOG_PYTHON:-$(command -v python3)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-torch-cuda) WITH_CUDA=1; shift ;;
    --no-torch) NO_TORCH=1; shift ;;   # 目标机已经装好 torch，或想先做个小包
    --for-python) FOR_PY="$2"; shift 2 ;;   # 目标机的 python 版本，如 3.10
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
# 失败时把半成品 stage 删掉：留着一个不完整的 dist/*.tar.gz 会被误当成完整包
cleanup() { [[ -d "$STAGE" ]] && rm -rf "$STAGE"; }
trap cleanup EXIT

say "0/5 检查"
HOST_PY=$($PY -c 'import sys;print("%d.%d"%sys.version_info[:2])')
echo "    打包机 python = $PY ($HOST_PY)"
echo "    平台         = $($PY -c 'import sysconfig;print(sysconfig.get_platform())')"
$PY -m pip --version >/dev/null || { echo "没有 pip" >&2; exit 1; }

# ★★ pip download 默认按【打包机自己的解释器与平台】拉 wheel。
#    目标机(N100)如果是 Ubuntu 24.04，python 是 3.12，而这里拉的是 cp38 的包，
#    到了那边 pip 会报 "not a supported wheel on this platform"。
#    所以目标机 python 版本与打包机不同时，必须 --for-python。
TORCH_PY="${FOR_PY:-$HOST_PY}"
DL_ARGS=()
if [[ -n "$FOR_PY" ]]; then
  ABITAG="cp${FOR_PY/./}"
  DL_ARGS+=(--only-binary=:all: --python-version "$FOR_PY" --implementation cp
            --abi "$ABITAG" --platform manylinux2014_x86_64)
  echo "    → 为【目标机 python $FOR_PY】拉 wheel（abi=$ABITAG, manylinux2014_x86_64）"
elif [[ "$HOST_PY" != "3.8" ]]; then
  echo "    ⚠️  没给 --for-python：将按本机 python $HOST_PY 拉 wheel。"
  echo "        目标机 python 版本不同的话会装不上，请加：--for-python 3.10（填目标机的版本）"
else
  echo "    ⚠️  没给 --for-python：按本机 python $HOST_PY 拉 wheel。"
  echo "        ★ 先确认 N100 上是哪个版本（python3 -V）；不是 $HOST_PY 就加 --for-python X.Y"
fi

say "1/5 准备目录 $STAGE"
rm -rf "$STAGE"
mkdir -p "$WHEELS" "$STAGE/sim2real"

say "2/5 拉 wheel（numpy / PyYAML / pyserial）"
# ★ torch 必须从这份清单里剔掉：pip download 会照单全收，而 requirements 里的 torch
#   会从【默认源】拉 CUDA 版（2GB+，N100 根本用不上）。
#   torch 在第 3 步单独按 CPU 版拉。
grep -v -i '^torch' "$HERE/requirements-deploy.txt" > "$STAGE/reqs-no-torch.txt"
$PY -m pip download -d "$WHEELS" "${DL_ARGS[@]}" -r "$STAGE/reqs-no-torch.txt"

say "2.5/5 拉 pip/setuptools/wheel（离线机器建 venv 用的兜底）"
$PY -m pip download -d "$WHEELS" "${DL_ARGS[@]}" pip setuptools wheel \
  || echo "    （拉不到也不致命：目标机装了 python3-venv 就用不上它们）"

say "3/5 拉 torch"
if [[ "$NO_TORCH" == "1" ]]; then
  echo "    --no-torch：跳过（目标机上自己装：pip install torch --index-url "
  echo "      https://download.pytorch.org/whl/cpu）"
elif [[ "$WITH_CUDA" == "1" ]]; then
  # ★ torch 的 wheel tag 与常规包不同（linux_x86_64 / manylinux_2_28），
  #   不能把 --platform/--abi 钉死，只按 python 版本挑
  $PY -m pip download -d "$WHEELS" --only-binary=:all: \
      --python-version "$TORCH_PY" --implementation cp torch \
    || die "拉 CUDA 版 torch 失败"
else
  # CPU 版：N100 没有独显，装 CUDA 版纯属浪费 2GB 磁盘
  if ! $PY -m pip download -d "$WHEELS" --only-binary=:all: \
        --python-version "$TORCH_PY" --implementation cp torch \
        --index-url https://download.pytorch.org/whl/cpu 2>/dev/null; then
    warn "CPU 源没拉到，退回默认源（会拉到 CUDA 版，大很多）"
    $PY -m pip download -d "$WHEELS" --only-binary=:all: \
        --python-version "$TORCH_PY" --implementation cp torch \
      || die "torch 拉不到。可以 --no-torch 先出个小包，torch 到目标机上单独装：
        pip install torch --index-url https://download.pytorch.org/whl/cpu"
  fi
fi
echo "    wheel 总数: $(ls -1 "$WHEELS" | wc -l)   累计: $(du -sh "$WHEELS" | cut -f1)"
echo "    wheel 数量: $(ls -1 "$WHEELS" | wc -l)   总大小: $(du -sh "$WHEELS" | cut -f1)"

say "4/5 拷代码（不含 .venv / logs / __pycache__）"
tar -C "$HERE" \
    --exclude='.venv' --exclude='logs' --exclude='__pycache__' --exclude='dist' \
    --exclude='*.pyc' --exclude='recorded_io' \
    -cf - . | tar -C "$STAGE/sim2real" -xf -
echo "    代码文件数: $(find "$STAGE/sim2real" -type f | wc -l)"

cat > "$STAGE/离线安装说明.txt" <<EOF
LgDog1 sim2real 离线安装包（$PKG）

1) 解压
     tar -xzf $PKG.tar.gz && cd $PKG

2) 装（自动建 venv + 从本地 wheels 装依赖 + 装 udev + 预检）
     cd sim2real
     ./install.sh --wheels ../wheels

3) 验证（不需要硬件）
     ./selftest.sh
     ./.venv/bin/python main.py --dry-run

4) 之后按 PORTING.md 的"第一次上电顺序"走。

包内容：
  sim2real/           上位机代码（含 models/policy.pt 与它的权重指纹）
  wheels/             pip 离线 wheels（numpy / PyYAML / pyserial / torch CPU）
  sim2real/PORTING.md 迁移与上电手册（★ 先读这个）
EOF

say "5/5 打包"
mkdir -p "$OUT"
tar -C "$OUT" -czf "$OUT/$PKG.tar.gz" "$PKG"
rm -rf "$STAGE"
trap - EXIT
ls -lh "$OUT/$PKG.tar.gz"
echo
echo "拷到 N100："
echo "  scp $OUT/$PKG.tar.gz <用户>@<N100-IP>:~/"
echo "  ssh <用户>@<N100-IP> 'tar -xzf $PKG.tar.gz && cd $PKG/sim2real && ./install.sh --wheels ../wheels'"
