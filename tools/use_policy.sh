#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# use_policy.sh —— 把某个归档策略切为 sim2sim/play 的【当前策略】(policy.pt)，
#                 并打印与它严格配套的 sim2sim 命令（含正确的 --snapshot）。
#
# 为什么需要它：sim2sim 默认读 exported/policies/policy.pt，而 policy.pt 会被每次导出覆盖；
#   更危险的是【脚本常量与历史 run 不一致】——例如 V14_h 时代 decimation=2(100Hz)、
#   现在常量是 4(50Hz)，不带 --snapshot 就会用错控制频率跑出 OOD 结果。
#   本脚本按归档名自动反查对应的 run，并把该 run 的 config_snapshot.json 配到命令行里。
#
# 用法:
#   bash tools/use_policy.sh policy_V14_6000.pt          # 切换并打印命令
#   bash tools/use_policy.sh --list                      # 列出所有归档策略
# ─────────────────────────────────────────────────────────────────────────────
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POL="$ROOT/legged_gym/logs/flat_LgDog1/exported/policies"
RUNS="$ROOT/legged_gym/logs/flat_LgDog1"
PY="$HOME/miniconda3/envs/himloco/bin/python"

if [ "${1:-}" = "--list" ] || [ -z "${1:-}" ]; then
  echo "可用归档策略（按版本）:"
  ls "$POL" | grep -E "^policy_" | sed 's/^/  /'
  exit 0
fi

SRC="$POL/$1"
[ -f "$SRC" ] || { echo "❌ 找不到归档策略: $SRC"; exit 1; }

# 从文件名解析版本号（policy_V14_6000.pt -> V14；policy_V16_t_11601.pt -> V16_t）
VER=$(echo "$1" | sed -E 's/^policy_(.*)_[0-9]+\.pt$/\1/')
RUN_DIR=""
for cand in "$RUNS/$VER" "$RUNS/${VER}_h"; do
  [ -d "$cand" ] && RUN_DIR="$cand" && break
done

# ★ 2026-10-10 补：run 还没按惯例改名成 V<NN> 时（轮次没结束就直接导出，V46 就是这样），
#   上面按版本名找必然落空 ⇒ 退回【按 model_<step> 反查】：文件名里的步数在哪个 run 里有
#   同名 checkpoint，那个 run 就是它的来源（同一份权重不会被两个 run 同时导出到同一步）。
if [ -z "$RUN_DIR" ]; then
  STEP=$(echo "$1" | sed -nE 's/^policy_.*_([0-9]+)\.pt$/\1/p')
  if [ -n "$STEP" ]; then
    RUN_DIR=$(ls -dt "$RUNS"/*/ 2>/dev/null | while read -r d; do
                [ -f "$d/model_$STEP.pt" ] && { echo "${d%/}"; break; }
              done)
  fi
  [ -n "$RUN_DIR" ] && echo "   ↳ 目录名不是 V<NN>（本轮还没改名）⇒ 按 model_$STEP.pt 反查到: $(basename "$RUN_DIR")"
fi

cp "$SRC" "$POL/policy.pt"
echo "✅ 已切换: $1  ->  $POL/policy.pt"
echo "   md5 = $(md5sum "$POL/policy.pt" | cut -d' ' -f1)"

SNAP_ARG=""
if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/config_snapshot.json" ]; then
  SNAP_ARG="--snapshot $RUN_DIR/config_snapshot.json"
  $PY - "$RUN_DIR/config_snapshot.json" <<'EOF'
import json, sys
e = json.load(open(sys.argv[1]))['env_cfg']; c = e['control']
print(f"   配套快照: {sys.argv[1].split('/')[-2]}/config_snapshot.json")
print(f"   ⚠️ 该 run 的控制参数: decimation={c['decimation']} action_scale={c['action_scale']} "
      f"kp={list(c['stiffness'].values())[0]} kd={list(c['damping'].values())[0]}")
print(f"   ⇒ 必须带 --snapshot（否则脚本常量(decimation=4)可能与训练不符 → OOD）")
EOF
else
  echo "   ⚠️ 未找到与「$VER」对应的 run 目录/快照 —— 请手动指定 --snapshot 或 --decimation"
fi

cat <<EOF

在 sim2sim 里跑它（在 legged_gym/legged_gym/scripts 目录下）:
  cd $ROOT/legged_gym/legged_gym/scripts
  python sim2sim.py $SNAP_ARG --no-render --no-keyboard --gamepad off --x_vel 1.0 --clamp-vel
  # 有显示器/想用键盘或手柄: 去掉 --no-render --no-keyboard --gamepad off
EOF
