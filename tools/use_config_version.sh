#!/usr/bin/env bash
# =============================================================================
# use_config_version.sh —— 在【V1 版】与【V43 版】配置/代码之间一键切换
#
# 为什么需要它：2026-10-08 按用户指令把 LgDog1 配置 + 奖励函数整体回退到 V1
#   （21 项奖励、270 维观测、100Hz、kp40/kd1.0、默认姿态 (0,-0.4,+0.5)）。
#   但【旧策略（V41/V43）必须配旧配置才能评测】—— 缺了切换脚本就会出现
#   "拿 V1 配置去评测 276 维策略"的静默错误（维度不匹配或更糟: 跑起来但语义错）。
#
# 涉及两个文件（必须成对切换）：
#   legged_gym/legged_gym/envs/LgDog1/LgDog1_config.py    配置值/奖励集
#   legged_gym/legged_gym/envs/base/legged_robot.py       奖励函数实现/观测前缀
#
# 用法:
#   bash tools/use_config_version.sh v1      # 切到 V1（当前生效的就是这套）
#   bash tools/use_config_version.sh v43     # 切到 V43（评测 V41/V43 策略用）
#   bash tools/use_config_version.sh status  # 看现在挂的是哪套
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BK="$ROOT/logs/_backup_20261008_preV1"
CFG="$ROOT/legged_gym/legged_gym/envs/LgDog1/LgDog1_config.py"
ROB="$ROOT/legged_gym/legged_gym/envs/base/legged_robot.py"

# V1 版（回退后的当前状态）另存一份，切换才不会把 V1 弄丢
V1DIR="$ROOT/logs/_config_v1"
mkdir -p "$V1DIR"
[ -f "$V1DIR/LgDog1_config.py" ] || cp "$CFG" "$V1DIR/LgDog1_config.py"
[ -f "$V1DIR/legged_robot.py" ] || cp "$ROB" "$V1DIR/legged_robot.py"

ver() {   # 判定当前配置版本
  python3 - "$CFG" <<'PY'
import re, sys
s = open(sys.argv[1], encoding='utf-8').read()
n = int(re.search(r'num_commands\s*=\s*(\d+)', s).group(1))
# 只数 class scales: 块里的权重行
i = s.index('class scales:')
j = s.find('\n    class ', i)          # scales 是 rewards 的嵌套类，下一个 4 空格 class 即结束
block = s[i:j if j > 0 else len(s)]
sc = len(re.findall(r'^\s+[a-z_]+\s*=\s*-?[0-9]', block, re.M))
print('V1' if n == 4 else 'V43', end='')
print('（num_commands=%d，奖励项 %d 个）' % (n, sc))
PY
}

case "${1:-status}" in
  status)
    echo "当前配置: $(ver)"
    echo "  V1 版备份: $V1DIR"
    echo "  V43 版备份: $BK/LgDog1_config.py.before_revert + legged_robot.py.before_revert"
    ;;
  v1)
    cp "$V1DIR/LgDog1_config.py" "$CFG"
    cp "$V1DIR/legged_robot.py"  "$ROB"
    echo "✅ 已切到 V1 版 → $(ver)"
    ;;
  v43)
    cp "$BK/LgDog1_config.py.before_revert" "$CFG"
    cp "$BK/legged_robot.py.before_revert"  "$ROB"
    echo "✅ 已切到 V43 版 → $(ver)"
    echo "   现在可以评测 V41/V43 策略（276 维 / 50Hz / kp30 kd1.2）"
    ;;
  *)
    echo "用法: $0 {status|v1|v43}"; exit 1;;
esac
