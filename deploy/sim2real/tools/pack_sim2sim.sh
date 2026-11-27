#!/usr/bin/env bash
# =============================================================================
# 打一个【可独立运行】的 sim2sim 包（含模型、策略、判读工具），可选的拷到 U 盘
#
#   tools/pack_sim2sim.sh                      # 只在本地打出包并实测，不碰 U 盘
#   tools/pack_sim2sim.sh --label F326         # 打包 + 拷到卷标 F326 的 U 盘（<盘>/LgRobot/sim2sim）
#   tools/pack_sim2sim.sh --dest /media/x/Y --sub ""   # 直接拷到盘根目录
#   tools/pack_sim2sim.sh --no-verify          # 跳过"从包里实跑一遍"的验证
#
# ★ 为什么需要"打包"而不是直接拷 scripts/ 目录：
#   sim2sim.py 的路径是【按 __file__ 往上两级】推出来的：
#     SCRIPT_DIR     = <X>/legged_gym/legged_gym/scripts
#     LEGGED_GYM_DIR = <X>/legged_gym                     ← 由上面推出来
#     SCENE_XML      = LEGGED_GYM_DIR/resources/robots/LgDog1/xml/scene.xml
#     POLICY_PATH    = LEGGED_GYM_DIR/logs/flat_LgDog1/exported/policies/policy.pt
#   所以必须【复刻这个目录层级】才跑得起来，单独拷一个 sim2sim.py 是跑不了的。
#   另外 xml 里 meshdir="../meshes/"、scene.xml 里 <include file="LgDog1.xml"/>，
#   网格和 xml 也必须按原位带上。
#
# ★ 随包放哪份策略：放【当前已在 sim2sim 里验过的那份】（默认取最新 policy_auto_*.pt），
#   并把它同时命名成 policy.pt —— 这样不带 --policy 也能直接跑。
#   仓库里 exported/policies/policy.pt 可能是训练中途覆盖的、没验过的版本，
#   直接拿它当默认很危险（实测有一份给走路指令 1.8 秒就倒）。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"      # -> sim2real/
RL="$(cd "$HERE/../../.." && pwd)"                            # -> rl/
R="$RL/LgRobotDog"
PY="${LGDOG_PYTHON:-python3}"

STAGE_DEFAULT="$RL/dist/sim2sim"
STAGE="$STAGE_DEFAULT"
DEST_ROOT=""
LABEL=""
SUB="LgRobot"          # 拷到 U 盘时放在这个子目录下（留空=盘根目录）
VERIFY=1
POLICY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)      DEST_ROOT="$2"; shift 2 ;;
    --label)     LABEL="$2"; shift 2 ;;
    --sub)       SUB="$2"; shift 2 ;;
    --stage)     STAGE="$2"; shift 2 ;;
    --policy)    POLICY="$2"; shift 2 ;;
    --no-verify) VERIFY=0; shift ;;
    -h|--help)   sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[x] %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ 找策略
POL_DIR="$R/legged_gym/logs/flat_LgDog1/exported/policies"
if [[ -z "$POLICY" ]]; then
  POLICY="$(ls -1 "$POL_DIR"/policy_auto_*.pt 2>/dev/null \
            | sed 's/.*policy_auto_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)" || true
fi
[[ -n "$POLICY" && -f "$POLICY" ]] || die "找不到可用的策略（$POL_DIR/policy_auto_*.pt）。
    先跑一次 tools/auto_sim2sim_check.sh 产出 policy_auto_<step>.pt，或用 --policy 指定。"

# ------------------------------------------------------------------ 打包
say "1/4 打包到 $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE/legged_gym/legged_gym/scripts" \
         "$STAGE/legged_gym/resources/robots/LgDog1" \
         "$STAGE/legged_gym/logs/flat_LgDog1/exported/policies" \
         "$STAGE/tools" "$STAGE/recorded_io"

# ① 本体 + 它本地 import 的模块（record_io 是 sim2sim 记录功能要用的）
cp "$R/legged_gym/legged_gym/scripts/sim2sim.py"   "$STAGE/legged_gym/legged_gym/scripts/"
cp "$R/legged_gym/legged_gym/scripts/record_io.py" "$STAGE/legged_gym/legged_gym/scripts/"
for t in check_dof_order.py measure_stand.py view_model.py urdf2xml.py; do
  [[ -f "$R/legged_gym/legged_gym/scripts/$t" ]] && \
    cp "$R/legged_gym/legged_gym/scripts/$t" "$STAGE/legged_gym/legged_gym/scripts/"
done

# ② 机器人模型（xml 靠 ../meshes/ 引网格；scene.xml include LgDog1.xml）
for d in xml urdf meshes check; do
  [[ -d "$R/legged_gym/resources/robots/LgDog1/$d" ]] || continue
  mkdir -p "$STAGE/legged_gym/resources/robots/LgDog1/$d"
  rsync -rt --no-perms --no-owner --no-group \
        "$R/legged_gym/resources/robots/LgDog1/$d/" \
        "$STAGE/legged_gym/resources/robots/LgDog1/$d/"
done

# ③ 策略：验过的那份，同时命名成 policy.pt（不带 --policy 也能跑）
cp "$POLICY" "$STAGE/legged_gym/logs/flat_LgDog1/exported/policies/policy.pt"
cp "$POLICY" "$STAGE/legged_gym/logs/flat_LgDog1/exported/policies/"

# ④ 训练配置快照 + 一份验过的录制（给判读工具当样板）
SNAP="$(ls -t "$R"/legged_gym/logs/flat_LgDog1/*/config_snapshot.json 2>/dev/null | head -1 || true)"
[[ -n "$SNAP" ]] && cp "$SNAP" "$STAGE/legged_gym/logs/flat_LgDog1/"
# 与随包策略同指纹的那份录制（有就带，没有就算了）
REC="$(ls -t "$R"/legged_gym/legged_gym/scripts/recorded_io/sim2sim_*.npz 2>/dev/null | head -1 || true)"
[[ -n "$REC" ]] && cp "$REC" "$STAGE/recorded_io/"

# ⑤ 判读/训练侧工具
cp "$R/tools/analyze_attitude.py" "$R/tools/crosscheck_pitch.py" "$STAGE/tools/" 2>/dev/null || true
cp "$R/tools/export_checkpoint.py" "$R/tools/auto_sim2sim_check.sh" "$STAGE/tools/" 2>/dev/null || true

# ⑥ 一键运行脚本 + 依赖清单 + 说明
cat > "$STAGE/run_sim2sim.sh" <<'RUNSH'
#!/usr/bin/env bash
# 从本包目录直接跑 sim2sim（不依赖仓库；任意位置可用）
#   ./run_sim2sim.sh                 # 默认：场景=障碍赛道 scene.xml，前进 1.0 m/s，跑 30s 墙钟
#   ./run_sim2sim.sh --no-render --x_vel 0.5
#   SECS=120 ./run_sim2sim.sh        # 想跑久一点
# 参数原样透传给 sim2sim.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECS="${SECS:-30}"
PY="${LGDOG_PYTHON:-python3}"
cd "$HERE/legged_gym/legged_gym/scripts"
# ★ 必须 cd 到 scripts/ 再跑：sim2sim.py 按 __file__ 往上两级推 SCENE_XML / POLICY_PATH
# ★ 必须带 -u：不加的话 Python 对管道用块缓冲，而 timeout 到点直接 SIGTERM，
#   缓冲区里的输出会全部丢掉 —— 现象是"跑了但什么都没打印"（踩过）。
exec timeout "$SECS" "$PY" -u sim2sim.py \
  --no-keyboard --gamepad off --clamp-vel \
  --snapshot "$HERE/legged_gym/logs/flat_LgDog1/config_snapshot.json" \
  "$@"
RUNSH
chmod 755 "$STAGE/run_sim2sim.sh"

cat > "$STAGE/requirements-sim2sim.txt" <<'REQ'
# sim2sim 需要的运行时依赖（真机部署栈不需要这些，两者是分开的）
numpy>=1.21
mujoco>=3.0            # 实测用 3.2.3
torch>=1.10            # 只用来加载 TorchScript 策略；CPU 版即可
# 可选（只有开窗口/手柄时才需要）：
# mujoco_viewer
# glfw
REQ

printf '%s\n' \
  "================================================================================" \
  " sim2sim 独立运行包" \
  " 打包时间: $(date '+%F %T')    来源: $R" \
  "================================================================================" \
  "" \
  "【怎么跑】" \
  "  ./run_sim2sim.sh                       # 默认跑 30 秒墙钟（约 57× 实时 = 约 28 分钟仿真）" \
  "  SECS=300 ./run_sim2sim.sh              # 跑 5 分钟墙钟（约 600 秒仿真，= auto_sim2sim_check 的口径）" \
  "  ./run_sim2sim.sh --x_vel 0.5 --no-render" \
  "  ./run_sim2sim.sh --scene scene.xml --spawn-x -3 --spawn-y 0.5" \
  "" \
  "  依赖： pip install -r requirements-sim2sim.txt" \
  "        （N100/无显卡机器：torch 装 CPU 版 pip install torch --index-url https://download.pytorch.org/whl/cpu）" \
  "" \
  "【里面有什么】" \
  "  legged_gym/legged_gym/scripts/sim2sim.py    ← 主程序（★ 目录层级不能动，见下）" \
  "  legged_gym/legged_gym/scripts/record_io.py  ← sim2sim 记录功能 import 的本地模块" \
  "  legged_gym/resources/robots/LgDog1/         ← xml + urdf + meshes（14MB）+ check 渲染图" \
  "  legged_gym/logs/flat_LgDog1/exported/policies/policy.pt          ← 随包策略（见下）" \
  "  legged_gym/logs/flat_LgDog1/config_snapshot.json                 ← 训练配置快照（--snapshot 用）" \
  "  tools/analyze_attitude.py                   ← 纯 numpy：量化 pitch/roll/站高（判据依据）" \
  "  tools/crosscheck_pitch.py                   ← 纯 numpy：交叉核对 pitch" \
  "  tools/export_checkpoint.py, auto_sim2sim_check.sh  ← 训练侧工具（要训练仓库，见下）" \
  "  recorded_io/                                ← 一份验过的录制，给 analyze_attitude.py 做样板" \
  "" \
  "【★ 为什么目录要长这样】" \
  "  sim2sim.py 用 __file__ 往上两级推路径：" \
  "    SCRIPT_DIR     = <X>/legged_gym/legged_gym/scripts" \
  "    LEGGED_GYM_DIR = <X>/legged_gym            ← 由上面推出来" \
  "    SCENE_XML      = LEGGED_GYM_DIR/resources/robots/LgDog1/xml/scene.xml" \
  "    POLICY_PATH    = LEGGED_GYM_DIR/logs/flat_LgDog1/exported/policies/policy.pt" \
  "  所以【必须保持这个层级】，而且 cwd 无所谓但脚本会自己按 __file__ 找。" \
  "  另外 LgDog1.xml 里 meshdir=\"../meshes/\"、scene.xml 里 <include file=\"LgDog1.xml\"/>，" \
  "  网格与 xml 都必须按原位带上，单独拷 sim2sim.py 是跑不起来的。" \
  "" \
  "【随包的策略】" \
  "  $(basename "$POLICY")  →  同时命名成 policy.pt" \
  "  这样不带 --policy 也能直接跑，且默认就是【已经验过】的那份。" \
  "  ⚠️ 不要直接用训练仓库里的 exported/policies/policy.pt：它会被训练轮反复覆盖，" \
  "     实测其中一份给走路指令 1.8 秒就倒了（随包这份跑满 600 秒没摔）。" \
  "  判断"是不是同一份权重"只看【权重指纹】（按参数名排序对张量字节做 md5），" \
  "     文件 md5 会因 torch.jit 每次序列化不同而变化，不可信。" \
  "" \
  "【场景】" \
  "  scene.xml       障碍测试场地（四组楼梯 + 五个高台 + 141 个 cm 数字标签）" \
  "  LgDog1.xml      机器人本体（由 urdf 转换；结构对齐 go1.xml）" \
  "  _chk_row.xml    当时做模型检查的临时文件" \
  "  出生点参考：平地 (-3.0, 0.5)；二层基座顶 (1.8, 0.0) 需 --spawn-z 0.98" \
  "" \
  "【训练侧工具（本包带了但在这里跑不了）】" \
  "  export_checkpoint.py / auto_sim2sim_check.sh 需要训练仓库（rsl_rl + logs 里的 model_N.pt）。" \
  "  它们的流程是：等 model_<step>.pt → 导出成 policy_auto_<step>.pt → 本包跑 sim2sim" \
  "  → analyze_attitude.py 量化 → 按 pitch≤±4° 且 z≥0.24 判好坏。" \
  "  所以在训练机上，把这些工具放回 <LgRobotDog>/tools/ 用。" \
  "================================================================================" \
  > "$STAGE/运行说明.txt"

echo "    文件数: $(find "$STAGE" -type f | wc -l)    体积: $(du -sh "$STAGE" | cut -f1)"
echo "    随包策略: $(basename "$POLICY")"

# ------------------------------------------------------------------ 本地实测
if [[ "$VERIFY" == "1" ]]; then
  say "2/4 从包里实跑 sim2sim（证明这个包真的能独立跑）"
  if ! "$PY" -c "import mujoco, torch" 2>/dev/null; then
    warn "当前 python 缺 mujoco/torch，跳过实跑。"
    warn "想验证请用有依赖的解释器： LGDOG_PYTHON=~/miniconda3/envs/himloco/bin/python $0 ..."
  else
    out="$(cd "$STAGE/legged_gym/legged_gym/scripts" && \
           timeout 60 "$PY" sim2sim.py --no-render --no-keyboard --gamepad off \
             --no-record --cmd-schedule "0:1.0" 2>&1 || true)"
    echo "$out" | grep -E "加载策略|权重指纹|加载场景|MuJoCo 关节顺序|就绪|稳定完成|性能|Error|错误|Traceback" | head -12 | sed 's/^/    /'
    if echo "$out" | grep -q "加载场景"; then
      echo "    ✓ 场景与策略都加载成功"
    else
      die "从包里跑不起来 —— 上面的输出是线索"
    fi
    # 清理这次实跑产生的垃圾
    rm -rf "$STAGE/legged_gym/legged_gym/scripts/recorded_io" \
           "$STAGE/legged_gym/legged_gym/scripts/MUJOCO_LOG.TXT" \
           "$STAGE/legged_gym/legged_gym/scripts/__pycache__"
  fi
else
  say "2/4 跳过实跑（--no-verify）"
fi

# ------------------------------------------------------------------ 拷到 U 盘
if [[ -z "$DEST_ROOT" && -n "$LABEL" ]]; then
  DEST_ROOT="$(lsblk -o LABEL,MOUNTPOINT -P 2>/dev/null \
               | awk -v L="$LABEL" -F'"' '$2==L && $4!="" {print $4; exit}')"
  [[ -n "$DEST_ROOT" ]] || die "找不到卷标 $LABEL 的已挂载 U 盘（lsblk -o NAME,LABEL,MOUNTPOINT 看看）"
fi

if [[ -n "$DEST_ROOT" ]]; then
  [[ -d "$DEST_ROOT" ]] || die "--dest 不存在: $DEST_ROOT"
  if [[ -n "$SUB" ]]; then DEST="$DEST_ROOT/$SUB/sim2sim"; else DEST="$DEST_ROOT/sim2sim"; fi
  say "3/4 拷到 $DEST"
  mkdir -p "$(dirname "$DEST")"
  rsync -rt --no-perms --no-owner --no-group --delete --info=stats2 \
        "$STAGE/" "$DEST/" | grep -E "Number of (regular )?files|Total file size" | sed 's/^/    /'
  echo "    校验:"
  if out="$(rsync -rtnc --no-perms --no-owner --no-group "$STAGE/" "$DEST/" 2>&1)"; then
    n="$(printf '%s' "$out" | grep -c . || true)"
    [[ "$n" == "0" ]] && echo "      0 处差异 ✓" || die "      $n 处差异 ✗"
  else
    die "      校验时 rsync 失败，校验无效: $out"
  fi
  FSTYPE="$(findmnt -no FSTYPE -T "$DEST_ROOT" 2>/dev/null || echo '?')"
  case "$FSTYPE" in
    vfat|exfat|msdos|ntfs)
      echo
      echo "    ⚠️ 这个盘是 $FSTYPE：没有 Linux 执行位（*.sh 会是 -rw-r--r--），"
      echo "       盘上直接 ./run_sim2sim.sh 会 Permission denied —— 用 bash run_sim2sim.sh，"
      echo "       或先拷到硬盘再跑。" ;;
  esac
  sync
  echo "    已刷盘"
else
  say "3/4 没给 --label/--dest，只打了本地包（$STAGE）"
fi

say "4/4 完成"
echo "  本地包: $STAGE"
[[ -n "$DEST_ROOT" ]] && echo "  U 盘:   ${DEST:-}"
echo "  再打包一次随时可重跑本脚本；U 盘不在也能先打好本地包。"
