#!/usr/bin/env bash
# =============================================================================
# 把 sim2real + 下位机固件 + 狗的模型同步到 U 盘（默认找标签 F326 的那个）
#
#   tools/sync_to_usb.sh                    # 自动找 F326，同步到 <U盘>/LgRobot
#   tools/sync_to_usb.sh --label MYUSB      # 指定卷标
#   tools/sync_to_usb.sh --dest /mnt/x      # 直接指定挂载点
#   tools/sync_to_usb.sh --dry-run          # 只看会做什么
#   tools/sync_to_usb.sh --check            # 只做校验，不拷
#
# ★ 为什么要有这个脚本（而不是手敲 rsync）：
#   手敲的时候踩过一个坑 —— U 盘已经拔了，rsync 因为建不出目标目录直接失败，
#   但命令是 `rsync ... | wc -l`，于是"输出 0 行"被当成了"0 处差异"，
#   假通过。这个脚本把"目标存在性"和"校验真实性"都做成硬判据。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # -> sim2real/
RL="$(cd "$HERE/../../.." && pwd)"                            # -> rl/
# 层级核对：sim2real → deploy → LgRobotDog → rl
LABEL="F326"
DEST_ROOT=""
MODE="sync"
FORCE=0
ONLY="all"      # all | sim2real | dm02 | LgDog1 | 逗号组合
ROOT_MODE=0     # 1 = 直接放挂载根目录（不套 LgRobot/ 这一层）

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --dest)  DEST_ROOT="$2"; shift 2 ;;
    --dry-run) MODE="dry"; shift ;;
    --check)   MODE="check"; shift ;;
    --pull)    MODE="pull"; shift ;;      # 把 U 盘上更新的文件【拉回】仓库
    --force)   FORCE=1; shift ;;          # 明知目标更新也照样覆盖
    --only)    ONLY="$2"; shift 2 ;;       # 只同步指定的（如 --only sim2real）
    --root)    ROOT_MODE=1; shift ;;       # 直接放到 U 盘根目录
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m[x] %s\033[0m\n' "$*" >&2; exit 1; }

EXCLUDES=(--exclude '.venv' --exclude 'logs' --exclude '__pycache__'
          --exclude 'dist' --exclude '*.pyc')

# ------------------------------------------------------------------ 找挂载点
if [[ -z "$DEST_ROOT" ]]; then
  DEST_ROOT="$(lsblk -o LABEL,MOUNTPOINT -P 2>/dev/null \
               | awk -v L="$LABEL" -F'"' '$2==L && $4!="" {print $4; exit}')"
fi
[[ -n "$DEST_ROOT" && -d "$DEST_ROOT" ]] || die "找不到卷标为 $LABEL 的已挂载 U 盘。
    查一下： lsblk -o NAME,SIZE,LABEL,MOUNTPOINT
    插上后在文件管理器里点一下让它自动挂载，或用 --dest /media/<用户>/<卷标> 指定。"
if [[ "$ROOT_MODE" == "1" ]]; then
  DEST="$DEST_ROOT"                       # 直接放根目录
else
  DEST="$DEST_ROOT/LgRobot"
  [[ -d "$DEST" ]] || die "$DEST 不存在（这个 U 盘上没有 LgRobot 目录）。
    想直接放到盘根目录：加 --root
    先确认插的是对的盘： ls -la $DEST_ROOT"
fi

say "目标: $DEST   （卷标 $LABEL 挂载在 $DEST_ROOT）"
df -h "$DEST_ROOT" | tail -1

# 三个源
SRC_SIM="$HERE"
SRC_DM02="$RL/dm02"
SRC_MODEL="$RL/LgRobotDog/legged_gym/resources/robots/LgDog1"

# 选择器：ONLY 为 all 或包含该名字时返回 0
want() { case ",$ONLY," in *",$1,"*|*",all,"*) return 0 ;; *) return 1 ;; esac; }
[[ "$ONLY" == "sim2real" || "$ONLY" == "dm02" || "$ONLY" == "LgDog1" || "$ONLY" == "all" \
   || "$ONLY" == *,* ]] || die "--only 只接受 all/sim2real/dm02/LgDog1（可逗号组合），收到: $ONLY"
# ★ 末尾必须 return 0：`set -o pipefail` 下，若最后一次 want 返回 1，
#   整条 `_each | wc -l` 就被判失败，配合 `set -e` 会让脚本【静默退出】。
_each() { for n in sim2real dm02 LgDog1; do want "$n" && echo "$n"; done; return 0; }
echo "    只同步: $(  _each | tr '\n' ' ' )"

for d in "$SRC_SIM" "$SRC_DM02" "$SRC_MODEL"; do
  [[ -d "$d" ]] || die "源目录不存在: $d"
done

# ------------------------------------------------------------------ ★ 目标更新检测
# 踩过的坑：U 盘带出去、在别的机器（Keil/编辑器）上改过源码，回来一同步就被
# 本地那份【旧的】覆盖掉，改动直接消失。rsync 默认（不加 -u）不管谁新谁旧，
# 只看"大小或时间不同"就覆盖。所以这里先比一遍：目标比源【新】的文件全都列出来，
# 不确认就不许覆盖。
find_newer_on_dest() {
  # $1..$3 = 三个源目录, $4..$6 = 对应的目标目录
  "${LGDOG_PYTHON:-python3}" - "$@" <<'PYEOF2'
import os, sys
srcs = sys.argv[1:4]
dsts = sys.argv[4:7]
names = ("sim2real", "dm02", "LgDog1")
TOL = 60          # 容差 60s：只关心"明显更晚"的，避免拷贝抖动/时钟偏差误报
SKIP_DIRS = {"__pycache__", "logs", ".venv", "dist", "Objects", "Listings"}
for name, src, dst in zip(names, srcs, dsts):
    if not (os.path.isdir(src) and os.path.isdir(dst)):
        continue
    for root, dirs, files in os.walk(dst):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith((".pyc", ".o", ".d", ".crf", ".lst", ".htm", ".map", ".dep")):
                continue                      # 编译器产物不当作"人的改动"
            dp = os.path.join(root, f)
            sp = os.path.join(src, os.path.relpath(dp, dst))
            if not os.path.exists(sp):
                print("DESTONLY\t%s\t%s" % (name, os.path.relpath(dp, dst)))
                continue
            try:
                dm, sm = os.path.getmtime(dp), os.path.getmtime(sp)
            except OSError:
                continue
            if dm > sm + TOL:
                print("NEWER\t%s\t%s\t+%ds" % (name, os.path.relpath(dp, dst), int(dm - sm)))
PYEOF2
}

NEWER="$(find_newer_on_dest "$SRC_SIM" "$SRC_DM02" "$SRC_MODEL" "$DEST/sim2real" "$DEST/dm02" "$DEST/LgDog1" || true)"
if [[ -n "$NEWER" ]]; then
  n_newer=$(printf '%s\n' "$NEWER" | grep -c '^NEWER' || true)
  n_only=$(printf '%s\n' "$NEWER" | grep -c '^DESTONLY' || true)
  printf '\n\033[1;33m[!] U 盘上有比仓库【更新】的文件 %s 个、仓库里没有的文件 %s 个：\033[0m\n' "$n_newer" "$n_only"
  printf '%s\n' "$NEWER" | head -20 | sed 's/^/      /'
  [[ $(printf '%s\n' "$NEWER" | wc -l) -gt 20 ]] && echo "      …（只列前 20 个）"
  echo
  echo "      ★ 这通常意味着：U 盘被带出去、在别的机器上改过（Keil/编辑器/编译器产物）。"
  echo "        直接同步会用仓库里【旧的】版本覆盖它们。三条路："
  echo "          ./tools/sync_to_usb.sh --pull      # 先把盘上更新的拉回仓库（会留备份），看一眼再决定"
  echo "          ./tools/sync_to_usb.sh --force     # 明知如此，仍按仓库覆盖过去"
  echo "          ./tools/sync_to_usb.sh --check     # 只看不动"
  if [[ "$MODE" == "sync" && "$FORCE" != "1" ]]; then
    echo
    die "已停止，什么都没覆盖（要覆盖请显式 --force）"
  fi
fi

# ------------------------------------------------------------------ 拉回（--pull）
if [[ "$MODE" == "pull" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  echo
  printf '%s\n' "$NEWER" | grep '^NEWER' | while IFS=$'\t' read -r _ name rel extra; do
    src="$SRC_SIM"; dst="$DEST/sim2real"
    case "$name" in
      dm02) src="$SRC_DM02"; dst="$DEST/dm02" ;;
      LgDog1) src="$SRC_MODEL"; dst="$DEST/LgDog1" ;;
    esac
    if [[ -f "$src/$rel" ]]; then
      cp -p "$src/$rel" "$src/$rel.pre_pull_$STAMP"
      printf '    拉回 %-10s %s（原文件备份为 *.pre_pull_%s）\n' "$name" "$rel" "$STAMP"
    else
      printf '    新增 %-10s %s\n' "$name" "$rel"
    fi
    mkdir -p "$(dirname "$src/$rel")"
    cp -p "$dst/$rel" "$src/$rel"
  done
  echo
  echo "✅ 已把 U 盘上更新的文件拉回仓库（原名.pre_pull_$STAMP 是本地旧版）。"
  echo "   请 diff 看一眼，确认后再跑一次同步把它推回去。"
  exit 0
fi

# ------------------------------------------------------------------ 只看不动（--check）
if [[ "$MODE" == "check" ]]; then
  echo
  if [[ -z "$NEWER" ]]; then echo "    ✓ 盘上没有比仓库更新的文件"; fi
  echo "    （--check 只看不动。要推过去：不带参数；要拉回来：--pull）"
  exit 0
fi

# ------------------------------------------------------------------ 同步
if [[ "$MODE" == "sync" || "$MODE" == "dry" ]]; then
  RS=(-rt --no-perms --no-owner --no-group --info=stats2)
  [[ "$MODE" == "dry" ]] && RS+=(--dry-run -v)
  i=0; n=$( _each | wc -l )
  if want sim2real; then
    i=$((i+1)); say "$i/$n sim2real（上位机部署栈，排除 .venv/logs/__pycache__/dist）"
    rsync "${RS[@]}" "${EXCLUDES[@]}" "$SRC_SIM/" "$DEST/sim2real/" \
      | grep -E "Number of (regular )?files|Total file size|created" || true
  fi
  if want dm02; then
    i=$((i+1)); say "$i/$n dm02（下位机固件）"
    rsync "${RS[@]}" "$SRC_DM02/" "$DEST/dm02/" \
      | grep -E "Number of (regular )?files|Total file size|created" || true
  fi
  if want LgDog1; then
    i=$((i+1)); say "$i/$n LgDog1（xml + urdf + meshes + check）"
    rsync "${RS[@]}" "$SRC_MODEL/" "$DEST/LgDog1/" \
      | grep -E "Number of (regular )?files|Total file size|created" || true
  fi
fi

[[ "$MODE" == "dry" ]] && { echo; echo "（--dry-run，没有真的写）"; exit 0; }

# ------------------------------------------------------------------ 清产物
say "4/4 清理自检产物 + 校验"
if want sim2real; then
  rm -rf "$DEST/sim2real/logs"
  find "$DEST/sim2real" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
  find "$DEST/sim2real" -name "*.pyc" -delete 2>/dev/null || true
fi

# ★ 校验：必须让 rsync 真跑起来并成功退出，再数差异行。
#   直接 `rsync | wc -l` 会在 rsync 失败时数到 0 行 → 假通过。
FAIL=0
PAIRS=()
want sim2real && PAIRS+=("sim2real:$SRC_SIM:$DEST/sim2real")
want dm02     && PAIRS+=("dm02:$SRC_DM02:$DEST/dm02")
want LgDog1   && PAIRS+=("LgDog1:$SRC_MODEL:$DEST/LgDog1")
for pair in "${PAIRS[@]}"; do
  name="${pair%%:*}"; rest="${pair#*:}"; src="${rest%%:*}"; dst="${rest#*:}"
  out="$(rsync -rtnc --no-perms --no-owner --no-group "${EXCLUDES[@]}" "$src/" "$dst/" 2>&1)" \
    || { printf '    %-10s \033[1;31mrsync 失败，校验无效\033[0m: %s\n' "$name" "$out"; FAIL=1; continue; }
  n="$(printf '%s' "$out" | grep -c . || true)"
  if [[ "$n" == "0" ]]; then
    printf '    %-10s \033[1;32m0 处差异 ✓\033[0m\n' "$name"
  else
    printf '    %-10s \033[1;31m%s 处差异\033[0m\n' "$name" "$n"; FAIL=1
  fi
done

sync
if [[ "$FAIL" != "0" ]]; then
  die "校验没通过 —— 别拔盘，先把上面的差异看一眼。"
fi
echo
echo "✅ 同步并校验完成，刷盘已执行，可以安全拔盘。"
[[ -f "$DEST/拷贝说明.txt" ]] && echo "   U 盘上的说明文件: $DEST/拷贝说明.txt（内容没变的话不用更新）"

# ★ vfat/exfat 没有 Linux 执行位（vfat 的 showexec 只给 .exe/.com/.bat 加 x），
#   所以盘上的 *.sh 拷过去是 -rw-r--r--：`./install.sh` 会 "Permission denied"。
#   用户手册本来就要求"先拷到目标机硬盘再跑"，这里再提醒一次。
FSTYPE="$(findmnt -no FSTYPE -T "$DEST_ROOT" 2>/dev/null || echo '?')"
case "$FSTYPE" in
  vfat|exfat|msdos|ntfs)
    echo
    echo "   ⚠️ 这个盘是 $FSTYPE —— 没有 Linux 执行位，盘上的 *.sh 不能直接 ./ 执行。"
    echo "      在目标机上先拷到硬盘再装（正确做法）："
    echo "        cp -r $DEST/sim2real ~/lgdog-sim2real && cd ~/lgdog-sim2real && ./install.sh"
    echo "      或临时用： bash install.sh / bash selftest.sh" ;;
esac
