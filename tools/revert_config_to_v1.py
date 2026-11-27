#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""revert_config_to_v1.py —— 把 LgDog1 配置逐项回退到 V1（2026-09 最初那版）

依据：`legged_gym/logs/_archive/V1/config_snapshot.json`（V1 训练时自动存下的完整配置，
它就是权威值——不是回忆、不是近似）。

做什么：
  1) 遍历 V1 快照里的每个字段，在当前 `LgDog1_config.py` 里找到【最后一个】同名赋值
     （后面的类定义覆盖前面的 ⇒ 最后一个是"生效的那个"），把右边的值改成 V1 的值，
     **行尾注释原样保留**。
  2) `rewards.scales`：先把 V1 那 21 项的值改回去，再删掉 V1 没有的项（= 关掉 V1 之后
     新增/改名的奖励项）。
  3) `init_state.default_joint_angles`：按行替换 thigh/calf 数字。
  4) 表达式型赋值（如 `num_one_step_privileged_obs = 46 + 3 + 3 + 187`）单独处理。

不做：不动类结构、不动注释、不动 V1 里没有出现过的字段（V1 之后新增的参数保留，
      它们对 V1 的奖励函数是惰性的；删掉反而可能踩到仍在校验它们的代码）。

用法：
    python3 tools/revert_config_to_v1.py --dry-run     # 只打印将要做的改动
    python3 tools/revert_config_to_v1.py               # 实际写入（会先备份）
"""
import argparse
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, 'legged_gym/legged_gym/envs/LgDog1/LgDog1_config.py')
SNAP = os.path.join(ROOT, 'legged_gym/logs/_archive/V1/config_snapshot.json')

# V1 的 21 项奖励（来自快照 rewards.scales）
V1_SCALES = None          # 运行时从快照填


def flat(d, p=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, '%s.%s' % (p, k) if p else k))
    else:
        out[p] = d
    return out


def lit(v):
    """按代码风格写字面量。"""
    if isinstance(v, bool):
        return 'True' if v else 'False'
    if isinstance(v, float):
        s = repr(v)
        return s
    if isinstance(v, int):
        return str(v)
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(lit(x) for x in v) + ']'
    if isinstance(v, dict):
        return '{' + ', '.join('%r: %s' % (k, lit(x)) for k, x in v.items()) + '}'
    if v is None:
        return 'None'
    return repr(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    v1 = json.load(open(SNAP, encoding='utf-8'))
    v1 = v1.get('env_cfg', v1)
    f1 = flat(v1)
    global V1_SCALES
    V1_SCALES = dict(v1['rewards']['scales'])

    lines = open(CFG, encoding='utf-8').read().split('\n')
    changes, deletions = [], []

    # ---- 建立"类链 → 属性"的行索引 -------------------------------------
    # 只用缩进判断层级；类名取自 `class X(...)` 或 `class X:`
    stack = []          # [(indent, name)]
    index = {}          # (chain tuple, attr) -> [line numbers]
    for i, ln in enumerate(lines):
        m = re.match(r'^(\s*)class\s+([A-Za-z_0-9]+)', ln)
        if m:
            ind = len(m.group(1))
            while stack and stack[-1][0] >= ind:
                stack.pop()
            stack.append((ind, m.group(2)))
            continue
        m = re.match(r'^(\s*)([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.+?)\s*$', ln)
        if m and stack:
            chain = tuple(n for _, n in stack[1:])     # 去掉最外层类名
            index.setdefault((chain, m.group(2)), []).append(i)

    def find_last(chain, attr):
        c = index.get((tuple(chain), attr))
        return c[-1] if c else None

    # ---- 1) 逐字段改值 --------------------------------------------------
    SPECIAL = {
        'env.num_one_step_privileged_obs': '45 + 3 + 3 + 187',
        'env.num_observations': 'num_one_step_observations * 6',
        'env.num_privileged_obs': 'num_one_step_privileged_obs * 1',
    }
    skip_prefix = ('init_member_classes.', 'seed', 'viewer.')
    expr_like = re.compile(r'[A-Za-z_]')      # 含字母 ⇒ 表达式（不是纯字面量）

    for path, val in sorted(f1.items()):
        if path.startswith(skip_prefix):
            continue
        if path.startswith('init_state.default_joint_angles.'):
            continue                            # 单独处理
        parts = path.split('.')
        attr = parts[-1]
        chain = parts[:-1]
        if chain and chain[0] in ('rewards',) and attr in ('scales',):
            continue
        # rewards.scales.X 走单独分支
        if len(chain) >= 2 and chain[-1] == 'scales':
            continue
        ln_no = find_last(chain, attr)
        if ln_no is None:
            continue
        ln = lines[ln_no]
        m = re.match(r'^(\s*)([A-Za-z_][A-Za-z_0-9]*)(\s*=\s*)(.*?)(\s*#.*)?$', ln)
        if not m:
            continue
        cur_val = m.group(4).strip()
        # 纯格式差异（0. vs 0.0、单双引号）不算改动：试着求值比较
        try:
            if not expr_like.search(cur_val):
                cv = eval(cur_val, {'__builtins__': {}}, {})
                if cv == val and type(cv) == type(val):
                    continue
        except Exception:
            pass
        want = SPECIAL.get(path, None)
        if want is None:
            want = lit(val)
            if expr_like.search(cur_val) and not expr_like.search(want):
                # 当前是表达式、V1 是字面量 ⇒ 只有当表达式求值结果与 V1 不同才改
                continue
        if cur_val == want:
            continue
        new = '%s%s%s%s%s' % (m.group(1), attr, m.group(3), want, m.group(5) or '')
        changes.append((ln_no + 1, ln, new, path))
        lines[ln_no] = new

    # ---- 2) default_joint_angles（按行替换 thigh/calf）-------------------
    ang = v1['init_state']['default_joint_angles']
    for i, ln in enumerate(lines):
        if '_thigh_joint' in ln and '_calf_joint' in ln:
            def sub(mo):
                leg, j, _ = mo.group(1), mo.group(2), mo.group(3)
                key = '%s_%s_joint' % (leg, j)
                return "'%s': %s" % (key, lit(ang[key])) if key in ang else mo.group(0)
            new = re.sub(r"'(F[LR]|R[LR])_(thigh|calf)_joint':\s*([-0-9.eE]+)", sub, ln)
            if new != ln:
                changes.append((i + 1, ln, new, 'init_state.default_joint_angles'))
                lines[i] = new

    # ---- 3) rewards.scales：整块重写为 V1 的 21 项（简短注释）----------
    start = end = None
    for i, ln in enumerate(lines):
        if re.match(r'^\s*class scales\s*:', ln):
            start = i
            base_ind = len(ln) - len(ln.lstrip())
            for j in range(i + 1, len(lines)):
                t = lines[j]
                if t.strip() and (len(t) - len(t.lstrip())) <= base_ind \
                        and re.match(r'^\s*(class |def )', t):
                    end = j
                    break
            end = end or len(lines)
            break
    if start is None:
        print('❌ 找不到 class scales'); return 1
    ind = ' ' * (len(lines[start]) - len(lines[start].lstrip()) + 4)
    DESC = {
        'tracking_lin_vel': '线速度跟踪 exp(−err²/tracking_sigma)',
        'tracking_ang_vel': '角速度跟踪 exp(−err²/tracking_sigma)',
        'lin_vel_z': '罚竖直速度²（防弹跳）',
        'ang_vel_xy': '罚横滚/俯仰角速度²（防晃）',
        'orientation': '罚 roll/pitch²（水平姿态）',
        'dof_acc': '罚关节加速度²（冲击）',
        'joint_power': '罚关节功率（省电/降热）',
        'base_height': '机身高度偏离 base_height_target',
        'foot_clearance': '摆动足离地高度不足（机身系口径，V1 版）',
        'action_rate': '罚动作变化率（平滑）',
        'smoothness': '二阶平滑 |a−2a₋₁+a₋₂|²',
        'torques': '罚力矩²',
        'dof_vel': '罚关节速度²',
        'collision': '罚 base/thigh/calf 触地',
        'termination': '非超时摔倒罚一次',
        'dof_pos_limits': '罚超出软限位（V1 权重 0）',
        'dof_vel_limits': '罚超速（V1 权重 0）',
        'torque_limits': '罚超力矩（V1 权重 0）',
        'feet_air_time': '奖励摆动时长（迈步）',
        'feet_stumble': '罚足端撞竖直面（绊）',
        'stand_still': '零指令时罚偏离默认姿态（V1 权重 0）',
    }
    # ⚠️ 不要重复放 class 行：lines[:start+1] 已经保留了它（首版在这里出过重复 class scales ⇒ IndentationError）
    block = [ind + '# ★ V1 原始奖励集（21 项）：值取自 logs/_archive/V1/config_snapshot.json']
    deleted = []
    for i in range(start + 1, end):
        m = re.match(r'^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=', lines[i])
        if m:
            deleted.append((i + 1, lines[i].strip()))
    for k, v in v1['rewards']['scales'].items():
        block.append('%s%-22s = %-10s # %s' % (ind, k, lit(v), DESC.get(k, '')))
    block.append('')
    deleted = [d for d in deleted if re.match(r'^\s*([A-Za-z_][A-Za-z_0-9]*)\s*=', d[1])]
    lines = lines[:start + 1] + block + lines[end:]
    changes.append((start + 1, '(class scales 整块重写)', '%d 项 V1 奖励' % len(v1['rewards']['scales']),
                    'rewards.scales'))

    # ---- 输出 --------------------------------------------------------
    print('=== 值改动 %d 处 ===' % len(changes))
    for no, old, new, path in changes:
        print('  L%-5d %-34s\n          - %s\n          + %s' % (no, path, old.strip(), new.strip()))
    print('=== 删除的奖励项 %d 项（V1 没有）===' % len(deletions))
    for no, ln in deletions:
        print('  L%-5d %s' % (no, ln.strip()))

    if args.dry_run:
        print('\n(dry-run，未写入)')
        return 0

    # 关键注释修正：值改了、注释不能再自相矛盾
    txt = '\n'.join(lines)
    fix = [
        (r"(num_commands = 4)(.*)$", r"\1  # x vel, y vel, yaw vel, heading（V1 是 4 维；第 5 维身高指令是 2026-10-01 才加的）"),
        (r"(decimation = 2)(.*控制频率设置.*)$", r"\1              # V1: 200/2 = 100Hz"),
        (r"(episode_length_s = 20)(.*)$", r"\1"),
    ]
    for pat, rep in fix:
        txt = re.sub(pat, rep, txt, count=1, flags=re.M)
    header = ("# ★★ 2026-10-08 按用户指令【完整回退到 V1】★★\n"
              "#   配置值逐项还原自 legged_gym/logs/_archive/V1/config_snapshot.json（V1 训练时自动存的完整配置）；\n"
              "#   奖励集 = V1 的 21 项；奖励函数实现 = V1 直系祖先 HIMLoco 版（V1 源码已不在本机，见 tuning_log）。\n"
              "#   回退前的版本备份在 logs/_backup_20261008_preV1/，改动清单见同目录 revert_report.md。\n")
    lines = (header + txt).split('\n')

    bk = os.path.join(ROOT, 'logs', '_backup_20261008_preV1')
    os.makedirs(bk, exist_ok=True)
    shutil.copy2(CFG, os.path.join(bk, 'LgDog1_config.py.before_revert'))
    open(CFG, 'w', encoding='utf-8').write('\n'.join(lines))
    with open(os.path.join(bk, 'revert_report.md'), 'w', encoding='utf-8') as f:
        f.write('# V1 回退改动清单（%s）\n\n' % __import__('time').strftime('%F %T'))
        f.write('权威来源: legged_gym/logs/_archive/V1/config_snapshot.json\n\n')
        for no, old, new, path in changes:
            f.write('- `%s`\n  - 旧: `%s`\n  - 新: `%s`\n' % (path, old.strip(), new.strip()))
        f.write('\n## 删除的奖励项（V1 没有）\n')
        for no, ln in deleted:
            f.write('- %s\n' % ln)
    print('\n✅ 已写入 %s（备份 %s/LgDog1_config.py.before_revert）' % (CFG, bk))
    return 0


if __name__ == '__main__':
    sys.exit(main())
