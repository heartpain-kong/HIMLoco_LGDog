#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""add_opendoge_rewards.py —— 把 OpenDoge_train-main 的奖励项加回本项目

来源：~/rl/OpenDoge_train-main/OpenDoge_train-main/legged_gym/envs/base/legged_robot.py
做法：逐字取 OpenDoge 的函数体，只做两处必要适配（都在下面注释里说明），
      不掺入我们之前那些版本里的定制逻辑（那些已随 V1 回退删除）。

加回的 8 项（第 9 项 ankle_pitch 用不了：LgDog1 没有踝关节）:
  feet_contact_forces        接触力超 max_contact_force 罚        （OpenDoge 原样）
  feet_parallel_to_ground    着地时脚底法向与地面平行奖励          （OpenDoge 原样）
  hip_joint_pos              髋关节偏离默认位的 exp(−err²)         （OpenDoge 原样）
  default_joint_pos          12 关节偏离默认位 L2 平方             （OpenDoge 原样）
  default_pos_linear         12 关节偏离默认位 L1（梯度恒定）      （OpenDoge 原样）
  base_height_linear         线性高度误差（★ 去掉它的 DEBUG 打印）  （适配：剥掉刷屏代码）
  diagonal_sync              trot 对角触地同步                     （OpenDoge 原样）
  hip_mirror_symmetry        髋关节左右镜像 (|fl+fr|+|rl+rr|)²     （适配：关节名匹配改成大小写无关，
                              OpenDoge 只认小写 'fl_hip'，而本项目 dof_names 是 'FL_hip_joint' ⇒ 原版会恒返回 0）

权重策略（写进 config 的 scales）:
  · OpenDoge 配置里**有明确值**的三个 ⇒ 用它的值: default_pos_linear -0.05、diagonal_sync -0.2、
    hip_mirror_symmetry -0.1
  · OpenDoge 只写了函数、没配权重的五个 ⇒ 加进去但权重 **0.0**（注释给出建议值与符号），
    以免未经确认就把训练分布改掉（尤其 base_height_linear 的 ×100 平方形式在本项目权重尺度下会压倒一切）

用法：
    python3 tools/add_opendoge_rewards.py --dry-run
    python3 tools/add_opendoge_rewards.py
"""
import argparse
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUR = os.path.join(ROOT, 'legged_gym/legged_gym/envs/base/legged_robot.py')
CFG = os.path.join(ROOT, 'legged_gym/legged_gym/envs/LgDog1/LgDog1_config.py')
OD = os.path.expanduser('~/rl/OpenDoge_train-main/OpenDoge_train-main/legged_gym/envs/base/legged_robot.py')
BK = os.path.join(ROOT, 'logs', '_backup_20261008_preV1')

# (函数名, 是否 OpenDoge 配过权重, 权重, 说明)
ITEMS = [
    ('feet_contact_forces',      False, 0.0,   '接触力超 max_contact_force 罚（建议 -0.01~-0.1）'),
    ('feet_parallel_to_ground',  False, 0.0,   '着地时脚底与地面平行奖励（建议 +0.05~+0.2）'),
    ('hip_joint_pos',            False, 0.0,   '髋关节朝默认位 exp(−err²)（建议 +0.05~+0.2）'),
    ('default_joint_pos',        False, 0.0,   '12 关节偏离默认位 L2²（建议 -0.02~-0.1）'),
    ('default_pos_linear',       True, -0.05,  '12 关节偏离默认位 L1（OpenDoge 值 -0.05）'),
    ('base_height_linear',       False, 0.0,   '线性高度误差（OpenDoge 形式含 ×100 平方，权重须慎用）'),
    ('diagonal_sync',            True, -0.2,   'trot 对角触地同步（OpenDoge 值 -0.2；我们上一轮用过 -0.1）'),
    ('hip_mirror_symmetry',      True, -0.1,   '髋左右镜像（OpenDoge 值 -0.1）'),
]
MARK = '    # ===== OpenDoge_train-main 奖励项（★ 2026-10-08 按用户指令加回；实现逐字取自 OpenDoge）=====\n'


def od_body(name):
    s = open(OD, encoding='utf-8').read()
    m = re.search(r'(?ms)^    def _reward_%s\(self.*?(?=^    def |^    # =|\Z)' % re.escape(name), s)
    return m.group(0).rstrip() + '\n' if m else None


def adapt_base_height_linear(body):
    """去掉 DEBUG 打印块（每 50 步 print 会刷屏），保留计算与返回值。"""
    return re.sub(r'(?ms)\n        # --- DEBUG 打印代码开始 ---.*?# --- DEBUG 打印代码结束 ---\n', '\n', body)


def adapt_hip_mirror(body):
    """关节名匹配改成大小写无关（本项目 dof_names 形如 FL_hip_joint）。"""
    old = """            if 'fl_hip' in name or 'LF_hip' in name or 'front_left_hip' in name:
                fl_idx = idx
            elif 'fr_hip' in name or 'RF_hip' in name or 'front_right_hip' in name:
                fr_idx = idx
            elif 'rl_hip' in name or 'LH_hip' in name or 'rear_left_hip' in name or 'hind_left_hip' in name:
                rl_idx = idx
            elif 'rr_hip' in name or 'RH_hip' in name or 'rear_right_hip' in name or 'hind_right_hip' in name:
                rr_idx = idx"""
    new = """            n = name.lower()          # ★ 适配本项目: dof_names 是 'FL_hip_joint' 这种大写前缀
            if 'fl_hip' in n or 'lf_hip' in n or 'front_left_hip' in n:
                fl_idx = idx
            elif 'fr_hip' in n or 'rf_hip' in n or 'front_right_hip' in n:
                fr_idx = idx
            elif 'rl_hip' in n or 'lh_hip' in n or 'rear_left_hip' in n or 'hind_left_hip' in n:
                rl_idx = idx
            elif 'rr_hip' in n or 'rh_hip' in n or 'rear_right_hip' in n or 'hind_right_hip' in n:
                rr_idx = idx"""
    assert old in body, 'hip_mirror_symmetry 匹配块未找到'
    return body.replace(old, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    cur = open(CUR, encoding='utf-8').read()
    cfg = open(CFG, encoding='utf-8').read()
    have = set(re.findall(r'(?m)^    def _reward_([a-z_0-9]+)\(self', cur))

    added, skipped = [], []
    blocks = []
    for name, has_w, w, desc in ITEMS:
        if name in have:
            skipped.append(name + '(已存在)')
            continue
        b = od_body(name)
        if b is None:
            skipped.append(name + '(OpenDoge 里找不到)')
            continue
        if name == 'base_height_linear':
            b = adapt_base_height_linear(b)
        if name == 'hip_mirror_symmetry':
            b = adapt_hip_mirror(b)
        blocks.append(b.rstrip() + '\n')
        added.append((name, w, desc))

    if not blocks:
        print('没有要加的项'); return 0

    # 插到 _reward_stand_still 之后（奖励函数区末尾）
    m = re.search(r'(?ms)^    def _reward_stand_still\(self.*?(?=\n    def |\n    # =|\Z)', cur)
    assert m, '找不到 _reward_stand_still 作为插入点'
    insert_at = m.end()
    cur = cur[:insert_at] + '\n\n' + MARK + '\n' + '\n'.join(blocks) + cur[insert_at:]

    # 配置里补 scales（插到 scales 块的最后一行之后）
    lines = cfg.split('\n')
    i = next(k for k, l in enumerate(lines) if l.strip() == 'class scales:')
    j = next(k for k in range(i + 1, len(lines)) if lines[k].strip() and not lines[k].startswith('            '))
    new_lines = ['            # --- OpenDoge 奖励项（★ 2026-10-08 加回；0.0 = 已接入但默认关闭）---']
    for name, w, desc in added:
        new_lines.append('            %-22s = %-9s # %s' % (name, w, desc))
    lines[j:j] = new_lines + ['']
    cfg = '\n'.join(lines)

    print('=== 加回的奖励函数 %d 个 ===' % len(added))
    for n, w, d in added:
        print('   %-26s 权重 %-7s %s' % (n, w, d))
    if skipped:
        print('=== 跳过 ===\n   ', skipped)

    if args.dry_run:
        print('\n(dry-run，未写入)')
        return 0
    os.makedirs(BK, exist_ok=True)
    shutil.copy2(CUR, os.path.join(BK, 'legged_robot.py.before_opendoge'))
    shutil.copy2(CFG, os.path.join(BK, 'LgDog1_config.py.before_opendoge'))
    open(CUR, 'w', encoding='utf-8').write(cur)
    open(CFG, 'w', encoding='utf-8').write(cfg)
    print('\n✅ 已写入（备份 *.before_opendoge）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
