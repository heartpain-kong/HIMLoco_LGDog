#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""revert_rewards_to_v1.py —— 把奖励函数实现回退到 V1（2026-09 最初那版）

背景：V1 的源码已不在本机（其 config_snapshot.json 里记的 legged_robot.py 指纹
      `0b327e4ea3…` 与现存所有副本都不匹配；git 最早提交是 10-03，比 V1 晚得多）。
      所以函数体取自 V1 的直系祖先 `~/rl/HIMLoco`（奖励函数名与 V1 的 21 项一一对应，
      见 tuning_log 2026-10-08 记录）——**函数体忠实，但 V1 当时的少量本地改动无法复原**。

做三件事：
  1) 把 V1 的 21 个奖励函数体替换成 HIMLoco 版（`stumble` → `feet_stumble` 改名对应）；
  2) 删掉 V1 没有的奖励函数（V1 之后新增/改名的 17 项）；
  3) 门控第 5 维身高命令的写入点（V1 只有 4 维命令 ⇒ `commands[:, 4] = …` 会越界）。

用法：
    python3 tools/revert_rewards_to_v1.py --dry-run
    python3 tools/revert_rewards_to_v1.py
"""
import argparse
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUR = os.path.join(ROOT, 'legged_gym/legged_gym/envs/base/legged_robot.py')
TPL = os.path.expanduser('~/rl/HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py')

V1 = ("tracking_lin_vel tracking_ang_vel lin_vel_z ang_vel_xy orientation dof_acc joint_power "
      "base_height foot_clearance action_rate smoothness torques dof_vel collision termination "
      "dof_pos_limits dof_vel_limits torque_limits feet_air_time feet_stumble stand_still").split()
# HIMLoco 里的函数名映射（V1 的 feet_stumble 在祖先里叫 stumble）
TPL_NAME = {n: ('stumble' if n == 'feet_stumble' else n) for n in V1}

MARK = '    # 【V1 回退 2026-10-08】实现取自 V1 直系祖先 HIMLoco（原版注释见 logs/_backup_20261008_preV1/）\n'


def block(src, name, indent='    '):
    """取出 `def _reward_<name>(self):` 及其上方紧邻的注释行。"""
    m = re.search(r'(?ms)^%sdef _reward_%s\(self\):.*?(?=^%sdef |^    # =|\Z)'
                  % (indent, re.escape(name), indent), src)
    if not m:
        return None
    start = m.start()
    # 往上吃掉紧邻的注释行（它们描述的是这个函数，替换后必须一起走）
    lines = src[:start].split('\n')
    k = len(lines) - 1
    while k > 0 and lines[k - 1].strip().startswith('#'):
        k -= 1
    start = sum(len(x) + 1 for x in lines[:k])
    return src[start:m.end()], start, m.end()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    cur = open(CUR, encoding='utf-8').read()
    tpl = open(TPL, encoding='utf-8').read()

    replaced, missing, deleted = [], [], []

    # ---- 1) 替换 21 个函数体 ----
    for name in V1:
        tb = block(tpl, TPL_NAME[name])
        if tb is None:
            missing.append(name)
            continue
        body = tb[0]
        if name == 'feet_stumble':                    # 改名对应
            body = body.replace('def _reward_stumble(self):', 'def _reward_feet_stumble(self):')
        cb = block(cur, name)
        if cb is None:
            missing.append(name + '(当前代码里找不到)')
            continue
        if body.strip() == cb[0].strip():
            continue
        cur = cur[:cb[1]] + MARK + body + '\n' + cur[cb[2]:]
        replaced.append(name)

    # ---- 2) 删除 V1 没有的奖励函数 ----
    have = re.findall(r'^    def _reward_([a-z_0-9]+)\(self\)', cur, re.M)
    for name in have:
        if name in V1:
            continue
        cb = block(cur, name)
        if cb is None:
            continue
        deleted.append(name)
        cur = cur[:cb[1]] + cur[cb[2]:]

    # ---- 3) 门控第 5 维身高命令的写入点 ----
    gates = []
    pat = "        self.commands[env_ids, 4] = torch_rand_float(h_lo, h_hi, (len(env_ids), 1), device=self.device).squeeze(1)\n"
    if pat in cur:
        cur = cur.replace(pat,
            "        # 【V1 回退】V1 只有 4 维命令（无身高通道）⇒ 5 维时才采样身高\n"
            "        if self.cfg.commands.num_commands >= 5:\n"
            "            self.commands[env_ids, 4] = torch_rand_float(h_lo, h_hi, (len(env_ids), 1), device=self.device).squeeze(1)\n")
        gates.append('commands[env_ids, 4]（身高采样）')
    pat2 = "                self.commands[stand_ids, 4] = float(getattr(self.cfg.commands, 'height_nominal', 0.28))"
    if pat2 in cur:
        cur = cur.replace(pat2,
            "                if self.cfg.commands.num_commands >= 5:\n"
            "                    self.commands[stand_ids, 4] = float(getattr(self.cfg.commands, 'height_nominal', 0.28))")
        gates.append('commands[stand_ids, 4]（站立样本身高）')

    print('=== 替换为 V1/HIMLoco 实现: %d 个 ===' % len(replaced))
    print('   ', ' '.join(replaced))
    print('=== 删除（V1 没有的奖励函数）: %d 个 ===' % len(deleted))
    print('   ', ' '.join(deleted))
    print('=== 门控第 5 维命令: %d 处 ===' % len(gates))
    for g in gates:
        print('   ', g)
    if missing:
        print('⚠️ 没找到:', missing)
    left = re.findall(r'^    def _reward_([a-z_0-9]+)\(self\)', cur, re.M)
    print('\n剩余奖励函数 %d 个: %s' % (len(left), ' '.join(sorted(left))))

    if args.dry_run:
        print('\n(dry-run，未写入)')
        return 0

    bk = os.path.join(ROOT, 'logs', '_backup_20261008_preV1')
    os.makedirs(bk, exist_ok=True)
    shutil.copy2(CUR, os.path.join(bk, 'legged_robot.py.before_revert'))
    open(CUR, 'w', encoding='utf-8').write(cur)
    print('\n✅ 已写入 %s（备份 %s/legged_robot.py.before_revert）' % (CUR, bk))
    return 0


if __name__ == '__main__':
    sys.exit(main())
