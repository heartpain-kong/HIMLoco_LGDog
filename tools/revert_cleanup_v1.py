#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""revert_cleanup_v1.py —— V1 回退的收尾清理（terrain / normalization / 死代码）

在 tools/revert_config_to_v1.py + revert_rewards_to_v1.py 之后，把 V1 里并不存在、
但还留在配置与代码里的东西一并清掉：

  配置（LgDog1_config.py）
    · terrain: platform_* / curriculum_by / climb_frac / climb_only_obstacle_cols /
      min_terrain_level（含 LgDog1FlatCfg 里的覆盖）—— 删掉后代码按 getattr 默认值走，
      即 V1 的『按行走距离晋升』课程（curriculum_by='distance'）、无平台地形、无最低等级钳制。
    · normalization.obs_scales.height_cmd（4 维命令下用不到）

  代码（legged_robot.py）
    · 删掉 V1 之后新增、现已无对应奖励函数的辅助与状态：
      _turn_intent / _move_intent / _swing_clearance / _hip_indices / _update_heading_ref、
      stand_flag（站立样本注入）、heading_ref（straight_hold 参考朝向）、
      open_loop_mask（V43 开环朝向暴露）、sl_*（step_length）、sf_*（step_freq）、gs_*（gait_symmetry）
    · 连同 _post_physics_step_callback 与 _resample_commands 里的调用点、以及 _init_buffers 里的缓冲创建

用法：
    python3 tools/revert_cleanup_v1.py --dry-run
    python3 tools/revert_cleanup_v1.py
"""
import argparse
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, 'legged_gym/legged_gym/envs/LgDog1/LgDog1_config.py')
ROB = os.path.join(ROOT, 'legged_gym/legged_gym/envs/base/legged_robot.py')
BK = os.path.join(ROOT, 'logs', '_backup_20261008_preV1')

TERRAIN_FIELDS = ['platform_height_min', 'platform_height_max', 'platform_band_depth',
                  'platform_band_gap', 'platform_spawn_size', 'curriculum_by', 'climb_frac',
                  'climb_only_obstacle_cols', 'min_terrain_level']


def drop_field_lines(text, field):
    """删掉 `field = ...` 那一行，以及紧邻其上的注释行（只删属于它的那段）。"""
    out, removed = [], 0
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r'^(\s*)%s\s*=' % re.escape(field), ln)
        if not m:
            out.append(ln)
            i += 1
            continue
        indent = m.group(1)
        # 往上回退，吃掉紧邻的注释行（缩进相同或更深，且不是空行）
        k = len(out)
        while k > 0 and out[k - 1].strip().startswith('#') and \
                (len(out[k - 1]) - len(out[k - 1].lstrip())) >= len(indent):
            k -= 1
        del out[k:]
        removed += 1
        i += 1
    return '\n'.join(out), removed


def drop_method(text, name):
    """删掉 `def <name>(self)...` 整块（含上方紧邻注释）与其后的一个空行。"""
    m = re.search(r'(?ms)^(    )def %s\(self.*?(?=^    def |^    @|^    # =|\Z)' % re.escape(name), text)
    if not m:
        return text, 0
    start = m.start()
    lines = text[:start].split('\n')
    k = len(lines) - 1
    while k > 0 and lines[k - 1].strip().startswith('#'):
        k -= 1
    start = sum(len(x) + 1 for x in lines[:k])
    end = m.end()
    while end < len(text) and text[end] == '\n':
        end += 1
    return text[:start] + text[end:], 1


def drop_exact(text, snippet, label, cnt):
    if snippet in text:
        cnt.append(label)
        return text.replace(snippet, '', 1)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    cfg = open(CFG, encoding='utf-8').read()
    rob = open(ROB, encoding='utf-8').read()
    report = []

    # ---------------- 配置 ----------------
    for f in TERRAIN_FIELDS:
        cfg, n = drop_field_lines(cfg, f)
        if n:
            report.append('config: 删除 terrain.%s ×%d' % (f, n))
    cfg, n = drop_field_lines(cfg, 'height_cmd')
    if n:
        report.append('config: 删除 normalization.obs_scales.height_cmd ×%d' % n)

    # ---------------- 代码：调用点 ----------------
    hit = []
    # (1) 站立锁定 + 开环暴露 + heading_ref 维护（_post_physics_step_callback 内）
    def cut_between(text, start_pat, end_pat, label):
        """删掉 [start_pat 起点, end_pat 起点) 之间的所有内容（含两端之间的注释/代码）。"""
        i = text.find(start_pat)
        j = text.find(end_pat, i + 1) if i >= 0 else -1
        if i < 0 or j < 0:
            return text, False
        return text[:i] + text[j:], True

    # (1) 站立锁定 + 开环暴露 + heading_ref 维护：从该段注释删到 measure_heights 那段之前
    rob, ok1 = cut_between(rob, '            # 【2026-10-06 修·破',
                           '        if self.cfg.terrain.measure_heights:', 'post_physics 块')
    # (2) 身高采样 + 站立样本注入：从该段注释删到 _compute_torques 之前
    rob, ok2 = cut_between(rob, '        # 【2026-10-01】身高指令（第 5 维，index 4）',
                           '    def _compute_torques(self, actions):', 'resample 块')
    hit += [l for l, ok in (('code: post_physics 块(站立锁定/开环/heading_ref)', ok1),
                            ('code: resample 块(身高采样/站立注入)', ok2)) if ok]
    rob = drop_exact(rob, "        p_stand = float(getattr(self.cfg.commands, 'stand_command_prob', 0.0))\n",
                     'code: p_stand 行', hit)
    rob = drop_exact(rob, "        _ol = float(getattr(self.cfg.commands, 'open_loop_heading_frac', 0.0) or 0.0)\n",
                     'code: open_loop 行(A)', hit)
    # (3) _init_buffers 里为已删奖励创建的缓冲
    for snip, label in [
        ("        self.sf_cycle = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)\n", 'sf_cycle'),
        ("        self.sf_last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)\n", 'sf_last_contacts'),
        ("        self.stand_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)\n", 'stand_flag'),
        ("        self.gs_air = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)\n", 'gs_air'),
        ("        self.gs_last_air = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)\n", 'gs_last_air'),
        ("        self.gs_last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)\n", 'gs_last_contacts'),
        ("        self.heading_ref = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)\n", 'heading_ref'),
    ]:
        rob = drop_exact(rob, snip, 'code: 缓冲 %s' % label, hit)
    # (4) _resample_commands 里的开环掩码采样块（V43 加的，V1 没有）
    rob, ok3 = cut_between(rob,
        '        # ★ 2026-10-07【开环朝向暴露】挑哪批环境在直行段不给 yaw 指令',
        '    def _compute_torques(self, actions):', 'resample 开环块')
    if ok3:
        hit.append('code: resample 开环掩码采样块')
    # (5) 其余零散行（stand_flag 清零等）
    for snip, label in [
        ("        # 本批重采样的 env 先一律清掉『站立样本』标记（下面 stand 注入会重新置位）\n", '注释'),
        ("        self.stand_flag[env_ids] = False\n", 'stand_flag 清标记'),
    ]:
        rob = drop_exact(rob, snip, 'code: %s' % label, hit)
    # 注释行（提到已删缓冲/奖励的段落注释）
    for pat in (r'(?m)^\s*#.*(sl_liftoff|sl_air|sl_last|sf_cycle|sf_last|gs_air|gs_last|stand_flag|heading_ref|open_loop|step_length|step_freq|gait_symmetry|straight_hold|axis_purity|diagonal_sync|hip_mirror|_reward_feet_air_time 拥有).*\n',):
        rob = re.sub(pat, '', rob)
    report += hit

    # ---------------- 代码：辅助方法 ----------------
    for name in ('_turn_intent', '_move_intent', '_swing_clearance', '_hip_indices', '_update_heading_ref'):
        rob, n = drop_method(rob, name)
        if n:
            report.append('code: 删除方法 %s' % name)

    print('=== 清理项 %d 条 ===' % len(report))
    for r in report:
        print('  ', r)
    left = {k: len(re.findall(k, rob)) for k in
            ('stand_flag', 'heading_ref', 'open_loop_mask', 'sf_cycle', 'gs_air', '_turn_intent',
             '_move_intent', '_swing_clearance', '_hip_indices', '_update_heading_ref')}
    print('=== 代码里残留引用 ===')
    for k, v in left.items():
        if v:
            print('   ⚠️ %-22s %d 处' % (k, v))
            for ln in rob.split('\n'):
                if k in ln:
                    print('        %s' % ln.strip()[:110])
    if not any(left.values()):
        print('   无 ✔')

    if args.dry_run:
        print('\n(dry-run，未写入)')
        return 0
    os.makedirs(BK, exist_ok=True)
    shutil.copy2(CFG, os.path.join(BK, 'LgDog1_config.py.before_cleanup'))
    shutil.copy2(ROB, os.path.join(BK, 'legged_robot.py.before_cleanup'))
    open(CFG, 'w', encoding='utf-8').write(cfg)
    open(ROB, 'w', encoding='utf-8').write(rob)
    print('\n✅ 已写入（备份 *.before_cleanup）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
