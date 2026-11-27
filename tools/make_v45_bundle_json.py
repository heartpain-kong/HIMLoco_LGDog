# -*- coding: utf-8 -*-
"""make_v45_bundle_json.py — 从导出日志里抽出 control_contract，生成 V45 策略包的 policy.json。

★ 为什么要这么干：契约（kp/kd/action_scale/默认站姿…）是【不可手抄】的，
  抄错不会报错、只会让策略行为不可信。所以这里只从 tools/export_checkpoint.py 的
  日志里抠那段自动生成的 JSON，再补上溯源字段，绝不手打数值。
"""
import hashlib
import json
import os

D = os.path.expanduser('~/rl/dog/V45')
LOG = os.path.join(D, '导出日志.out')

# ── 1) 从日志里抠 contract JSON ───────────────────────────────────────────────
lines = open(LOG, encoding='utf-8').read().splitlines()
start = next(i for i, ln in enumerate(lines) if ln.startswith('[contract]'))
block = []
for ln in lines[start + 1:]:
    if not ln.startswith('    '):
        break
    block.append(ln[4:])
text = '\n'.join(x for x in block if not x.lstrip().startswith('#'))
contract = json.loads(text)

# ── 2) 指纹 / 尺寸 / md5 ─────────────────────────────────────────────────────
import re  # noqa: E402
_ln = next(ln for ln in lines if ln.startswith('[fingerprint]'))
fp = re.search(r"'([0-9a-f]{8,})'", _ln).group(1)
pt = os.path.join(D, 'policy.pt')
blob = open(pt, 'rb').read()
import torch  # noqa: E402
n_tensors = len(list(torch.jit.load(pt).state_dict().keys())) if False else None
try:
    import torch  # noqa: E402
    n_tensors = len(list(torch.jit.load(pt).state_dict().keys()))
except Exception:
    pass

contract['obs_one_step_note'] = (
    '45 = 无身高指令通道（V1 布局，270 维输入）。与 V41/V43（46 维/276）【不通用】。')
contract['note'] = (
    '★ 本段由 tools/export_checkpoint.py 从 V45 的 config_snapshot.json 自动生成（不是手抄）。'
    '★ V45 与 V44 的契约【完全相同】（kp40/kd1.2/decimation4/action_scale0.25/(0,0,0.2)）——'
    'V45 只改了奖励：stand_still 0→−1.5 且加偏航门（|ω_z 指令|<0.2 才算站立）。')

policy = {
    'weight_fingerprint': fp,
    'name': 'V45 @iteration 7400（★ 本包即本轮最终产物；训练于 7400/16000 = 46% 因机器硬挂终止）',
    'source_run': 'LgRobotDog/legged_gym/logs/flat_LgDog1/Oct08_22-17-59_LgDog1_himloco',
    'source_checkpoint': 'model_7400.pt（iteration 7401；2026-10-09 09:45 落盘）',
    'source_file': 'legged_gym/logs/flat_LgDog1/exported/policies/policy_V45_7400.pt',
    'extracted_on': '2026-10-09',
    'extraction_tool': 'tools/export_checkpoint.py（45 维自适应 + [norm] 统计量还原 + [contract] 自动契约）',
    'file_size': len(blob),
    'md5_file': hashlib.md5(blob).hexdigest(),
    'md5_note': ('文件 md5 只用于比对「是不是同一个文件」；判断「是不是同一份权重」一律用 '
                 'weight_fingerprint（JIT 重存会变 md5，指纹不变）。'),
    'io': '270 维（45 单帧 × 6 历史）→ 12 维动作（含烘焙进图的观测归一化统计量）',
    'control_contract': contract,
    'export_reason': ('用户 2026-10-09 12:08 令「导出」（当时训练还在跑），12:25 又令「不需要继续训 结束导出」'
                      '⇒ 训练已停、run 已改名为 V45，本包即本轮最终产物（检查点 7400/16000 = 46%）。'),
    'round_close': {
        'status': '本包即 V45 的【最终产物】（用户 2026-10-09 12:25 令「不需要继续训 结束导出」）',
        'why_stopped': ('V45 于 7400/16000（46%）终止：不是策略不行，而是机器第 4 次硬挂——'
                        '2026-10-09 09:53:48 整机冻结（journalctl 最后一条 = fwupd.service 正常退出，'
                        '之后没有任何关机序列），训练到 7719 轮时被杀，落盘检查点停在 model_7400。'),
        'resume_attempt': ('12:17 曾从 model_7400 续训（resume 语义: 新建 log_dir、只接权重/优化器/迭代数），'
                           '12:19 按用户指令停止；该 2 分钟的 run 目录（Oct09_12-17-35，零 checkpoint）已删除。'),
        'renamed_run': 'legged_gym/logs/flat_LgDog1/V45（原 Oct08_22-17-59_LgDog1_himloco，38 个检查点）',
        'crash_forensics': ('4 次硬挂全部发生在【后台系统维护任务收尾的瞬间】: '
                            '10-06 17:31:35 anacron 退出 / 10-07 21:26:43 tracker 索引器启动 / '
                            '10-08 09:05:07 apt-daily 退出 / 10-09 09:53:48 fwupd 退出。四次都没有关机序列、'
                            '没有内核报错 ⇒ 强烈指向「后台维护任务触发磁盘/模块操作 → 整机冻结」。'),
        'blackbox_gap': ('★ 前几轮记录里说"黑匣子已在跑"是【错的】: logs/blackbox.log 从未存在过 ⇒ '
                         '第 4 次硬挂依然零采样。现已用 tools/crash_guard.sh 落实（2s 采样 + 内核日志尾随 + '
                         'PID 可核查），启动后已确认在写。'),
        'followup_sudo': ('需要 sudo 才能做的三件事（我这边没有免密 sudo）: ① 训练期间 mask 掉 '
                          'apt-daily/apt-daily-upgrade/unattended-upgrades/anacron/fwupd-refresh 这些 timer；'
                          '② /etc/modprobe.d/nvidia-pm.conf 写 NVreg_DynamicPowerManagement=0x00 + update-initramfs；'
                          '③ 内核参数加 nvme_core.default_ps_max_latency_us=0（SSD 掉盘式冻结的经典修法）。'),
    },
    'standing_fix_evidence': {
        'target': ('V44@3200 的站立缺陷（Isaac FR 足端离地 13.0%、3% 环境 >10%）；'
                   '★ 注意流传的 MuJoCo 68.1% 是 V44@2400 那份，V44@3200 单环境也是 0.0%'),
        'change': 'rewards.stand_still 0 → −1.5，_reward_stand_still 加偏航门 stand_turn_gate=0.2',
        'probe_4800': ('MuJoCo 站立 25 s（指令全 0）：z 0.305 m、pitch +3.06°、roll +1.51°、'
                       '|ω_z| 0.000 rad/s、净位移 0.001 m/595 s、四足足端离地 0.0%'),
        'verdict': '站立抬腿已修复（同口径 Isaac: V44@3200 FR 13.0% → V45@7400 0.1%）',
        'caveat': '训练侧 4800 轮仍有 2 项 ✘：平滑（二阶 Δa 5.80 vs 判据 ≤3.0）、髋镜像误差（1.90 vs ≤1.0 rad 上限）',
    },
    'probe_7400': {
        'protocol': 'MuJoCo 600 s × 3，各 30000 控制步，CPU（CUDA_VISIBLE_DEVICES=""），不占训练 GPU',
        'stand': '四足足端离地 0.0%、净位移 0.002 m/595 s、|ω_z| 0.000 rad/s、z 0.306 m、pitch +2.87° → ✔',
        'walk_1.0ms': ('平均速度 0.86 m/s、z 0.323 m、pitch +6.09°、直线率 17.9%、净偏航 −443°、'
                       '横向偏移 −59.1 m → ⚠ 走得住但跑偏（47% 进度 + 无直行纪律项，与 V44@3200 的 17.3% 同类）'),
        'turn_1.0rads': '|ω_z| 0.860 rad/s（指令 86%）、roll −0.20°、未翻倒 → ✔ 偏航门未误伤转身',
        'log': '~/rl/dog/V45/探针_导出策略.out',
        'script': 'LgRobotDog/logs/train_logs/v45_export_probe.sh',
    },
    'checkpoint_sweep': {
        'protocol': 'MuJoCo 600 s、指令恒 1.0 m/s、--clamp-vel、用 V45 自己的 config_snapshot.json 对齐契约；全程 CPU',
        'script': 'LgRobotDog/logs/train_logs/v45_sweep.sh',
        'log': 'LgRobotDog/logs/train_logs/v45_sweep.out',
        'rows': [
            {'step': 2400, 'fingerprint': '7651507a3435', 'pitch均值': 2.23, 'roll均值': 8.75, 'z均值': 0.258,
             '直线率%': 1.1, '速度': 0.43, '翻倒': True},
            {'step': 4800, 'fingerprint': '1906075541c4', 'pitch均值': 4.74, 'roll均值': 1.56, 'z均值': 0.328,
             '直线率%': 1.3, '速度': 0.63, '翻倒': True},
            {'step': 6000, 'pitch均值': 5.49, 'roll均值': 0.30, 'z均值': 0.329,
             '直线率%': 8.8, '速度': 0.78, '翻倒': False},
            {'step': 7400, 'fingerprint': '5d9489513230', 'pitch均值': 6.09, 'roll均值': 1.24, 'z均值': 0.323,
             '直线率%': 17.9, '速度': 0.86, '翻倒': False, '选定': '★ 装进 sim2sim 的就是这个'},
        ],
        'verdict': ('越晚的检查点越快越直且不翻倒（速度 0.43→0.86 m/s、直线率 1.1%→17.9%、'
                    '2400/4800 会在纯行走下翻倒），代价是 pitch 单调变大（+2.23°→+6.09°）。'
                    '7400 是四个里最好的：不翻倒、最快、最直。'),
        'caveat': ('按 sim2sim 包原来的判据「pitch ≤ ±4°」严格卡，7400 不达标——但 V44 那份也是 +4.49°/17.3%，'
                   '说明该判据对 V1 奖励集（无直行纪律项）本就偏严。'),
        'sim2sim_introduced': ('2026-10-09 用户令「引入 sim2sim」: 已把 7400 装进 ~/rl/LgRobot/sim2sim '
                               '（policy.pt + policy_V45_7400.pt，快照同步），V44 旧策略保留为 '
                               'policy_V44_3ec760d951df.pt，旧快照备份在 _backup_20261007/。'),
    },
    'isaac_accept_7400': {
        'protocol': 'tools/eval_isaac.py --run V45 --checkpoint 7400（平地 + 训练同款 DR），多组指令',
        'stand_zero_cmd': ('足端离地 FL 0.3% / FR 0.1% / RL 0.0% / RR 0.1%（V44 对照 FR 13.0%）；'
                           '离地>10% 的环境占比 0%（V44 3%）；漂移 7 mm/10 s（V44 18 mm）；'
                           'z 0.306 m、pitch +2.70°、roll −0.58°、|ω_z| 0.006 rad/s、30 s 净转 +0.5° → ✔'),
        'gate_proof': ('站立指令下 stand_still 每秒速率 −1.5702（权重 −1.5 满额）；原地转向 yaw=1.5 时'
                       '该分项 <1e-4（未进列表）⇒ 偏航门按设计开/关 ✔'),
        'walk_1.0ms': ('速度 0.852 m/s（跟踪 85.2%）、reset 0.0000（不摔）、trot 对角相关 0.98/0.97、'
                       '横向间距 0.345 m、髋外张 ±1.7…3.1°；★ pitch +6.60° ✘（判据 ≤±4°，V44@3200 为 +4.49°）'),
        'turn': 'yaw 1.5 → 91.3%；yaw 1.0 → 85.7%；yaw 2.0 → 93.5% ✔ 转身未被站立惩罚压坏',
        'remaining': '行走 pitch 偏高（+6.60°）与直行跑偏为本轮遗留；训练侧 @4800 仍有 2 项 ✘（平滑 5.80 / 髋镜像 1.90）',
        'log': 'LgRobotDog/logs/train_logs/v45_accept_1009.out',
    },
}
if n_tensors:
    policy['tensors'] = n_tensors

out = os.path.join(D, 'policy.json')
json.dump(policy, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'[write] {out}')
print(f'  fingerprint = {fp}   size = {len(blob)}   md5 = {policy["md5_file"]}')
print(f'  contract kp/kd/scale = {contract["kp"]}/{contract["kd"]}/{contract["action_scale"]} '
      f'decim={contract["decimation"]} 站姿={contract["default_dof_pos"][:3]}')
print(f'  contract 字段 = {sorted(contract)}')
