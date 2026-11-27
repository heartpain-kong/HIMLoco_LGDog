#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_criteria.py —— 把训练侧的奖励项【换算成轮次判据】并给出达标判定（不与训练抢 GPU）。

为什么需要它: 中途监控点不能跑 Isaac 评测（8GB 卡并发会 OOM/Xid，2026-10-02 事故），
所以每个监控点只能看训练侧曲线。但奖励项本身就是判据的代理量，换算一下就能"自动检测":

  V40 的换算关系（权重见 LgDog1_config.py）:
    |vx| 侧移串扰      = |rew_axis_purity| / 1.0      ← 该项 = 纯横移 env 的 |vx|
    |Δ朝向| 直行偏离    = |rew_straight_hold| / 0.3    ← 直行 env 的 |wrap(yaw−参考)|
    步态左右时长差      = |rew_gait_symmetry|（每秒速率，单调趋 0 即对称性在改善；稀疏事件项，不换算成秒）
    |ω_z| 不自转        = |rew_yaw_hold| / 0.8         ← 未转向 env（含站立）的 |ω_z|
用法:
  python tools/check_criteria.py --run V40            # 用 logs/flat_LgDog1/V40
  python tools/check_criteria.py --run Oct06_12-31-04_LgDog1_himloco --round V40
"""
import argparse
import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

# 轮次判据表: (显示名, 奖励项名, 缩放(判据=值/缩放), 判据上限, 越小越好, 基线说明)
CRITERIA = {
    'V40': [
        ('侧移串扰 |vx|        ', 'rew_axis_purity',    1.0, 0.06, 'V39 基线 0.131/0.214'),
        ('直行朝向偏离 |Δ|     ', 'rew_straight_hold',  0.3, 0.03, 'V39 基线 ≈0.05 rad'),
        ('不自转 |ω_z|         ', 'rew_yaw_hold',       0.8, 0.05, 'V39 基线 0.049（含站立+直行）'),
        ('步态左右时长差(速率) ', 'rew_gait_symmetry',  0.0, 0.02, 'V39 无此项；越接近 0 越好'),
    ],
    'V44': [   # V1 奖励集 + 用户启用/微调的 OpenDoge 项；★ 阈值按【物理量】标定（不是拍脑袋的 0.05）
        # 写法: (显示名, 奖励项, 缩放, 上限, 说明) —— 判据 = |速率| / 缩放，再把物理意义写在说明里
        ('姿态 倾斜rms(rad)     ', 'rew_orientation',      1.5,  0.060, '= sqrt(|值|/1.5) ⇒ 上限 0.060 即 rms 倾斜 ≤0.20 rad(11.5°)'),
        ('高度 偏差(m)          ', 'rew_base_height',      1.5,  0.050, '= sqrt(|值|/1.5)/10 ⇒ 上限 0.05 即高度偏差 ≤0.058 m'),
        ('跳弹 vz_rms(m/s)      ', 'rew_lin_vel_z',        2.4,  0.050, '= sqrt(|值|/2.4) ⇒ 上限 0.05 即竖直速度 rms ≤0.144 m/s'),
        ('平滑 二阶Δa           ', 'rew_smoothness',       0.08, 3.000, '= |值|/0.08 = Σ(二阶差分)² ⇒ 上限 3.0 即每关节 rms ≤0.50'),
        ('对角同步 error        ', 'rew_diagonal_sync',    0.2,  1.000, '= |值|/0.2（两组对角触地不一致之和）⇒ 上限 1.0 即 0.2'),
        ('髋镜像 err(rad)       ', 'rew_hip_mirror_symmetry', 0.05, 1.000, '= sqrt(|值|/0.05)/10 ⇒ 上限 1.0 即髋左右镜像误差 ≤0.10 rad(6°)'),
        ('姿态锚 Σ|q−q_def|(rad)', 'rew_default_pos_linear', 0.05, 36.0, '= |值|/0.05 ⇒ 上限 36 即 12 关节 Σ 偏差 ≤1.8 rad（每关节 ≤0.15）'),
    ],
    'V51': [   # ★ 2026-10-12 22:3x 用户又改了 3 项并把这一轮重开（旧 run 只跑了 1155 轮就作废）:
        #   orientation −1.2→−1.5、smoothness −0.1→−0.04、action_rate −0.06→−0.04
        #   （其余沿用 V50→V51 那批: collision/default_joint_pos/default_pos_linear 归零、
        #     lin_vel_z −0.8、ang_vel_xy −0.02、hip_joint_pos 0.01、feet_parallel_to_ground 0.1、
        #     tracking_sigma 0.18、tracking_lin_vel 1.6、max_curriculum 1.0）
        # ★ 第 3 列 = 该奖励项【当前实际权重】；权重 0 的项判据退化成"看绝对值/趋势"（脚本自动标 ·）。
        # ⚠️ 训练侧能反映"姿态/步态"的项已所剩无几（高度/锚/对角/髋镜像/碰撞全为 0），
        #   验收必须靠 Isaac/MuJoCo 实测（V50 的教训: 判据全绿但 MuJoCo 直行直线率只有 3%）。
        ('姿态 倾斜rms(rad)     ', 'rew_orientation',      1.5,  0.060, '= sqrt(|值|/1.5) ⇒ 上限 0.060 即 rms 倾斜 ≤0.20 rad(11.5°)（权重 −1.2→−1.5 略微加严）'),
        ('跳弹 vz_rms(m/s)      ', 'rew_lin_vel_z',        0.8,  0.050, '= sqrt(|值|/0.8) ⇒ 上限 0.05 即竖直速度 rms ≤0.141 m/s'),
        ('平滑 二阶Δa           ', 'rew_smoothness',       0.04, 3.000, '= |值|/0.04 = Σ(二阶差分)² ⇒ 上限 3.0 即每关节 rms ≤0.50（★ 权重 −0.1→−0.04，判据随之换算，别和 V50 同行直接比）'),
        ('落地惩罚(速率)        ', 'rew_termination',      0.5,  0.010, '= |值|/0.5 ⇒ 摔倒/终止频率，越小越好'),
        ('★已关: 高度/锚/对角/髋镜像', 'rew_default_pos_linear', 0.0, 36.0, '★ 这几项权重为 0（日志里已不记录）⇒ 训练侧不再约束姿态锚/高度/对角同步/髋镜像'),
    ],
    'V50': [   # V49 基础上用户又放松 8 项: orientation −2.0→−1.5、tracking_lin_vel 2.0→1.6、
        #   base_height −1.0→0（关）、diagonal_sync −0.1→0（关）、hip_mirror_symmetry −0.05→0（关）、
        #   default_joint_pos/default_pos_linear −0.02→−0.01、commands.max_curriculum 2.0→1.0（指令上限降）
        # ★ 第 3 列 = 该奖励项【当前实际权重】；权重 0 的项判据退化成"看绝对值/趋势"（脚本会自动标 ·）。
        ('姿态 倾斜rms(rad)     ', 'rew_orientation',      1.5,  0.060, '= sqrt(|值|/1.5) ⇒ 上限 0.060 即 rms 倾斜 ≤0.20 rad(11.5°)（权重 −2.0→−1.5 更松；重点看 Isaac 实测 pitch，别只看这条达标线）'),
        ('高度 偏差(速率)       ', 'rew_base_height',      0.0,  0.050, '★ 本轮权重归零（该项已关）⇒ 只能看绝对值/趋势，不能再换算成"高度偏差多少米"'),
        ('跳弹 vz_rms(m/s)      ', 'rew_lin_vel_z',        1.0,  0.050, '= sqrt(|值|/1.0) ⇒ 上限 0.05 即竖直速度 rms ≤0.20 m/s'),
        ('平滑 二阶Δa           ', 'rew_smoothness',       0.1,  3.000, '= |值|/0.1 = Σ(二阶差分)² ⇒ 上限 3.0 即每关节 rms ≤0.50'),
        ('对角同步(速率)        ', 'rew_diagonal_sync',    0.0,  1.000, '★ 本轮权重归零（已关）⇒ 只看绝对值/趋势'),
        ('髋镜像(速率)          ', 'rew_hip_mirror_symmetry', 0.0, 1.000, '★ 本轮权重归零（已关）⇒ 只看绝对值/趋势'),
        ('姿态锚 Σ|q−q_def|(rad)', 'rew_default_pos_linear', 0.01, 36.0, '= |值|/0.01 ⇒ 上限 36 即 12 关节 Σ 偏差 ≤0.36 rad（权重再减半 ⇒ 锚继续放松）'),
        ('落地惩罚(速率)        ', 'rew_termination',      0.5,  0.010, '= |值|/0.5 ⇒ 摔倒/终止频率，越小越好'),
    ],
    'V49': [   # V48 基础上用户又调 10 项（多为次级项放松）: diagonal_sync −0.3→−0.1、
        #   feet_stumble −0.4→−0.1、collision −0.2→−0.1、default_joint_pos −0.05→−0.02、
        #   default_pos_linear −0.05→−0.02、lin_vel_z −1.5→−1.0、ang_vel_xy −0.01→−0.03、
        #   feet_parallel_to_ground 0.1→0.05、hip_joint_pos 0.1→0.05、base_height_linear −0.01→0
        # ★ 第 3 列 = 该奖励项【当前实际权重】（判据 = |速率| / 权重）。
        # ⚠️ 权重每轮都在变 ⇒ 同一行的数字跨轮【不可直接比】，只能看是否达标与趋势。
        ('姿态 倾斜rms(rad)     ', 'rew_orientation',      2.0,  0.060, '= sqrt(|值|/2.0) ⇒ 上限 0.060 即 rms 倾斜 ≤0.20 rad(11.5°)（权重回到 −2.0：更容易达标，但姿态本身会更差，重点看绝对值）'),
        ('高度 偏差(m)          ', 'rew_base_height',      1.0,  0.050, '= sqrt(|值|/1.0)/10 ⇒ 上限 0.05 即高度偏差 ≤0.058 m'),
        ('跳弹 vz_rms(m/s)      ', 'rew_lin_vel_z',        1.0,  0.050, '= sqrt(|值|/1.0) ⇒ 上限 0.05 即竖直速度 rms ≤0.20 m/s（权重 −1.5→−1.0，判据随之放宽）'),
        ('平滑 二阶Δa           ', 'rew_smoothness',       0.1,  3.000, '= |值|/0.1 = Σ(二阶差分)² ⇒ 上限 3.0 即每关节 rms ≤0.50'),
        ('对角同步 error        ', 'rew_diagonal_sync',    0.1,  1.000, '= |值|/0.1 ⇒ 上限 1.0 即 0.1（权重 −0.3→−0.1：同值下允许的物理误差大三倍）'),
        ('髋镜像 err(rad)       ', 'rew_hip_mirror_symmetry', 0.05, 1.000, '= sqrt(|值|/0.05)/10 ⇒ 上限 1.0 即髋左右镜像误差 ≤0.10 rad(6°)'),
        ('姿态锚 Σ|q−q_def|(rad)', 'rew_default_pos_linear', 0.02, 36.0, '= |值|/0.02 ⇒ 上限 36 即 12 关节 Σ 偏差 ≤0.72 rad（权重减半 ⇒ 锚明显放松）'),
        ('落地惩罚(速率)        ', 'rew_termination',      0.5,  0.010, '= |值|/0.5 ⇒ 摔倒/终止频率，越小越好'),
    ],
    'V48': [   # V47 基础上用户再大改一轮（12 项）: episode 20→30s、orientation −3.5→−5.0、
        #           base_height −1.5→−1.0 且 target 0.28→0.30、tracking_lin_vel 2.5→2.0、
        #           tracking_sigma 0.2→0.25、action_rate −0.12→−0.06、ang_vel_xy −0.05→−0.01、
        #           feet_stumble −0.8→−0.4、foot_clearance −0.02→−0.01、stand_still −1.5→−1.0、
        #           clearance_height_target −0.20→−0.10（★ 隐含摆动净空目标 = 0.30−0.10 = 0.20 m）
        # ★ 第 3 列 = 该奖励项【当前实际权重】（判据 = |速率| / 权重）。
        ('姿态 倾斜rms(rad)     ', 'rew_orientation',      5.0,  0.060, '= sqrt(|值|/5.0) ⇒ 上限 0.060 即 rms 倾斜 ≤0.20 rad(11.5°)（本轮权重 −3.5→−5.0）'),
        ('高度 偏差(m)          ', 'rew_base_height',      1.0,  0.050, '= sqrt(|值|/1.0)/10 ⇒ 上限 0.05 即高度偏差 ≤0.058 m（权重 −1.5→−1.0，target→0.30 与被动站高一致）'),
        ('跳弹 vz_rms(m/s)      ', 'rew_lin_vel_z',        2.4,  0.050, '= sqrt(|值|/2.4) ⇒ 上限 0.05 即竖直速度 rms ≤0.144 m/s'),
        ('平滑 二阶Δa           ', 'rew_smoothness',       0.1,  3.000, '= |值|/0.1 = Σ(二阶差分)² ⇒ 上限 3.0 即每关节 rms ≤0.50'),
        ('对角同步 error        ', 'rew_diagonal_sync',    0.3,  1.000, '= |值|/0.3（两组对角触地不一致之和）⇒ 上限 1.0 即 0.3'),
        ('髋镜像 err(rad)       ', 'rew_hip_mirror_symmetry', 0.05, 1.000, '= sqrt(|值|/0.05)/10 ⇒ 上限 1.0 即髋左右镜像误差 ≤0.10 rad(6°)'),
        ('姿态锚 Σ|q−q_def|(rad)', 'rew_default_pos_linear', 0.05, 36.0, '= |值|/0.05 ⇒ 上限 36 即 12 关节 Σ 偏差 ≤1.8 rad（每关节 ≤0.15）'),
        ('落地惩罚(速率)        ', 'rew_termination',      0.5,  0.010, '= |值|/0.5 ⇒ 摔倒/终止频率，越小越好'),
        ('摆动净空亏欠(m²·m/s)  ', 'rew_foot_clearance',   0.01, 0.500, '★ 本轮 clearance_height_target −0.20→−0.10 ⇒ 隐含目标净空 0.08→0.20 m（要求抬脚翻倍多）；这项变小=抬得更够'),
    ],
    'V47': [   # V46 基础上用户再改权重（orientation −2.5→−3.5、tracking_lin_vel 2.0→2.5、
        #           ang_vel_xy −0.2→−0.05、tracking_sigma 0.22→0.2、指令回 ±1.0）
        # ★ 2026-10-10 修正：第 3 列【必须是该奖励项当前的实际权重】（判据 = |速率| / 权重，
        #   物理意义写在说明里）。V45/V46 的表里 smoothness 写 0.08、diagonal_sync 写 0.2，
        #   而实际权重是 0.1 / 0.3 ⇒ 那两行算出来的数字被缩小了 1.25 / 1.5 倍，跨轮不能直接比。
        #   V47 起按实际权重填，并逐项与 LgDog1_config.py 的 scales 对齐。
        ('姿态 倾斜rms(rad)     ', 'rew_orientation',      3.5,  0.060, '= sqrt(|值|/3.5) ⇒ 上限 0.060 即 rms 倾斜 ≤0.20 rad(11.5°)（本轮权重 −2.5→−3.5）'),
        ('高度 偏差(m)          ', 'rew_base_height',      1.5,  0.050, '= sqrt(|值|/1.5)/10 ⇒ 上限 0.05 即高度偏差 ≤0.058 m；★ target 0.28 与被动站高 0.305 冲突，只看趋势'),
        ('跳弹 vz_rms(m/s)      ', 'rew_lin_vel_z',        2.4,  0.050, '= sqrt(|值|/2.4) ⇒ 上限 0.05 即竖直速度 rms ≤0.144 m/s'),
        ('平滑 二阶Δa           ', 'rew_smoothness',       0.1,  3.000, '= |值|/0.1 = Σ(二阶差分)² ⇒ 上限 3.0 即每关节 rms ≤0.50（★ 权重实为 0.1，V45/V46 表写成 0.08 是错的）'),
        ('对角同步 error        ', 'rew_diagonal_sync',    0.3,  1.000, '= |值|/0.3（两组对角触地不一致之和）⇒ 上限 1.0 即 0.3（★ 权重实为 0.3，V45/V46 表写成 0.2 是错的）'),
        ('髋镜像 err(rad)       ', 'rew_hip_mirror_symmetry', 0.05, 1.000, '= sqrt(|值|/0.05)/10 ⇒ 上限 1.0 即髋左右镜像误差 ≤0.10 rad(6°)'),
        ('姿态锚 Σ|q−q_def|(rad)', 'rew_default_pos_linear', 0.05, 36.0, '= |值|/0.05 ⇒ 上限 36 即 12 关节 Σ 偏差 ≤1.8 rad（每关节 ≤0.15）'),
        ('落地惩罚(速率)        ', 'rew_termination',      0.5,  0.010, '= |值|/0.5 ⇒ 摔倒/终止频率，越小越好（V46 起开的项）'),
    ],
    'V46': [   # 同 V45（V1 奖励集 + OpenDoge 项 + stand_still=-1.5 带 yaw 门）
        # ★ 与 V45 的差别（2026-10-09 用户手改配置 + 阈值改动），判据阈值本身沿用 V45——
        #   本轮改动是"难度/纪律"方向（速度 ±1.2、摩擦下限 0.1、推力 2.0、termination −0.5、
        #   base_height_target 0.28、clearance_height_target −0.20、feet_air_time 阈值 0.05s），
        #   不影响这些判据的物理含义，所以同一张表可以跨 V45/V46 比。
        ('姿态 倾斜rms(rad)     ', 'rew_orientation',      1.5,  0.060, '= sqrt(|值|/1.5) ⇒ 上限 0.060 即 rms 倾斜 ≤0.20 rad(11.5°)'),
        ('高度 偏差(m)          ', 'rew_base_height',      1.5,  0.050, '= sqrt(|值|/1.5)/10 ⇒ 上限 0.05 即高度偏差 ≤0.058 m；★ 本轮 target 0.3→0.28，同值下物理偏差更大，看趋势'),
        ('跳弹 vz_rms(m/s)      ', 'rew_lin_vel_z',        2.4,  0.050, '= sqrt(|值|/2.4) ⇒ 上限 0.05 即竖直速度 rms ≤0.144 m/s'),
        ('平滑 二阶Δa           ', 'rew_smoothness',       0.08, 3.000, '= |值|/0.08 = Σ(二阶差分)² ⇒ 上限 3.0 即每关节 rms ≤0.50'),
        ('对角同步 error        ', 'rew_diagonal_sync',    0.2,  1.000, '= |值|/0.2（两组对角触地不一致之和）⇒ 上限 1.0 即 0.2'),
        ('髋镜像 err(rad)       ', 'rew_hip_mirror_symmetry', 0.05, 1.000, '= sqrt(|值|/0.05)/10 ⇒ 上限 1.0 即髋左右镜像误差 ≤0.10 rad(6°)'),
        ('姿态锚 Σ|q−q_def|(rad)', 'rew_default_pos_linear', 0.05, 36.0, '= |值|/0.05 ⇒ 上限 36 即 12 关节 Σ 偏差 ≤1.8 rad（每关节 ≤0.15）'),
        ('落地惩罚(速率)        ', 'rew_termination',      0.5,  0.010, '★ 本轮新开 termination=−0.5（V45 是 0）⇒ 这条反映"摔倒/终止频率"，越小越好'),
    ],
    'V45': [   # 同 V44（V1 奖励集 + OpenDoge 项）+ ★V45 新开 stand_still=-1.5（yaw 门控）治站立抬腿
        # 写法: (显示名, 奖励项, 缩放, 上限, 说明) —— 判据 = |速率| / 缩放，再把物理意义写在说明里
        ('姿态 倾斜rms(rad)     ', 'rew_orientation',      1.5,  0.060, '= sqrt(|值|/1.5) ⇒ 上限 0.060 即 rms 倾斜 ≤0.20 rad(11.5°)'),
        ('高度 偏差(m)          ', 'rew_base_height',      1.5,  0.050, '= sqrt(|值|/1.5)/10 ⇒ 上限 0.05 即高度偏差 ≤0.058 m'),
        ('跳弹 vz_rms(m/s)      ', 'rew_lin_vel_z',        2.4,  0.050, '= sqrt(|值|/2.4) ⇒ 上限 0.05 即竖直速度 rms ≤0.144 m/s'),
        ('平滑 二阶Δa           ', 'rew_smoothness',       0.08, 3.000, '= |值|/0.08 = Σ(二阶差分)² ⇒ 上限 3.0 即每关节 rms ≤0.50'),
        ('对角同步 error        ', 'rew_diagonal_sync',    0.2,  1.000, '= |值|/0.2（两组对角触地不一致之和）⇒ 上限 1.0 即 0.2'),
        ('髋镜像 err(rad)       ', 'rew_hip_mirror_symmetry', 0.05, 1.000, '= sqrt(|值|/0.05)/10 ⇒ 上限 1.0 即髋左右镜像误差 ≤0.10 rad(6°)'),
        ('姿态锚 Σ|q−q_def|(rad)', 'rew_default_pos_linear', 0.05, 36.0, '= |值|/0.05 ⇒ 上限 36 即 12 关节 Σ 偏差 ≤1.8 rad（每关节 ≤0.15）'),
    ],
    'V43': [   # 从零新轮（16000 轮）+【开环朝向暴露】open_loop_heading_frac=0.3
        ('侧移串扰 |vx|        ', 'rew_axis_purity',    1.0, 0.06, 'V41@2400 -0.091/-0.141(Isaac)；V40@600 0.043/0.178'),
        ('直行朝向偏离 |Δ|     ', 'rew_straight_hold',  0.3, 0.03, '★本轮该项才真正发力（30% 环境直行段无 yaw 指令）；V41@2400 -0.007 rad/s、V41b@5400 -0.036'),
        ('不自转 |ω_z|         ', 'rew_yaw_hold',       0.8, 0.05, '同上：开环环境靠它顶住漂移；V39/V40 基线 0.049'),
        ('步态左右时长差(速率) ', 'rew_gait_symmetry',  0.0, 0.02, '越接近 0 越好'),
        ('顶限位罚(速率)       ', 'rew_dof_pos_limits', 0.0, 0.01, 'V41 实测 0.0014~0.0021（几乎不发力，仅看趋势）'),
    ],
    'V42': [   # 从零新轮（16000 轮），奖励集与 V41 相同 ⇒ 判据阈值沿用
        ('侧移串扰 |vx|        ', 'rew_axis_purity',    1.0, 0.06, 'V41@2400 -0.091/-0.141(Isaac)；V40@600 0.043/0.178'),
        ('直行朝向偏离 |Δ|     ', 'rew_straight_hold',  0.3, 0.03, 'V41@2400 净偏置 -0.007 rad/s；V41b@5400 -0.036'),
        ('不自转 |ω_z|         ', 'rew_yaw_hold',       0.8, 0.05, 'V39/V40 基线 0.049；V41 站立 0.002'),
        ('步态左右时长差(速率) ', 'rew_gait_symmetry',  0.0, 0.02, '越接近 0 越好'),
        ('顶限位罚(速率)       ', 'rew_dof_pos_limits', 0.0, 0.01, 'V41 实测 0.0014~0.0021（该项在训练里几乎不发力，仅看趋势）'),
    ],
    'V41b': [   # V41 续训（同一套奖励，判断标准完全一致）
        ('侧移串扰 |vx|        ', 'rew_axis_purity',    1.0, 0.06, 'V40@600 基线 0.043/0.178；V41@2400 -0.091/-0.141(Isaac)'),
        ('直行朝向偏离 |Δ|     ', 'rew_straight_hold',  0.3, 0.03, 'V40@600 +0.005 rad/s；V41@2400 净偏置 -0.007'),
        ('不自转 |ω_z|         ', 'rew_yaw_hold',       0.8, 0.05, 'V39/V40 基线 0.049'),
        ('步态左右时长差(速率) ', 'rew_gait_symmetry',  0.0, 0.02, '越接近 0 越好'),
        ('顶限位罚(速率)       ', 'rew_dof_pos_limits', 0.0, 0.01, 'V41 新项 权重-0.5；V41@2400 是 0.0020'),
    ],
    'V41': [
        ('侧移串扰 |vx|        ', 'rew_axis_purity',    1.0, 0.06, 'V40@600 基线 0.043(vy=+1)/0.178(vy=−1)'),
        ('直行朝向偏离 |Δ|     ', 'rew_straight_hold',  0.3, 0.03, 'V40@600 基线 +0.005 rad/s（≈6°/40s）'),
        ('不自转 |ω_z|         ', 'rew_yaw_hold',       0.8, 0.05, 'V39/V40 基线 0.049 → V40@600 0.002'),
        ('步态左右时长差(速率) ', 'rew_gait_symmetry',  0.0, 0.02, '越接近 0 越好'),
        ('顶限位罚(速率)       ', 'rew_dof_pos_limits', 0.0, 0.01, 'V41 新项 权重−0.5：进软限位外沿才罚，看趋势趋 0'),
    ],
}


def load(run_dir):
    from tensorboard.backend.event_processing import event_accumulator as ea
    ev = glob.glob(os.path.join(run_dir, 'events.out.tfevents.*'))
    if not ev:
        sys.exit('❌ %s 下没有 events 文件' % run_dir)
    acc = ea.EventAccumulator(max(ev, key=os.path.getmtime), size_guidance={ea.SCALARS: 0})
    acc.Reload()
    return {t: {e.step: e.value for e in acc.Scalars(t)} for t in acc.Tags()['scalars']}


def latest(c, key, step=None):
    d = c.get(key)
    if not d:
        return None
    ks = [k for k in d if (step is None or k <= step)]
    return d[max(ks)] if ks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--round', default='V40')
    ap.add_argument('--step', type=int, default=None)
    args = ap.parse_args()
    run_dir = args.run if os.path.isdir(args.run) else os.path.join(
        ROOT, 'legged_gym', 'logs', 'flat_LgDog1', args.run)
    if not os.path.isdir(run_dir):
        sys.exit('❌ 找不到 run 目录: %s' % run_dir)
    c = load(run_dir)
    tag = os.path.basename(run_dir.rstrip('/'))
    print('  ── 本轮判据推导（训练侧代理量，%s%s）──' % (tag, '' if args.step is None else ' @%d' % args.step))
    for name, key, scale, limit, base in CRITERIA.get(args.round, []):
        v = latest(c, 'Episode/' + key, args.step)
        if v is None:
            print('    %s —（本项未记录）' % name)
            continue
        crit = abs(v) / scale if scale else abs(v)
        ok = '✔' if crit <= limit else '✘'
        extra = ''
        if scale == 0.0:      # 稀疏事件项: 只看趋势
            ok = '·'
            extra = '（只看趋势）'
        print('    %s %8.4f  （判据 ≤%.3f，%s）%s %s'
              % (name, crit, limit, base, ok, extra))
    for key, nm in [('Train/mean_episode_length', '回合长度'),
                    ('Episode/terrain_level', '地形等级'),
                    ('Episode/rew_tracking_lin_vel', '速度跟踪'),
                    ('Episode/rew_stand_still', '站立锚')]:
        v = latest(c, key, args.step)
        if v is not None:
            print('      参考: %-8s %.3f' % (nm, v))


if __name__ == '__main__':
    main()
