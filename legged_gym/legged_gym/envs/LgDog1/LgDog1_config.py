# ★★ 2026-10-08 按用户指令【完整回退到 V1】★★
#   配置值逐项还原自 legged_gym/logs/_archive/V1/config_snapshot.json（V1 训练时自动存的完整配置）；
#   奖励集 = V1 的 21 项；奖励函数实现 = V1 直系祖先 HIMLoco 版（V1 源码已不在本机，见 tuning_log）。
#   回退前的版本备份在 logs/_backup_20261008_preV1/，改动清单见同目录 revert_report.md。
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# ============================================================================
# ⚠️【V 编号于 2026-09-20 被重置过 —— 读下面注释里的 V 编号前必看】
#   当时只保留 4 个 run 并重新编号:
#     新 V1 = 原 V1（PD 40/1.0 + 域随机化加严）
#     新 V2 = 原 V11（release 惩罚 + 降 tracking 权重；唯一崩解后自愈）
#     新 V3 = 原 V14（base_height 治蹲姿；最稳、tracking 最好 0.7069）
#     新 V4 = 原 V17（姿态锚把零指令站高瞄到 0.32）
#   其余 run 目录已删除。因此本文件注释里出现的 V5~V17 都是【旧编号】，
#   它们对应的 run 已不存在（策略归档移到 exported/policies/_obsolete/，
#   配置快照归档在 logs/_archive/<run>/）。
#   旧→新: 旧V11→V2、旧V14→V3、旧V17→V4；其余旧编号无现存对应。
#   下面这些注释保留原编号是为了**如实记录当时的实验依据**，不要按新编号去读它们。
# ============================================================================

#机器人参数：大腿长0.19m，小腿长0.2447m 正常站立高度

# ============================================================================
# LgDog1 机型事实（实测，改配置前先看）
#   腿长(矢状面)  大腿 0.1900 m + 小腿 0.2450 m，极限伸直 0.4347 m
#                 （大腿三维长 0.2054，差的 0.078 是 thigh 轴↔膝轴的侧向 y 偏移）
#   关节限位      hip -1.0~1.2(FL/RL) / -1.2~1.0(FR/RR)（镜像），thigh ±1.3，calf ±0.7
#                 放开依据：实测策略在 0.3/1.0/-0.8/0.5 m/s 下 thigh 用到 ±1.06、
#                 hip 到 -0.83，原 thigh ±1.0 两侧都被顶穿 → 放开到 ±1.3、hip 各放 0.2。
#                 ⚠️ calf ±0.7 是实机硬限位，保持不变（策略目前会顶死在这里）
#   站高          默认姿态 thigh=-0.4 / calf=+0.5 → 几何 0.3540 m（PD 悬停 0.3413）
#                 该姿态腿伸展率 89%（旧零位 72%），可达最高 0.378，最蹲 0.138
#   力矩          calf 带 1:2 减速比 → 20 Nm；hip/thigh 直驱 → 10 Nm
#   质量          ≈10.06 kg
#   关节顺序      URDF（= Isaac 加载顺序）= [FL,FR,RL,RR]×[hip,thigh,calf]
#                 MuJoCo XML（urdf2xml.py 按 body 层级生成）= [FR,FL,RR,RL]×[hip,thigh,calf]
#                 → 两者不同，sim2sim.py 的 REMAP=[3,4,5,0,1,2,9,10,11,6,7,8] 已逐个关节核对通过
# ============================================================================

class LgDog1Cfg( LeggedRobotCfg ): 

    # ==========================
    # 1. 环境与地形 (适配 HimLoco)
    # ==========================
    class env(LeggedRobotCfg.env):
        num_envs = 3072

        # 角速度(3) + 重力向量(3) + 指令(4) + 关节位置(12) + 关节速度(12) + 上一帧动作(12) = 43？→ 实测 45
        # 【历史】2026-10-02 曾加过第 5 个指令（目标机身高度）把单帧顶到 46；2026-10-08 起本轮线
        #   退回 **45**（去掉身高通道，见 V44/V45/V47/V48/V49 的 config_snapshot）。
        #   ⚠️ 这里必须与 num_one_step_observations 一致：两者不一致时 play.py 会以
        #   「size mismatch for critic.0.weight: 238 vs 239」的方式报错（2026-10-12 实测踩到）。
        num_one_step_observations = 45

        # [历史信息] 输入过去 6 帧的观测，用于让网络推断环境参数和自身状态
        num_observations = num_one_step_observations * 6 

        # 特权观测 (用于 Teacher Policy)
        # 包含：单步观测 + 线速度(3) + 外力(3) + 地形点(187)
        # ★ 2026-10-12 修：原来写 46 + 3 + 3 + 187 = 239，与本类的 45 单帧观测自相矛盾
        #   ⇒ 任何 45 维 checkpoint 用 `--task LgDog1` 加载都会崩。改成与单帧观测同步的 45+3+3+187=238。
        num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs * 1
        
        episode_length_s = 30

    class terrain(LeggedRobotCfg.terrain):   # 必须继承基类，否则基类新增字段会静默丢失
        mesh_type = 'trimesh' #'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)# 地形网格中的行数，对应训练中的难度等级，difficulty 随行数递增
        # num_cols=20 时 choice = j/20 + 0.001（0.001~0.951），按下面 proportions 的累积值分带选类型
        num_cols = 20 # number of terrain cols (types)# 地形网格中的列数，对应地形类型的多样性
        # ===================== 【7 类地形】(2026-09-25 新增第 7 类：纯平地) =====================
        # ★2026-09-29 提高平地占比（用户指令『把平地的占比调高』）: 0.20 → 0.34
        #   累积比例 = [0.08, 0.18, 0.24, 0.34, 0.50, 0.66, 1.00]，对应
        #   0.08 金字塔斜坡(正/负各半) | 0.10 斜坡+随机起伏 | 0.06 下楼 | 0.10 上楼
        #   0.16 随机方块障碍(含坑)   | 0.16 ★【障碍带】 | 0.34 ★【纯平地】(本次调高)
        #   20 列的实际落位(实测 terrain_type_of, choice=j/20+0.001): 斜坡2 | 斜坡起伏2 | 下楼1 | 上楼2
        #     | 障碍3 | 障碍带4 | 【纯平地 6 列 = 30%】(原为 4 列 = 20%，相对 +50%)
        # ⚠️ 顺序必须与 utils/terrain.py 里的 Terrain.TERRAIN_TYPES 前 7 项一致
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]   # 2026-09-29 平地 0.20→0.34
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    # ==========================
    # 2. 初始状态
    # ==========================
    class init_state(LeggedRobotCfg.init_state):
        # 默认姿态 thigh=-0.4 / calf=+0.5（四腿相同）：
        #   几何站高 0.3540 m，kp=30/kd=0.8 悬停实测 0.3321（倾角 4.75°、四足着地）
        #   腿伸展率 89%（旧零位 72%）→ 姿态更高、离地间隙更大，但向上蹬伸的余量变小
        #   （calf 还有 +0.2 rad 到限位，thigh 还有 0.9 rad）
        #   质心仍在支撑多边形内 ✔
        # ⚠️ 原 pos=0.15 会让脚出生就插进地里 13 cm，求解器猛弹 → "出生就倒"
        pos = [0.0, 0.0, 0.31]  # 用户值        # ★ 2026-10-08 用户指令: 出生高度改为 0.31 m
        #   依据: 用 sim2sim 的"落地稳定"段（纯 PD 保持默认姿态）实测，当前默认姿态
        #   (hip 0, thigh 0.0, calf 0.2) 的被动站高 = 0.303 m ⇒ 0.31 ≈ 脚刚好触地 + 7 mm 余量。
        #   ⚠️ 改默认姿态后必须重测这个值（站高 ≈0.30，别沿用 0.35/0.37 那些旧值）。
        # ⚠️ 曾误用 go1 的 calf=-1.5，超出 URDF 限位 ±0.7 一倍多，PD 会顶死在限位上
        default_joint_angles = { 
            # 【2026-10-02 用户指令『站立高度和行走高度一样，避免行走切站立突然蹦起来』】
            #   原默认 (thigh -0.4, calf 0.5) 被动站高 0.357m，而行走高度只有 ~0.30m ⇒ 行走→站立要弹起 5.5cm。
            #   新默认 (thigh -0.20, calf 0.10) 经 MuJoCo FK 网格搜索: 被动站高 ≈0.299m，足端仍在髋下方
            #   (+2.9cm)，远离 calf ±0.85 限位。改后请同步 sim2sim.py 的 _POSE（两边必须一致）。
            'FL_hip_joint': 0.0,  'FL_thigh_joint': 0.0,  'FL_calf_joint': 0.2,   # ★ 2026-10-08 用户指令
            'FR_hip_joint': 0.0,  'FR_thigh_joint': 0.0,  'FR_calf_joint': 0.2,   # ★ 2026-10-08 用户指令
            'RL_hip_joint': 0.0,  'RL_thigh_joint': 0.0,  'RL_calf_joint': 0.2,   # ★ 2026-10-08 用户指令
            'RR_hip_joint': 0.0,  'RR_thigh_joint': 0.0,  'RR_calf_joint': 0.2,   # ★ 2026-10-08 用户指令
        }                      
    # ==========================
    # 3. 控制参数
    # ==========================
    class control(LeggedRobotCfg.control):
        control_type = 'P'          # 位置控制 (PD Controller)
        stiffness = {'joint': 40.0}   # V1: kp 40   # 提升刚度让关节响应更快，配合大步幅
        damping   = {'joint': 1.2}   # V1: kd 1.0     # 增加阻尼，动作更柔顺自然     
        # 【2026-10-06 用户指令】0.3 → 0.25: 与 V6_h/V28 时代及实机现用值统一；
        #   副作用: 关节偏移整体缩 17%（动作权限更温和，实机上那种 FL_calf 目标冲到 +2.31rad
        #   被 safety 急刹的极端情形会收敛一些）。部署侧 sim2real/config.yaml 同步改 0.25。
        action_scale = 0.25  # 【2026-09-30 用 URDF+真实录制反推的标定】（用户提问『action_scale 最好开多少』）: URDF 可用行程 thigh 向上 1.9 rad、calf 向上 0.35 rad（限位放宽到 ±0.85 后）；4 个健康策略 14.9 万样本 |a| P95≈4.4；分布对齐法理想值 hip 0.284/thigh 0.254/calf 0.246（均值 0.261）。实测顶限位率: 0.25→整体1.6%(calf 4.9%) | 0.30→整体3.2%(calf 9.4%，安全上限) | 0.40→10.4%(calf 24.6%) | 0.50→19.6%(calf 38.9%，被削平)
                                     # 【2026-09-24 回退: 0.5 -> 0.25（V16 值；0.5 只在 V17 那轮用过）】
        decimation = 4              # V1: 200/2 = 100Hz
        hip_reduction = 1.0

    # ==========================
    # 4. 指令范围 (低速限制)   
    # ==========================
    class commands(LeggedRobotCfg.commands):
        """★ 2026-10-08 完整回退到 V1（值取自 logs/_archive/V1/config_snapshot.json）。
           V1 之后新增的 height_nominal / stand_command_prob / open_loop_heading_frac 已删除；
           对应的身高指令通道与站立样本注入在 V1 里并不存在。"""

        curriculum = True        # 指令课程：按地形成绩逐步放宽速度范围
        max_curriculum = 1.0     # 课程能放到的最大速度 [m/s]（V1 值）
        num_commands = 4         # vx, vy, yaw, heading（V1 没有第 5 维身高指令）
        resampling_time = 3.0    # 每 3 s 重采样一次指令（V1 值）
        heading_command = True   # 第 4 维是朝向指令，yaw 速度由朝向控制器算出

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x   = [-1.0, 1.0]      # 前进速度范围 [m/s]
            lin_vel_y   = [-1.0, 1.0]      # 侧向速度范围 [m/s]
            ang_vel_yaw = [-3.5, 3.5]      # 偏航角速度范围 [rad/s]（V1 值）
            heading     = [-3.14, 3.14]    # 朝向指令范围 [rad]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/LgDog1/urdf/LgDog1.urdf'
        name = 'LgDog1'
        # foot_name 是 body 名的子串匹配（legged_robot.py:829）。URDF 里已有 4 个
        # *foot link（固定关节 + dont_collapse="true"），否则 feet_indices 为空，
        # feet_air_time / feet_stumble / foot_clearance / feet_contact_forces 全静默变 0。
        foot_name = 'foot'
        # 碰撞惩罚覆盖的 body（子串匹配 body 名）：thigh×4 / calf×4 / base_link。
        # 三项在 URDF 里都有碰撞体，且都不含足端（足端接触是正常行为，不能罚）。
        # ⚠️ calf 依赖 urdf 里小腿的【胶囊】碰撞体；若把它删掉（比如为了绕开 PhysX
        #    的网格膨胀问题而整段移除 collision），这一项就变成永不触发的死配置。
        penalize_contacts_on = ['thigh', 'calf', 'base']
        terminate_after_contacts_on = ['base']
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        # ⚠️ 这两个是 PhysX 刚体速度钳制，不是电机限速；基类默认 1000=不钳制。
        #    9.5 rad/s 会切掉快速摆腿的速度。
        max_angular_velocity = 9.5
        max_linear_velocity = 20.
        # MuJoCo 模型里 armature=0.01（calf 因减速比 0.04），Isaac 只能设全局标量。
        armature = 0.

    # ==========================
    # 6. 域随机化 
    # ==========================
    class domain_rand(LeggedRobotCfg.domain_rand):
        # ────────────────────────────────────────────────────────────────
        # 本段 = 域随机化(DR)。按【重采样时机】分三类，各条注释已标注：
        #   [一次]      建环境时按环境采样一次，之后永不改变（等于换一批机器人）
        #   [每次reset] 每个回合重采样（同一环境不同回合参数会变）
        #   [定时]      按步数计时触发
        # 实现均在 legged_robot.py，行号见各条注释。
        # ⚠️ 本段【并非】与基类默认完全一致。2026-09-30 逐项核对时发现 5 处与"注释本意/参考实现"
        #    都不符的手滑漂移（com_displacement_range 8 倍、randomize_link_mass 反了、link_mass_range
        #    与 kp/kd/motor_strength_range 偏大、friction 上限偏高），已全部归位；改动理由写在各自条目下。
        #    因为是显式赋值，改基类 legged_robot_config.py 不会影响这里，反之亦然——两处要同步改。
        # ────────────────────────────────────────────────────────────────

        # ── 负载随机化 [一次] ────────────────────────────────────────────
        randomize_payload_mass = True
        payload_mass_range = [-0.75, 2.5]    # 机身附加质量 [kg]，叠加在 base_link 原质量 3.487 kg 上
        #   机制：legged_robot.py:423  props[0].mass = 原质量 + payload（只改 base 一个刚体）
        #   ⚠️ 范围含负值（-1 kg ≈ 减重）；+2 kg = base 增重 57%、整机(10.06 kg)的 ±20%。
        #      且只压在 base 上 → 显著改变质心高度与惯量，不是"均匀加重"。
        #   ⚠️ 只在建环境时采样（:711），reset 不重采样。

        # ── 质心偏移 [一次] ──────────────────────────────────────────────
        randomize_com_displacement = True
        com_displacement_range = [-0.2, 0.2]   # base_link 质心三轴偏移 [m]
        #   机制：legged_robot.py:426  props[0].com = Vec3(...)
        #   ⚠️ ±5cm 相对机身长 0.404m 是 ±12%，属很激进的范围 → 直接影响平衡与步态对称性。
        #   ⚠️ 同样只在建环境时采样（:713），reset 不重采样。
        #   【2026-09-30 修正】原值 [-0.4, 0.4]（±40cm，而机身才 0.404m 长），与本注释写的 ±5cm、
        #     与基类/OpenDoge 的 ±0.05 都不符 ⇒ 判定为 8 倍手滑 typo。±40cm 的质心偏移在大量环境里
        #     物理上站不住，策略只能用"最大支撑面+保守步态"换存活 → 实测 V25 走成蛙式外八

        # ── 连杆质量缩放 [一次]（当前关闭）──────────────────────────────
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]    # 除 base 外每个刚体独立乘一个随机系数
        #   机制：legged_robot.py:429，逐 body 用 np.random.uniform(rng[0], rng[1]) 缩放质量。
        #   【2026-09-30 修正】原为 True，与本行标题"（当前关闭）"矛盾，且基类/OpenDoge 均为
        #     False、范围 [0.9,1.1] ⇒ 按注释本意与参考实现归位。

        # ── 摩擦 [每次reset] ─────────────────────────────────────────────
        randomize_friction = True
        friction_range = [0.1, 1.5]    # 摩擦系数（标称 1.0）：0.2≈冰面、1.25≈橡胶
        #   机制：legged_robot.py:373 refresh_actor_rigid_shape_props()，每回合重采样
        #   ⚠️ 整机所有 rigid shape 共用同一个系数（不是逐足独立）→ 是"全身滑"而非"单脚打滑"。
        #   【2026-09-30 修正】上限 1.5 → 1.25（对齐基类/OpenDoge）。

        # ── 弹性 [每次reset]（当前关闭）─────────────────────────────────
        randomize_restitution = False
        restitution_range = [0., 1.0]   # 恢复系数（0=完全非弹性，1=完全弹性）

        # ── 电机力矩系数 [每次reset] ─────────────────────────────────────
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]   # ±10% 力矩缩放，模拟电机个体差异/电池电压波动
        #   【2026-09-30 修正】以下 kp/kd/motor_strength 三项原为 [0.8,1.2]：既超出注释本意(±10%)、
        #     也宽于基类/OpenDoge 的 [0.9,1.1]（同时压窄 DR 有助稳住外八那轮的站立姿态），统一收到 ±10%。
        #   机制：reset 时采样（legged_robot.py:206），_compute_torques 里乘到力矩上（:497）。

        # ── PD 增益 [每次reset] ──────────────────────────────────────────
        randomize_kp = True
        kp_range = [0.8, 1.2]           # ±10% 刚度误差（装配/关节摩擦导致的等效刚度差异）
        randomize_kd = True
        kd_range = [0.8, 1.2]           # ±10% 阻尼误差
        #   机制：reset 时采样 Kp/Kd 系数（legged_robot.py:202 / :204）：
        #   torques = p_gains·Kp_factors·(target−q) − d_gains·Kd_factors·dq   (:497)

        # ── 初始关节位置随机化（❌ 未实现，打开也没用）───────────────────
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]
        #   ❌ 死配置：全仓库（含 rsl_rl）搜 randomize_initial_joint_pos / initial_joint_pos_range
        #      命中 0 处 → 打开不会有任何效果。要启用需自己在 _reset_dofs 里给 dof_pos 加扰动；
        #      且语义未定义：[0.5, 1.5] 到底是"倍率"还是"rad 增量"？

        # ── 外力干扰 [定时] ──────────────────────────────────────────────
        disturbance = True
        disturbance_range = [-30.0, 30.0]   # 三维随机外力 [N]，逐环境独立采样
        disturbance_interval = 8            # 重采样间隔 [步] → 8×0.01s = 0.08s 换一次方向
        #   机制：legged_robot.py:453 → :550 _disturbance_robots()，
        #      apply_rigid_body_force_tensors(..., LOCAL_SPACE) 作用在 base（机体系）。
        #   ⚠️ ±30 N / 约 10 kg ≈ ±3 m/s²，比早先版本的 ±1.5 N(≈0.15 m/s²) 强 20 倍，
        #      属强力连续抖动而非偶发踢击。
        #   ⚠️ 代码里没有清除调用 → Isaac 的 force tensor 保持到下次覆盖，
        #      即"持续 0.08s 的恒力、方向每 0.08s 换一次"。
        #   ⚠️ 观测可见性：disturbance 只拼进【特权观测】，45 维学生观测在拼接前已被截断 →
        #      学生只能靠 6 帧历史推断外力，这正是 HIM estimator 的训练目标（看 estimation_loss）。

        # ── 推力扰动 [定时] ──────────────────────────────────────────────
        push_robots = True
        push_interval_s = 16            # 间隔 [s]；启动时换算成步数 ceil(16/dt)=1601（legged_robot.py:930）
        max_push_vel_xy = 2.            # 强度 [m/s]：把水平速度【直接设成】±1 的随机值，不是"加一个力"
        #   机制：legged_robot.py:451 → :543 _push_robots()，改写 root_states[:,7:9]。
        #   ⚠️ 换算后的 push_interval 会写进 config_snapshot（那里显示 1601.0 而不是 16），属正常。
        #   ⚠️ common_step_counter 是跨 episode 的全局计数 → 所有环境【同时】被推；
        #      episode 长 2002 步、间隔 1601 步 → 大致每个回合挨一次推。

        # ── 动作延迟 [每个控制步] ────────────────────────────────────────
        delay = True
        #   机制：legged_robot.py:92 每环境随机 delay_steps = randint(0, control.decimation)，
        #      在 decimation 个子步内让 last_actions → actions 线性过渡。
        #   ⚠️ 延迟粒度受 decimation 限制：decimation=2 → delay_steps ∈ {0,1}，
        #      只能表达 0 或 1 个 sim 子步 = 0 或 5ms（sim.dt=0.005），上限半个控制周期。

    # ==========================
    # 7. 奖励函数
    # ==========================
    class rewards( LeggedRobotCfg.rewards ):
        """★ 2026-10-08 按用户指令完整回退：参数与权重全部取自 logs/_archive/V1/config_snapshot.json。
           之后新增的 26 个参数（turn/step/clearance_*/axis/attitude/stuck… 相关）已删除——它们对应的
           奖励函数在 V1 里并不存在。"""

        only_positive_rewards   = False  # 不把总奖励裁剪为非负（保留完整惩罚信号）
        tracking_sigma          = 0.18   # 线/角速度跟踪核宽度 exp(−err²/σ²)
        base_height_target      = 0.30   # 机身高度目标 [m]
        clearance_height_target = -0.10   # 摆动足净空目标（机身系 z）
        max_contact_force       = 100.0  # 足端接触力超过它开始罚 [N]
        # ★ V45 补回: stand_still 的 yaw 门控阈值 [rad/s]（xy 与 yaw 同时低于它才算站立）
        stand_turn_gate         = 0.2
        soft_dof_pos_limit      = 0.9    # 关节位置软限位比例（目标夹取基准）
        soft_dof_vel_limit      = 1.0    # 关节速度软限位比例
        soft_torque_limit       = 1.0    # 关节力矩软限位比例

        class scales:
            # ===== V1 原始奖励集（21 项，权重逐一取自 V1 快照）=====
            action_rate            = -0.04      # 罚动作变化率（平滑）
            ang_vel_xy             = -0.02      # 罚横滚/俯仰角速度²（防晃）
            base_height            = -0.0      # 机身高度偏离 base_height_target
            collision              = -0.05      # 罚 base/thigh/calf 触地
            dof_acc                = -2.5e-07  # 罚关节加速度²（冲击）
            dof_pos_limits         = -0.02      # 罚超出软限位（V1 权重 0）
            dof_vel                = -0.0      # 罚关节速度²
            dof_vel_limits         = -0.0      # 罚超速（V1 权重 0）
            feet_air_time          = 0.1       # 奖励摆动时长（迈步）★ 这里是【权重】；
            #   起步阈值在 legged_robot.py:_reward_feet_air_time 里（2026-10-09 起 = 0.05 s，
            #   历史 0.20 → 0.10 → 0.05），两者不是一个量：权重决定这项值多少，阈值决定多长算"迈步"。
            feet_stumble           = -0.1      # 罚足端撞竖直面（绊）
            foot_clearance         = -0.01      # 摆动足离地不足（机身系口径，V1 版）
            joint_power            = -2e-05    # 罚关节功率
            lin_vel_z              = -0.8      # 罚竖直速度²（防弹跳）
            orientation            = -1.5      # 罚 roll/pitch²（水平姿态）
            smoothness             = -0.04     # 二阶平滑 |a−2a₋₁+a₋₂|²
            stand_still            = -1.0      # ★ V45: 0 → -1.5（治站立抬腿；配 stand_turn_gate 门控，不罚原地转弯）
            termination            = -0.5      # 非超时摔倒罚一次
            torque_limits          = -0.0      # 罚超力矩（V1 权重 0）
            torques                = -0.0      # 罚力矩²
            tracking_ang_vel       = 0.8       # 角速度跟踪 exp(−err²/σ²)
            tracking_lin_vel       = 1.6       # 线速度跟踪 exp(−err²/σ²)

            # --- OpenDoge 奖励项（★ 2026-10-08 加回；0.0 = 已接入但默认关闭）---
            feet_contact_forces    = -0.01       # 接触力超 max_contact_force 罚（建议 -0.01~-0.1）
            feet_parallel_to_ground = 0.1       # 着地时脚底与地面平行奖励（建议 +0.05~+0.2）
            hip_joint_pos          = 0.01      # 髋关节朝默认位 exp(−err²)（建议 +0.05~+0.2）
            default_joint_pos      = -0.00       # 12 关节偏离默认位 L2²（建议 -0.02~-0.1）
            default_pos_linear     = -0.00     # 12 关节偏离默认位 L1（OpenDoge 值 -0.05）
            base_height_linear     = -0.0       # 线性高度误差（OpenDoge 形式含 ×100 平方，权重须慎用）
            diagonal_sync          = -0.0      # trot 对角触地同步（OpenDoge 值 -0.2；我们上一轮用过 -0.1）
            hip_mirror_symmetry    = -0.0      # 髋左右镜像（OpenDoge 值 -0.1）

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.
    
    # ==========================
    # 9. 噪声
    # ==========================
    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01    # 关节编码器噪声
            dof_vel = 1.5     # 速度计算噪声 (通常比较大)
            lin_vel = 0.1     # IMU/状态估计噪声
            ang_vel = 0.2     # 陀螺仪噪声
            gravity = 0.05    # 重力感应噪声
            height_measurements = 0.1 # 高度图噪声
    # ==========================
    # 10. 物理引擎
    # ==========================
    class sim(LeggedRobotCfg.sim):
        dt = 0.005 # 物理仿真步长 5ms 关系到上位机频率
        substeps = 1 # 在每个 dt 内部，物理引擎还要细分几步来算
        gravity = [0., 0., -9.81] # 重力加速度
        up_axis = 1 # Z轴为上方向
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10 
            solver_type = 1 # 0: PGS, 1: TGS
            num_position_iterations = 4 # 次数越高，关节连接越紧密，不容易“脱臼”。
            num_velocity_iterations = 1 # 求解器计算“速度误差”的次数。主要解决“碰撞后的反弹速度”和“摩擦力”
            contact_offset = 0.01 # 碰撞检测距离
            rest_offset = 0.0 # 物体静止时的平衡距离
            bounce_threshold_velocity = 0.5 # 反弹阈值。如果撞击速度小于 0.5m/s，就不发生反弹（完全非弹性碰撞）
            max_depenetration_velocity = 1.0 # 最大去穿模速度。如果物体不幸穿模了（嵌进地里），引擎会把它推出来。这个参数限制推出来的最大速度。
            default_buffer_size_multiplier = 5
            max_gpu_contact_pairs = 2**23 # Opendoge mesh contacts are dense; bump capacity to avoid dropped aggregate pairs

# ==========================
# 训练参数 (PPO Runner)
# ==========================
# ⚠️ 本类只写"与基类不同"的项，其余全部继承 LeggedRobotCfgPPO
#    (legged_gym/envs/base/legged_robot_config.py)。继承值在下面用注释列出，
#    想改继承值必须在本类【显式写一行】覆盖，否则改了看不出效果、也进不了快照 diff。
class LgDog1CfgPPO( LeggedRobotCfgPPO ):
    # ---- 继承值（未在本类覆盖）----
    # seed = 1                       # ⚠️ 固定 seed：同配置从零重训≈复现同一条轨迹，
    #                                #    所以"重训一遍看方差"这个做法在本仓库是无效的
    # class policy:                  # 网络结构：actor 输入 45+3+16=64 维（含估计器隐变量）
    #     init_noise_std = 1.0       # 动作噪声标准差初值（探索幅度起点）
    #     actor_hidden_dims = [512, 256, 128]
    #     critic_hidden_dims = [512, 256, 128]   # critic 输入 238 维特权观测
    #     activation = 'elu'

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # ---- 本类覆盖值 ----
        entropy_coef = 0.01             # 熵奖励系数（探索压力）。后期退化时可考虑调低
        learning_rate = 1.e-3           # adaptive schedule 下的基准学习率（实际步长由 desired_kl 反馈调整）
        num_mini_batches = 16           # 基类 4 → 16：降低单次更新峰值显存，以保住 3072 并行环境
        schedule = 'adaptive'           # 由 KL 反馈自适应调 LR

        # ---- 继承值（默认生效，此处仅记录）----
        # desired_kl = 0.01             # ⚠️ commands.md 表里记的是 0.03，且文档写明
        #                               #    『mean_episode_length 突然暴跌 → LR 归零了，提高 desired_kl』。
        #                               #    第 8 轮 2900~3100 轮的 episode 1756→85 发散与该症状吻合，
        #                               #    但当时未改此值——下次遇到同类发散，这是第一个该试的旋钮。
        # num_learning_epochs = 5       # ⚠️ commands.md 记的是 3（"减半防过拟合"）
        # clip_param = 0.2              # PPO 策略比率裁剪范围
        # gamma = 0.99 / lam = 0.95     # 折扣因子 / GAE λ
        # max_grad_norm = 1.0           # 梯度裁剪范数
        # value_loss_coef = 1.0 / use_clipped_value_loss = True

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'LgDog1_himloco'
        experiment_name = 'flat_LgDog1' # ⚠️ 名字里是 flat，但 terrain.mesh_type 早已改成 trimesh
                                        #    （崎岖地形），只是没改实验名以免旧日志目录散掉

        policy_class_name = 'HIMActorCritic'   # HIM：actor 吃 45 步观测 + 估计器输出的 vel(3)+latent(16)
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100         # 每次迭代每环境采样 100 步
        max_iterations = 200000         # 迭代上限（实际每轮都远早于此就手动停了）
        save_interval = 200              # 每 20 次迭代存一次 model_<iter>.pt

        # ---- 继承值 ----
        # resume = False / load_run = -1 / checkpoint = -1   # 命令行 --resume / --load_run /
        #                                                    # --checkpoint 会覆盖这三项，
        #                                                    # 并如实写进 config_snapshot 的 train_cfg


# =====================================================================================
# 【低难度起步任务·2026-09-29】用户指令『开启 trimesh 然后从最低难度开始 + 恢复特权观测维度 + 日志目录』
#   与 LgDog1(trimesh) 的唯一差异 = 地形课程【从最低难度起步】:
#     min_terrain_level 3→0、max_init_terrain_level 5→0
#   其余全部沿用: trimesh 地形、高度扫描开启(特权观测 239)、越障晋升课程、OpenDoge 化奖励集。
# =====================================================================================
class LgDog1FlatCfg( LgDog1Cfg ):
    class env( LgDog1Cfg.env ):
        # 【2026-10-02 随高度指令通道同步】46(单步) + 3(线速度) + 3(扰动) + 187(高度扫描) = 239
        #   （46 = 3 指令 + 3 角速度 + 3 重力 + 1 身高指令 + 12 关节位置 + 12 关节速度 + 12 上帧动作）
        num_one_step_privileged_obs = 45 + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs * 1

    class terrain( LgDog1Cfg.terrain ):
        mesh_type = 'trimesh'              # ★开启 trimesh（回到真实崎岖地形）
        measure_heights = True             # ★高度扫描开启（特权观测 239）
        curriculum = True                  # 课程开启
        max_init_terrain_level = 5

class LgDog1FlatCfgPPO( LgDog1CfgPPO ):
    class runner( LgDog1CfgPPO.runner ):
        experiment_name = 'flat_LgDog1'    # ★日志目录恢复为 trimesh 版（与 LgDog1 共用，run 子目录区分）
