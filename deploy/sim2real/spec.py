#!/usr/bin/env python3
"""LgDog1 sim2real —— 与训练/sim2sim 对齐的常量（唯一来源）

★ 这个文件是整个部署栈里唯一允许出现"训练接口常量"的地方。
  motor / policy / obs / safety 各层都从这里 import，不许各自抄一份。

为什么这么做：`sim2sim.py` 自己的注释里写着 action_scale 经历过
0.25→0.5→0.25 的反复，V15 的三个 60000 步测试就因 0.25/0.5 错配而作废重跑。
再抄一份常量必然重蹈覆辙。用 `tools/check_alignment.py` 可以把本文件与
`sim2sim.py` 的模块级常量逐项断言相等（开发机上跑，不依赖硬件）。

口径来源（2026-09-24 版 sim2sim.py）：
  SIM_DT=0.005, DECIMATION=4 → 控制 50Hz；KP=30, KD=0.8；ACTION_SCALE=0.25
  观测单帧 45 维 × 历史 6 帧 = 270；REMAP 见下
  默认姿态每腿 (hip, thigh, calf) = (0, -0.4, +0.5)
"""

import numpy as np

# ==================== 维度与频率 ====================
NUM_JOINTS         = 12          # 12 个关节，无轮
NUM_LEGS           = 4
HISTORY_LEN        = 6           # 历史帧数

# ★★ 单帧观测有两套布局，由【身高指令通道】决定（2026-10-01 训练侧新增）★★
#   v1（旧，45 维）: cmd[0:3] omega[3:6] pg[6:9] q[9:21] dq[21:33] act[33:45]
#   v2（新，46 维）: cmd[0:3] 【身高指令[3]】 omega[4:7] pg[7:10] q[10:22] dq[22:34] act[34:46]
#   第 4 维 = (目标身高 − 名义高度) × HC；不给身高指令时就等于名义值 → 该维恒 0
#   （与 sim2sim.py 的 --height 缺省行为一致：obs[3] = 0）。
#   布局属于【策略契约】：老策略必须用 v1，新训练线（Oct02_06-50-35 起）用 v2。
HEIGHT_CMD_ENABLED = True        # 当前训练线的默认布局
# ★ 名义站高：身高指令通道的【编码基准】—— obs[3] = (目标身高 − 名义站高) × HC。
#   它是【每个 run 各自的】，和 kp/kd/action_scale 一样属于策略契约：
#     V41 / V39 / V41b 等 Oct-04 之后的 run：0.30（命令区间 [0.28, 0.34]）
#     V32 及更早带身高通道的 run：          0.28（命令区间 [0.26, 0.33]）
#   这里的值只是"策略没声明时"的默认；正常路径由 policy.json 的 control_contract.height_nominal
#   覆盖（--snapshot 也能给）。sim2sim.py 里同名常量必须与它一致，check_alignment.py 会核对。
#   ⚠️ 写错的后果：不给 --height 时 obs[3] 恒 0，一切正常（"碰巧对上"）；
#      一旦给 --height，编码基准差 2cm 就会把身高指令整体放大一倍以上。
HEIGHT_NOMINAL     = 0.30
HC                 = 10.0        # 身高指令缩放（sim2sim.py: HC，训练侧 normalization.height_cmd）


def obs_offsets(height_cmd=None) -> dict:
    """返回单帧观测各段的下标（slice / int）。布局是【可以切换】的，别在别处写死数字。"""
    h = HEIGHT_CMD_ENABLED if height_cmd is None else bool(height_cmd)
    if h:
        return {"cmd": slice(0, 3), "height": 3, "omega": slice(4, 7), "pg": slice(7, 10),
                "q": slice(10, 22), "dq": slice(22, 34), "act": slice(34, 46), "n": 46}
    return {"cmd": slice(0, 3), "height": None, "omega": slice(3, 6), "pg": slice(6, 9),
            "q": slice(9, 21), "dq": slice(21, 33), "act": slice(33, 45), "n": 45}


def _refresh_obs_dims():
    global NUM_ONE_STEP_OBS, NUM_OBS, OBS
    OBS = obs_offsets()
    NUM_ONE_STEP_OBS = OBS["n"]
    NUM_OBS = NUM_ONE_STEP_OBS * HISTORY_LEN


OBS                = obs_offsets()
NUM_ONE_STEP_OBS   = OBS["n"]                             # 45 或 46
NUM_OBS            = NUM_ONE_STEP_OBS * HISTORY_LEN       # 270 或 276
CONTROL_FREQ       = 50.0        # Hz = 1/(SIM_DT*DECIMATION) = 1/(0.005*4)
CONTROL_DT         = 1.0 / CONTROL_FREQ                  # 0.02 s
SIM_DT             = 0.005       # 仅用于与 sim2sim 对齐时的自检
DECIMATION         = 4

# ==================== 关节顺序 ====================
# ★ 策略的输入/输出永远是【Isaac 顺序】：[FL, FR, RL, RR] × [hip, thigh, calf]
#   （sim2sim.py 里 q_isaac = q_mj[REMAP]，REMAP 把 MJCF 的 FR,FL,RR,RL 换成这个序）
#   sim2real 侧把电机反馈直接组织成这个顺序，因此运行时不需要再做 REMAP。
LEG_ORDER          = ("FL", "FR", "RL", "RR")
JOINT_ORDER        = ("hip", "thigh", "calf")
JOINT_LABELS       = tuple("%s_%s" % (leg, j) for leg in LEG_ORDER for j in JOINT_ORDER)
# 与 sim2sim.py 的 REMAP 保持一致（MJCF 顺序 FR,FL,RR,RL → Isaac 顺序 FL,FR,RL,RR）
REMAP_MJCF_TO_ISAAC = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# ==================== 默认姿态 ====================
# 每腿 (hip, thigh, calf) = (0, -0.4, +0.5)，四腿相同；几何站高 0.3540 m
# ⚠️ sim2sim.py 注释：曾误用 go1 的站姿（calf -1.5），而 LgDog1 的 calf 限位只有 ±0.7，
#    会 PD 顶死在限位上、稳定阶段直接趴下。
# ★ 2026-10-02 改：(0,-0.4,+0.5) → (0,-0.20,+0.10)，与 sim2sim.py 同步。
#   依据 sim2sim.py 里的注释：原站姿被动站高 0.354 m，而行走高度约 0.30 m，
#   站立切行走时会弹一下；按 MuJoCo FK 网格搜索改成 (0,-0.20,+0.10) ⇒ 被动站高 ≈0.299 m。
#   当前训练线（logs/flat_LgDog1/Oct02_06-50-35_LgDog1_himloco 起）已用新站姿。
#   ⚠️ 老策略（models/、V6_h/、dog/ 里那三份）是按老站姿训的 —— 它们的
#      policy.json 的 control_contract.default_dof_pos 各自记了真值，main.py 会自动应用。
DEFAULT_DOF_POS    = np.array([0.0, -0.20, 0.10] * NUM_LEGS, dtype=np.float64)

# ==================== 控制参数 ====================
KP                 = 30.0
# ★ 2026-10-02：0.8 → 1.2，与当前训练线（V32 等）和 sim2sim.py 的常量一致。
#   同样地，老策略（models/、V6_h/）训练时是 0.8，由它们各自的
#   policy.json: control_contract.kd 声明并自动应用，不会被这个默认值带偏。
KD                 = 1.2
# ★ 2026-10-07：默认值改回 0.25，与当前训练线（V41 等）和 sim2sim.py 的常量一致
#   （tools/check_alignment.py 会逐项核对这两个文件，不一致会直接报错）。
#   来历：0.25 → 0.30（2026-10-02，V28/V32 那批线）→ 0.25（2026-10-07，V41 线）。
#   ★ 为什么改默认值是安全的：**契约跟着策略走** —— 每份策略旁边的 policy.json 里有
#     control_contract（action_scale / kd / decimation / kp / 观测布局），main.py 启动时自动应用。
#     所以按 0.30 训的 V28/V32 各自声明 0.30，不会被这个默认值带偏；反过来也一样。
#   这里的值只在"策略没有声明契约"时才起作用。
ACTION_SCALE       = 0.25
# 逐关节力矩上限 = URDF 的 effort：calf 带 1:2 减速比 → 20 Nm，hip/thigh 直驱 → 10 Nm
TAU_LIMIT          = np.array([10.0, 10.0, 20.0] * NUM_LEGS, dtype=np.float64)
# URDF 关节速度上限（Isaac/PhysX 当硬约束执行；真机由电机固件限制）
VEL_LIMIT          = np.array([18.85, 18.85, 9.425] * NUM_LEGS, dtype=np.float64)
# URDF 关节位置限位（顺序同上：每腿 hip,thigh,calf）
# 2026-09-17 起：hip 左右对称 ±1.2，thigh ±1.5（原 ±1.3）
# ★ 2026-10-02 修正：calf 由 ±0.7 改为 ±0.85，与模型文件对齐 ——
#   urdf/LgDog1.urdf 与 xml/LgDog1.xml 在 2026-09-30 都被改成 range ±0.85
#   （hip ±1.2、thigh ±1.5 未变；effort 仍是 calf 20 / 其余 10，velocity calf 9.425 / 其余 18.85）。
#   为什么要改：策略是按【模型的限位】(±0.85) 训的，而这些常量是上位机
#   RuntimeGuard 的越限判据 + SafetyMonitor 的关节限位夹取 —— 留在 0.7 会把策略
#   合法范围内的动作夹掉，还会误报"关节越限"。
JOINT_LOWER        = np.array([-1.2, -1.5, -0.85] * NUM_LEGS, dtype=np.float64)
JOINT_UPPER        = np.array([+1.2, +1.5, +0.85] * NUM_LEGS, dtype=np.float64)

# ==================== IMU 安装朝向（★ 硬件事实，2026-09 由用户确认）====================
# 机体系（与 legged_gym/Isaac 一致）: X 前, Y 左, Z 上  （FLU）
#   依据：MJCF 里 FR_hip 在 y=-0.07、FL_hip 在 y=+0.07 → 右腿在 -Y、左腿在 +Y
# IMU 实际安装: X 向后, Y 向右, Z 向上, 位于机身几何中央
#
#   IMU 三根轴在机体系里的方向:
#       IMU_X = 机体 -X（向后）   IMU_Y = 机体 -Y（向右）   IMU_Z = 机体 +Z（向上）
#   → 绕机体 Z 转 180°，v_body = M @ v_imu, M = diag(-1,-1,+1)
#
# ★★ 为什么这一段最重要：策略观测的
#       obs[3:6] = 机体系角速度 × 0.25      ← 必须是机体系
#       obs[6:9] = 投影重力（机体系）        ← 必须是机体系
#    而下位机的 IMU 直接读的是【IMU 坐标系】的量。坐标系不换算，观测就错，
#    机器人一上电就会往一边倒 —— 这是解析阶段标记的最高危项。
#    （"放在几何中央"意味着没有安装偏置，不需要补偿平移/离心项。）
IMU_MOUNT_PERM = (0, 1, 2)               # 机体的第 i 个轴取 IMU 的第 IMU_MOUNT_PERM[i] 个轴
IMU_MOUNT_SIGN = (-1.0, -1.0, 1.0)       # 以及各自的符号


def imu_mount_matrix():
    """v_body = M @ v_imu。"""
    import numpy as np
    M = np.zeros((3, 3))
    for i in range(3):
        M[i, IMU_MOUNT_PERM[i]] = IMU_MOUNT_SIGN[i]
    return M


def imu_vec_to_body(v):
    """把一个 IMU 系三维向量换算到机体系（角速度、重力、加速度都适用）。"""
    import numpy as np
    return imu_mount_matrix() @ np.asarray(v, dtype=np.float64)


# ==================== 观测构造 ====================
# 单帧 45 维布局（与 sim2sim.py:build_obs 逐行对应）：
#   [0:3]   cmd * (LV, LV, AV)
#   [3:6]   base_ang_vel(体坐标系) * AV
#   [6:9]   projected_gravity（机体坐标系，机体系下的重力方向）
#   [9:21]  (q - DEFAULT_DOF_POS) * DP
#   [21:33] dq * DV
#   [33:45] last_action（上一控制步的原始动作，未乘 ACTION_SCALE）
LV                 = 2.0         # lin_vel 缩放（x, y）
AV                 = 0.25        # ang_vel 缩放（观测 yaw 与 指令 yaw）
DP                 = 1.0         # dof_pos 缩放
DV                 = 0.05        # dof_vel 缩放
CLIP               = 100.0       # 单帧观测与动作的裁剪幅值

# ★ 指令死区：sim2sim.py:496 `if np.hypot(cmd[0], cmd[1]) < 0.2: cmd[0]=cmd[1]=0`
#   策略是在这个规则下评测的，sim2real 必须照搬，否则平地微速指令会进 OOD。
CMD_DEADZONE_XY    = 0.2

# ★ 历史帧语义（sim2sim.py:362-364, 498-508）：
#   - reset 时历史 = HISTORY_LEN 帧【全零】，且【第一次】策略调用就用这个零历史
#     （对应 Isaac play.py 的观测时序，不能用稳定站立帧预热）
#   - frame_order='newest' → 最新帧在最前（history[0] 是当前帧）
FRAME_ORDER        = "newest"


# ==================== 控制频率覆盖（★ 换"不同训练频率"的策略时必须走这里）====================
# 背景：policy 的观测历史是 6 帧，帧距 = SIM_DT × DECIMATION。所以【控制频率是策略契约的一部分】：
#   50Hz(decimation=4) → 历史跨度 100ms      100Hz(decimation=2) → 历史跨度 50ms
# 换一份按别的频率训出来的策略，如果只换权重、不改这个频率，策略拿到的是【错的时间尺度】——
# 现象是"能动但动作节奏全乱"，且不会报任何错。
# 所以这里给一个受控的覆盖入口：spec 仍然是常量的唯一来源，只是允许在启动时按 profile 重算一次。
def apply_decimation(decimation: int):
    """按 decimation 重算 CONTROL_FREQ / CONTROL_DT。返回 (旧, 新) 供打印。

    ⚠️ 必须在构造任何使用 CONTROL_DT 的对象【之前】调用（main.py 在启动早期调）。
    """
    global DECIMATION, CONTROL_FREQ, CONTROL_DT
    old = (DECIMATION, CONTROL_FREQ, CONTROL_DT)
    DECIMATION = int(decimation)
    if DECIMATION < 1:
        raise ValueError("decimation 必须 >= 1")
    CONTROL_FREQ = 1.0 / (SIM_DT * DECIMATION)
    CONTROL_DT = 1.0 / CONTROL_FREQ
    return old, (DECIMATION, CONTROL_FREQ, CONTROL_DT)


def apply_contract(obs_scales=None, clip=None, default_dof_pos=None, decimation=None,
                   height_cmd=None, height_nominal=None):
    """按【训练配置快照】把观测/控制契约对齐到策略真正被训练时的值。

    为什么需要：策略是"在这个契约下"训出来的 —— 控制频率决定历史帧距、观测缩放决定
    每一维的数值范围、默认站姿决定 obs[9:21] 的零点。用错了不会报错，只会让策略行为
    变得不可信（这个仓库里 action_scale 0.25/0.5 错配就作废过三个 60000 步的测试）。
    所以 main.py 的 --snapshot 会把这些一次对齐到位，并逐项打印对照。

    返回 [(项, 旧值, 新值)] 供打印；只覆盖传进来的项。
    """
    global LV, AV, DP, DV, CLIP, DEFAULT_DOF_POS, HEIGHT_CMD_ENABLED, HEIGHT_NOMINAL
    changes = []
    if obs_scales:
        for key, name in (("lin_vel", "LV"), ("ang_vel", "AV"),
                          ("dof_pos", "DP"), ("dof_vel", "DV")):
            if key in obs_scales:
                old_v = globals()[name]
                globals()[name] = float(obs_scales[key])
                if abs(old_v - globals()[name]) > 1e-9:
                    changes.append((name, old_v, globals()[name]))
    if clip is not None and abs(float(clip) - CLIP) > 1e-9:
        changes.append(("CLIP", CLIP, float(clip)))
        CLIP = float(clip)
    if default_dof_pos is not None:
        import numpy as _np
        new_pos = _np.asarray(default_dof_pos, dtype=_np.float64)
        if new_pos.shape == DEFAULT_DOF_POS.shape and not _np.allclose(new_pos, DEFAULT_DOF_POS):
            changes.append(("DEFAULT_DOF_POS", DEFAULT_DOF_POS.copy(), new_pos.copy()))
            DEFAULT_DOF_POS = new_pos
    if decimation is not None:
        _o, _n = apply_decimation(int(decimation))
        if _o[0] != _n[0]:
            changes.append(("DECIMATION", _o[0], _n[0]))
    if height_cmd is not None and bool(height_cmd) != HEIGHT_CMD_ENABLED:
        _old_n = NUM_ONE_STEP_OBS
        HEIGHT_CMD_ENABLED = bool(height_cmd)
        _refresh_obs_dims()
        changes.append(("单帧观测布局(%d维)" % NUM_ONE_STEP_OBS, _old_n, NUM_ONE_STEP_OBS))
    if height_nominal is not None:
        _hn = float(height_nominal)
        if not (0.10 <= _hn <= 0.60):
            raise ValueError("height_nominal=%r 不在合理范围 [0.10, 0.60] m" % (height_nominal,))
        if abs(_hn - HEIGHT_NOMINAL) > 1e-9:
            changes.append(("HEIGHT_NOMINAL", HEIGHT_NOMINAL, _hn))
            HEIGHT_NOMINAL = _hn
    return changes


def set_obs_layout(n_one_step: int):
    """按"单帧多少维"选布局（45=v1 无身高通道 / 46=v2 有）。返回是否改动了。"""
    global HEIGHT_CMD_ENABLED
    n = int(n_one_step)
    if n not in (45, 46):
        raise ValueError("单帧观测只支持 45(v1) 或 46(v2)，收到 %d —— "
                         "说明这份策略的观测布局与部署栈实现的不兼容" % n)
    want = (n == 46)
    if want == HEIGHT_CMD_ENABLED:
        return False
    HEIGHT_CMD_ENABLED = want
    _refresh_obs_dims()
    return True


def obs_history_span_ms() -> float:
    """观测历史覆盖的时间跨度 [ms] = (HISTORY_LEN-1) × 控制周期。"""
    return (HISTORY_LEN - 1) * CONTROL_DT * 1000.0


def obs_layout_report() -> str:
    """打印单帧观测的分段，便于人工核对（跟着当前布局走）。"""
    o = OBS
    rows = [
        (o["cmd"].start, o["cmd"].stop, "cmd[0]*%.2f, cmd[1]*%.2f, cmd[2]*%.2f" % (LV, LV, AV)),
    ]
    if o["height"] is not None:
        rows.append((o["height"], o["height"] + 1,
                     "(目标身高 − %.2f) × %.1f   ← 身高指令通道（不给就是名义值→恒 0）"
                     % (HEIGHT_NOMINAL, HC)))
    rows += [
        (o["omega"].start, o["omega"].stop, "base_ang_vel * %.2f" % AV),
        (o["pg"].start, o["pg"].stop, "projected_gravity(机体系)"),
        (o["q"].start, o["q"].stop,
         "(q - default) * %.2f  [Isaac序 FL,FR,RL,RR × hip,thigh,calf]" % DP),
        (o["dq"].start, o["dq"].stop, "dq * %.2f" % DV),
        (o["act"].start, o["act"].stop, "last_action (原始,未乘 ACTION_SCALE)"),
    ]
    out = ["单帧观测 %d 维：" % NUM_ONE_STEP_OBS]
    for a, b, desc in rows:
        out.append("  [%2d:%2d] %s" % (a, b, desc))
    out.append("历史 %d 帧 × %d = %d 维，%s" % (HISTORY_LEN, NUM_ONE_STEP_OBS, NUM_OBS, FRAME_ORDER))
    out.append("控制周期 %.1f ms，历史覆盖时间跨度 %.1f ms" % (CONTROL_DT * 1000.0,
                                                         obs_history_span_ms()))
    return "\n".join(out)


if __name__ == "__main__":
    print(obs_layout_report())
    print()
    print("控制频率 %.1f Hz (dt=%.4f s)" % (CONTROL_FREQ, CONTROL_DT))
    print("关节顺序:", ", ".join(JOINT_LABELS))
    print("默认姿态:", DEFAULT_DOF_POS)
    print("力矩上限:", TAU_LIMIT)
    print("位置限位: lower", JOINT_LOWER, " upper", JOINT_UPPER)
