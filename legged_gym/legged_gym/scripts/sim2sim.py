#!/usr/bin/env python3
"""LgRobotDog Sim2Sim — MuJoCo 策略验证（LgDog1）

所有常量必须与【训练时的 config_snapshot.json】一致，否则策略在 OOD 观测下必然失败。
本次对齐依据：logs/flat_LgDog1/Sep14_07-58-28_LgDog1_himloco/config_snapshot.json
  * 关节顺序：Isaac 实测 = [FL,FR,RL,RR]（check_dof_order.py --task LgDog1）
              LgDog1.xml = [FR,FL,RR,RL]（每腿 hip,thigh,calf）
    → 不一致，用 REMAP 交换 FR↔FL、RL↔RR
  * 力矩上限：URDF 里 calf=20（1:2 减速比）、hip/thigh=10 → TAU_LIMIT 逐关节不同
  * 控制参数：decimation=2（100Hz）、action_scale=0.25、kp=30、kd=0.8
  * 默认姿态：每腿 (hip, thigh, calf) = (0, -0.4, +0.5)
  * 出生高度：0.37 m（与最新训练配置一致）
  * 传感器：Body_Quat / Body_Gyro（LgDog1.xml 已确认存在）

用法：
  python sim2sim.py                                # GLFW 键盘
  python sim2sim.py --no-keyboard --no-render --x_vel 1.0
  python sim2sim.py --frame-order oldest           # A/B 帧顺序

键盘：
  W/S 前后，Q/E 左右，A/D 转向，K 向躯干施加一次短时侧向踢击，Space/R 复位，Esc 退出。

手柄（飞智冰原狼一代等，GLFW 原生 API，无需 pygame/evdev）：
  python sim2sim.py --list-gamepads        # 先确认系统认到了（打印 jid/名称/轴数/键数）
  python sim2sim.py                        # 有手柄就自动接管（默认 --gamepad auto）
  左摇杆 = 前后/左右（模拟量），右摇杆左右 = 转向，A = 复位，X = 踢击，Start = 退出
  手柄与键盘可同时操作；无手柄时自动退回键盘/固定速度。

  轴/键序号不对时（不同模式序号会变）：
  python sim2sim.py --gamepad-debug        # 实时打印原始轴值，推摇杆看哪个轴在变
  python sim2sim.py --gp-axes 1,0,2 --gp-buttons 0,2,7     # 按实测序号指定
  ⚠️ 飞智手柄建议切到 XInput 模式（默认即 Xbox 布局，1,0,2 / 0,2,7 可直接用）；
     DInput/Switch 模式下 Linux 可能不暴露 /dev/input/js*，需要另配映射。
  无渲染模式（--no-render）下会建一个隐藏窗口专门用于手柄轮询，不会弹窗。
"""

import json, os, sys, argparse
import numpy as np
from collections import deque
import mujoco
try:
    import glfw                 # 手柄/键盘轮询用；模块级绑定，供 list_gamepads/Gamepad 使用
except Exception:               # 没装 glfw 也不该让脚本挂掉（只是没有窗口输入）
    glfw = None

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
LEGGED_GYM_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SCENE_XML      = os.path.join(LEGGED_GYM_DIR, "resources", "robots", "LgDog1", "xml", "scene.xml")
# --scene 传相对路径时的基准目录（scene.xml / scene_flat.xml 都在这）
SCENE_DIR      = os.path.join(LEGGED_GYM_DIR, "resources", "robots", "LgDog1", "xml")
POLICY_PATH    = os.path.join(LEGGED_GYM_DIR, "logs", "flat_LgDog1", "exported", "policies", "policy.pt")

# ==================== 关节顺序映射 ====================
# ✅ 已实测确认（check_dof_order.py --task LgDog1）：
#   Isaac Gym 训练 dof 顺序 = [FL,FR,RL,RR]（每腿 hip,thigh,calf）
#   MuJoCo LgDog1.xml 顺序 = [FR,FL,RR,RL]（每腿 hip,thigh,calf）
#   → 不一致！需要交换映射：isaac_q[i] = mujoco_q[REMAP[i]]
REMAP = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]   # 交换 FR↔FL、RL↔RR（自逆置换）

# 默认站立姿态（rad）—— 每腿 (hip, thigh, calf) = (0, -0.20, +0.10)，四腿相同
# 【2026-10-02 改】原 (0, -0.4, +0.5) 站高 0.354m；为与行走高度(~0.30m)一致、避免行走切站立弹跳，
#   按 MuJoCo FK 网格搜索改为 (0, -0.20, +0.10) ⇒ 被动站高 ≈0.299m。与 config 的 default_joint_angles 一致。
# ⚠️ 曾误用 go1 的站姿 (hip∓0.1, thigh 0.8/1.0, calf -1.5)：LgDog1 的 calf 限位只有 ±0.7，
#    -1.5 超限位两倍多，会让 PD 一直顶死在限位上、稳定阶段直接趴下。
_POSE = [0.0, 0.0, 0.2] * 4   # ★ 2026-10-08 与 LgDog1_config 默认关节角同步
DEFAULT_DOF_POS_MUJOCO = np.array(_POSE, dtype=np.float64)   # MuJoCo 顺序 FR,FL,RR,RL
DEFAULT_DOF_POS_ISAAC  = np.array(_POSE, dtype=np.float64)   #  Isaac 顺序 FL,FR,RL,RR

# ==================== 控制参数（与 config_snapshot 一致）====================
SIM_DT = 0.005
DECIMATION = 4             # 训练 control.decimation=4（★2026-09-23 对齐当前配置）→ 控制频率 200/4 = 50Hz
KP = 40.0                  # 训练 control.stiffness={'joint':40.0}（★2026-10-08 对齐当前 V1 系配置；原 30 是 V41 时代）
KD = 1.2                   # 最新训练配置 control.damping={'joint':1.2}（【2026-10-02 对齐】0.8 是 V27 时代的旧值，V28 起训练已用 1.2；不带 --snapshot 的裸跑也应对齐）
ACTION_SCALE = 0.25      # 训练 control.action_scale=0.25（★2026-10-07 对齐：V41 起训练与部署统一 0.25，
                           #   本常量之前错写成 0.3，裸跑（不带 --snapshot）时会比训练放大 20%）。
                           #   带 --snapshot 时以快照为准，会打印 ✅/❌ 对照；此值经历过
                           #   0.25→0.5→0.25→0.3→0.25 的反复——务必与训练时的 config_snapshot 一致，
                           #   V15 的三个 60000 步测试就曾因 0.25/0.5 错配而作废重跑）
# 逐关节力矩上限 = URDF 的 effort：calf 带 1:2 减速比 → 20 Nm，hip/thigh 直驱 → 10 Nm
# 顺序 = MuJoCo [FR,FL,RR,RL] × [hip,thigh,calf]
TAU_LIMIT = np.array([10., 10., 20.] * 4, dtype=np.float64)
# URDF 的关节速度上限 [rad/s]：hip/thigh=18.85, calf=9.425（顺序同 LgDog1.xml: FR,FL,RR,RL × hip,thigh,calf）。
# Isaac/PhysX 把它当硬约束执行（实测 Isaac 小腿速度峰值 9.44 ≈ 上限 9.425）；
# 而 MJCF 没有 maxvel 属性，MuJoCo 默认不限制 → 小腿实测冲到 24.1 rad/s（2.5 倍超限），
# 落地冲击被放大，机身角速度被踢到 4.05 rad/s（Isaac 只有 0.82）后翻倒。
# 用 --clamp-vel 打开这个逐物理步钳制，复现 PhysX 的行为。
VEL_LIMIT = np.array([18.85, 18.85, 9.425] * 4, dtype=np.float64)


def clamp_joint_vel(data, rdof):
    """把关节速度钳制到 URDF 上限（复现 PhysX 的关节速度约束）。

    只作用于关节自由度（qvel 末尾 rdof 个），不碰自由关节的线/角速度。
    """
    data.qvel[-rdof:] = np.clip(data.qvel[-rdof:], -VEL_LIMIT, VEL_LIMIT)
SETTLE_TIME = 0.5          # 落地稳定时长 [s]（A1 教训：0.01s 等于没稳定，自由落体历史回归）
SPAWN_Z = 0.31             # 出生高度 [m]（★2026-10-08 对齐当前配置 init_state.pos=[0,0,0.31]；
                           #   该姿态 (hip 0, thigh 0, calf 0.2) 的被动站高实测 ≈0.303 m）
#                            ⚠️ 改默认姿态后必须同步改这里，否则出生就插进地里
KICK_FORCE = 100.0         # 踢击力大小 [N]
KICK_DURATION = 0.1       # 踢击持续时间 [s]

# ==================== 观测缩放（与训练 obs_scales 一致）====================
LV = 2.0; AV = 0.25; DP = 1.0; DV = 0.05; CLIP = 100.0
HISTORY_LEN = 6; NUM_ONE_STEP_OBS = 45   # ★ 2026-10-08 默认对齐当前契约（V1 系 45/270）；
#   跑 276 维老策略时不带 --snapshot 也会【自动切换】布局（见下方按策略图的维度自检）
# 【2026-10-01 身高指令通道】obs 第 4 维 = (目标身高 − 名义高度) × HC；--height 下发，缺省=名义值
HC = 10.0
# ⚠️【2026-10-04 修正】必须等于训练配置 commands.height_nominal（当前 0.30）。
#   旧值 0.28 是 height_nominal 从 0.33 改到 0.30 之前的残留 ⇒ 编码基准差 2cm:
#   不传 --height 时 obs[3]=0，策略按它自己的名义 0.30 理解（碰巧对上）；
#   一旦传 --height 0.30，策略收到 (0.30−0.28)×10 = +0.2 ⇒ 以为要 0.32m。
HEIGHT_NOMINAL = 0.30

# 键盘命令必须落在训练范围内（lin_vel_x/y=[-1,1] m/s）。
# 旧值 2.0 会让按 W/Q 时策略直接进入训练分布外。
SPEED_X = 1.; SPEED_Y = 1.; SPEED_YAW = 2.
# ==================== 【朝向保持层】(2026-10-06 新增·部署侧方案) ====================
# 为什么需要（用户 MuJoCo 反馈『为什么会在原地旋转』）:
#   策略的观测里【没有自己的绝对朝向】（只有重力向量=roll/pitch 与陀螺=yaw 角速度），
#   训练时 yaw 指令由环境的朝向控制器给出: cmd_yaw = 0.5×wrap(朝向目标 − 当前朝向)（clip ±2）。
#   也就是说训练时【是环境在闭合这个环】。而 sim2sim/实机若直接下发 wz=0（开环），
#   就没有任何人纠正偏航 ⇒ 策略的微小偏航偏置会累积成"直行画圈/原地旋转"。
#   实测（Isaac，零 yaw 指令）: V38 站立 |ω_z|=2.998 rad/s(172°/s)、V37 直行 0.076；
#   MuJoCo 里同样（直行 |ω_z|=0.311 rad/s=17.8°/s ⇒ 30 秒转 535°）。
# 做法: 部署侧实现【与训练完全一致】的朝向控制器 ——
#   操作杆/按键的 yaw 输入不再直接下发，而是【积分成朝向目标】（按住 A/D = 以 2 rad/s 转动目标），
#   松开则目标冻结 ⇒ 控制器把朝向拉回并保持（= 直行不画圈、站立不旋转）。
#   下发给策略的 wz 仍是 0.5×误差（clip ±2）⇒ 与训练分布一致，不需要重训。
HEADING_P = 0.5        # 与训练 _post_physics_step_callback 同增益
WZ_CMD_MAX = 2.0       # 与训练相同的 clip（超出即 out-of-distribution，不要放大）



def quat_rotate_inverse(q, v):
    """q 为 xyzw；公式与 legged_gym/math.py 一致"""
    qw, qv = q[-1], q[:3]
    return v * (2. * qw**2 - 1) - np.cross(qv, v) * qw * 2 + qv * np.dot(qv, v) * 2


def get_obs(data):
    q = data.qpos.astype(np.float64)
    dq = data.qvel.astype(np.float64)
    quat = data.sensor('Body_Quat').data[[1, 2, 3, 0]].astype(np.float64)  # wxyz -> xyzw
    omega = data.sensor('Body_Gyro').data.astype(np.float64)               # 体坐标系
    return q, dq, quat, omega


def _yaw_of_quat_xyzw(quat):
    """四元数(xyzw) → 世界系 yaw。朝向保持层用它算朝向误差。"""
    x, y, z, w = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap_pi(a):
    """把角度归一化到 (−π, π]（与训练 wrap_to_pi 同语义）。"""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def build_obs(q_mj, dq_mj, quat, omega, cmd, last_actions_isaac, remap, default_isaac, cmd_height=None):
    pg = quat_rotate_inverse(quat, np.array([0., 0., -1.]))
    obs = np.zeros(NUM_ONE_STEP_OBS, dtype=np.float32)
    obs[0] = cmd[0] * LV; obs[1] = cmd[1] * LV; obs[2] = cmd[2] * AV
    q_isaac = q_mj[remap]
    dq_isaac = dq_mj[remap]
    # ★ 2026-10-08 支持两种观测布局（46 = 含身高指令通道；45 = V1 时代没有这一维）
    #   为什么必须支持: 之前把 46 写死 ⇒ 跑 V1 契约（270 维）的策略时直接"拒绝运行"，
    #   监控探针静默失败、体检读到的是旧录制（2026-10-08 实测踩到）。
    if NUM_ONE_STEP_OBS >= 46:
        h = HEIGHT_NOMINAL if cmd_height is None else cmd_height
        obs[3] = (h - HEIGHT_NOMINAL) * HC          # 身高指令（与训练同口径）
        obs[4:7] = omega * AV; obs[7:10] = pg
        obs[10:22] = (q_isaac - default_isaac) * DP
        obs[22:34] = dq_isaac * DV
        obs[34:46] = last_actions_isaac
    else:
        obs[3:6] = omega * AV; obs[6:9] = pg
        obs[9:21] = (q_isaac - default_isaac) * DP
        obs[21:33] = dq_isaac * DV
        obs[33:45] = last_actions_isaac
    return np.clip(obs, -CLIP, CLIP)


# ==================== 手柄支持（GLFW 原生，无需 pygame/evdev）====================
# 为什么走【原始摇杆 API】而不是 glfw.get_gamepad_state()：
#   get_gamepad_state() 只在设备命中 SDL 手柄映射时才有值（否则全 0），而国产手柄
#   （飞智冰原狼等）在 DInput/Switch 模式下常认不出映射。原始 API 对任何 HID 摇杆都有效，
#   代价是轴/按键序号跟设备走 → 用 --gp-axes / --gp-buttons 指定，或 --gamepad-debug 现场标定。
#
# 默认按键（XInput 布局，飞智在 XInput 模式下即此布局）：
#   左摇杆=前后/左右，右摇杆左右=转向，A=复位，X=踢击，Start=退出
GP_AXIS_DEFAULT = (1, 0, 2)        # (vx轴, vy轴, yaw轴)
GP_INVERT_DEFAULT = (True, True, True)   # Linux 摇杆 Y 轴通常"上为负"→ 需要取反
GP_BTN_DEFAULT = (0, 2, 7)         # (复位A, 踢击X, 退出Start)
GP_DEADZONE = 0.12


def gp_floats(vals):
    """GLFW 返回的元素可能是三种东西之一，必须逐一兼容：
       ① ctypes 指针 LP_c_float → 取 a.contents.value
       ② ctypes 数值对象        → 取 a.value
       ③ 普通 Python float      → 直接用
       （直接 float(a) / round(a) / np.asarray 都会报错，别用）"""
    out = []
    for a in vals:
        if hasattr(a, 'contents'):
            out.append(float(a.contents.value))
        elif hasattr(a, 'value'):
            out.append(float(a.value))
        else:
            out.append(float(a))
    return out


def gp_buttons(vals):
    """按键同理：可能是 LP_c_ubyte 指针、c_ubyte 或 int。返回 [0/1] 列表。"""
    out = []
    for b in vals:
        if hasattr(b, 'contents'):
            out.append(int(b.contents.value))
        elif hasattr(b, 'value'):
            out.append(int(b.value))
        else:
            out.append(int(b))
    return out


def list_gamepads():
    """枚举手柄，返回 [(jid, name, 轴数, 按键数)]。需要在 glfw.init() 之后调用。"""
    out = []
    if glfw is None:
        return out
    for jid in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1):
        if not glfw.joystick_present(jid):
            continue
        glfw.poll_events()                       # 首次 poll 后才有轴/键的初值
        try:
            nax, nbtn = len(glfw.get_joystick_axes(jid)), len(glfw.get_joystick_buttons(jid))
        except Exception:
            nax = nbtn = -1
        out.append((jid, glfw.get_joystick_name(jid), nax, nbtn))
    return out


class Gamepad:
    """GLFW 原始摇杆 → 模拟量速度指令 + 按键边沿。"""

    def __init__(self, jid, axes=GP_AXIS_DEFAULT, invert=GP_INVERT_DEFAULT,
                 buttons=GP_BTN_DEFAULT, deadzone=GP_DEADZONE, debug=False):
        self.jid = jid
        self.name = glfw.get_joystick_name(jid)
        self.spec = [(int(i), bool(inv)) for i, inv in zip(axes, invert)]
        self.btn_reset, self.btn_kick, self.btn_exit = [int(b) for b in buttons]
        self.dz = float(deadzone)
        self.debug = debug
        self._prev = (False, False, False)
        self._n = 0
        self._bad_axis_warned = False

    def _axis(self, idx, invert):
        vals = gp_floats(glfw.get_joystick_axes(self.jid))
        if idx < 0 or idx >= len(vals):
            return 0.0
        v = float(vals[idx])
        if invert:
            v = -v
        # ★ 2026-10-14 修 bug（实测: 一跑 sim2sim 就倒地）:
        #   GLFW 会把【触摸板/键盘/鼠标接收器】也列成 joystick，它们的轴值可能是 ±255、±32767
        #   这类【未归一化】原始值。原实现直接当摇杆用 ⇒ cmd 变成 vx=−7.82 m/s ⇒ 策略一上电就趴下。
        #   现在: |v|>1.5 视为"非标准手柄"，只警告一次并按 0 处理（不参与控制）；
        #   其余情况再夹到 [−1,1]，避免个别手柄偶发越界。
        if abs(v) > 1.5:
            if not self._bad_axis_warned:
                print("  ⚠️ 轴%d 原始值 %+.1f 超出 [−1,1] ⇒ 该设备不是标准手柄（%r），"
                      "已忽略它的模拟量输入" % (idx, v, self.name))
                self._bad_axis_warned = True
            return 0.0
        v = float(np.clip(v, -1.0, 1.0))
        if abs(v) <= self.dz:
            return 0.0
        # 死区后线性重映射到 [0,1]，避免死区边缘的跳变
        return float(np.sign(v) * (abs(v) - self.dz) / (1.0 - self.dz))

    def _btn(self, idx):
        btns = gp_buttons(glfw.get_joystick_buttons(self.jid))
        return idx >= 0 and idx < len(btns) and bool(btns[idx])

    def poll(self):
        """返回 (vx, vy, yaw, reset, kick, exit)，后三个是按下瞬间的边沿。"""
        vx = self._axis(*self.spec[0])
        vy = self._axis(*self.spec[1])
        yaw = self._axis(*self.spec[2])
        cur = (self._btn(self.btn_reset), self._btn(self.btn_kick), self._btn(self.btn_exit))
        reset = cur[0] and not self._prev[0]
        kick = cur[1] and not self._prev[1]
        exit_ = cur[2] and not self._prev[2]
        self._prev = cur
        if self.debug and self._n % 20 == 0:      # 每 20 个控制步打一次，别刷屏
            ax = gp_floats(glfw.get_joystick_axes(self.jid))
            bt = [i for i, b in enumerate(gp_buttons(glfw.get_joystick_buttons(self.jid))) if b]
            print(f"  [手柄] {self.name}: 轴={[round(a,2) for a in ax]}  按下键={bt}  "
                  f"→ vx={vx:+.2f} vy={vy:+.2f} yaw={yaw:+.2f}")
        self._n += 1
        return vx, vy, yaw, reset, kick, exit_


def init_gamepad(which='auto', axes=None, buttons=None, deadzone=None, debug=False):
    """初始化 GLFW 并选一个手柄。返回 (Gamepad|None, 隐藏窗口|None, 是否新建了窗口)。

    which: 'off' 不用 / 'auto' 取第一个 / 数字指定 jid
    GLFW 必须 poll_events 才更新手柄状态，而 poll_events 需要一个窗口/上下文；
    无渲染模式下这里创建一个【隐藏窗口】专门用于轮询。
    """
    if which == 'off':
        return None, None, False
    if glfw is None:
        print("  ⚠️ 未安装 glfw，手柄不可用")
        return None, None, False
    _glfw = glfw
    if not _glfw.init():
        print("  ⚠️ GLFW 初始化失败，手柄不可用")
        return None, None, False
    _glfw.poll_events()
    pads = list_gamepads()
    if not pads:
        print("  ⚠️ 未检测到手柄（检查：是否插上 / 是否在 XInput 模式 / ls /dev/input/js*）")
        return None, None, False
    print("  检测到手柄:")
    for jid, name, nax, nbtn in pads:
        print(f"    jid={jid}  {name!r}  轴数={nax} 按键数={nbtn}")
    if which == 'auto':
        # ★ 2026-10-14: 只有【轴数 >= 4】的设备才当手柄（标准手柄至少 4 轴）；
        #   触摸板/键盘/鼠标接收器都是 2 轴 —— 以前明知 nax<4 还照用（日志里那句警告形同虚设）。
        usable = [p for p in pads if p[2] is not None and p[2] >= 4]
        if not usable:
            print("  ⚠️ 列出的设备都不是标准手柄（轴数 < 4，多半是触摸板/键盘/鼠标接收器）"
                  "⇒ 按『无手柄』处理: 模拟量指令改由 --x_vel/--y_vel/--yaw_vel 或键盘给出")
            return None, None, False
        jid = usable[0][0]
    else:
        jid = int(which)
    if jid not in [p[0] for p in pads]:
        print(f"  ⚠️ 指定的 jid={jid} 不在列表里，改用 {pads[0][0]}")
        jid = pads[0][0]
    ax = GP_AXIS_DEFAULT if axes is None else tuple(axes)
    bt = GP_BTN_DEFAULT if buttons is None else tuple(buttons)
    dz = GP_DEADZONE if deadzone is None else float(deadzone)
    # 需要窗口才能 poll：由调用方决定是否新建（有 viewer 时复用它的窗口）
    gp = Gamepad(jid, axes=ax, buttons=bt, deadzone=dz, debug=debug)
    print(f"  ✔ 使用 {gp.name}：vx=轴{ax[0]} vy=轴{ax[1]} yaw=轴{ax[2]} | "
          f"A(键{bt[0]})=复位 X(键{bt[1]})=踢击 Start(键{bt[2]})=退出 | 死区={dz}")
    if nax is not None and nax < 4:
        print("  ⚠️ 该设备只有 %d 个轴 —— 大概率不在手柄模式！" % nax)
        print("     飞智冰原狼：长按 Mode/Fn 键（或用飞智游戏厅 App）切到 XInput/手柄模式，")
        print("     正常应有 6~8 个轴、11+ 个按键。当前 2 轴 2 键只是键盘/鼠标映射。")
    if debug:
        print("  [debug] 下面是实时原始值，用来标定轴/键序号：推动摇杆看哪个轴变化即可")
    return gp, jid, False



def main(use_keyboard=True, speed_x=1.0, speed_y=0.0, speed_yaw=0.0,
         no_render=False, frame_order='newest', settle_time=SETTLE_TIME,
         kp=KP, kd=KD, remap_mode='swap', spawn_z=SPAWN_Z,
         kick_force=KICK_FORCE, kick_duration=KICK_DURATION,
         kick_direction='left', record=True, record_every=1, record_csv=False,
         clamp_vel=False, scene_xml=None, policy_path=None, cmd_height=None,
         match_isaac_dofs=False, friction=None, render_every=0,
         uniform_joint_passives=False, cmd_schedule=None,
         spawn_x=0.0, spawn_y=0.0, snapshot=None,
         heading_hold=False, heading_p=HEADING_P,
         action_scale=None, decimation=None,
         gamepad='auto', gp_axes=None, gp_buttons=None, gp_deadzone=None,
         gp_debug=False):
    # ★【保真度对齐】(2026-09-23 新增) —— 治"硬编码常量与训练配置不一致"这个反复踩的坑:
    #   症状: V15 训练用 action_scale=0.5，而 sim2sim 常量长期是 0.25 ⇒ 策略动作被砍半，
    #         三个 60000 步测试全部作废。
    #   做法: --snapshot <run>/config_snapshot.json 时【直接从训练快照读控制参数】并逐项对照打印，
    #         默认采用快照值；--action-scale/--decimation 可显式覆盖（用于扫保真度旋钮）。
    global ACTION_SCALE, DECIMATION, NUM_ONE_STEP_OBS, SPAWN_Z, KP, KD
    if not snapshot:
        print("[提示] 未给 --snapshot ⇒ 用脚本模块常量（kp/kd/decimation/spawn_z 已按当前训练配置对齐）。"
              "若这份策略来自别的 run，请带 --snapshot <run>/config_snapshot.json")
    if snapshot:
        with open(snapshot, "r", encoding="utf-8") as _f:
            _snap = json.load(_f)
        _ec = _snap["env_cfg"]
        _c, _s = _ec["control"], _ec["sim"]
        _vals = dict(sim_dt=float(_s["dt"]), decimation=int(_c["decimation"]),
                     kp=float(list(_c["stiffness"].values())[0]),
                     kd=float(list(_c["damping"].values())[0]),
                     action_scale=float(_c["action_scale"]),
                     spawn_z=float(_ec["init_state"]["pos"][2]))
        # 【2026-09-25 补齐】观测侧常量也纳入对齐核对（此前只查控制参数，观测缩放是硬编码的）
        try:
            _os_ = _ec["normalization"]["obs_scales"]
            _vals.update(obs_lin=float(_os_["lin_vel"]), obs_ang=float(_os_["ang_vel"]),
                         obs_dof_pos=float(_os_["dof_pos"]), obs_dof_vel=float(_os_["dof_vel"]),
                         clip_obs=float(_ec["normalization"]["clip_observations"]),
                         n_one_step=int(_ec["env"]["num_one_step_observations"]),
                         n_obs=int(_ec["env"]["num_observations"]))
        except (KeyError, TypeError) as _e:
            print(f"    ⚠️ 快照缺少观测侧字段，跳过观测对齐核对（{_e}）")
        # ★ 2026-10-08: 采纳快照的观测布局（45/46）—— 只对照不采纳会导致
        #   跑 V1 契约（270 维）策略时"拒绝运行"，探针静默失败。
        if int(_vals["n_one_step"]) != NUM_ONE_STEP_OBS:
            print(f"[对齐] 观测布局按快照切换: 单步 {NUM_ONE_STEP_OBS} → {int(_vals['n_one_step'])} 维"
                  f"（历史 {HISTORY_LEN} 帧 ⇒ {int(_vals['n_one_step'])*HISTORY_LEN} 维）")
            NUM_ONE_STEP_OBS = int(_vals["n_one_step"])
        print(f"[对齐] 读训练快照 {snapshot}")
        print("[对齐] 逐项对照（sim2sim 当前值 vs 训练值）:")
        _all_ok = True
        _checks = [("sim.dt", SIM_DT, _vals["sim_dt"]),
                   ("decimation", DECIMATION, _vals["decimation"]),
                   ("kp", kp, _vals["kp"]), ("kd", kd, _vals["kd"]),
                   ("action_scale", ACTION_SCALE, _vals["action_scale"]),
                   ("spawn_z", spawn_z, _vals["spawn_z"])]
        if "obs_lin" in _vals:
            _checks += [("obs_scales.lin_vel", LV, _vals["obs_lin"]),
                        ("obs_scales.ang_vel", AV, _vals["obs_ang"]),
                        ("obs_scales.dof_pos", DP, _vals["obs_dof_pos"]),
                        ("obs_scales.dof_vel", DV, _vals["obs_dof_vel"]),
                        ("clip_observations", CLIP, _vals["clip_obs"]),
                        ("obs 历史维度", HISTORY_LEN * NUM_ONE_STEP_OBS, _vals["n_obs"]),
                        ("单步 obs 维度", NUM_ONE_STEP_OBS, _vals["n_one_step"])]
        for _name, _cur, _val in _checks:
            _ok = abs(float(_cur) - _val) < 1e-9
            _all_ok &= _ok
            print(f"    {'✅' if _ok else '❌'} {_name}: sim2sim={_cur} 训练={_val}")
        print(f"    ⇒ {'全部一致' if _all_ok else '存在不一致 → 采用训练快照的值（命令行显式覆盖优先）'}")
        kp, kd, spawn_z = _vals["kp"], _vals["kd"], _vals["spawn_z"]
        if action_scale is None:
            action_scale = _vals["action_scale"]
        if decimation is None:
            decimation = _vals["decimation"]
        if abs(SIM_DT - _vals["sim_dt"]) > 1e-12:
            print(f"    ⚠️ sim.dt 不一致（{SIM_DT} vs {_vals['sim_dt']}）: 物理步长是模块常量，如需对齐请改脚本常量。")
    if action_scale is not None:
        ACTION_SCALE = float(action_scale)
    if decimation is not None:
        DECIMATION = int(decimation)
    print(f"[控制] 最终生效: decimation={DECIMATION}（控制频率 {1.0/(SIM_DT*DECIMATION):.0f}Hz）"
          f" action_scale={ACTION_SCALE} kp={kp} kd={kd} spawn_z={spawn_z}")

    # 关节映射（已实测确认）：Isaac=FL,FR,RL,RR vs LgDog1.xml=FR,FL,RR,RL → 需要交换
    #   swap     = 正确（默认，实测确认）
    #   identity = 错误假设（仅 A/B 用）
    if remap_mode == 'swap':
        remap = REMAP
        default_isaac = DEFAULT_DOF_POS_ISAAC
        print(f"[2.5/4] 交换映射（实测确认：Isaac=FL,FR,RL,RR vs LgDog1.xml=FR,FL,RR,RL）")
    else:
        remap = list(range(12))
        default_isaac = DEFAULT_DOF_POS_MUJOCO.copy()
        print(f"[2.5/4] 恒等映射（A/B 用）")

    # ★策略路径显式化：POLICY_PATH 是写死的 policy.pt，一旦有别的导出覆盖它，
    #   整轮 sim2sim 结果就会被贴错标签（"这是 V13 的结果"其实不是）。
    #   所以默认仍是 policy.pt（保持原行为），但允许 --policy 指定。
    #
    # ⚠️【不要用文件 md5 当策略身份！】实测踩到：同一份权重重新导出一次，
    #    文件的 md5 就变了（34622e7d vs 4d7170db），因为 torch.jit 的 zip 里
    #    `policy/.data/serialization_id` 和自动生成的 `___torch_mangle_N` 源码条目
    #    每次序列化都不同 —— 而 14 个真正存权重的 `policy/data/*` 条目 CRC 完全一致。
    #    所以我用【权重指纹】(按参数名排序后对张量字节做 md5) 作为真正的身份标识：
    #    policy.pt 与 policy_V13_6000.pt 的文件 md5 不同，但权重指纹都是 e4fee6f97eaf。
    policy_file = policy_path or POLICY_PATH
    import hashlib
    print(f"[1/4] 加载策略: {policy_file}")
    import torch
    policy = torch.jit.load(policy_file, map_location='cpu' ); policy.eval()
    # 【硬维度校验】policy 的 norm_mean 维度必须等于 sim2sim 构建的单步观测×历史帧数，
    #   否则会在起步稳定后第一帧推理时才崩（2026-10-04 用户实测）。不符就直接退出并给出原因。
    _policy_dim = int(policy.norm_mean.shape[0])
    _sim_dim = NUM_ONE_STEP_OBS * HISTORY_LEN
    # ★ 2026-10-08【自动布局识别】裸跑（不给 --snapshot）时模块常量是写死的，
    #   而策略可能是另一种布局（45/270 = V1 系、46/276 = V41 系）⇒ 以前直接"拒绝运行"，
    #   用户看到的就是"❌ 策略输入维度 270 ≠ sim2sim 观测维度 276"。
    #   策略图自己的输入维度才是权威（与部署侧 probe_contract 同思路）⇒ 差 6 维就自动切换。
    if _policy_dim != _sim_dim and _policy_dim in (270, 276) and _sim_dim in (270, 276):
        _want = _policy_dim // HISTORY_LEN
        print(f"[布局] 策略是 {_policy_dim} 维（{_want}/帧）而当前 sim2sim 是 {_sim_dim} 维"
              f"（{NUM_ONE_STEP_OBS}/帧）⇒ 自动切换观测布局为 {_want}/帧")
        NUM_ONE_STEP_OBS = _want
        _sim_dim = NUM_ONE_STEP_OBS * HISTORY_LEN
    if _policy_dim != _sim_dim:
        print(f"❌ 策略输入维度 {_policy_dim} ≠ sim2sim 观测维度 {_sim_dim}，拒绝运行。")
        print(f"   原因: 这个 policy.pt 是【旧观测格式(45/270)】或其它导出的策略 ——")
        print(f"   常见于 play.py 自动覆盖了 policy.pt（helpers.py 的导出固定 obs[:, 0:45]）。")
        print(f"   解决: 用 tools/use_policy.sh 切回 276 维的策略，")
        print(f"         或 bash tools/export_checkpoint.py <run_dir> <step> 重新导出。")
        sys.exit(1)
    try:
        _sd = policy.state_dict()
        _h = hashlib.md5()
        for _k in sorted(_sd):
            _h.update(_k.encode()); _h.update(_sd[_k].detach().cpu().numpy().tobytes())
        _policy_wfp = _h.hexdigest()[:12]
        print(f"      权重指纹 = {_policy_wfp}   ← 判断『是不是同一个策略』只看这个")
    except Exception as _e:
        _policy_wfp = 'unavailable'
        print(f"      权重指纹不可用（{str(_e)[:40]}）—— 请勿用文件 md5 代替")
    try:
        policy(torch.zeros(1, NUM_ONE_STEP_OBS * HISTORY_LEN))
        print(f"      策略输入 {NUM_ONE_STEP_OBS * HISTORY_LEN} 维 ✓")
    except Exception as e:
        print("      ⚠️ 策略输入维度不符（应为 %d），请检查导出方式：" % (NUM_ONE_STEP_OBS * HISTORY_LEN), e)
    # 【2026-09-25 新增】观测经验归一化状态核对：
    #   归一化(若训练时启用)已被烘焙进导出图 ⇒ 本脚本只需喂【原始 obs】；
    #   这里把烘焙的统计量打印出来，供判断"导出是否还原了训练分布"。
    try:
        _bufs = dict(policy.named_buffers())
        if "norm_mean" in _bufs and "norm_var" in _bufs:
            _nm, _nv = _bufs["norm_mean"], _bufs["norm_var"]
            _dev_m = float(_nm.abs().max()); _dev_v = float((_nv - 1).abs().max())
            _ident = (_dev_m < 1e-6 and _dev_v < 1e-6)
            print(f"      观测归一化: 已烘焙 mean|max|={_dev_m:.3e} var|偏差|={_dev_v:.3e}"
                  f" ⇒ {'⚠️ 恒等统计量（训练统计量丢失，输入分布可能 OOD）' if _ident else '训练统计量已还原 ✅'}")
        else:
            print("      观测归一化: 导出图中无归一化缓冲（该 run 未启用观测归一化）")
    except Exception as _e:
        print(f"      观测归一化状态未知（{str(_e)[:40]}）")

    scene_path = scene_xml or SCENE_XML
    if not os.path.isabs(scene_path):
        scene_path = os.path.join(SCENE_DIR, scene_path)
    print(f"[2/4] 加载场景: {scene_path}")
    model = mujoco.MjModel.from_xml_path(scene_path)
    model.opt.timestep = SIM_DT
    # ==================== 保真度旋钮（让 MuJoCo 对齐 Isaac 训练侧）====================
    # 审计结论（实测读自 MjModel vs config_snapshot）——MuJoCo 与 Isaac 的差异：
    #   ① 关节摩擦/阻尼/惯量: MuJoCo hip/thigh armature=0.01 damping=0.01 frictionloss=0.2、
    #      calf armature=0.04 damping=0.04 frictionloss=0.4；而 Isaac 侧 cfg.asset.armature=0、
    #      URDF dynamics damping=0 friction=0 → 【Isaac 完全没有这三项】。
    #      calf 的 armature=0.04 相对小腿自身转动惯量(≈0.003 kg·m²)是十几倍，量级很大。
    #   ② 摩擦: MuJoCo 全部 geom friction=0.6；训练标称 static/dynamic_friction=1.0（低了 40%）。
    #   ③ 关节限速: MuJoCo 无（--clamp-vel 才能对齐 URDF 的 18.85/18.85/9.425）。
    #   这些旋钮就是用来逐个归因"到底哪一项让策略在 MuJoCo 里走不动"的。
    # ★"模仿 go1" 选项：把全部关节的被动参数统一成 MJCF 的 <default> 关节类值
    #   （damping=0.01 / armature=0.01 / frictionloss=0.2），也就是 go1.xml 的实际形态。
    #   用途: 回答"LgDog1 唯一的模型级差异（calf 因 1:2 减速比被折算成 0.04/0.04/0.4）
    #   是不是 sim2sim 走不动的原因" —— 这是个可测的问题，不必争论。
    if uniform_joint_passives:
        import numpy as _np
        model.dof_armature[:] = 0.01
        model.dof_damping[:] = 0.01
        model.dof_frictionloss[:] = 0.2
        # 只改 12 个铰链关节（自由关节的 6 个 DOF 保持 0，与原来一致）
        for _j in range(model.njnt):
            if model.jnt_type[_j] != mujoco.mjtJoint.mjJNT_HINGE:
                _a = model.jnt_dofadr[_j]
                model.dof_armature[_a] = 0.0
        print("      [保真] 已把全部铰链关节统一为 go1 式被动参数 "
              "(armature=0.01 damping=0.01 frictionloss=0.2)")
    if match_isaac_dofs:
        model.dof_armature[:] = 0.0
        model.dof_damping[:] = 0.0
        model.dof_frictionloss[:] = 0.0
        print("      [保真] 已把 armature/damping/frictionloss 全部置 0（对齐 Isaac 训练侧）")
    if friction is not None:
        # MuJoCo 的接触摩擦取两 geom 的【逐元素最大值】，所以把所有 geom 的滑动摩擦
        # 统一设为该值即可让接触摩擦等于该值（训练标称 1.0）。
        model.geom_friction[:, 0] = float(friction)
        print(f"      [保真] 已把全部 geom 滑动摩擦置为 {friction}（训练标称 1.0；模型原值 0.6）")
    data = mujoco.MjData(model)
    rdof = model.nu
    trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'trunk')
    if trunk_body_id < 0:
        raise RuntimeError("场景中找不到名为 'trunk' 的 body，无法施加踢击力")
    kick_directions = {
        'forward': np.array([1.0, 0.0, 0.0]),
        'backward': np.array([-1.0, 0.0, 0.0]),
        'left': np.array([0.0, 1.0, 0.0]),
        'right': np.array([0.0, -1.0, 0.0]),
    }
    if kick_direction not in kick_directions:
        raise ValueError(f"未知踢击方向: {kick_direction}")
    kick_force = abs(float(kick_force))
    kick_duration = max(0.0, float(kick_duration))
    kick_vector = kick_directions[kick_direction] * kick_force
    kick_steps = max(1, int(round(kick_duration / SIM_DT))) if kick_duration > 0 else 0
    names = [model.joint(i).name for i in range(1, model.njnt)]
    print(f"      MuJoCo 关节顺序: {names}")
    print(f"      预期(LgDog1.xml)顺序: [FR_hip,FR_thigh,FR_calf, FL_hip,FL_thigh,FL_calf, "
          f"RR_hip,RR_thigh,RR_calf, RL_hip,RL_thigh,RL_calf]")
    print(f"      REMAP = {REMAP}  帧顺序 = {frame_order}  出生 z = {spawn_z}")
    print(f"      踢击设置: K 键触发，方向={kick_direction}，力={kick_force:.1f}N，持续={kick_duration:.3f}s")

    # ★出生点可指定(x,y): 场景里有并排的多条车道（如三组不同高度的楼梯）时，
    #   sim2sim 的指令是固定向前的，机器人只会沿 y≈0 走 —— 没有这个参数就永远测不到别的车道。
    data.qpos[:7] = np.array([spawn_x, spawn_y, spawn_z, 1., 0., 0., 0.], dtype=np.float64)
    data.qpos[7:] = DEFAULT_DOF_POS_MUJOCO
    mujoco.mj_step(model, data)

    viewer = None
    if not no_render:
        try:
            import mujoco_viewer
            viewer = mujoco_viewer.MujocoViewer(model, data)
        except Exception as e:
            print(f"      ⚠️ viewer 创建失败，降级无渲染: {e}")
            no_render = True
    if heading_hold:
        print(f"[朝向保持层] 开（双模式）: 无转向输入时下发 wz = {heading_p:.2f}×朝向误差 "
              f"(clip ±{WZ_CMD_MAX:.1f}) 保持朝向（与训练同律，直行不画圈）；"
              f"按住转向键/调度表给速率时【直接透传】，手感与原始行为一致。")
    else:
        print("[朝向保持层] 关（默认=原始行为）: 操作杆/调度表的 wz 直接下发（开环）"
              "——策略偏航偏置会累积成画圈/自转；加 --heading-hold 可开。")
    # ★渲染频率：mujoco_viewer 内部开了 glfw.swap_interval(1)（vsync），
    #   所以每次 viewer.render() 会阻塞到下一个垂直同步 → 渲染被【显示器刷新率】卡住。
    #   若每个物理步都渲染（N=1），每显示一帧只推进 SIM_DT=0.005s：
    #     60Hz 屏 → 0.3 倍实时（慢放 3.3 倍）；144Hz → 0.72 倍；165Hz → 0.83 倍。
    #   要 1.0 倍实时需要 N = 1/(刷新率 × SIM_DT)：60Hz→3.3、144Hz→1.4、165Hz→1.2。
    #   所以默认【自动按本机刷新率算】，而不是写死一个数（写死 4 在 165Hz 屏上会变成 3.3 倍快放）。
    #   --render-every 传正数可手动覆盖；末尾仍会打印实测倍率供核对。
    if viewer is not None and render_every <= 0:
        try:
            import glfw as _glfw
            _rr = _glfw.get_video_mode(_glfw.get_primary_monitor()).refresh_rate
            render_every = max(1, int(round(1.0 / (_rr * SIM_DT))))
            print(f"      [渲染] 自动: 显示器 {_rr}Hz × 物理步 {SIM_DT}s "
                  f"→ 每 {render_every} 个物理步渲染一次（目标 ≈1.0 倍实时）")
        except Exception as _e:
            render_every = 4
            print(f"      [渲染] 刷新率探测失败({str(_e)[:40]})，退回每 {render_every} 步一次")
    render_every = max(1, render_every)
    if viewer is not None:
        print(f"[3/4] 渲染: 每 {render_every} 个物理步一次 "
              f"(请求 {1/(SIM_DT*render_every):.1f}Hz；每帧推进 {render_every*SIM_DT*1000:.0f}ms 仿真时间)")
    if no_render:
        use_keyboard = False
        print(f"[3/4] 无渲染模式（固定速度 x={speed_x}）")

    glfw = None
    window = None
    if viewer is not None:
        import glfw
        window = glfw.get_current_context()
        if use_keyboard:
            print("[3/4] GLFW键盘：W/S前后，Q/E左右，A/D转向，K=踢击，Space/R=复位")
        else:
            print(f"[3/4] 固定速度 x={speed_x}")
    # ==================== 手柄初始化 ====================
    # GLFW 手柄状态必须靠 poll_events 更新，而 poll_events 需要窗口/上下文；
    # 有 viewer 时复用它的窗口，无渲染模式下建一个【隐藏窗口】专门用于轮询（不弹窗）。
    gp = None
    gp_win = None
    if gamepad != 'off':
        import glfw
        if viewer is None:
            if glfw.init():
                glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
                gp_win = glfw.create_window(64, 64, "sim2sim-gamepad", None, None)
                if gp_win:
                    glfw.make_context_current(gp_win)
                else:
                    print("  ⚠️ 隐藏窗口创建失败，手柄无法轮询")
            else:
                print("  ⚠️ GLFW 初始化失败，手柄不可用")
        gp, _gp_jid, _ = init_gamepad(gamepad, axes=gp_axes, buttons=gp_buttons,
                                      deadzone=gp_deadzone, debug=gp_debug)
        if gp is not None and viewer is None and gp_win is None:
            print("  ⚠️ 无窗口可用 → 手柄停用（用 --no-render 时必须能建隐藏窗口）")
            gp = None
        if gp is not None:
            print("[3.5/4] 手柄已就绪（左摇杆前后/左右，右摇杆转向；可用键盘同时操作）")
    print("[4/4] 就绪")

    # ==================== 落地稳定 ====================
    print(f"      落地稳定 {settle_time}s（纯 PD 保持默认姿态，KP={kp} KD={kd}）...")
    # Always advance at least one physics step so sensor values (including
    # quaternion) are initialized even when settle_time is below SIM_DT.
    settle_steps = max(1, int(np.ceil(max(0.0, settle_time) / SIM_DT)))
    for i in range(settle_steps):
        q, dq, quat, omega = get_obs(data)
        tau = np.clip((DEFAULT_DOF_POS_MUJOCO - q[-rdof:]) * kp - dq[-rdof:] * kd, -TAU_LIMIT, TAU_LIMIT)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if clamp_vel:
            clamp_joint_vel(data, rdof)
        if viewer is not None and (i % render_every == 0):
            try:
                viewer.render()
            except Exception:
                viewer.close(); viewer = None; no_render = True
        if i % int(0.25 / SIM_DT) == 0:
            pg = quat_rotate_inverse(quat, np.array([0., 0., -1.]))
            print(f"        t={i * SIM_DT:4.2f}s  z={data.qpos[2]:.3f}  pg=({pg[0]:+.2f},{pg[1]:+.2f},{pg[2]:+.2f})")
    z_settled = data.qpos[2]
    pg = quat_rotate_inverse(quat, np.array([0., 0., -1.]))
    print(f"      稳定完成 z={z_settled:.3f}  pg=({pg[0]:+.2f},{pg[1]:+.2f},{pg[2]:+.2f})")
    if z_settled < 0.20:
        print("      ⚠️ 稳定阶段狗已趴下（z<0.2）——模型撑不住静态站立，先别测策略！")
        print("        排查：① 试 --kp 55 / --kp 70 ② --spawn-z 调出生高度")
        print("        若高 KP 仍塌 → 模型与训练 URDF 差异，需对齐模型")
    elif abs(pg[0]) > 0.5 or abs(pg[1]) > 0.5:
        print("      ⚠️ 稳定阶段狗已侧翻/倾斜（pg 水平分量大）——姿态失衡，检查模型质心/站位")

    # ---- 历史缓冲（与训练 reset 后的 obs_buf 一致）----
    # play.py 的第一帧策略输入仍是 reset 时的全零历史；不要用稳定站立帧
    # 预热 estimator，否则 sim2sim 的第一轮动作与 Isaac 不同。
    hist = deque(maxlen=HISTORY_LEN)
    for _ in range(HISTORY_LEN):
        hist.append(np.zeros(NUM_ONE_STEP_OBS, dtype=np.float32))
    last_actions_isaac = np.zeros(12, dtype=np.float64)
    target_q = DEFAULT_DOF_POS_MUJOCO.copy()
    cmd = np.array([speed_x, speed_y, speed_yaw], dtype=np.float64)
    # ★脚本化指令(用于测【站立→行走】这类切换瞬态): "--cmd-schedule '5:1.0,25:0'"
    #   格式: 逗号分隔的 时刻秒:vx[:vy[:wz]]，到点后一直保持该指令。
    #   为什么需要它: "站立→行走会蹦一下"这类问题只能在【切换瞬间】观察到，
    #   而此前只能靠键盘手切(不可复现、也没有精确的时间基准)。有了它就能量化
    #   切换瞬间的高度超调/竖直速度峰值，并且不同模型/策略之间可重复对比。
    sched = []
    if cmd_schedule:
        for item in str(cmd_schedule).split(','):
            item = item.strip()
            if not item:
                continue
            parts = item.split(':')
            _t = float(parts[0])
            _v = [float(x) for x in parts[1:]]
            while len(_v) < 3:
                _v.append(0.0)
            sched.append((_t, np.array(_v[:3], dtype=np.float64)))
        sched.sort(key=lambda kv: kv[0])
        print("      指令调度表: " + ", ".join(
            f"t={t:g}s→({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})" for t, v in sched))
    kick_steps_remaining = 0
    kick_key_was_down = False
    history_bootstrap = True

    def _oh():
        if frame_order == 'newest':
            return np.concatenate(list(hist)).reshape(1, -1)   # [最新 ... 最旧]
        return np.concatenate(list(hist)).reshape(1, -1)       # [最旧 ... 最新]

    # ===== [记录] 策略输入/输出记录器（与 play.py 同一 schema，可直接对比）=====
    # 数据落在 scripts/recorded_io/ 下，文件名 sim2sim_<月日_时分秒>.npz
    # 关掉：--no-record，或设环境变量 LG_RECORD_IO=0
    rec = None
    if record and os.environ.get('LG_RECORD_IO', '1') != '0':
        from record_io import IORecorder
        rec = IORecorder(tag='sim2sim', every=record_every, enabled=True,
                         env_index=0, csv=record_csv,
                         note=(f'policy={os.path.basename(policy_file)} wfp={_policy_wfp} '
                               f'scene={os.path.basename(scene_path)} '
                               f'frame_order={frame_order} remap={remap_mode} '
                               f'kp={kp} kd={kd} action_scale={ACTION_SCALE} '
                               f'clamp_vel={clamp_vel}(上限18.85/18.85/9.425) '
                               f'match_isaac_dofs={match_isaac_dofs} uniform_passives={uniform_joint_passives} friction={friction} '
                               f'cmd_schedule={cmd_schedule} spawn=({spawn_x},{spawn_y}) | '
                               f'event 位标志: 1=复位(Space/R) 2=踢击(K) 4=指令变化 | '
                               f'base_quat_wxyz 为 MuJoCo 原始 wxyz 序'))
        rec.__enter__()
        print(f"[记录] 策略输入/输出 -> {rec.out_dir}")

    # ===== [记录] 操作事件标记 =====
    # 用位标志按【控制步】记录"我做了什么"，回放时可以把操作与策略反应对齐：
    #   bit0(1)=复位   bit1(2)=踢击   bit2(4)=指令变化
    # 同一步可同时置多位（例如复位的同时指令也被清零）。
    EV_RESET, EV_KICK, EV_CMD = 1, 2, 4
    step_events = 0
    prev_cmd_rec = None

    count = 0
    total = int(600.0 / SIM_DT)
    # ★渲染频率统计：物理步 200Hz，而 viewer.render() 会受 GLFW vsync 限制
    #   （默认 60Hz 显示 → 每帧最多 60 次/秒）。若每个物理步都渲染，
    #   每显示一帧只推进 SIM_DT=0.005s → 实际只有 60×0.005 = 0.3 倍实时（慢放 3.3 倍）。
    #   改为每 render_every 步渲染一次后，每帧推进 render_every×0.005s：
    #     render_every=3 → 0.9 倍实时；=4 → 1.2 倍实时。
    #   末尾会打印实测实时倍率，可直接照着调。
    render_count = 0
    import time as _time
    t_wall0 = _time.time()
    for _ in range(total):
        # xfrc_applied is user-owned state in MuJoCo; clear it every step so
        # the kick remains a pulse instead of becoming a permanent force.
        data.xfrc_applied[:] = 0.0
        if viewer is not None and glfw.window_should_close(window):
            print("[用户关闭窗口，退出]")
            break
        # ---- 输入：手柄（模拟量） + 键盘（数字量），两者可同时使用 ----
        # 【朝向保持层】状态: 目标朝向（None=下一步用当前朝向初始化；复位时清零重来）
        hdg_target = None
        _hdg_err_last = 0.0

        x = y = yaw = 0.0; reset = False; exit_ = False; kick_pressed = False
        if gp is not None:
            if viewer is None:
                glfw.poll_events()          # 无渲染模式下没有别人 poll，手柄不会更新
            _gx, _gy, _gyaw, _greset, _gkick, _gexit = gp.poll()
            x += _gx; y += _gy; yaw += _gyaw
            reset = reset or _greset
            exit_ = exit_ or _gexit
            kick_pressed = kick_pressed or _gkick
        if viewer is not None and use_keyboard:
            for key, (axis, sign) in {
                glfw.KEY_W: ('x', 1), glfw.KEY_S: ('x', -1),
                glfw.KEY_A: ('yaw', 1), glfw.KEY_D: ('yaw', -1),
                glfw.KEY_Q: ('y', 1), glfw.KEY_E: ('y', -1),
                glfw.KEY_KP_8: ('x', 1), glfw.KEY_KP_2: ('x', -1),
                glfw.KEY_KP_4: ('yaw', 1), glfw.KEY_KP_6: ('yaw', -1),
                glfw.KEY_KP_7: ('y', 1), glfw.KEY_KP_9: ('y', -1),
            }.items():
                if glfw.get_key(window, key) == glfw.PRESS:
                    if axis == 'x': x += sign
                    elif axis == 'y': y += sign
                    elif axis == 'yaw': yaw += sign
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS: exit_ = True
            if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS: reset = True
            if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS: reset = True
            if glfw.get_key(window, glfw.KEY_K) == glfw.PRESS: kick_pressed = True

        # 【2026-09-25 修 bug】原条件 `sched is not None` 恒为真（sched 是列表，空列表也非 None）
        #   ⇒ 该分支永远执行，无手柄/键盘输入时 cmd 被 [0,0,0] 覆盖
        #   ⇒ `--no-render --no-keyboard --x_vel 1.0` 的【固定速度路径失效】(实测 cmd 恒为 0、狗只站着)。
        #   改为 `len(sched) > 0`：无任何输入源时保留 cmd = [speed_x, speed_y, speed_yaw]。
        if gp is not None or (viewer is not None and use_keyboard) or len(sched) > 0:
            if kick_pressed and not kick_key_was_down:
                kick_steps_remaining = kick_steps
                step_events |= EV_KICK          # [记录] 标记踢击发生的控制步
                if kick_steps_remaining:
                    print(f"[踢击] 对 trunk 施加 {kick_force:.1f}N 的 {kick_direction} 向力，持续 {kick_duration:.3f}s")
                else:
                    print("[踢击] 已禁用：kick duration 为 0")
            kick_key_was_down = kick_pressed
            cmd = np.array([x * SPEED_X, y * SPEED_Y, yaw * SPEED_YAW], dtype=np.float64)
            if sched:
                _ts = count * SIM_DT
                _cv = sched[0][1]
                for _t, _v in sched:
                    if _ts >= _t: _cv = _v
                cmd = _cv.copy()
            if exit_: break
            if reset:
                step_events |= EV_RESET         # [记录] 标记复位发生的控制步
                data.qpos[:7] = np.array([spawn_x, spawn_y, spawn_z, 1., 0., 0., 0.], dtype=np.float64)
                data.qpos[7:] = DEFAULT_DOF_POS_MUJOCO
                data.qvel[:] = 0
                data.ctrl[:] = 0
                data.xfrc_applied[:] = 0.0
                hist.clear()
                for _ in range(HISTORY_LEN):
                    hist.append(np.zeros(NUM_ONE_STEP_OBS, dtype=np.float32))
                last_actions_isaac[:] = 0
                target_q = DEFAULT_DOF_POS_MUJOCO.copy()
                kick_steps_remaining = 0
                history_bootstrap = True
                hdg_target = None          # 【朝向保持层】复位后重新以当前朝向为目标
                mujoco.mj_forward(model, data)
                print("[复位]")

        q, dq, quat, omega = get_obs(data)
        if count % DECIMATION == 0:
            if np.hypot(cmd[0], cmd[1]) < 0.2:
                cmd[0] = cmd[1] = 0.0
            # ---- 【朝向保持层 2026-10-06】把"操作员的转向速率"积分成"朝向目标"，
            #      再按训练同一公式 cmd_wz = clip(HEADING_P × 朝向误差, ±2) 下发。
            #      默认【关】= 原行为（操作杆 wz 直接下发）; 加 --heading-hold 才开启本层。
            if heading_hold:
                # ⚠️【2026-10-06 修正·转弯失效】第一版把操作杆速率【积分进朝向目标】再让 P 控制器追,
                #   目标以 r 前进 ⇒ 误差稳定在 r/p = 4 rad > π ⇒ wrap 把误差翻反 ⇒ 控制器反复反向,
                #   实测按住转向键 3s 只转 0.25 rad（应 6 rad）、键盘 4/6 转弯"用不了"。
                #   改成【两种模式分开】: 有转向输入 → 直接下发该速率（与原行为逐位一致）并把目标
                #   同步到当前朝向；无输入 → 才由本层接管保持朝向。
                _cur_yaw = _yaw_of_quat_xyzw(quat)
                _hdg_err_last = 0.0
                if abs(float(cmd[2])) > 1e-3:             # 转向模式（操作杆/调度表在转）
                    hdg_target = _cur_yaw                 # 松手即以此为新的保持点
                else:                                     # 保持模式
                    if hdg_target is None:
                        hdg_target = _cur_yaw             # 首次/复位后: 以当前朝向为目标
                    _hdg_err_last = _wrap_pi(hdg_target - _cur_yaw)
                    cmd[2] = float(np.clip(heading_p * _hdg_err_last, -WZ_CMD_MAX, WZ_CMD_MAX))
            # The first policy call after reset uses the zero history returned by
            # Isaac's BaseTask. Actual state enters the history after that action
            # has been simulated, matching play.py's observation timing.
            if not history_bootstrap:
                obs = build_obs(q[-rdof:], dq[-rdof:], quat, omega, cmd, last_actions_isaac,
                                remap, default_isaac, cmd_height=cmd_height)
                if frame_order == 'newest':
                    hist.appendleft(obs)
                else:
                    hist.append(obs)
            oh = _oh()
            with torch.no_grad():
                a = policy(torch.from_numpy(oh).float()).numpy().flatten()
            a = np.clip(a, -CLIP, CLIP)
            # ===== [记录] 与 play.py 完全同一口径：原始 obs 历史 + 裁剪后的原始动作 =====
            # 注意记的是 a（未 remap、未乘 ACTION_SCALE），保证与 Isaac 侧逐元素可比。
            if rec is not None:
                # step 用【控制步序号】(count//DECIMATION)，与 play.py 的口径一致；
                # 若记 count（仿真步）会与 Isaac 侧的步号差 DECIMATION 倍，无法直接对比。
                # [记录] 指令变化检测放在这里（而不是键盘处），因为小指令归零规则
                #   （np.hypot(cmd[0],cmd[1]) < 0.2 → 置零）也发生在这一步里。
                if prev_cmd_rec is None or np.abs(cmd - prev_cmd_rec).max() > 1e-9:
                    step_events |= EV_CMD
                prev_cmd_rec = cmd.copy()
                rec.add(step=count // DECIMATION, t=count * SIM_DT, obs=oh, action=a,
                        cmd=cmd,
                        # [记录] 机身运动轨迹：位置 + 四元数。只记速度看不出"走到哪、在哪摔的"。
                        base_pos=data.qpos[0:3],
                        base_quat_wxyz=data.qpos[3:7],
                        event=step_events,
                        dof_pos=q[-rdof:], dof_vel=dq[-rdof:],
                        base_lin_vel=data.qvel[0:3], base_ang_vel=omega,
                        target_q=target_q)
                step_events = 0                 # 事件只属于它发生的那一个控制步
            last_actions_isaac = a.copy()
            action_mj = a[remap]
            target_q = action_mj * ACTION_SCALE + DEFAULT_DOF_POS_MUJOCO
            history_bootstrap = False

            if count % (DECIMATION * 50) == 0:
                q_isaac = q[-rdof:][remap]
                vx = data.qvel[0]                      # 机体系前进速度
                print(f"[{count // DECIMATION:4d}] cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) hdg_err={_hdg_err_last:+.2f} "
                      f"z={data.qpos[2]:.3f} "
                      f"xy=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f}) vx={vx:+.2f} "
                      f"act[min/max]={a.min():+.2f}/{a.max():+.2f} "
                      f"q_isaac_min/max={q_isaac.min():+.2f}/{q_isaac.max():+.2f} "
                      f"dq_max={abs(dq[-rdof:]).max():.2f}")

        tau = np.clip((target_q - q[-rdof:]) * kp - dq[-rdof:] * kd, -TAU_LIMIT, TAU_LIMIT)
        data.ctrl[:] = tau
        if kick_steps_remaining > 0:
            data.xfrc_applied[trunk_body_id, :3] = kick_vector
            kick_steps_remaining -= 1
        mujoco.mj_step(model, data)
        if clamp_vel:
            clamp_joint_vel(data, rdof)
        if viewer is not None and (count % render_every == 0):
            try:
                viewer.render()
                render_count += 1
            except Exception as e:
                print(f"⚠️ render 失败，切换无渲染: {e}")
                viewer.close(); viewer = None
        count += 1

    if viewer is not None:
        viewer.close()
    # ===== 渲染/实时倍率实测（照这个数调 --render-every）=====
    _wall = _time.time() - t_wall0
    _sim = count * SIM_DT
    if no_render:
        print(f"[性能] 无渲染: 仿真 {_sim:.1f}s / 墙钟 {_wall:.1f}s = {_sim/max(_wall,1e-9):.2f} 倍实时")
    else:
        print(f"[性能] 渲染 {render_count} 次（每 {render_every} 个物理步一次，"
              f"物理 {1/SIM_DT:.0f}Hz → 请求 {1/(SIM_DT*render_every):.1f}Hz）")
        print(f"[性能] 仿真 {_sim:.1f}s / 墙钟 {_wall:.1f}s = {_sim/max(_wall,1e-9):.2f} 倍实时"
              f"（实际渲染 {render_count/max(_wall,1e-9):.1f}Hz）")
        print(f"       想要更接近 1.0 倍实时就把 --render-every 调成 "
              f"{max(1, round(render_every / max(_sim/max(_wall,1e-9), 1e-9)))}")
    # ===== [记录] 落盘（Ctrl-C / 异常时由 IORecorder 的 atexit 兜底，不会白跑）=====
    if rec is not None:
        rec.save()
    print("Done")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--no-keyboard", action="store_true")
    p.add_argument("--gamepad", default="auto", metavar="auto|off|N",
                   help="手柄控制：auto=用第一个检测到的 / off=关闭 / N=指定 jid（默认 auto）")
    p.add_argument("--list-gamepads", action="store_true",
                   help="只枚举手柄并退出（检查是否识别到、有多少轴多少个键）")
    p.add_argument("--gp-axes", default=None, metavar="vx,vy,yaw",
                   help="三个摇杆轴序号，默认 1,0,2（XInput 布局）。轴不对时用 --gamepad-debug 看实时值")
    p.add_argument("--gp-buttons", default=None, metavar="reset,kick,exit",
                   help="复位/踢击/退出三个按键序号，默认 0,2,7（A/X/Start）")
    p.add_argument("--gp-deadzone", type=float, default=None, help="摇杆死区，默认 0.12")
    p.add_argument("--gamepad-debug", action="store_true",
                   help="打印手柄原始轴/键值（标定轴序号用）")
    p.add_argument("--no-render", action="store_true", help="无渲染（解决 GLFW 窗口崩溃）")
    p.add_argument("--render-every", type=int, default=0,
                   help="每 N 个物理步渲染一次。0=自动（默认，按本机显示器刷新率算，目标 ≈1.0 倍实时）。"
                        "viewer 开了 vsync，所以渲染被刷新率卡住：N=1 在 60Hz 屏上只有 0.3 倍实时（慢放 3.3 倍），"
                        "在 165Hz 屏上 0.83 倍。给正数即手动覆盖；末尾会打印实测倍率照着调。")
    p.add_argument("--frame-order", choices=["newest", "oldest"], default="newest",
                   help="历史帧顺序：newest=最新在前（训练约定，默认）；oldest=兼容旧版实测")
    p.add_argument("--settle-time", type=float, default=SETTLE_TIME,
                   help="落地稳定时长 [s]（默认 0.5）")
    p.add_argument("--kp", type=float, default=KP, help="PD 刚度（训练值 30）")
    p.add_argument("--kd", type=float, default=KD, help="PD 阻尼（训练值 0.8）")
    p.add_argument("--remap", choices=["swap", "identity"], default="swap",
                   help="关节映射：swap=交换（实测确认，默认）；identity=恒等（A/B 用）")
    p.add_argument("--spawn-x", type=float, default=0.0, help="出生点 x（默认 0）。配合 --spawn-y 可对准不同车道")
    p.add_argument("--spawn-y", type=float, default=0.0, help="出生点 y（默认 0）。多车道场景用它对准某条车道")
    p.add_argument("--spawn-z", type=float, default=SPAWN_Z,
                   help="出生高度 [m]（默认 0.37=最新训练配置）")
    p.add_argument("--kick-force", type=float, default=KICK_FORCE,
                   help="K 键踢击力大小 [N]（默认 150）")
    p.add_argument("--kick-duration", type=float, default=KICK_DURATION,
                   help="K 键踢击持续时间 [s]（默认 0.10）")
    p.add_argument("--kick-direction", choices=["forward", "backward", "left", "right"], default="left",
                   help="K 键踢击方向（世界坐标，默认 left=+y）")
    p.add_argument("--x_vel", type=float, default=1.0)
    p.add_argument("--height", type=float, default=None,
                   help="目标机身高度 [m]（身高指令通道，缺省=名义高度 0.28；范围参考 0.26~0.33）")
    p.add_argument("--y_vel", type=float, default=0.0)
    p.add_argument("--yaw_vel", type=float, default=0.0)
    # ---- 关节速度限幅（复现 Isaac/PhysX 的关节速度约束）----
    p.add_argument("--clamp-vel", action="store_true",
                   help="每物理步把关节速度钳到 URDF 上限(18.85/18.85/9.425 rad/s)。"
                        "Isaac/PhysX 默认执行该约束，MJCF 无 maxvel 属性故 MuJoCo 默认不限速")
    # ---- 场景选择 ----
    p.add_argument("--scene", default=None,
                   help="场景 xml（相对路径以 resources/robots/LgDog1/xml/ 为基准）。"
                        "默认 scene.xml=整套障碍赛道；对比 Isaac 平地 play 请用 scene_flat.xml")
    # ---- 保真度旋钮（逐个归因 MuJoCo 与 Isaac 的物理差异）----
    p.add_argument("--heading-hold", action="store_true",
                   help="开启【朝向保持层】（默认关=原始行为）。开启时: 没有转向输入时由本地控制器"
                        "下发 wz = 0.5×朝向误差（与训练一致）⇒ 直行不画圈；按住转向键时仍直接"
                        "透传该速率，手感与原始行为完全一致。")
    p.add_argument("--heading-p", type=float, default=HEADING_P,
                   help="朝向保持层的比例增益（训练值是 0.5，一般不用改）")
    p.add_argument("--cmd-schedule", default=None,
                   help="脚本化指令，用于测切换瞬态: \"5:1.0,25:0\" = 第5秒起 vx=1.0、第25秒起停下。"
                        "格式 时刻秒:vx[:vy[:wz]]，逗号分隔。给了它就忽略键盘指令")
    p.add_argument("--uniform-joint-passives", action="store_true",
                   help="把全部铰链关节的被动参数统一为 MJCF <default> 关节类的值"
                        "(armature=0.01 damping=0.01 frictionloss=0.2) = go1.xml 的形态。"
                        "用于检验'calf 的 1:2 减速比折算(0.04/0.04/0.4)是不是 sim2sim 走不动的原因'")
    p.add_argument("--match-isaac-dofs", action="store_true",
                   help="把关节 armature/damping/frictionloss 全置 0，对齐 Isaac 训练侧"
                        "（cfg.asset.armature=0、URDF damping/friction=0；MuJoCo 模型原为 0.01/0.04 等）")
    # ---- ★保真度对齐（2026-09-23 新增）----
    p.add_argument("--snapshot", default=None,
                   help="训练 run 的 config_snapshot.json 路径: 自动读取并对照 sim.dt/decimation/kp/kd/"
                        "action_scale/spawn_z，默认采用快照值（治『硬编码常量与训练配置不一致』）")
    p.add_argument("--action-scale", type=float, default=None,
                   help="覆盖 action_scale（不传则由 --snapshot 决定）。MuJoCo 侧最优值不一定等于训练值，"
                        "可用它 0.25/0.35/0.5 扫一遍")
    p.add_argument("--decimation", type=int, default=None,
                   help="覆盖 decimation（不传则由 --snapshot 决定；控制频率 = 200/decimation Hz）")
    p.add_argument("--friction", type=float, default=None,
                   help="覆盖全部 geom 的滑动摩擦（训练标称 1.0；MuJoCo 模型原值 0.6）。不传=保持模型值")
    # ---- 策略选择 ----
    p.add_argument("--policy", default=None,
                   help="要验证的策略文件（默认 exported/policies/policy.pt）。"
                        "验归档策略请显式指定，如 policy_V13_6000.pt —— "
                        "policy.pt 会被后续每次导出覆盖，不指定就可能验错策略。")
    # ---- 策略输入/输出记录（数据落在 scripts/recorded_io/）----
    p.add_argument("--no-record", action="store_true",
                   help="关掉策略输入/输出记录（默认开启，存到 scripts/recorded_io/）")
    p.add_argument("--record-every", type=int, default=1,
                   help="抽帧间隔：1=每步都记（默认）；跑很久可设 5/10 降体积")
    p.add_argument("--record-csv", action="store_true",
                   help="额外存一份 CSV（270 列很宽，默认只存 npz）")
    args = p.parse_args()
    if args.list_gamepads:
        import glfw as _g           # 模块级 import 已绑定；这里只是确保可用
        if _g is None:
            raise SystemExit("未安装 glfw")
        if not _g.init():
            raise SystemExit("GLFW 初始化失败")
        _g.poll_events()
        pads = list_gamepads()
        if not pads:
            print("未检测到手柄。排查：① 是否插好/开机 ② 是否切到 XInput 模式 ③ ls /dev/input/js*")
        for jid, name, nax, nbtn in pads:
            print(f"jid={jid}  name={name!r}  轴数={nax}  按键数={nbtn}")
            _ax = gp_floats(_g.get_joystick_axes(jid))
        print(f"  轴的实时值: {[round(a,2) for a in _ax]}")
        if len(_ax) < 4:
            print("  ⚠️ 轴数少于 4 —— 设备不在手柄模式（飞智请切 XInput/手柄模式）")
            print(f"  按下的键  : {[i for i,b in enumerate(gp_buttons(_g.get_joystick_buttons(jid))) if b]}")
        _g.terminate()
        raise SystemExit(0)

    main(use_keyboard=not (args.no_keyboard or args.no_render),
         speed_x=args.x_vel, speed_y=args.y_vel, speed_yaw=args.yaw_vel, cmd_height=args.height,
         no_render=args.no_render, frame_order=args.frame_order,
         settle_time=args.settle_time, kp=args.kp, kd=args.kd,
         remap_mode=args.remap, spawn_z=args.spawn_z,
         kick_force=args.kick_force, kick_duration=args.kick_duration,
         kick_direction=args.kick_direction,
         record=not args.no_record, record_every=args.record_every,
         record_csv=args.record_csv, clamp_vel=args.clamp_vel,
         scene_xml=args.scene, policy_path=args.policy,
         match_isaac_dofs=args.match_isaac_dofs, friction=args.friction,
         uniform_joint_passives=args.uniform_joint_passives,
         cmd_schedule=args.cmd_schedule,
         heading_hold=args.heading_hold, heading_p=args.heading_p,
         spawn_x=args.spawn_x, spawn_y=args.spawn_y,
         render_every=args.render_every,
         snapshot=args.snapshot, action_scale=args.action_scale,
         decimation=args.decimation,
         gamepad=args.gamepad,
         gp_axes=[int(x) for x in args.gp_axes.split(",")] if args.gp_axes else None,
         gp_buttons=[int(x) for x in args.gp_buttons.split(",")] if args.gp_buttons else None,
         gp_deadzone=args.gp_deadzone,
         gp_debug=args.gamepad_debug)
