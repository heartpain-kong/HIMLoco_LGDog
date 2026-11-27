# LgDog1 sim2real —— 上位机部署栈

在 **N100** 上跑的策略控制器：手柄输入 → HIM 策略（270 维观测历史）→ 关节目标 →
经 **Type-C 到下位机（dm02 / STM32H7）** → 两条 FDCAN → 12 个 **灵足 RS-02**。
IMU 由下位机处理成欧拉角 + 陀螺仪角速度后回传。

```
  USB 手柄 ─┐
            ├─► N100（本目录）── Type-C ──► 下位机 ── FDCAN1/2 ──► 12×RS-02
   IMU ◄────┘        策略 / 安全 / 日志              纯桥，无本地控制律
```

## 三条命令上车

```bash
./install.sh                              # 一次性：venv + 依赖 + udev + 权限 + 预检
./selftest.sh                             # 自测：所有模块 + 预检 + 干跑整条链路
.venv/bin/python main.py --check --live    # 插上下位机的只读预检（通电机电前）
```

然后按 **[PORTING.md](PORTING.md)** 的"第一次上电顺序"走 —— 那份手册里有标定流程、
三个高危项（IMU 安装朝向 / 欧拉角单位 / 固件 `IMU.q` 语义）和一张排障对照表。

**没硬件也想先看它跑起来**（策略、状态机、安全链、日志全都会真跑一遍）：

```bash
.venv/bin/python main.py --dry-run
```

## 朝向保持层（2026-10-06 新增，默认关/opt-in）

**策略的观测里没有绝对朝向**（46 维单步观测只有 cmd/身高指令/角速度/重力投影/关节量），
训练时 yaw 指令是【环境的朝向控制器】算出来的：`cmd_yaw = 0.5×wrap(朝向目标 − 当前朝向)`（clip ±2）。
所以部署侧若把遥控的 `wz` 直接下发（开环，站住时 wz=0），就再没有东西纠正偏航 —— 策略那点
偏航偏置一直累积，表现为"直行画圈、原地旋转"。实测（Isaac，V38@8000，直行 40 s）：

| | 累积朝向漂移 | 偏置 ω̄ |
|---|---|---|
| 开环（直接下发 wz） | **−75.7°** ±4.0° | −0.034 rad/s |
| **本层（朝向保持）** | **−8.4°** ±1.7° | −0.003 rad/s |

做法（`policy/heading_hold.py`，与训练同一控制律，不需要重训）——**双模式**：
- 有转向输入（按住转向键）→ **直接透传该速率**（与原行为逐位一致），并把朝向目标同步到当前朝向；
- 无转向输入 → 下发 `wz = clamp(0.5×朝向误差, ±2)` 保持朝向。
⇒ 松开转向键后朝向被保持住（直行不画圈），而转弯手感与改动前完全一样。
⚠️ 首版把速率积分进目标再让 P 控制器追，误差稳定在 r/p≈4rad>π ⇒ wrap 反相 ⇒ 转弯卡死
（按住转向键 3 秒只转 0.25rad），已按上面双模式修正。
yaw 来源用接口已有的姿态四元数（`st["quat_wxyz"]`，策略算重力投影用的同一个）。

配置（`config.yaml` 的 `teleop` 段）：`heading_hold: false`（**默认关=原行为**，验证满意再打开）、`heading_p: 0.5`、
`heading_wz_max: 2.0`（训练里的 clip 值，别放大）、`heading_hold_dt: 0.02`。
`heading_hold: false` 即退回开环。

⚠️ 能力边界：本层靠 ±2 rad/s 的指令限幅工作，能压住的偏置约 ≤1.8 rad/s。
若某轮策略自身站立时会以 3 rad/s 自转（如 V38），本层只能把残余压到 ~2.4 rad/s，
必须同时靠策略侧修（见 tuning_log V39：站立样本锁朝向 + 姿态锚定 + yaw_hold）。

## 安全设计（读一眼再上电）

| 机制 | 行为 |
|---|---|
| **死人开关** | 不按键 = 保持站立；**按住 run 键**才放行策略。松手这个最自然的动作直接等于停机 |
| **急停键** | 闩锁（再按一下才解除），触发即阻尼刹车 |
| **阻尼刹车** | 急停 / 失联 / 倾倒 / NaN / IMU 陈旧 / 链路中断 / 目标持续超限 → 阻尼，且**必须人工按回车**才能重新起立 |
| **心搏看门狗** | 下位机 100ms 收不到上位机心跳就自己刹车（上位机崩了也不会失控） |
| **映射表闸门** | `hardware_map.yaml` 的 `verified` 为 false 时**拒绝进入策略控制**（只允许低速标定） |
| **策略指纹闸门** | 权重指纹与 `models/policy.json` 不符时拒绝启动（防"拿没验过的权重上真机"） |
| **双路守护** | `RuntimeGuard` 看**实测世界**（NaN/陈旧/倾倒/角速度/越限/链路）；`SafetyMonitor` 看**策略想干什么**（目标偏移限幅） |

## 目录

| 路径 | 作用 |
|---|---|
| `main.py` | 入口：状态机 + 预检 + 自动日志 → 见 `--help` |
| `spec.py` | **唯一**允许出现训练接口常量的地方（50Hz / kp=30 / kd=0.8 / action_scale=0.25 / 观测布局） |
| `config.yaml` | 硬件与安全阈值（不含训练常量） |
| `hardware_map.yaml` | 12 关节的总线/id/方向/减速比/零位 —— **默认全是猜测，必须标定** |
| `backends/` | `host_link`（真机）/ `stub`（干跑，带故障注入） |
| `interface/` | 协议、IMU（含安装变换）、关节映射、手柄 |
| `policy/` | 270 维观测历史 → 12 维动作（与 `sim2sim.py` 同口径） |
| `safety/` `startup/` | 守护与起立流程 |
| `tools/` | 标定 / 嗅探 / 与 sim2sim 对拍 / 日志 / 手柄探测 / IMU 定标 |
| `models/` | 随包策略 + 权重指纹登记 |
| `V6_h/` | **另一个方案**：V6_h 轮的策略（指纹 `bd3c045be439`）+ 它的训练配置 + 600s sim2sim 验证录制 + **移植说明**（★ 它要 100 Hz，见该目录 README） |
| `logs/` | 每次运行自动落盘（`logs/latest` 是软链接） |

## 常用命令

```bash
./run.sh                          # 真机跑策略（等价 .venv/bin/python main.py）
./run.sh --mode init              # 只起立保持，不跑策略（标定/吊架用）
./run.sh --dry-run                # 干跑
LGDOG_CPU=2 ./run.sh              # 控制环钉在 2 号核（N100 是 4 核）
./run.sh --decimation 2 --policy V6_h/policy.pt   # 跑 V6_h 方案（它是 100Hz 训的）
.venv/bin/python main.py --help   # 全部开关
.venv/bin/python main.py --list-devices        # 串口 + 手柄一览
.venv/bin/python tools/gamepad_probe.py --suggest   # 手柄轴号/键号标定
```

## 开发机上（不接硬件）

开发机通常用 conda 环境而不是 `.venv`，且没装 `pyserial`：

```bash
LGDOG_PYTHON=~/miniconda3/envs/himloco/bin/python ./selftest.sh
```

`--check` 会因缺 `pyserial` 报 ✗（这是**预期**的，开发机不接硬件）；
用 `--check --dry-run` 按"仅验证代码"的口径评估。

> ⚠️ **别直接用训练仓库里的 `exported/policies/policy.pt`** —— 它会被正在跑的训练轮
> 反复覆盖（实测：09-25 13:17 还是 V19 的 `fdf3053d8b29`，20:05 被覆盖成 `28599e9bf8b1`，
> 09-26 18:50 又变成 `92a645131d55` —— 而那份在 sim2sim 里刚给走路指令 **1.8 秒就倒**）。
> 部署一律用 `models/policy.pt`，换策略用 **`python tools/adopt_policy.py`**
> （自动算指纹、按指纹备份旧版、更新 `policy.json`、并把 sim2sim 的姿态验证数字抄进去）。
> 当前捆的是 `a13b6ff17f79`（= sim2sim 侧 `policy_auto_4000.pt`，跑满 600 s 未翻倒，
> pitch +7.10° 尚未达 ≤±4° 判据），详见 `models/README.md`。
>
> ℹ️ 另有一份 **V6_h**（`bd3c045be439`，pitch +4.52°，更接近判据）收在 `V6_h/`，
> 但它按 **100 Hz（decimation=2）** 训的 —— 跑它必须带 `--decimation 2`，见 `V6_h/README.md`。

## 文档

* **`../../../../dog/说明.md`** —— ★ 策略仓库说明（用哪份策略 / 怎么跑 / 契约怎么生效 / 上电前清单）
* **`PORTING.md`** —— 迁移 + 上电手册（拷过去 / 装 / 验证 / 标定 / 排障 / 本次改动清单）
* `logs/latest/summary.txt`、`events.jsonl` —— 出事故先看这两个
* `../../dm02/User_File/task/HOST_LINK_README.md` —— 下位机侧协议
* `../../dm02/ARCHITECTURE.md` —— 下位机固件结构
