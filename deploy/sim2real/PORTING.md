# LgDog1 sim2real —— N100 迁移与上电手册

> 目标：把 `deploy/sim2real/` 这一个目录拷到 N100，装一次，跑起来。
> 上位机代码**不依赖**训练仓库（Isaac Gym / MuJoCo / legged_gym），只依赖
> `numpy / PyYAML / pyserial / torch(CPU)` 和一个随包的 `models/policy.pt`。

---

## 0. 拓扑与实际分工（先确认你手上的东西和这张图一致）

```
                 Type-C（USB CDC 虚拟串口）
  ┌────────────┐ ─────────────────────────► ┌──────────────────────┐
  │  N100      │                             │ 下位机 dm02          │
  │  上位机     │ ◄───────────────────────── │ STM32H7，裸机，无 RTOS│
  │  · 跑策略   │   IMU(欧拉角+陀螺) + 12电机  └───┬──────────────┬───┘
  │  · 安全链   │      状态，周期回传               │ FDCAN1       │ FDCAN2
  │  · 手柄输入 │                            6×灵足 RS-02     6×灵足 RS-02
  └────────────┘
        ▲
        │ USB（手柄直接插上位机，不经过下位机）
   ┌─────────┐
   │  USB 手柄 │   ← 人机输入；死人开关 = 按住 run 键才放行策略
   └─────────┘
```

**下位机是纯桥**：只做「MIT 指令中继到 CAN」+「IMU 回传」+「电机状态回传」，
没有任何本地控制律。所以：

* 控制频率、PD 增益、观测构造**全在 N100 上**，改代码不用烧固件；
* 下位机只保留两个自己的安全动作：**心跳看门狗**（100ms 收不到 PC 心跳就自动阻尼刹车）
  和**电机软限位**。除此之外所有安全判断都在上位机 `safety/` 里。

### 上位机目录结构

```
sim2real/
├── main.py                  ★ 入口：状态机 + 安全链 + 自动日志（本次新增）
├── spec.py                  ★ 唯一允许出现"训练接口常量"的地方（50Hz/kp/kd/观测布局…）
├── config.yaml              硬件相关 + 安全/启动阈值（不含训练常量）
├── hardware_map.yaml        ★ 12 个关节的总线/id/方向/减速比/零位 —— 默认全是猜测，必须标定
├── models/
│   ├── policy.pt            随包策略（270 维输入 / 12 维输出）
│   └── policy.json          它的【权重指纹】与来源/验证记录
├── backends/                ★ 本次新增
│   ├── __init__.py          后端工厂 + 串口自动挑选
│   ├── host_link_backend.py 真机（经下位机中转）
│   └── stub_backend.py      干跑假机器人（无硬件，带故障注入）
├── interface/
│   ├── host_link.py         PC↔下位机 二进制协议（与固件 lgdog_link.h 逐字节对应）
│   ├── imu_mount.py         ★ 本次新增：IMU 系 → 机体系（装反 180° 的换算）
│   ├── imu.py               IMU 数据源抽象
│   ├── motor_mapping.py     关节空间 ↔ 电机空间
│   └── teleop.py            手柄（evdev/joydev）+ 键盘兜底
├── policy/policy_runner.py  270 维观测历史 → 12 维动作（与 sim2sim 同口径）
├── safety/                  runtime_guard（看"实测世界"）+ safety_monitor（看"策略想干什么"）
├── startup/                 起立流程 + 站姿保持
├── tools/                   标定/嗅探/对拍/日志/手柄探测/IMU 定标
├── install.sh               ★ 一键装环境（venv/依赖/udev/权限/预检）
├── run.sh                   启动包装（.venv + 调度小优化）
├── selftest.sh              ★ 一键自测（移植到新机器的第一件事）
├── make_offline_bundle.sh   离线安装包（N100 没网时用）
└── systemd/                 udev 规则 + 可选的开机自启单元
```

---

## 1. 拷过去

### 方式 A：有网 / 同一个局域网

```bash
# 在开发机上（把整个 sim2real 目录同步过去；logs/.venv/__pycache__ 不用带）
rsync -av --delete \
      --exclude '.venv' --exclude 'logs' --exclude '__pycache__' \
      ~/rl/LgRobotDog/deploy/sim2real/  <用户>@<N100-IP>:~/lgdog-sim2real/
```

### 方式 B：N100 没网（推荐，最省事）

```bash
# 开发机上造离线包（含 wheels 与代码）
./make_offline_bundle.sh --for-python 3.10      # ★ 填 N100 上 python3 -V 的版本
scp dist/lgdog-sim2real-offline-*.tar.gz <用户>@<N100-IP>:~/

# N100 上
tar -xzf lgdog-sim2real-offline-*.tar.gz
cd lgdog-sim2real-offline-*/sim2real
./install.sh --wheels ../wheels
```

**`--for-python` 不能省**（除非两台机器 python 版本一样）：`pip download` 默认按
**打包机自己的解释器**拉 wheel —— 这里是 3.8（`numpy-...-cp38-...`），而 N100 的
Ubuntu 22.04 / 24.04 是 3.10 / 3.12，cp38 的包装不进去，pip 会报
`not a supported wheel on this platform`。看 `wheels/` 里文件名的 `cpXX` 就知道是给谁拉的。

**Ubuntu 缺 `python3-venv` 也能装**：离线包里带了 `pip-*.whl`，`install.sh` 发现
`ensurepip` 不可用时会自动改走 `--without-pip` + 本地 wheel 引导，不需要联网、
也不需要 `apt install python3-venv`。只有完全手工装依赖时才需要它：

```bash
sudo apt install python3-venv        # 仅当你不走 install.sh 时才需要
```

> **最少要带的东西**：整个 `sim2real/` 目录 + `models/policy.pt`。
> 其他都可以重新生成，但 `models/policy.pt` 是训练产物，拷错版本就得回开发机重导。

---

## 2. 装一次

```bash
cd sim2real
./install.sh                    # 有网
./install.sh --wheels ../wheels # 离线
./install.sh --no-sudo          # 不碰 udev/用户组（自己已经有权限时）
```

它会做六件事，并逐条打印结果：环境自检 → 建 `.venv` → 装依赖
（CPU 版 torch）→ 串口权限（`dialout` 组）→ udev 规则 → 策略检查 → 跑 `main.py --check`。

**装完必须注销重登一次**（`usermod -aG` 对已登录的会话不生效）。
不想重登就先 `newgrp dialout` 开个子 shell 试试。

装完应该看到：

```
预检结果: ✅ 通过
```

---

## 3. 验证（不接硬件，先确认代码是好的）

```bash
./selftest.sh                     # 全部模块自测 + 预检 + 干跑（约 1 分钟）
./selftest.sh --quick             # 跳过干跑
.venv/bin/python main.py --dry-run # 干跑：假机器人跑通整条链路，约 17 秒
```

`--dry-run` 会：加载策略 → 一阶假机器人"起立" → 进 RUNNING → 靠零指令抑制保持站姿
→ 8 秒后自动刹车退出，并打印：

```
控制环: 目标 50.0 Hz，实际 600 步 / 12.0 s = 50.00 Hz
节拍质量: 周期 n=601  中位 20.00 ms  p99 25.41 ms  最大 26.83 ms  超期 0 次
```

**节拍质量这一行是 N100 上最该盯的数字**：p99 明显超过 20 ms 说明有东西在偷时间
（torch 推理被抢 CPU、串口阻塞、日志刷盘），而观测历史是按固定 20ms 构造的，
节拍抖动 = 给策略喂了错误的时间尺度。

还想验证安全链真的会跳闸（很值得做一次）：

```bash
.venv/bin/python main.py --dry-run --duration 8 --stub-fault-imu-freeze 1.0   # IMU 停更
.venv/bin/python main.py --dry-run --duration 8 --stub-fault-nan 1.0          # NaN 进观测
.venv/bin/python main.py --dry-run --duration 8 --stub-fault-drop-link 2.0    # 链路中断
.venv/bin/python main.py --dry-run --duration 8 --stub-tilt 80,0              # 机身倾倒 80°
```

每一条都应该看到 `🛑 阻尼刹车: guard: ...`。前三条的故障发生在**起立过程中**，
所以你会看到 `[PoseInit] guard 急停，放弃起立` —— 也就是起立被当场打断，
而不是傻乎乎地把机器人往站姿掰完再停。这条路径是本次移植专门补的（原来
`guard.check()` 只在主循环里调，起立期间守护根本没被喂过），别跳过。

对照：`--stub-tilt 5,0`（只倾 5°）应当**照常跑完**，状态行里 `倾角=5.0°`。

### 期望的验证结果（开发机实测，供你对照）

```
./selftest.sh                  → 自测结果: 通过 21  失败 0
main.py --dry-run              → 控制环 50.00 Hz，p99 ≈ 25 ms，超期 0 次
main.py --check                → 预检结果: ✅ 通过（装好 pyserial 之后）
main.py --dry-run --cmd 0.3,0,0 → blend 在 0.30 s 内 0→1（释放混合按预期工作）
```

---

## 4. 第一次上电顺序 ★★★

> 机器人**架起来**（吊起来或垫支架，四腿悬空），前几步不需要它承重。
> 每个阶段结束先断电确认，不要一路跑到底。

### 第 0 步：机械与电气检查

- [ ] 12 个电机都能被手盘动，无异响/卡死；小腿那一级能感觉到 2:1 减速的阻力
- [ ] 电池电压正常，急停按钮（如果有硬件急停）可用
- [ ] 下位机上电，`lsusb` 能看到 STM32 设备；`ls /dev/ttyACM*` 或 `/dev/lgdog_board`
- [ ] 手柄插上，`main.py --list-devices` 能看到它

### 第 1 步：只读检查（**不通电机电**）

```bash
.venv/bin/python main.py --check --live
```

看四件事：

1. **串口打开 + 收帧**：`link_alive=True`，`rx` 在涨
2. **IMU 原始诊断**（这一步最关键，见下面第 5 节的三个高危项）：
   ```
   |gyro|=0.0123 rad/s  静止倾角=1.8°  欧拉角vs四元数=0.42°（与共轭=179.6°）
   ```
3. **12 个关节回读**：数值应当**接近零位附近且互不雷同**。全部为 0 → 没收到反馈；
   某个值离谱 → 映射表/减速比填错
4. **温度/缺反馈/错误位**：`missing=0x000`，温度是室温

### 第 2 步：标定映射表（`hardware_map.yaml`）

`hardware_map.yaml` 的 `verified` 默认是 `false`，此时**策略被闸门拦住**
（只允许做低速标定动作）。这不是麻烦，是防止"关节和电机错位 → 一上电就乱甩"。

按文件头注释里的 7 步走：

```bash
.venv/bin/python tools/calibrate.py --map hardware_map.yaml check      # 纯静态
.venv/bin/python tools/calibrate.py --map hardware_map.yaml scan       # 扫 CAN ID
.venv/bin/python tools/calibrate.py --map hardware_map.yaml read       # 读 12 个角度
.venv/bin/python tools/calibrate.py --map hardware_map.yaml wiggle --joint FL_hip
.venv/bin/python tools/calibrate.py --map hardware_map.yaml direction --joint FL_hip
.venv/bin/python tools/calibrate.py --map hardware_map.yaml gear --joint FL_calf
.venv/bin/python tools/calibrate.py --map hardware_map.yaml offsets
```

标完把 `verified: true`，然后**再跑一次** `main.py --check --live` 确认 `(bus, can_id)`
与下位机固件里的 `LGD_JOINT_TO_MOTOR` 一致（两处不一致 = 关节错位）。

### 第 3 步：只起立，不跑策略

```bash
.venv/bin/python main.py --mode init
```

它会：软保持（低 kp）→ kp 斜坡 → 余弦插值到默认站姿 → 保持 → 等回车。

**这一阶段唯一要确认的事：每个关节是不是朝"生理方向"动。**
任何反向/乱动 → 立刻按手柄 estop 或 Ctrl-C，回去改 `hardware_map.yaml` 的 `sign`。
`sign` 填错的后果是策略一发指令关节就往限位里顶，非常容易烧电机。

### 第 4 步：策略（仍然架着）

```bash
.venv/bin/python main.py
```

- 起立 → 按回车确认 → HOLD（只维持站姿）
- **按住手柄 run 键** → 放行策略；松开就立刻回到站立保持
- 先给**极小**指令（`config.yaml` 里 `max_vx/max_vy/max_yaw` 默认已是保守值 0.6/0.2/0.4）
- 悬空状态下先看"腿的动作是否有节奏、幅度是否合理"，再看落地

### 第 5 步：落地

先落平地在人扶着的情况下试，别一上来过障碍。任何异常 → 松手（死人开关）比找急停键更快。

---

## 5. 三个高危项（**这三条错了，机器人一上电就会往一边倒**）

### (1) IMU 安装朝向 `spec.IMU_MOUNT_PERM / IMU_MOUNT_SIGN`

芯片在机身上是**绕 Z 转了 180°** 装的（IMU 的 X 向后、Y 向右、Z 向上，装在几何中央）。
`interface/imu_mount.py` 负责把它换成机体系（X 前 / Y 左 / Z 上）。
关掉它（`imu.apply_mount_transform: false`）等于把 IMU 系角速度与重力直接喂给策略。

自测里有专门的不变量：**IMU 系下水平 ⇒ 机体系下也水平**，而且自测明确打印了一句
"静置水平时左乘/右乘看不出区别，必须用倾斜姿态验证" —— 因为 180° 偏航在水平姿态下
恰好不可观测。所以：

```bash
.venv/bin/python interface/imu_mount.py       # 自测（含左乘/右乘反例）
```

### (2) 欧拉角单位：**固件发的是弧度**

`config.yaml` 的 `imu.euler_unit_is_degree` 已按固件实测改成 `false`
（依据 `dm02/User_File/task/imu_task.c`：`Roll/Pitch/Yaw` 直接取 `mahony.*`，
偏航回绕判据用 `3.1415926f` 而不是 `180.0f`）。
如果 `--check --live` 打出"静止倾角 500°"这种数，说明单位反了。

### (3) 固件 `IMU.q` 的语义未定 —— 用加计当地面真值定标

`imu_task.c` 的 Mahony 用旋转矩阵**第 3 行**当机体重力（世界→机体），
而矩阵公式匹配的却是姿态助手的**机体→世界**；且 `pitch = asin(+R20)` 与标准 ZYX 疑似反号。
所以**不要猜**，按下面三步定：

```bash
.venv/bin/python tools/imu_check.py --port /dev/ttyACM0
```

用**加速度计**当地面真值，摆三个姿态（水平 / 抬头 20° / 侧倾 20°），逐个对比：

| 现象 | 结论 | 怎么改 |
|---|---|---|
| 欧拉角与四元数夹角 ≈ 0° | 两者一致，用哪个都行 | 保持 `attitude_source: euler` |
| 夹角 ≈ 180° | 固件 q 是「世界→机体」 | `attitude_source: quat` 且对 q 取共轭（或直接用 euler） |
| 某个轴的符号不对 | 固件欧拉角符号约定与标准不同 | 改 `imu.euler_sign: [1, -1, 1]` |
| 陀螺仪模长在 5~20 之间 | 固件发的是**度/秒** | 该字段需要单位换算（别用 `gyro_sign` 凑） |

`main.py` 启动时的静止自检会把上面这些数字**全部打印出来并给提示**，
`--check --live` 也一样。策略模式下自检有疑点会**拒绝启动**（`--allow-bad-imu` 可强行继续，
但那样等于闭着眼睛跑）。

---

## 6. 零点 / 符号 / 减速比错了会怎样、怎么查

> ### ★ 2026-09-27 重要变更：固件删除了电机软限位
>
> 下位机固件里那 12 个 `Pos_min/Pos_max` **已删除**（桥接模式下它们只会
> ①静默夹住上位机目标、②在实测位置越界时静默把前馈力矩置 0 —— 制造
> "PC 发的角度和电机到的角度对不上"的假象）。
> 上位机侧对应改动：`hardware_map.yaml: board_limits_active: false`、
> `--check` 的 `[7/7]` 改为跳过并提示、`[8/8]` 新增"行程保护现状"。
>
> **后果（必须清楚）**：
>
> | 之前 | 现在 |
> |---|---|
> | 板侧有独立软限位，是一道**与零点无关**的护栏 | 板侧只剩**心跳看门狗**一条兜底 |
> | `RuntimeGuard` 的关节越限 + 板侧软限位，两道 | 行程保护**全在上位机**，且是**零点依赖**的 |
> | 6.3 节的 L0（静态交叉检查）与 L3（板侧越界）可用 | **这两个参考系都没了** → 只剩 L1/L2 |
>
> 补上的那一层是 `safety.SafetyMonitor` 的**关节限位夹取**（`clamp_to_joint_limits`，
> 默认开）：把最终目标**无条件**夹进 URDF 限位。它为什么必需 —— 目标偏移框是
> "相对默认站姿 ±0.6"，而 calf 默认 +0.5、框到 **+1.1**，URDF 限位只到 **+0.7**：
> 偏移框自己就会越限（仿真里由 PhysX 的关节限位兜住，真机原来由板侧软限位兜住，
> 现在两边都没了）。
>
> ⚠️ **所以零点标定从"重要"变成"关键"**：保护边界随零点平移，没有独立护栏了。
> 标定顺序见 6.5 节；L1（差分定符号）与 L2（机械限位探测）是现在唯一能
> 真正**测出/验证**零点的手段（这两个工具还没写，见第 12 节）。

这一节单独成章，因为**零点错是这套栈里最难自查的一类故障**：
它不会让任何数值越界，不会触发任何阈值，程序会一路报"成功"。

### 6.1 先说清楚为什么难查：闭环会自证

`PoseInitializer` 的到位判据是【**上报的** q 到达目标】。而上报的 q 是

```
q_上报 = (电机实际角度 − zero_offset) / sign / gear_ratio
```

零点错了一个常数 δ 时：

1. 控制器一直推电机，直到 **q_上报** 等于目标；
2. 此时机器人**物理姿态**与"默认站姿"差的正好是 δ；
3. 而所有自检看到的 q_上报 都完美收敛 —— 起立流程报
   `[PoseInit] ✅ 默认站姿初始化完成`。

**凡是"从上报值出发、再回到上报值"的检查，对常数零点误差完全不敏感。**
属于这一类的有：

| 检查 | 为什么看不见零点错 |
|---|---|
| `--mode init` 起立成功/失败 | 收敛判据就是上报值到目标 |
| `SafetyMonitor` 目标偏移限幅 | 比的是上报值相对 `DEFAULT_DOF_POS` 的偏移 |
| `RuntimeGuard` 关节越限 | 比的是上报值相对 URDF 限位 |
| `obs[9:21]` = (q − default) | 偏差被"吸收"进实际姿态，观测看起来正常 |
| `--check --live` 打印的 12 个关节角 | 它们本身就是用待验证的零点算出来的 |
| 温度 / 力矩 / 静态 PD 误差 | 稳态时上报误差为 0，PD 误差也为 0，不发热 |

★ **更要紧的推论**：`RuntimeGuard` 的关节越限保护是**零点依赖**的 ——
零点偏 δ，保护边界也跟着偏 δ。关节真顶到机械限位时上报值可能还在范围内，
越限保护**不会响**。所以不要把 guard 的关节检查当最后一道防线。
真正与我们的零点假设无关的，是**电机固件自己的软限位**。

### 6.2 一个常数偏差会造成什么

| 影响 | 说明 |
|---|---|
| 实际姿态整体偏 δ | 四足各关节 δ 不同 → 站姿扭曲、站高不对、可能根本站不住 |
| calf 行程被吃掉 | calf 限位只有 ±0.7 rad（±40°）。偏 0.4 rad 就吃掉一半行程，一侧直接顶死 |
| 板子软限位夹住目标 | 见 6.4：目标越出固件窗口 → 被静默夹到边界 → 关节到不了指令位置 |
| 前馈力矩被静默置 0 | 固件 `motor_task.cpp:204-205`：**实测**位置越出窗口时 `__send_bl=1`，前馈被丢 |
| 越限保护整体平移 | 见 6.1 的推论 |
| 上电就往一边倒 | δ 大到观测里的姿态与真实姿态严重不符时，策略直接 OOD |

### 6.3 四层检查：各自能看见什么、看不见什么

按"能不能真的看见"从弱到强（**前两层已实现，后两层要动手/待补**）：

| 层 | 手段 | 能查出 | 查不出 |
|---|---|---|---|
| **L0** 静态交叉检查<br>（~~已实现~~ **09-27 起失效**：固件软限位已删除，参考系不存在） | 把 URDF 关节行程映射到电机侧，看是否落在**固件软限位**内 | 零点/符号/减速比**自相矛盾**（含大偏差）；给出能自洽的候选 | 小偏差（窗口有富余就会"装进去"）；**符号的物理正确性**（两种 sign 各配一个 offset 都能装进去） |
| **L1** 差分检查<br>（待补：`tools/wiggle`） | 只动一个关节一点点，看**上报变化量**与**物理反应方向** | sign、gear_ratio（差分把零点消掉了）；配合 IMU 还能定符号 | 零点本身（差分天然消掉） |
| **L2** 机械限位探测<br>（待补：`tools/zero_probe`） | 用小 kp/限力矩**慢慢往机械端头顶**，记录上报 q 停在哪 → 得到"机械端点在上报坐标里的位置" | ★ **唯一能自动"测出"绝对零点的方法**（端点位置 − 已知机械行程 = 零点误差） | 需要知道机械端点的真值；要小心力矩与发热 |
| **L3** 运行时越界<br>（~~已实现~~ **09-27 起失效**：板子不再夹目标，bit4 不再出现） | 板子 `err_flags` bit4 + 我们逐帧算的窗口越界 → 守护急停 | 零点/符号错的**运行期表现**；定位到具体关节和差值 | 同上：只在越出窗口时才看得见 |

L0 与 L3 都建立在同一个独立参考系上：**固件对每个电机的 `Pos_min/Pos_max`**
（硬编码在 `motor_task.cpp`，桥接模式下依然生效）。这是除"机械/加计"之外，
唯一与我们的零点假设无关的一把尺子。

### 6.4 ★ 当时那张表的实测结论（**已由"删除固件软限位"解决，留作记录**）

`main.py --check` 的 `[7/7]` 段会逐关节算。**现在的结果是 4/12 关节冲突**：

```
关节          URDF行程→电机侧        固件窗口            默认姿态→电机侧  判定
FL_calf       [ -1.40, +1.40]  [  -3.67,  +0.00]     +1.00      ✗ 默认姿态被夹死
FR_calf       [ -1.40, +1.40]  [  +0.00,  +3.67]     +1.00      ✗ 行程越界 80°
RL_calf       [ -1.40, +1.40]  [  +0.00,  +3.67]     +1.00      ✗ 行程越界 80°
RR_calf       [ -1.40, +1.40]  [  -3.67,  +0.00]     +1.00      ✗ 默认姿态被夹死
（其余 8 个 hip/thigh 都在窗口内；但 thigh 只剩 4° 余量，也很贴边）
```

含义：

* `hardware_map.yaml` 现在是**全 sign=+1、calf gear=2、zero_offset=0**（未标定的猜测值）；
* calf 的固件窗口是**单边**的（0..210° 或 −210..0°），说明固件认为
  **calf 电机的零位在某个机械端点**，而不是在关节零位；
* 于是把 ±0.7 rad 的关节行程（= ±80° 电机侧）按 `zero_offset=0` 映射过去，
  有 80° 落在窗外；FL_calf / RR_calf 更严重：**连默认站姿都被夹死**，
  那条腿的膝盖根本伸不开，`--mode init` 必然起不来。

能自洽的候选（`--check` 会打印，两符号余量相同）：

| 方式 | 结果 |
|---|---|
| (a) 认为固件窗口是对的 | FL/RR_calf: `sign=+1, zero_offset=−1.833 rad (−105°)`<br>FR/RL_calf: `sign=+1, zero_offset=+1.833 rad (+105°)`<br>（`sign=−1` 配同样 offset 也能装进去 → **窗口定不了符号**，必须实测） |
| (b) 认为窗口是遗留值 | 下位机现在只是桥，那 12 个 `Pos_min/Pos_max` 是它自跑步态时代的产物 → 改成机械真实行程更合理 |

**当时的推荐是 (b) + 重新清零**（见 6.5），实际就是按 (b) 走的：固件软限位已删除：窗口一旦按机械行程放宽，它就变成一把
可信的、独立的运行时护栏；而 (a) 会让 `zero_offset` 背负一个 ±105° 的大偏置，
真正的机械零点仍然没被锚定，只是"看起来能跑"。

### 6.5 正确的零点标定顺序

固件提供了唯一能**锚定绝对零点**的手段：`LGD_MSG_SET_ZERO` → 电机自己的 `zero()`
（写进电机存储，断电不丢；固件只在**未使能**时接受）。上位机侧已经接好：

```bash
# 0) 先看现在的矛盾在哪
.venv/bin/python main.py --check                 # [7/7] 段

# 1) 【人动手】把机器人物理摆到关节零位姿态
#    URDF 里 12 个关节 q=0 的构型。用夹具/水平仪/量角器，不要用程序命令过去 ——
#    零点错的时候，命令过去到达的位置本身就不对（见 6.1）
#    ⚠️ 同时确认板子对 12 个电机的软限位窗口不会再夹住目标（6.4 的 (b)）

# 2) 记零位（写进电机）
.venv/bin/python main.py --set-zero              # 会打印当前读数 + 清单，要你输 YES

# 3) 复核：零位姿态下 12 个关节应都 ≈ 0
.venv/bin/python main.py --check --live

# 4) 定符号（唯一必须"用眼睛看"的一步）
#    小幅度动一个关节，看它实际朝哪边动；方向反了就把该关节的 sign 改成 −1
.venv/bin/python main.py --mode init             # 整条腿一起动，能看出方向对不对

# 5) 复核窗口一致性：改完 sign/offset 后 [7/7] 必须过
.venv/bin/python main.py --check
```

顺序不能颠倒：**先锚零点 → 再定符号 → 最后才允许 `--mode init` 承重**。
每改一次 `sign` / `zero_offset`，都要重跑 `--check`（L0 会立刻告诉你是否又矛盾了）。

---

## 7. 推理频率 / 时序怎么对齐

**这一节回答的是"策略为什么能在真机上按训练时的时间尺度跑"。**
对齐不是"记得设 50Hz"，而是三层机制叠起来：静态对拍 → 启动资格 → 运行时闸门。

### 7.1 要对齐的量（以及现在靠什么保证）

| 环节 | 训练侧（口径来源） | 真机侧 | 靠什么对齐 |
|---|---|---|---|
| 推理/控制周期 | `SIM_DT=0.005` × `DECIMATION=4` → **50Hz** | 上位机循环 | `spec.CONTROL_FREQ` 唯一来源；`config.control_freq` 必须相等（`--check` 硬校验）；`Pacer` 忙等到点 |
| 目标在一个周期内不变 | Isaac 把动作保持 4 个物理步（=20ms） | 板子 TIM8 每 2ms 把最新目标写进电机 | ★ **免费对齐**：PC 按 50Hz 发、板子 500Hz 零阶保持 = 训练语义 |
| 观测历史帧距 | 6 帧 × 20ms = **100ms** | `policy.step()` 的相邻调用间隔 | `StageTimer.mark_step()` 实测；启动资格 + 运行时闸门判它 |
| 历史帧顺序/首帧 | `frame_order='newest'`；reset 时 6 帧全零、首帧就用零历史 | `PolicyRunner.reset()` + `history_bootstrap` | 与 `sim2sim.py` 逐行同构（`check_alignment.py` 数值对拍） |
| `last_action` 语义 | 上一控制步的**原始动作**（未乘 action_scale） | 同上 | `policy.step()` 里 `self.last_action = a`，写目标时再乘 scale |
| 角速度新鲜度 | Isaac 每步给当前值（0 延迟） | 固件 200Hz 回传，取最新 | `--check --live` 量 `imu_max_age_ms`（最坏新鲜度） |
| 位置/速度新鲜度 | PhysX 当前值 | 电机状态 100Hz | 同上，量 `motor_max_age_ms` |
| 板子时钟 | — | HAL 1ms tick | `--check --live` 量"下位机时钟/上位机时钟"应 ≈1.0000 |

### 7.2 三层机制

**① 静态对拍（开发机跑，不碰硬件）**

```bash
python tools/check_alignment.py          # 把 spec.py 与 sim2sim.py 逐项断言相等
```
它比的是 `SIM_DT / DECIMATION / CONTROL_FREQ / HISTORY_LEN / NUM_OBS / KP / KD /
ACTION_SCALE / DEFAULT_DOF_POS / REMAP / TAU_LIMIT / VEL_LIMIT`，再做一次
"同一状态 → 观测/动作逐位相同"的数值对拍。**这是"常量口径"层面的对齐。**

**② 启动资格自检（真机，起立之后、放行策略之前）**

起立刚完成时机器人稳稳站着 —— 这是唯一能**安全地满负载测时序**的窗口：
每周期照常推理，但只写站姿目标。跑 `--timing-qual-seconds`（默认 2s）后打印：

```
推理间隔 p50 20.00 ms / p99 20.05 ms（阈值 26.0 ms）
观测历史跨度 p50 100.0 ms / p99 100.2 ms（训练口径恒 100.0 ms）
  read     p50  0.16 ms  p99  0.22 ms
  policy   p50  0.29 ms  p99  0.41 ms      ← torch 推理
  safety   p50  0.04 ms  p99  0.06 ms
  write    p50  0.02 ms  p99  0.04 ms
  log      p50  0.09 ms  p99  5.52 ms      ← 日志刷盘（最容易被忽略的尖峰）
计算占用：p50 0.48 ms / 预算 20.0 ms = 2%（余量 41.6x）
```
（以上是**开发机干跑**的实测值，供对照。N100 上 `policy` 那一行会明显更大。）

不达标 → **不放行策略**（进入 DAMPED，按回车可重试），除非显式 `--allow-slow-loop`。

**③ 运行时闸门 + 持续观测**

抑制期里（按键着但摇杆居中）策略照常在推理、写的却是站姿 ——
所以 RUNNING 分支里保留了**同一条闸门**：最近 200 个周期的**推理间隔** p99
超过 `--timing-max-period-ms`（默认 1.3×周期 = 26ms）就继续抑制、不放行。

> ★ 判据用的是**推理间隔**，不是"全体循环周期"。全体周期会把 DAMPED/HOLD 那些
> 不喂观测历史的周期（有刹车重发、有打印）也算进来，实测 p99 25.4ms vs
> 推理间隔 p99 20.1ms —— 用前者会虚高、误拦。

### 7.3 三个"不在我们控制内"的延迟（知道就好，别当成 bug）

| 量 | 训练侧 | 真机侧 | 影响 |
|---|---|---|---|
| `dq`（关节速度） | PhysX 瞬时速度 | 电机固件自己差分 + 滤波 | 分布不同（有相位滞后与量化噪声），`obs[21:33]` 的 0.05 缩放是按训练侧定的 |
| 姿态 | Isaac 直接给四元数 | 固件 Mahony 解算（有滞后） | **所以 `obs[3:6]` 的角速度必须用陀螺仪直读**，不要用欧拉角差分（这就是 `interface/imu.py` 里 `gyro_mode` 那条告警的意思） |
| 链路 | 无 | USB CDC 可能批处理 | `imu_max_age_ms` 就是它的量化指标 |

### 7.3 前 ★ 小腿减速比 1:2 → PD 增益要除以 4

`spec.KP/KD`（以及策略契约里的 kp/kd）是**关节侧**的值；写进电机 MIT 指令前要按减速比换算：

```
Kp_电机 = Kp_关节 / gear_ratio²      Kd_电机 = Kd_关节 / gear_ratio²
```

小腿有 1:2 额外减速 → **30 → 7.5、0.8 → 0.2**；hip/thigh 无额外减速 → 不变。
代码在 `MotorMapping.kp_kd_to_motor()` 里自动做，`--check` 的 `[8/9]` 段打印对照表；
开关是 `hardware_map.yaml: gain_correction`（正常必须 true）。
**完整推导、电机余量核算、台架核实方法见 `rl/dog/小腿减速比_kp_kd_计算.md`。**

### 7.3c ★★ 观测布局也是契约：45 维（旧）/ 46 维（新，含身高指令通道）

2026-10-01 训练侧给观测加了**身高指令通道**，单帧从 **45 维变 46 维**、总计 **270 → 276**：

| 段 | 45 维（v1，旧） | **46 维（v2，新）** |
|---|---|---|
| 指令 cmd(vx,vy,wz) | [0:3] | [0:3] |
| **身高指令** | — | **[3]** = (目标身高 − `height_nominal`) × 10 |
| 角速度 | [3:6] | [4:7] |
| 投影重力 | [6:9] | [7:10] |
| 关节角 − 默认站姿 | [9:21] | [10:22] |
| 关节速度 | [21:33] | [22:34] |
| 上一步动作 | [33:45] | [34:46] |

- 布局由每份策略的 `policy.json: control_contract.obs_one_step` 声明，
  `main.py` / `PolicyRunner` 会**自动切换**（`spec.set_obs_layout()`）；
- `--snapshot` 也会按训练快照里的 `num_one_step_observations` 切；
- **真机上不给身高指令时该维恒 0**（= 该策略的名义站高，与 `sim2sim.py --height` 缺省一致）；
  要用它下发高度就加 `--height 0.32`（单位 m）；
- ★ **`height_nominal` 是每个 run 各自的**（和 kp/kd/action_scale 同类），写在
  `policy.json: control_contract.height_nominal`，`main.py` 自动应用、`--snapshot` 也能取。
  **V41/V39/V41b 系是 0.30，V32 是 0.28。**
  ⚠️ 这里踩过坑：训练侧把它从 0.33 改成 0.30 后，`sim2sim.py` 改了而 `spec.py` 留在 0.28 ——
  **不传 `--height` 时完全看不出来**（该维恒 0，"碰巧对上"），一传 `--height 0.32` 就变成
  `(0.32−0.28)×10 = +0.4`，策略以为要 0.34 m，**指令被放大一倍**。
  `tools/check_alignment.py` 现在会逐点数值核对 obs[3] 的两边编码；
- 写观测的地方**一律从 `spec.OBS` 取分段**（如 `obs[o["omega"]]`），不要写死下标 ——
  两套布局并存，写死必然错位。

### 7.3d 身份登记表的查找顺序（一个目录放多份策略时）

`main.py` 找"这台机器该跑哪一版"的登记表时，顺序是：

```
<策略目录>/<策略文件名>.json      ← 同名的专用登记（例: V43/policy_X.pt → V43/policy_X.json）
  ↓ 没有才退
<策略目录>/policy.json            ← 目录级登记（常规情况）
  ↓ 还没有才退
<sim2real>/models/policy.json     ← 兜底（并把两边指纹摆出来）
```

为什么要"同名优先"：`dog/V43/` 这种目录会同时放**主策略 + 备选检查点**，
两份指纹不同却只有一份 `policy.json` ⇒ 备选那份一定会被【权重指纹闸门】拦住，
而绕过闸门（`--allow-policy-mismatch`）恰恰是最不该做的事。
给每份策略配一份同名 sidecar，就能各自声明身份与契约。

### 7.3a ★★ 契约跟着策略走：`policy.json` 的 `control_contract`

**不同 run 的训练契约本来就不一样**，实测三份：

| 策略 | 单帧 obs | decimation | 控制频率 | 历史跨度 | kp | kd | action_scale |
|---|---|---|---|---|---|---|---|
| **`rl/dog/V32/policy.pt`（39c92a0634d3，推荐）** | **46** | 4 | 50 Hz | 100 ms | 30 | **1.2** | **0.3** |
| `rl/dog/policy.pt`（0bee8f5b8c87，V28） | 45 | 4 | 50 Hz | 100 ms | 30 | 1.2 | 0.3 |
| `models/policy.pt`（a13b6ff17f79） | 45 | 4 | 50 Hz | 100 ms | 30 | 0.8 | 0.25 |
| `V6_h/policy.pt`（bd3c045be439） | 45 | **2** | **100 Hz** | **50 ms** | 30 | 0.8 | 0.25 |

所以**契约不能是一份全局常量**：用一份常量去套所有策略，必然有一批是静默错的
（本仓库 action_scale 0.25/0.5 错配就作废过三个 60000 步的测试）。
现在的做法：**每份策略旁边的 `policy.json` 里写 `control_contract`，`main.py` 启动时自动应用并打印来源。**

优先级（高→低）：

```
--kp/--kd/--action-scale (CLI，仅 A/B)
  > --snapshot <训练配置原件>          ← 会逐项对照并拒绝硬项不符
    > 策略自带 policy.json 的 control_contract
      > config.yaml
        > spec.py 默认值
```

⚠️ 注意：`config.yaml` 里那份 `kp/kd` 是**训练值的副本**（文件头也写了"不在这里定义第二遍"），
所以契约**会覆盖它** —— 实测过 config 里写死的 `kd=0.8` 会把 V28 契约里的 `1.2` 挡住，
于是静默按 0.8 跑，正是要消灭的那类错配。

所以换策略只需给 `--policy`（+ 建议给 `--snapshot` 复核），不必手改 config。
`--check` 的 `[2/8]` 段会把"当前生效"的频率与 PD 打印出来。

### 7.3b ★ 换一份"按别的频率训出来"的策略：用 `--decimation`

控制频率是**策略契约的一部分**（观测历史的帧距 = 控制周期），所以换策略时频率必须一起换：

| 策略 | decimation | 控制频率 | 历史跨度 | 每周期推理预算 |
|---|---|---|---|---|
| 主策略 `a13b6ff17f79`（默认） | 4 | 50 Hz | 100 ms | 20 ms |
| **`V6_h/bd3c045be439`** | **2** | **100 Hz** | **50 ms** | **10 ms** |

```bash
python main.py --decimation 2 --policy V6_h/policy.pt
```

- 覆盖入口在 `spec.apply_decimation()`（spec 仍是常量的唯一来源），启动时会打印醒目提示；
- 它会**同时**把 `config.yaml` 的 `control_freq` 按新频率处理，否则 `--check` 会报不一致；
- 时序资格自检的阈值变成 `1.3 × 10 ms = 13 ms`，**推理预算减半** —— N100 上必须实测；
- 只换权重不改频率的现象是「能动但节奏全乱、站不稳」，**不会报错**。`sim2sim.py --snapshot`
  会替你查这一项（它会把 `decimation` 与训练快照逐项对照并标 ❌）。

### 7.4 在真机上怎么量、怎么判

```bash
.venv/bin/python main.py --check --live      # 上行速率 + 时钟比 + 新鲜度
.venv/bin/python main.py --mode init          # 只想看时序，不想跑策略时
.venv/bin/python main.py                      # 启动时会自动跑时序资格自检
```

| 数字 | 正常 | 不正常说明 |
|---|---|---|
| 推理间隔 p99 | ≤ 26 ms（≈20.0 最好） | 控制环跑不进预算 → 观测历史时间尺度失真 |
| 观测历史跨度 p99 | ≈ 100 ms | 同上；这是**最直接的"对齐了吗"指标** |
| `policy` 阶段 p99 | 远小于 20 ms | torch CPU 被抢时间片 → `OMP_NUM_THREADS=2` / `LGDOG_CPU=2` |
| `log` 阶段 p99 | 远小于 20 ms | 日志刷盘尖峰 → `log.every` 调大 / `flush_interval_s` 调大 |
| IMU 实测速率 | ≈200 Hz | 明显偏低 → USB CDC/读线程跟不上，角速度会变旧 |
| IMU 最大龄 | ≈5 ms | 远大于 5ms → 丢帧或批处理 |
| 时钟比 | ≈1.0000 | 偏很多 → 板子时钟树与固件注释不一致 |

**不达标的旋钮**（按见效顺序）：`LGDOG_CPU=2` 钉核 → `OMP_NUM_THREADS=2` →
`config.log.every: 5` → `--no-log`（先排除日志）→ 换 USB3 口/短线。
`--allow-slow-loop` 能硬跑，但等于让策略用错的时间轴。

---

## 8. 日常用法

```bash
./run.sh                        # 真机跑策略 = .venv/bin/python main.py
./run.sh --mode init            # 只起立保持
./run.sh --dry-run              # 干跑
LGDOG_CPU=2 ./run.sh            # 把控制环钉在 2 号核（N100 有 4 核，留核给串口线程）

.venv/bin/python main.py --help # 全部开关（约 40 个，按用途分组）
```

常用开关：

| 开关 | 用途 |
|---|---|
| `--check` / `--check --live` | 预检；`--live` 加做真机只读检查 |
| `--dry-run` | 干跑（= `--backend stub --yes --gamepad-off --auto-run`） |
| `--mode init` | 只起立保持，不需要 `verified` 映射表 |
| `--cmd 0.3,0,0 --cmd-is-fixed` | 固定速度指令（台架/吊架测试） |
| `--auto-run-seconds 20` | 跑 20 秒后自动刹车退出 |
| `--port /dev/lgdog_board` | 指定串口（默认自动挑） |
| `--policy <path>` | 指定别的策略（会与 `models/policy.json` 的指纹比对） |
| `--stub-fault-{nan,imu-freeze,motor-missing,drop-link,backend} SEC` | 干跑故障注入（见第 3 节） |
| `--timing-qual-seconds 2` | 起立后的时序资格自检时长（0=跳过） |
| `--timing-max-period-ms 26` | 推理间隔 p99 的放行阈值 |
| `--allow-slow-loop` | 时序不达标也放行策略（★ 时间轴会失真） |

**安全设计：死人开关。** 不按任何键 = 保持站立，按住 run 键 = 放行策略。
为什么"按住才跑"：四足失控就是真摔，松手这个最自然的动作直接等价于停机，
不需要人在慌张时还记得该按哪个键。急停键是**闩锁**的，再按一下才解除。

---

## 9. 出问题怎么查

| 现象 | 原因 | 处理 |
|---|---|---|
| `找不到下位机串口` | 没插 / 没权限 / 被别的程序占用 | `main.py --list-devices`；`sudo usermod -aG dialout $USER` 后**重登** |
| `Permission denied: /dev/ttyACM0` | 不在 dialout 组 | 同上；或者 `--no-sudo` 装完忘了重登 |
| `not a supported wheel on this platform` | 离线包的 wheel 是别的 python 版本 | 重打包，加 `--for-python <N100 的版本>` |
| `ensurepip is not available` / venv 建不起来 | Ubuntu 缺 python3-venv | 用离线包 + `./install.sh --wheels`（会自动引导）；或 `apt install python3-venv` |
| `5.0s 内没收齐上行帧` | 固件没烧 host_link / 串口选错 / 死了 | `tools/link_sniffer.py` 看有没有帧 |
| 手柄没反应 | 不在 input 组 / 轴号不对 | `tools/gamepad_probe.py --suggest`，把实测号填进 config |
| 键盘兜底按不住 | 终端关了键自动重复 | `xset r on`，或 `keyboard.run_is_toggle: true` |
| `Sim2RealMapNotVerified` | `verified: false` | 按第 4 节第 2 步标定；或先 `--mode init` |
| `策略指纹不符` | `policy.pt` 被覆盖/拷错 | 见下一条 |
| 关节朝反方向动 | `sign` 填错 | 回第 4 节第 3 步，改 `hardware_map.yaml` |
| 小腿动一半/两倍 | `gear_ratio` 该是 2.0 | `tools/calibrate.py gear --joint <腿>_calf` |
| 静止倾角 500° | 欧拉角单位是度而不是弧度 | `--imu-degrees`，或改 config |
| 一上电就往一边倒 | IMU 安装变换/符号/姿态源错 | 第 5 节三条，逐个用 `imu_check.py` 定 |
| 某条腿膝盖伸不开 / 起立到一半失败 | **关节行程越出固件软限位，被夹住** | `main.py --check` 的 `[7/7]` 段，见第 6 节 |
| 小腿比其他关节"硬"很多 / 一动就顶限位 | PD 增益没按减速比换算（应除以 gear²=4） | `main.py --check` 的 `[8/9]` 段；见 `rl/dog/小腿减速比_kp_kd_计算.md` |
| 时序自检不达标 / 策略迟迟不放行 | 控制环跑不进 20ms | 看阶段耗时分解找是谁在吃预算，见第 7.4 节 |
| `err_flags` 出现 `0x10` | 同上（板子把目标夹了）。它**闩锁**，要复位下位机才清 | 第 6.4/6.5 节 |
| 起立报"✅ 已到位"但姿态明显不对 | **零点错的典型签名**（闭环自证，见 6.1） | 第 6.5 节的标定顺序 |
| `--set-zero` 回 `rejected` | 下位机处于使能态（固件只在未使能时接受） | 断电重启下位机再试 |
| 零点姿态下 `--check --live` 读到的不是 0 | 要么没摆到零位，要么 `zero_offset/sign` 还没配 | 第 6.5 节 3)~4) 步 |
| 控制环 p99 远大于 20ms | torch 被抢 CPU / 日志刷盘 / 串口阻塞 | `LGDOG_CPU=2`，`OMP_NUM_THREADS=2`，`log.every: 5` |
| 起来一半就刹车 | guard 抓到东西（IMU 陈旧/NaN/倾倒/链路断） | 看日志 `events.jsonl` 里的 `BRAKE` 原因 |

### 日志在哪

每次启动自动建 `logs/<tag>_<MMDD_HHMMSS>/`，`logs/latest` 是软链接：

| 文件 | 内容 |
|---|---|
| `meta.json` | 配置快照 + spec 指纹 + **策略权重指纹** + 后端描述（排查"配置漂移"全靠它） |
| `state.csv` | 控制周期逐帧摘要（一行 = 一个周期，50Hz） |
| `events.jsonl` | 事件流（状态切换/急停/失联/安全触发），**每条立即 flush** |
| `data.npz` | 完整 obs / action / target_q / q / dq（只在下发策略时有） |
| `summary.txt` | 人读总结 |

出事故先看 `events.jsonl`：`BRAKE` 那条的 `msg` 就是原因，往前翻 `STATE` 看是在哪个阶段。

### 紧急情况

* 手柄 **estop 键**（闩锁）→ 立刻阻尼刹车
* **松开 run 键** → 立刻回到站立保持（最快）
* **Ctrl-C** → 正常收尾（刹车 + 存日志）
* 都不行 → 断电池

---

## 10. 移植到别的机器时要改什么

| 位置 | 项 | 说明 |
|---|---|---|
| `config.yaml` | `serial_port` | 或命令行 `--port`；留空/无效会自动扫 |
| `config.yaml` | `hardware_map` | 映射表路径，默认同目录 |
| `config.yaml` | `imu.*` | 单位/符号/姿态源 —— **每台机器都要实测** |
| `config.yaml` | `teleop.axes/buttons` | 手柄型号不同就不一样，用 `gamepad_probe.py --suggest` |
| `hardware_map.yaml` | 12 个关节的 bus/id/sign/gear/offset | 每次重装/换电机都要重新标定 |
| `models/policy.pt` + `policy.json` | 策略与它的指纹 | 换策略必须同步改指纹 |
| `rl/dog/`（仓库外） | **策略仓库**：V32（推荐）/ V28 / PD 增益推导 / 入口说明 | 部署用 `rl/dog/V32/policy.pt`，先读 `rl/dog/说明.md` |
| `spec.py` | 训练接口常量 | **只在训练配置真的改了时才动**，并且要在 sim2sim 里同步验证 |

**不需要改的**：所有 Python 路径都是相对 `__file__` 的，没有写死 `/home/xxx`。

### 与开发机的差异

开发机上跑 `--check` 时，`pyserial` 缺失或"没发现串口"属于**预期失败**
（开发机不接硬件）。`./selftest.sh` 会自动识别这种情况并退到 `--check --dry-run`。
开发机如果用的是 conda 环境而不是 `.venv`：

```bash
LGDOG_PYTHON=~/miniconda3/envs/himloco/bin/python ./selftest.sh
```

---

## 11. 本次为移植新增/改动了什么（便于 review）

| 文件 | 变化 |
|---|---|
| `main.py` | **新增**。之前整条栈缺入口：状态机（INIT→CONNECTED→ENABLED→HOLD→RUNNING/DAMPED）、`--check` 预检、自动日志、收尾。 |
| `backends/__init__.py` | **新增**。后端工厂 + 串口自动挑选（按 USB vid，不写死 `ttyACM0`）。`auto` 找不到串口会**报错**而不是悄悄退化。 |
| `backends/host_link_backend.py` | **新增**。真机 IO：`HostLinkBus` + IMU 解析 + 安装变换 + 电机错误位解读 + 静止自检。 |
| `backends/stub_backend.py` | **新增**。无硬件干跑后端（纯标准库），带虚拟时钟与 4 种故障注入。 |
| `interface/imu_mount.py` | **新增**。IMU 系 → 机体系。原来 `spec.imu_vec_to_body()` **只在 `imu_check.py` 里用过，运行路径上没人调用** —— 也就是角速度和姿态根本没做变换。 |
| `safety/runtime_guard.py` | 加了一条判据：`link_alive=False` → STOP。理由见下一条。 |
| `main.py: GuardedIO` | **新增的一层**。原来 `guard.check()` 只在主循环里调，**起立流程期间守护完全没被喂过** —— IMU 在起立中途挂掉、关节读出 NaN，起立照样一路把机器人往站姿掰。现在"谁读状态谁喂守护"，起立/保持也走这条路。 |
| `policy/policy_runner.py` | 加 `weight_fingerprint()`（按参数名排序对张量字节做 md5）。 |
| `models/` | 随包策略 + 指纹登记。**实测 `exported/policies/policy.pt` 会被训练过程覆盖**（13:17 还是 V19，20:05 被正在跑的 V20 覆盖成中间快照），所以不能直接用训练仓库里那份。 |
| `config.yaml` | `imu.euler_unit_is_degree` 改成 `false`（按固件实测）；新增 `backend` / `hardware_map` / `attitude_source` / `euler_sign` / `gyro_sign` / `apply_mount_transform`。 |
| `interface/motor_mapping.py` | `map_path=None` 视作"用默认路径"（原来会抛 `TypeError`）。 |
| `install.sh` / `run.sh` / `selftest.sh` / `make_offline_bundle.sh` / `systemd/` | **新增**。install.sh 会在缺 `python3-venv` 时自动用离线包里的 pip wheel 引导 venv（完全离线可装）。 |
| `safety/safety_monitor.py`（**新增关节限位夹取**） | 固件删除软限位后，这是**唯一**的绝对行程保护：把最终目标无条件夹进 URDF 限位。为什么必需 —— 目标偏移框（站姿 ±0.6）本身会越限（calf 到 +1.1，限位只到 +0.7，仿真里由 PhysX 兜住）。日志新增 `tgt_off_max` 供现场看夹取是否生效。 |
| `spec.py` / `main.py`（**`--decimation`**） | 新增受控的控制频率覆盖：`spec.apply_decimation()` + `main.py --decimation N`。理由见 7.3b —— 换按别的频率训出的策略（如 `V6_h` 是 100Hz）必须一起换频率。 |
| `V6_h/` | **新增**：把 V6_h 轮的策略收进 sim2real（`policy.pt` + `policy.json` + 训练配置 + 600s 验证录制 + `README.md` 移植说明）。 |
| `tools/adopt_policy.py` | **新增**：把 sim2sim 当前验证的策略收编进 `models/` —— 算指纹 → 按指纹备份旧版 → 覆盖 → 重写 `policy.json` 并抄入 `analyze_attitude.py` 的验证数字；**没在 sim2sim 里验过会给出警告**。换策略一律用它，别手敲 `cp`。 |
| `tools/sync_to_usb.sh` | **新增**：同步到 U 盘并按校验复核；同步前会拦住"盘上比仓库更新"的文件（防止把带出去改过的源码覆盖掉），支持 `--pull` 拉回。 |
| `tools/check_alignment.py`（不拷） | 开发机专用：要 mujoco + 训练仓库的 sim2sim.py。`selftest.sh` 在目标机上会自动跳过它，不算失败。 |

**验证方式**（都是可复现的）：

```bash
./selftest.sh                                  # 21 项全绿
.venv/bin/python main.py --dry-run             # 整条链路，50.00 Hz，超期 0 次
.venv/bin/python main.py --dry-run --duration 8 --stub-fault-imu-freeze 1.0
#   → 起立到一半被打断：guard 急停，放弃起立（补上这个洞的直接证据）
.venv/bin/python main.py --dry-run --duration 8 --stub-tilt 80,0
#   → guard STOP tilt: pg.z=-0.17 > -0.30（机身倾倒判据生效）
```

---

## 12. 还没做 / 已知限制

* **`hardware_map.yaml` 仍然是猜测值**（`verified: false`）。上真机第一件事就是标定它。
* 固件侧：`safety_pos_limit_violation` 一旦触发**闩锁且没有恢复路径**（要切 init 才能清）；
  电机 `MError` 只上报不处理；CAN 发送返回值没检查。
* `interface/host_link.py` 假设 `IMU` 帧的 `flags` bit0 表示姿态就绪（已按固件实现）。
  固件上电前 ~3 秒欧拉角是 0，`main.py` 会等它置位并报告等待情况。
* 下位机的 FDCAN 波特率不确定（若内核时钟是 HSE 24MHz 则约 200kbps，若 PLL1Q 120MHz 则 1Mbps）。
  **以能正常驱动的电机为准**，改之前先记录当前值。
* **零点标定目前只能"静态查矛盾"，不能"自动测出零点"**：L0（已实现）能查出
  自相矛盾的 sign/gear/offset，但零点必须由机械参考锚定。两个还没做的工具：
  `tools/wiggle.py`（L1 差分定符号）与 `tools/zero_probe.py`（L2 用小力矩顶机械
  限位反推零点）。在只走"PC→下位机→CAN"的拓扑下，这两个是唯一能替代
  USB-CAN 直连标定的手段。
* 还没做 MuJoCo 后端（`backends/` 只做了 stub 与 host_link）。
  想在 N100 上做"半实物"回放的话，可以照 `stub_backend.py` 的接口再写一个。
* 手柄的轴号/键号默认值来自 XInput 布局，**必须现场用 `gamepad_probe.py` 确认**。
