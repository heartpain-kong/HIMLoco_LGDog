# V6_h —— 一个「按 100 Hz 训练」的策略方案（含移植说明）

> 这是 `legged_gym/logs/flat_LgDog1/V6_h` 那轮训练的最终策略，收编进 sim2real 是为了
> **在真机上试它**。它比当前主策略的前倾更小（pitch +4.52° vs +7.10°），
> **但它要求 100 Hz 控制，而 sim2real 默认是 50 Hz** —— 这是引入它唯一需要小心的地方。

| 项 | 值 |
|---|---|
| 权重指纹 | **`bd3c045be439`** |
| 来源 | `exported/policies/policy_V6_12400.pt` ← `V6_h/model_12400.pt`（iteration 12400） |
| 张量数 / 输入输出 | 14 / 270 → 12 |
| **控制契约** | **`decimation = 2` → 100 Hz，周期 10 ms，观测历史跨度 50 ms** |
| 当前主策略 | `a13b6ff17f79`（`models/policy.pt`），`decimation = 4` → 50 Hz，跨度 100 ms |

目录内容：

```
V6_h/
├── policy.pt                              策略本体（指纹 bd3c045be439）
├── policy.json                            身份登记：指纹 / 契约 / 验证证据 / 限制
├── config_snapshot.json                   这轮训练的训练配置（sim2sim --snapshot 用它自动对齐）
└── 验证录制_sim2sim_600s_bd3c045be439.npz  600 s @100Hz 的实测录制（23.8MB，可删）
```

---

## 1. ★★ 为什么不能"只换权重"：控制频率是策略契约的一部分

策略的观测是 **6 帧历史**，帧距就是控制周期。所以：

| | 主策略 | **V6_h** |
|---|---|---|
| `decimation` | 4 | **2** |
| 控制频率 | 50 Hz | **100 Hz** |
| 控制周期 | 20 ms | **10 ms** |
| 观测历史覆盖 | 100 ms | **50 ms** |
| 每周期推理预算 | 20 ms | **10 ms** |

只换权重不改频率 → 策略拿到的历史在时间上被拉伸成 2 倍。
**它不会报任何错**，表现是"能动，但动作节奏全乱、站不稳"。这就是 `--decimation` 这个开关存在的全部理由。

（`sim2sim.py` 自己也会查这件事：带 `--snapshot V6_h/config_snapshot.json` 跑时它会逐项对照，
直接打出 `❌ decimation: sim2sim=4 训练=2`。）

其余契约项 **与主策略完全一致**（所以只有频率这一处要动）：
`sim.dt=0.005`、`kp=30`、`kd=0.8`、`action_scale=0.25`、
`obs_scales = {lin_vel 2.0, ang_vel 0.25, dof_pos 1.0, dof_vel 0.05}`、
`clip=100`、默认站姿 `(0, -0.4, +0.5)`×4、`spawn_z=0.37`。

> 与当前 run(V20) 的配置共有 38 处差异，但**只有 `decimation` 影响部署**；
> 其余 37 处是 `rewards.*` / `terrain.*` / `domain_rand.*` / `commands.*` 这类训练侧参数。

---

## 2. 怎么跑（两条命令）

```bash
cd <sim2real>

# ① 无硬件干跑：验证 100Hz 下控制环跑不跑得动 + 策略能加载
.venv/bin/python main.py --dry-run --decimation 2 --policy V6_h/policy.pt

# ② 真机（第一次务必先 --check --live 与 --mode init）
.venv/bin/python main.py --decimation 2 --policy V6_h/policy.pt --mode init
.venv/bin/python main.py --decimation 2 --policy V6_h/policy.pt
```

**★ `--decimation 2` 现在不用手打了**：本目录 `policy.json` 的 `control_contract` 里写了
`decimation: 2`，`main.py` 启动时会**自动应用**并打印（"按策略自带契约设 decimation=2（控制周期 10.0 ms，
历史跨度 50.0 ms）"）。所以下面命令里的 `--decimation 2` 只是显式写法，去掉也一样。
控制频率被改动时仍会打印醒目提示，把"周期 20→10ms、历史跨度 100→50ms、预算减半"说清楚，
并把 `config.yaml` 的 `control_freq` 一并按 100 Hz 处理（否则 `--check` 会报不一致）。

**指纹核对**：身份登记表按【所加载策略旁边那份】找，所以 `V6_h/policy.json` 会与
`V6_h/policy.pt`（`bd3c045be439`）自动对上，不会因为和主策略不同而被拦。
（若你指向一个没有登记的 `.pt`，就会退回 `models/policy.json` 比对并**拒绝启动** —— 这是防呆。）

---

## 3. 它在 sim2sim 里的实测水平（本次补做的验证）

**此前 V6_h 从未出现在任何 sim2sim 录制里**，也就是它没被验过。我按你的流程补跑了一次：

```bash
cd <LgRobotDog>/legged_gym/legged_gym/scripts
python -u sim2sim.py \
  --snapshot ../../logs/flat_LgDog1/V6_h/config_snapshot.json \
  --policy   ../../logs/flat_LgDog1/exported/policies/policy_V6_12400.pt \
  --no-render --no-keyboard --gamepad off --clamp-vel --cmd-schedule "0:1.0"
```

```
[控制] 最终生效: decimation=2（控制频率 100Hz） action_scale=0.25 kp=30.0 kd=0.8 spawn_z=0.37
[性能] 无仿真: 仿真 600.0s / 墙钟 16.4s = 36.57 倍实时
```

`analyze_attitude.py` 的结果：

| 指标 | V6_h | 主策略 `a13b6ff17f79` | 判据 |
|---|---|---|---|
| 未翻倒 | ✅ 600 s | ✅ 600 s | — |
| **pitch（前高后低）** | **均值 +4.52°**（中位 +5.39°, p5 −0.03°, p95 +7.14°） | 均值 +7.10° | **≤ ±4°** |
| roll | +0.82° | +0.20° | — |
| 机身高度 z | 均值 0.336 m（中位 0.326） | 0.318 / 0.330 | ≥ 0.24 |
| 航向 | 路径 404 m / 直线 4.3 m，**yaw 累计 +307°** | 同量级偏航 | — |

**结论：pitch 明显更接近判据（+4.52° vs +7.10°），但仍未达标；两者都会打圈（yaw 累计几百度的漂移）。**
所以 V6_h 是"历史最好之一"，不是"已验证可用"。

---

## 4. 移植清单（要改什么、别踩什么）

**只是想在真机上试一次**（推荐）：

```bash
python main.py --decimation 2 --policy V6_h/policy.pt --mode init   # 先只起立
python main.py --decimation 2 --policy V6_h/policy.pt             # 再跑策略
```

**决定长期用 V6_h 当主策略**，再动这三处（否则每次都得带两个参数）：

1. 把它收编成主策略：`python tools/adopt_policy.py --from V6_h/policy.pt --yes`
   （会把当前主策略按指纹备份成 `models/policy_<指纹>.pt`）
2. 改默认频率：`spec.py` 里 `DECIMATION = 4` → **`2`**
   （`spec.py` 是常量的唯一来源；`--check` 会核对 `config.yaml` 的 `control_freq`）
3. `config.yaml` 的 `control_freq: 50` → `100`

**★ 上真机前必看的三件事**：

| 事项 | 为什么 |
|---|---|
| **先测 100 Hz 时序资格** | 推理预算从 20 ms 砍到 10 ms。`main.py` 起立后会做时序资格自检（满负载只写站姿），不达标会**拒绝放行策略**。N100 是 4 核小机器，10 ms 周期比 20 ms 难得多。缓解：`LGDOG_CPU=2` 钉核 / `OMP_NUM_THREADS=2` / `log.every` 调大 |
| **`hardware_map.yaml` 仍是 `verified: false`** | 12 个关节的总线/id/方向/减速比/零位还是猜测值，策略控制会被闸门拦住。**换策略不解决这个问题**，得先标定 |
| **板侧已无软限位** | 固件软限位已删除，行程保护只有上位机的关节限位夹取，而它**零点依赖**。零点标定比之前更关键（`PORTING.md` 第 6 节） |

**别做的事**：

- 别把 V6_h 的策略配 50 Hz 跑（动作节奏会乱），也别把主策略配 100 Hz 跑
- 别用 `cp` 手换 `models/policy.pt` —— 指纹闸门会拦住，用 `tools/adopt_policy.py`
- 别指望它做长距离直行（yaw 累计 +307°，会打圈）

---

## 5. 相关入口

| 想做什么 | 命令 |
|---|---|
| 看当前生效的控制频率与历史跨度 | `python main.py --check`（`[2/8]` 段会打印） |
| 确认某份策略的频率契约 | 读它旁边的 `policy.json` → `control_contract` |
| 在 sim2sim 里复现上面的验证 | 见第 3 节的命令（`--snapshot` 会自动把 decimation 对齐到 2） |
| 换回主策略 | 不带 `--decimation` / `--policy` 就是默认的 50 Hz 主策略 |
