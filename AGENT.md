# LgRobotDog 调参工作流（AGENT 说明）

> 本文档指导 **AI 训练调参 agent** 使用 LgRobotDog 的三层参数调整记录体系，
> 实现 OpenDoge_train 那种"每次修改可追溯、防重复踩坑"的纪律。
> 参考：OpenDoge_train 的 tuning_log.json + CLAUDE.md 方法论。

---

## 一、三层记录体系（先理解）

```
┌──────────────────────────────────────────────────────────┐
│ 第一层：配置快照（train.py 自动）                          │
│   每次训练启动 → env_cfg + train_cfg + URDF 指纹 →         │
│   logs/<exp>/<run>/config_snapshot.json                   │
├──────────────────────────────────────────────────────────┤
│ 第二层：调参日志（tools/tuning_log.py）                    │
│   run / add（含 --diff 自动对比快照）/ update / show / diff │
├──────────────────────────────────────────────────────────┤
│ 第三层：指标报告（tools/report.py + parse_tb.py）          │
│   配置改动 vs 指标 总表 → 哪个改动导致什么结果              │
└──────────────────────────────────────────────────────────┘
```

**快照里有什么（自动 diff 能看到的范围）**：

| 层 | 快照字段 | diff 键前缀 | 例子 |
|---|---|---|---|
| 环境/奖励/地形 | `env_cfg` | 无（裸点路径） | `terrain.mesh_type`、`rewards.scales.feet_air_time` |
| **训练超参** | `train_cfg` | `train.` | `train.algorithm.learning_rate`、`train.runner.max_iterations` |
| **URDF 几何** | `urdf_fingerprint` | `urdf.` | `urdf.joint_limits.FL_thigh_joint`、`urdf.collision_geometry.FL_calf` |
| **源码** | `source_fingerprint` | `src.` | `src.files.legged_gym/…/legged_robot.py`（44 个 .py 的逐文件 md5 + 聚合 md5） |

> ⚠️ 改这四层里的**任何一层**都要进 tuning_log——训练超参、URDF 限位、奖励函数源码
> 过去都是盲区（只 diff `env_cfg`，改了 lr/epoch、关节限位或代码里的硬编码门限，
> 自动 diff 会说"配置无改动"）。

**你的职责**：在"改配置"这个动作发生时，把**为什么改、改了什么、预期什么**记录进第二层；
训练后**回填结果**；用第三层总表判断每个改动的影响。

> ★ **铁律：每一个训练 run 都必须在 tuning_log.json 里有且仅有一条记录。**
> 改了配置的要记，**没改配置的纯跑一轮也要记**。
> 训练启动前用 `run` 建记录，训练结束后用 `update` 回填。
> `show` 会自动列出「最后一条记录之后新产生但没记录」的 run，看到就补。

### 版本命名（V 方案）

**只有"训练出来的正式版本"才进 V 序列**；探索期、失败轮、中途掐掉的产物不编号。

| 版本 | 含义 | run | 导出归档 |
|---|---|---|---|
| V1 | 首版：PD 40/1.0 + 域随机化全面加严（连杆质量随机化、力矩 ±20%、质心 ±0.2） | `V1` | `policy_V1_3000.pt` |
| V2 | 放松惩罚 + 降 tracking 权重（原编号 V11）：tracking 曾达 0.79，是唯一崩解后自愈的轮次 | `V2` | `policy_V2_4200.pt` / `policy_V2_7800.pt` |
| V3 | base_height 那条腿治蹲姿（原编号 V14）：全部轮次里最稳、tracking 最好（0.7069） | `V3` | `policy_V3_8800.pt` |
| V4 | 姿态锚把零指令站高瞄到 0.32（原编号 V17）：站立≈行走基本达成（差 2mm） | `V4` | `policy_V4_3200.pt` |
| V5 | 提高抬腿高度（clearance −0.18→−0.13）：抬腿饱和在 ≈0.14m，达不到 0.17 目标 | `V5` | `policy_V5_5200.pt` |
| V6_h | 8 项改动（episode 60s、指令范围 ±1.6、max_curriculum 3.0 等）：本序列最强，tracking 峰值 0.8219 | `V6_h` | `policy_V6_12400.pt` |
| V7 | 用户手改 9 项后的长跑（3636/12000 被提前停）：tracking 峰值 0.7373，未推进指令课程 | `V7` | `policy_V7_3600.pt` |
| V8 | **训练地形新增第 6 类【高台】**（唯一变量，2038/12000 被提前停并导出）：崩解 0 处、tracking 与 V7 持平，但**没学会爬 20cm 高台** | `V8` | `policy_V8_2000.pt` |
| V9 | **过障碍能力四项齐改**（抬腿奖励换摆动相门控+净空 0.27m；高台→横贯全宽障碍带；新增卡住惩罚；课程改按越障晋升）：进步=terrain_level 2.55→5.16、tracking 峰值 0.7832（全序列最高）；代价=**步态崩坏**（抖动中位 80.9、末值 230、2012 段崩解，首个在第 3049 轮）。⚠️**出口存档 8400 已坏**（平地原地打转、撞障碍带会塌）；哨兵建议的回滚点 `policy_V9_2400.pt` 平地直行正常但也不会爬。结论: 四项改动【未换来过障碍能力】，且越障课程可能把难度推得太快（等级 5.16 ≫ V8 的 2.93） | `V9` | `policy_V9_8400.pt`（坏）+ `policy_V9_2400.pt`（可用） |
| V10 | **V9 结算后修三项**（★修 climb_max 量错对象的 bug + 抬腿目标 0.27→0.20m + climb_frac 1.0 且越障判据只在障碍列生效）：★成果=**首次真的爬上 20cm 台面**（z 峰值 0.582、83 步 >0.45，但 2 秒后滑落）、**抬腿首次真的变高**（p90 0.0957→**0.1375**、max 0.1749→**0.2099**、≥0.12m 从 0.1%→**27%**）、tracking 峰值 **0.8239**（全序列最高、首次越过 0.80 线）、课程不再白送等级（2.51→1.75）；⚠️ 问题=**2775 轮后又崩**（抖动 6→290，255 段）—— 与 V9 崩在同一区间，源头未定位；用户导出崩坏前的 2600 | `V10` | `policy_V10_2600.pt` |
| V11 | **专治 10cm 楼梯被绊倒**（②地形等级下限 3、③抬腿权重 -1.0→-2.0、④指令保持 3→7s、⑤新增『四足都踏上被抬高地面』奖励）：②✅等级下限生效（terrain_level 1.8→3.5，10cm 台阶终于进了训练分布）、③部分✅抬腿分布上移（中位 0.095→0.112、首次有摆动到 0.22m）、⑤✅机制首次非零；❌**但 10cm 楼梯仍上不去（比 V10 更差: 只到第 1~2 级）且训练更早崩**（首个持续崩解在第 **1298** 轮，V9 3049/V10 2775 ⇒ 奖励压力越大崩得越早）| `V11` | `policy_V11_1200.pt`(最后健康点) / `_1600` / `_4400`(已崩) |
| V12 | **单变量归因实验**（= V11 去掉抬腿权重翻倍，foot_clearance 回 -1.0；1135 轮时结束、未导出）：证明**权重 -2.0 才是 V11 提前崩坏的原因**（崩坏点 V9 3050 / V10 2776 / V11 1299 / V12 到 1135 轮仍未崩、抖动峰值 58、tracking 0.791）⇒ 抬腿权重安全上限 = -1.0 | `V12` | （未导出） |
| V13 | **两类高度目标全部关闭**（foot_clearance 整项=0、base_height 整项=0；stand_still/climb_up/feet_stumble/feet_air_time 保留）：**1200 轮 0 崩解**（抖动中位 8.2）；★零指令站高 **0.3656** 自然落在几何站高上（V8 折中值 0.3233 的问题消失）、行走 0.3100；抬腿 ≥0.12m 占比 **39%**（V8 0.1%）—— 证明高度目标确实可以不要。⚠️ 遗留=航向保持差（平地弧线行走、楼梯测试被带偏） | `V13` | `policy_V13_1200.pt` |
| V14_h | **V13 配置 + 姿态/角速度惩罚的地形对齐算法**（orientation 罚机身 vs 脚下地形法向、ang_vel_xy 按地形平坦度缩放；climb_up 2.0、stuck -3.0）：训练稳定窗口创纪录（前 6000 轮抖动 7~8、0 崩解，**6038 轮起崩**）；但 10cm 楼梯只到第 1~2 级、20cm 测试向后翻倒、抬腿 ≥0.12m 仅 0.06%（退步）；平地弧线漂移 | `V14_h` | `policy_V14_6600.pt`（坏）+ `policy_V14_6000.pt`（候选） |
| V15 | **V14 基础上回调激励**（climb_up 2.0→0.5、stuck -3.0→-2.0）+ **用户手改 3 处**（action_scale 0.25→0.5、tracking 2.4/1.2）：全程 **0 崩解**（抖动中位 6.6）；★抬腿全序列最好（中位 0.084/p90 **0.166**/max 0.194，≥0.12m 占比 **37%**）；20cm 立面**稳定顶住 600s 不塌**；❌ 10cm 楼梯仍卡在第 2 级翻倒、平地仍弧线漂移 | `V15` | `policy_V15_2600.pt` |

> ⚠️ **编号在 2026-09-20 被重置过**：当时只保留 4 个 run 并重新编号为 V1~V4，
> 其余 run 目录已删除（配置注释里出现的 V5~V17 都是**旧编号**，现已不对应任何现存 run）。
> 旧→新映射：**旧V11→V2、旧V14→V3、旧V17→V4**（V1 不变）；旧 V2/V3/V5/V12/V13/V15/V16 等
> 对应的 run 与策略归档已移出（策略在 `exported/policies/_obsolete/`）。
> 已删除 run 的 `config_snapshot.json` 归档在 `legged_gym/logs/_archive/<run>/`，
> 所以 `tuning_log.py diff` 与历史记录的 `changes` 仍可追溯。

规则：
1. 每条记录的 `changes` 里带一个 `"版本"` 字段（如 `"V2（= V1 + 3 项奖励微调）"`），
   `diagnosis` 以 `【V2】` 开头 → `show` 里一眼定位版本。
2. 导出归档名 `policy_V<n>_<step>.pt`；`policy.pt` 始终是**最新那次导出**（sim2sim/play 读它）。
3. V1 之前的轮次保留 `policy_round<n>_<step>.pt` 旧名（不在 V 序列内），**不要混用两套编号**。
4. 回填 `actual_effect` 时，用**上一版**做跨轮对照锚点（如 V2 对 V1），不要跨版本乱比。


---

## 二、工作流（每轮训练循环）

### Step 1 训练前
```bash
# 看历史记录，避免重复已证明无效的修改（必须先看！）
python tools/tuning_log.py show

# 看最近两次配置差异（可选）
python tools/tuning_log.py diff --to <最近run>
```

### Step 2 训练中（每 200~300 轮看一次指标，第一次退化为正常现象）
```bash
# 方式 A：TensorBoard 网页
tensorboard --logdir=legged_gym/logs

# 方式 B：命令行解析（零依赖）
python tools/parse_tb.py legged_gym/logs/<exp>/<最新run>/    # 看关键指标曲线

# 方式 C：全 run 总表
python tools/report.py
```

### Step 3 诊断
对照已知失败模式（OpenDoge/HIMLoco 经验）：
| 指标组合 | 诊断 |
|---|---|
| `mean_episode_length` 短且下降 | 站不起来/频繁倒 → 先查出生环境，别加惩罚 |
| `rew_ang_vel_xy` 很负（<-3） | 机身摇晃 → 检查是否奖励权重过载/PD |
| `rew_lin_vel_z` 很负（<-3） | 弹跳 → 检查 feet_air 与 lin_vel_z 冲突 |
| `rew_base_height` 持续负 | 蹲着走/站高不对 → 检查 base_height_target |
| `value_function` 峰值巨大 | 奖励尺度爆炸 → 权重回退 |
| `mean_noise_std` 后期回升 | 探索过大/后期退化 → 降 entropy 或提前停止 |

### Step 4 改配置
- **一次只改一个变量**（或强相关一组），否则无法归因
- 结构性改动（网络/PD 基线/奖励结构）从零重训；权重微调可从 checkpoint 续
- **改完先跑通再记录，记录在训练启动前完成**

### Step 5 记录（每个 run 必做，训练启动前完成）
```bash
# ★ 首选：为本轮 run 建记录 —— 自动定位快照、自动 diff 出改了哪些配置项，
#   没改配置也会记一条（不会漏），同一个 run 重复记录会被拒绝
python tools/tuning_log.py run \
    --run_dir <本次run> \
    --diagnosis "小腿胶囊修复后首轮" \
    --expected "rew_collision 非零且 episode 不降"

# 改配置时手动写清（值里可以安全带逗号）
python tools/tuning_log.py add \
    --diagnosis "出生就倒，回退 V1 权重" \
    --changes "rewards.scales.lin_vel_z: -5.2 -> -2.0, rewards.scales.orientation: -5.2 -> -2.4" \
    --expected "episode 回到 1000" \
    --run_dir <本次run>

# 或自动 diff 最近两个 run 的配置快照（改了什么自动生成）
python tools/tuning_log.py add --diff \
    --diagnosis "V2 惩罚过载导致崩" \
    --expected "能站稳"
```

> ⚠️ `--changes` 的分隔符：`;` 永远分隔；`,` 只有**后面紧跟 `key:`** 时才算分隔符。
> 逗号/冒号有歧义时用 `--changes-json '{"k": "旧 -> 新"}'`。
> （早期版本按任意逗号切分，值里带逗号会生成 `param2` 之类的垃圾键。）

### Step 6 训练后回填结果（必须！）
```bash
# 解析本次 run 的关键指标
python tools/parse_tb.py legged_gym/logs/<exp>/<run>/

# 回填实际效果（index 用 show 查，-1 = 最后一条）
python tools/tuning_log.py update --index <N> --effect "FAILED: ep=17 仍倒，不是权重问题"
# 规范：SUCCESS / PARTIAL / FAILED + 关键指标值
```

### Step 7 生成报告（可选，定期）
```bash
python tools/report.py            # 全部 run：配置改动 vs 指标 总表
python tools/report.py --exp rough_a1 --tags mean_episode_length,rew_ang_vel_xy
```

---

## 三、记录规范（写 tuning_log 时）

1. `diagnosis`：一句话说清**当前问题**（含证据，如 "rew_ang_vel_xy=-7 摇晃"）
2. `changes`：**旧值 -> 新值**，键用点路径，前缀按上表分三层：
   - `env_cfg`：裸路径（`rewards.scales.lin_vel_z`、`terrain.mesh_type`）
   - `train_cfg`：加 `train.`（`train.algorithm.learning_rate`）
   - URDF：加 `urdf.`（`urdf.joint_limits.FL_thigh_joint`、`urdf.*_calf`）
3. `expected_effect`：可验证的预期（带数值："episode > 800"、"rew_base_height > -0.03"）
4. `actual_effect`：必须回填，格式 `SUCCESS/PARTIAL/FAILED: 指标证据`
5. `run_dir`：填本次训练 run **目录名**（如 `Sep13_04-26-04_LgDog1_himloco`），
   **不要填路径**——`show` 靠它反查实验，填路径会让漏记检测失效

---

## 四、禁止事项

- ❌ **训练 run 不记录**（每个 run 都必须有一条记录，纯跑也要记）
- ❌ **改配置不记录**（体系失效的根源）
- ❌ 不 `show` 就直接重复尝试已 FAILED 的修改
- ❌ 一次改多个无关变量（无法归因）
- ❌ 出生就倒时继续加大惩罚权重（先查环境：出生高度/地形/push/disturbance/初始随机）
- ❌ 执行任何 git 操作（本项目不需版本控制介入）

---

## 五、工具速查

| 命令 | 用途 |
|---|---|
| `python tools/tuning_log.py show` | 查看调参历史（防重复）+ 漏记 run 提醒 |
| `python tools/tuning_log.py run --run_dir <run>` | ★每个 run 建记录（自动 diff 出改动） |
| `python tools/tuning_log.py add --diagnosis "..." --changes "..." --expected "..."` | 记录一次修改 |
| `python tools/tuning_log.py add --diff --diagnosis "..." --expected "..."` | 自动 diff 快照记录 |
| `python tools/tuning_log.py update --index N --effect "..."` | 回填结果（N 可为 -1） |
| `python tools/tuning_log.py diff --to <run>` | 对比某 run 与前一个的配置 |
| `python tools/parse_tb.py <run目录>` | 解析某 run 的指标 |
| `python tools/report.py` | 全部 run 配置 vs 指标总表 |

---

## 六、典型排查顺序（站不起来时）

1. `python tools/report.py` → 看哪个 run 配置改动后 episode 骤降
2. `python tools/tuning_log.py show` → 历史里是否已试过类似修改
3. `python tools/parse_tb.py <该run>` → 确认指标（ang_vel_xy/lin_vel_z/value_function）
4. 排查环境：出生高度、地形（高台/断桥出生即倒?）、push/disturbance/initial_joint 随机
5. 回退奖励权重到可学的基线（如 V1），验证能站
6. 记录 → 回填 → 再迭代
