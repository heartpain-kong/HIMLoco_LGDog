# HIMLoco 常用命令速查

## 环境

```bash
conda activate himloco
cd ~/rl/LgRobotDog/legged_gym/legged_gym/scripts
python train.py --task go1 
python train.py --task LgDog1 --headless --num_envs 3072
python sim2sim.py
python play.py --task go1 --resume --load_run Sep09_10-57-46_round20_after_v6_mb16
python play.py --task LgDog1 --resume --load_run V50 --checkpoint 7200  
```

---

## 训练

### 标准训练

```bash
python train.py --task a1
python train.py --task a1 --num_envs 256                    # 8G 显存
python train.py --task a1 --experiment_name rough_a1_v2     # 新实验（不覆盖旧日志）
python train.py --task a1 --headless

python train.py --task LgDog1 --headless --num_envs 3072  2048 1024 
```

### 无渲染训练（服务器/后台）

```bash
# 默认已 headless，训练时不弹窗口
python train.py --task a1 --num_envs 256

# 后台运行 + 日志
nohup python train.py --task a1 --num_envs 256 > train.log 2>&1 &

# 查看进度
tail -f train.log
# 或另开终端
python extract_tb.py --watch
```

### 断点续训

```bash
python train.py --task a1 --resume                                    # 自动最新
python train.py --task a1 --resume --checkpoint 3000                  # 指定迭代
python train.py --task a1 --resume --load_run Aug13_03-36-17_ --checkpoint 400          # 指定 run
```

---

## Play（Isaac Gym 测试）

```bash
# 最新模型
python play.py --task a1 --resume

# 指定 checkpoint
python play.py --task a1 --resume --checkpoint 3000
python play.py --task a1 --resume --load_run  Aug19_13-22-27_ --checkpoint 2700
```

---

## Sim2Sim（MuJoCo 验证）

**一条命令，全局键盘控制，不挑窗口：**

```bash
python sim2sim.py
```

| 键 | 功能 |
|---|---|
| 8/W | 前进 | 2/S | 后退 |
| 4/A | 左转 | 6/D | 右转 |
| 7/Q | 左移 | 9/E | 右移 |
| Space/R | 复位 | Esc | 退出 |

其他用法：

```bash
python sim2sim.py --x_vel 0.3                   # 固定速度
python sim2sim.py --no-keyboard                 # 禁用键盘（只看）
python sim2sim.py --policy /path/to/policy.pt   # 指定策略
```

> 依赖：`conda activate himloco && pip install pynput`（已装）
> 场景文件：`resources/robots/a1/xml/scene.xml`（含平地+斜坡+台阶）

---

## 训练监控

### TensorBoard（网页端）

```bash

conda activate himloco
tensorboard --logdir=~/rl/LgRobotDog/legged_gym/logs/rough_go1 --port 6006
conda activate himloco
tensorboard --logdir=~/rl/LgRobotDog/legged_gym/logs/flat_LgDog1 --port 6006

# 浏览器: http://localhost:6006
```

### 命令行文本监控

```bash
conda activate himloco
cd ~/rl/HIMLoco/legged_gym/legged_gym/scripts
python extract_tb.py --watch
# 提取最新 run 数据到 training_metrics.md
python extract_tb.py

# 边训练边看（每60秒刷新）
python extract_tb.py --watch

```bash
# 最新模型
python play.py --task a1 --resume

# 指定 checkpoint
# 所有历史 run
python extract_tb.py --all

# 指定 run
python extract_tb.py --run Jul30_21-53-16_
```

输出 `training_metrics.md` 包含：摘要表（最低/最高/趋势/状态）+ 逐步骤数据 + 异常检测。

---


## mujoco

```bash
cd /home/heartpain/rl/LgRobotDog/legged_gym/legged_gym/scripts
~/miniconda3/envs/himloco/bin/python view_model.py ../../resources/robots/LgDog1/xml/LgDog1.xml
# 蹲一半（大腿 +0.25、小腿 -0.35，自动按 URDF 限位裁剪）
python view_model.py <xml> --crouch 0.5

# 手写 12 个目标角（顺序 = data.ctrl 顺序，即 FR,FL,RR,RL × hip,thigh,calf）
python view_model.py <xml> --target 0,0.3,-0.6,0,0.3,-0.6,0,0.3,-0.6,0,0.3,-0.6

# 不要 PD，纯自由落体（看纯物理/惯性）
python view_model.py <xml> --no-hold

# 换增益（KP 太大时会抖，看是否稳定）
python view_model.py <xml> --kp 60 --kd 2
```


## 关键训练参数（最新值）

| 参数 | 值 | 位置 | 说明 |
|---|---|---|---|
| `learning_rate` | 1e-3 | `legged_robot_config.py` algorithm | PPO 学习率 |
| `desired_kl` | 0.03 | 同上 | 放宽 KL 防 LR 归零 |
| `num_learning_epochs` | 3 | 同上 | 减半防过拟合 |
| `estimator_learning_rate` | 3e-4 | 同上 | 估计器独立 LR |
| `base_height_target` | 0.30 | `a1_config.py` rewards | A1 自然行走高度 |
| `base_height` | -2.0 | `a1_config.py` rewards.scales | 高度惩罚 |
| `tracking_sigma` | 0.25 | `a1_config.py` rewards | 跟踪高斯核 |
| `estimator_temperature` | 5.0 | `legged_robot_config.py` policy | Swap loss 温度 |
| `estimator_max_grad_norm` | 5.0 | 同上 | 估计器梯度裁剪 |

---

## sim2sim 关键参数

| 参数 | 值 | 说明 |
|---|---|---|
| KP | 55.0 | PD 刚度（MuJoCo 需要 >50） |
| KD | 1.3 | PD 阻尼 |
| 髋力矩 | ±20 Nm | |
| 大腿/小腿力矩 | ±55 Nm | |
| 默认速度 | 0.3 m/s | 按住走/松手停 |

---

## 可用任务

| `--task` | 机器人 |
|----------|--------|
| `a1` (默认) | Unitree A1 |
| `go1` | Unitree Go1 |
| `aliengo` | Aliengo |
| `anymal_b` | ANYmal B |
| `anymal_c` | ANYmal C |
| `cassie` | Cassie |

---

## 显存参考

| GPU | `--num_envs` |
|-----|-------------|
| 8 GB | 256 |
| 12 GB | 512~1024 |
| 16 GB | 1024~2048 |
| 24 GB | 4096 |

---

## 常见问题

| 问题 | 解决 |
|------|------|
| 训练时 estimation_loss 反升 | 检查 `estimator_learning_rate` 是否太高 |
| swap_loss 剧烈波动 | 提高 `estimator_temperature`（5.0→8.0） |
| mean_episode_length 突然暴跌 | LR 归零了，提高 `desired_kl` |
| rew_base_height 一直负 | `base_height_target` 太高（A1 正常 0.30） |
| MuJoCo 里狗倒了 | 按键复位，或降低 KP/KD |
| 松手狗还走 | 确认 `python sim2sim.py`（已集成键盘，不需要桥接） |

---

## 奖励函数 ↔ TensorBoard 指标对应

### tracking_lin_vel=3.0, sigma=0.2 → rew_tracking_lin_vel

`rew = 3.0 × exp(-vel_error² / 0.2)`

| rew | 偏差 | 含义 |
|-----|------|------|
| 3.0 | 0 m/s | 完美 |
| 1.5 | 0.28 m/s | 有点跟不上 |
| 0.7 | 0.53 m/s | 基本没走 |

### base_height=-2.0, target=0.30 → rew_base_height

`rew = -2.0 × (h - 0.30)²`

| rew | 高度 | 含义 |
|-----|------|------|
| 0 | 0.30m | 完美 |
| -0.01 | 0.23~0.37m | 正常 |
| -0.06 | 0.13m | 严重蹲着走 |

> 蹲着更稳 → tracking 分更高 → 策略算账：蹲 = 赚。越训越蹲。

### ang_vel_xy=-0.5 → rew_ang_vel_xy

`rew = -0.5 × (ω²)`

| rew | ω | 含义 |
|-----|---|------|
| 0~-0.2 | <0.6 | 稳 |
| -2.0 | 2 | 晃 |
| -8.0 | 4 | 狂摇 |

> **最早死前信号**。→0 正常，负向加速 → 马上崩。

### only_positive_rewards=True 的影响

`总奖励 = max(0, 正+负)`。True 会把惩罚砍到 0 → 策略不知道在蹲/晃。**必须 False。**

### estimation_loss

| est | 含义 |
|-----|------|
| <0.05 | 健康 |
| 0.1~0.3 | 正常 |
| >0.5 | 速度预测出错 |
| >1.0 | 快崩 |

### 死亡时间线（V15）

```
正常 → base_h先恶化 → 50步后ang恶化 → 100步后est恶化 → 全崩
```

### 预警速查

| 指标 | 正常 | 警告 | 崩 |
|------|------|------|-----|
| rew_tracking | >1.5 | <1.0 | <0.7 |
| rew_base_height | >-0.02 | <-0.03且恶化 | <-0.06 |
| rew_ang_vel_xy | >-0.5 | <-2.0且加速 | <-8.0 |
| est | <0.1 | >0.3且上升 | >0.5 |
| mean_ep | >800 | <400 | <100 |
