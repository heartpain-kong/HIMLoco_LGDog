# models/ —— 随包带的策略（让部署不依赖训练仓库）

| 文件 | 说明 |
|---|---|
| `policy.pt` | **当前上栈**的策略，指纹 **`a13b6ff17f79`**（270 维输入 / 12 维输出） |
| `policy.json` | 身份登记：指纹 + 来源 + sim2sim 验证证据（`main.py` 启动时据此核对） |
| `policy_fdf3053d8b29.pt` | **上一版**（V19）按指纹留档，可随时切回 |

> 判断"机器人上跑的到底是哪一版"**只看权重指纹**。启动时会打印并与 `policy.json`
> 自动比对；不符会**拒绝启动**（可用 `--allow-policy-mismatch` 强行放行）。

---

## 当前策略：`a13b6ff17f79`

来源：`exported/policies/policy_auto_4000.pt`
—— 由 `tools/auto_sim2sim_check.sh` 从当前训练 run
`logs/flat_LgDog1/Sep26_20-58-09_LgDog1_himloco` 的 `model_4000.pt` 导出
（导出脚本 `tools/export_checkpoint.py`，含 obs 归一化统计量还原，所以张量数是 16 而不是 14）。

**sim2sim 验证证据**（`tools/analyze_attitude.py` 量化，录制 `sim2sim_0927_034104.npz`）：

```
✅ 全程未翻倒（30000 控制步 / 600 s，cmd = 前进 1.0 m/s）
pitch 前高后低: 均值 +7.10°  中位 +7.10°  p5 +5.53°  p95 +8.76°  std 1.00°
roll  侧倾    : 均值 +0.20°
机身高度 z    : 均值 0.318 m  中位 0.330 m
```

⚠️ **对照判据**：`auto_sim2sim_check.sh` 里写的是 **pitch ≤ ±4° 且 z ≥ 0.24**。
z 达标 ✓，**pitch +7.10° 还没达标**（对照基准：V16 行走 +12.81°/z=0.267、
V17@3600 +7.56°/z=0.210）。
也就是说这份是**"当前最优、仍在迭代"**的一份 —— 上真机时要知道它还没满足姿态判据：
能走、不摔，但机身是前高后低的。

## 换策略：用 `tools/adopt_policy.py`，别手敲 `cp`

```bash
python tools/adopt_policy.py --show         # 先看：现在捆的是哪份、sim2sim 侧最新是哪份
python tools/adopt_policy.py --yes          # 收编最新的 policy_auto_*.pt
python tools/adopt_policy.py --step 6000    # 指定某一步的 auto 策略
python tools/adopt_policy.py --from 别的.pt  # 指定任意策略文件
```

它做四件事（都是手敲 `cp` 容易漏的）：

1. 算**权重指纹**（唯一可信的身份；文件名、文件 md5 都不可信）；
2. 覆盖前把旧策略**按指纹命名**留一份（`policy_fdf3053d8b29.pt` 就是这么来的）；
3. 重写 `policy.json` 的指纹与来源 —— 不改的话 `main.py` 会**拒绝启动**（防呆设计）；
4. 顺手在 `recorded_io` 里找用这个指纹跑过的 sim2sim 录制，把 `analyze_attitude.py`
   的姿态数字抄进 `policy.json` 当验证证据；**找不到就警告"这份还没验过"**。

## 上一版：`fdf3053d8b29`（V19）

来源：`legged_gym/logs/flat_LgDog1/exported/policies/policy_V16_t_11601.pt`
（iteration 11601；**2026-09-25 时这个文件叫 `policy_V19_11601.pt`，后来被改名了** ——
尺寸 993706 B / mtime 09-25 13:17 / 指纹 `fdf3053d8b29` 三项都吻合，按指纹核对是同一份文件）。

原始训练 checkpoint：`logs/flat_LgDog1/V16_t/model_11601.pt`
（核对过：导出里的 14 个张量**全部**能在该 checkpoint 的 `model_state_dict` 里
按「张量字节哈希 + 形状」找到，多出的 16 个是 critic / 训练专用参数。）

> ⚠️ 注意：训练仓库里 `policy_V19_11601.pt` 这个名字**已经不存在了**。
> 以名字找文件会找不到，**一律按 `weight_fingerprint` 找**。

## ⚠️ 为什么不能直接用训练仓库里的 `exported/policies/policy.pt`

实测：2026-09-25 13:17 它还是 V19（`fdf3053d8b29`），**20:05 就被正在跑的 V20 训练轮覆盖**；
09-26 18:50 又变成 `92a645131d55` —— 而那一份在 sim2sim 里刚给走路指令 **1.8 秒就倒了**
（对照 `policy_auto_4000` 跑满 600 s 没摔）。
忘带 `--policy` 就可能拿一个没验过的权重上真机 —— 所以随包带一份 `models/policy.pt`，
并让启动流程核对指纹。

## 为什么不用文件 md5

`torch.jit` 每次序列化都会改 zip 里的 `policy/.data/serialization_id` 和自动生成的
`___torch_mangle_N` 源码条目 —— **同一份权重重新导出一次，文件 md5 就变了**。
（`legged_gym/scripts/sim2sim.py` 里踩过：policy.pt 与 policy_V13_6000.pt 文件 md5 不同，
权重指纹都是 `e4fee6f97eaf`。）
