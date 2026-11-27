# 许可范围说明 / Scope of the License

本仓库的 `LICENSE`（MIT）覆盖**本项目自行编写**的部分：

- LgDog1 的适配与配置（`legged_gym/legged_gym/envs/LgDog1/`）
- 实机部署栈（`deploy/sim2real/`）
- sim2sim 的增强与新增脚本（`legged_gym/legged_gym/scripts/sim2sim.py`、`tools/eval_stairs.py`、`tools/render_stairs.py`）
- 调参体系（`tools/tuning_log.py` 等）与全部文档、演示素材

以下文件**继承自上游**，保留各自原始许可，不被 MIT 覆盖：

| 来源 | 涉及内容 | 许可 |
|---|---|---|
| [OpenRobotLab/HIMLoco](https://github.com/OpenRobotLab/HIMLoco) | 整体框架、`legged_gym`/`rsl_rl` 的改造、HIM 估计器与 HIM-PPO | CC BY-NC-SA 4.0（原文：`docs/LICENSE_HIMLoco_CC-BY-NC-SA.txt`） |
| [OpenDogeRobotics/OpenDoge_train](https://github.com/OpenDogeRobotics/OpenDoge_train) | 参考移植的奖励函数（`default_pos_linear`、`diagonal_sync`、`hip_mirror_symmetry`，以及 `base_height` / `feet_air_time` / `stand_still` 的表单） | Apache-2.0 |
| [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) | 训练环境骨架 | BSD-3-Clause（`legged_gym/LICENSE`） |
| [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) | PPO 实现骨架 | BSD-3-Clause（`rsl_rl/LICENSE`） |

实操含义：

- 用 `deploy/`、`tools/` 和文档做自己的项目 —— MIT，随便用。
- 用整套训练代码做**商业产品** —— HIMLoco 那部分的非商用条款仍然约束着你，需要另行获得 OpenRobotLab 的授权。
- 引用 OpenDoge 的奖励实现时请保留 Apache-2.0 的署名与许可声明。
