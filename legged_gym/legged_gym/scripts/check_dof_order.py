#!/usr/bin/env python3
"""check_dof_order.py — 打印 Isaac Gym 环境的真实 dof 顺序与 default_dof_pos

用途：一锤定音验证 sim2sim 的 REMAP 假设。
运行（在 legged_gym/legged_gym/scripts 下，conda 环境 himloco）：
  python check_dof_order.py --num_envs 4
输出：
  Isaac dof 顺序: [...]
  default_dof_pos: [...]
然后与 sim2sim 打印的 "MuJoCo 关节顺序" 对比：
  - Isaac 顺序 == [FR,FL,RR,RL] 且 MuJoCo == [FL,FR,RL,RR] → REMAP=[3,4,5,0,1,2,9,10,11,6,7,8] 正确（当前值）
  - Isaac 顺序 == [FL,FR,RL,RR]（与 MuJoCo 相同）      → REMAP 应为恒等 [0..11]（当前是错的！）
  - 其它 → 按两边打印结果重排 REMAP

注意：Isaac Gym 要求 isaacgym 模块先于 torch 导入，
      所以这里 import legged_gym 必须在 import torch 之前。
"""

from legged_gym.envs import *          # noqa: F401  注册任务（内部加载 isaacgym，必须先于 torch）
from legged_gym.utils import get_args, task_registry
import torch


def main():
    args = get_args()
    args.headless = True                # 不弹窗口
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    print("\n========== Isaac Gym 环境信息 ==========")
    print(f"task: {args.task}")
    print(f"num_envs: {env.num_envs}")
    print(f"num_dofs: {env.num_dofs}")
    print("Isaac dof 顺序:", list(env.dof_names))
    print("default_dof_pos (Isaac 顺序):", env.default_dof_pos.cpu().numpy().round(3))
    print("========== MuJoCo 侧（从 sim2sim 打印获取）==========")
    print("MuJoCo 顺序:   [FL_hip,FL_thigh,FL_calf, FR_hip,FR_thigh,FR_calf, "
          "RL_hip,RL_thigh,RL_calf, RR_hip,RR_thigh,RR_calf]")
    print("\n判断：")
    print("  若 Isaac == [FR,FL,RR,RL] → 当前 REMAP 正确")
    print("  若 Isaac == [FL,FR,RL,RR] → REMAP 应为恒等 [0,1,...,11]，当前交换映射是错的！")

    # 顺便打印 default_dof_pos 对应关系（确认符号）
    print("\ndefault_dof_pos 与 config default_joint_angles 对比：")
    print("  config: FL_hip=0.1 RL_hip=0.1 FR_hip=-0.1 RR_hip=-0.1 / thigh FL=0.8 RL=1.0 FR=0.8 RR=1.0 / calf 全 -1.5")


if __name__ == '__main__':
    main()
