# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go1RoughCfg( LeggedRobotCfg ): 
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }                          

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.0}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0

    class commands( LeggedRobotCfg.commands ):
            curriculum = True
            max_curriculum = 2.0
            num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            resampling_time = 10. # time before command are changed[s]
            heading_command = False # 3-command mode uses yaw velocity directly
            class ranges( LeggedRobotCfg.commands.ranges):
                lin_vel_x = [-1., 1.] # min max [m/s]
                lin_vel_y = [-1., 1.]   # min max [m/s]
                ang_vel_yaw = [-3.5, 3.5]    # min max [rad/s]
                heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # 非超时摔倒时的终止奖励/惩罚；0 表示关闭。
            termination = -0.0
            # 机身平面线速度跟踪指令（x/y）。
            tracking_lin_vel = 2.0
            # yaw 角速度跟踪指令。
            tracking_ang_vel = 0.8
            # 惩罚竖直速度，抑制上下弹跳。
            lin_vel_z = -2.0
            # 惩罚横滚/俯仰角速度，抑制机身摇晃。
            ang_vel_xy = -0.2
            # 惩罚机身偏离水平姿态。
            orientation = -0.2
            # 惩罚关节加速度，减少冲击和抖动。
            dof_acc = -2.5e-7
            # 惩罚关节功率，降低能耗和电机负载。
            joint_power = -2e-5
            # 惩罚机身高度偏离 base_height_target。
            base_height = -1.0
            # 惩罚足端运动时偏离 clearance_height_target。
            foot_clearance = -0.01
            # 惩罚相邻动作突变。
            action_rate = -0.1
            # 惩罚二阶动作变化，增强动作平滑性。
            smoothness = -0.02
            # 鼓励运动指令下形成更长的腾空步态。
            feet_air_time = 0.5
            # 惩罚指定身体碰撞；0 表示关闭。
            collision = -0.4
            # 惩罚足端侧向撞击/绊倒；0 表示关闭。
            feet_stumble = -0.5
            # 零指令时惩罚偏离默认站姿；0 表示关闭。
            stand_still = -0.
            # 惩罚关节力矩平方；0 表示关闭。
            torques = -0.0
            # 惩罚关节速度平方；0 表示关闭。
            dof_vel = -0.0
            # 惩罚接近关节位置软限位；0 表示关闭。
            dof_pos_limits = -0.0
            # 惩罚接近关节速度软限位；0 表示关闭。
            dof_vel_limits = -0.0
            # 惩罚接近关节力矩软限位；0 表示关闭。
            torque_limits = -0.0

        # 是否把总负奖励裁剪为非负；False 保留完整惩罚信号。
        only_positive_rewards = False
        # 速度跟踪核宽度，越小越强调精确跟踪。
        tracking_sigma = 0.25
        # 关节位置软限位比例，超过后开始惩罚。
        soft_dof_pos_limit = 1.
        # 关节速度软限位比例，超过后开始惩罚。
        soft_dof_vel_limit = 1.
        # 关节力矩软限位比例，超过后开始惩罚。
        soft_torque_limit = 1.
        # 机身高度目标（米）。
        base_height_target = 0.32  
        # 足端接触力超过该阈值（牛）后开始惩罚。
        max_contact_force = 100.
        # 足端净空目标高度（米）。
        clearance_height_target = -0.16

class Go1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        # Reduce PPO update peak memory while retaining 3072 parallel envs.
        num_mini_batches = 16
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go1'

  
