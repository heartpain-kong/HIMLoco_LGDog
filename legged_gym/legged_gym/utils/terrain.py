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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    # 地形类型顺序 —— 前 5 类与历史行为逐位一致（go1/a1/aliengo 继承基类的 5 条
    # terrain_proportions 只用到前 5 类，故行为不变）；【platforms 高台】是 2026-09-21
    # 新增的第 6 类，插在 discrete 之后、stepping_stones 之前；【flat 纯平地】是
    # 2026-09-25 新增的第 7 类，插在 platforms 之后（同样不影响前面所有索引）。
    TERRAIN_TYPES = ['slope',          # 0 金字塔斜坡（含正/负两半）
                     'slope_rough',    # 1 斜坡 + 随机起伏
                     'stairs_down',    # 2 金字塔楼梯（向下）
                     'stairs_up',      # 3 金字塔楼梯（向上）
                     'discrete',       # 4 随机方块障碍（含坑）
                     'platforms',      # 5 【新】等高方块高台
                     'flat',           # 6 【2026-09-25 新】纯平地（治"平地机身前高后低"：
                                       #   原 mix 无纯平地类型，平地行走+机身的经验几乎为零）
                     'stepping_stones',# 7
                     'gap',            # 8
                     'pit']            # 9

    def terrain_type_of(self, choice):
        """把 [0,1] 的 choice 按 terrain_proportions 的累积值映射成类型名。
        超出已配置带数的 choice（原代码的 else 分支）→ 'pit'。"""
        for k, cum in enumerate(self.proportions):
            if choice < cum:
                return self.TERRAIN_TYPES[k] if k < len(self.TERRAIN_TYPES) else 'pit'
        return 'pit'

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        amplitude = 0.01 + 0.07 * difficulty
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.1
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        kind = self.terrain_type_of(choice)
        if kind == 'slope':
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif kind == 'slope_rough':
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-amplitude, max_height=amplitude, step=0.005, downsampled_scale=0.2)
        elif kind == 'stairs_down':
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.30, step_height=-step_height, platform_size=3.)
        elif kind == 'stairs_up':
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.30, step_height=step_height, platform_size=3.)
        elif kind == 'discrete':
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif kind == 'platforms':
            # 【障碍带】高度随难度（= 行号）线性递增；参数在 config 里可调
            h_min = getattr(self.cfg, 'platform_height_min', 0.05)
            h_max = getattr(self.cfg, 'platform_height_max', 0.40)
            platform_terrain(terrain,
                             platform_height=h_min + (h_max - h_min) * difficulty,
                             band_depth=getattr(self.cfg, 'platform_band_depth', 1.0),
                             band_gap=getattr(self.cfg, 'platform_band_gap', 0.5),
                             spawn_size=getattr(self.cfg, 'platform_spawn_size', 2.0))
        elif kind == 'flat':
            # 【纯平地】SubTerrain 初始高度全 0，不做任何加工即为平地。
            # 目的: 给"平地行走且机身水平"提供充足训练正样本（配合 orientation v3
            # 的平地全额罚——纯平地上高差豁免恒为全额，mixed≡0）。
            pass
        elif kind == 'stepping_stones':
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif kind == 'gap':
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def platform_terrain(terrain, platform_height, band_depth=1.0, band_gap=0.5, spawn_size=2.0):
    """【高台/障碍带】地形（2026-09-21 第 2 版：从"孤立方块"改成"横贯全宽的障碍带"）。

    第 1 版是中心四周 8 个孤立方块（1.2×1.6m）。实测它的致命问题：
    从出生点出发有【16.2% 的方向可以完全不碰台子走满 4m】（= 地形课程的晋升门槛），
    平地占 79% —— 机器人一路只在平地游走就能晋升，等于从没被要求爬过任何东西。

    第 2 版沿 x 方向横贯【整片地形宽度】摆等高障碍带：任何沿 ±x 的移动都必须翻越障碍带，
    不存在"从旁边绕过去"的路线。中心留一条 spawn_size 宽的平坦出生带（横贯全宽），
    带与带之间用 band_gap 平地隔开（让机器人有落脚和再起跳的余地）。
    障碍高度由难度（行号）决定，同一行严格等高、完全确定（不吃随机数）。

    Parameters:
        platform_height (float): 障碍带高度 [m]
        band_depth (float): 单条障碍带沿 x 的进深 [m]
        band_gap (float): 障碍带之间的平地间隙 [m]
        spawn_size (float): 中心平坦出生带沿 x 的宽度 [m]（须 >= 2m：env_origin_z 取中心 2m 窗口最大值）
    Returns:
        terrain (SubTerrain): 带 _n_bands 属性记录实际摆放条数
    """
    h  = int(round(platform_height / terrain.vertical_scale))
    d  = max(1, int(round(band_depth   / terrain.horizontal_scale)))
    g  = max(1, int(round(band_gap     / terrain.horizontal_scale)))
    cs = max(2, int(round(spawn_size   / terrain.horizontal_scale)))

    hf = terrain.height_field_raw
    ni, nj = hf.shape                       # (width, length)；本工程里两者都是 80
    ci = ni // 2

    # 1) 中心平坦出生带（横贯全宽）→ 保证机器人在平地出生、env_origin_z = 0
    i0, i1 = max(0, ci - cs // 2), min(ni, ci + cs // 2)
    hf[i0:i1, :] = 0

    # 2) 出生带两侧交替摆【全宽障碍带 + 平地间隙】，直到放不下为止
    n = 0
    for sgn in (1, -1):
        cursor = i1 if sgn > 0 else i0      # 出生带的外沿
        while True:
            start = cursor + sgn * g
            end   = start + sgn * d
            lo, hi = (start, end) if sgn > 0 else (end, start)
            if lo < 0 or hi > ni or hi <= lo:
                break
            hf[lo:hi, :] = h                # 全宽（j 方向整条）
            n += 1
            cursor = hi if sgn > 0 else lo
    terrain._n_bands = n
    return terrain


def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
