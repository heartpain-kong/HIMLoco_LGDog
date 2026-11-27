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
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args, headless=True):
    args.headless = headless
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # ============ [调参记录体系] 保存配置快照（每个 run 自动存）============
    import json as _json
    from legged_gym.utils.helpers import class_to_dict
    from legged_gym import LEGGED_GYM_ROOT_DIR

    def _urdf_fingerprint(cfg):
        """URDF 指纹：关节限位/碰撞几何的改动在 env_cfg 里完全看不到，
        不单独留痕的话 tuning_log 的自动 diff 会误报'配置无改动'。"""
        import hashlib
        import xml.etree.ElementTree as ET
        path = str(cfg.asset.file).replace('{LEGGED_GYM_ROOT_DIR}', LEGGED_GYM_ROOT_DIR)
        with open(path, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        root = ET.parse(path).getroot()
        limits, collisions = {}, {}
        for j in root.findall('joint'):
            lim = j.find('limit')
            if lim is not None:
                limits[j.get('name')] = [lim.get('lower'), lim.get('upper')]
        for link in root.findall('link'):
            geoms = [shape.tag for c in link.findall('collision')
                     for g in c.findall('geometry') for shape in g]
            collisions[link.get('name')] = geoms
        return {"md5": md5, "joint_limits": limits, "collision_geometry": collisions}

    def _source_fingerprint():
        """源码指纹：奖励函数/算法代码里的常量（如 _reward_feet_air_time 里硬编码的
        0.5 秒门限）改起来在 env_cfg/train_cfg 里完全看不到，会重演'改了却没记录'。"""
        import hashlib
        repo_root = os.path.dirname(LEGGED_GYM_ROOT_DIR)
        roots = [os.path.join(repo_root, "legged_gym", "legged_gym"),
                 os.path.join(repo_root, "rsl_rl", "rsl_rl")]
        files = {}
        for r in roots:
            for dirpath, dirnames, filenames in os.walk(r):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for fn in sorted(filenames):
                    if not fn.endswith(".py"):
                        continue
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, repo_root)
                    with open(full, "rb") as fh:
                        files[rel] = hashlib.md5(fh.read()).hexdigest()[:10]
        agg = hashlib.md5("".join(f"{k}:{v}" for k, v in sorted(files.items())).encode()).hexdigest()
        return {"aggregate_md5": agg, "files": files}

    try:
        # The runner creates the timestamped run directory; the environment
        # config has no run_dir attribute of its own.
        run_dir = ppo_runner.log_dir
        if run_dir is None:
            raise RuntimeError("training runner has no log directory")
        os.makedirs(run_dir, exist_ok=True)
        try:
            urdf_fp = _urdf_fingerprint(env_cfg)
        except Exception as e:
            urdf_fp = {"error": str(e)}
        try:
            src_fp = _source_fingerprint()
        except Exception as e:
            src_fp = {"error": str(e)}
        with open(os.path.join(run_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
            _json.dump({"env_cfg": class_to_dict(env_cfg),
                        "train_cfg": class_to_dict(train_cfg),
                        "urdf_fingerprint": urdf_fp,
                        "source_fingerprint": src_fp},
                       f, indent=2, ensure_ascii=False, default=str)
        print(f"[LOG] 配置快照已保存: {run_dir}/config_snapshot.json "
              f"(含 URDF 指纹 md5={urdf_fp.get('md5', '?')[:8]}, "
              f"源码指纹 {src_fp.get('aggregate_md5', '?')[:8]})")
    except Exception as e:
        print(f"[LOG] 配置快照保存失败（不影响训练）: {e}")
    # =====================================================================

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args, headless=True)
