#!/usr/bin/env python3
"""viewer_terrain.py — 直接查看训练地形（Isaac Gym 真实 3D 网格），不启动训练

原理：复用 legged_gym 的 Terrain 类生成地形（含 vertices/triangles），
创建**空的** PhysX sim（不创建任何机器人环境），用 Isaac Gym viewer
渲染 trimesh 地形网格。可自由旋转 / 缩放 / 平移视角，观察断桥/高台/楼梯等。

用法（Linux，himloco 环境，有显示环境）：
  python viewer_terrain.py                 # 默认视角对准第 5 行、第 10 列
  python viewer_terrain.py --row 3 --col 9 # 对准某地形块（如断桥列 9-11）
  python viewer_terrain.py --col 13        # 看 20cm 高台列

控制：鼠标左键旋转 / 滚轮缩放 / 右键平移；Esc 退出。
"""

import argparse
import numpy as np

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.terrain import Terrain

from isaacgym import gymapi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row", type=int, default=5, help="初始相机对准的行（难度，0~9）")
    p.add_argument("--col", type=int, default=10, help="初始相机对准的列（类型，0~19）")
    p.add_argument("--elev", type=float, default=25, help="相机俯仰角（度）")
    p.add_argument("--dist", type=float, default=10, help="相机距离 [m]")
    args = p.parse_args()

    print("生成训练地形（纯 CPU）...")
    cfg = LeggedRobotCfg.terrain()
    terrain = Terrain(cfg, num_robots=1)
    print(f"地形: {cfg.num_rows}行(难度) × {cfg.num_cols}列(类型)，分布 {cfg.terrain_proportions}")
    print(f"顶点 {terrain.vertices.shape[0]}，三角面 {terrain.triangles.shape[0]}")

    gym = gymapi.acquire_gym()

    # ---- 创建空的 PhysX sim（不加载任何机器人）----
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.01
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("❌ 创建 sim 失败（检查 GPU/驱动）")
        return

    # ---- 添加地形 mesh（同 legged_gym _create_trimesh 逻辑）----
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = terrain.vertices.shape[0]
    tm_params.nb_triangles = terrain.triangles.shape[0]
    tm_params.transform.p.x = terrain.cfg.border_size
    tm_params.transform.p.y = terrain.cfg.border_size
    tm_params.transform.p.z = 0.0
    gym.add_triangle_mesh(sim, terrain.vertices.flatten(order='C'),
                          terrain.triangles.flatten(order='C'), tm_params)

    # ---- viewer ----
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1600
    cam_props.height = 900
    viewer = gym.create_viewer(sim, cam_props)
    if viewer is None:
        print("❌ 创建 viewer 失败（无显示环境？SSH 需 X 转发）")
        gym.destroy_sim(sim)
        return

    # ---- 初始相机对准某地形块 ----
    row = max(0, min(args.row, cfg.num_rows - 1))
    col = max(0, min(args.col, cfg.num_cols - 1))
    cx = (row + 0.5) * terrain.env_length
    cy = (col + 0.5) * terrain.env_width
    # 从 -y 方向斜上方看向目标块中心
    eye = gymapi.Vec3(cx, cy - args.dist * np.cos(np.deg2rad(args.elev)),
                      args.dist * np.sin(np.deg2rad(args.elev)))
    target = gymapi.Vec3(cx, cy, 0.0)
    gym.viewer_camera_look_at(viewer, None, eye, target)

    print(f"相机对准: 行{row} 列{col}（坐标 {cx:.1f},{cy:.1f}）")
    print("控制：鼠标左键旋转 / 滚轮缩放 / 右键平移；Esc 退出")
    print("列分布参考: 0-1斜坡 2-5粗糙 6-8三角坡 9-11断桥 12-15高台 16-19楼梯")

    # ---- 渲染循环（不 step 物理、不训练）----
    while not gym.query_viewer_has_closed(viewer):
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Done")


if __name__ == "__main__":
    main()