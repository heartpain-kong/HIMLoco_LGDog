#!/usr/bin/env python3
"""check_model.py — MuJoCo 模型自检（结构 / 运动学 / 碰撞 / 动力学 / 离屏渲染）

用途：换 URDF、改 XML、动减速比之后，一条命令确认模型没坏：
       关节数、质量、执行器顺序、传感器、关节轴向与镜像约定、
       自碰撞、站立稳定性、小跑不发散，并渲染成图（默认 3/4 侧前视角）。

运行（conda 环境 himloco；有 DISPLAY 时也可 --gl glfw 弹窗看）：
  cd legged_gym/legged_gym/scripts
  MUJOCO_GL=egl python check_model.py ../../resources/robots/LgDog1/xml/LgDog1.xml
输出：
  终端打印各项检查结论（✔/✘），图片写到 --out-dir（默认 /tmp/mj_model_check）。

注意：MuJoCo 的 free camera 需要先 mjv_defaultFreeCamera 再改姿态。
"""

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")   # 必须在 import mujoco 之前

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", help="待检查的 MuJoCo XML")
    ap.add_argument("--root", default=None, help="根 body 名（默认自动取 worldbody 下第一个 body）")
    ap.add_argument("--kp", type=float, default=40.0, help="站立 PD 的 P 增益")
    ap.add_argument("--kd", type=float, default=1.0, help="站立 PD 的 D 增益")
    ap.add_argument("--out-dir", default=None,
                    help="图片输出目录（默认 <机器人目录>/check，即 XML 的上上级目录）")
    ap.add_argument("--no-render", action="store_true", help="跳过渲染")
    args = ap.parse_args()

    if args.out_dir is None:                    # .../robots/<机器人>/xml/<x>.xml → .../robots/<机器人>/check
        args.out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.xml))), "check")

    import mujoco

    os.makedirs(args.out_dir, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(os.path.abspath(args.xml))
    d = mujoco.MjData(m)
    name = lambda o, i: mujoco.mj_id2name(m, o, i)          # noqa: E731
    ok = lambda b: "✔" if b else "✘"                         # noqa: E731

    root = args.root
    if root is None:
        root = name(mujoco.mjtObj.mjOBJ_BODY, 1)
    root_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root)
    if root_id < 0:
        sys.exit(f"找不到 body: {root}")
    print(f"模型 {args.xml}\n根 body = {root}\n")

    # ───────────────────────── 1. 结构 ─────────────────────────
    print("=" * 62)
    print("1. 结构")
    print("=" * 62)
    print(f"  nq={m.nq} nv={m.nv} nu={m.nu} njnt={m.njnt} nbody={m.nbody} "
          f"ngeom={m.ngeom} nsite={m.nsite} nsensor={m.nsensor}")
    print(f"  总质量 = {m.body_mass.sum():.4f} kg")
    free = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    print(f"  free joint: {[name(mujoco.mjtObj.mjOBJ_JOINT, i) for i in free]} {ok(len(free) == 1)}")
    hinge = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    print(f"  铰链关节 {len(hinge)} 个；执行器 {m.nu} 个 {ok(m.nu == len(hinge))}（应一一对应）")
    print("  执行器顺序（写入 data.ctrl 的顺序）:")
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        print(f"    [{i:2d}] {name(mujoco.mjtObj.mjOBJ_ACTUATOR, i):10s} → "
              f"{name(mujoco.mjtObj.mjOBJ_JOINT, jid):16s} ctrlrange={np.round(m.actuator_ctrlrange[i], 1)}")
    need = ["Body_Quat", "Body_Gyro"]
    have = [name(mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)]
    print(f"  传感器 {len(have)} 个；sim2sim 需要的 {need}: "
          f"{ok(all(s in have for s in need))} {[s for s in need if s in have]}")
    print(f"  质量分布（应等于 URDF 各 link 质量之和）:")
    for i in range(1, m.nbody):
        print(f"    {name(mujoco.mjtObj.mjOBJ_BODY, i):10s} {m.body_mass[i]:8.4f} kg")

    # ───────────────────────── 2. 运动学 ─────────────────────────
    print()
    print("=" * 62)
    print("2. 运动学（零位 + 各关节对足端的影响）")
    print("=" * 62)
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    feet = [name(mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)
            if name(mujoco.mjtObj.mjOBJ_SITE, i).endswith("_foot")]
    zs = {}
    for f in feet:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f)
        p = d.site_xpos[sid] - d.xpos[root_id]
        zs[f] = p[2]
        print(f"  零位 {f:8s} 相对 {root}: x={p[0]:+.4f} y={p[1]:+.4f} z={p[2]:+.4f}")
    if zs:
        print(f"  四足等高 {ok(max(zs.values()) - min(zs.values()) < 1e-6)}"
              f"（站高 {abs(np.mean(list(zs.values()))):.4f} m）")
        ys = sorted(round(d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f)][1], 4) for f in feet)
        print(f"  左右对称 y = {ys} {ok(abs(ys[0] + ys[-1]) < 1e-6 and abs(ys[1] + ys[-2]) < 1e-6)}")

    qadr = m.jnt_qposadr
    print("\n  关节轴向约定（+0.2 rad 时足端位移；判断 default_dof_pos 的符号与镜像方式）:")
    print(f"    {'关节':16s} {'左腿(FL/RL) 足端 Δ':>22s}   {'右腿(FR/RR) 足端 Δ':>22s}")
    for part in ["hip", "thigh", "calf"]:
        row = []
        for side in ["FL", "FR"]:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{part}_joint")
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
            d.qpos[:] = 0
            mujoco.mj_forward(m, d)
            p0 = (d.site_xpos[sid] - d.xpos[root_id]).copy()
            d.qpos[qadr[jid]] = 0.2
            mujoco.mj_forward(m, d)
            row.append(d.site_xpos[sid] - d.xpos[root_id] - p0)
        print(f"    {part:16s} Δ=({row[0][0]:+.3f},{row[0][1]:+.3f},{row[0][2]:+.3f})   "
              f"Δ=({row[1][0]:+.3f},{row[1][1]:+.3f},{row[1][2]:+.3f})")
    print("    → 髋轴同号时左右足端 y 位移应反号（外摆用左右相反的默认角）；"
          "膝/大腿同号位移应同向（y 位移≈0）")

    # 质心 vs 支撑多边形
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    com = d.subtree_com[root_id]
    pts = np.array([d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f)][:2] for f in feet])
    print(f"\n  质心 xy = ({com[0]:+.4f}, {com[1]:+.4f})，足端 xy 包围盒 = "
          f"x[{pts[:,0].min():+.3f},{pts[:,0].max():+.3f}] y[{pts[:,1].min():+.3f},{pts[:,1].max():+.3f}]")
    inside = (pts[:, 0].min() <= com[0] <= pts[:, 0].max()) and (pts[:, 1].min() <= com[1] <= pts[:, 1].max())
    print(f"  静立时质心落在支撑多边形内 {ok(inside)}（否则零位就会自己翻倒）")

    # 关节限位可达性
    print("\n  关节限位（MuJoCo range，应等于 URDF 的 lower/upper）:")
    for i in hinge:
        print(f"    {name(mujoco.mjtObj.mjOBJ_JOINT, i):16s} range={np.round(m.jnt_range[i], 3)} "
              f"limited={bool(m.jnt_limited[i])} "
              f"actfrc={np.round(m.jnt_actfrcrange[i], 1)} "
              f"armature={m.dof_armature[m.jnt_dofadr[i]]:.3f}")

    # ───────────────────────── 3. 自碰撞 ─────────────────────────
    print()
    print("=" * 62)
    print("3. 自碰撞（网格凸包相交会导致机器人被顶翻）")
    print("=" * 62)
    rng = np.random.default_rng(0)
    lo = np.array([m.jnt_range[i][0] for i in hinge])
    hi = np.array([m.jnt_range[i][1] for i in hinge])
    worst = 0
    for trial in range(200):
        d.qpos[:] = 0
        d.qpos[7:] = lo + (hi - lo) * rng.random(len(hinge))
        mujoco.mj_forward(m, d)
        robot_geoms = set(range(m.ngeom))
        self_con = sum(1 for c in d.contact
                       if c.geom1 in robot_geoms and c.geom2 in robot_geoms
                       and m.geom_bodyid[c.geom1] != 0 and m.geom_bodyid[c.geom2] != 0)
        worst = max(worst, self_con)
    print(f"  200 个随机位形下最大自碰撞数 = {worst} {ok(worst == 0)}"
          f"（XML 里机器人几何 conaffinity=0 时必然为 0）")

    # ───────────────────────── 4. 动力学 ─────────────────────────
    print()
    print("=" * 62)
    print("4. 动力学（PD 站立 + 正弦小跑）")
    print("=" * 62)
    m.opt.timestep = 0.005
    lim = m.actuator_ctrlrange[:, 1]
    d = mujoco.MjData(m)
    d.qpos[2] = float(args.xml and abs(np.mean(list(zs.values()))) + 0.015) if zs else 0.3
    for _ in range(400):                       # 2 s 站立
        d.ctrl[:] = np.clip(args.kp * (0 - d.qpos[7:]) - args.kd * d.qvel[6:], -lim, lim)
        mujoco.mj_step(m, d)
    tilt = np.degrees(2 * np.arccos(min(1.0, abs(d.qpos[3]))))
    print(f"  站立 2s: z={d.qpos[2]:.4f} 倾角={tilt:.2f}° 接触={d.ncon} "
          f"有限值={ok(bool(np.isfinite(d.qpos).all()))}")
    print(f"  站姿倾斜 < 5° {ok(tilt < 5)}；四足着地 {ok(d.ncon >= 4)}")
    stand_z = d.qpos[2]

    amp = np.array([0, 0.35, -0.6] * 4)
    for i in range(1200):                      # 6 s 小跑
        tgt = amp * np.sin(2 * np.pi * 1.5 * i * 0.005)
        d.ctrl[:] = np.clip(args.kp * (tgt - d.qpos[7:]) - args.kd * d.qvel[6:], -lim, lim)
        mujoco.mj_step(m, d)
    print(f"  小跑 6s: z={d.qpos[2]:.4f} 接触={d.ncon} "
          f"关节速峰值={np.abs(d.qvel[6:]).max():.1f} rad/s "
          f"有限值={ok(bool(np.isfinite(d.qpos).all()))}")
    print(f"  小跑未发散（z 在 0.1~1.0 m 内）{ok(0.1 < d.qpos[2] < 1.0)}")

    if args.no_render:
        return

    # ───────────────────────── 5. 渲染 ─────────────────────────
    print()
    print("=" * 62)
    print("5. 离屏渲染（MUJOCO_GL=%s）" % os.environ["MUJOCO_GL"])
    print("=" * 62)
    import imageio.v2 as iio

    def shot(tag, cam_az, cam_el, dist=1.3, lookat=None, pose=None, trunk_z=None, data=None):
        dd = mujoco.MjData(m) if data is None else data
        if pose is not None:
            dd.qpos[:] = 0
            dd.qpos[7:] = pose
        if trunk_z is not None:
            dd.qpos[2] = trunk_z
        mujoco.mj_forward(m, dd)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, cam)
        cam.distance, cam.azimuth, cam.elevation = dist, cam_az, cam_el
        cam.lookat[:] = lookat if lookat is not None else [0, 0, stand_z]
        rr.update_scene(dd, cam)
        img = rr.render()
        path = os.path.join(args.out_dir, f"{tag}.png")
        iio.imwrite(path, img)
        print(f"  {tag:14s} → {path}")
        return img

    rr = mujoco.Renderer(m, height=340, width=480)
    dz = mujoco.MjData(m)                       # 重新复现一次站立姿态
    dz.qpos[2] = abs(np.mean(list(zs.values()))) + 0.015 if zs else 0.3
    for _ in range(400):
        dz.ctrl[:] = np.clip(args.kp * (0 - dz.qpos[7:]) - args.kd * dz.qvel[6:], -lim, lim)
        mujoco.mj_step(m, dz)
    z_stand = dz.qpos[2]

    imgs = []
    imgs.append(("零位 3/4 视角", shot("01_zero_3q", 130, -18, 1.3, [0, 0, 0.28], pose=np.zeros(12), trunk_z=0.30)))
    imgs.append(("站立 3/4 视角", shot("02_stand_3q", 130, -18, 1.3, [0, 0, z_stand], data=dz)))
    imgs.append(("站立 侧视", shot("03_stand_side", 90, -3, 1.2, [0, 0, z_stand], data=dz)))
    imgs.append(("站立 前视（看左右对称）", shot("04_stand_front", 180, -8, 1.2, [0, 0, z_stand], data=dz)))
    imgs.append(("站立 俯视（看四足位置）", shot("05_stand_top", 90, -89, 1.4, [0, 0, z_stand], data=dz)))
    p = np.zeros(12)
    for leg in (0, 3, 6, 9):
        p[leg + 2] = 0.7
    imgs.append(("calf=+0.7", shot("06_calf_pos", 90, -3, 1.3, [0, 0, 0.32], pose=p, trunk_z=0.36)))
    p = np.zeros(12)
    for leg in (0, 3, 6, 9):
        p[leg + 2] = -0.7
    imgs.append(("calf=-0.7", shot("07_calf_neg", 90, -3, 1.3, [0, 0, 0.16], pose=p, trunk_z=0.20)))
    p = np.zeros(12)
    for leg in (0, 3, 6, 9):
        p[leg] = 0.5
    imgs.append(("hip 外摆 ±0.5", shot("08_hip_splay", 180, -8, 1.3, [0, 0, 0.30], pose=p, trunk_z=0.34)))
    p = np.zeros(12)
    for leg in (0, 3, 6, 9):
        p[leg + 1] = 0.5
    imgs.append(("thigh=+0.5", shot("09_thigh_pos", 90, -3, 1.3, [0, 0, 0.30], pose=p, trunk_z=0.34)))
    # 小跑三帧
    dt = mujoco.MjData(m)
    dt.qpos[2] = z_stand
    for i in range(1200):
        tgt = amp * np.sin(2 * np.pi * 1.5 * i * 0.005)
        dt.ctrl[:] = np.clip(args.kp * (tgt - dt.qpos[7:]) - args.kd * dt.qvel[6:], -lim, lim)
        mujoco.mj_step(m, dt)
        if i in (300, 330, 360):
            imgs.append((f"trot t={i * 0.005:.2f}s",
                         shot(f"10_trot_{i}", 130, -18, 1.4, [0, 0, dt.qpos[2]], data=mujoco.MjData(m) if False else dt)))

    # 拼图（2 列）
    def montage(items, path):
        rows = []
        for i in range(0, len(items), 2):
            pair = [im for _, im in items[i:i + 2]]
            if len(pair) == 1:
                pair.append(np.zeros_like(pair[0]))
            rows.append(np.concatenate(pair, axis=1))
        iio.imwrite(path, np.concatenate(rows, axis=0))
        print(f"  拼图 → {path}")

    montage(imgs[:6], os.path.join(args.out_dir, "montage_A_姿态.png"))
    montage(imgs[6:], os.path.join(args.out_dir, "montage_B_极限位形.png"))
    print("\n检查完毕。")


if __name__ == "__main__":
    main()
