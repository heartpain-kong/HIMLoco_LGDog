#!/usr/bin/env python3
"""LgRobotDog：URDF → MuJoCo XML 转换器（输出对齐 legged_gym/resources/robots/go1/xml/go1.xml 风格）

用法（需要装 mujoco 的解释器，例如 conda 环境 himloco）：
    ~/miniconda3/envs/himloco/bin/python urdf2xml.py \
        ../../resources/robots/LgDog1/urdf/LgDog1.urdf

默认输出到 <urdf 同级目录>/../xml/<机器人名>.xml，可用 -o 覆盖。

════════════════════════════════════════════════════════════════════════
URDF 直接喂给 MuJoCo 会踩的三个坑（本脚本自动修掉）：
  1) 网格路径：MuJoCo 的 URDF 解析器把 `../meshes/x.STL` 按「URDF 所在目录 + 文件名」
     解析，`../` 会被丢掉 → 报 "Error opening file .../urdf/FL_hip.STL"。
     修法：往 URDF 里注入 <mujoco><compiler meshdir="../meshes"/></mujoco>。
  2) 根 link 丢失：URDF 没有 floating joint 时，根 link 被并进 worldbody（静态），
     整个躯干的质量/惯量全部丢失（LgDog1：10.015kg → 6.528kg，少了 3.487kg 的 trunk）。
     修法：注入 <link name="world"/> + <joint type="floating"> 把根 link 挂到 world 上，
     MuJoCo 会生成真正的 free joint（nq=7+12=19）并保留惯量。
  3) 没有驱动器/传感器：URDF 没有 <transmission>，转换结果 nu=0，也没有 Body_Quat 等传感器，
     sim2sim / 训练脚本无法使用。修法：本脚本按 go1.xml 约定补 <actuator> 与 <sensor>。

另外两个工程约定（训练/sim2sim 直接依赖，见 legged_gym/legged_gym/scripts/sim2sim.py）：
  * 根 body 命名为 trunk（= URDF 的 base_link），sim2sim.py 按名字 'trunk' 找躯干
  * 执行器顺序 = [FR,FL,RR,RL] × [hip,thigh,calf]，sim2sim 的 data.ctrl 按该顺序写入
  * 网格碰撞用凸包：躯干凸包与大腿凸包在零位就相交，自碰撞力会把机器人顶翻，
    所以机器人几何统一 conaffinity=0（只与地形/地面碰撞，机器人自身不互撞）
════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

# 执行器顺序（与 go1.xml 一致，sim2sim.py 的 ctrl 写入顺序）
LEG_ORDER = ("FR", "FL", "RR", "RL")
JOINT_ORDER = ("hip", "thigh", "calf")


# ─────────────────────────── 1. 修 URDF 的两个坑 ───────────────────────────
def patch_urdf(urdf_path):
    """返回注入了 meshdir 与 floating-base 的 URDF 文本"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    meshdir = os.path.relpath(
        os.path.join(os.path.dirname(os.path.abspath(urdf_path)), "..", "meshes"),
        os.path.dirname(os.path.abspath(urdf_path)),
    )

    # 坑 1：显式指定 meshdir（放在 <robot> 的第一个子元素，MuJoCo 才会读到）
    # 坑 1b：MuJoCo 的 URDF 解析器默认 discardvisual=true —— 会把 <visual> 网格全丢掉，
    #        只留 <collision>。后果：没有 collision 的 link 会整个消失（渲染不出来），
    #        且 visual 与 collision 网格不同的模型会用碰撞网格去渲染。必须显式关掉。
    mj = ET.Element("mujoco")
    ET.SubElement(mj, "compiler", {"meshdir": meshdir, "discardvisual": "false"})
    root.insert(0, mj)

    # 坑 2：world + floating joint，保住根 link 的质量/惯量
    root_link = root.find("link")  # URDF 的第一个 link 即根 link
    world = ET.Element("link", {"name": "world"})
    floating = ET.Element("joint", {"name": "floating_base", "type": "floating"})
    ET.SubElement(floating, "parent", {"link": "world"})
    ET.SubElement(floating, "child", {"link": root_link.get("name")})
    root.insert(1, world)
    root.insert(2, floating)
    return ET.tostring(root, encoding="unicode")


# ─────────────────────────── 2. 让 MuJoCo 做几何/惯量换算 ───────────────────────────
def raw_convert(urdf_path):
    """URDF → MuJoCo 原始 XML（几何、惯量矩阵→quat+diaginertia、关节全部换算好）"""
    try:
        import mujoco
    except ImportError:
        sys.exit("需要 mujoco：请用带 mujoco 的解释器运行，例如 ~/miniconda3/envs/himloco/bin/python")

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    fd, tmp_urdf = tempfile.mkstemp(suffix=".urdf", dir=urdf_dir)  # 必须与源 URDF 同目录
    os.close(fd)
    tmp_xml = tmp_urdf[:-5] + ".xml"
    try:
        with open(tmp_urdf, "w") as f:
            f.write(patch_urdf(urdf_path))
        model = mujoco.MjModel.from_xml_path(tmp_urdf)
        mujoco.mj_saveLastXML(tmp_xml, model)
        return ET.parse(tmp_xml).getroot(), model
    finally:
        for p in (tmp_urdf, tmp_xml):
            if os.path.exists(p):
                os.remove(p)


# ─────────────────────────── 3. 后处理成 go1.xml 风格 ───────────────────────────
def _attr(el, key, default=None):
    return el.get(key, default)


def rebuild_body(src, is_root, root_name, foot_bodies, model, out_sites, joint_overrides):
    """把 raw XML 的 body 递归改写成 legged_gym 风格"""
    name = root_name if is_root else src.get("name")
    body = ET.Element("body", {"name": name})
    if src.get("pos"):
        body.set("pos", src.get("pos"))
    if src.get("quat"):
        body.set("quat", src.get("quat"))

    inertial = src.find("inertial")
    if inertial is not None:
        body.append(inertial)

    for joint in src.findall("joint"):
        j = ET.Element("joint", dict(joint.attrib))
        if j.get("type") == "free":
            j.set("name", "root")
            j.attrib.pop("limited", None)
            j.attrib.pop("actuatorfrclimited", None)
        elif j.get("name") in joint_overrides:
            # 带减速比的关节：被动参数按减速比折算（见文件头说明）
            j.attrib.update(joint_overrides[j.get("name")])
        body.append(j)

    # 几何：一个视觉（group1，不参与碰撞）+ 一个碰撞（conaffinity=0，只撞地形）
    geoms = src.findall("geom")
    visual = [g for g in geoms if g.get("contype", "1") == "0"]
    colliding = [g for g in geoms if g.get("contype", "1") != "0"]
    if not visual and colliding:  # URDF 里 visual 与 collision 用同一网格时 MuJoCo 只留碰撞
        visual = colliding
    for tag, group in (("visual", 1), ("collision", 3)):
        sources = visual if tag == "visual" else colliding
        for i, g in enumerate(sources):
            attrs = dict(g.attrib)
            attrs["name"] = f"{name}_{tag}" + (f"_{i}" if len(sources) > 1 else "")
            # group：1=视觉（默认显示）、3=碰撞（默认隐藏）。
            # MuJoCo 默认把所有 geom 都渲染出来，所以 URDF 里"只做碰撞"的小腿胶囊
            # 会明晃晃地插在小腿外面，很怪。放到 group 3 后 viewer/Renderer 默认隐藏，
            # 物理完全不变；要看接触几何时在 viewer 的 Rendering→Geom Group 里勾上 3。
            attrs["group"] = str(group)
            if tag == "visual":
                attrs["contype"] = "0"
                attrs["conaffinity"] = "0"
            else:
                attrs["contype"] = "1"
                attrs["conaffinity"] = "0"
            attrs.pop("density", None)
            # URDF 没有 capsule 图元，只能用 cylinder 表达；而 Isaac 侧
            # asset.replace_cylinder_with_capsule=True 会把它转成胶囊。
            # 为让两个仿真器形状一致，这里同步输出 capsule（尺寸语义相同）。
            if attrs.get("type") == "cylinder":
                attrs["type"] = "capsule"
            body.append(ET.Element("geom", attrs))

    if is_root:
        ET.SubElement(body, "site", {"name": "imu", "pos": "0 0 0"})
    if name in foot_bodies:
        pos = _lowest_point_centroid(model, name)
        # group=3：MuJoCo 默认 sitegroup=[1,1,1,0,0,0] → 默认不显示，只做位置标记用。
        # 想看接地点时在 viewer 的 Rendering→Site Group 里勾 3。
        ET.SubElement(body, "site", {"name": f"{name.rsplit('_', 1)[0]}_foot",
                                     "pos": pos, "size": "0.01", "rgba": "1 0 0 1",
                                     "group": "3"})
        out_sites.append((name, pos))

    for child in src.findall("body"):
        body.append(rebuild_body(child, False, root_name, foot_bodies, model, out_sites,
                                 joint_overrides))

    # 腿的顺序排成 go1.xml 的 [FR,FL,RR,RL]，便于和参考模型对照
    order = {leg: i for i, leg in enumerate(LEG_ORDER)}
    body[:] = sorted(
        body,
        key=lambda c: (0, order.get(c.get("name", "").split("_")[0], 99))
        if c.tag == "body" else (-1, 0),
    )
    return body


def _lowest_point_centroid(model, body_name):
    """body 的网格最低 5mm 顶点的质心，换算到 body 局部坐标系（= 足端接触中心）"""
    import mujoco
    import numpy as np

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pts = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        v = v.reshape(-1, 3)
        pts.append(v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
    if not pts:
        return "0 0 0"
    w = np.vstack(pts)
    sel = w[w[:, 2] < w[:, 2].min() + 0.005]
    local = data.xmat[bid].reshape(3, 3).T @ (sel.mean(axis=0) - data.xpos[bid])
    return f"{local[0]:.6f} {local[1]:.6f} {local[2]:.6f}"


def joint_overrides_from_gear(gear, armature, damping, frictionloss):
    """减速比 N（= 电机转速 / 关节转速）折算到关节侧的被动参数

        armature     转子惯量   J_j = N² · J_motor  → ×N²  （阻抗型：力矩随速度线性变化）
        damping      粘性阻尼   b_j = N² · b_motor  → ×N²
        frictionloss 库仑摩擦   τ_j = N  · τ_motor  → ×N   （扭矩型：与速度无关，只按力矩比放大）

    注意 frictionloss 是 ×N 而非 ×N²：它是恒定摩擦扭矩，折算关系和「力矩」相同只乘 N；
    只有惯量和粘性阻尼这类阻抗才乘 N²。
    """
    overrides = {}
    for leg in LEG_ORDER:
        for part in JOINT_ORDER:
            n = gear.get(part)
            if not n or n == 1:
                continue
            overrides[f"{leg}_{part}_joint"] = {
                "armature": f"{armature * n ** 2:g}",
                "damping": f"{damping * n ** 2:g}",
                "frictionloss": f"{frictionloss * n:g}",
            }
    return overrides


def build_xml(raw, model, robot_name, root_name, torques, root_z, foot_bodies, note, joint_overrides):
    """torques: {关节名: 力矩上限[Nm]}，逐关节生效（减速比不同的关节上限不同）"""
    out_sites = []
    root = ET.Element("mujoco", {"model": robot_name})
    root.append(ET.Comment(note))

    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": raw.find("compiler").get("meshdir")})
    ET.SubElement(root, "size", {"njmax": "500", "nconmax": "100"})
    ET.SubElement(root, "option", {"gravity": "0 0 -9.806", "iterations": "50",
                                   "solver": "Newton", "timestep": "0.002"})

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", {"contype": "1", "conaffinity": "1", "friction": "0.6 0.3 0.3",
                                    "rgba": "0.5 0.6 0.7 1", "margin": "0.001", "group": "0"})
    ET.SubElement(default, "light", {"castshadow": "false", "diffuse": "1 1 1"})
    ET.SubElement(default, "motor", {"ctrlrange": f"-{max(torques.values()):g} {max(torques.values()):g}",
                                     "ctrllimited": "true"})
    ET.SubElement(default, "camera", {"fovy": "60"})
    ET.SubElement(default, "joint", {"damping": "0.01", "armature": "0.01", "frictionloss": "0.2"})

    asset = ET.SubElement(root, "asset")
    for mesh in raw.find("asset").findall("mesh"):
        asset.append(mesh)

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "rgba", {"com": "0.502 1.0 0 0.5", "contactforce": "0.98 0.4 0.4 0.7",
                                   "contactpoint": "1.0 1.0 0.6 0.4"})
    ET.SubElement(visual, "scale", {"com": "0.2", "forcewidth": "0.035",
                                    "contactwidth": "0.10", "contactheight": "0.04"})

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"directional": "true", "diffuse": ".8 .8 .8",
                                   "pos": "0 0 10", "dir": "0 0 -10"})
    ET.SubElement(world, "camera", {"name": "track", "mode": "trackcom", "pos": "0 -1.3 1.6",
                                    "xyaxes": "1 0 0 0 0.707 0.707"})
    ET.SubElement(world, "geom", {"name": "floor", "type": "plane", "conaffinity": "1", "condim": "3",
                                  "contype": "1", "rgba": "0.5 0.9 0.9 0.1", "pos": "0 0 0",
                                  "size": "0 0 1"})

    raw_root_body = raw.find("worldbody").find("body")
    trunk = rebuild_body(raw_root_body, True, root_name, foot_bodies, model, out_sites, joint_overrides)
    trunk.set("pos", f"0 0 {root_z}")
    world.append(trunk)

    # 执行器：按 LEG_ORDER × JOINT_ORDER（sim2sim 的 ctrl 顺序）；
    # ctrlrange 逐关节取 URDF 的 effort（小腿有减速比，力矩上限与直驱关节不同）
    actuator = ET.SubElement(root, "actuator")
    for leg in LEG_ORDER:
        for part in JOINT_ORDER:
            joint = f"{leg}_{part}_joint"
            attrs = {"name": f"{leg}_{part}", "gear": "1", "joint": joint}
            if joint in torques:
                attrs["ctrlrange"] = f"-{torques[joint]:g} {torques[joint]:g}"
            ET.SubElement(actuator, "motor", attrs)

    sensor = ET.SubElement(root, "sensor")
    for leg in LEG_ORDER:
        for part in JOINT_ORDER:
            ET.SubElement(sensor, "jointpos", {"name": f"{leg}_{part}_pos", "joint": f"{leg}_{part}_joint"})
    for leg in LEG_ORDER:
        for part in JOINT_ORDER:
            ET.SubElement(sensor, "jointvel", {"name": f"{leg}_{part}_vel", "joint": f"{leg}_{part}_joint"})
    ET.SubElement(sensor, "accelerometer", {"name": "Body_Acc", "site": "imu"})
    ET.SubElement(sensor, "gyro", {"name": "Body_Gyro", "site": "imu"})
    ET.SubElement(sensor, "framepos", {"name": "Body_Pos", "objtype": "site", "objname": "imu"})
    ET.SubElement(sensor, "framequat", {"name": "Body_Quat", "objtype": "site", "objname": "imu"})

    _indent(root)
    return root, out_sites


def _indent(elem, level=0):
    """ET.indent 的替代实现（兼容 python3.8）"""
    pad = "\n" + "    " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "    "
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


# ─────────────────────────── 主流程 ───────────────────────────
def main():
    ap = argparse.ArgumentParser(description="URDF → MuJoCo XML（legged_gym 风格）")
    ap.add_argument("urdf", help="输入 URDF 路径")
    ap.add_argument("-o", "--output", help="输出 XML 路径（默认 <机器人>/xml/<机器人名>.xml）")
    ap.add_argument("--name", help="模型名（默认取 URDF 的 robot name）")
    ap.add_argument("--root-name", default="trunk", help="根 body 名（默认 trunk，sim2sim.py 依赖）")
    ap.add_argument("--torque", type=float, help="统一覆盖所有关节的力矩上限（默认逐关节取 URDF effort）")
    ap.add_argument("--spawn-z", type=float, help="根 body 出生高度（默认自动算：零位足底刚好离地）")
    ap.add_argument("--feet", default="calf", help="为其生成 *_foot site 的 body 名关键词（默认 calf）")
    ap.add_argument("--gear", nargs="*", default=[], metavar="关节=比值",
                    help="带减速比的关节，如 calf=2（电机:关节 = 2:1）。"
                         "该关节被动参数按减速比折算：armature/damping ×N²、frictionloss ×N")
    ap.add_argument("--armature", type=float, default=0.01, help="直驱关节的转子惯量（默认 0.01）")
    ap.add_argument("--damping", type=float, default=0.01, help="直驱关节的关节阻尼（默认 0.01）")
    ap.add_argument("--frictionloss", type=float, default=0.2, help="直驱关节的库仑摩擦（默认 0.2）")
    args = ap.parse_args()

    gear = {}
    for item in args.gear:
        if "=" not in item:
            sys.exit(f"--gear 需要 关节=比值 的形式，收到：{item}")
        part, ratio = item.split("=", 1)
        gear[part.strip()] = float(ratio)

    urdf = os.path.abspath(args.urdf)
    robot_name = args.name or ET.parse(urdf).getroot().get("name") or "robot"
    out_path = args.output or os.path.join(os.path.dirname(urdf), "..", "xml", f"{robot_name}.xml")
    out_path = os.path.abspath(out_path)

    print(f"[1/4] 修 URDF 两个坑（meshdir / floating base）→ MuJoCo 原始转换")
    raw, model = raw_convert(urdf)
    print(f"      raw: nbody={model.nbody} njnt={model.njnt} nq={model.nq} "
          f"nmesh={model.nmesh} 总质量={model.body_mass.sum():.3f}kg")

    # 力矩上限：逐关节取 URDF 的 <limit effort>（小腿带减速比，通常与其他关节不同），
    # --torque 给定时则统一覆盖
    torques = {}
    for joint in ET.parse(urdf).getroot().iter("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("effort"):
            torques[joint.get("name")] = float(limit.get("effort"))
    if not torques:
        torques = {j: 33.5 for j in (f"{l}_{p}_joint" for l in LEG_ORDER for p in JOINT_ORDER)}
    if args.torque is not None:
        torques = {k: args.torque for k in torques}
    calf = [v for k, v in torques.items() if "calf" in k]
    other = [v for k, v in torques.items() if "calf" not in k]
    print(f"      力矩上限：{'calf=' + str(calf[0]) if calf else ''}"
          f"{' 其余=' + str(other[0]) if other else ''} Nm"
          f"{'（逐关节不同）' if len(set(torques.values())) > 1 else ''}")

    # 出生高度：零位足底最低点刚好离地 +1.5cm
    if args.spawn_z is None:
        import numpy as np
        data = __import__("mujoco").MjData(model)
        __import__("mujoco").mj_forward(model, data)
        lows = []
        for gid in range(model.ngeom):
            if model.geom_type[gid] != __import__("mujoco").mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[gid]
            v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            v = v.reshape(-1, 3)
            w = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            lows.append(w[:, 2].min())
        args.spawn_z = round(-min(lows) + 0.015, 3)
        print(f"      零位足底最低点 z={min(lows):.5f} → 出生高度 trunk z={args.spawn_z}")

    foot_bodies = set()
    for body in ET.parse(urdf).getroot().iter("link"):
        if args.feet in body.get("name", ""):
            foot_bodies.add(body.get("name"))

    print(f"[2/4] 后处理：free joint / trunk / 执行器×{len(LEG_ORDER) * len(JOINT_ORDER)} / 传感器 / 碰撞分组")
    joint_overrides = joint_overrides_from_gear(gear, args.armature, args.damping, args.frictionloss)
    for part, n in gear.items():
        if n and n != 1:
            print(f"      减速比 {part}=1:{n:g} → 该关节 armature/damping ×{n ** 2:g}"
                  f"（{args.armature:g}→{args.armature * n ** 2:g} / {args.damping:g}→{args.damping * n ** 2:g}）、"
                  f"frictionloss ×{n:g}（{args.frictionloss:g}→{args.frictionloss * n:g}）")
    gear_note = (
        f"\n           * 被动参数：直驱关节 armature={args.armature:g} / damping={args.damping:g}"
        f" / frictionloss={args.frictionloss:g}"
        + "".join(
            f"；{part} 带 1:{n:g} 减速比 → armature=damping={args.armature * n ** 2:g}"
            f"（阻抗 ×N²）、frictionloss={args.frictionloss * n:g}（摩擦是扭矩，×N）"
            for part, n in gear.items() if n and n != 1)
    )
    note = (
        f"\n         {robot_name} —— 由 {os.path.relpath(urdf, os.path.dirname(out_path))} 转换而来"
        f"\n         转换脚本：legged_gym/legged_gym/scripts/urdf2xml.py（可重新生成，勿手动大改结构）"
        f"\n         结构约定对齐 resources/robots/go1/xml/go1.xml，sim2sim.py 可直接复用："
        f"\n           * 根 body 名 = {args.root_name}（= URDF 的根 link），带 free joint"
        f"\n           * 执行器顺序 = [{'/'.join(LEG_ORDER)}] × [{'/'.join(JOINT_ORDER)}]（与 go1.xml 一致）"
        f"\n           * 传感器 Body_Quat / Body_Gyro / Body_Acc / Body_Pos 齐全，imu site 在根 body 原点"
        f"\n           * 机器人几何 conaffinity=0：网格碰撞是凸包，trunk 凸包与 thigh 凸包零位就相交，"
        f"\n             自碰撞会把机器人顶翻；需要自碰撞时请改回 conaffinity=1 并用 <contact><exclude> 排除"
        f"\n           * 关节范围/力矩上限均沿用 URDF（力矩逐关节生效：小腿带减速比，"
        f"calf={max(torques.values()):g} Nm / 其余={min(torques.values()):g} Nm）；"
        f"URDF 的 velocity 在 MuJoCo 里无对应字段，只在训练侧生效"
        f"\n           * 关节角度范围（range）是关节侧数值，不随减速比换算："
        f"减速箱在电机与关节之间，关节能摆多少与它无关"
        f"{gear_note}"
        f"\n        "
    )
    root, sites = build_xml(raw, model, robot_name, args.root_name, torques, args.spawn_z,
                            foot_bodies, note, joint_overrides)
    for name, pos in sites:
        print(f"      {name} 足端 site（局部坐标）= {pos}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=False)
    print(f"[3/4] 写出 {out_path}")

    import mujoco
    m2 = mujoco.MjModel.from_xml_path(out_path)
    print(f"[4/4] 验证通过：nq={m2.nq} nv={m2.nv} nu={m2.nu} ngeom={m2.ngeom} "
          f"nsite={m2.nsite} nsensor={m2.nsensor} 总质量={m2.body_mass.sum():.3f}kg")
    print("      执行器顺序:", [mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m2.nu)])


if __name__ == "__main__":
    main()
