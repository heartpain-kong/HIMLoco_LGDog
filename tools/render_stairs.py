#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""楼梯视频导出 —— MuJoCo 离屏渲染 + 参考视频（02_stepping_stones_cmd0p6mps_front.mp4）同款 HUD。

画面: 1920x1080 / 25fps
  · 第三人称追尾视角（主画面）
  · 右上角「机身前景相机 · 下俯20°」画中画（青色边框 + 标题牌）
  · 顶部标题栏（任务 / 目标速度 / 模型名 / 推理说明）
  · 底部数据条（时间 / 实际平面速度 / 爬升高度 / 已上台阶）+ 青色进度条

用法:
  python render_stairs.py --lane 1.0 --step-h 0.10 --label 10 --plat-h 1.0 --out out.mp4
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # 仓库根目录
SCRIPTS = os.path.join(ROOT, 'legged_gym', 'legged_gym', 'scripts')
sys.path.insert(0, SCRIPTS)
import sim2sim as S                            # noqa: E402
import mujoco                                  # noqa: E402
import torch                                   # noqa: E402
from PIL import Image, ImageDraw, ImageFont    # noqa: E402
import imageio.v2 as iio                       # noqa: E402

SCENE = os.path.join(ROOT, 'legged_gym', 'resources', 'robots', 'LgDog1', 'xml', 'scene.xml')
W, H = 1920, 1080
FPS = 25
STEPS_PER_FRAME = int(round((1.0 / FPS) / S.SIM_DT))       # 200Hz / 25fps = 8

CYAN = (28, 208, 201)
DARK = (18, 20, 24)
FONT_R = r'C:\Windows\Fonts\msyh.ttc'
FONT_B = r'C:\Windows\Fonts\msyhbd.ttc'


def font(size, bold=True):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT_R, size)
    except Exception:
        return ImageFont.truetype(FONT_R, size)


def cam_from_pos_lookat(cam, pos, lookat):
    """由『相机位置 + 注视点』反推 MuJoCo free camera 的 lookat/distance/azimuth/elevation。

    实测约定: pos = lookat - d * (cos e * cos a, cos e * sin a, sin e)
    """
    pos = np.asarray(pos, dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)
    v = pos - lookat
    d = float(np.linalg.norm(v))
    if d < 1e-6:
        d = 1e-6
    u = v / d
    sin_e = float(np.clip(-u[2], -1.0, 1.0))
    e = np.arcsin(sin_e)
    a = np.arctan2(-u[1], -u[0])
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = d
    cam.azimuth = float(np.degrees(a))
    cam.elevation = float(np.degrees(e))
    return cam


def body_forward(quat_wxyz):
    """机身朝向在水平面的单位向量（世界系）"""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, quat_wxyz)
    f = m.reshape(3, 3)[:, 0].astype(np.float64)
    f[2] = 0.0
    n = np.linalg.norm(f)
    return f / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])


def main():
    ap = argparse.ArgumentParser()
    pol_dir = os.path.join(ROOT, 'legged_gym', 'logs', 'flat_LgDog1', 'exported', 'policies')
    ap.add_argument('--policy', default=os.path.join(pol_dir, 'policy_V51_18600.pt'))
    ap.add_argument('--json', default=None, help='策略侧车 json（控制契约），缺省用同名 .json')
    ap.add_argument('--lane', type=float, default=1.0, help='楼梯车道的 y 坐标')
    ap.add_argument('--step-h', type=float, default=0.10, help='每级高度')
    ap.add_argument('--label', default='10', help='HUD 上的每级高度文字')
    ap.add_argument('--plat-h', type=float, default=1.0, help='顶部平台高度（进度条满格）')
    ap.add_argument('--plat-x', type=float, default=4.0, help='顶部平台起始 x')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cmd', type=float, default=0.6)
    ap.add_argument('--cmd-y', type=float, default=0.0, help='横向指令（抵消策略直行漂移）')
    ap.add_argument('--secs', type=float, default=16.0)
    ap.add_argument('--friction', type=float, default=1.0)
    ap.add_argument('--spawn-x', type=float, default=0.0)
    ap.add_argument('--hold-secs', type=float, default=2.6, help='到顶后继续录制时长')
    ap.add_argument('--speed-label', default='0.60')
    ap.add_argument('--idx', type=int, default=1, help='本段序号（配合 --total 显示 第 n/m 段）')
    ap.add_argument('--total', type=int, default=1)
    ap.add_argument('--pip', action='store_true',
                    help='打开右上角前视画中画（默认关：这是 HIMLoco 盲策略，画面里不该出现相机）')
    a = ap.parse_args()

    ctrl = {}
    jf = a.json or os.path.splitext(a.policy)[0] + '.json'
    if os.path.exists(jf):
        ctrl = json.load(open(jf, encoding='utf-8')).get('control_contract', {})
    kp = float(ctrl.get('kp', S.KP)); kd = float(ctrl.get('kd', S.KD))
    dec = int(ctrl.get('decimation', S.DECIMATION))
    asc = float(ctrl.get('action_scale', S.ACTION_SCALE))
    S.NUM_ONE_STEP_OBS = int(ctrl.get('obs_one_step', S.NUM_ONE_STEP_OBS))
    S.HISTORY_LEN = int(ctrl.get('obs_history_len', S.HISTORY_LEN))
    ddp = ctrl.get('default_dof_pos')
    default_isaac = np.array(ddp, dtype=np.float64) if ddp else S.DEFAULT_DOF_POS_ISAAC
    default_mj = default_isaac[S.REMAP]
    osc = ctrl.get('obs_scales') or {}
    S.LV = float(osc.get('lin_vel', S.LV)); S.AV = float(osc.get('ang_vel', S.AV))
    S.DP = float(osc.get('dof_pos', S.DP)); S.DV = float(osc.get('dof_vel', S.DV))
    S.CLIP = float(ctrl.get('clip', S.CLIP))
    print(f'[契约] obs={S.NUM_ONE_STEP_OBS}/帧 × {S.HISTORY_LEN}  decimation={dec}  '
          f'action_scale={asc}  kp={kp} kd={kd}', flush=True)

    policy = torch.jit.load(a.policy, map_location='cpu'); policy.eval()

    model = mujoco.MjModel.from_xml_path(SCENE)
    model.opt.timestep = S.SIM_DT
    model.geom_friction[:, 0] = float(a.friction)
    model.vis.global_.offwidth = W
    model.vis.global_.offheight = H
    model.vis.global_.fovy = 42
    # 画面调亮 + 换色（只改渲染，不改 XML）: 参考视频是「灰地面 + 绿色台阶」，白机器人才跳得出来
    model.vis.headlight.ambient[:] = [0.26, 0.27, 0.29]
    model.vis.headlight.diffuse[:] = [0.42, 0.42, 0.42]
    model.vis.headlight.specular[:] = [0.04, 0.04, 0.04]
    _gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    if _gid < 0:
        for _i in range(model.ngeom):
            if model.geom_type[_i] == mujoco.mjtGeom.mjGEOM_PLANE:
                _gid = _i
                break
    if _gid >= 0:
        model.geom_rgba[_gid] = [0.33, 0.34, 0.36, 1.0]
    for _i in range(model.ngeom):
        _nm = model.geom(_i).name or ''
        if _nm.startswith('uw_stair'):
            if _nm.endswith('_top'):
                model.geom_rgba[_i] = [0.24, 0.45, 0.27, 1.0]     # 顶部平台：深一点的绿
            else:
                model.geom_rgba[_i] = [0.31, 0.57, 0.34, 1.0]     # 台阶：参考视频的绿

    data = mujoco.MjData(model)
    rdof = model.nu
    r_main = mujoco.Renderer(model, height=H, width=W)
    r_pip = mujoco.Renderer(model, height=420, width=660) if a.pip else None

    data.qpos[:7] = np.array([a.spawn_x, a.lane, 0.31, 1., 0., 0., 0.])
    data.qpos[7:] = default_mj
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for _ in range(int(0.5 / S.SIM_DT)):
        q, dq, quat, omega = S.get_obs(data)
        data.ctrl[:] = np.clip((default_mj - q[-rdof:]) * kp - dq[-rdof:] * kd,
                               -S.TAU_LIMIT, S.TAU_LIMIT)
        mujoco.mj_step(model, data)
        S.clamp_joint_vel(data, rdof)
    z_stand = float(data.qpos[2])

    hist = [np.zeros(S.NUM_ONE_STEP_OBS, dtype=np.float32) for _ in range(S.HISTORY_LEN)]
    last_actions = np.zeros(12)
    target_q = default_mj.copy()
    hdg_target = None
    cmd_now = np.array([a.cmd, a.cmd_y, 0.0])
    speed_f = 0.0

    cam_main = mujoco.MjvCamera()
    cam_pip = mujoco.MjvCamera()
    total = int(a.secs / S.SIM_DT)
    frames = 0
    reached_at = None
    on_plate_since = None
    log = []

    f_title = font(38); f_sub = font(27, False); f_pip = font(23, False)
    f_data = font(30); f_data_b = font(31)

    def draw_hud(img, t, v, climb, steps_done):
        canvas = Image.fromarray(img).convert('RGBA')
        ov = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        d.rectangle([0, 0, W, 104], fill=DARK + (215,))
        d.text((30, 10), f'上楼梯 · 每级 {a.label} cm │ 第 {a.idx}/{a.total} 段 │ 目标 {a.speed_label} m/s',
               font=f_title, fill=(255, 255, 255, 255))
        d.text((32, 62), f'{os.path.splitext(os.path.basename(a.policy))[0]} + HIMLoco 盲策略 · 仅本体感知 · 1 倍速',
               font=f_sub, fill=(168, 176, 186, 255))
        d.rectangle([0, H - 96, W, H], fill=DARK + (215,))
        d.text((30, H - 78), f't = {t:5.2f} s', font=f_data_b, fill=CYAN + (255,))
        d.text((300, H - 78), f'实际平面速度 {v:4.2f} m/s', font=f_data, fill=(235, 238, 242, 255))
        d.text((720, H - 78), f'爬升 {climb:4.2f} / {a.plat_h:4.2f} m',
               font=f_data, fill=(235, 238, 242, 255))
        d.text((1150, H - 78), f'已上台阶 {min(steps_done, 10):2d} / 10 级',
               font=f_data, fill=(235, 238, 242, 255))
        bar_y = H - 16
        d.rectangle([30, bar_y - 6, W - 30, bar_y], fill=(210, 216, 222, 60))
        pw = int((W - 60) * max(0.0, min(1.0, climb / a.plat_h)))
        if pw > 0:
            d.rectangle([30, bar_y - 6, 30 + pw, bar_y], fill=CYAN + (255,))
        return Image.alpha_composite(canvas, ov).convert('RGB')

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    writer = iio.get_writer(a.out, fps=FPS, codec='libx264', quality=8,
                            macro_block_size=None, ffmpeg_log_level='error')
    t = 0.0
    for count in range(total):
        data.xfrc_applied[:] = 0.0
        q, dq, quat, omega = S.get_obs(data)
        # ---- 到顶（或指定高度）后收指令，避免冲出平台末端掉下去 ----
        x_now, z_now = float(data.qpos[0]), float(data.qpos[2])
        on_plate = z_now > a.plat_h + z_stand - 0.14
        if on_plate:
            if on_plate_since is None:
                on_plate_since = t
        else:
            on_plate_since = None
        if reached_at is None and ((on_plate and x_now > a.plat_x + 0.18) or
                                  (on_plate_since is not None and (t - on_plate_since) > 1.0)):
            reached_at = t
        if reached_at is not None:
            k = min(1.0, (t - reached_at) / 0.30)
            cmd_now[0] = a.cmd * (1.0 - k)
            cmd_now[1] = a.cmd_y * (1.0 - k)
        if count % dec == 0:
            cur_yaw = S._yaw_of_quat_xyzw(quat)
            if hdg_target is None:
                hdg_target = cur_yaw
            cmd_now[2] = float(np.clip(S.HEADING_P * S._wrap_pi(hdg_target - cur_yaw),
                                       -S.WZ_CMD_MAX, S.WZ_CMD_MAX))
            if count > 0:
                obs = S.build_obs(q[-rdof:], dq[-rdof:], quat, omega, cmd_now, last_actions,
                                  S.REMAP, default_isaac)
                hist.insert(0, obs); hist.pop()
            oh = np.concatenate(hist).reshape(1, -1)
            with torch.no_grad():
                act = policy(torch.from_numpy(oh).float()).numpy().flatten()
            act = np.clip(act, -S.CLIP, S.CLIP)
            last_actions = act.copy()
            target_q = act[S.REMAP] * asc + default_mj
        data.ctrl[:] = np.clip((target_q - q[-rdof:]) * kp - dq[-rdof:] * kd,
                               -S.TAU_LIMIT, S.TAU_LIMIT)
        mujoco.mj_step(model, data)
        S.clamp_joint_vel(data, rdof)
        t += S.SIM_DT
        log.append([t, float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])])

        if count % STEPS_PER_FRAME == 0:
            base = np.array(data.qpos[:3])
            fwd = body_forward(np.array(data.qpos[3:7]))
            right = np.array([fwd[1], -fwd[0], 0.0])
            v_planar = float(np.linalg.norm(data.qvel[0:2]))
            speed_f = 0.75 * speed_f + 0.25 * v_planar
            climb = max(0.0, base[2] - z_stand)
            steps_done = int(climb / a.step_h + 0.35)
            # 主视角: 追尾（后 3.25m、上 1.80m、略偏右，注目前方 1.15m）；到顶后缓慢环绕
            u = 0.0
            if reached_at is not None:
                u = min(1.0, (t - reached_at) / max(a.hold_secs, 1e-6))
            th = np.radians(22.0) * u
            zoom = 1.0 - 0.16 * u
            look = base + fwd * (1.15 * (1.0 - 0.62 * u)) + np.array([0, 0, -0.02 - 0.08 * u])
            pos = base - (fwd * np.cos(th) + right * np.sin(th)) * (3.25 * zoom) \
                + right * (0.30 * (1.0 - u)) + np.array([0, 0, 1.80 * zoom * (1.0 - 0.06 * u)])
            cam_from_pos_lookat(cam_main, pos, look)
            model.vis.global_.fovy = 42
            r_main.update_scene(data, camera=cam_main)
            frame = r_main.render()
            pip_img = None
            if r_pip is not None:
                # 机身前景相机: 机身中心、略高，宽角下俯 20°（能看到自己的前腿）
                cf = base + fwd * 0.02 + np.array([0, 0, 0.02])
                lp = cf + fwd * 1.6 + np.array([0, 0, -0.58])
                cam_from_pos_lookat(cam_pip, cf, lp)
                model.vis.global_.fovy = 82
                r_pip.update_scene(data, camera=cam_pip)
                pip_img = Image.fromarray(r_pip.render()).resize((520, 331), Image.LANCZOS)
            canvas = draw_hud(frame, t, speed_f, climb, steps_done)
            if pip_img is not None:
                px, py = W - 520 - 34, 104 + 26
                d2 = ImageDraw.Draw(canvas)
                d2.rectangle([px - 3, py - 3, px + 520 + 3, py + 331 + 3], outline=CYAN, width=3)
                canvas.paste(pip_img, (px, py))
                d2.rectangle([px, py, px + 520, py + 38], fill=(10, 12, 15, 205))
                d2.text((px + 12, py + 6), '机身前景相机 · 下俯 20°',
                        font=f_pip, fill=(235, 240, 244, 255))
            writer.append_data(np.asarray(canvas))
            frames += 1
            if frames % 50 == 0:
                print(f'  t={t:5.2f}s  z={base[2]:.2f}  climb={climb:.2f}  steps={steps_done}',
                      flush=True)
        if reached_at is not None and (t - reached_at) > a.hold_secs:
            break

    writer.close()
    npz = os.path.splitext(a.out)[0] + '_traj.npz'
    np.savez(npz, traj=np.array(log), lane=a.lane, step_h=a.step_h, plat_h=a.plat_h)
    print(f'DONE {a.out} frames={frames} dur={frames / FPS:.2f}s traj={npz}')


if __name__ == '__main__':
    main()
