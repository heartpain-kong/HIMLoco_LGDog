#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""妤兼鑳藉姏蹇€熻瘎娴嬶紙澶嶇敤 sim2sim 鐨勮娴?鎺у埗濂戠害锛屾棤娓叉煋锛夈€?
鐢ㄦ硶:
  python eval_stairs.py --policy <pt> --json <sidecar.json> --lane 1.0 --step-h 0.10 --secs 25

杈撳嚭涓€琛?JSON: 鏈€楂樻満韬珮搴︺€佸埌杈剧殑鏈€杩?x銆佹攢涓婄殑鍙伴樁绾ф暟銆佹槸鍚︾炕鍊掋€?"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # 仓库根目录
SCRIPTS = os.path.join(ROOT, 'legged_gym', 'legged_gym', 'scripts')
sys.path.insert(0, SCRIPTS)
import sim2sim as S  # noqa: E402
import mujoco          # noqa: E402

SCENE = os.path.join(ROOT, 'legged_gym', 'resources', 'robots', 'LgDog1', 'xml', 'scene.xml')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--json', default=None, help='policy 渚ц溅 json锛堟帶鍒跺绾︽潈濞佹潵婧愶級')
    ap.add_argument('--lane', type=float, default=1.0)
    ap.add_argument('--step-h', type=float, default=0.10)
    ap.add_argument('--secs', type=float, default=25.0)
    ap.add_argument('--cmd', type=float, default=0.6)
    ap.add_argument('--cmd-y', type=float, default=0.0)
    ap.add_argument('--friction', type=float, default=1.0)
    ap.add_argument('--spawn-x', type=float, default=0.0)
    ap.add_argument('--clamp-vel', action='store_true', default=True)
    ap.add_argument('--heading-hold', action='store_true', default=True)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    # ---- 鎺у埗濂戠害 ----
    ctrl = {}
    if a.json and os.path.exists(a.json):
        ctrl = json.load(open(a.json, encoding='utf-8')).get('control_contract', {})
    kp = float(ctrl.get('kp', S.KP)); kd = float(ctrl.get('kd', S.KD))
    dec = int(ctrl.get('decimation', S.DECIMATION))
    asc = float(ctrl.get('action_scale', S.ACTION_SCALE))
    S.NUM_ONE_STEP_OBS = int(ctrl.get('obs_one_step', S.NUM_ONE_STEP_OBS))
    S.HISTORY_LEN = int(ctrl.get('obs_history_len', S.HISTORY_LEN))
    ddp = ctrl.get('default_dof_pos')
    default_isaac = np.array(ddp, dtype=np.float64) if ddp else S.DEFAULT_DOF_POS_ISAAC
    default_mj = default_isaac[S.REMAP]
    os_sc = ctrl.get('obs_scales') or {}
    S.LV = float(os_sc.get('lin_vel', S.LV)); S.AV = float(os_sc.get('ang_vel', S.AV))
    S.DP = float(os_sc.get('dof_pos', S.DP)); S.DV = float(os_sc.get('dof_vel', S.DV))
    S.CLIP = float(ctrl.get('clip', S.CLIP))

    import torch
    policy = torch.jit.load(a.policy, map_location='cpu'); policy.eval()

    model = mujoco.MjModel.from_xml_path(SCENE)
    model.opt.timestep = S.SIM_DT
    model.geom_friction[:, 0] = float(a.friction)
    data = mujoco.MjData(model)
    rdof = model.nu

    # 鍑虹敓
    data.qpos[:7] = np.array([a.spawn_x, a.lane, 0.31, 1., 0., 0., 0.])
    data.qpos[7:] = default_mj
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    # 钀藉湴绋冲畾锛堢函 PD 淇濇寔榛樿濮挎€侊級
    for _ in range(int(0.5 / S.SIM_DT)):
        q, dq, quat, omega = S.get_obs(data)
        tau = np.clip((default_mj - q[-rdof:]) * kp - dq[-rdof:] * kd, -S.TAU_LIMIT, S.TAU_LIMIT)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if a.clamp_vel:
            S.clamp_joint_vel(data, rdof)
    z_stand = float(data.qpos[2])

    hist = [np.zeros(S.NUM_ONE_STEP_OBS, dtype=np.float32) for _ in range(S.HISTORY_LEN)]
    last_actions = np.zeros(12); target_q = default_mj.copy()
    cmd = np.array([a.cmd, a.cmd_y, 0.0])
    hdg_target = None

    total = int(a.secs / S.SIM_DT)
    zs = []; xs = []; ys = []; vs = []; ws = []
    for count in range(total):
        data.xfrc_applied[:] = 0.0
        q, dq, quat, omega = S.get_obs(data)
        if count % dec == 0:
            if a.heading_hold:
                cur_yaw = S._yaw_of_quat_xyzw(quat)
                if hdg_target is None:
                    hdg_target = cur_yaw
                cmd[2] = float(np.clip(S.HEADING_P * S._wrap_pi(hdg_target - cur_yaw),
                                       -S.WZ_CMD_MAX, S.WZ_CMD_MAX))
            if count > 0:
                obs = S.build_obs(q[-rdof:], dq[-rdof:], quat, omega, cmd, last_actions,
                                  S.REMAP, default_isaac)
                hist.insert(0, obs); hist.pop()
            oh = np.concatenate(hist).reshape(1, -1)
            with torch.no_grad():
                act = policy(torch.from_numpy(oh).float()).numpy().flatten()
            act = np.clip(act, -S.CLIP, S.CLIP)
            last_actions = act.copy()
            target_q = act[S.REMAP] * asc + default_mj
        tau = np.clip((target_q - q[-rdof:]) * kp - dq[-rdof:] * kd, -S.TAU_LIMIT, S.TAU_LIMIT)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if a.clamp_vel:
            S.clamp_joint_vel(data, rdof)
        zs.append(float(data.qpos[2])); xs.append(float(data.qpos[0]))
        ys.append(float(data.qpos[1]))
        vs.append(float(np.linalg.norm(data.qvel[0:2])))
        g = S.quat_rotate_inverse(S.get_obs(data)[2], np.array([0., 0., -1.]))
        ws.append(float(g[2]))

    zs = np.array(zs); xs = np.array(xs); ys = np.array(ys)
    keep = zs > (z_stand * 0.5)
    climbed = max(0.0, float(zs.max()) - z_stand) / max(a.step_h, 1e-6)
    out = dict(tag=a.tag, policy=os.path.basename(a.policy), lane=a.lane, step_h=a.step_h,
               z_stand=round(z_stand, 4), z_max=round(float(zs.max()), 4),
               z_end=round(float(zs[-1]), 4), x_final=round(float(xs[-1]), 3),
               x_max=round(float(xs.max()), 3), y_end=round(float(ys[-1]), 3),
               steps_climbed=round(climbed, 2), v_mean=round(float(np.mean(vs[-int(2 / S.SIM_DT):])), 3),
               upright_end=bool(ws[-1] > 0.5),
               dbg=(S.NUM_ONE_STEP_OBS, dec, asc, kp, kd))
    print('RESULT ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()

