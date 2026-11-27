#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mj_stand_report.py —— 从 sim2sim 录制里出【站立质量体检】（全程 CPU，可与训练并行）。

为什么要它: 中途监控点想"自动检测"，但训练占着显卡（8GB 卡上并发跑 Isaac 评测会 OOM/Xid，
2026-10-02 就是这么把 V34 搞崩的）⇒ 只能走【CPU 侧】的独立探测。项目既有结论是
"MuJoCo 只信站立/姿态"（推进与爬障复现不出来），所以这个探针专测站立这一类:
  站立时的偏航角速度（自转）、机身高度、pitch/roll、位置漂移、四足离地占比。

yaw 角速度一律【由四元数差分】算，不用录制里的 base_ang_vel 字段
（2026-10-06 发现该字段与"朝向是否真的在变"对不上，口径存疑）。

用法:
  python tools/mj_stand_report.py <recording.npz> [--skip 200]
  python tools/mj_stand_report.py --latest        # 自动取 recorded_io 里最新的一份
"""
import argparse
import glob
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
REC_DIR = os.path.join(ROOT, 'legged_gym', 'legged_gym', 'scripts', 'recorded_io')


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rec', nargs='?', default=None)
    ap.add_argument('--latest', action='store_true')
    ap.add_argument('--skip', type=int, default=250, help='跳过开头这么多帧（起步过渡）')
    args = ap.parse_args()

    rec = args.rec
    if args.latest or not rec:
        cands = sorted(glob.glob(os.path.join(REC_DIR, '*.npz')), key=os.path.getmtime)
        if not cands:
            sys.exit('❌ recorded_io 里没有录制')
        rec = cands[-1]
    d = np.load(rec, allow_pickle=True)
    q = d['base_quat_wxyz']                      # MuJoCo 原始 wxyz
    bp = d['base_pos']
    cmd = d['cmd'] if 'cmd' in d.files else None
    t = d['t'] if 't' in d.files else None
    n = len(bp)
    sk = min(args.skip, n // 4)
    dt = 0.01
    if t is not None and n > 2:
        dt = float(np.median(np.diff(t[:, 0]))) or 0.01

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    yaw_rate = np.gradient(yaw, dt)
    fz = 2 * (x * z - w * y)                     # 机头方向 z 分量 → pitch
    fx = 1 - 2 * (y * y + z * z)
    fy = 2 * (x * y + w * z)
    pitch = np.degrees(np.arctan2(fz, np.sqrt(fx ** 2 + fy ** 2)))
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    roll = np.degrees(np.arctan2(r21, r22))

    m = slice(sk, n)
    dur = (n - sk) * dt
    print('录制 %s  帧 %d (%.0fs，统计后 %.0fs)  dt=%.3fs'
          % (os.path.basename(rec), n, n * dt, dur, dt))
    if cmd is not None:
        print('  指令中位: vx=%+.2f vy=%+.2f wz=%+.2f'
              % tuple(np.median(cmd[sk:], axis=0)))
    print('  机身高度 z 均值 %.3f m    pitch %+.2f°   roll %+.2f°'
          % (bp[m, 2].mean(), pitch[m].mean(), roll[m].mean()))
    print('  ★ 站立/移动自转 |ω_z| 均值 %.3f rad/s (%.1f °/s)   净转 %+.1f°  ← 站立时应≈0'
          % (np.abs(yaw_rate[m]).mean(), np.degrees(np.abs(yaw_rate[m]).mean()),
             np.degrees(wrap(yaw[-1] - yaw[sk]))))
    print('  位移: 净 %.3f m / 路径 %.3f m（每 %.0fs）'
          % (np.linalg.norm(bp[-1, :2] - bp[sk, :2]),
             np.linalg.norm(np.diff(bp[sk:, :2], axis=0), axis=1).sum(), dur))
    # 足端离地占比（用高度阈值判接触：足底 <1cm 视为支撑）
    if 'dof_pos' in d.files:
        try:
            import mujoco
            xml = os.path.join(ROOT, 'legged_gym', 'resources', 'robots', 'LgDog1', 'xml', 'LgDog1.xml')
            mod = mujoco.MjModel.from_xml_path(xml)
            dat = mujoco.MjData(mod)
            legs = ['FR', 'FL', 'RR', 'RL']
            qadr = {nm: mod.jnt_qposadr[mujoco.mj_name2id(mod, mujoco.mjtObj.mjOBJ_JOINT, nm)]
                    for nm in ['%s_%s_joint' % (l, j) for l in legs for j in ('hip', 'thigh', 'calf')]}
            gid = {l: mujoco.mj_name2id(mod, mujoco.mjtObj.mjOBJ_GEOM, '%s_calf_collision_1' % l) for l in legs}
            rq = mod.jnt_qposadr[mujoco.mj_name2id(mod, mujoco.mjtObj.mjOBJ_JOINT, 'root')]
            adr = np.array([qadr['%s_%s_joint' % (l, j)] for l in legs for j in ('hip', 'thigh', 'calf')])
            zs = np.zeros((n, 4))
            for i in range(sk, n):
                dat.qpos[:] = 0.0
                dat.qpos[rq:rq + 3] = bp[i]
                dat.qpos[rq + 3:rq + 7] = q[i]
                dat.qpos[adr] = d['dof_pos'][i]
                mujoco.mj_kinematics(mod, dat)
                for k, l in enumerate(legs):
                    zs[i, k] = dat.geom_xpos[gid[l]][2] - 0.012
            air = (zs[sk:] > 0.01).mean(axis=0)
            print('  足端离地占比: ' + '  '.join('%s %.1f%%' % (legs[k], 100 * air[k]) for k in range(4))
                  + '   ← 站立时应≈0%')
        except Exception as e:      # 缺 mujoco/模型时降级为不打印
            print('  （跳过足端离地统计: %s）' % e)


if __name__ == '__main__':
    main()
