# -*- coding: utf-8 -*-
"""交叉检验: 用【录制 obs 里的 projected_gravity】独立复算 pitch，与 quaternion 路径对比。
obs 布局(L277-283, 45 维单步): [0:3]commands  [3:6]ang_vel  [6:9]projected_gravity
                              [9:21]dof_pos   [21:33]dof_vel  [33:45]actions
history 拼接为 newest-first ⇒ obs[0:45] 是最新帧。
重力关系: g_body = R^T · (0,0,-1)，R 由 base_quat_wxyz 得到。
"""
import sys, numpy as np

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])

d = np.load(sys.argv[1], allow_pickle=True)
t = np.asarray(d['t']).reshape(-1)
quat = np.asarray(d['base_quat_wxyz']).reshape(-1, 4)
obs = np.asarray(d['obs']).reshape(len(t), -1)
pg_obs = obs[:, 6:9]                      # 策略真正吃到的重力(归一化单位矢量)
errs, pitch_q, pitch_g = [], [], []
for i in range(0, len(t), 200):           # 每 200 步抽一个样
    R = quat_to_R(quat[i])
    g_body = R.T @ np.array([0., 0., -1.])          # 由四元数推重力
    errs.append(np.abs(g_body - pg_obs[i]).max())
    fwd = R[:, 0]
    pitch_q.append(np.degrees(np.arctan2(fwd[2], np.hypot(fwd[0], fwd[1]))))
    # 由重力反推 pitch: 纯俯仰时机体系重力 x 分量 = -sin(pitch)
    pitch_g.append(np.degrees(np.arcsin(np.clip(-g_body[0], -1, 1))))
errs = np.array(errs); pitch_q = np.array(pitch_q); pitch_g = np.array(pitch_g)
print(f"样本数 {len(errs)}（每 200 控制步一个）")
print(f"① 四元数推重力 vs 录制 obs 里的 projected_gravity: max|Δ|={errs.max():.4f} "
      f"(均值 {errs.mean():.4f})  ⇒ {'一致 ✔' if errs.max() < 0.02 else '不一致 ✘'}")
diff = np.abs(pitch_q - pitch_g)
print(f"② 两条独立路径的 pitch: 平均差 {diff.mean():.2f}° (max {diff.max():.2f}°)")
print(f"   四元数路径 pitch 均值 {pitch_q.mean():+.2f}° | 重力路径 pitch 均值 {pitch_g.mean():+.2f}°")
print(f"③ 符号约定核对: 抬头(前高后低) ⇒ 机体系重力 x 分量为负 ⇒ pg_obs[0]<0")
print(f"   本段 pg_obs[0] 均值 = {pg_obs[::200, 0].mean():+.4f}  ⇒ "
      f"{'抬头(与 pitch>0 一致) ✔' if pg_obs[::200, 0].mean() < 0 else '低头'}")
