# -*- coding: utf-8 -*-
"""plot_sim2sim_run.py — 把 sim2sim 录制画成"看一眼就知道发生了什么"的图。

为什么要单独有个画图工具:
  mj_stand_report.py / analyze_attitude.py 给的是【均值】——平均值会把"第 30 秒摔了、之后躺了 570 秒"
  这种最要命的事抹平（V46 在 x=1.2 下就是这样：均值 z=0.06 m 看着像"蹲得低"，其实是躺了）。
  所以要看"效果"就得看时间序列。

用法:
  python tools/plot_sim2sim_run.py <out.png> <标签1>=<npz1> [<标签2>=<npz2> ...]
输出: 4 个面板 —— 机身高度 z / roll / pitch / 前进速度（虚线 = 指令）
      ★ 每张图上用红点标出"翻倒时刻"（|roll|>45° 或 |pitch|>60° 或 z<0.14）
"""
import os
import sys
import numpy as np


def quat_to_R(q):          # q = wxyz（与 analyze_attitude.py 同一套公式）
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load(path):
    d = np.load(path, allow_pickle=True)
    t = np.asarray(d['t']).reshape(-1)
    pos = np.asarray(d['base_pos']).reshape(-1, 3)
    quat = np.asarray(d['base_quat_wxyz']).reshape(-1, 4)
    vel = np.asarray(d['base_lin_vel']).reshape(-1, 3)
    cmd = np.asarray(d['cmd']).reshape(-1, 3)
    roll, pitch = [], []
    for q in quat:
        R = quat_to_R(q)
        # 与 analyze_attitude 一致：从旋转矩阵取 pitch/roll
        pitch.append(np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1))))
        roll.append(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
    return dict(t=t, z=pos[:, 2], roll=np.array(roll), pitch=np.array(pitch),
                vx=vel[:, 0], cmd_vx=cmd[:, 0], quat=quat)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    out = sys.argv[1]
    runs = []
    for spec in sys.argv[2:]:
        label, path = spec.split('=', 1)
        runs.append((label, load(path)))
        print('读入 %-28s %s' % (label, path))

    import matplotlib
    matplotlib.use('Agg')
    # ★ 中文字体: 不加这一段，图里所有中文都会渲染成方框（用户 2026-10-13 反馈的"乱码"就是它）
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mpl_cjk import setup as _cjk_setup, check_glyphs
    _cjk_setup()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for label, r in runs:
        # 翻倒判定：与 analyze_attitude 同口径
        bad = (np.abs(r['roll']) > 45) | (np.abs(r['pitch']) > 60) | (r['z'] < 0.14)
        first = int(np.argmax(bad)) if bad.any() else None
        fall_txt = ('翻倒 @ %.1fs' % r['t'][first]) if first is not None else '未翻倒'
        axes[0].plot(r['t'], r['z'], lw=1.0, label='%s（%s）' % (label, fall_txt))
        axes[1].plot(r['t'], r['roll'], lw=1.0, label='%s' % label)
        axes[2].plot(r['t'], r['pitch'], lw=1.0, label='%s' % label)
        axes[3].plot(r['t'], r['vx'], lw=1.0, label='%s' % label)
        if first is not None:
            for ax in axes:
                ax.axvline(r['t'][first], color='red', ls=':', lw=1.0)
                ax.plot(r['t'][first], ax.get_ylim()[0] if False else 0, 'r.', ms=0)  # 占位，保持图例干净
    # 指令曲线（取第一条有阶梯的 run）
    for label, r in runs:
        if np.ptp(r['cmd_vx']) > 0.05:
            axes[3].plot(r['t'], r['cmd_vx'], 'k--', lw=1.0, label='指令 vx')
            break

    axes[0].set_ylabel('机身高度 z [m]'); axes[0].axhline(0.14, color='gray', ls='--', lw=0.8)
    axes[1].set_ylabel('roll [deg]');    axes[1].axhline(0, color='gray', ls='--', lw=0.8)
    axes[2].set_ylabel('pitch [deg]');   axes[2].axhline(0, color='gray', ls='--', lw=0.8)
    axes[3].set_ylabel('前进速度 vx [m/s]'); axes[3].set_xlabel('仿真时间 [s]')
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('V46@7400 引入 sim2sim：x=1.2 m/s 效果（红线虚线=翻倒时刻）', fontsize=12)
    fig.tight_layout()
    _font, _miss = check_glyphs(fig)   # ★ 交付前自查（字体是否为中文字体 + 有无字形缺失）
    print('   字体 = %s  缺失字形 = %s' % (_font, _miss or '无'))
    fig.savefig(out, dpi=110)
    print('已保存', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
