# -*- coding: utf-8 -*-
"""从 sim2sim 录制 npz 里算【机身前高后低】: 用 base_quat_wxyz 推 pitch/roll。
用法: python analyze_attitude.py <npz> [--skip-sec 2.0]
输出: 行走段（跳过前 skip-sec）的 pitch/roll 均值/中位/p95，以及 z/速度参考。
"""
import sys, numpy as np

def quat_to_R(q):  # q = wxyz
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])

def main():
    path = sys.argv[1]
    skip = 2.0
    if '--skip-sec' in sys.argv:
        skip = float(sys.argv[sys.argv.index('--skip-sec') + 1])
    d = np.load(path, allow_pickle=True)
    keys = [k for k in d.keys() if not k.startswith('_meta')]
    t = np.asarray(d['t']).reshape(-1)
    quat = np.asarray(d['base_quat_wxyz']).reshape(-1, 4)
    pos = np.asarray(d['base_pos']).reshape(-1, 3)

    # 【2026-09-26 新增】摔倒检测: |roll|>45° 或 |pitch|>60° 或 z<0.14 视为翻倒。
    #   为什么需要: 中途翻倒会把整段统计污染成"侧躺姿态"(V19@2200 实测 roll 中位 -111°)，
    #   真正要看的"行走姿态"必须只看翻倒之前那一段。
    all_pitch, all_roll = [], []
    for q in quat:
        R = quat_to_R(q)
        fwd = R[:, 0]
        all_pitch.append(np.degrees(np.arctan2(fwd[2], np.hypot(fwd[0], fwd[1]))))
        all_roll.append(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
    all_pitch = np.array(all_pitch); all_roll = np.array(all_roll)
    zz = pos[:, 2]
    # 【2026-09-27 修·第2版】把两类异常【分开】处理（第1版把 z<0.14 也当"翻倒"，导致
    #   V20@15200 的一次 1.2s 塌坐(z 0.045~0.134, pitch/roll 正常)把整段统计截断）:
    #     · 翻倒(flip)   = |roll|>45° 或 |pitch|>60° 且【连续 persist 步】——真翻转
    #     · 低姿态(lowz) = z<0.14（塌坐/趴下），单独计数与报时长，不当作翻倒
    #   主统计 = 全程剔除这两类样本后的有效段（不再截断），并如实报出异常占比。
    persist = 10
    raw_flip = ((np.abs(all_roll) > 45) | (np.abs(all_pitch) > 60)).astype(int)
    flip_persist = np.convolve(raw_flip, np.ones(persist, dtype=int), mode='same') >= persist
    lowz = zz < 0.14
    excl = flip_persist | lowz
    m = (t >= skip) & (~excl)
    pitch = all_pitch[m]; roll = all_roll[m]
    z = pos[m][:, 2]

    def _longest_run(mask):
        best = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    print(f"记录: {path}")
    print(f"  长度 {len(t)} 控制步 ({t[-1]:.1f}s); 有效统计段 {m.sum()} 步 "
          f"(剔除 {int((t >= skip).sum() - m.sum())} 步异常, 占 {100*(1-m.sum()/max((t>=skip).sum(),1)):.2f}%)")
    if flip_persist.any():
        print(f"  ❌ 翻倒: t≈{t[int(np.argmax(flip_persist))]:.1f}s 起（连续≥{persist}步 |roll|>45° 或 |pitch|>60°）")
    else:
        print("  ✅ 未翻倒")
    if lowz.any():
        print(f"  ⚠️ 低姿态(塌坐/趴下, z<0.14): {int(lowz.sum())} 步, 最长连续 {_longest_run(lowz)} 步 "
              f"({_longest_run(lowz)*0.02:.1f}s), 首次 t≈{t[int(np.argmax(lowz))]:.1f}s")
    if m.sum() == 0:
        print("  ⚠️ 无有效行走段")
        return
    print(f"  pitch 前高后低: 均值 {pitch.mean():+.2f}°  中位 {np.median(pitch):+.2f}°  "
          f"p5 {np.percentile(pitch,5):+.2f}°  p95 {np.percentile(pitch,95):+.2f}°  "
          f"std {pitch.std():.2f}°")
    print(f"  roll  侧倾    : 均值 {roll.mean():+.2f}°  中位 {np.median(roll):+.2f}°  std {roll.std():.2f}°")
    print(f"  机身高度 z    : 均值 {z.mean():.3f} m  中位 {np.median(z):.3f} m")
    # 【2026-09-27 新增】航向保持（用户关注的"平地弧线漂移"）:
    #   直线率 = 起终点直线距离 / 实际路径长度（1.0=笔直，越小越像绕圈）；
    #   另报 yaw 变化与横向偏移，便于跨轮对比。
    xy = pos[m][:, :2]
    if len(xy) > 10:
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        path = float(d.sum()); straight = float(np.linalg.norm(xy[-1] - xy[0]))
        yaw = np.degrees(np.arctan2(2*(quat[m][:, 0]*quat[m][:, 3] + quat[m][:, 1]*quat[m][:, 2]),
                                    1 - 2*(quat[m][:, 2]**2 + quat[m][:, 3]**2)))
        yaw_unwrapped = np.unwrap(yaw)
        print(f"  航向保持      : 直线率 {100*straight/max(path,1e-6):.1f}% "
              f"(路径 {path:.1f}m / 直线 {straight:.1f}m) | yaw 累计变化 {yaw_unwrapped[-1]-yaw_unwrapped[0]:+.0f}° "
              f"| 横向偏移 {xy[-1][1]-xy[0][1]:+.1f}m")
        # 【2026-09-28 新增】步频/步幅估算（验收「减少步频、加大步幅」这类改动用）:
        #   机身竖直起伏的【主频】≈ 步频（trot 每个落脚引起一次起伏），据此 估算步幅 ≈ 速度/步频。
        #   注意这是【估算】（依赖步态近似），跨轮对比的趋势可靠；精确足端步幅需 FK，见 tools/foot_audit.py。
        dur = float(t[m][-1] - t[m][0])
        if dur > 5.0:
            zt = z - z.mean()
            n = len(zt); fps = n / dur
            spec = np.abs(np.fft.rfft(zt * np.hanning(n)))
            freqs = np.fft.rfftfreq(n, d=1.0 / fps)
            band = (freqs > 0.5) & (freqs < 6.0)
            if band.any():
                f_bob = float(freqs[band][np.argmax(spec[band])])
                speed = path / max(dur, 1e-6)
                print(f"  步频/步幅估算 : 机身起伏主频 ≈ {f_bob:.2f} Hz（≈步频） | 平均速度 {speed:.2f} m/s "
                      f"| 估算步幅 ≈ {speed/max(f_bob,1e-6):.3f} m")

if __name__ == '__main__':
    main()
