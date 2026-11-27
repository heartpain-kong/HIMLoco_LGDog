#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""策略输入/输出记录器 —— play.py(Isaac Gym) 与 sim2sim.py(MuJoCo) 共用。

用途
----
把策略每一步的【输入(观测)】与【输出(动作)】原样落盘，用于：
  1. 排查 play 与 sim2sim 行为不一致（同一份 .pt，两边观测是否真的对齐？）
  2. 复现某一步的异常动作（回放/对比）
  3. 给后续分析（接触力、关节轨迹）留原始数据

统一 schema（两个仿真器一致，便于直接对比）
--------------------------------------------
  obs      (N, 270)  策略输入：6 帧 × 45 维历史（最新帧在前的约定见各脚本）
  action   (N, 12)   策略输出：原始动作（未乘 action_scale、未 remap）
  step     (N,)      控制步序号
  t        (N,)      仿真时间 [s]
  cmd      (N, 3)    指令 (vx, vy, wz)
  ---- 以下为可选附加量（有哪些取决于调用方传了什么）----
  dof_pos / dof_vel / base_lin_vel / base_ang_vel / contact_z / tau / target_q ...

用法
----
    from record_io import IORecorder

    with IORecorder(tag='play', every=1, enabled=True) as rec:
        for i in range(N):
            obs = ...; act = policy(obs)
            rec.add(step=i, t=i*dt, obs=obs, action=act, cmd=cmd,
                    dof_pos=dof_pos, dof_vel=dof_vel)
    # 退出 with 时自动落盘（含 Ctrl-C / 异常，不会白跑）

命令行
------
    python record_io.py list                      # 列出已记录的会话
    python record_io.py show  <a.npz>             # 看单个记录的摘要
    python record_io.py compare <a.npz> <b.npz>   # 对比两份记录（play vs sim2sim）
"""

import os
import sys
import csv
import glob
import time
import atexit
import numpy as np

# 数据默认落在本文件同级的 recorded_io/ 下（"放 scripts 里"）
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(_HERE, 'recorded_io')


class IORecorder:
    """策略输入/输出记录器。

    Args:
        tag:        会话标签，用于文件名，如 'play' / 'sim2sim' / 'go1'
        out_dir:    输出目录，默认 scripts/recorded_io/
        every:      抽帧间隔（every=5 表示每 5 个控制步记一次），默认 1 = 全记
        enabled:    False 时所有 add() 直接返回（便于用一个开关同时关掉两边）
        max_steps:  上限，防止长时间跑把内存撑爆（默认 200000 步 ≈ 270 维 × 20 万）
        csv:        额外写一份 CSV（270 列很宽，默认关；需要人眼查看时打开）
        env_index:  多环境时只记第几个 env（play.py 默认只在 env0 上操作）
        note:       自由文本，写进元数据
    """

    def __init__(self, tag='run', out_dir=None, every=1, enabled=True,
                 max_steps=200000, csv=False, env_index=0, note=''):
        self.tag = tag
        self.out_dir = out_dir or DEFAULT_OUT_DIR
        self.every = max(int(every), 1)
        self.enabled = bool(enabled)
        self.max_steps = int(max_steps)
        self.write_csv = bool(csv)
        self.env_index = int(env_index)
        self.note = note

        self._cols = {}          # name -> list of np.ndarray
        self._n = 0              # 已记录步数
        self._seen = 0           # 已调用 add 次数（用于抽帧）
        self._saved = False
        self._warned_full = False
        self.t_start = time.strftime('%Y-%m-%d %H:%M:%S')

    # ---------------- 记录 ----------------
    def add(self, step=None, t=None, obs=None, action=None, **extra):
        """记录一步。obs/action 传 torch.Tensor 或 np.ndarray 均可（会自动脱离计算图）。

        多环境张量默认只取 env_index 那一行；传长度不匹配的会被广播/截断保护。
        """
        if not self.enabled:
            return
        self._seen += 1
        if (self._seen - 1) % self.every != 0:
            return
        if self._n >= self.max_steps:
            if not self._warned_full:
                print(f"[IORecorder:{self.tag}] 已达 max_steps={self.max_steps}，停止记录")
                self._warned_full = True
            return

        row = {}
        if obs is not None:
            row['obs'] = _flat(obs, self.env_index)
        if action is not None:
            row['action'] = _flat(action, self.env_index)
        if step is not None:
            row['step'] = np.asarray([step], dtype=np.float64)
        if t is not None:
            row['t'] = np.asarray([t], dtype=np.float64)
        for k, v in extra.items():
            if v is None:
                continue
            row[k] = _flat(v, self.env_index)

        # 首次遇到某个键时确定宽度；后续宽度不一致的直接丢弃并告警（避免存出参差不齐的数组）
        for k, v in row.items():
            if k not in self._cols:
                self._cols[k] = []
            self._cols[k].append(v)
        self._n += 1

    # ---------------- 落盘 ----------------
    def save(self, path=None):
        if not self.enabled or self._saved or self._n == 0:
            return None
        os.makedirs(self.out_dir, exist_ok=True)
        if path is None:
            stamp = time.strftime('%m%d_%H%M%S')
            path = os.path.join(self.out_dir, f'{self.tag}_{stamp}.npz')

        # 对齐宽度：不同来源的同一键宽度可能不同（如 obs 270 vs 45），取最小公共宽度
        arrays, meta = {}, {
            'tag': self.tag, 'n_steps': self._n, 'every': self.every,
            'env_index': self.env_index, 'note': self.note,
            'started': self.t_start, 'saved': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        for k, vs in self._cols.items():
            widths = {v.shape[0] for v in vs}
            if len(widths) > 1:
                w = min(widths)
                print(f"[IORecorder:{self.tag}] 键 '{k}' 宽度不一致 {sorted(widths)}，截断到 {w}")
                vs = [v[:w] for v in vs]
            arrays[k] = np.stack(vs).astype(np.float32)
        np.savez_compressed(path, **arrays, **{f'_meta_{k}': np.array(v) for k, v in meta.items()})

        if self.write_csv:
            csv_path = os.path.splitext(path)[0] + '.csv'
            keys = sorted(arrays)
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow([f'{k}[{i}]' if arrays[k].ndim > 1 else k
                            for k in keys for i in range(arrays[k].shape[1] if arrays[k].ndim > 1 else 1)])
                for r in range(self._n):
                    w.writerow([arrays[k][r, i] if arrays[k].ndim > 1 else arrays[k][r]
                                for k in keys for i in range(arrays[k].shape[1] if arrays[k].ndim > 1 else 1)])
            print(f"[IORecorder:{self.tag}] CSV -> {csv_path}")

        print(f"[IORecorder:{self.tag}] 已保存 {self._n} 步 -> {path}")
        for k in sorted(arrays):
            print(f"    {k:16s} {tuple(arrays[k].shape)}")
        self._saved = True
        return path

    # ---------------- 上下文管理（保证异常/Ctrl-C 也能落盘）----------------
    def __enter__(self):
        atexit.register(self.save)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.save()
        return False   # 不吞异常


def _flat(v, env_index=0):
    """任意输入 -> 一维 float64 ndarray（**强制拷贝**）。

    ⚠️ 必须拷贝，这是本模块最容易踩的坑：
       - MuJoCo 的 data.qvel / data.qpos / q / dq 都是【原地更新的缓冲区视图】，
         np.asarray 不会复制，存下来的是"同一块内存的别名"，
         等仿真跑完再落盘时所有行都变成最后一个时刻的值（表现为整列恒定）。
       - Isaac 侧 env.get_observations() 返回的是预分配的 self.obs_buf，
         每一步被原地改写；torch 的 .numpy() 默认与 tensor 共享内存，同样会中招。
       所以这里统一 np.array(..., copy=True)。
    """
    if hasattr(v, 'detach'):
        v = v.detach().cpu().numpy()
    a = np.array(v, dtype=np.float64, copy=True)
    if a.ndim == 0:
        return a.reshape(1)
    if a.ndim >= 2:
        # (num_envs, dim) -> 取指定 env；若第一维不是 env 维（比如 1×270），也安全
        if a.shape[0] > env_index:
            a = a[env_index]
        a = a.reshape(-1)
    return a.ravel()


# ======================== 命令行工具 ========================
def _load(path):
    d = np.load(path, allow_pickle=False)
    meta = {k[len('_meta_'):]: d[k] for k in d.files if k.startswith('_meta_')}
    arrs = {k: d[k] for k in d.files if not k.startswith('_meta_')}
    return arrs, meta


def cmd_list(_args):
    files = sorted(glob.glob(os.path.join(DEFAULT_OUT_DIR, '*.npz')))
    if not files:
        print(f"（{DEFAULT_OUT_DIR} 下还没有记录）")
        return
    print(f"{'文件':<44}{'步数':>8}{'大小':>10}  标签")
    for f in files:
        try:
            arrs, meta = _load(f)
            n = int(meta.get('n_steps', arrs.get('obs', np.zeros((0,))).shape[0]))
            tag = str(meta.get('tag', '?'))
        except Exception as e:
            n, tag = -1, f'读取失败: {e}'
        print(f"{os.path.basename(f):<44}{n:>8}{os.path.getsize(f)/1024:>9.1f}K  {tag}")


def cmd_show(args):
    arrs, meta = _load(args.path)
    print(f"===== {args.path} =====")
    for k, v in meta.items():
        print(f"  meta.{k} = {v}")
    print("  ---- 数组 ----")
    for k in sorted(arrs):
        print(f"  {k:16s} {str(arrs[k].shape):>16}  "
              f"范围 [{arrs[k].min():+.3f}, {arrs[k].max():+.3f}]")


def cmd_compare(args):
    a_arr, a_meta = _load(args.a)
    b_arr, b_meta = _load(args.b)
    print(f"===== 对比 =====\n  A: {os.path.basename(args.a)}  tag={a_meta.get('tag')}  "
          f"n={a_meta.get('n_steps')}\n  B: {os.path.basename(args.b)}  tag={b_meta.get('tag')}  "
          f"n={b_meta.get('n_steps')}")
    common = sorted(set(a_arr) & set(b_arr))
    n = min(v.shape[0] for k, v in a_arr.items() if k in common)
    print(f"  公共键: {common}   对齐长度 = {n} 步"
          f"{'  ⚠️ 两份长度不同，已截断到较短者' if a_arr[common[0]].shape[0] != b_arr[common[0]].shape[0] else ''}")
    for k in common:
        x, y = a_arr[k][:n], b_arr[k][:n]
        if x.shape != y.shape:
            print(f"  {k:14s} 形状不同 {x.shape} vs {y.shape} → 跳过")
            continue
        d = np.abs(x - y)
        print(f"  {k:14s} 最大差 {d.max():.6g}   平均差 {d.mean():.6g}   "
              f"A范围[{x.min():+.2f},{x.max():+.2f}] B范围[{y.min():+.2f},{y.max():+.2f}]")

    # ---- 语义分块：270 = 6 帧 × 45 维（最新帧在前）----
    # 这一步回答的是 sim2sim 验证最核心的问题：两边的观测【构造】是否一致。
    # 直接比 obs 逐元素没意义（状态本来就不同），比【各块的量级/缩放】才有意义。
    if args.blocks and 'obs' in a_arr and 'obs' in b_arr and a_arr['obs'].shape[1] >= 45:
        m = min(n, args.block_steps)
        blk = [('commands*scale', 0, 3), ('base_ang_vel*0.25', 3, 6), ('projected_gravity', 6, 9),
               ('dof_pos-default', 9, 21), ('dof_vel*0.05', 21, 33), ('last_actions', 33, 45)]
        print(f"\n  ---- 观测语义分块（第0帧=最新帧，前 {m} 步平均）----")
        for name, i0, i1 in blk:
            x = a_arr['obs'][:m, i0:i1].mean(axis=0)
            y = b_arr['obs'][:m, i0:i1].mean(axis=0)
            print(f"  {name:<18} A={np.array2string(x, precision=3, floatmode='fixed', max_line_width=200)}")
            print(f"  {'':<18} B={np.array2string(y, precision=3, floatmode='fixed', max_line_width=200)}")
        print("\n  ---- 历史帧方向自检（滑动填充时，越旧的帧'全零占比'越高）----")
        for tag, D in (('A', a_arr['obs'][:m]), ('B', b_arr['obs'][:m])):
            zr = [f'帧{f}:{(D[:, f*45:(f+1)*45] == 0).mean()*100:.1f}%' for f in range(6)]
            print(f"  [{tag}] " + '  '.join(zr))
        print("      两边应同为递增。若一个递增一个递减 → 历史帧顺序不一致（--frame-order 设反了）")

    # ---- 健康度：用 projected_gravity z 判定有没有倒 ----
    if 'obs' in a_arr and 'obs' in b_arr and a_arr['obs'].shape[1] >= 9:
        print(f"\n  ---- 健康度（obs[8] = projected_gravity_z，≈-1 站立 / >-0.5 基本躺平）----")
        skip = 10   # ⚠️ 跳过开头：两边最初几步的历史缓冲仍是全零，gz=0 会被误判成"已倒"
        for tag, o in (('A', a_arr['obs'][:n]), ('B', b_arr['obs'][:n])):
            gz = o[skip:, 8]
            bad = np.nonzero(gz > -0.7)[0]
            t_fall = f'{(bad[0] + skip) * 0.01:.2f}s' if bad.size else '从未（全程站立）'
            print(f"  [{tag}] gz 末值 {gz[-1]:+.3f}   首次 >-0.7: {t_fall}   "
                  f"(跳过前 {skip} 步的零缓冲)")

    if 'action' in a_arr and 'action' in b_arr:
        d = np.abs(a_arr['action'][:n] - b_arr['action'][:n])
        print(f"\n  ★ 动作最大绝对差 = {d.max():.6g}   平均 {d.mean():.6g}")
        if d.max() < 1e-3:
            print("     → 两边策略输出几乎一致（观测对齐良好）")
        else:
            print("     → 差异显著。逐关节平均差（按 BLOCKS 语义看哪一组最可疑）:")
            print(f"       {np.array2string(d.mean(axis=0), precision=3, floatmode='fixed')}")
            print("       优先排查：关节顺序映射 / 观测缩放 / 历史帧顺序 / 默认关节角 / PD 增益")


def main():
    import argparse
    p = argparse.ArgumentParser(description='策略输入输出记录器 / 查看 / 对比')
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list', help='列出已记录的会话').set_defaults(func=cmd_list)
    s = sub.add_parser('show', help='查看单个记录'); s.add_argument('path'); s.set_defaults(func=cmd_show)
    s = sub.add_parser('compare', help='对比两份记录'); s.add_argument('a'); s.add_argument('b')
    s.add_argument('--no-blocks', dest='blocks', action='store_false',
                   help='不做观测语义分块对比（默认做）')
    s.add_argument('--block-steps', type=int, default=300,
                   help='语义分块统计用前多少步（默认 300；两边指令不同时可调小）')
    s.set_defaults(func=cmd_compare, blocks=True)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
