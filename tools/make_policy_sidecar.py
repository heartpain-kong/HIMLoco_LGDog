# -*- coding: utf-8 -*-
"""make_policy_sidecar.py — 给 exported/policies/ 里的某份策略生成 sidecar `policy.json`。

为什么需要它（2026-10-10 实测踩到）:
  收尾链导出的策略只有 .pt、没有 sidecar json。部署侧 `tools/check_alignment.py` 与
  sim2real 的 main.py 都是**按策略包 policy.json 取契约**（kp/kd/action_scale/decimation/
  默认站姿/观测布局）——读不到 sidecar 就退回 spec.py 的模块默认值（46 维 / kp30），
  于是对 45 维 kp40 的策略必然报「❌ 1 项不一致：KP」。这不是真的漂移，但**看着像**，
  而且真上机时 main.py 会按错误默认值跑。

  ⇒ 每份要拿出去用的导出策略，旁边都该有一份自带契约的 sidecar。

用法:
  python tools/make_policy_sidecar.py <run目录名或路径> <step> [策略文件名]
  例: python tools/make_policy_sidecar.py V47 4200 policy_V47_4200.pt
       # 省略第三参数时按 policy_V<run>_<step>.pt 推断
  产物: legged_gym/logs/flat_LgDog1/exported/policies/<策略名>.json
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'rsl_rl'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('ec', os.path.join(ROOT, 'tools', 'export_checkpoint.py'))
_ec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ec)
import torch  # noqa: E402

POLROOT = os.path.join(ROOT, 'legged_gym', 'logs', 'flat_LgDog1', 'exported', 'policies')
RUNROOT = os.path.join(ROOT, 'legged_gym', 'logs', 'flat_LgDog1')


def sidecar(run, step, name=None):
    rundir = run if os.path.isdir(run) else os.path.join(RUNROOT, run)
    if not os.path.isdir(rundir):
        raise SystemExit('找不到 run 目录: %s' % rundir)
    runname = os.path.basename(rundir.rstrip('/'))
    name = name or ('policy_%s_%s.pt' % (re.sub(r'\W+', '', runname) if False else runname, step))
    if name is None:
        raise SystemExit('请给出策略文件名')
    pt = os.path.join(POLROOT, name)
    if not os.path.isfile(pt):
        raise SystemExit('找不到策略文件: %s' % pt)

    snap_p = os.path.join(rundir, 'config_snapshot.json')
    snap = json.load(open(snap_p, encoding='utf-8')) if os.path.isfile(snap_p) else {}
    e = snap.get('env_cfg', snap)
    ctrl = e.get('control', {}) or {}
    norm = e.get('normalization', {}) or {}
    init = e.get('init_state', {}) or {}
    n_one = int((e.get('env') or {}).get('num_one_step_observations', 45))

    cc = {
        'obs_one_step': n_one,
        'obs_history_len': 6,
        'obs_history_span_ms': 100.0,
        'decimation': int(ctrl.get('decimation', 4)),
        'control_freq_hz': round(1.0 / (float((e.get('sim') or {}).get('dt', 0.005))
                                        * int(ctrl.get('decimation', 4))), 1),
        'kp': (ctrl.get('stiffness') or {}).get('joint'),
        'kd': (ctrl.get('damping') or {}).get('joint'),
        'action_scale': ctrl.get('action_scale'),
        'clip': norm.get('clip_observations'),
        'obs_scales': {k: v for k, v in (norm.get('obs_scales') or {}).items()
                       if k in ('lin_vel', 'ang_vel', 'dof_pos', 'dof_vel')},
    }
    dja = init.get('default_joint_angles')
    if isinstance(dja, dict):
        order = ['%s_%s_joint' % (leg, j) for leg in ('FL', 'FR', 'RL', 'RR')
                 for j in ('hip', 'thigh', 'calf')]
        cc['default_dof_pos'] = [round(float(dja[k]), 4) for k in order if k in dja]
    if n_one == 46:
        cmds = e.get('commands', {}) or {}
        cc['height_nominal'] = cmds.get('height_nominal')

    data = open(pt, 'rb').read()
    fp = _ec.weight_fingerprint(torch.jit.load(pt))
    out = {
        'weight_fingerprint': fp,
        'name': '%s @iteration %s（收尾链导出）' % (runname, step),
        'source_run': 'LgRobotDog/legged_gym/logs/flat_LgDog1/%s' % runname,
        'source_checkpoint': 'model_%s.pt' % step,
        'source_file': 'exported/policies/%s' % name,
        'file_size': len(data),
        'md5_file': hashlib.md5(data).hexdigest(),
        'md5_note': '文件 md5 只用于比对"是不是同一个文件"；判断"是不是同一份权重"一律用 weight_fingerprint',
        'io': '%d 维（%d 单帧 × 6 历史）→ 12 维动作' % (n_one * 6, n_one),
        'control_contract': cc,
        'sidecar_note': ('本文件由 tools/make_policy_sidecar.py 从该 run 的 config_snapshot.json 自动生成；'
                         '没有它，部署侧 check_alignment.py / main.py 会退回 spec.py 的模块默认契约，'
                         '对 45 维/kp40 的策略会误报「KP 不一致」。'),
    }
    outp = os.path.join(POLROOT, name.rsplit('.pt', 1)[0] + '.json')
    json.dump(out, open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('[write] %s' % outp)
    print('  指纹 = %s   kp/kd/scale/decim = %s/%s/%s/%s   obs = %d×6'
          % (fp, cc['kp'], cc['kd'], cc['action_scale'], cc['decimation'], n_one))
    return outp


if __name__ == '__main__':
    import re  # noqa: E402  （放在这里避免与上面的 lambda 冲突）
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sidecar(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
