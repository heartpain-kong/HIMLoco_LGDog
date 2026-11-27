# -*- coding: utf-8 -*-
"""export_checkpoint.py — 把训练 checkpoint 导出成 JIT 策略（含观测归一化统计量还原）。

为什么需要它（而不是直接用 play.py 的导出）：
  play.py 只能导出【它自己跑起来的那次】策略；离线导出指定 checkpoint 需要手工构造网络。
  观测经验归一化的统计量在 2026-09-25 之后由 him_on_policy_runner.save() 存进 ckpt
  （'obs_norm_stats' 键）——本脚本会把它还原并冻结烘焙进导出图，保证 sim2sim/play 的
  输入分布与训练一致（V18 那次因为统计量丢失只能用恒等统计量导出，策略直接瘫了）。

用法:
  python tools/export_checkpoint.py <run_dir> <step> [输出名] [--no-policy-pt]
  例: python tools/export_checkpoint.py legged_gym/logs/flat_LgDog1/V18 8400 policy_V18_8400.pt
输出: <run_dir>/../exported/policies/<输出名>（并默认同时覆盖 policy.pt）
"""
import argparse, copy, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'rsl_rl'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules.him_actor_critic import HIMActorCritic

# 网络维度（与 LgDog1Cfg / snapshot 一致）
# 【2026-10-01 身高指令通道】单步观测 45→46（新增第 4 维指令=目标身高）：
#   NUM_OBS 46×6=276、NUM_CRITIC_OBS 46+3+3+187=239。旧 270 维 ckpt 与本维度不兼容。
# ★ 但仓库里 Oct-01 之前的老 run（V28/V6_h 等 15 个 run）仍是 45 维（270 维）——
#   所以下面这些在 main() 里会【按 ckpt 的 actor 输入维度自动判定】，这里的字面量只是兜底：
#     actor 输入 = 单帧观测 + 3(估计速度) + 16(潜变量)  ⇒ 单帧 = actor_in − 19
NUM_OBS, NUM_CRITIC_OBS, NUM_ONE_STEP, NUM_ACTIONS = 276, 239, 46, 12
HISTORY_LEN = 6
TAIL_DIM = 19                     # 估计器输出的 3 速度 + 16 潜变量，接在单帧观测之后
HIDDEN = [512, 256, 128]


class PolicyExporterHIM(nn.Module):
    """与 legged_gym/utils/helpers.py 同一实现（不 import isaacgym，便于离线导出）。"""

    def __init__(self, actor_critic, num_one_step: int = NUM_ONE_STEP):
        super().__init__()
        # TorchScript 不能闭包全局 int ⇒ 存成模块属性（45 维老 run 也要能导出）
        self.num_one_step = int(num_one_step)
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        norm = actor_critic.obs_normalizer
        self.register_buffer("norm_mean", copy.deepcopy(norm.running_ms.mean).cpu())
        self.register_buffer("norm_var", copy.deepcopy(norm.running_ms.var).cpu())

    def forward(self, obs_history):
        obs = (obs_history - self.norm_mean) / torch.sqrt(self.norm_var + 1e-4)
        parts = self.estimator(obs)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        # 单帧观测 45（老，无身高通道）或 46（2026-10-01 起，含身高指令通道）
        return self.actor(torch.cat((obs[:, 0:self.num_one_step], vel, z), dim=1))


def weight_fingerprint(mod):
    h = hashlib.md5()
    sd = mod.state_dict()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('step', type=int)
    ap.add_argument('out_name', nargs='?', default=None)
    ap.add_argument('--no-policy-pt', action='store_true', help='不覆盖 policy.pt')
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    step = args.step
    ckpt = os.path.join(run_dir, f'model_{step}.pt')
    if not os.path.isfile(ckpt):
        sys.exit(f'找不到 checkpoint: {ckpt}')
    out_dir = os.path.join(os.path.dirname(run_dir), 'exported', 'policies')
    os.makedirs(out_dir, exist_ok=True)
    out_name = args.out_name or f'policy_{os.path.basename(run_dir)}_{step}.pt'

    ck = torch.load(ckpt, map_location='cpu')
    # ★2026-09-29 自适应 critic 观测维度: 平地版任务去掉了 187 维高度扫描 ⇒ 238→51。
    #   直接从权重形状读，避免硬编码（否则导出平地版 checkpoint 会 shape mismatch）。
    _sd = ck['model_state_dict']
    critic_in = int(_sd['critic.0.weight'].shape[1]) if 'critic.0.weight' in _sd else NUM_CRITIC_OBS
    actor_in = int(_sd['actor.0.weight'].shape[1]) if 'actor.0.weight' in _sd else None
    if critic_in != NUM_CRITIC_OBS:
        print(f"[dims] 从 ckpt 读到 critic 观测维度 = {critic_in}（默认 {NUM_CRITIC_OBS}）")

    # ★ 单帧观测维度也自适应：actor 输入 = 单帧 + 19（3 速度 + 16 潜变量）。
    #   Oct-01 之前的老 run 是 45 维（270），之后是 46 维（276，多了身高指令通道）。
    #   写死 46 会让老 run 导出直接 shape mismatch（仓库里 25 个 run 有 15 个是 45 维）。
    n_one_step = NUM_ONE_STEP
    if actor_in is not None:
        if actor_in - TAIL_DIM not in (45, 46):
            sys.exit(f'[dims] ckpt 的 actor 输入维度 = {actor_in}，推出单帧观测 '
                     f'{actor_in - TAIL_DIM} 维 —— 既不是 45 也不是 46，拒绝导出'
                     f'（{ckpt} 可能不是本任务的权重）')
        n_one_step = actor_in - TAIL_DIM
    n_obs = n_one_step * HISTORY_LEN
    if n_one_step != NUM_ONE_STEP:
        print(f"[dims] ckpt 是【{n_one_step} 维单帧观测】（{n_obs} 维历史）"
              f"—— 老布局，无身高指令通道")
    else:
        print(f"[dims] {n_one_step} 维单帧观测（{n_obs} 维历史，含身高指令通道 obs[3]）")

    ac = HIMActorCritic(n_obs, critic_in, n_one_step, NUM_ACTIONS, HIDDEN, HIDDEN, 'elu')
    ac.load_state_dict(ck['model_state_dict'], strict=True)
    ac.eval()
    print(f"[load] {ckpt} (iter={ck.get('iter')}) 载入成功")

    stats = ck.get('obs_norm_stats', None)
    if stats is None:
        print("[warn] ckpt 无 obs_norm_stats ⇒ 用恒等统计量(mean=0,var=1)导出；"
              "若该 run 训练时启用了观测归一化，导出策略将处于 OOD（V18 的教训）")
    else:
        ac.obs_normalizer.running_ms.mean = stats['mean'].cpu()
        ac.obs_normalizer.running_ms.var = stats['var'].cpu()
        ac.obs_normalizer.running_ms.n = stats['n']
        m, v = stats['mean'], stats['var']
        print(f"[norm] 已从 ckpt 还原观测归一化统计量 (n={stats['n']:.0f}); "
              f"mean 范围[{m.min():+.3f},{m.max():+.3f}] std 范围[{v.sqrt().min():.3f},{v.sqrt().max():.3f}]")

    exp = PolicyExporterHIM(ac, n_one_step).eval()
    jit = torch.jit.script(exp)

    x = torch.randn(8, n_obs)
    with torch.no_grad():
        _m = ac.obs_normalizer.running_ms.mean.cpu()
        _v = ac.obs_normalizer.running_ms.var.cpu()
        obs_n = (x - _m) / torch.sqrt(_v + 1e-4)
        parts = ac.estimator.encoder(obs_n)[:, 0:19]  # 估计器输出 19 维(3 速度 + 16 潜变量)，与维度改动无关
        ref = ac.actor(torch.cat((obs_n[:, :n_one_step], parts[..., :3],
                                  F.normalize(parts[..., 3:], dim=-1, p=2.0)), dim=1))
        out = jit(x)
    assert out.shape == (8, NUM_ACTIONS) and torch.isfinite(out).all()
    print(f"[verify] JIT {n_obs}→{NUM_ACTIONS} 形状OK 有限值OK vs 手工前向 "
          f"max|Δ|={(out - ref).abs().max().item():.2e}")

    names = [out_name] + ([] if args.no_policy_pt else ['policy.pt'])
    for name in names:
        p = os.path.join(out_dir, name)
        jit.save(p)
        md5 = hashlib.md5(open(p, 'rb').read()).hexdigest()
        print(f"[export] {p}  size={os.path.getsize(p)//1024}KB  md5={md5}")
    fp = [weight_fingerprint(torch.jit.load(os.path.join(out_dir, n))) for n in names]
    print(f"[fingerprint] {' / '.join(names)} = {fp}  一致={len(set(fp)) == 1}")
    _print_contract(run_dir, n_one_step, n_obs, ck)
    print("ALL OK")


def _print_contract(run_dir, n_one_step, n_obs, ck):
    """打印可直接粘贴进 policy.json 的 control_contract。

    ★ 为什么要有这一步：sim2real 的契约（kp/kd/action_scale/身高指令基准…）以前靠人手抄，
      抄错【不会报错、只会让策略行为不可信】——项目里 action_scale 0.25/0.5 错配作废过
      三个 60000 步的测试；height_nominal 也漂过一次（0.28 vs 训练的 0.30，一传 --height
      就把身高指令放大一倍）。所以导出时直接从该 run 的训练配置里生成一份，别再手抄。
    """
    snap = os.path.join(run_dir, 'config_snapshot.json')
    if not os.path.isfile(snap):
        print(f"[contract] （{run_dir} 里没有 config_snapshot.json，跳过契约生成）")
        return
    try:
        d = json.load(open(snap, encoding='utf-8'))
    except Exception as e:
        print(f"[contract] config_snapshot.json 读不了: {e}")
        return
    e_ = d.get('env_cfg', d)
    ctrl = e_.get('control', {}) or {}
    norm = e_.get('normalization', {}) or {}
    cmds = e_.get('commands', {}) or {}
    init = e_.get('init_state', {}) or {}
    cc = {
        'obs_one_step': int(n_one_step),
        'obs_history_len': int(HISTORY_LEN),
        # 6 帧历史跨度 = (6−1) × 控制周期，50Hz 下是 100 ms（不是 6 × 20 = 120 ms）
        'obs_history_span_ms': round(1000.0 * (HISTORY_LEN - 1)
                                     * float((e_.get('sim') or {}).get('dt', 0.005))
                                     * int(ctrl.get('decimation', 4)), 1),
        'decimation': int(ctrl.get('decimation', 4)),
        'control_freq_hz': round(1.0 / (float((e_.get('sim') or {}).get('dt', 0.005))
                                        * int(ctrl.get('decimation', 4))), 1),
        'kp': (ctrl.get('stiffness') or {}).get('joint'),
        'kd': (ctrl.get('damping') or {}).get('joint'),
        'action_scale': ctrl.get('action_scale'),
        'clip': norm.get('clip_observations'),
        'obs_scales': {k: v for k, v in (norm.get('obs_scales') or {}).items()
                       if k in ('lin_vel', 'ang_vel', 'dof_pos', 'dof_vel')},
    }
    if n_one_step == 46:
        cc['height_nominal'] = cmds.get('height_nominal')
        cc['height_cmd_scale'] = norm.get('height_cmd')
    dja = init.get('default_joint_angles')
    if isinstance(dja, dict):
        order = [f'{leg}_{j}_joint' for leg in ('FL', 'FR', 'RL', 'RR')
                 for j in ('hip', 'thigh', 'calf')]
        if all(k in dja for k in order):
            cc['default_dof_pos'] = [round(float(dja[k]), 4) for k in order]
        else:  # 关节命名可能不同，按快照里的顺序兜底
            cc['default_dof_pos'] = [round(float(v), 4) for v in dja.values()]
    print('[contract] 这份策略的 control_contract（★ 直接拷进 policy.json，别手抄）:')
    for ln in json.dumps(cc, ensure_ascii=False, indent=2).splitlines():
        print('    ' + ln)
    if n_one_step == 46:
        print(f"    # obs[3] = (目标身高 − {cc.get('height_nominal')}) × "
              f"{cc.get('height_cmd_scale')}；不给 --height 时该维恒 0")


if __name__ == '__main__':
    main()
