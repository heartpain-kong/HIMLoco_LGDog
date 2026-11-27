#!/usr/bin/env python3
"""LgDog1 策略运行时 —— 270 维观测历史 → 12 维动作

★ 与 sim2sim.py 严格同口径。逐行对应关系：
    sim2sim.py:build_obs   -> self.build_obs()
    sim2sim.py:362-364     -> self.reset() 的零历史
    sim2sim.py:498-508     -> step() 里的 history_bootstrap 分支
    sim2sim.py:511-534     -> 裁剪 / last_action / target_q 计算

★ 为什么历史帧不能省：导出的 policy.pt 是 PolicyExporterHIM
  （legged_gym/utils/helpers.py），内部含 HIM 估计器：
      parts = estimator(obs_history)[:, 0:19]; vel = parts[0:3]; z = normalize(parts[3:19])
      actor(concat(obs_history[:, 0:45], vel, z))
  观测里【没有】base_lin_vel —— 它完全靠这 6 帧历史推断出来。
  历史构造错一点（顺序/帧数/首帧）→ 估计速度就错 → 策略直接 OOD。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec  # noqa: E402
from tools.math_utils import quat_rotate_inverse  # noqa: E402


class PolicyRunner:
    """加载 TorchScript 策略并维护观测历史。"""

    def __init__(self, policy_path: str, action_scale=None, device: str = "cpu"):
        import torch  # 延迟导入，便于在没装 torch 的机器上跑静态检查

        self.torch = torch
        self.device = device
        if not os.path.isfile(policy_path):
            raise FileNotFoundError("找不到策略文件: %s" % policy_path)
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()
        self.policy_path = policy_path

        # action_scale：默认取 spec（与训练一致）。显式传值只用于 A/B，会有配错风险。
        if action_scale is None:
            self.action_scale = np.array(spec.ACTION_SCALE, dtype=np.float64)
        else:
            self.action_scale = np.asarray(action_scale, dtype=np.float64)
            if self.action_scale.ndim == 0:
                self.action_scale = np.full(spec.NUM_JOINTS, float(self.action_scale))
            assert self.action_scale.shape == (spec.NUM_JOINTS,), "action_scale 维度必须是 12"

        self._adopt_obs_layout()
        self._check_policy_io()
        self.weight_fp = self.weight_fingerprint()
        self.reset()

    def _adopt_obs_layout(self):
        """按【策略旁边那份 policy.json】声明的观测布局切换 spec 的 45/46 维设置。

        ★ 为什么放在这里：布局（有没有身高指令通道）是策略契约的一部分，而且它决定
          网络输入维度。任何用到 PolicyRunner 的地方（main.py / 自测 / 别的工具）
          都必须用对，所以让 runner 自己认领，而不是指望调用方记得先切。
          没有 policy.json 时不动（保持 spec 默认）。
        """
        side = os.path.join(os.path.dirname(os.path.abspath(self.policy_path)), "policy.json")
        if not os.path.isfile(side):
            return
        try:
            import json as _json
            cc = (_json.load(open(side, encoding="utf-8")) or {}).get("control_contract") or {}
            n = cc.get("obs_one_step")
            if n and spec.set_obs_layout(int(n)):
                print("[策略契约] 切到 %d 维单帧观测（总 %d 维，身高指令通道=%s）"
                      % (spec.NUM_ONE_STEP_OBS, spec.NUM_OBS,
                         "有" if spec.HEIGHT_CMD_ENABLED else "无"))
        except Exception as e:
            print("⚠️  读 %s 的 obs 布局失败（按 spec 默认继续）: %s" % (side, e))

    # ------------------------------------------------------------------ 自检
    def _check_policy_io(self):
        """用全零输入探测策略的输入/输出维度，不匹配立刻报错（宁可在启动时炸）。"""
        try:
            with self.torch.no_grad():
                out = self.policy(self.torch.zeros(1, spec.NUM_OBS))
            shape_in = spec.NUM_OBS
            shape_out = int(out.shape[-1])
        except Exception as e:  # pragma: no cover
            raise RuntimeError("策略加载自检失败（不是 270 维输入？）: %s" % e)
        if shape_out != spec.NUM_JOINTS:
            raise RuntimeError(
                "策略输出 %d 维，期望 %d 维。这个 policy.pt 不是给 LgDog1 12 关节用的。"
                % (shape_out, spec.NUM_JOINTS)
            )
        self.policy_in_dim = shape_in
        self.policy_out_dim = shape_out

    # ------------------------------------------------------------------ 身份
    def weight_fingerprint(self, n: int = 12) -> str:
        """策略的【权重指纹】：按参数名排序后对张量字节做 md5。

        ★ 为什么不用文件 md5 当身份：同一份权重重新导出一次，torch.jit 的 zip 里
          `policy/.data/serialization_id` 和自动生成的 `___torch_mangle_N` 源码条目
          每次序列化都不同 → 文件 md5 变了而权重一模一样。
          （legged_gym/scripts/sim2sim.py 里踩过这个坑，注释写得很清楚：
           policy.pt 与 policy_V13_6000.pt 文件 md5 不同、权重指纹都是 e4fee6f97eaf。）
        部署时判断"机器人上跑的到底是哪一版"只能看这个值。
        """
        try:
            sd = self.policy.state_dict()
        except Exception:
            return "unavailable"
        import hashlib
        h = hashlib.md5()
        for k in sorted(sd):
            h.update(k.encode())
            h.update(sd[k].detach().cpu().numpy().tobytes())
        return h.hexdigest()[:n]

    # ------------------------------------------------------------------ 状态
    def reset(self):
        """复位历史。

        ★ 与 sim2sim.py:362-364 完全一致：历史 = HISTORY_LEN 帧【全零】，
          且【第一次】策略调用就用这个零历史（history_bootstrap=True 时不再追加）。
          对应 Isaac play.py 在 reset 后的第一帧观测。不要用站立帧预热。
        """
        self.hist = [np.zeros(spec.NUM_ONE_STEP_OBS, dtype=np.float32)
                     for _ in range(spec.HISTORY_LEN)]
        self.history_bootstrap = True
        self.last_action = np.zeros(spec.NUM_JOINTS, dtype=np.float64)
        self.step_count = 0

    # ------------------------------------------------------------------ 观测
    def build_obs(self, q, dq, quat_wxyz, omega, cmd, height_cmd=None) -> np.ndarray:
        """构造单帧 45 维观测。逐行对应 sim2sim.py:build_obs。

        q    : 12 关节角 [rad]，Isaac 顺序（spec.JOINT_LABELS）
        dq   : 12 关节速度 [rad/s]，同上
        quat : 机身姿态四元数 (w,x,y,z)，机体系相对世界系
        omega: 机体系角速度 [rad/s]
        cmd  : (vx, vy, yaw_rate) 未缩放
        height_cmd : 目标站高 [m]（只有 46 维布局才用；None = 名义值 → 该维恒 0）

        ★ 各段下标一律从 spec.OBS 取，不在这里写死数字 ——
          布局有两套（45 维无身高通道 / 46 维有），写死必然在切换时错位。
        """
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        cmd = np.asarray(cmd, dtype=np.float64)
        assert q.shape == (spec.NUM_JOINTS,) and dq.shape == (spec.NUM_JOINTS,)

        pg = quat_rotate_inverse(np.asarray(quat_wxyz, dtype=np.float64),
                                 np.array([0.0, 0.0, -1.0]))
        o = spec.OBS
        obs = np.zeros(spec.NUM_ONE_STEP_OBS, dtype=np.float32)
        obs[o["cmd"]] = cmd[:3] * np.array([spec.LV, spec.LV, spec.AV])
        if o["height"] is not None:
            h = spec.HEIGHT_NOMINAL if height_cmd is None else float(height_cmd)
            obs[o["height"]] = (h - spec.HEIGHT_NOMINAL) * spec.HC
        obs[o["omega"]] = np.asarray(omega, dtype=np.float64) * spec.AV
        obs[o["pg"]] = pg
        obs[o["q"]] = (q - spec.DEFAULT_DOF_POS) * spec.DP
        obs[o["dq"]] = dq * spec.DV
        obs[o["act"]] = self.last_action
        return np.clip(obs, -spec.CLIP, spec.CLIP).astype(np.float32)

    def flatten_history(self) -> np.ndarray:
        """拼成 270 维。newest 语义 = hist[0] 是最新帧（与 sim2sim 一致）。"""
        return np.concatenate(self.hist, axis=0).astype(np.float32)

    # ------------------------------------------------------------------ 推理
    def step(self, q, dq, quat_wxyz, omega, cmd, height_cmd=None):
        """一个控制周期：更新历史 → 推理 → 返回 (action, one_step_obs, obs_history)。

        action 是【原始策略输出】（已裁剪到 ±CLIP），未乘 ACTION_SCALE，
        与 sim2sim 记录的 action 同口径，便于直接对比排查。
        """
        cmd = np.asarray(cmd, dtype=np.float64).copy()
        # ★ 指令死区：sim2sim.py:496，小速度指令归零（策略在该规则下评测）
        if np.hypot(cmd[0], cmd[1]) < spec.CMD_DEADZONE_XY:
            cmd[0] = 0.0
            cmd[1] = 0.0

        one_step = None
        if not self.history_bootstrap:
            one_step = self.build_obs(q, dq, quat_wxyz, omega, cmd, height_cmd)
            if spec.FRAME_ORDER == "newest":
                self.hist.insert(0, one_step)
                self.hist.pop()
            else:
                self.hist.pop(0)
                self.hist.append(one_step)

        oh = self.flatten_history()
        with self.torch.no_grad():
            a = self.policy(self.torch.from_numpy(oh).unsqueeze(0)).numpy().flatten()
        a = np.clip(a, -spec.CLIP, spec.CLIP).astype(np.float64)

        self.last_action = a.copy()
        self.history_bootstrap = False
        self.step_count += 1
        return a, one_step, oh

    # ------------------------------------------------------------------ 目标
    def target_from_action(self, action) -> np.ndarray:
        """动作 → 关节目标角。sim2sim.py:534: target = a * ACTION_SCALE + DEFAULT。"""
        return np.asarray(action, dtype=np.float64) * self.action_scale + spec.DEFAULT_DOF_POS


def _selftest():
    """不接硬件，直接喂零状态跑几步，确认策略能加载且输出合理。"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    args = ap.parse_args()

    # 查找顺序与 main.py 一致：随包 models/policy.pt 优先，其次训练仓库
    # ★ 目标机(N100)上【没有】训练仓库，所以随包策略必须是第一顺位 ——
    #   否则这个自测在真机上会因为找不到 <LgRobotDog>/legged_gym/... 而假失败。
    _sim2real = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _cands = [
        os.path.join(_sim2real, "models", "policy.pt"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_sim2real))),
                     "legged_gym", "logs", "flat_LgDog1", "exported", "policies", "policy.pt"),
    ]
    default_policy = next((c for c in _cands if os.path.isfile(c)), _cands[0])
    path = args.policy or default_policy
    if not os.path.isfile(path):
        print("找不到策略文件: %s" % path)
        print("  目标机上请把策略放到 %s/models/policy.pt" % _sim2real)
        return 1
    print("加载策略:", path)

    r = PolicyRunner(path)
    print("输入 %d 维 / 输出 %d 维 ✓" % (r.policy_in_dim, r.policy_out_dim))
    print("action_scale =", r.action_scale)
    print()
    print(spec.obs_layout_report())
    print()

    q = spec.DEFAULT_DOF_POS.copy()
    dq = np.zeros(spec.NUM_JOINTS)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.zeros(3)
    for i in range(4):
        cmd = np.array([1.0, 0.0, 0.0]) if i >= 2 else np.zeros(3)
        a, one_step, oh = r.step(q, dq, quat, omega, cmd)
        tgt = r.target_from_action(a)
        print("step %d cmd=%s  hist_boot=%s  |a|max=%.3f  |target-default|max=%.3f"
              % (i, np.round(cmd, 2), i == 0, np.abs(a).max(), np.abs(tgt - spec.DEFAULT_DOF_POS).max()))
        if one_step is not None:
            nz = np.nonzero(one_step)[0]
            print("       单帧观测非零段: %s" % (nz[:6].tolist() + (["..."] if len(nz) > 6 else [])))
    print()
    print("零历史首步动作（应与 sim2sim 第一帧一致）: ", np.round(a, 4))
    print("✅ policy_runner 自测通过")
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_selftest())
