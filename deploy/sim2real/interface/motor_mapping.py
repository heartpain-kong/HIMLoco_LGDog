#!/usr/bin/env python3
"""真机硬件映射：策略关节顺序 ↔ 电机 (bus, can_id, sign, zero_offset)

设计要点：
  * 表放在外部 `hardware_map.yaml`，改接线不用动代码。
  * `verified: false` 时**拒绝进入策略控制**（Sim2RealMapNotVerified），
    只允许零位标定 / 单关节测试这类低速动作。理由：CAN ID 或方向填错，
    策略一上电就会把关节往反方向猛推，烧电机或摔机器。
  * 换算公式（在 RC_WheelLeg 的基础上多了一级减速比）：
        下发: motor = sign * (sim * gear_ratio) + zero_offset
        回读: sim   = (motor - zero_offset) / sign / gear_ratio
        速度: motor_vel = sign * sim_vel * gear_ratio
    gear_ratio = 关节转 1 圈需要电机转几圈。URDF 里 calf 的速度上限(9.425)正好是
    hip/thigh(18.85) 的一半 → 小腿有 2:1 减速，gear_ratio 必须是 2.0。
    ⚠️ 这一条是从 URDF 推出来的，上电前请手动盘一下确认。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec  # noqa: E402

DEFAULT_MAP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hardware_map.yaml")


class Sim2RealMapNotVerified(RuntimeError):
    """映射表还没人工确认就想进策略控制 —— 直接拦住。"""


class MotorMapping:
    """12 个关节的硬件映射。关节索引一律指【策略顺序】(spec.JOINT_LABELS)。"""

    def __init__(self, map_path: str = None):
        import yaml  # 只在需要时依赖

        # ★ map_path=None 视作"用默认路径"：调用方常常是
        #   MotorMapping(cfg.get("hardware_map") or None)，传 None 本意就是"没配就用默认"。
        #   旧写法会把 None 直接塞进 os.path.isfile 而抛 TypeError。
        if map_path is None:
            map_path = DEFAULT_MAP_PATH
        self.path = map_path
        if not os.path.isfile(map_path):
            raise FileNotFoundError("找不到硬件映射表: %s" % map_path)
        with open(map_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.verified = bool(cfg.get("verified", False))
        self.gain_correction = bool(cfg.get("gain_correction", True))
        self.motor_model = cfg.get("motor_model", "rs-02")
        kp_r = cfg.get("mit_kp_range", [0.0, 500.0])
        kd_r = cfg.get("mit_kd_range", [0.0, 5.0])
        self.mit_kp_range = (float(kp_r[0]), float(kp_r[1]))
        self.mit_kd_range = (float(kd_r[0]), float(kd_r[1]))

        joints = cfg.get("joints") or {}
        missing = [n for n in spec.JOINT_LABELS if n not in joints]
        extra = [n for n in joints if n not in spec.JOINT_LABELS]
        if missing:
            raise ValueError("hardware_map.yaml 缺少关节: %s" % ", ".join(missing))
        if extra:
            raise ValueError("hardware_map.yaml 有多余关节（名字拼错？）: %s" % ", ".join(extra))

        self.bus = np.zeros(spec.NUM_JOINTS, dtype=int)
        self.can_id = np.zeros(spec.NUM_JOINTS, dtype=int)
        self.sign = np.zeros(spec.NUM_JOINTS, dtype=np.float64)
        self.gear_ratio = np.ones(spec.NUM_JOINTS, dtype=np.float64)
        self.zero_offset = np.zeros(spec.NUM_JOINTS, dtype=np.float64)
        for i, name in enumerate(spec.JOINT_LABELS):
            j = joints[name]
            self.bus[i] = int(j["bus"])
            self.can_id[i] = int(j["can_id"])
            self.sign[i] = float(j["sign"])
            self.gear_ratio[i] = float(j.get("gear_ratio", 1.0))
            self.zero_offset[i] = float(j["zero_offset"])

        # ---- 下位机固件的电机软限位 ----
        # ★ 2026-09-27 起固件已删除这 12 个软限位（见 hardware_map.yaml 的说明）。
        #   board_limits_active=false 时本节自检会跳过而不是判失败；
        #   默认 True 是为了兼容"还在跑旧固件"的板子。
        # 没有这一节时相关自检会跳过（不报错），但会少掉一个独立参考系
        self.board_limits_active = bool(cfg.get("board_limits_active", True))
        self.board_limits_deg = cfg.get("board_limits_deg") or {}
        self.board_lo = np.full(spec.NUM_JOINTS, np.nan)
        self.board_hi = np.full(spec.NUM_JOINTS, np.nan)
        for i, name in enumerate(spec.JOINT_LABELS):
            lim = self.board_limits_deg.get(name)
            if lim:
                self.board_lo[i] = np.radians(float(lim[0]))
                self.board_hi[i] = np.radians(float(lim[1]))

        self._validate()

    def _validate(self):
        if not np.all(np.isin(self.sign, (-1.0, 1.0))):
            raise ValueError("sign 只能是 +1 或 -1，实际: %s" % self.sign)
        if len(set(self.can_id.tolist())) != spec.NUM_JOINTS:
            # ★ 唯一性必须按 (bus, can_id) 组合判断，不能只看 can_id ——
            #   LgDog1 是【两条 CAN × 每条 6 个电机】，can_id 1..6 会在两条总线上各出现一次。
            pairs = list(zip(self.bus.tolist(), self.can_id.tolist()))
            if len(set(pairs)) != spec.NUM_JOINTS:
                dup = sorted({p for p in pairs if pairs.count(p) > 1})
                raise ValueError("(bus, can_id) 有重复: %s" % dup)
        if np.any(self.can_id <= 0):
            raise ValueError("can_id 必须为正整数: %s" % self.can_id)
        if np.any(self.bus < 0):
            raise ValueError("bus 必须 >= 0: %s" % self.bus)
        if np.any(self.gear_ratio <= 0):
            raise ValueError("gear_ratio 必须为正: %s" % self.gear_ratio)
        # URDF 证据自检：calf 的速度上限是 hip/thigh 的一半 → 减速比应为 2
        for i, name in enumerate(spec.JOINT_LABELS):
            if name.endswith("_calf") and abs(self.gear_ratio[i] - 2.0) > 1e-9:
                print("⚠️  %s 的 gear_ratio=%.3f，但 URDF 显示 calf 速度上限是 "
                      "hip/thigh 的一半（疑似 2:1 减速），请确认。" % (name, self.gear_ratio[i]))
        for i, name in enumerate(spec.JOINT_LABELS):
            # 零位偏移不该大于半个关节行程，超出基本是填错了
            span = spec.JOINT_UPPER[i] - spec.JOINT_LOWER[i]
            if abs(self.zero_offset[i]) > span:
                raise ValueError(
                    "%s 的 zero_offset=%.3f 超过关节行程 %.3f，疑似填错"
                    % (name, self.zero_offset[i], span))

    # ------------------------------------------------------------------ 闸门
    def require_verified(self):
        if not self.verified:
            raise Sim2RealMapNotVerified(
                "hardware_map.yaml 的 verified 仍是 false —— 拒绝进入策略控制。\n"
                "  请先逐项确认 12 个关节的 can_id / sign / zero_offset，"
                "再把 verified 改成 true。\n"
                "  （没有真机映射表也能先验证整条链路：`main.py --dry-run` 用"
                " backends/stub_backend.py 的假机器人跑通全栈，不需要硬件。）")

    # ------------------------------------------------------------------ 换算
    def sim_to_real(self, q_sim) -> np.ndarray:
        """策略侧关节角 → 电机侧角度。"""
        q_sim = np.asarray(q_sim, dtype=np.float64)
        return self.sign * (q_sim * self.gear_ratio) + self.zero_offset

    def real_to_sim(self, q_real) -> np.ndarray:
        """电机侧角度 → 策略侧关节角。"""
        q_real = np.asarray(q_real, dtype=np.float64)
        return (q_real - self.zero_offset) / self.sign / self.gear_ratio

    def sim_vel_to_real(self, dq_sim) -> np.ndarray:
        return self.sign * np.asarray(dq_sim, dtype=np.float64) * self.gear_ratio

    def real_vel_to_sim(self, dq_real) -> np.ndarray:
        return np.asarray(dq_real, dtype=np.float64) / self.sign / self.gear_ratio

    def sim_tau_to_real(self, tau_sim) -> np.ndarray:
        """关节侧力矩 → 电机侧力矩（减速比越大，电机侧力矩越小）。"""
        return np.asarray(tau_sim, dtype=np.float64) / self.gear_ratio

    # ================================================================== PD 增益换算
    def kp_kd_to_motor(self, kp, kd):
        """关节侧 PD 增益 → 电机侧 MIT 增益。★ 除以 gear_ratio²（不是乘、也不是除以 n）。

        推导（n = gear_ratio = 电机输出轴转角 / 关节转角）：
            MIT 模式的 PD 在【电机自己的编码器坐标系】里算：
                T_m = Kp_m·(q*_m − q_m) − Kd_m·q̇_m
            减速级的功率守恒给出： q_m = n·q_j， T_j = n·T_m
            代入：
                T_j = n·[Kp_m·n·(q*_j − q_j) − Kd_m·n·q̇_j]
                    = n²·Kp_m·(q*_j − q_j) − n²·Kd_m·q̇_j
            ⇒ 关节等效增益 Kp_j = n²·Kp_m，Kd_j = n²·Kd_m
            ⇒ 反解： Kp_m = Kp_j / n²， Kd_m = Kd_j / n²

        为什么必须是 n² 而不是 n：力矩放大 n 只是一半，另一半是"同样的关节角误差，
        电机侧要转 n 倍的角度"。只除 n 会让关节等效刚度偏大 n 倍。
        小腿 n=2 → 除以 4；hip/thigh n=1 → 不变。

        ⚠️ 前提：MIT 的 Kp/Kd 是【电机输出侧】量（Nm/rad、Nm·s/rad，编码器装在输出侧）。
           这是 RobStride 的通行解释，但真机第一次上电请用文档里那两个台架试验核实。
        """
        g2 = np.asarray(self.gear_ratio, dtype=np.float64) ** 2
        kp = np.broadcast_to(np.asarray(kp, dtype=np.float64), (spec.NUM_JOINTS,))
        kd = np.broadcast_to(np.asarray(kd, dtype=np.float64), (spec.NUM_JOINTS,))
        return kp / g2, kd / g2

    def gain_report(self, kp_joint, kd_joint):
        """给 --check 打印用：逐关节把关节侧增益换算到电机侧。"""
        kp_m, kd_m = self.kp_kd_to_motor(kp_joint, kd_joint)
        rows = []
        for i, name in enumerate(spec.JOINT_LABELS):
            rows.append((name, float(np.broadcast_to(kp_joint, (spec.NUM_JOINTS,))[i]),
                         float(kp_m[i]), float(kd_m[i]), float(self.gear_ratio[i])))
        return rows

    def clip_kp_kd(self, kp, kd):
        """把 PD 增益裁到 RS-02 MIT 模式的合法范围。"""
        kp = np.clip(np.asarray(kp, dtype=np.float64), *self.mit_kp_range)
        kd = np.clip(np.asarray(kd, dtype=np.float64), *self.mit_kd_range)
        return kp, kd

    # ================================================================== 软限位自检
    def check_board_window(self, default_dof_pos=None, margin_warn_rad=0.05):
        """★ 逐关节检查「关节行程 + 默认姿态」映射到电机侧后是否落在固件窗口内。

        这是【零点/符号/减速比填错】最廉价、最先能做的检测 —— 不需要通电机电、
        不需要任何外部仪器，纯粹用固件的软限位表当独立参考系做交叉验证。

        为什么能查出零点错：
            关节行程（URDF，物理事实）→ 电机侧 = sign*(q*gear) + zero_offset
            固件窗口（电机自己的坐标系）必须能容纳这段行程。
            zero_offset 错一个常数，整段行程就会整体平移出窗口 —— 于是矛盾暴露。
        为什么查不出"小"的零点错：
            calf 窗口有 210° 宽而关节行程只占 160°，所以 ±25° 以内的零点偏差仍能
            "装进去"。窗口一致是【必要条件不是充分条件】。

        返回 list[dict]，每个关节一条：
            name / sim_lo / sim_hi / motor_lo / motor_hi / win_lo / win_hi
            default_motor / default_ok / range_ok / margin_lo / margin_hi
            verdict: "ok" | "default_clipped" | "range_over" | "no_window" | "tight"
            fix    : 若是 range_over，给出「能装进去」的候选 (sign, zero_offset)
        """
        dflt = (spec.DEFAULT_DOF_POS if default_dof_pos is None
                else np.asarray(default_dof_pos, dtype=np.float64))
        out = []
        for i, name in enumerate(spec.JOINT_LABELS):
            rec = {"name": name, "index": i,
                   "sim_lo": float(spec.JOINT_LOWER[i]), "sim_hi": float(spec.JOINT_UPPER[i]),
                   "sign": float(self.sign[i]), "gear": float(self.gear_ratio[i]),
                   "zero_offset": float(self.zero_offset[i]),
                   "default_sim": float(dflt[i])}
            if not np.isfinite(self.board_lo[i]):
                rec["verdict"] = "no_window"
                out.append(rec)
                continue
            w_lo, w_hi = float(self.board_lo[i]), float(self.board_hi[i])
            # 电机侧范围（注意 sign 可能为负，要排好序）
            # ★ 这里必须逐关节算标量：sim_to_real 是向量化接口，传标量会返回 12 维数组
            a = self._motor_of(i, rec["sim_lo"])
            b = self._motor_of(i, rec["sim_hi"])
            m_lo, m_hi = min(a, b), max(a, b)
            d = self._motor_of(i, rec["default_sim"])
            rec.update({"win_lo": w_lo, "win_hi": w_hi,
                        "motor_lo": float(m_lo), "motor_hi": float(m_hi),
                        "default_motor": d,
                        "margin_lo": float(m_lo - w_lo), "margin_hi": float(w_hi - m_hi)})
            rec["default_ok"] = bool(w_lo - 1e-9 <= d <= w_hi + 1e-9)
            rec["range_ok"] = bool(rec["margin_lo"] >= -1e-9 and rec["margin_hi"] >= -1e-9)
            if not rec["default_ok"]:
                rec["verdict"] = "default_clipped"
            elif not rec["range_ok"]:
                rec["verdict"] = "range_over"
            elif min(rec["margin_lo"], rec["margin_hi"]) < margin_warn_rad:
                rec["verdict"] = "tight"
            else:
                rec["verdict"] = "ok"
            if not rec["range_ok"]:
                rec["fix"] = self._propose_window_fix(i)
            out.append(rec)
        return out

    def _motor_of(self, i: int, q_sim: float) -> float:
        """单个关节的电机侧角度（标量版 sim_to_real）。"""
        return float(self.sign[i] * (float(q_sim) * self.gear_ratio[i]) + self.zero_offset[i])

    def _propose_window_fix(self, i: int):
        """给一个"能装进固件窗口"的 (sign, zero_offset) 候选（只保证窗口自洽）。

        ⚠️ 只解决"装不下"这个矛盾，【不能】据此判定符号或零点的物理正确性 ——
           对单边窗口，两种 sign 各自配一个 offset 都能装进去。
        做法：对 sign ∈ {+1,−1} 解出可行 offset 区间 [lo,hi]，取中点（余量最大）。
        """
        q_lo, q_hi = float(spec.JOINT_LOWER[i]), float(spec.JOINT_UPPER[i])
        gear = float(self.gear_ratio[i])
        w_lo, w_hi = float(self.board_lo[i]), float(self.board_hi[i])
        cands = []
        for sg in (+1.0, -1.0):
            a, b = sg * q_lo * gear, sg * q_hi * gear
            m_lo, m_hi = min(a, b), max(a, b)
            lo, hi = w_lo - m_lo, w_hi - m_hi        # off ∈ [lo, hi]
            if lo <= hi:
                off = 0.5 * (lo + hi)
                cands.append({"sign": sg, "zero_offset": off,
                              "margin": float(0.5 * (hi - lo)),
                              "offset_deg": float(np.degrees(off))})
        cands.sort(key=lambda c: -c["margin"])
        return cands

    def window_fit_report(self, default_dof_pos=None):
        """返回 (ok: bool, lines: list[str])，供 --check 打印。"""
        if not self.board_limits_active:
            # 固件已删除软限位：这一路交叉检查的前提没有了，跳过而不是判失败。
            # ★ 但要把"失去的是什么"讲清楚 —— 少了一道与零点无关的护栏。
            return True, [
                "      ⚠️  固件已删除电机软限位（hardware_map.yaml: board_limits_active=false）",
                "          → 跳过程序行程↔软限位交叉检查（这个参考系不存在了）",
                "          → 板侧不再夹目标 → 行程保护只剩上位机 safety 的关节限位夹取",
                "          → 而那一层是【零点依赖】的：零点偏 δ，保护边界也偏 δ，",
                "            没有独立护栏了。零点标定因此比之前更关键（PORTING.md 第 6 节）。",
                "          （若板子还跑着旧固件，err_flags bit4 仍会出现，那是『该刷固件』的信号）",
            ]
        recs = self.check_board_window(default_dof_pos)
        if all(r["verdict"] == "no_window" for r in recs):
            return True, ["      hardware_map.yaml 里没有 board_limits_deg —— 跳过硬限位交叉检查",
                          "      （建议补上：固件 motor_task.cpp 里那 12 个 Pos_min/Pos_max）"]
        lines = []
        bad = [r for r in recs if r["verdict"] in ("default_clipped", "range_over")]
        warn = [r for r in recs if r["verdict"] == "tight"]
        lines.append("      关节          URDF行程→电机侧        固件窗口            默认姿态→电机侧  判定")
        for r in recs:
            if r["verdict"] == "no_window":
                lines.append("      %-12s （无窗口数据）" % r["name"])
                continue
            mark = {"ok": "✓", "tight": "⚠ 余量小", "range_over": "✗ 行程越界",
                    "default_clipped": "✗ 默认姿态被夹死"}[r["verdict"]]
            lines.append("      %-12s [%+6.2f,%+6.2f]  [%+7.2f,%+7.2f]  %+8.2f      %s"
                         % (r["name"], r["motor_lo"], r["motor_hi"], r["win_lo"], r["win_hi"],
                            r["default_motor"], mark))
        for r in bad:
            lines.append("      ✗ %s：当前 sign=%+g gear=%.2f zero_offset=%+.3f rad"
                         % (r["name"], r["sign"], r["gear"], r["zero_offset"]))
            lines.append("        电机侧行程要走 %.0f°，窗口只有 %.0f°（%.0f° 落在窗外）"
                         % (np.degrees(r["motor_hi"] - r["motor_lo"]),
                            np.degrees(r["win_hi"] - r["win_lo"]),
                            np.degrees(max(0.0, -r["margin_lo"]) + max(0.0, -r["margin_hi"]))))
            if r.get("fix"):
                lines.append("        若窗口本身可信（= 电机零位在某个机械端点），"
                             "能让窗口自洽的候选：")
                for c in r["fix"]:
                    lines.append("          sign=%+g  zero_offset=%+.3f rad (%+7.1f°)   "
                                 "余量 ±%.0f°" % (c["sign"], c["zero_offset"],
                                                c["offset_deg"], np.degrees(c["margin"])))
                if len(r["fix"]) > 1 and abs(r["fix"][0]["margin"] - r["fix"][1]["margin"]) < 1e-9:
                    lines.append("        ★ 两个符号的余量完全一样 —— 窗口【定不了符号】："
                                 "同样是『装得进去』，关节会朝相反方向转。")
            else:
                lines.append("        两种 sign 都装不进这个窗口 —— 窗口与机械行程本身矛盾，"
                             "要么改固件 motor_task.cpp 的 Pos_min/Pos_max，要么电机零位/减速比另有问题")
            lines.append("        ⚠️ 无论如何：窗口自洽只是【必要】条件，"
                         "符号与零点的物理正确性必须靠机械参考实测（PORTING.md 第 5 节）")
        if warn:
            lines.append("      ⚠️ 余量 <5° 的关节: %s（本来就贴边，别再往限位方向加偏置）"
                         % ", ".join(r["name"] for r in warn))
        ok = not bad
        if ok:
            lines.append("      ✓ 12 个关节的行程与默认姿态都在固件窗口内（交叉检查通过）")
        return ok, lines

    # ------------------------------------------------------------------ 展示
    def describe(self) -> str:
        lines = ["电机映射表: %s" % self.path,
                 "  verified = %s%s" % (self.verified, "" if self.verified else "   ★ 未确认！"),
                 "  motor_model = %s   MIT kp∈%s kd∈%s"
                 % (self.motor_model, self.mit_kp_range, self.mit_kd_range),
                 "  %-10s %-6s %7s %5s %7s %12s"
                 % ("关节", "总线", "can_id", "sign", "gear", "zero_offset")]
        for i, n in enumerate(spec.JOINT_LABELS):
            lines.append("  %-10s FDCAN%d %7d %+5d %7.2f %12.4f"
                         % (n, self.bus[i] + 1, self.can_id[i], int(self.sign[i]),
                            self.gear_ratio[i], self.zero_offset[i]))
        return "\n".join(lines)


if __name__ == "__main__":
    m = MotorMapping()
    print(m.describe())
    print()
    q_sim = spec.DEFAULT_DOF_POS
    q_real = m.sim_to_real(q_sim)
    back = m.real_to_sim(q_real)
    print("默认站姿 sim -> real:", np.round(q_real, 4))
    print("回读 real -> sim     :", np.round(back, 4))
    assert np.allclose(back, q_sim), "sim<->real 往返不一致"
    print("✅ 往返换算自洽")
    if not m.verified:
        print()
        print("⚠️  verified=false：进策略控制会被拦。可用的路子："
              "① --dry-run 干跑验证整条链路；② --mode init 只起立保持（不起策略）；"
              "③ 标定完把 verified 改成 true。")
