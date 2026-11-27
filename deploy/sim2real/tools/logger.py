#!/usr/bin/env python3
"""上位机自动日志 —— 每次运行自动建一个 session 目录并落盘

设计目标（按重要性排序）：
  1. **自动**：不需要任何 flag，main.py 一起来就记；退出/异常/被 Ctrl-C / 被 kill
     都要把数据落干净。跑完一次机器人却什么都没留下是最亏的。
  2. **不写满磁盘**：真机上日志能轻松写到几百 MB/小时。所以两条闸门 ——
     剩余空间低于阈值、或本 session 超过上限 → 停止写 CSV 并记一条事件。
     机器人因为日志写满磁盘而挂掉，是很难受的故障。
  3. **事后能查**：CSV 给人看（一行一个控制周期），JSONL 记事件（状态切换/急停/
     失联/安全触发），npz 存完整数组（观测/动作/目标——CSV 放不下 270 维观测）。

产物（logs/<session>/）：
    meta.json     配置快照 + 各种指纹（策略 md5、hardware_map、代码 git 状态）
    state.csv     控制周期逐帧摘要（列名见 STATE_FIELDS）
    events.jsonl  事件流，一行一个 JSON，**每条立即 flush**（崩溃也不丢事件）
    data.npz      完整 obs / action / target_q / q / dq（结束时一次性写）
    summary.txt   人读总结（时长、帧数、事件计数、主要安全触发原因）

用法见文件末尾自测。main.py 里只需：
    log = LogSession(cfg, meta={...})
    log.state(t=..., step=..., ...)
    log.event("SAFETY", "倾倒", gz=-0.25)
    log.close()
"""

import atexit
import json
import os
import shutil
import signal
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import spec  # noqa: E402

# state.csv 的列顺序（改这里会让新旧日志列不一致，注意版本）
STATE_FIELDS = (
    ["t", "wall", "step", "level"]
    + ["cmd_vx", "cmd_vy", "cmd_wz", "run_policy", "estop"]
    + ["blend", "suppressed"]                         # 目标混合系数 / 是否被零指令抑制
    + ["act_min", "act_max", "act_mean"]
    + ["tgt_err_max", "tgt_sat_n"]                    # 实测 q 相对站姿的最大偏移 / 越过偏移框的关节数
    + ["tgt_off_max"]                                 # ★ 真正【下发目标】相对站姿的最大偏移
    #   （固件删除软限位后，safety 的关节限位夹取是唯一的绝对行程保护，
    #     这一列就是"夹取有没有在生效/是否长期贴限位"的现场证据）
    + ["q_min", "q_max", "dq_max", "tau_max", "temp_max"]
    + ["roll", "pitch", "yaw"]
    + ["gyro_x", "gyro_y", "gyro_z", "pg_x", "pg_y", "pg_z"]
    + ["imu_age_ms", "pad_alive", "link_alive", "rx_frames", "tx_drops", "motor_missing"]
)


class LogSession:
    """一次运行的日志会话。close() 之后所有写操作变成空操作（不抛异常）。"""

    def __init__(self, cfg: dict = None, meta: dict = None, tag: str = "run",
                 root: Optional[str] = None):
        cfg = cfg or {}
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True))
        self.every = max(1, int(cfg.get("every", 1)))
        self.flush_interval = float(cfg.get("flush_interval_s", 1.0))
        self.max_size_mb = float(cfg.get("max_size_mb", 500.0))
        self.min_free_mb = float(cfg.get("min_free_mb", 500.0))
        self.save_npz = bool(cfg.get("save_npz", True))

        self.root = root or os.path.join(_ROOT, cfg.get("dir", "logs"))
        stamp = time.strftime("%m%d_%H%M%S")
        self.dir = os.path.join(self.root, "%s_%s" % (tag, stamp))
        self.state_path = os.path.join(self.dir, "state.csv")
        self.event_path = os.path.join(self.dir, "events.jsonl")
        self.npz_path = os.path.join(self.dir, "data.npz")
        self.meta_path = os.path.join(self.dir, "meta.json")
        self.summary_path = os.path.join(self.dir, "summary.txt")

        self._csv = None
        self._evt = None
        self._closed = False
        self._rows = 0
        self._skipped = 0
        self._events = Counter()
        self._arrays: Dict[str, List[np.ndarray]] = {}
        self._t0 = time.time()
        self._last_flush = self._t0
        self._stopped_reason = ""
        self._prev_signals = {}

        if not self.enabled:
            return

        os.makedirs(self.dir, exist_ok=True)
        # state.csv 用缓冲写（50Hz 逐行 flush 太费）；events.jsonl 每条立即 flush
        self._csv = open(self.state_path, "w", encoding="utf-8", buffering=1 << 16)
        self._csv.write(",".join(STATE_FIELDS) + "\n")
        self._evt = open(self.event_path, "a", encoding="utf-8", buffering=1)

        self._write_meta(meta or {})
        # ★ 三重兜底：正常 close / 异常退出 / 被信号杀，都要落盘
        atexit.register(self.close)
        self._install_signal_handlers()
        self.event("SESSION_START", "日志会话开始", dir=self.dir)
        print("[LOG] session: %s" % self.dir)

    # ------------------------------------------------------------------ 元信息
    def _write_meta(self, extra: dict):
        meta = {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tag": os.path.basename(self.dir).split("_")[0],
            "hostname": _safe(os.uname().nodename),
            "python": sys.version.split()[0],
            "spec": _spec_fingerprint(),
            "config": _redact(self.cfg),
        }
        meta.update(extra)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    # ------------------------------------------------------------------ 事件
    def event(self, kind: str, msg: str = "", **fields):
        """记一条事件。★ 立即 flush —— 崩溃时事件流不能丢，它是最有价值的线索。"""
        if not self.enabled or self._closed:
            return
        self._events[kind] += 1
        rec = {"t": round(time.time() - self._t0, 4),
               "wall": time.strftime("%H:%M:%S"),
               "kind": kind, "msg": msg}
        rec.update({k: _jsonable(v) for k, v in fields.items()})
        try:
            self._evt.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._evt.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------ 状态
    def state(self, **kv):
        """写一行 state.csv。不需要每次都调 —— 内部按 cfg.every 抽帧。"""
        if not self.enabled or self._closed:
            return
        self._rows += 1
        if (self._rows % self.every) != 0:
            return
        if not self._check_budget():
            return

        t = kv.pop("t", time.time() - self._t0)
        row = {"t": "%.4f" % t,
               "wall": time.strftime("%H:%M:%S"),
               "step": self._rows}
        for f in STATE_FIELDS[3:]:
            v = kv.get(f, "")
            row[f] = ("%.5g" % v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
        try:
            self._csv.write(",".join(str(row.get(f, "")) for f in STATE_FIELDS) + "\n")
        except Exception:
            pass
        if (time.time() - self._last_flush) > self.flush_interval:
            self._flush_csv()

    def add_array(self, name: str, value):
        """累积数组，close() 时一起写进 data.npz（给 270 维观测这类宽数据用）。"""
        if not self.enabled or self._closed or not self.save_npz:
            return
        self._arrays.setdefault(name, []).append(
            np.asarray(value, dtype=np.float32).copy())   # ★ copy：防别名（见 record_io 的教训）

    # ------------------------------------------------------------------ 预算闸门
    def _check_budget(self):
        """磁盘保护：超上限或剩余空间不足就停写 CSV（事件流继续记，它很小）。"""
        if self._stopped_reason:
            self._skipped += 1
            return False
        try:
            used = os.path.getsize(self.state_path) / 1e6
            if used > self.max_size_mb:
                self._stop_csv("state.csv 超过 %.0f MB 上限" % self.max_size_mb)
                return False
            free = shutil.disk_usage(self.dir).free / 1e6
            if free < self.min_free_mb:
                self._stop_csv("磁盘剩余 %.0f MB 低于阈值 %.0f MB"
                               % (free, self.min_free_mb))
                return False
        except Exception:
            pass
        return True

    def _stop_csv(self, why: str):
        self._stopped_reason = why
        self._flush_csv()
        self.event("LOG_BUDGET", "停止写 state.csv：" + why)
        print("[LOG] ⚠️ %s（事件流继续记录）" % why)

    def _flush_csv(self):
        try:
            if self._csv is not None:
                self._csv.flush()
                os.fsync(self._csv.fileno())
        except Exception:
            pass
        self._last_flush = time.time()

    # ------------------------------------------------------------------ 退出
    def _install_signal_handlers(self):
        """被 kill/Ctrl-C 时也要落盘。只接管 TERM/INT，其余保持原样。"""
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._prev_signals[sig] = signal.getsignal(sig)

                def handler(signum, frame, _sig=sig):
                    self.event("SIGNAL", "收到信号 %d，落盘退出" % signum)
                    self.close()
                    prev = self._prev_signals.get(_sig)
                    if callable(prev):
                        prev(signum, frame)
                    else:
                        raise SystemExit(128 + signum)

                signal.signal(sig, handler)
            except Exception:
                pass

    def close(self):
        """落盘并关闭。幂等（可多次调用）。

        ⚠️ 顺序：必须【先写 SESSION_END 事件、再置 _closed】，否则 event() 会因为
        _closed 已置位而跳过 —— 结束事件永远记不上（自测抓到过）。"""
        if self._closed or not self.enabled:
            self._closed = True
            return
        dur = time.time() - self._t0
        self.event("SESSION_END", "日志会话结束",
                   duration_s=round(dur, 2), rows=self._rows,
                   skipped=self._skipped, stopped=self._stopped_reason)
        self._flush_csv()
        try:
            if self._csv:
                self._csv.close()
        except Exception:
            pass
        try:
            if self._evt:
                self._evt.close()
        except Exception:
            pass

        if self.save_npz and self._arrays:
            try:
                np.savez_compressed(self.npz_path,
                                    **{k: np.stack(v) for k, v in self._arrays.items()})
            except Exception as e:
                print("[LOG] ⚠️ npz 保存失败: %s" % e)

        try:
            with open(self.summary_path, "w", encoding="utf-8") as f:
                f.write("session   : %s\n" % self.dir)
                f.write("duration  : %.1f s\n" % dur)
                f.write("rows      : %d（每 %d 周期一行）\n" % (self._rows, self.every))
                f.write("skipped   : %d\n" % self._skipped)
                if self._stopped_reason:
                    f.write("csv 停止  : %s\n" % self._stopped_reason)
                f.write("events    :\n")
                for k, v in self._events.most_common():
                    f.write("    %-18s %d\n" % (k, v))
                f.write("\n文件: state.csv / events.jsonl%s\n"
                        % (" / data.npz" if self._arrays else ""))
        except Exception:
            pass

        # 方便下次直接看最近一次
        try:
            link = os.path.join(self.root, "latest")
            if os.path.islink(link):
                os.unlink(link)
            os.symlink(os.path.basename(self.dir), link)
        except Exception:
            pass
        print("[LOG] 已保存 %d 行 -> %s" % (self._rows, self.dir))


# ============================================================================
def _safe(s):
    try:
        return str(s)
    except Exception:
        return "?"


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def _redact(cfg) -> dict:
    """配置快照。递归处理，保证可 JSON 化。"""
    if isinstance(cfg, dict):
        return {str(k): _redact(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_redact(v) for v in cfg]
    return _jsonable(cfg)


def _spec_fingerprint() -> dict:
    """把与训练对齐的关键常量记进 meta —— 事后排查"是不是配置漂移"全靠它。"""
    return {
        "NUM_OBS": spec.NUM_OBS,
        "HISTORY_LEN": spec.HISTORY_LEN,
        "NUM_ONE_STEP_OBS": spec.NUM_ONE_STEP_OBS,
        "CONTROL_FREQ": spec.CONTROL_FREQ,
        "ACTION_SCALE": spec.ACTION_SCALE,
        "KP": spec.KP, "KD": spec.KD,
        "DEFAULT_DOF_POS": spec.DEFAULT_DOF_POS.tolist(),
        "FRAME_ORDER": spec.FRAME_ORDER,
        "CMD_DEADZONE_XY": spec.CMD_DEADZONE_XY,
    }


def _selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="lgdlog_")
    print("自测 session 目录: %s" % tmp)
    log = LogSession({"enabled": True, "every": 2, "max_size_mb": 0.05,
                      "save_npz": True}, meta={"note": "selftest"}, root=tmp)
    log.event("TEST", "开始", k=1)

    # 写 200 帧（every=2 → 100 行），并塞大数组
    for i in range(200):
        log.state(t=i * 0.02, cmd_vx=0.5, cmd_vy=0.0, cmd_wz=0.0,
                  run_policy=True, estop=False, act_max=1.234,
                  tgt_err_max=0.31, q_min=-1.2, q_max=1.5, dq_max=8.8,
                  tau_max=9.9, temp_max=41.0, roll=0.01, pitch=-0.02, yaw=1.0,
                  gyro_x=0.1, gyro_y=0.2, gyro_z=0.3,
                  pg_x=0.0, pg_y=-0.1, pg_z=-0.99,
                  imu_age_ms=2.0, pad_alive=True, link_alive=True,
                  rx_frames=1234, tx_drops=0, motor_missing=0, level="NORMAL")
        log.add_array("obs", np.zeros(spec.NUM_OBS))
        log.add_array("action", np.zeros(spec.NUM_SIMPLE_OBS if hasattr(spec, "NUM_SIMPLE_OBS")
                                         else spec.NUM_JOINTS))
        if i == 50:
            log.event("ESTOP", "测试急停", reason="selftest")
    log.close()

    files = sorted(os.listdir(log.dir))
    print("产物:", files)
    for need in ("meta.json", "state.csv", "events.jsonl", "summary.txt", "data.npz"):
        assert need in files, "缺少 %s" % need

    # 校验 CSV 行数与列数
    import csv
    with open(log.state_path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    print("state.csv: %d 行（1 表头 + %d 数据）" % (len(rows), len(rows) - 1))
    assert rows[0] == list(STATE_FIELDS), "列名不匹配"
    assert all(len(r) == len(STATE_FIELDS) for r in rows), "有行的列数不对"
    assert len(rows) - 1 == 100, "every=2 时 200 帧应写 100 行，实际 %d" % (len(rows) - 1)

    # 校验 events
    evs = [json.loads(l) for l in open(log.event_path, encoding="utf-8")]
    kinds = [e["kind"] for e in evs]
    print("events.jsonl: %d 条 %s" % (len(evs), kinds))
    for k in ("SESSION_START", "TEST", "ESTOP", "SESSION_END"):
        assert k in kinds, "缺少事件 %s" % k

    # 校验 npz 形状
    d = np.load(log.npz_path)
    print("data.npz: obs%s action%s" % (d["obs"].shape, d["action"].shape))
    assert d["obs"].shape == (200, spec.NUM_OBS)

    # 磁盘闸门：上限 0.05MB，继续写应触发停止
    log2 = LogSession({"enabled": True, "every": 1, "max_size_mb": 0.01}, root=tmp)
    for i in range(3000):
        log2.state(t=i, cmd_vx=0.1, level="NORMAL")
    stopped = log2._stopped_reason
    log2.close()
    print("磁盘闸门: %s" % (stopped or "未触发"))
    assert stopped, "超过 max_size_mb 应触发停止"
    evs2 = [json.loads(l) for l in open(log2.event_path, encoding="utf-8")]
    assert any(e["kind"] == "LOG_BUDGET" for e in evs2), "应记一条 LOG_BUDGET 事件"
    print("  ✓ 超限后停止 CSV 且记了 LOG_BUDGET 事件")

    # close 之后继续写不应抛异常
    log2.state(t=1); log2.event("X", "y")
    print("  ✓ close 后写入是安全的空操作")

    shutil.rmtree(tmp, ignore_errors=True)
    print()
    print("✅ logger 自测通过")


if __name__ == "__main__":
    _selftest()
