#!/usr/bin/env python3
"""把「sim2sim 当前在验证的那份策略」收编进 sim2real 的 models/

    python tools/adopt_policy.py                 # 取最新的 policy_auto_*.pt（= auto_sim2sim_check.sh 导出的）
    python tools/adopt_policy.py --step 4000     # 指定 auto 策略的步数
    python tools/adopt_policy.py --from 其他.pt   # 指定任意策略文件
    python tools/adopt_policy.py --show          # 只看现在捆的是哪份、有没有更新的

★ 为什么要有这个工具，而不是 `cp xxx models/policy.pt`：
  1) **策略身份只能看权重指纹**，不能看文件名/文件 md5 —— 这个仓库里已经有
     `policy_V19_11601.pt` 被改名成 `policy_V16_t_11601.pt` 的先例，按名字找必然出错；
  2) 覆盖前必须**按指纹留一份旧策略**（否则"上一版能跑的策略"就没了，
     V19 那次就是只剩 models/policy.pt 一处副本）；
  3) policy.json 里的指纹必须同步改，否则 main.py 启动时会因为
     "指纹与登记不符"**拒绝启动**（这是有意设计的防呆）；
  4) 顺手把"这份策略在 sim2sim 里跑到什么水平"记进 policy.json ——
     以后回头看就知道当初为什么敢把它放上机器人。

它做的事：算指纹 → 备份旧 policy.pt（按指纹命名）→ 覆盖 → 重写 policy.json
（含来源、provenance、sim2sim 验证证据）→ 打印摘要。
**不动 models/README.md**（那是给人读的长文，自动改容易越改越乱）。
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM2REAL = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_SIM2REAL))          # -> <LgRobotDog>
_POL_DIR = os.path.join(_REPO, "legged_gym", "logs", "flat_LgDog1", "exported", "policies")
_REC_DIR = os.path.join(_REPO, "legged_gym", "legged_gym", "scripts", "recorded_io")
MODELS = os.path.join(_SIM2REAL, "models")
TARGET = os.path.join(MODELS, "policy.pt")
SIDE = os.path.join(MODELS, "policy.json")


# ============================================================================
def weight_fingerprint(path: str, n: int = 12) -> str:
    """按参数名排序后对张量字节做 md5。★ 唯一可信的策略身份（文件 md5 不可信）。"""
    import torch
    m = torch.jit.load(path, map_location="cpu")
    m.eval()
    sd = m.state_dict()
    h = hashlib.md5()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:n]


def check_io(path: str):
    import torch
    m = torch.jit.load(path, map_location="cpu")
    m.eval()
    out = m(torch.zeros(1, 270))
    return int(out.shape[-1])


def find_auto_policy(step=None):
    if not os.path.isdir(_POL_DIR):
        return None
    cands = [f for f in os.listdir(_POL_DIR)
             if f.startswith("policy_auto_") and f.endswith(".pt")]
    if step is not None:
        want = "policy_auto_%s.pt" % step
        return os.path.join(_POL_DIR, want) if want in cands else None
    if not cands:
        return None
    # 按文件名里的步数取最大（比 mtime 可靠：mtime 会被拷贝/解压改变）
    def step_of(f):
        try:
            return int(f[len("policy_auto_"):-3])
        except ValueError:
            return -1
    return os.path.join(_POL_DIR, max(cands, key=step_of))


def find_validation(fp: str):
    """在 recorded_io 里找最近一次用这个指纹跑过的 sim2sim 录制（作为验证证据）。"""
    import numpy as np
    if not os.path.isdir(_REC_DIR):
        return None
    fs = sorted([os.path.join(_REC_DIR, f) for f in os.listdir(_REC_DIR)
                 if f.endswith(".npz")], key=os.path.getmtime, reverse=True)[:60]
    for f in fs:
        try:
            d = np.load(f, allow_pickle=True)
            note = str(d["_meta_note"])
        except Exception:
            continue
        if ("wfp=%s" % fp) not in note:
            continue
        ev = {"recording": os.path.relpath(f, _REPO),
              "started": str(d["_meta_started"]),
              "steps": int(d["_meta_n_steps"]),
              "note": note[:200]}
        # 顺带用用户自己的姿态分析工具量化一下
        try:
            r = subprocess.run([sys.executable,
                                os.path.join(_REPO, "tools", "analyze_attitude.py"), f],
                               capture_output=True, text=True, timeout=60)
            lines = [ln.strip() for ln in r.stdout.splitlines()
                     if ("pitch" in ln or "未翻倒" in ln or "翻倒" in ln or "机身高度" in ln)]
            if lines:
                ev["attitude"] = lines
        except Exception as e:
            ev["attitude"] = ["（analyze_attitude 未跑成: %s）" % e]
        return ev
    return None


# ============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="把 sim2sim 当前验证的策略收编进 sim2real/models/")
    ap.add_argument("--from", dest="src", default=None, help="指定策略文件（默认取最新 policy_auto_*.pt）")
    ap.add_argument("--step", type=int, default=None, help="指定 auto 策略的步数，如 4000")
    ap.add_argument("--show", action="store_true", help="只看不换")
    ap.add_argument("--yes", action="store_true", help="跳过确认")
    a = ap.parse_args(argv)

    # ---- 现状 ----
    cur_fp = None
    if os.path.isfile(TARGET):
        cur_fp = weight_fingerprint(TARGET)
    print("=" * 74)
    print("sim2real 当前捆的策略")
    print("=" * 74)
    if cur_fp:
        print("  models/policy.pt  指纹 = %s" % cur_fp)
        try:
            side = json.load(open(SIDE, encoding="utf-8"))
            print("  policy.json 登记    = %s%s"
                  % (side.get("weight_fingerprint"),
                     "" if side.get("weight_fingerprint") == cur_fp else "   ✗ 不一致！"))
            if side.get("source_file"):
                print("  来源: %s" % str(side["source_file"])[:100])
        except Exception as e:
            print("  ✗ policy.json 读不了: %s" % e)
    else:
        print("  （models/policy.pt 不存在）")

    src = a.src or find_auto_policy(a.step)
    if a.src and not os.path.isfile(a.src):
        print("\n✗ 找不到 %s" % a.src)
        return 1
    if src is None:
        print("\n✗ 没找到 policy_auto_*.pt（跑 auto_sim2sim_check.sh 导出后再生效），"
              "或用 --from 指定文件")
        return 1

    print()
    print("=" * 74)
    print("sim2sim 侧最新（auto_sim2sim_check.sh 的产物）")
    print("=" * 74)
    print("  %s" % src)
    print("  %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(src))))
    new_fp = weight_fingerprint(src)
    out_dim = check_io(src)
    print("  权重指纹 = %s   输出维度 = %d %s"
          % (new_fp, out_dim, "✓" if out_dim == 12 else "✗ 不是 12 维，不能用于 LgDog1！"))

    if new_fp == cur_fp:
        print("\n✅ 已经是同一份权重（指纹相同）—— 不用换。")
        return 0
    if out_dim != 12:
        print("\n✗ 输出维度不是 12，拒绝收编。")
        return 1
    if a.show:
        print("\n（--show：只看不换。去掉 --show 才会真的换。）")
        return 0

    print()
    print("  ⚠️ 换策略等于换行为：换完必须**先在 sim2sim 里跑过**再上真机。")
    ev = find_validation(new_fp)
    if ev:
        print("  在 recorded_io 里找到用这个指纹跑过的录制: %s" % ev["recording"])
        for ln in ev.get("attitude", []):
            print("      %s" % ln)
    else:
        print("  ⚠️ 没在 recorded_io 里找到用这个指纹跑过的录制 —— 这份还没在 sim2sim 里验过！")
        print("     建议先: nohup bash tools/auto_sim2sim_check.sh <run> <step> &")

    if not a.yes:
        print("\n确认收编？输入大写 YES 回车，其它任何输入放弃：")
        try:
            if input().strip() != "YES":
                print("已放弃，什么都没改。")
                return 0
        except EOFError:
            print("没有 stdin（脚本化请加 --yes）—— 已放弃。")
            return 0

    # ---- 备份旧策略（按【指纹】命名，这样按文件名也能认出是哪一份权重）----
    os.makedirs(MODELS, exist_ok=True)
    if cur_fp:
        bak = os.path.join(MODELS, "policy_%s.pt" % cur_fp)
        if not os.path.exists(bak):
            shutil.copy2(TARGET, bak)
            print("  已备份旧策略 → models/policy_%s.pt" % cur_fp)
        else:
            print("  （models/policy_%s.pt 已存在，跳过备份）" % cur_fp)

    shutil.copy2(src, TARGET)
    got = weight_fingerprint(TARGET)
    assert got == new_fp, "拷贝后指纹变了？！%s != %s" % (got, new_fp)

    # ---- 重写 policy.json ----
    side = {}
    if os.path.isfile(SIDE):
        try:
            side = json.load(open(SIDE, encoding="utf-8"))
        except Exception:
            side = {}
    side.update({
        "weight_fingerprint": new_fp,
        "source_file": os.path.relpath(src, _REPO),
        "file_size": os.path.getsize(src),
        "adopted_on": time.strftime("%Y-%m-%d %H:%M:%S"),
        "adopted_by": "tools/adopt_policy.py",
        "previous_fingerprint": cur_fp,
        "sync_note": ("本目录的 policy.pt 是当前上栈的策略；历史版本按指纹留在 "
                      "models/policy_<指纹>.pt。换策略请用 tools/adopt_policy.py，"
                      "别手敲 cp（会被指纹闸门拦住）。"),
    })
    if ev:
        side["validated"] = ev
    side["note"] = ("★ 不要用文件 md5 判断是不是同一份权重：torch.jit 每次序列化都会改 "
                    "zip 里的 serialization_id 与自动生成的源码条目，文件 md5 会变、权重不变。"
                    "所有地方一律用 weight_fingerprint。")
    json.dump(side, open(SIDE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print()
    print("✅ 收编完成")
    print("   models/policy.pt        指纹 = %s   （旧: %s）" % (new_fp, cur_fp))
    print("   models/policy.json      已更新")
    print("   历史版本: %s" % ", ".join(
        sorted(f for f in os.listdir(MODELS) if f.startswith("policy_") and f.endswith(".pt"))))
    print()
    print("   下一步： ./selftest.sh  看策略段；  python main.py --check  看指纹核对")
    return 0


if __name__ == "__main__":
    sys.exit(main())
