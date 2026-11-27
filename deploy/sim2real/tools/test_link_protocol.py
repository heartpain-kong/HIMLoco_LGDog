#!/usr/bin/env python3
"""★ C 端 ↔ Python 端 协议交叉验证

为什么需要它：协议有两份实现（固件 `lgdog_link.c` 与上位机 `host_link.py`），
"两段代码看起来一样"不等于一致 —— 一个字节序、一个 CRC 初值、一个 len 语义
不同，就会让机器人收不到指令或收到乱码。这个脚本把固件里的编解码【原封不动】
用 gcc 编译出来，喂同一批输入，与 Python 的结果逐字节比对。

跑法：
    python tools/test_link_protocol.py
依赖：gcc（本机已装）。不需要硬件、不需要目标板。

覆盖的用例：
  * 各消息类型的标准打包（含 0 长度载荷）
  * CRC8 标准检验向量 "123456789" -> 0xF4
  * 半包（分两次喂）、粘包（一次喂 3 帧）
  * 前置垃圾字节、帧内假 SOF、CRC 单比特翻转、超长 len
  * 两个方向（0xA5 / 0x5A）各自只认自己的 SOF
"""

import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_ROOT))          # LgRobotDog
sys.path.insert(0, _ROOT)

from interface import host_link as HL  # noqa: E402

FIRMWARE_DIR = None
HARNESS = os.path.join(_HERE, "link_test_harness.c")


def find_firmware_dir():
    """定位固件里 lgdog_link.c / .h 所在目录。

    优先级：环境变量 LGDOG_FIRMWARE_DIR → 仓库根同级的 dm02 → 仓库根下的 dm02
    """
    cands = []
    env = os.environ.get("LGDOG_FIRMWARE_DIR")
    if env:
        cands.append(env)
    parent = os.path.dirname(_REPO)          # /home/heartpain/rl
    for base in (parent, _REPO):
        cands.append(os.path.join(base, "dm02", "User_File", "task"))
        cands.append(os.path.join(base, "User_File", "task"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "lgdog_link.c")):
            return c
    return None


FAIL = []


def build_harness(workdir):
    """编译固件的 lgdog_link.c + 测试宿主。"""
    exe = os.path.join(workdir, "linktest")
    cmd = ["gcc", "-O1", "-Wall", "-Wextra", "-std=c99",
           "-I", FIRMWARE_DIR, HARNESS,
           os.path.join(FIRMWARE_DIR, "lgdog_link.c"),
           "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("❌ gcc 编译失败：")
        print(r.stderr)
        sys.exit(1)
    if r.stderr.strip():
        # 有警告也要显示出来（固件代码质量）
        print("gcc 警告（供参考）：")
        print(r.stderr.rstrip())
        print()
    return exe


class CHarness:
    """把指令喂给 C 宿主并收集输出。"""

    def __init__(self, exe):
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)

    def run(self, *lines):
        for ln in lines:
            self.p.stdin.write(ln + "\n")
        self.p.stdin.flush()
        out = []
        for _ in lines:
            out.append(self.p.stdout.readline().rstrip("\n"))
        return out

    def close(self):
        try:
            self.p.stdin.write("Q\n"); self.p.stdin.flush()
        except Exception:
            pass
        self.p.terminate()


def check(name, c_line, py_line):
    ok = (c_line == py_line)
    print("  %s %-46s" % ("✓" if ok else "✗", name))
    if not ok:
        print("      C     : %s" % c_line)
        print("      Python: %s" % py_line)
        FAIL.append(name)


def main():
    global FIRMWARE_DIR
    if shutil.which("gcc") is None:
        print("❌ 找不到 gcc")
        return 1
    FIRMWARE_DIR = find_firmware_dir()
    if FIRMWARE_DIR is None:
        print("❌ 找不到固件里的 lgdog_link.c。用环境变量指定：")
        print("     LGDOG_FIRMWARE_DIR=/path/to/dm02/User_File/task python tools/test_link_protocol.py")
        return 1

    print("=" * 78)
    print("协议交叉验证：固件 lgdog_link.c  (gcc 编译)  vs  host_link.py")
    print("=" * 78)
    print("固件源码: %s" % FIRMWARE_DIR)
    print()

    workdir = tempfile.mkdtemp(prefix="lgdlink_")
    exe = build_harness(workdir)
    h = CHarness(exe)

    rng = random.Random(20260924)
    import numpy as _np
    nprng = _np.random.default_rng(20260924)

    try:
        # ---------------- 1. 打包比对 ----------------
        print("【1/4】打包（Encode）逐字节比对")
        cases = [
            ("HEARTBEAT", HL.SOF_HOST2BOARD, HL.MSG_HEARTBEAT, HL.pack_heartbeat(0x1234)),
            ("SET_MODE", HL.SOF_HOST2BOARD, HL.MSG_SET_MODE, HL.pack_set_mode(HL.MODE_PC_CONTROL)),
            ("SET_ENABLE", HL.SOF_HOST2BOARD, HL.MSG_SET_ENABLE, HL.pack_set_enable(True)),
            ("SET_ZERO(空载荷)", HL.SOF_HOST2BOARD, HL.MSG_SET_ZERO, b""),
            ("MIT_CMD", HL.SOF_HOST2BOARD, HL.MSG_MIT_CMD,
             HL.pack_mit_cmd(nprng.uniform(-1, 1, 12), np_full(30.0), np_full(0.8), np_full(0.0))),
            ("DAMPING_BRAKE", HL.SOF_HOST2BOARD, HL.MSG_DAMPING_BRAKE,
             HL.pack_damping_brake(np_full(2.5))),
            ("IMU(上行)", HL.SOF_BOARD2HOST, HL.MSG_IMU, make_imu()),
            ("MOTOR_CHUNK(上行)", HL.SOF_BOARD2HOST, HL.MSG_MOTOR_STATE, make_chunk()),
            ("STATUS(上行)", HL.SOF_BOARD2HOST, HL.MSG_STATUS, make_status()),
            ("随机载荷 200B(上限)", HL.SOF_HOST2BOARD, 0x77, bytes(rng.randrange(256) for _ in range(200))),
            ("随机载荷 1B", HL.SOF_BOARD2HOST, 0x78, bytes([0xAB])),
        ]
        c_lines = h.run(*["E %02x %02x %s" % (sof, mid, pl.hex())
                          for _, sof, mid, pl in cases])
        for (name, sof, mid, pl), c_line in zip(cases, c_lines):
            py_frame = HL.encode(sof, mid, pl)
            check("encode " + name, c_line, "ENC " + py_frame.hex())

        # ---------------- 1.5 载荷长度（结构体 sizeof）----------------
        print()
        print("【1.5/4】载荷长度：固件结构体 sizeof  vs  Python 常量")
        (c_sizes,) = h.run("Z")
        got = [int(x) for x in c_sizes.split()[1:]]
        exp = [HL.LEN_MIT_CMD, HL.LEN_BRAKE, HL.LEN_IMU,
               HL.LEN_MOTOR_CHUNK, HL.LEN_STATUS]
        names = ["MIT_CMD", "BRAKE", "IMU", "MOTOR_CHUNK", "STATUS"]
        for nm, g, e in zip(names, got, exp):
            check("sizeof %s" % nm, str(g), str(e))
        print("      （若 C 侧结构体没加 LGD_PACKED，IMU 会是 60、MOTOR_CHUNK 会是 108）")

        # ---------------- 2. CRC 标准向量 ----------------
        print()
        print("【2/4】CRC8 标准检验向量")
        (c_crc,) = h.run("E 00 00 %s" % b"123456789".hex())
        py = HL.encode(0x00, 0x00, b"123456789")
        check("CRC8(\"123456789\") 内嵌于帧尾", c_crc, "ENC " + py.hex())
        print("      （CRC-8/SMBUS 标准值应为 0xF4，帧尾字节 = %02x）" % py[-1])

        # ---------------- 3. 解帧比对 ----------------
        print()
        print("【3/4】解帧（Parse）与计数器比对")
        f1 = HL.encode(HL.SOF_HOST2BOARD, HL.MSG_HEARTBEAT, HL.pack_heartbeat(7))
        f2 = HL.encode(HL.SOF_HOST2BOARD, HL.MSG_SET_MODE, HL.pack_set_mode(1))
        f3 = HL.encode(HL.SOF_HOST2BOARD, HL.MSG_MIT_CMD,
                       HL.pack_mit_cmd(np_full(0.1), np_full(30.0), np_full(0.8), np_full(0.0)))
        # 最大帧：payload 正好 200 B → 整帧 204 B（接近 parser 缓冲 205 B）
        f_big = HL.encode(HL.SOF_HOST2BOARD, 0x7A, bytes(range(200)))
        bad = bytearray(f2); bad[-1] ^= 0xFF
        streams = [
            ("单帧", f1),
            ("粘包 3 帧", f1 + f2 + f3),
            ("前置垃圾字节", b"\x00\x11\x22\x33" + f1),
            ("半包(整个再补)", f1),          # 特殊处理见下
            ("CRC 错 1 帧 + 好帧", bytes(bad) + f1),
            ("假 SOF 在载荷里", bytes([HL.SOF_HOST2BOARD, 0x7F, 0x02, HL.SOF_HOST2BOARD, 0x99, 0x00]) + f1),
            ("超长 len", bytes([HL.SOF_HOST2BOARD, 0x7F, 0xFF]) + f1),
            ("空输入", b""),
            ("两个方向混发", f1 + HL.encode(HL.SOF_BOARD2HOST, HL.MSG_IMU, make_imu())),
            # ★ 大突发：一次喂进远超 parser 缓冲(205B)的字节。
            #   固件原先的实现会因"缓冲满丢最旧字节"把流开头吃掉、首帧消失，
            #   这个用例就是锁住那个已修复的 bug。
            ("大突发 5 帧(1025B 一次喂)", f_big * 5),
        ]
        c_lines = h.run(*["P %02x %s" % (HL.SOF_HOST2BOARD, s.hex()) for _, s in streams])
        for (name, s), c_line in zip(streams, c_lines):
            check("parse " + name, c_line, py_parse(HL.SOF_HOST2BOARD, s))

        # ---------------- 4. 半包 / 跨调用状态 ----------------
        print()
        print("【4/4】半包：同一条流分两次喂（C 用 S/PN 保持同一个 parser 会话）")
        half = len(f3) // 2
        c_a, c_b = h.run("S %02x" % HL.SOF_HOST2BOARD,
                         "PN %02x %s" % (HL.SOF_HOST2BOARD, f3[:half].hex()))
        c_c = h.run("PN %02x %s" % (HL.SOF_HOST2BOARD, f3[half:].hex()))[0]
        p_py = HL.LinkParser(HL.SOF_HOST2BOARD)
        py_a = py_parse_from(HL.SOF_HOST2BOARD, f3[:half], p_py)
        py_b = py_parse_from(HL.SOF_HOST2BOARD, f3[half:], p_py)
        check("半包第 1 段（应无可解帧）", c_b, py_a)
        check("半包第 2 段（应解出完整帧）", c_c, py_b)

        # 逐字节喂（模拟 USB 每次回调 1~64 字节）
        print()
        print("逐字节喂（模拟 CDC 回调每次 1 字节，最坏情况）")
        h.run("S %02x" % HL.SOF_HOST2BOARD)
        c_lines = []
        for i in range(len(f1 + f2)):
            c_lines += h.run("PN %02x %s" % (HL.SOF_HOST2BOARD, (f1 + f2)[i:i+1].hex()))
        p_py = HL.LinkParser(HL.SOF_HOST2BOARD)
        py_lines = [py_parse_from(HL.SOF_HOST2BOARD, (f1 + f2)[i:i+1], p_py)
                    for i in range(len(f1 + f2))]
        check("逐字节喂 %d 次后的最终状态" % len(c_lines), c_lines[-1], py_lines[-1])
        n_mismatch = sum(1 for a, b in zip(c_lines, py_lines) if a != b)
        check("逐字节喂的每一步都一致", "一致" if n_mismatch == 0 else "%d 步不一致" % n_mismatch,
              "一致")

    finally:
        h.close()
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    print("=" * 78)
    if FAIL:
        print("❌ %d 项不一致：%s" % (len(FAIL), ", ".join(FAIL)))
        print("   固件的 lgdog_link.c 与上位机 host_link.py 已经漂移，必须先对齐再上真机。")
        return 1
    print("✅ C 端与 Python 端编解码【逐字节一致】，协议无漂移")
    return 0


# ---------------------------------------------------------------- 造帧助手
def np_full(v):
    import numpy as np
    return np.full(12, v)


def make_imu():
    return struct.pack("<13fIB", 0.11, -0.22, 1.33, 0.1, -0.2, 0.3,
                       0.0, 0.0, 9.81, 1.0, 0.0, 0.0, 0.0, 987654, 1)


def make_chunk():
    import numpy as np
    out = struct.pack("<I", 4242)
    for _ in range(4):
        out += np.arange(6, dtype="<f4").tobytes()
    out += bytes([0, 1, 2, 0, 4, 0])
    return out


def make_status():
    # mode, enabled, pc_alive, err_flags, motor_missing, loop_us, rx, crc, drops, tick
    return struct.pack("<4BH5I", 1, 1, 1, 0, 0x0041, 512, 100, 0, 0, 987654)


def py_parse_from(sof, stream, parser):
    """用已有的 parser 实例解析，格式与 C 宿主一致。"""
    msgs = parser.parse(stream)
    s = "PARSE %d" % len(msgs)
    for m in msgs:
        s += " %02x:%s" % (m.msg_id, m.payload.hex())
    s += " CRC=%d LEN=%d SOF=%d" % (parser.crc_errors, parser.len_errors, parser.sof_drops)
    return s


def py_parse(sof, stream):
    """Python 侧解析结果，格式与 C 宿主一致。"""
    p = HL.LinkParser(sof)
    msgs = p.parse(stream)
    s = "PARSE %d" % len(msgs)
    for m in msgs:
        s += " %02x:%s" % (m.msg_id, m.payload.hex())
    s += " CRC=%d LEN=%d SOF=%d" % (p.crc_errors, p.len_errors, p.sof_drops)
    return s


if __name__ == "__main__":
    sys.exit(main())
