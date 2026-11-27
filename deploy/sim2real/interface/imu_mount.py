#!/usr/bin/env python3
"""IMU 安装朝向变换 —— IMU 坐标系 → 机体坐标系（+X 前 / +Y 左 / +Z 上）

★★ 为什么必须单独一个文件、单独一层，而不是在各个后端里顺手乘一下 ★★

  下位机的 IMU 直接读的是【IMU 芯片自己的坐标系】。这个芯片在机身上是
  【转了 180° 绕 Z 装上去】的（用户 2026-09 确认：IMU 的 X 向后、Y 向右、Z 向上，
  装在机身几何中央）。而策略要的是机体系：

      obs[3:6] = 机体系角速度 × 0.25      ← 必须是机体系
      obs[6:9] = 投影重力（机体系）        ← 必须是机体系

  换算关系（spec.py 里已定义，这里实现）：
      v_body = M @ v_imu,      M = diag(-1, -1, +1)   ← 绕 Z 转 180°，det=+1
  说明：M 的行列式为 +1，是【正常旋转】不是镜像 —— 所以下面既可以用矩阵直接
  乘向量（角速度、加速度），也可以转成四元数去旋转姿态，两条路结果一致。

  姿态（容易写错的地方，务必看完）：
      设 IMU 报的是 R_wi（IMU 系 → 世界系，这是"姿态"的物理含义）
      我们要的是 R_wb（机体系 → 世界系），因为 policy_runner.build_obs 里
      quat_rotate_inverse 要求 quat 是【机体系 → 世界系】。
          v_world = R_wi · v_imu = R_wi · Mᵀ · v_body   （因为 v_imu = Mᵀ v_body）
      ⇒   R_wb = R_wi · Mᵀ
      即四元数形式：  q_wb = q_wi ⊗ conj(q_M)
      ★ 注意是【右乘】共轭。写成左乘（q_M ⊗ q_wi）在世界系里转，是错的：
        那样算出来的 roll/pitch 会随 yaw 变化，静置时你会发现 pg 在一个圆上漂。

  这一层不变量在最坏情况下的表现：装反了不报错，机器人一上电就往一边倒。
  所以下面附了自测，把"水平静置时变换后仍水平"这件事直接断言掉。

  用 `tools/imu_check.py --port /dev/ttyACM0` 可以在真机上用加计做地面真值
  验证三种欧拉角约定 + 本变换，属于上电前的必做项。
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import spec  # noqa: E402
from interface.imu import ImuSample, ImuSource  # noqa: E402


# ============================================================================
# 四元数/旋转矩阵小工具（只依赖 numpy，不引 torch）
# ============================================================================
def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_conj(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b) -> np.ndarray:
    """a ⊗ b（wxyz）。语义：先 b 再 a，即 R(a⊗b) = R(a)·R(b)。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_matrix(q) -> np.ndarray:
    """wxyz 四元数 → 旋转矩阵（3×3）。"""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quat(R) -> np.ndarray:
    """旋转矩阵 → wxyz 四元数（Shepperd 分支法，数值稳定）。"""
    R = np.asarray(R, dtype=np.float64)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return quat_normalize(q)


def quat_rotate(q, v) -> np.ndarray:
    """v_world = R(q) · v。"""
    return quat_to_matrix(q) @ np.asarray(v, dtype=np.float64)


def attitude_error_deg(qa, qb) -> float:
    """两个姿态之间的夹角 [度]，用于判定"两种约定哪个对"。"""
    qa = quat_normalize(qa)
    qb = quat_normalize(qb)
    d = abs(float(np.dot(qa, qb)))
    d = min(1.0, max(-1.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


# ============================================================================
class ImuMountTransform(ImuSource):
    """把一个 IMU 系数据源包成【机体系】数据源（角速度、姿态、加速度都变换）。

    :param inner:  下层的 ImuSource（host_link 适配器 / usb_euler / stub / derived…）
    :param matrix: 安装矩阵 M（v_body = M @ v_imu）。默认取 spec.imu_mount_matrix()。
    :param tag:    source_tag 后缀，便于日志里看清是哪一层。
    """

    def __init__(self, inner: ImuSource, matrix=None, tag: str = "body"):
        self.inner = inner
        self.M = (spec.imu_mount_matrix() if matrix is None
                  else np.asarray(matrix, dtype=np.float64))
        if self.M.shape != (3, 3):
            raise ValueError("安装矩阵必须是 3×3")
        det = float(np.linalg.det(self.M))
        if abs(det - 1.0) > 1e-6:
            # det = -1 是镜像（比如把某一行符号写错），它不可能由"装歪了"产生
            raise ValueError(
                "安装矩阵行列式 = %.6f，应当是 +1（纯旋转）。det = -1 说明符号写错了，"
                "例如应为 diag(-1,-1,+1) 而写成了 diag(-1,-1,-1)。" % det)
        self._q_M_inv = quat_conj(matrix_to_quat(self.M))   # conj(q_M)：右乘用
        self.tag = tag
        self.applied = 0

    # ---------------------------------------------------------------- 变换
    def transform_sample(self, s: ImuSample) -> ImuSample:
        gyro = None if s.gyro is None else self.M @ np.asarray(s.gyro, dtype=np.float64)
        accel = None if s.accel is None else self.M @ np.asarray(s.accel, dtype=np.float64)
        quat = None
        if s.quat_wxyz is not None:
            # R_wb = R_wi · Mᵀ  ⇔  q_wb = q_wi ⊗ conj(q_M)   （右乘！见文件头）
            quat = quat_normalize(quat_mul(quat_normalize(s.quat_wxyz), self._q_M_inv))
        return ImuSample(gyro=gyro, quat_wxyz=quat, accel=accel,
                         source_tag="%s+%s" % (s.source_tag, self.tag),
                         recv_time=s.recv_time)

    # ---------------------------------------------------------------- 接口
    def start(self, timeout: float = 8.0):
        return self.inner.start(timeout)

    def get_latest(self) -> ImuSample:
        s = self.inner.get_latest()
        out = self.transform_sample(s)
        self.applied += 1
        return out

    def feed_usb_frame(self, frame: bytes) -> bool:
        return self.inner.feed_usb_frame(frame)

    def stop(self):
        self.inner.stop()

    def describe(self) -> str:
        return ("IMU 安装变换: v_body = M @ v_imu, M =\n%s\n"
                "  （IMU 的 X 向后、Y 向右 → 绕 Z 转 180°；姿态做 q_wi ⊗ conj(q_M) 右乘共轭）"
                % np.array2string(self.M, precision=0))


# ============================================================================
def _selftest():
    print("=" * 72)
    print("ImuMountTransform 自测")
    print("=" * 72)

    M = spec.imu_mount_matrix()
    print("spec 安装矩阵 M =")
    print(np.array2string(M, precision=0))
    assert abs(np.linalg.det(M) - 1.0) < 1e-9, "M 必须是纯旋转（det=+1）"
    assert np.allclose(M @ M.T, np.eye(3)), "M 必须是正交阵"
    print("  det(M)=+1、M·Mᵀ=I ✓（是旋转，不是镜像）")

    # ---- 1) 四元数旋转 与 矩阵乘 必须一致 ----
    q_M = matrix_to_quat(M)
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.normal(size=3)
        assert np.allclose(quat_rotate(q_M, v), M @ v, atol=1e-12)
    print("  q_M 的旋转 与 M @ v 一致 ✓（姿态与向量两条路自洽）")

    # ---- 2) 角速度变换 == spec.imu_vec_to_body ----
    tr = ImuMountTransform(inner=None)      # 只测 transform_sample，不碰 inner
    for _ in range(50):
        w = rng.normal(size=3)
        s = ImuSample(gyro=w, quat_wxyz=np.array([1.0, 0, 0, 0]),
                      accel=rng.normal(size=3), source_tag="t")
        o = tr.transform_sample(s)
        assert np.allclose(o.gyro, spec.imu_vec_to_body(w), atol=1e-12)
    print("  角速度变换 == spec.imu_vec_to_body(g) ✓")

    # ---- 3) ★核心不变量：IMU 系下水平 ⇒ 机体系下也水平（pg 仍为 (0,0,-1)）----
    from tools.math_utils import quat_rotate_inverse
    s = ImuSample(gyro=np.zeros(3), quat_wxyz=np.array([1.0, 0, 0, 0]),
                  accel=np.array([0.0, 0.0, 9.81]), source_tag="t")
    o = tr.transform_sample(s)
    pg = quat_rotate_inverse(o.quat_wxyz, np.array([0.0, 0.0, -1.0]))
    print("  静置水平: q_body = %s  →  pg = %s"
          % (np.round(o.quat_wxyz, 6), np.round(pg, 6)))
    assert np.allclose(pg, [0.0, 0.0, -1.0], atol=1e-12), \
        "水平静置变换后 pg 不是 (0,0,-1) —— 姿态变换写错了（左乘 vs 右乘？）"
    # 四元数有双重覆盖：(0,0,0,1) 与 (0,0,0,-1) 表示同一个旋转，所以比姿态夹角而不是比分量
    assert attitude_error_deg(o.quat_wxyz, [0.0, 0.0, 0.0, 1.0]) < 1e-6, \
        "水平时 body 姿态应为绕 Z 180°"
    print("  水平静置变换后仍水平 ✓（这条错了机器人一定一上电就往一边倒）")

    # ---- 4) 左乘 vs 右乘的反例：绕 IMU X 轴倾 10° ----
    # IMU_X = 机体 −X ⇒ 在 IMU 系里绕 X 倾 10°，等于机体绕 −X 横滚 10°。
    # 期望：pg = (0, +sin10°, −cos10°)（用 math_utils 的约定反推过）
    ang = np.radians(10.0)
    q_tilt = np.array([np.cos(ang / 2), np.sin(ang / 2), 0.0, 0.0])
    o = tr.transform_sample(ImuSample(quat_wxyz=q_tilt, source_tag="t"))
    pg = quat_rotate_inverse(o.quat_wxyz, np.array([0.0, 0.0, -1.0]))
    roll = np.degrees(np.arctan2(-float(pg[1]), max(1e-9, -float(pg[2]))))
    pitch = np.degrees(np.arctan2(float(pg[0]),
                                   np.sqrt(float(pg[1]) ** 2 + float(pg[2]) ** 2)))
    print("  绕 IMU-X 倾 10°: pg = %s  → body roll=%+.2f° pitch=%+.2f°"
          % (np.round(pg, 4), roll, pitch))
    assert abs(roll + 10.0) < 1e-6 and abs(pitch) < 1e-6, \
        "符号不对：IMU_X = 机体 −X，所以在 IMU 系绕 X 倾 +10° 应当得到 body roll = −10°"
    right = quat_mul(q_tilt, quat_conj(q_M))
    wrong = quat_mul(quat_conj(q_M), q_tilt)
    assert np.allclose(o.quat_wxyz, right, atol=1e-12)
    err_left = attitude_error_deg(right, wrong)
    print("  右乘(正确)与左乘(错误)在这组姿态上差 %.2f° —— 静置水平时两者恰好相同，"
          % err_left)
    print("  所以【只测水平是分辨不出来的】，必须用倾斜姿态验证（imu_check.py 就是干这个的）")
    assert err_left > 1.0, "这组测试姿态分辨不出左右乘，换一组"

    # ---- 5) det=-1 必须被拦住 ----
    try:
        ImuMountTransform(inner=None, matrix=np.diag([-1.0, -1.0, -1.0]))
    except ValueError as e:
        print("  镜像矩阵被拦住 ✓ (%s)" % str(e).splitlines()[0][:48] + "…)")
    else:
        raise AssertionError("det=-1 的矩阵没有被拦住")

    print()
    print("✅ imu_mount 自测通过")


if __name__ == "__main__":
    _selftest()
