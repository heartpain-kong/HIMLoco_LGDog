# -*- coding: utf-8 -*-
"""check_config.py — 配置体检（改完配置先跑它，能挡住本项目踩过的几类坑）

检查项（每一项都对应历史上真实发生过的事故）：
  1. 【类内重复赋值】ast 扫描 config 文件：同名赋值后者静默生效
     —— 2026-09-26 的 clearance_form 事故（我写的 'terrain' 被下面的 'relu_swing' 覆盖）。
  2. 【权重 ↔ 奖励函数配对】scales 里的每个键必须在 legged_robot.py 有 _reward_<键>；
     反之有函数但没注册权重只是不生效（提示，不算错误）。
  3. 【代码引用的参数是否存在】legged_robot.py 里 self.cfg.rewards.<attr> 必须在 config 里定义；
     若代码用了 getattr(..., 默认值) 则记为"有兜底"。
  4. 【数值合理性】权重符号/量级明显异常的提示（如 step_length 应为正、cap 应为正）。
  5. 【指令课程耦合】打印指令课程推进门槛（历史口径 = 0.8 × tracking_lin_vel 权重）。

用法: python tools/check_config.py
"""
import ast, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, 'legged_gym/legged_gym/envs/LgDog1/LgDog1_config.py')
BASE_CFG = os.path.join(ROOT, 'legged_gym/legged_gym/envs/base/legged_robot_config.py')
ROBOT = os.path.join(ROOT, 'legged_gym/legged_gym/envs/base/legged_robot.py')


def audit_duplicates(paths):
    total = 0
    for path in paths:
        tree = ast.parse(open(path, encoding='utf-8').read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                seen = {}
                for st in node.body:
                    if isinstance(st, ast.Assign):
                        for t in st.targets:
                            if isinstance(t, ast.Name):
                                if t.id in seen:
                                    print(f"  ❌ 重复赋值 {os.path.basename(path)}::{node.name}.{t.id} "
                                          f"(行{seen[t.id]} 被 行{st.lineno} 覆盖)")
                                    total += 1
                                seen[t.id] = st.lineno
    print(f"  {'✔ 无重复赋值' if total == 0 else f'共 {total} 处重复'}")
    return total


def main():
    os.environ.setdefault('LD_LIBRARY_PATH', os.path.expanduser('~/miniconda3/envs/himloco/lib'))
    sys.path.insert(0, os.path.join(ROOT, 'legged_gym'))
    import isaacgym  # noqa: F401  (必须先于 torch)
    from legged_gym.envs.LgDog1.LgDog1_config import LgDog1Cfg as C

    src = open(ROBOT, encoding='utf-8').read()
    reward_funcs = set(re.findall(r'def (_reward_\w+)\(', src))
    scales = {k: v for k, v in vars(C.rewards.scales).items() if not k.startswith('_')}

    print("=" * 74)
    print("1. 类内重复赋值审计")
    print("=" * 74)
    bad = audit_duplicates([CFG, BASE_CFG])

    print("\n" + "=" * 74)
    print("2. 权重 ↔ 奖励函数配对")
    print("=" * 74)
    missing_fn = []
    for k, v in sorted(scales.items()):
        if f'_reward_{k}' not in reward_funcs:
            if v not in (0, 0.0, -0.0):
                print(f"  ❌ scales.{k} = {v} 但 legged_robot.py 里没有 _reward_{k}（启动即报错）")
                missing_fn.append(k)
            else:
                print(f"  ⚠️ scales.{k} = {v} 无对应函数（权重 0，暂不报错，但建议删掉）")
    if not missing_fn:
        print("  ✔ 所有非零权重都有对应奖励函数")
    unregistered = sorted(f[8:] for f in reward_funcs
                          if f[8:] not in scales and f[8:] not in ('log',))
    if unregistered:
        print(f"  ℹ️ 有函数未注册权重（不生效）: {', '.join(unregistered)}")

    print("\n" + "=" * 74)
    print("3. 代码引用的 rewards 参数是否存在")
    print("=" * 74)
    defined = {k for k in vars(C.rewards) if not k.startswith('_')}
    # ★用 AST 而不是正则: 正则会把【注释里】的 self.cfg.rewards.x 也算进来(本工具第一版就这样误报过)。
    #   同时区分 getattr(有默认值=有兜底) 与 getattr(无默认值=仍会报错)。
    tree = ast.parse(src)
    skip_ids, used_plain, used_getattr = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'getattr':
            if node.args and isinstance(node.args[0], ast.Attribute) and node.args[0].attr == 'rewards':
                skip_ids.add(id(node.args[0]))
                if len(node.args) >= 3 and isinstance(node.args[1], ast.Constant):
                    used_getattr.add(node.args[1].value)
                elif len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    used_plain.add(node.args[1].value)          # getattr 无默认值 → 同样会炸
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and id(node) not in skip_ids
                and isinstance(node.value, ast.Attribute) and node.value.attr == 'rewards'
                and isinstance(node.value.value, ast.Name) and node.value.value.id == 'self'):
            used_plain.add(node.attr)
    miss = sorted(used_plain - defined)
    for m in miss:
        print(f"  ❌ 代码用 self.cfg.rewards.{m} 但 config 未定义（运行到即 AttributeError）")
    for g in sorted((used_getattr | (used_getattr & defined)) - defined):
        print(f"  ⚠️ self.cfg.rewards.{g} 未定义，但有 getattr 兜底（用代码内默认值）")
    if not miss:
        print("  ✔ 代码直接引用的 rewards 参数全部已定义")

    print("\n" + "=" * 74)
    print("4. 数值合理性")
    print("=" * 74)
    notes = 0
    for k in ('step_length',):
        if k in scales and scales[k] <= 0:
            print(f"  ⚠️ scales.{k} = {scales[k]} 应为【正】(它是奖励项)"); notes += 1
    cap = getattr(C.rewards, 'step_length_cap', None)
    if cap is not None and cap <= 0:
        print(f"  ⚠️ step_length_cap = {cap} 应为正"); notes += 1
    for k in ('orientation', 'ang_vel_xy', 'lin_vel_z', 'base_height', 'smoothness', 'action_rate'):
        if k in scales and scales[k] > 0:
            print(f"  ⚠️ scales.{k} = {scales[k]} 应为【负】(它是惩罚项)"); notes += 1
    if not notes:
        print("  ✔ 权重符号未发现异常")

    print("\n" + "=" * 74)
    print("5. 关键耦合提醒")
    print("=" * 74)
    tl = scales.get('tracking_lin_vel', 0)
    print(f"  · 指令课程推进门槛 = 0.8 × tracking_lin_vel = {0.8*tl:.2f} "
          f"（权重越低课程越易推进；跨轮比较 max_command_x 时注意基线变化）")
    if 'step_length' in scales:
        print(f"  · step_length={scales['step_length']}（正=奖励步幅）; cap={cap} m")
    else:
        print("  · step_length 未注册（步幅奖励关闭）")

    print("\n" + ("=" * 74))
    print(f"结论: {'✅ 未发现硬错误' if not bad and not miss else '❌ 存在硬错误，见上'}")
    return 0 if (not bad and not miss) else 1


if __name__ == '__main__':
    sys.exit(main())
