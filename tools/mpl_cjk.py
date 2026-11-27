# -*- coding: utf-8 -*-
"""mpl_cjk.py — matplotlib 中文字体设置（一处配置，所有画图脚本共用）。

为什么需要它（2026-10-13 实测踩到）:
  matplotlib 默认字体是 DejaVu Sans，**没有任何中文字形** ⇒ 图里所有中文/℃/≈ 之类
  会变成方框（用户看到的就是"乱码"）。而中文标标签在本项目里到处都是（轴名、图例、标题），
  所以每个画图脚本都必须先调用本模块的 setup()。

用法:
  from mpl_cjk import setup
  setup()                      # 在 import matplotlib.pyplot 之前或之后都行
  import matplotlib.pyplot as plt
"""
import warnings

_CANDIDATES = [
    'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Sans CJK TC', 'Noto Sans CJK HK',
    'Source Han Sans SC', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
    'Droid Sans Fallback', 'AR PL UMing CN', 'AR PL UKai CN',
    'Microsoft YaHei', 'SimHei',
]


def setup(verbose=False):
    """把 matplotlib 的字体设成【系统里真实存在】的中文字体；找不到就明确警告。"""
    import matplotlib
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in _CANDIDATES if n in have), None)
    if chosen is None:
        warnings.warn('mpl_cjk: 系统里找不到任何候选中文字体，图里的中文会变方框。'
                      '装字体: sudo apt install fonts-noto-cjk')
        return None
    matplotlib.rcParams['font.sans-serif'] = [chosen] + matplotlib.rcParams.get('font.sans-serif', [])
    matplotlib.rcParams['font.family'] = 'sans-serif'
    # ★ 负号: DejaVu/Noto 都能画 −，但显式关掉 unicode_minus 更保险（否则刻度上的负号也可能方框）
    matplotlib.rcParams['axes.unicode_minus'] = False
    # ★ 预热字体缓存: 换字体后的第一次 findfont 会触发缓存重建，可能抛异常/报"找不到"
    #   （2026-10-13 实测: 同一段检查第一次失败、第二次却成功）。这里主动预热并把缓存问题吃掉。
    try:
        font_manager.findfont(font_manager.FontProperties(family=chosen))
    except Exception:
        try:
            font_manager._load_fontmanager(try_read_cache=False)   # 强制重建缓存
            font_manager.findfont(font_manager.FontProperties(family=chosen))
        except Exception:
            pass
    if verbose:
        print('mpl_cjk: 使用中文字体 %s' % chosen)
    return chosen


def resolved_font():
    """返回 matplotlib 画中文时会用的字体名（best-effort）。

    ★ 刻意【不走 font_manager.findfont】: 实测它在同一进程里会因为字体缓存重建而时好时坏
      （2026-10-13: 同一段代码第一次抛异常、第二次成功），拿它当判据会误报。
      直接读 rcParams 更可靠 —— 那正是 matplotlib 排版文字时实际使用的第一顺位字体。
    """
    import matplotlib
    fam = matplotlib.rcParams.get('font.sans-serif') or []
    return fam[0] if fam else None


def check_glyphs(fig=None, strict=False):
    """交付前自查（默认只警告、不抛异常，避免因为 matplotlib 字体缓存的怪行为把画图整个搞挂）。

    返回 (字体文件名, 缺失字形列表)。strict=True 时若字体不是中文或缺字形就抛异常。
    ★ 真正的保证是 setup() 里【显式指定中文字体】；本函数只是把它打印/记录下来供核查。
    """
    import matplotlib
    import matplotlib.pyplot as plt
    font = resolved_font()
    # ★ 归一化后再匹配: 字体名带空格（如 'Noto Sans CJK JP'），直接子串匹配会漏判（踩过）
    cjk_markers = ('notosanscjk', 'notoserifcjk', 'sourcehansans', 'wqy', 'wenquanyi',
                   'droidsansfallback', 'uming', 'ukai', 'msyh', 'simhei', 'notosanssc')
    key = (font or '').lower().replace(' ', '').replace('-', '')
    ok_font = any(m in key for m in cjk_markers)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            (fig or plt.gcf()).canvas.draw()
        miss = [str(x.message) for x in w if 'missing from font' in str(x.message)]
    except Exception:
        miss = []
    if not ok_font:
        msg = ('当前字体 %s 不是中文字体 ⇒ 图里的中文会变方框（请先调 mpl_cjk.setup()）' % font)
        warnings.warn('mpl_cjk: ' + msg)
        if strict:
            raise RuntimeError(msg)
    if miss:
        warnings.warn('mpl_cjk: 图里有字形缺失: %s' % '; '.join(miss[:3]))
        if strict:
            raise RuntimeError('字形缺失: %s' % '; '.join(miss[:3]))
    return font, miss
