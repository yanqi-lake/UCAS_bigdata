"""图表统一入口：中文字体、样式、保存。

字体来源：优先用本机 Windows 的微软雅黑（复制到 ~/.local/share/fonts），
其次 WSL 自带的 Noto CJK / 文泉驿。都找不到时退回 DejaVu（中文会显示为方块）。
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

from src import config  # noqa: E402

_FONT_CANDIDATES = [
    "~/.local/share/fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

_INSTALLED_FONT: str | None = None


def setup_font(verbose: bool = True) -> str | None:
    """注册中文字体并写入 rcParams；返回字体名（失败返回 None）。"""
    global _INSTALLED_FONT
    if _INSTALLED_FONT is not None:
        return _INSTALLED_FONT

    target = Path("~/.local/share/fonts/msyh.ttc").expanduser()
    windows_font = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    if not target.exists() and windows_font.exists():
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(windows_font, target)
        except OSError:
            pass

    for raw in _FONT_CANDIDATES:
        p = Path(raw).expanduser()
        if not p.exists():
            continue
        try:
            fm.fontManager.addfont(str(p))
            name = fm.FontProperties(fname=str(p)).get_name()
        except Exception:  # 字体损坏/格式不支持时继续尝试下一个
            continue
        plt.rcParams.update(
            {
                "font.family": name,
                "axes.unicode_minus": False,
                "figure.dpi": 110,
                "savefig.dpi": 150,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
        _INSTALLED_FONT = name
        if verbose:
            print(f"[plotting] 使用中文字体: {name} ({p})")
        return name

    warnings.warn("未找到中文字体，图中中文可能显示为方块。可将 C:\\Windows\\Fonts\\msyh.ttc 复制到 ~/.local/share/fonts/")
    _INSTALLED_FONT = ""
    return None


def save(fig, path: str | Path) -> Path:
    """保存并关闭图（自动建目录、去白边）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def new_fig(width: float = 8.0, height: float = 4.5):
    setup_font(verbose=False)
    return plt.subplots(figsize=(width, height))