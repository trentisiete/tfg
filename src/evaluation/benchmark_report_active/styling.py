from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import seaborn as sns


HATCHES = ["", "//", "\\\\", "..", "xx", "--", "oo", "++"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]
LINESTYLES = ["-", "--", "-.", ":"]
IMAGE_SUFFIXES = {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".tif", ".tiff"}


def apply_publication_style(dpi: int = 300) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.8,
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def build_model_style_map(models: Iterable[str]) -> Dict[str, Dict[str, str]]:
    model_list = sorted(set(models))
    palette = sns.color_palette("colorblind", n_colors=max(3, len(model_list)))
    style_map: Dict[str, Dict[str, str]] = {}

    for i, m in enumerate(model_list):
        style_map[m] = {
            "color": palette[i % len(palette)],
            "hatch": HATCHES[i % len(HATCHES)],
            "marker": MARKERS[i % len(MARKERS)],
            "linestyle": LINESTYLES[i % len(LINESTYLES)],
        }
    return style_map


def place_legend(ax, outside: bool = True) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if outside:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    else:
        ax.legend(handles, labels, loc="best", frameon=True)


def sanitize_filename(value: str) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "item"


def _resolve_base_output_path(path: Path) -> Path:
    p = Path(path)
    # If suffix is an image extension, remove it and keep base stem.
    if p.suffix.lower() in IMAGE_SUFFIXES:
        return p.with_suffix("")
    return p


def save_figure(fig, path: Path, dpi: int = 300, save_svg: bool = False) -> None:
    base = _resolve_base_output_path(Path(path))
    base.parent.mkdir(parents=True, exist_ok=True)

    # Robust layout for dense scientific multipanel figures.
    try:
        fig.tight_layout(pad=1.25)
    except Exception:
        pass

    png_path = Path(str(base) + ".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.30)
    if save_svg:
        svg_path = Path(str(base) + ".svg")
        fig.savefig(svg_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
