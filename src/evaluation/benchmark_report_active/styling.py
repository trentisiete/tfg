from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import seaborn as sns


HATCHES = ["", "//", "\\\\", "..", "xx", "--", "oo", "++"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*"]
LINESTYLES = ["-", "--", "-.", ":"]


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


def save_figure(fig, path: Path, dpi: int = 300, save_svg: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if save_svg:
        fig.savefig(path.with_suffix(".svg"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
