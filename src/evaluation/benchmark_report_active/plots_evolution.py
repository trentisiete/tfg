from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .styling import build_model_style_map, place_legend, save_figure


EVOLUTION_METRICS = [
    ("mae", "MAE"),
    ("coverage_95", "Coverage 95%"),
    ("nlpd", "NLPD"),
]


def _plot_noise_grid(
    bench_df: pd.DataFrame,
    benchmark: str,
    metric: str,
    ylabel: str,
    out_path: Path,
    style_map,
    dpi: int,
    save_svg: bool,
) -> None:
    noises: List[str] = sorted(bench_df["noise"].dropna().unique().tolist())
    if not noises:
        return

    nrows = len(noises)
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(12.0, 3.4 * nrows), sharex=True)
    if nrows == 1:
        axs = [axs]

    for i, noise in enumerate(noises):
        ax = axs[i]
        block = bench_df[bench_df["noise"] == noise]
        if block.empty:
            ax.set_axis_off()
            continue

        grouped = (
            block.groupby(["n_train_current", "model", "sampler"], as_index=False)[metric]
            .mean()
            .sort_values("n_train_current")
        )
        for (model, sampler), sb in grouped.groupby(["model", "sampler"]):
            st = style_map[model]
            label = f"{model} | {sampler}"
            ax.plot(
                sb["n_train_current"],
                sb[metric],
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                label=label,
            )
        ax.set_title(f"{benchmark} | ruido={noise}")
        ax.set_ylabel(ylabel)
        place_legend(ax, outside=True)

    axs[-1].set_xlabel("n_train_current")
    fig.suptitle(f"Evolucion de {ylabel} por ruido ({benchmark})")
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _plot_aggregated_curve(
    bench_df: pd.DataFrame,
    benchmark: str,
    metric: str,
    ylabel: str,
    out_path: Path,
    style_map,
    dpi: int,
    save_svg: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    g = (
        bench_df.groupby(["n_train_current", "model"], as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "metric_mean", "std": "metric_std"})
    )
    if g.empty:
        plt.close(fig)
        return
    for model, mb in g.groupby("model"):
        st = style_map[model]
        x = mb["n_train_current"].to_numpy()
        y = mb["metric_mean"].to_numpy()
        s = mb["metric_std"].fillna(0.0).to_numpy()
        ax.plot(x, y, color=st["color"], marker=st["marker"], linestyle=st["linestyle"], label=model)
        ax.fill_between(x, y - s, y + s, color=st["color"], alpha=0.18)

    ax.set_title(f"Evolucion agregada de {ylabel} (media ± std) - {benchmark}")
    ax.set_xlabel("n_train_current")
    ax.set_ylabel(ylabel)
    place_legend(ax, outside=True)
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def generate_evolution_plots(
    master_df: pd.DataFrame,
    out_dir: Path,
    dpi: int = 300,
    save_svg: bool = False,
) -> None:
    if master_df.empty:
        return
    out_dir = Path(out_dir)
    style_map = build_model_style_map(master_df["model"].unique())

    for benchmark, bench_df in master_df.groupby("benchmark", dropna=False):
        bdir = out_dir / str(benchmark)
        bdir.mkdir(parents=True, exist_ok=True)

        for metric, ylabel in EVOLUTION_METRICS:
            if metric not in bench_df.columns:
                continue
            _plot_noise_grid(
                bench_df=bench_df,
                benchmark=str(benchmark),
                metric=metric,
                ylabel=ylabel,
                out_path=bdir / f"evolucion_{metric}_filas_por_ruido",
                style_map=style_map,
                dpi=dpi,
                save_svg=save_svg,
            )
            _plot_aggregated_curve(
                bench_df=bench_df,
                benchmark=str(benchmark),
                metric=metric,
                ylabel=ylabel,
                out_path=bdir / f"evolucion_{metric}_agregado_media_std",
                style_map=style_map,
                dpi=dpi,
                save_svg=save_svg,
            )
