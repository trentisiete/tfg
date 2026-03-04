from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .styling import build_model_style_map, place_legend, save_figure


METRICS: List[Tuple[str, str]] = [
    ("mae", "MAE"),
    ("coverage_95", "Coverage 95%"),
    ("nlpd", "NLPD"),
    ("r2", "R2"),
]

def _add_sampler_summary_box(ax, summary_lines: List[str]) -> None:
    if not summary_lines:
        return
    text = "\n".join(summary_lines)
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#666666", "alpha": 0.78},
    )


def _plot_benchmark_sampler_mae(
    benchmark: str,
    block: pd.DataFrame,
    style_map,
    out_dir: Path,
    dpi: int,
    save_svg: bool,
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)
    legend_handles = []
    legend_labels = []

    for i, sampler in enumerate(["sobol", "random"]):
        ax = axs[i]
        sb = block[block["sampler"] == sampler]
        if sb.empty:
            ax.set_axis_off()
            continue

        grouped = (
            sb.groupby(["n_train_current", "model"], as_index=False)["mae"]
            .mean()
            .sort_values("n_train_current")
        )
        for model, mb in grouped.groupby("model"):
            st = style_map[model]
            line = ax.plot(
                mb["n_train_current"],
                mb["mae"],
                marker=st["marker"],
                linestyle=st["linestyle"],
                color=st["color"],
                label=model,
            )[0]
            if model not in legend_labels:
                legend_labels.append(model)
                legend_handles.append(line)
        ax.set_title(f"{benchmark} | sampler={sampler}")
        ax.set_xlabel("n_train_current")
        ax.set_ylabel("MAE")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=min(5, len(legend_labels)),
            frameon=True,
        )
    fig.subplots_adjust(top=0.86, wspace=0.12)
    fig.suptitle(f"Comparacion Sobol vs Random - Evolucion MAE ({benchmark})")
    save_figure(fig, out_dir / f"{benchmark}_sobol_vs_random_evolucion_mae", dpi=dpi, save_svg=save_svg)


def _plot_benchmark_sampler_multimetric(
    benchmark: str,
    block: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    save_svg: bool,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(15.5, 10.0), sharex=True)
    axs = axs.ravel()

    for idx, (metric, ylabel) in enumerate(METRICS):
        ax = axs[idx]
        if metric not in block.columns:
            ax.set_axis_off()
            continue

        grouped = (
            block.groupby(["sampler", "n_train_current"], as_index=False)[metric]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "metric_mean", "std": "metric_std"})
        )
        summary_lines: List[str] = []
        for _, (sampler, sb) in enumerate(grouped.groupby("sampler")):
            sb = sb.sort_values("n_train_current")
            x = sb["n_train_current"].to_numpy()
            y = sb["metric_mean"].to_numpy()
            s = sb["metric_std"].fillna(0.0).to_numpy()
            ax.plot(x, y, marker="o", label=str(sampler))
            ax.fill_between(x, y - s, y + s, alpha=0.20)
            if metric == "mae":
                if len(y) > 0:
                    summary_lines.append(f"{sampler}: init={y[0]:.3g} | final={y[-1]:.3g}")

        if metric == "mae":
            _add_sampler_summary_box(ax=ax, summary_lines=summary_lines)

        ax.set_title(f"{ylabel} por sampler")
        ax.set_xlabel("n_train_current")
        ax.set_ylabel(ylabel)
        place_legend(ax, outside=False)

    fig.suptitle(f"Sampling effects multimetric ({benchmark})")
    save_figure(fig, out_dir / f"{benchmark}_sampling_effects_multimetric", dpi=dpi, save_svg=save_svg)


def generate_sampling_effects_plots(
    master_df: pd.DataFrame,
    final_df: pd.DataFrame,
    out_dir: Path,
    dpi: int = 300,
    save_svg: bool = False,
) -> None:
    if master_df.empty:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samplers = sorted(master_df["sampler"].dropna().unique().tolist())
    if len(samplers) < 2:
        return

    style_map = build_model_style_map(master_df["model"].unique())

    # Per benchmark: sobol vs random evolution (MAE by model).
    for benchmark, block in master_df.groupby("benchmark", dropna=False):
        _plot_benchmark_sampler_mae(
            benchmark=str(benchmark),
            block=block,
            style_map=style_map,
            out_dir=out_dir,
            dpi=dpi,
            save_svg=save_svg,
        )
        _plot_benchmark_sampler_multimetric(
            benchmark=str(benchmark),
            block=block,
            out_dir=out_dir,
            dpi=dpi,
            save_svg=save_svg,
        )

    # Aggregated initial/final MAE by sampler and model.
    g = (
        master_df.sort_values("n_train_current")
        .groupby(["sampler", "model"], as_index=False)
        .agg(mae_inicial=("mae", "first"), mae_final=("mae", "last"))
    )
    if not g.empty:
        fig, ax = plt.subplots(figsize=(10.8, 5.8))
        for model, block in g.groupby("model"):
            st = style_map[model]
            ax.plot(
                block["sampler"],
                block["mae_final"],
                marker=st["marker"],
                linestyle=st["linestyle"],
                color=st["color"],
                label=f"{model} (final)",
            )
            ax.scatter(
                block["sampler"],
                block["mae_inicial"],
                marker="x",
                color=st["color"],
                s=70,
                label=f"{model} (inicial)",
            )
        ax.set_title("Sampling effects: MAE inicial y final por sampler")
        ax.set_xlabel("Sampler")
        ax.set_ylabel("MAE")
        place_legend(ax, outside=True)
        save_figure(fig, out_dir / "sampling_effects_mae_inicial_final", dpi=dpi, save_svg=save_svg)

    # Additional global comparison per sampler (all key metrics).
    for metric, ylabel in METRICS:
        if metric not in final_df.columns:
            continue
        g2 = final_df.groupby(["sampler", "model"], as_index=False).agg(metric_value=(metric, "mean"))
        if g2.empty:
            continue
        fig, ax = plt.subplots(figsize=(10.8, 5.8))
        for sampler in sorted(g2["sampler"].unique()):
            sb = g2[g2["sampler"] == sampler].sort_values("metric_value")
            ax.plot(sb["model"], sb["metric_value"], marker="o", label=f"{sampler}")
        ax.set_title(f"{ylabel} final medio por modelo y sampler")
        ax.set_xlabel("Modelo")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        place_legend(ax, outside=True)
        save_figure(fig, out_dir / f"sampler_vs_model_{metric}_final", dpi=dpi, save_svg=save_svg)
