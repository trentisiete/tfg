from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .styling import build_model_style_map, place_legend, save_figure


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
        # Keep behavior explicit: no crash, just skip.
        return

    style_map = build_model_style_map(master_df["model"].unique())

    # Per benchmark: sobol vs random evolution (MAE)
    for benchmark, block in master_df.groupby("benchmark", dropna=False):
        fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
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
                ax.plot(
                    mb["n_train_current"],
                    mb["mae"],
                    marker=st["marker"],
                    linestyle=st["linestyle"],
                    color=st["color"],
                    label=model,
                )
            ax.set_title(f"{benchmark} | sampler={sampler}")
            ax.set_xlabel("n_train_current")
            ax.set_ylabel("MAE")
            place_legend(ax, outside=True)
        fig.suptitle(f"Comparacion Sobol vs Random - Evolucion MAE ({benchmark})")
        save_figure(fig, out_dir / f"{benchmark}_sobol_vs_random_evolucion_mae", dpi=dpi, save_svg=save_svg)

    # Aggregated initial/final MAE by sampler and model
    g = (
        master_df.sort_values("n_train_current")
        .groupby(["sampler", "model"], as_index=False)
        .agg(mae_inicial=("mae", "first"), mae_final=("mae", "last"))
    )
    if not g.empty:
        fig, ax = plt.subplots(figsize=(10.0, 5.5))
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

    # Additional global comparison per sampler
    g2 = final_df.groupby(["sampler", "model"], as_index=False).agg(mae=("mae", "mean"))
    if not g2.empty:
        fig, ax = plt.subplots(figsize=(10.0, 5.5))
        for sampler in sorted(g2["sampler"].unique()):
            sb = g2[g2["sampler"] == sampler].sort_values("mae")
            ax.plot(sb["model"], sb["mae"], marker="o", label=f"{sampler}")
        ax.set_title("MAE final medio por modelo y sampler")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("MAE")
        ax.tick_params(axis="x", rotation=30)
        place_legend(ax, outside=True)
        save_figure(fig, out_dir / "sampler_vs_model_mae_final", dpi=dpi, save_svg=save_svg)
