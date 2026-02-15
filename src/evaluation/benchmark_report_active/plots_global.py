from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .styling import build_model_style_map, place_legend, save_figure


def generate_global_plots(
    final_df: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    out_dir: Path,
    overview_dir: Path,
    dpi: int = 300,
    save_svg: bool = False,
) -> None:
    if final_df.empty:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overview_dir.mkdir(parents=True, exist_ok=True)
    style_map = build_model_style_map(final_df["model"].unique())

    # Violin MAE
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    order = final_df.groupby("model")["mae"].mean().sort_values().index.tolist()
    data = [final_df.loc[final_df["model"] == m, "mae"].dropna().values for m in order]
    vp = ax.violinplot(data, showmeans=True, showextrema=True)
    for i, body in enumerate(vp["bodies"]):
        model = order[i]
        body.set_facecolor(style_map[model]["color"])
        body.set_alpha(0.75)
        body.set_edgecolor("black")
        body.set_hatch(style_map[model]["hatch"])
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_title("Distribucion global de MAE por modelo")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("MAE")
    save_figure(fig, out_dir / "violin_mae_global", dpi=dpi, save_svg=save_svg)

    # Violin RMSE
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    order_rmse = final_df.groupby("model")["rmse"].mean().sort_values().index.tolist()
    data_rmse = [final_df.loc[final_df["model"] == m, "rmse"].dropna().values for m in order_rmse]
    vp = ax.violinplot(data_rmse, showmeans=True, showextrema=True)
    for i, body in enumerate(vp["bodies"]):
        model = order_rmse[i]
        body.set_facecolor(style_map[model]["color"])
        body.set_alpha(0.75)
        body.set_edgecolor("black")
        body.set_hatch(style_map[model]["hatch"])
    ax.set_xticks(np.arange(1, len(order_rmse) + 1))
    ax.set_xticklabels(order_rmse, rotation=25, ha="right")
    ax.set_title("Distribucion global de RMSE por modelo")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("RMSE")
    save_figure(fig, out_dir / "violin_rmse_global", dpi=dpi, save_svg=save_svg)

    # Heatmap global benchmark x model
    pivot_global = final_df.pivot_table(
        index="benchmark", columns="model", values="mae", aggfunc="mean"
    )
    if not pivot_global.empty:
        fig, ax = plt.subplots(figsize=(1.1 * len(pivot_global.columns) + 4, 0.6 * len(pivot_global.index) + 2))
        sns.heatmap(pivot_global, annot=True, fmt=".3g", cmap="viridis_r", ax=ax, cbar_kws={"label": "MAE"})
        ax.set_title("Heatmap MAE promedio (benchmark x modelo)")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Benchmark")
        save_figure(fig, out_dir / "heatmap_mae_global_promedio", dpi=dpi, save_svg=save_svg)

    # Heatmap by noise
    for noise, block in final_df.groupby("noise", dropna=False):
        piv = block.pivot_table(index="benchmark", columns="model", values="mae", aggfunc="mean")
        if piv.empty:
            continue
        fig, ax = plt.subplots(figsize=(1.1 * len(piv.columns) + 4, 0.6 * len(piv.index) + 2))
        sns.heatmap(piv, annot=True, fmt=".3g", cmap="mako_r", ax=ax, cbar_kws={"label": "MAE"})
        ax.set_title(f"Heatmap MAE por ruido: {noise}")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Benchmark")
        save_figure(fig, out_dir / f"heatmap_mae_noise_{noise}", dpi=dpi, save_svg=save_svg)

    # MAE + STD aggregated
    agg = (
        final_df.groupby("model", as_index=False)
        .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"))
        .sort_values("mae_mean")
    )
    if not agg.empty:
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        xs = np.arange(len(agg))
        for i, row in agg.reset_index(drop=True).iterrows():
            model = row["model"]
            st = style_map[model]
            ax.bar(
                xs[i],
                row["mae_mean"],
                yerr=0.0 if pd.isna(row["mae_std"]) else row["mae_std"],
                color=st["color"],
                hatch=st["hatch"],
                edgecolor="black",
                linewidth=0.8,
                label=model,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(agg["model"], rotation=30, ha="right")
        ax.set_title("MAE medio y variacion global por modelo")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("MAE medio")
        place_legend(ax, outside=True)
        save_figure(fig, out_dir / "mae_std_global_por_modelo", dpi=dpi, save_svg=save_svg)

    # Time-error tradeoff
    te = final_df.copy()
    te["total_time"] = te["total_time"].fillna(te["fit_time"].fillna(0.0) + te["predict_time"].fillna(0.0))
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    for model, block in te.groupby("model"):
        st = style_map[model]
        ax.scatter(
            block["total_time"],
            block["mae"],
            color=st["color"],
            marker=st["marker"],
            edgecolor="black",
            alpha=0.70,
            label=model,
        )
        # centroid
        cx = block["total_time"].mean()
        cy = block["mae"].mean()
        ax.scatter([cx], [cy], color=st["color"], marker="*", s=220, edgecolor="black")
    ax.set_title("Trade-off tiempo-error (MAE)")
    ax.set_xlabel("Tiempo total (fit + predict)")
    ax.set_ylabel("MAE")
    place_legend(ax, outside=True)
    save_figure(fig, out_dir / "tradeoff_tiempo_error_mae", dpi=dpi, save_svg=save_svg)

    # Global NLPD and Coverage by benchmark
    for metric, ylabel in [("nlpd", "NLPD"), ("coverage_95", "Coverage 95%")]:
        if metric not in final_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10.8, 5.5))
        g = (
            final_df.groupby(["benchmark", "model"], as_index=False)[metric]
            .mean()
            .sort_values(["benchmark", metric], ascending=[True, True])
        )
        for model, block in g.groupby("model"):
            st = style_map[model]
            ax.plot(
                block["benchmark"],
                block[metric],
                marker=st["marker"],
                linestyle=st["linestyle"],
                color=st["color"],
                label=model,
            )
        ax.set_title(f"{ylabel} por benchmark y modelo")
        ax.set_xlabel("Benchmark")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        place_legend(ax, outside=True)
        save_figure(fig, out_dir / f"{metric}_por_benchmark_modelo", dpi=dpi, save_svg=save_svg)

    # Overview dashboard
    lb = tables.get("leaderboard_global_mae", pd.DataFrame())
    wins = tables.get("wins_summary_mae", pd.DataFrame())
    robust = tables.get("robustness_by_noise_mae", pd.DataFrame())
    sampler = tables.get("sampler_effects_summary", pd.DataFrame())

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.ravel()

    if not lb.empty:
        axs[0].bar(lb["model"], lb["mae_mean"], color=[style_map[m]["color"] for m in lb["model"]], edgecolor="black")
        axs[0].set_title("Ranking global MAE")
        axs[0].set_xlabel("Modelo")
        axs[0].set_ylabel("MAE medio")
        axs[0].tick_params(axis="x", rotation=30)
    else:
        axs[0].set_axis_off()

    if not wins.empty:
        axs[1].bar(wins["model"], wins["wins"], color=[style_map.get(m, {"color": "gray"})["color"] for m in wins["model"]], edgecolor="black")
        axs[1].set_title("Wins summary (MAE)")
        axs[1].set_xlabel("Modelo")
        axs[1].set_ylabel("Numero de wins")
        axs[1].tick_params(axis="x", rotation=30)
    else:
        axs[1].set_axis_off()

    if not robust.empty:
        r = robust.groupby("model", as_index=False).agg(mae_mean=("mae_mean", "mean")).sort_values("mae_mean")
        axs[2].plot(r["model"], r["mae_mean"], marker="o")
        axs[2].set_title("Robustez (MAE medio entre ruidos)")
        axs[2].set_xlabel("Modelo")
        axs[2].set_ylabel("MAE medio")
        axs[2].tick_params(axis="x", rotation=30)
    else:
        axs[2].set_axis_off()

    if not sampler.empty and "final_gap_random_minus_sobol" in sampler.columns:
        s = sampler.groupby("model", as_index=False)["final_gap_random_minus_sobol"].mean().sort_values("final_gap_random_minus_sobol")
        axs[3].bar(s["model"], s["final_gap_random_minus_sobol"], color=[style_map.get(m, {"color": "gray"})["color"] for m in s["model"]], edgecolor="black")
        axs[3].axhline(0.0, color="black", linewidth=1.0)
        axs[3].set_title("Efecto sampler (random - sobol) en MAE final")
        axs[3].set_xlabel("Modelo")
        axs[3].set_ylabel("Delta MAE")
        axs[3].tick_params(axis="x", rotation=30)
    else:
        axs[3].set_axis_off()

    fig.suptitle("Overview agregado del benchmark report (active-only)")
    save_figure(fig, overview_dir / "overview_dashboard", dpi=dpi, save_svg=save_svg)
