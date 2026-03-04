from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.benchmarks import get_benchmark

from .styling import build_model_style_map, place_legend, sanitize_filename, save_figure


def _with_dimension_column(final_df: pd.DataFrame) -> pd.DataFrame:
    df = final_df.copy()
    dim_map: Dict[str, int] = {}
    for bench in sorted(df["benchmark"].dropna().unique().tolist()):
        try:
            dim_map[str(bench)] = int(get_benchmark(str(bench)).dim)
        except Exception:
            dim_map[str(bench)] = np.nan
    df["dim"] = df["benchmark"].astype(str).map(dim_map)
    return df


def _pivot_rank_by_benchmark(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    piv = df.pivot_table(index="benchmark", columns="model", values=metric, aggfunc="mean")
    if piv.empty:
        return piv
    return piv.rank(axis=1, method="average", ascending=True)


def _plot_rank_heatmap(
    rank_df: pd.DataFrame,
    title: str,
    out_path: Path,
    dpi: int,
    save_svg: bool,
) -> None:
    if rank_df.empty:
        return
    fig, ax = plt.subplots(figsize=(1.1 * len(rank_df.columns) + 4.0, 0.6 * len(rank_df.index) + 2.2))
    sns.heatmap(
        rank_df,
        annot=True,
        fmt=".2g",
        cmap="viridis_r",
        vmin=1.0,
        vmax=float(len(rank_df.columns)),
        ax=ax,
        cbar_kws={"label": "Rank MAE (1 = mejor)"},
    )
    ax.set_title(title)
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Benchmark")
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _plot_dimension_panels(final_df: pd.DataFrame, out_dir: Path, style_map, dpi: int, save_svg: bool) -> None:
    df = _with_dimension_column(final_df)
    dims = [d for d in sorted(df["dim"].dropna().unique().tolist()) if pd.notna(d)]
    if not dims:
        return

    ncols = 2
    nrows = int(math.ceil(len(dims) / float(ncols)))
    fig, axs = plt.subplots(nrows, ncols, figsize=(14.5, 4.3 * nrows))
    axs = np.atleast_1d(axs).reshape(nrows, ncols)

    for i, dim in enumerate(dims):
        r = i // ncols
        c = i % ncols
        ax = axs[r, c]
        block = df[df["dim"] == dim]
        g = (
            block.groupby("model", as_index=False)
            .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"))
            .sort_values("mae_mean")
        )
        if g.empty:
            ax.set_axis_off()
            continue
        xs = np.arange(len(g))
        for j, row in g.reset_index(drop=True).iterrows():
            m = row["model"]
            st = style_map[m]
            ax.bar(
                xs[j],
                row["mae_mean"],
                yerr=0.0 if pd.isna(row["mae_std"]) else row["mae_std"],
                color=st["color"],
                hatch=st["hatch"],
                edgecolor="black",
                linewidth=0.8,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(g["model"], rotation=20, ha="right")
        ax.set_title(f"d = {int(dim)}")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("MAE medio")

    total_cells = nrows * ncols
    for k in range(len(dims), total_cells):
        r = k // ncols
        c = k % ncols
        axs[r, c].set_axis_off()

    fig.suptitle("Comparativa global por dimensionalidad (2 columnas)")
    save_figure(fig, out_dir / "mae_por_dimension_panel", dpi=dpi, save_svg=save_svg)


def _plot_robustness_rank(final_df: pd.DataFrame, out_dir: Path, style_map, dpi: int, save_svg: bool) -> None:
    scenario_cols = ["benchmark", "noise", "sampler", "n_train", "cv_mode"]
    tmp = final_df.dropna(subset=["mae"]).copy()
    if tmp.empty:
        return
    tmp["rank_mae"] = tmp.groupby(scenario_cols)["mae"].rank(method="average", ascending=True)
    g = (
        tmp.groupby("model", as_index=False)
        .agg(rank_mean=("rank_mae", "mean"), rank_std=("rank_mae", "std"), n=("rank_mae", "count"))
        .sort_values("rank_mean")
    )
    if g.empty:
        return

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    for _, row in g.iterrows():
        m = row["model"]
        st = style_map[m]
        ax.scatter(
            row["rank_mean"],
            0.0 if pd.isna(row["rank_std"]) else row["rank_std"],
            s=90,
            color=st["color"],
            marker=st["marker"],
            edgecolor="black",
            alpha=0.85,
            label=m,
        )
        ax.annotate(str(m), (row["rank_mean"], 0.0 if pd.isna(row["rank_std"]) else row["rank_std"]), xytext=(5, 4), textcoords="offset points", fontsize=8)

    ax.set_title("Robustez global (justa): rank MAE medio vs variacion")
    ax.set_xlabel("Rank MAE medio (1 = mejor)")
    ax.set_ylabel("Desviacion estandar del rank")
    place_legend(ax, outside=True)
    save_figure(fig, out_dir / "robustez_rank_mean_vs_std", dpi=dpi, save_svg=save_svg)


def _plot_tradeoff_interactive(te: pd.DataFrame, out_dir: Path) -> None:
    try:
        import plotly.express as px

        fig = px.scatter_3d(
            te,
            x="total_time",
            y="mae",
            z="dim",
            color="model",
            symbol="sampler",
            hover_data=["benchmark", "noise", "n_train"],
            title="Trade-off Tiempo-Error interactivo (3D)",
            labels={"total_time": "Tiempo total", "mae": "MAE", "dim": "Dimension"},
        )
        fig.update_traces(marker={"size": 4, "opacity": 0.72})
        out_path = out_dir / "tradeoff_tiempo_error_3d_interactivo.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
    except Exception:
        # Keep report generation robust even if Plotly export fails.
        return


def _plot_metric_bounded_heatmaps(final_df: pd.DataFrame, out_dir: Path, dpi: int, save_svg: bool) -> None:
    metric_specs: Tuple[Tuple[str, str, Dict[str, float]], ...] = (
        ("coverage_95", "Heatmap Coverage95 (acotado [0,1])", {"vmin": 0.0, "vmax": 1.0}),
        ("r2", "Heatmap R2 (truncado para legibilidad)", {"vmin": -1.0, "vmax": 1.0}),
    )
    for metric, title, lims in metric_specs:
        if metric not in final_df.columns:
            continue
        piv = final_df.pivot_table(index="benchmark", columns="model", values=metric, aggfunc="mean")
        if piv.empty:
            continue
        fig, ax = plt.subplots(figsize=(1.1 * len(piv.columns) + 4.0, 0.6 * len(piv.index) + 2.2))
        sns.heatmap(
            piv,
            annot=True,
            fmt=".2g",
            cmap="coolwarm",
            ax=ax,
            cbar_kws={"label": metric},
            vmin=lims.get("vmin"),
            vmax=lims.get("vmax"),
        )
        ax.set_title(title)
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Benchmark")
        save_figure(fig, out_dir / f"heatmap_{metric}_global", dpi=dpi, save_svg=save_svg)


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

    df = _with_dimension_column(final_df)
    style_map = build_model_style_map(df["model"].unique())

    # Violin MAE
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    order = df.groupby("model")["mae"].mean().sort_values().index.tolist()
    data = [df.loc[df["model"] == m, "mae"].dropna().values for m in order]
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
    order_rmse = df.groupby("model")["rmse"].mean().sort_values().index.tolist()
    data_rmse = [df.loc[df["model"] == m, "rmse"].dropna().values for m in order_rmse]
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

    # Robust heatmaps using rank (avoid scale distortion between benchmarks).
    rank_global = _pivot_rank_by_benchmark(df, metric="mae")
    _plot_rank_heatmap(
        rank_df=rank_global,
        title="Heatmap rank MAE global (escala justa por benchmark)",
        out_path=out_dir / "heatmap_rank_mae_global",
        dpi=dpi,
        save_svg=save_svg,
    )
    for noise, block in df.groupby("noise", dropna=False):
        rank_noise = _pivot_rank_by_benchmark(block, metric="mae")
        if rank_noise.empty:
            continue
        noise_tag = sanitize_filename(str(noise))
        _plot_rank_heatmap(
            rank_df=rank_noise,
            title=f"Heatmap rank MAE por ruido: {noise}",
            out_path=out_dir / f"heatmap_rank_mae_noise_{noise_tag}",
            dpi=dpi,
            save_svg=save_svg,
        )

    # Bounded heatmaps with interpretable ranges.
    _plot_metric_bounded_heatmaps(df, out_dir=out_dir, dpi=dpi, save_svg=save_svg)

    # MAE + STD aggregated
    agg = (
        df.groupby("model", as_index=False)
        .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"))
        .sort_values("mae_mean")
    )
    if not agg.empty:
        fig, ax = plt.subplots(figsize=(9.8, 5.4))
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

    # Time-error tradeoff (static + interactive 3D).
    te = df.copy()
    te["total_time"] = te["total_time"].fillna(te["fit_time"].fillna(0.0) + te["predict_time"].fillna(0.0))
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for model, block in te.groupby("model"):
        st = style_map[model]
        ax.scatter(
            block["total_time"],
            block["mae"],
            color=st["color"],
            marker=st["marker"],
            edgecolor="black",
            alpha=0.45,
            s=20,
            label=model,
        )
        cx = block["total_time"].mean()
        cy = block["mae"].mean()
        ax.scatter([cx], [cy], color=st["color"], marker="*", s=140, edgecolor="black")
    ax.set_title("Trade-off tiempo-error (MAE)")
    ax.set_xlabel("Tiempo total (fit + predict)")
    ax.set_ylabel("MAE")
    place_legend(ax, outside=True)
    save_figure(fig, out_dir / "tradeoff_tiempo_error_mae", dpi=dpi, save_svg=save_svg)
    _plot_tradeoff_interactive(te=te, out_dir=out_dir)

    # Requested structural plots.
    _plot_dimension_panels(df, out_dir=out_dir, style_map=style_map, dpi=dpi, save_svg=save_svg)
    _plot_robustness_rank(df, out_dir=out_dir, style_map=style_map, dpi=dpi, save_svg=save_svg)

    # Overview figures.
    lb = tables.get("leaderboard_global_mae", pd.DataFrame())
    wins = tables.get("wins_summary_mae", pd.DataFrame())
    robust = tables.get("robustness_by_noise_mae", pd.DataFrame())
    sampler = tables.get("sampler_effects_summary", pd.DataFrame())

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
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

    # Extra overview: rank heatmap + tradeoff
    fig2, axs2 = plt.subplots(1, 2, figsize=(16, 6.2))
    if not rank_global.empty:
        sns.heatmap(
            rank_global,
            annot=False,
            cmap="viridis_r",
            vmin=1.0,
            vmax=float(len(rank_global.columns)),
            ax=axs2[0],
            cbar_kws={"label": "Rank MAE"},
        )
        axs2[0].set_title("Heatmap rank MAE global")
        axs2[0].set_xlabel("Modelo")
        axs2[0].set_ylabel("Benchmark")
    else:
        axs2[0].set_axis_off()

    for model, block in te.groupby("model"):
        st = style_map[model]
        axs2[1].scatter(
            block["total_time"],
            block["mae"],
            color=st["color"],
            marker=st["marker"],
            edgecolor="black",
            alpha=0.50,
            s=18,
            label=model,
        )
    axs2[1].set_title("Trade-off tiempo-error (puntos por configuracion)")
    axs2[1].set_xlabel("Tiempo total")
    axs2[1].set_ylabel("MAE")
    place_legend(axs2[1], outside=True)
    fig2.suptitle("Overview final: rendimiento y coste")
    save_figure(fig2, overview_dir / "overview_heatmap_tradeoff", dpi=dpi, save_svg=save_svg)
