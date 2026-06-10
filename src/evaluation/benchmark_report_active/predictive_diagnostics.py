from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .styling import save_figure


BASE_TRAJECTORY_KEYS = ["benchmark", "n_train", "model", "sampler", "noise", "cv_mode"]
OPTIONAL_TRAJECTORY_KEYS = [
    "seed",
    "random_state",
    "run_id",
    "replicate",
    "replicate_id",
    "trial",
    "fold",
    "split",
    "config_id",
]


def _normalise_active_df(master_df: pd.DataFrame) -> pd.DataFrame:
    df = master_df.copy()
    rename_map = {
        "mae_test": "mae",
        "rmse_test": "rmse",
        "r2_test": "r2",
        "nlpd_test": "nlpd",
        "coverage_95_test": "coverage_95",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "cv_mode" in df.columns:
        df = df[df["cv_mode"].astype(str).str.lower() == "active"].copy()
    if "model" in df.columns:
        df = df[~df["model"].astype(str).str.contains("dummy", case=False, na=False)].copy()

    for col in ["n_train", "step", "mae", "rmse", "r2", "nlpd", "coverage_95"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = {"benchmark", "n_train", "model", "sampler", "noise", "cv_mode", "step"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.dropna(subset=["benchmark", "n_train", "model", "sampler", "noise", "cv_mode", "step"]).copy()
    df["n_train"] = df["n_train"].astype(int)
    df["step"] = df["step"].astype(int)
    trajectory_keys = [
        c
        for c in BASE_TRAJECTORY_KEYS + OPTIONAL_TRAJECTORY_KEYS
        if c in df.columns and not df[c].isna().all()
    ]
    df["trajectory_id"] = df[trajectory_keys].astype(str).agg("|".join, axis=1)
    return df.sort_values(trajectory_keys + ["step"]).reset_index(drop=True)


def _classify_mae(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "sin datos"
    if mean_value <= 0:
        return "sin mejora predictiva clara"
    if pd.notna(std_value) and std_value > abs(mean_value):
        return "mejora irregular"
    if mean_value >= 0.25:
        return "mejora clara del MAE"
    if mean_value >= 0.10:
        return "mejora debil del MAE"
    return "sin mejora predictiva clara"


def _classify_dummy(mean_gain: float, pct_better: float) -> str:
    if pd.isna(mean_gain) or pd.isna(pct_better):
        return "sin datos"
    if mean_gain > 0.10 and pct_better >= 0.70:
        return "GP supera claramente a Dummy"
    if mean_gain > 0 and pct_better >= 0.50:
        return "GP supera a Dummy de forma moderada"
    if mean_gain > 0:
        return "GP supera a Dummy en media, irregular"
    return "Dummy no queda superado"


def _classify_calibration(coverage_mean: float, calibration_error: float) -> str:
    if pd.isna(coverage_mean):
        return "sin datos"
    if coverage_mean > 0.985:
        return "cobertura alta; revisar anchura"
    if coverage_mean < 0.90:
        return "subcobertura"
    if calibration_error <= 0.03:
        return "calibracion cercana a 0.95"
    if calibration_error <= 0.07:
        return "calibracion aceptable"
    return "calibracion irregular"


def _load_table(tables_dir: Path, name: str) -> pd.DataFrame:
    path = Path(tables_dir) / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _build_mae_selected(tables_dir: Path) -> pd.DataFrame:
    summary = _load_table(tables_dir, "summary_mae_relative_improvement_by_benchmark_model.csv")
    required = {
        "benchmark",
        "model",
        "mean_relative_improvement_mae",
        "std_relative_improvement_mae",
        "n_runs",
        "best_n_train_initial",
    }
    if summary.empty or not required.issubset(summary.columns):
        return pd.DataFrame()

    ranked = summary.copy()
    ranked["mean_relative_improvement_mae"] = pd.to_numeric(
        ranked["mean_relative_improvement_mae"], errors="coerce"
    )
    ranked = ranked.dropna(subset=["mean_relative_improvement_mae"]).sort_values(
        ["benchmark", "mean_relative_improvement_mae", "model"],
        ascending=[True, False, True],
    )
    selected = ranked.groupby("benchmark", as_index=False).first()
    selected = selected.rename(columns={"model": "best_model_mae"})
    selected["comentario"] = [
        _classify_mae(mean_value, std_value)
        for mean_value, std_value in zip(
            selected["mean_relative_improvement_mae"],
            selected["std_relative_improvement_mae"],
        )
    ]
    return selected[
        [
            "benchmark",
            "best_model_mae",
            "mean_relative_improvement_mae",
            "std_relative_improvement_mae",
            "best_n_train_initial",
            "n_runs",
            "comentario",
        ]
    ].copy()


def _build_dummy_selected(tables_dir: Path) -> pd.DataFrame:
    summary = _load_table(tables_dir, "summary_mae_gp_vs_dummy_by_benchmark_model.csv")
    required = {
        "benchmark",
        "model",
        "mean_relative_gain_gp_final_vs_dummy_mae",
        "pct_gp_final_better_than_dummy_mae",
        "n_runs",
    }
    if summary.empty or not required.issubset(summary.columns):
        return pd.DataFrame()

    ranked = summary.copy()
    ranked["mean_relative_gain_gp_final_vs_dummy_mae"] = pd.to_numeric(
        ranked["mean_relative_gain_gp_final_vs_dummy_mae"], errors="coerce"
    )
    ranked = ranked.dropna(subset=["mean_relative_gain_gp_final_vs_dummy_mae"]).sort_values(
        ["benchmark", "mean_relative_gain_gp_final_vs_dummy_mae", "model"],
        ascending=[True, False, True],
    )
    selected = ranked.groupby("benchmark", as_index=False).first()
    selected = selected.rename(columns={"model": "best_model_vs_dummy"})
    selected["comentario"] = [
        _classify_dummy(mean_gain, pct_better)
        for mean_gain, pct_better in zip(
            selected["mean_relative_gain_gp_final_vs_dummy_mae"],
            selected["pct_gp_final_better_than_dummy_mae"],
        )
    ]
    return selected[
        [
            "benchmark",
            "best_model_vs_dummy",
            "mean_relative_gain_gp_final_vs_dummy_mae",
            "pct_gp_final_better_than_dummy_mae",
            "n_runs",
            "comentario",
        ]
    ].copy()


def _final_rows_by_trajectory(active_df: pd.DataFrame) -> pd.DataFrame:
    if active_df.empty:
        return pd.DataFrame()
    return (
        active_df.sort_values("step")
        .groupby("trajectory_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _iqr(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return np.nan
    return float(finite.quantile(0.75) - finite.quantile(0.25))


def _q1(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(finite.quantile(0.25)) if not finite.empty else np.nan


def _q3(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(finite.quantile(0.75)) if not finite.empty else np.nan


def _median_finite(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(finite.median()) if not finite.empty else np.nan


def _nonfinite_count(values: pd.Series) -> int:
    numeric = pd.to_numeric(values, errors="coerce")
    return int((~np.isfinite(numeric)).sum())


def _build_probabilistic_summary(active_df: pd.DataFrame) -> pd.DataFrame:
    final_df = _final_rows_by_trajectory(active_df)
    if final_df.empty or not {"coverage_95", "nlpd"}.issubset(final_df.columns):
        return pd.DataFrame()

    grouped = (
        final_df.groupby(["benchmark", "model"], as_index=False)
        .agg(
            coverage_95_final_mean=("coverage_95", "mean"),
            coverage_95_final_std=("coverage_95", "std"),
            nlpd_final_median=("nlpd", _median_finite),
            nlpd_final_q1=("nlpd", _q1),
            nlpd_final_q3=("nlpd", _q3),
            nlpd_final_iqr=("nlpd", _iqr),
            n_nonfinite_nlpd=("nlpd", _nonfinite_count),
            n_runs=("trajectory_id", "nunique"),
        )
        .sort_values(["benchmark", "model"])
    )
    grouped["calibration_error_95_mean"] = (grouped["coverage_95_final_mean"] - 0.95).abs()
    grouped["comentario_calibracion"] = [
        _classify_calibration(coverage, error)
        for coverage, error in zip(
            grouped["coverage_95_final_mean"],
            grouped["calibration_error_95_mean"],
        )
    ]
    return grouped[
        [
            "benchmark",
            "model",
            "coverage_95_final_mean",
            "coverage_95_final_std",
            "calibration_error_95_mean",
            "nlpd_final_median",
            "nlpd_final_iqr",
            "n_nonfinite_nlpd",
            "n_runs",
            "comentario_calibracion",
            "nlpd_final_q1",
            "nlpd_final_q3",
        ]
    ].copy()


def _plot_coverage(summary: pd.DataFrame, out_path: Path, dpi: int, save_svg: bool) -> None:
    if summary.empty:
        return
    benchmarks = sorted(summary["benchmark"].dropna().astype(str).unique().tolist())
    if not benchmarks:
        return

    ncols = 2
    nrows = int(np.ceil(len(benchmarks) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.4, 4.25 * nrows))
    axs = np.atleast_1d(axs).ravel()

    for i, benchmark in enumerate(benchmarks):
        ax = axs[i]
        block = (
            summary[summary["benchmark"].astype(str) == benchmark]
            .sort_values(["calibration_error_95_mean", "model"], ascending=[True, True])
            .reset_index(drop=True)
        )
        x = np.arange(len(block))
        y = block["coverage_95_final_mean"].to_numpy(dtype=float)
        yerr = block["coverage_95_final_std"].fillna(0.0).to_numpy(dtype=float)
        ax.axhspan(0.90, 1.00, color="#d8f3dc", alpha=0.35, zorder=0)
        ax.bar(x, y, yerr=yerr, capsize=3, color="#4c78a8", edgecolor="black", linewidth=0.6)
        ax.axhline(0.95, color="black", linewidth=1.1, linestyle="--")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(block["model"].astype(str).tolist(), rotation=35, ha="right")
        ax.set_title(str(benchmark))
        ax.set_ylabel("Coverage 95 final")

    for k in range(len(benchmarks), len(axs)):
        axs[k].set_axis_off()

    fig.suptitle("Calibracion predictiva final por benchmark")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.90, hspace=0.58, wspace=0.28)
    fig._skip_tight_layout = True
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _plot_nlpd(summary: pd.DataFrame, out_path: Path, dpi: int, save_svg: bool) -> None:
    if summary.empty or summary["nlpd_final_median"].dropna().empty:
        return
    benchmarks = sorted(summary["benchmark"].dropna().astype(str).unique().tolist())
    if not benchmarks:
        return

    ncols = 2
    nrows = int(np.ceil(len(benchmarks) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.4, 4.25 * nrows))
    axs = np.atleast_1d(axs).ravel()

    for i, benchmark in enumerate(benchmarks):
        ax = axs[i]
        block = (
            summary[summary["benchmark"].astype(str) == benchmark]
            .dropna(subset=["nlpd_final_median"])
            .sort_values(["nlpd_final_median", "model"], ascending=[True, True])
            .reset_index(drop=True)
        )
        if block.empty:
            ax.set_axis_off()
            continue
        x = np.arange(len(block))
        median = block["nlpd_final_median"].to_numpy(dtype=float)
        q1 = block["nlpd_final_q1"].fillna(block["nlpd_final_median"]).to_numpy(dtype=float)
        q3 = block["nlpd_final_q3"].fillna(block["nlpd_final_median"]).to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(0.0, median - q1), np.maximum(0.0, q3 - median)])
        ax.bar(x, median, yerr=yerr, capsize=3, color="#f58518", edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(block["model"].astype(str).tolist(), rotation=35, ha="right")
        ax.set_title(str(benchmark))
        ax.set_ylabel("NLPD final mediana")

    for k in range(len(benchmarks), len(axs)):
        axs[k].set_axis_off()

    fig.suptitle("NLPD final por benchmark (menor es mejor)")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.90, hspace=0.58, wspace=0.28)
    fig._skip_tight_layout = True
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _first_rows_by_trajectory(block: pd.DataFrame) -> pd.DataFrame:
    if block.empty:
        return pd.DataFrame()
    return (
        block.sort_values("step")
        .groupby("trajectory_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _build_step_diagnostics_best_model(active_df: pd.DataFrame, mae_selected: pd.DataFrame) -> pd.DataFrame:
    required = {"benchmark", "best_model_mae"}
    metric_required = {"trajectory_id", "step", "mae", "nlpd", "coverage_95"}
    if active_df.empty or mae_selected.empty or not required.issubset(mae_selected.columns):
        return pd.DataFrame()
    if not metric_required.issubset(active_df.columns):
        return pd.DataFrame()

    rows: List[pd.DataFrame] = []
    for selected in mae_selected.sort_values("benchmark").itertuples(index=False):
        benchmark = str(selected.benchmark)
        model = str(selected.best_model_mae)
        block = active_df[
            (active_df["benchmark"].astype(str) == benchmark)
            & (active_df["model"].astype(str) == model)
        ].copy()
        if block.empty:
            continue

        step0 = block[block["step"].astype(int) == 0].copy()
        reference = step0 if not step0.empty else _first_rows_by_trajectory(block)
        if reference.empty:
            continue

        reference = reference[["trajectory_id", "step", "mae", "nlpd"]].rename(
            columns={"step": "reference_step", "mae": "mae_initial", "nlpd": "nlpd_initial"}
        )
        merged = block.merge(reference, on="trajectory_id", how="inner")
        if merged.empty:
            continue

        merged["mae"] = pd.to_numeric(merged["mae"], errors="coerce")
        merged["nlpd"] = pd.to_numeric(merged["nlpd"], errors="coerce")
        merged["coverage_95"] = pd.to_numeric(merged["coverage_95"], errors="coerce")
        merged["mae_initial"] = pd.to_numeric(merged["mae_initial"], errors="coerce")
        merged["nlpd_initial"] = pd.to_numeric(merged["nlpd_initial"], errors="coerce")
        merged["relative_mae"] = np.where(
            merged["mae_initial"] > 0,
            merged["mae"] / merged["mae_initial"],
            np.nan,
        )
        finite_nlpd_initial = np.isfinite(merged["nlpd_initial"]) & (merged["nlpd_initial"] > 0)
        finite_nlpd = np.isfinite(merged["nlpd"])
        merged["relative_nlpd"] = np.where(
            finite_nlpd_initial & finite_nlpd,
            merged["nlpd"] / merged["nlpd_initial"],
            np.nan,
        )
        merged["coverage_error_95"] = (merged["coverage_95"] - 0.95).abs()

        counts = merged.groupby("step")["trajectory_id"].nunique()
        full_n_runs = int(counts.max()) if not counts.empty else 0
        common_steps = set(counts[counts == full_n_runs].index.tolist())
        comparable = merged[merged["step"].isin(common_steps)].copy()
        if comparable.empty:
            comparable = merged.copy()

        summary = (
            comparable.groupby(["benchmark", "model", "step"], as_index=False)
            .agg(
                relative_mae_mean=("relative_mae", "mean"),
                relative_mae_std=("relative_mae", "std"),
                relative_nlpd_median=("relative_nlpd", _median_finite),
                relative_nlpd_q1=("relative_nlpd", _q1),
                relative_nlpd_q3=("relative_nlpd", _q3),
                relative_nlpd_iqr=("relative_nlpd", _iqr),
                coverage_95_mean=("coverage_95", "mean"),
                coverage_95_std=("coverage_95", "std"),
                coverage_error_95_mean=("coverage_error_95", "mean"),
                coverage_error_95_std=("coverage_error_95", "std"),
                n_runs=("trajectory_id", "nunique"),
                n_runs_nlpd=("relative_nlpd", lambda s: int(pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).notna().sum())),
                reference_step=("reference_step", "min"),
            )
            .sort_values(["benchmark", "model", "step"])
        )
        summary["best_model_source"] = "summary_mae_selected_for_tfg"
        summary["common_n_runs"] = full_n_runs
        all_steps = sorted(pd.to_numeric(merged["step"], errors="coerce").dropna().astype(int).unique().tolist())
        used_steps = sorted(pd.to_numeric(summary["step"], errors="coerce").dropna().astype(int).unique().tolist())
        summary["max_step_available"] = max(all_steps) if all_steps else np.nan
        summary["max_step_common"] = max(used_steps) if used_steps else np.nan
        summary["uses_common_steps_only"] = True
        rows.append(summary)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _plot_predictive_quality_by_step(
    diagnostics: pd.DataFrame,
    out_path: Path,
    dpi: int,
    save_svg: bool,
) -> None:
    if diagnostics.empty:
        return

    benchmarks = sorted(diagnostics["benchmark"].dropna().astype(str).unique().tolist())
    if not benchmarks:
        return

    ncols = 2
    nrows = int(np.ceil(len(benchmarks) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13.2, 4.3 * nrows))
    axs = np.atleast_1d(axs).ravel()

    colors = {
        "mae": "#2f6f9f",
        "nlpd": "#c45536",
        "coverage": "#238b45",
        "reference": "#333333",
    }

    for i, benchmark in enumerate(benchmarks):
        ax = axs[i]
        ax_cov = ax.twinx()
        block = diagnostics[diagnostics["benchmark"].astype(str) == benchmark].sort_values("step")
        if block.empty:
            ax.set_axis_off()
            ax_cov.set_axis_off()
            continue

        x = block["step"].to_numpy(dtype=float)
        mae_mean = block["relative_mae_mean"].to_numpy(dtype=float)
        mae_std = block["relative_mae_std"].fillna(0.0).to_numpy(dtype=float)
        nlpd_median = block["relative_nlpd_median"].to_numpy(dtype=float)
        nlpd_q1 = block["relative_nlpd_q1"].fillna(block["relative_nlpd_median"]).to_numpy(dtype=float)
        nlpd_q3 = block["relative_nlpd_q3"].fillna(block["relative_nlpd_median"]).to_numpy(dtype=float)
        coverage_mean = block["coverage_95_mean"].to_numpy(dtype=float)
        coverage_std = block["coverage_95_std"].fillna(0.0).to_numpy(dtype=float)

        ax.fill_between(
            x,
            np.maximum(0.0, mae_mean - mae_std),
            mae_mean + mae_std,
            color=colors["mae"],
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(x, mae_mean, color=colors["mae"], marker="o", markersize=3.2, label="MAE relativo")
        ax.fill_between(
            x,
            np.maximum(0.0, nlpd_q1),
            np.maximum(0.0, nlpd_q3),
            color=colors["nlpd"],
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            x,
            nlpd_median,
            color=colors["nlpd"],
            linestyle="--",
            marker="s",
            markersize=3.0,
            label="NLPD relativo",
        )
        ax.axhline(1.0, color=colors["reference"], linestyle=":", linewidth=1.0)

        ax_cov.fill_between(
            x,
            np.clip(coverage_mean - coverage_std, 0.0, 1.05),
            np.clip(coverage_mean + coverage_std, 0.0, 1.05),
            color=colors["coverage"],
            alpha=0.10,
            linewidth=0,
        )
        ax_cov.plot(
            x,
            coverage_mean,
            color=colors["coverage"],
            linestyle="-.",
            marker="^",
            markersize=3.0,
            label="Coverage 95",
        )
        ax_cov.axhline(0.95, color=colors["coverage"], linestyle=":", linewidth=1.0)

        finite_left = np.concatenate(
            [
                mae_mean[np.isfinite(mae_mean)],
                (mae_mean + mae_std)[np.isfinite(mae_mean + mae_std)],
                nlpd_median[np.isfinite(nlpd_median)],
                nlpd_q3[np.isfinite(nlpd_q3)],
            ]
        )
        y_max = 1.25
        if finite_left.size:
            y_max = max(1.25, float(np.nanmax(finite_left)) * 1.08)
        ax.set_ylim(0.0, min(max(y_max, 1.25), 3.5))
        ax_cov.set_ylim(0.0, 1.05)
        ax.set_xlabel("Iteracion de infill")
        if i % ncols == 0:
            ax.set_ylabel("MAE/NLPD relativo")
        else:
            ax.set_ylabel("")
        if i % ncols == ncols - 1:
            ax_cov.set_ylabel("Coverage 95")
        else:
            ax_cov.set_ylabel("")

        model = str(block["model"].iloc[0])
        n_runs = int(block["n_runs"].min()) if "n_runs" in block.columns and block["n_runs"].notna().any() else 0
        max_common = int(block["max_step_common"].max()) if block["max_step_common"].notna().any() else int(block["step"].max())
        max_available = int(block["max_step_available"].max()) if block["max_step_available"].notna().any() else max_common
        step_note = f"pasos comunes 0-{max_common}"
        if max_available > max_common:
            step_note += f" de {max_available}"
        ax.set_title(f"{benchmark} - {model}\n{step_note}, n={n_runs}")
        ax.grid(True, alpha=0.25)
        ax_cov.grid(False)

    for k in range(len(benchmarks), len(axs)):
        axs[k].set_axis_off()

    handles = [
        Line2D([0], [0], color=colors["mae"], marker="o", linewidth=1.8, label="MAE relativo medio"),
        Line2D([0], [0], color=colors["nlpd"], marker="s", linestyle="--", linewidth=1.8, label="NLPD relativo mediano"),
        Line2D([0], [0], color=colors["coverage"], marker="^", linestyle="-.", linewidth=1.8, label="Coverage 95 medio"),
        Line2D([0], [0], color=colors["reference"], linestyle=":", linewidth=1.4, label="Referencias: 1 y 0.95"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle("Calidad predictiva y calibracion durante el infill")
    fig.text(
        0.5,
        0.925,
        "Mejor modelo MAE por benchmark; NLPD agregado con mediana/IQR y cobertura con media/desviacion tipica",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.15, top=0.86, hspace=0.46, wspace=0.30)
    fig._skip_tight_layout = True
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _best_by_benchmark(table: pd.DataFrame, value_col: str, ascending: bool) -> pd.DataFrame:
    if table.empty or value_col not in table.columns:
        return pd.DataFrame()
    ranked = table.dropna(subset=[value_col]).sort_values(
        ["benchmark", value_col, "model" if "model" in table.columns else table.columns[1]],
        ascending=[True, ascending, True],
    )
    return ranked.groupby("benchmark", as_index=False).first()


def _write_interpretation(
    path: Path,
    mae_selected: pd.DataFrame,
    dummy_selected: pd.DataFrame,
    prob_summary: pd.DataFrame,
    incumbent_summary: pd.DataFrame,
    nlpd_nonfinite_total: int,
    nlpd_outlier_note: str,
) -> None:
    lines: List[str] = [
        "# Interpretacion predictiva para el TFG",
        "",
        "Esta lectura resume la capacidad predictiva global del GP durante el infill. No evalua el exito de optimizacion; para eso debe usarse el analisis de incumbent.",
        "",
    ]

    if not mae_selected.empty:
        lines.append("## MAE relativo")
        for row in mae_selected.sort_values("benchmark").itertuples(index=False):
            extra = ""
            if not incumbent_summary.empty and {"benchmark", "model"}.issubset(incumbent_summary.columns):
                inc_block = incumbent_summary[incumbent_summary["benchmark"].astype(str) == str(row.benchmark)]
                if not inc_block.empty:
                    inc_best = inc_block.sort_values(
                        "mean_relative_incumbent_improvement", ascending=False
                    ).iloc[0]
                    if str(inc_best["model"]) != str(row.best_model_mae):
                        extra = (
                            f" El mejor modelo predictivo no coincide con el mejor en incumbent "
                            f"({inc_best['model']})."
                        )
            lines.append(
                f"- {row.benchmark}: {row.best_model_mae} obtiene la mayor mejora media de MAE "
                f"({row.mean_relative_improvement_mae:.3f}); {row.comentario}.{extra}"
            )
        lines.append("")

    if not dummy_selected.empty:
        lines.append("## Comparacion con Dummy")
        for row in dummy_selected.sort_values("benchmark").itertuples(index=False):
            lines.append(
                f"- {row.benchmark}: {row.best_model_vs_dummy} supera a Dummy con una ganancia relativa "
                f"media de {row.mean_relative_gain_gp_final_vs_dummy_mae:.3f} y gana en "
                f"{row.pct_gp_final_better_than_dummy_mae:.1%} de las configuraciones; {row.comentario}."
            )
        lines.append("")

    if not prob_summary.empty:
        lines.append("## Diagnostico probabilistico")
        best_cal = _best_by_benchmark(prob_summary, "calibration_error_95_mean", ascending=True)
        for row in best_cal.sort_values("benchmark").itertuples(index=False):
            lines.append(
                f"- {row.benchmark}: la cobertura mas cercana a 0.95 corresponde a {row.model} "
                f"(coverage={row.coverage_95_final_mean:.3f}); {row.comentario_calibracion}."
            )
        lines.append(
            "- Coverage 95 diagnostica calibracion: una cobertura alta no implica mejor optimizacion ni necesariamente mejor MAE."
        )
        if nlpd_nonfinite_total:
            lines.append(f"- NLPD contiene {nlpd_nonfinite_total} valores no finitos, excluidos de medianas e IQR.")
        if nlpd_outlier_note:
            lines.append(f"- {nlpd_outlier_note}")
        lines.append("")

    lines.extend(
        [
            "## Figuras recomendadas",
            "- `figures/evolution/evolution_relative_mae_best_model_by_benchmark.png` como figura principal de capacidad predictiva.",
            "- `figures/evolution/predictive_quality_vs_uncertainty_by_step_best_model.png` como diagnostico compacto de MAE, NLPD y coverage durante el infill.",
            "- `figures/evolution/final_relative_gain_mae_gp_vs_dummy_by_benchmark.png` o la tabla compacta Dummy como referencia predictiva.",
            "- `figures/evolution/final_coverage95_calibration_by_benchmark.png` como diagnostico probabilistico compacto.",
            "",
            "## Figuras no recomendadas como principales",
            "- Curvas largas separadas de NLPD/coverage por step: son densas y menos interpretables para la memoria.",
            "- `final_median_nlpd_by_benchmark.png` solo debe usarse como diagnostico secundario si se desea discutir NLPD.",
            "- Figuras agregadas por `n_train_current` no deben usarse como evidencia principal de mejora por infill.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_verification(
    active_df: pd.DataFrame,
    counts_mae: pd.DataFrame,
    step_diagnostics: pd.DataFrame,
    mae_selected: pd.DataFrame,
    dummy_selected: pd.DataFrame,
    prob_summary: pd.DataFrame,
    generated_tables: List[Path],
    generated_figures: List[Path],
    nlpd_nonfinite_total: int,
    nlpd_outlier_note: str,
) -> str:
    lines: List[str] = [
        "# Verificacion de metricas predictivas",
        "",
        "## Checks pasados",
    ]
    if active_df.empty:
        lines.append("- WARNING: no se pudo construir `active_df`.")
    else:
        min_steps = active_df.groupby("trajectory_id")["step"].min()
        lines.append(f"- Todas las trayectorias empiezan en step=0: {bool(min_steps.eq(0).all())}.")
        lines.append(f"- Benchmarks cubiertos: {', '.join(sorted(active_df['benchmark'].astype(str).unique()))}.")
        lines.append(f"- Modelos GP cubiertos: {', '.join(sorted(active_df['model'].astype(str).unique()))}.")
        dummy_mixed = active_df["model"].astype(str).str.contains("dummy", case=False, na=False).any()
        lines.append(f"- Dummy mezclado en curvas GP: {bool(dummy_mixed)}.")

    if not counts_mae.empty:
        variable = counts_mae.groupby(["benchmark", "n_train", "model"])["n_runs"].nunique()
        variable_count = int((variable > 1).sum())
        n_min = int(counts_mae["n_runs"].min())
        n_max = int(counts_mae["n_runs"].max())
        lines.append(f"- Curvas MAE con n_runs variable: {variable_count}.")
        lines.append(f"- n_runs MAE min/max por punto: {n_min}/{n_max}.")
        compact_counts = (
            counts_mae.groupby(["benchmark", "model", "step"], as_index=False)["n_runs"].sum()
            .groupby(["benchmark", "model"])["n_runs"]
            .agg(["min", "max"])
            .reset_index()
        )
        lines.append("")
        lines.append("## Trayectorias por benchmark/modelo/step para MAE")
        lines.append("")
        lines.append(compact_counts.to_markdown(index=False))

    if not step_diagnostics.empty:
        common_summary = (
            step_diagnostics.groupby(["benchmark", "model"], as_index=False)
            .agg(
                n_runs=("n_runs", "min"),
                max_step_common=("max_step_common", "max"),
                max_step_available=("max_step_available", "max"),
            )
            .sort_values(["benchmark", "model"])
        )
        lines.append("")
        lines.append("## Pasos usados en la figura MAE/NLPD/Coverage")
        lines.append("")
        lines.append(common_summary.to_markdown(index=False))

    lines.extend(
        [
            "",
            "## Warnings y decisiones",
            f"- Valores no finitos de NLPD final: {nlpd_nonfinite_total}.",
            f"- {nlpd_outlier_note or 'NLPD agregado con mediana e IQR para robustez.'}",
            "- Dummy se usa solo como baseline predictiva entrenada con el diseno inicial.",
            "- La mejora de MAE se resume por trayectoria antes de promediar, usando `step=0` como referencia.",
            "- La figura conjunta MAE/NLPD/Coverage usa pasos comunes del mejor modelo de cada benchmark para evitar cambios de composicion.",
            "",
            "## Figuras recomendadas para la memoria",
            "- `evolution_relative_mae_best_model_by_benchmark.png`.",
            "- `predictive_quality_vs_uncertainty_by_step_best_model.png` como diagnostico conjunto MAE/NLPD/coverage.",
            "- `final_relative_gain_mae_gp_vs_dummy_by_benchmark.png` o `summary_dummy_selected_for_tfg.csv`.",
            "- `final_coverage95_calibration_by_benchmark.png`.",
            "",
            "## Figuras que no se recomiendan como principales",
            "- Curvas largas separadas de NLPD/coverage por step.",
            "- Figuras de MAE agregadas por `n_train_current`.",
            "- `final_median_nlpd_by_benchmark.png` salvo como diagnostico secundario.",
            "",
            "## Salidas generadas",
        ]
    )
    for path in generated_tables:
        lines.append(f"- Tabla: `{path.as_posix()}`")
    for path in generated_figures:
        lines.append(f"- Figura: `{path.as_posix()}`")
    lines.append("")
    return "\n".join(lines)


def _nlpd_outlier_note(active_df: pd.DataFrame) -> Tuple[int, str]:
    final_df = _final_rows_by_trajectory(active_df)
    if final_df.empty or "nlpd" not in final_df.columns:
        return 0, "NLPD no disponible."
    nlpd = pd.to_numeric(final_df["nlpd"], errors="coerce")
    nonfinite = int((~np.isfinite(nlpd)).sum())
    finite = nlpd[np.isfinite(nlpd)]
    if finite.empty:
        return nonfinite, "NLPD no tiene valores finitos para resumir."
    median = float(finite.median())
    max_value = float(finite.max())
    if median > 0 and max_value / median > 10:
        return (
            nonfinite,
            f"NLPD presenta outliers fuertes (max={max_value:.3f}, mediana={median:.3f}); se recomienda usar tabla/mediana, no media.",
        )
    return nonfinite, "NLPD se resume con mediana e IQR para evitar sensibilidad a outliers."


def generate_predictive_diagnostics_outputs(
    master_df: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    dpi: int = 300,
    save_svg: bool = False,
) -> Dict[str, pd.DataFrame]:
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    active_df = _normalise_active_df(master_df)
    counts_mae = _load_table(tables_dir, "counts_by_step_mae.csv")
    incumbent_summary = _load_table(tables_dir, "summary_incumbent_relative_improvement_by_benchmark_model.csv")

    mae_selected = _build_mae_selected(tables_dir)
    dummy_selected = _build_dummy_selected(tables_dir)
    prob_summary = _build_probabilistic_summary(active_df)
    step_diagnostics = _build_step_diagnostics_best_model(active_df, mae_selected)

    mae_selected_path = tables_dir / "summary_mae_selected_for_tfg.csv"
    dummy_selected_path = tables_dir / "summary_dummy_selected_for_tfg.csv"
    prob_summary_path = tables_dir / "summary_probabilistic_selected_for_tfg.csv"
    step_diagnostics_path = tables_dir / "predictive_quality_by_step_best_model.csv"
    interpretation_path = tables_dir / "interpretacion_predictiva_tfg.md"
    verification_path = tables_dir / "verification_predictive_metrics.md"

    coverage_figure_path = figures_dir / "final_coverage95_calibration_by_benchmark.png"
    nlpd_figure_path = figures_dir / "final_median_nlpd_by_benchmark.png"
    step_diagnostics_figure_path = figures_dir / "predictive_quality_vs_uncertainty_by_step_best_model.png"

    generated_tables = [
        mae_selected_path,
        dummy_selected_path,
        prob_summary_path,
        step_diagnostics_path,
        interpretation_path,
        verification_path,
    ]
    generated_figures = [
        coverage_figure_path,
        nlpd_figure_path,
        step_diagnostics_figure_path,
    ]

    mae_selected.to_csv(mae_selected_path, index=False)
    dummy_selected.to_csv(dummy_selected_path, index=False)
    prob_summary.to_csv(prob_summary_path, index=False)
    step_diagnostics.to_csv(step_diagnostics_path, index=False)

    _plot_coverage(
        summary=prob_summary,
        out_path=figures_dir / "final_coverage95_calibration_by_benchmark",
        dpi=dpi,
        save_svg=save_svg,
    )
    _plot_nlpd(
        summary=prob_summary,
        out_path=figures_dir / "final_median_nlpd_by_benchmark",
        dpi=dpi,
        save_svg=save_svg,
    )
    _plot_predictive_quality_by_step(
        diagnostics=step_diagnostics,
        out_path=figures_dir / "predictive_quality_vs_uncertainty_by_step_best_model",
        dpi=dpi,
        save_svg=save_svg,
    )

    nlpd_nonfinite_total, nlpd_note = _nlpd_outlier_note(active_df)
    _write_interpretation(
        path=interpretation_path,
        mae_selected=mae_selected,
        dummy_selected=dummy_selected,
        prob_summary=prob_summary,
        incumbent_summary=incumbent_summary,
        nlpd_nonfinite_total=nlpd_nonfinite_total,
        nlpd_outlier_note=nlpd_note,
    )
    verification = _build_verification(
        active_df=active_df,
        counts_mae=counts_mae,
        step_diagnostics=step_diagnostics,
        mae_selected=mae_selected,
        dummy_selected=dummy_selected,
        prob_summary=prob_summary,
        generated_tables=generated_tables,
        generated_figures=generated_figures,
        nlpd_nonfinite_total=nlpd_nonfinite_total,
        nlpd_outlier_note=nlpd_note,
    )
    verification_path.write_text(verification, encoding="utf-8")

    print("Predictive diagnostics verification")
    if not counts_mae.empty:
        variable = counts_mae.groupby(["benchmark", "n_train", "model"])["n_runs"].nunique()
        print(f"- MAE curves with variable n_runs: {int((variable > 1).sum())}")
        print(f"- MAE n_runs min/max: {int(counts_mae['n_runs'].min())}/{int(counts_mae['n_runs'].max())}")
    if not active_df.empty:
        print(f"- All trajectories use step=0: {bool(active_df.groupby('trajectory_id')['step'].min().eq(0).all())}")
        print(f"- Benchmarks: {', '.join(sorted(active_df['benchmark'].astype(str).unique()))}")
        print(f"- Models: {', '.join(sorted(active_df['model'].astype(str).unique()))}")
    print(f"- Non-finite final NLPD values: {nlpd_nonfinite_total}")
    for path in generated_tables:
        print(f"- Table: {path}")
    for path in generated_figures:
        print(f"- Figure: {path}")

    return {
        "summary_mae_selected_for_tfg": mae_selected,
        "summary_dummy_selected_for_tfg": dummy_selected,
        "summary_probabilistic_selected_for_tfg": prob_summary,
        "predictive_quality_by_step_best_model": step_diagnostics,
    }
