from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarks import get_benchmark
from src.benchmarks.dataset_generator import generate_benchmark_dataset

from .styling import build_model_style_map, place_legend, sanitize_filename, save_figure


EVOLUTION_METRICS = [
    ("mae", "MAE"),
    ("coverage_95", "Coverage 95%"),
    ("nlpd", "NLPD"),
]

PREDICTIVE_METRICS = {
    "mae": {
        "label": "MAE",
        "ylabel": "MAE en test",
        "relative_ylabel": "MAE relativo respecto al inicio",
        "direction": "min",
    },
    "rmse": {
        "label": "RMSE",
        "ylabel": "RMSE en test",
        "relative_ylabel": "RMSE relativo respecto al inicio",
        "direction": "min",
    },
    "r2": {
        "label": "R2",
        "ylabel": "R2 en test",
        "relative_ylabel": "Mejora de R2 respecto al inicio",
        "direction": "max",
    },
}

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


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


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
        is_dummy = df["model"].astype(str).str.contains("dummy", case=False, na=False)
        df = df[~is_dummy].copy()

    numeric_cols = ["n_train", "step", "n_train_current", "mae", "rmse", "r2"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = {"benchmark", "n_train", "model", "step"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.dropna(subset=["benchmark", "n_train", "model", "step"]).copy()
    df["step"] = df["step"].astype(int)
    df["n_train"] = df["n_train"].astype(int)

    trajectory_keys = [
        c
        for c in BASE_TRAJECTORY_KEYS + OPTIONAL_TRAJECTORY_KEYS
        if c in df.columns and not df[c].isna().all()
    ]
    if not trajectory_keys:
        trajectory_keys = ["benchmark", "n_train", "model"]

    df["_trajectory_keys"] = "|".join(trajectory_keys)
    df["trajectory_id"] = df[trajectory_keys].astype(str).agg("|".join, axis=1)
    return df


def _infer_y_test_ranges(
    df: pd.DataFrame,
    metadata: Optional[Dict[str, object]],
) -> Tuple[pd.DataFrame, List[str]]:
    """Attach y-test ranges when they are already present or reproducible."""
    if df.empty:
        return df, []

    notes: List[str] = []
    range_candidates = ["range_y_test", "y_test_range", "rango_y_test"]
    existing = next((c for c in range_candidates if c in df.columns), None)
    if existing:
        df["range_y_test"] = pd.to_numeric(df[existing], errors="coerce")
        return df, notes

    if metadata is None:
        notes.append("No hay metadata para recuperar range_y_test; se usa MAE relativo por trayectoria.")
        return df, notes

    seed = int(metadata.get("seed", 42) or 42)
    n_test = int(metadata.get("n_test", 200) or 200)
    range_map: Dict[Tuple[str, str], float] = {}

    if "sampler" not in df.columns:
        notes.append("No existe columna sampler; no se recupera range_y_test.")
        return df, notes

    for benchmark, sampler in df[["benchmark", "sampler"]].drop_duplicates().itertuples(index=False):
        try:
            dataset = generate_benchmark_dataset(
                benchmark=str(benchmark),
                n_train=1,
                n_test=n_test,
                sampler=str(sampler).lower(),
                noise="none",
                seed=seed,
            )
            y = np.asarray(dataset.y_test_clean, dtype=float).ravel()
            y_range = float(np.nanmax(y) - np.nanmin(y))
            if np.isfinite(y_range) and y_range > 0:
                range_map[(str(benchmark), str(sampler))] = y_range
        except Exception as exc:
            notes.append(f"No se pudo recuperar range_y_test para {benchmark}/{sampler}: {exc}")

    if range_map:
        df["range_y_test"] = [
            range_map.get((str(row.benchmark), str(row.sampler)), np.nan)
            for row in df.itertuples(index=False)
        ]
        recovered = int(pd.Series(range_map).notna().sum())
        notes.append(
            f"range_y_test recuperado de forma reproducible para {recovered} pares benchmark/sampler "
            f"(seed={seed}, n_test={n_test})."
        )
    else:
        notes.append("No se pudo recuperar range_y_test; se usa MAE relativo por trayectoria.")

    return df, notes


def _build_counts_by_step(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame(columns=["benchmark", "n_train", "model", "step", "n_runs"])

    valid = df.dropna(subset=[metric]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["benchmark", "n_train", "model", "step", "n_runs"])

    counts = (
        valid.groupby(["benchmark", "n_train", "model", "step"], as_index=False)
        .agg(n_runs=("trajectory_id", "nunique"))
        .sort_values(["benchmark", "n_train", "model", "step"])
    )
    return counts


def _count_warnings(counts: pd.DataFrame, metric: str) -> List[str]:
    warnings: List[str] = []
    if counts.empty:
        return warnings
    for (benchmark, n_train, model), block in counts.groupby(["benchmark", "n_train", "model"]):
        observed = sorted(block["n_runs"].dropna().astype(int).unique().tolist())
        if len(observed) > 1:
            warnings.append(
                f"WARNING: {metric.upper()} tiene n_runs variable en "
                f"{benchmark}, n_train={n_train}, model={model}: {observed}"
            )
    return warnings


def _aggregate_step_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    valid = df.dropna(subset=[metric]).copy()
    if valid.empty:
        return pd.DataFrame()

    return (
        valid.groupby(["benchmark", "n_train", "step", "model"], as_index=False)
        .agg(
            metric_mean=(metric, "mean"),
            metric_std=(metric, "std"),
            n_runs=("trajectory_id", "nunique"),
        )
        .sort_values(["benchmark", "n_train", "model", "step"])
    )


def _plot_metric_by_step_split_ntrain(
    grouped: pd.DataFrame,
    benchmark: str,
    metric_label: str,
    ylabel: str,
    title: str,
    out_path: Path,
    style_map: Dict[str, Dict[str, str]],
    dpi: int,
    save_svg: bool,
    horizontal_zero: bool = False,
) -> None:
    bench = grouped[grouped["benchmark"].astype(str) == str(benchmark)].copy()
    if bench.empty:
        return

    n_trains = sorted(bench["n_train"].dropna().astype(int).unique().tolist())
    if not n_trains:
        return

    ncols = min(3, len(n_trains))
    nrows = int(np.ceil(len(n_trains) / ncols))
    fig = plt.figure(figsize=(5.2 * ncols + 4.0, 3.8 * nrows))
    grid = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[1.0] * ncols + [0.85],
    )
    axs = [
        fig.add_subplot(grid[r, c])
        for r in range(nrows)
        for c in range(ncols)
    ]
    legend_ax = fig.add_subplot(grid[:, -1])
    legend_ax.set_axis_off()

    legend_handles = []
    legend_labels = []

    for i, n_train in enumerate(n_trains):
        ax = axs[i]
        block = bench[bench["n_train"].astype(int) == int(n_train)]
        for model, mb in block.groupby("model"):
            mb = mb.sort_values("step")
            st = style_map.get(model, {})
            color = st.get("color")
            marker = st.get("marker", "o")
            linestyle = st.get("linestyle", "-")
            x = mb["step"].to_numpy(dtype=float)
            y = mb["metric_mean"].to_numpy(dtype=float)
            s = mb["metric_std"].fillna(0.0).to_numpy(dtype=float)
            line = ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=4,
                linewidth=1.8,
                label=str(model),
            )[0]
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.16)
            if model not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(model)

        if horizontal_zero:
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_title(f"n_train inicial = {n_train}")
        ax.set_xlabel("Iteracion de infill")
        ax.set_ylabel(ylabel)

    for k in range(len(n_trains), len(axs)):
        axs[k].set_axis_off()

    if legend_handles:
        legend_ax.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            ncol=1,
            frameon=True,
        )

    fig.suptitle(f"{title} - {benchmark}")
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.82, hspace=0.45, wspace=0.30)
    fig._skip_tight_layout = True
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _attach_relative_metric(df: pd.DataFrame, metric: str) -> Tuple[pd.DataFrame, str]:
    out = df.copy()
    info = PREDICTIVE_METRICS[metric]
    relative_col = f"relative_{metric}"
    valid = out.dropna(subset=[metric]).copy()
    if valid.empty:
        out[relative_col] = np.nan
        return out, relative_col

    first_values = (
        valid.sort_values("step")
        .groupby("trajectory_id", as_index=False)
        .first()[["trajectory_id", "step", metric]]
        .rename(columns={"step": "initial_step", metric: f"{metric}_initial"})
    )
    out = out.merge(first_values, on="trajectory_id", how="left")

    if info["direction"] == "max":
        out[relative_col] = out[metric] - out[f"{metric}_initial"]
    else:
        denom = out[f"{metric}_initial"].replace(0.0, np.nan)
        out[relative_col] = out[metric] / denom

    return out, relative_col


def _build_run_improvements(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    info = PREDICTIVE_METRICS[metric]
    base_cols = [
        "trajectory_id",
        "benchmark",
        "n_train",
        "model",
        "sampler",
        "noise",
        "cv_mode",
    ]
    base_cols = [c for c in base_cols if c in df.columns]

    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    valid = df.dropna(subset=[metric]).sort_values("step").copy()
    rows: List[Dict[str, object]] = []
    for trajectory_id, block in valid.groupby("trajectory_id", dropna=False):
        if block.empty:
            continue
        first = block.iloc[0]
        last = block.iloc[-1]
        initial = float(first[metric])
        final = float(last[metric])
        if info["direction"] == "max":
            absolute_improvement = final - initial
            relative_improvement = absolute_improvement
        else:
            absolute_improvement = initial - final
            relative_improvement = (
                absolute_improvement / abs(initial)
                if np.isfinite(initial) and abs(initial) > 1e-12
                else np.nan
            )

        row: Dict[str, object] = {
            "trajectory_id": trajectory_id,
            "initial_step": int(first["step"]),
            "final_step": int(last["step"]),
            "n_steps": int(block["step"].nunique()),
            f"{metric}_initial": initial,
            f"{metric}_final": final,
            f"absolute_improvement_{metric}": absolute_improvement,
            f"relative_improvement_{metric}": relative_improvement,
        }
        for col in base_cols:
            if col == "trajectory_id":
                continue
            row[col] = first[col]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _classify_improvement(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "sin datos"
    if mean_value <= 0:
        return "sin mejora"
    if mean_value >= 0.10 and (pd.isna(std_value) or std_value <= abs(mean_value)):
        return "mejora clara"
    return "mejora debil"


def _summarise_improvements(
    run_improvements: pd.DataFrame,
    metric: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if run_improvements.empty:
        return pd.DataFrame(), pd.DataFrame()

    rel_col = f"relative_improvement_{metric}"
    abs_col = f"absolute_improvement_{metric}"
    initial_col = f"{metric}_initial"
    final_col = f"{metric}_final"

    by_ntrain = (
        run_improvements.groupby(["benchmark", "n_train", "model"], as_index=False)
        .agg(
            **{
                f"{metric}_initial_mean": (initial_col, "mean"),
                f"{metric}_final_mean": (final_col, "mean"),
                f"mean_absolute_improvement_{metric}": (abs_col, "mean"),
                f"mean_relative_improvement_{metric}": (rel_col, "mean"),
                f"std_relative_improvement_{metric}": (rel_col, "std"),
                "n_runs": (rel_col, "count"),
                "initial_step_min": ("initial_step", "min"),
                "initial_step_max": ("initial_step", "max"),
                "final_step_min": ("final_step", "min"),
                "final_step_max": ("final_step", "max"),
            }
        )
        .sort_values(["benchmark", "n_train", f"mean_relative_improvement_{metric}"], ascending=[True, True, False])
    )
    by_ntrain["improvement_direction"] = np.where(
        by_ntrain[f"mean_relative_improvement_{metric}"] > 0,
        "positiva",
        "negativa",
    )

    aggregated = (
        run_improvements.groupby(["benchmark", "model"], as_index=False)
        .agg(
            **{
                f"mean_relative_improvement_{metric}": (rel_col, "mean"),
                f"std_relative_improvement_{metric}": (rel_col, "std"),
                "n_runs": (rel_col, "count"),
            }
        )
        .sort_values(["benchmark", f"mean_relative_improvement_{metric}"], ascending=[True, False])
    )

    best_rows: List[Dict[str, object]] = []
    best_by_ntrain = (
        run_improvements.groupby(["benchmark", "model", "n_train"], as_index=False)
        .agg(mean_improvement=(rel_col, "mean"))
        .dropna(subset=["mean_improvement"])
    )
    for (benchmark, model), block in best_by_ntrain.groupby(["benchmark", "model"]):
        best = block.sort_values("mean_improvement", ascending=False).iloc[0]
        best_rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "best_n_train_initial": int(best["n_train"]),
            }
        )
    best_df = pd.DataFrame(best_rows)
    if not best_df.empty:
        aggregated = aggregated.merge(best_df, on=["benchmark", "model"], how="left")
    else:
        aggregated["best_n_train_initial"] = np.nan

    aggregated["comentario"] = [
        _classify_improvement(mean, std)
        for mean, std in zip(
            aggregated[f"mean_relative_improvement_{metric}"],
            aggregated[f"std_relative_improvement_{metric}"],
        )
    ]

    return by_ntrain, aggregated


def _plot_final_improvement_bars(
    summary: pd.DataFrame,
    metric: str,
    out_path: Path,
    dpi: int,
    save_svg: bool,
) -> None:
    if summary.empty:
        return

    value_col = f"mean_relative_improvement_{metric}"
    std_col = f"std_relative_improvement_{metric}"
    label = PREDICTIVE_METRICS[metric]["label"]
    ylabel = (
        f"Mejora relativa media de {label}"
        if PREDICTIVE_METRICS[metric]["direction"] == "min"
        else f"Mejora media de {label}"
    )

    benchmarks = sorted(summary["benchmark"].dropna().astype(str).unique().tolist())
    if not benchmarks:
        return

    ncols = min(2, len(benchmarks))
    nrows = int(np.ceil(len(benchmarks) / ncols))
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.0 * ncols, 4.2 * nrows),
        sharey=False,
    )
    axs = np.atleast_1d(axs).ravel()

    for i, benchmark in enumerate(benchmarks):
        ax = axs[i]
        block = (
            summary[summary["benchmark"].astype(str) == benchmark]
            .dropna(subset=[value_col])
            .sort_values(value_col, ascending=False)
        )
        if block.empty:
            ax.set_axis_off()
            continue
        xs = np.arange(len(block))
        y = block[value_col].to_numpy(dtype=float)
        yerr = block[std_col].fillna(0.0).to_numpy(dtype=float)
        colors = ["#2a9d8f" if val > 0 else "#c44536" for val in y]
        ax.bar(xs, y, yerr=yerr, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(block["model"].astype(str).tolist(), rotation=35, ha="right")
        ax.set_title(str(benchmark))
        ax.set_ylabel(ylabel)

    for k in range(len(benchmarks), len(axs)):
        axs[k].set_axis_off()

    fig.suptitle(f"Mejora predictiva final por benchmark ({label})")
    fig.subplots_adjust(hspace=0.55, wspace=0.25, top=0.90, bottom=0.18)
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _select_best_mae_models_by_benchmark(mae_summary: pd.DataFrame) -> Dict[str, str]:
    value_col = "mean_relative_improvement_mae"
    required = {"benchmark", "model", value_col}
    if mae_summary.empty or not required.issubset(mae_summary.columns):
        return {}

    ranked = mae_summary.loc[:, ["benchmark", "model", value_col]].copy()
    ranked[value_col] = pd.to_numeric(ranked[value_col], errors="coerce")
    ranked = (
        ranked.dropna(subset=[value_col])
        .sort_values(["benchmark", value_col, "model"], ascending=[True, False, True])
    )
    if ranked.empty:
        return {}

    best = ranked.groupby("benchmark", as_index=False).first()
    return {
        str(row.benchmark): str(row.model)
        for row in best[["benchmark", "model"]].itertuples(index=False)
    }


def _plot_best_model_relative_mae_by_benchmark(
    grouped_relative: pd.DataFrame,
    relative_df: pd.DataFrame,
    mae_summary: pd.DataFrame,
    out_path: Path,
    dpi: int,
    save_svg: bool,
) -> None:
    best_models = _select_best_mae_models_by_benchmark(mae_summary)
    if grouped_relative.empty or not best_models:
        return

    required = {"benchmark", "model", "n_train", "step", "metric_mean", "metric_std"}
    if not required.issubset(grouped_relative.columns):
        return

    available: List[Tuple[str, str]] = []
    for benchmark, model in best_models.items():
        mask = (
            (grouped_relative["benchmark"].astype(str) == benchmark)
            & (grouped_relative["model"].astype(str) == model)
        )
        if mask.any():
            available.append((benchmark, model))
    if not available:
        return

    plot_mask = pd.Series(False, index=grouped_relative.index)
    for benchmark, model in available:
        plot_mask |= (
            (grouped_relative["benchmark"].astype(str) == benchmark)
            & (grouped_relative["model"].astype(str) == model)
        )
    plot_df = grouped_relative[plot_mask].copy()

    n_trains = sorted(plot_df["n_train"].dropna().astype(int).unique().tolist())
    if not n_trains:
        return

    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    ntrain_styles = {
        n_train: {
            "color": cmap(i % 10),
            "marker": markers[i % len(markers)],
        }
        for i, n_train in enumerate(n_trains)
    }

    ncols = 2
    nrows = int(np.ceil(len(available) / ncols))
    fig = plt.figure(figsize=(12.4, 3.35 * nrows + 0.85))
    grid = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[1.0, 1.0, 0.55],
    )
    axs = [
        fig.add_subplot(grid[r, c])
        for r in range(nrows)
        for c in range(ncols)
    ]
    legend_ax = fig.add_subplot(grid[:, -1])
    legend_ax.set_axis_off()

    legend_handles = []
    legend_labels = []

    for i, (benchmark, model) in enumerate(available):
        ax = axs[i]
        block = plot_df[
            (plot_df["benchmark"].astype(str) == benchmark)
            & (plot_df["model"].astype(str) == model)
        ].copy()

        for n_train, nb in block.groupby("n_train"):
            n_train_int = int(n_train)
            nb = nb.sort_values("step")
            style = ntrain_styles[n_train_int]
            x = nb["step"].to_numpy(dtype=float)
            y = nb["metric_mean"].to_numpy(dtype=float)
            s = nb["metric_std"].fillna(0.0).to_numpy(dtype=float)
            label = f"n_train={n_train_int}"
            line = ax.plot(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                markersize=4,
                linewidth=1.8,
                label=label,
            )[0]
            ax.fill_between(x, y - s, y + s, color=style["color"], alpha=0.16)
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_title(f"{benchmark}\n{model}", fontsize=10.5)
        ax.set_xlabel("Iteraci\u00f3n de infill")
        ax.set_ylabel("MAE relativo respecto al inicio")
        ax.margins(x=0.03)

    for k in range(len(available), len(axs)):
        axs[k].set_axis_off()

    if legend_handles:
        legend_ax.legend(
            legend_handles,
            legend_labels,
            title="n_train inicial",
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            borderaxespad=0.0,
            frameon=True,
        )

    ref_source = relative_df.copy()
    if not ref_source.empty and {"benchmark", "model"}.issubset(ref_source.columns):
        ref_mask = pd.Series(False, index=ref_source.index)
        for benchmark, model in available:
            ref_mask |= (
                (ref_source["benchmark"].astype(str) == benchmark)
                & (ref_source["model"].astype(str) == model)
            )
        ref_source = ref_source[ref_mask]

    initial_steps = (
        pd.to_numeric(ref_source.get("initial_step", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
    )
    uses_step_zero = bool(not initial_steps.empty and initial_steps.eq(0).all())
    title = "Evoluci\u00f3n relativa del MAE del mejor modelo por benchmark"
    if not uses_step_zero:
        title += "\nReferencia: primer paso registrado (no hay step=0 en todas las trayectorias)"

    fig.suptitle(title, y=0.985)
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.105,
        top=0.80 if not uses_step_zero else 0.88,
        hspace=0.52,
        wspace=0.30,
    )
    fig._skip_tight_layout = True
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _write_interpretation(
    path: Path,
    mae_summary: pd.DataFrame,
    run_improvements: pd.DataFrame,
    warnings: List[str],
    range_notes: List[str],
) -> None:
    lines: List[str] = [
        "# Interpretacion inicial: evolucion de la capacidad predictiva durante el infill",
        "",
        "Esta interpretacion resume mejoras por trayectoria individual: primero se compara el primer y ultimo paso observado de cada trayectoria, y despues se promedia. No compara medias con distinta composicion de configuraciones.",
        "",
    ]

    if range_notes:
        lines.append("## Notas sobre normalizacion")
        for note in range_notes:
            lines.append(f"- {note}")
        lines.append("")

    if not run_improvements.empty and int(run_improvements["initial_step"].min()) > 0:
        lines.extend(
            [
                "## Aviso sobre el paso inicial",
                "- El archivo no contiene `step=0`; las mejoras se calculan contra el primer paso observado. Para medir estrictamente el efecto desde el diseno inicial previo al primer infill, conviene registrar tambien una fila `step=0` en futuras ejecuciones.",
                "",
            ]
        )

    if warnings:
        lines.append("## Avisos metodologicos")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if mae_summary.empty:
        lines.append("No hay datos suficientes de MAE para generar una interpretacion automatica.")
    else:
        lines.append("## Lectura por benchmark")
        value_col = "mean_relative_improvement_mae"
        std_col = "std_relative_improvement_mae"
        for benchmark, block in mae_summary.groupby("benchmark"):
            block = block.sort_values(value_col, ascending=False)
            positive = block[block[value_col] > 0]
            if positive.empty:
                lines.append(f"### {benchmark}")
                lines.append("- Ningun modelo muestra mejora media positiva de MAE en las trayectorias agregadas.")
                continue

            best = block.iloc[0]
            clear = block[block["comentario"] == "mejora clara"]["model"].astype(str).tolist()
            weak = block[block["comentario"] == "mejora debil"]["model"].astype(str).tolist()
            high_std = block[
                block[std_col].fillna(0.0) > block[value_col].abs().fillna(0.0)
            ]["model"].astype(str).tolist()

            lines.append(f"### {benchmark}")
            lines.append(
                f"- Mayor reduccion media de MAE: {best['model']} "
                f"({best[value_col]:.3f}, n_runs={int(best['n_runs'])})."
            )
            if clear:
                lines.append(f"- Mejora clara: {', '.join(clear)}.")
            if weak:
                lines.append(f"- Mejora debil: {', '.join(weak)}.")

            gp_models = block[block["model"].astype(str).str.startswith("GP_")].copy()
            if not gp_models.empty:
                kernel_text = _kernel_comment(gp_models, value_col)
                if kernel_text:
                    lines.append(f"- Diferencias entre kernels: {kernel_text}")
                ard_text = _ard_comment(gp_models, value_col)
                if ard_text:
                    lines.append(f"- ARD: {ard_text}")
            if high_std:
                lines.append(
                    f"- Desviacion tipica alta frente a la media en: {', '.join(high_std)}; interpretar con cautela."
                )

    lines.extend(
        [
            "",
            "## Nota para el TFG",
            "Una mejora de MAE/RMSE/R2 en test mide capacidad predictiva global del surrogate. No implica necesariamente mejor busqueda del optimo, porque Expected Improvement selecciona puntos para mejorar el incumbente esperado y explorar incertidumbre, no para minimizar directamente el error medio global en todo el dominio.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _kernel_comment(block: pd.DataFrame, value_col: str) -> str:
    ordered = block.dropna(subset=[value_col]).sort_values(value_col, ascending=False)
    if ordered.empty:
        return ""
    best = ordered.iloc[0]
    worst = ordered.iloc[-1]
    if len(ordered) == 1:
        return f"solo hay un kernel GP evaluable ({best['model']})."
    return (
        f"{best['model']} aparece como el kernel con mayor mejora media; "
        f"{worst['model']} queda al final de este benchmark."
    )


def _ard_comment(block: pd.DataFrame, value_col: str) -> str:
    diffs: List[float] = []
    for base in ["GP_RBF", "GP_Matern52"]:
        ard = f"{base}_ARD"
        base_row = block[block["model"].astype(str) == base]
        ard_row = block[block["model"].astype(str) == ard]
        if base_row.empty or ard_row.empty:
            continue
        diffs.append(float(ard_row.iloc[0][value_col]) - float(base_row.iloc[0][value_col]))
    if not diffs:
        return ""
    mean_diff = float(np.nanmean(diffs))
    if mean_diff > 0.02:
        return "las variantes ARD mejoran de media frente a sus equivalentes no ARD."
    if mean_diff < -0.02:
        return "las variantes ARD no ayudan de media frente a sus equivalentes no ARD."
    return "no se observa una diferencia media clara entre variantes ARD y no ARD."


def _plot_incumbent_gap(
    bench_df: pd.DataFrame,
    benchmark: str,
    out_path: Path,
    style_map: Dict[str, Dict[str, str]],
    dpi: int,
    save_svg: bool,
) -> None:
    bench_obj = get_benchmark(str(benchmark))
    opt = getattr(bench_obj, "optimal_value", None)
    if opt is None:
        return

    bench_df = bench_df.copy()
    if "incumbent_best" not in bench_df.columns or bench_df["incumbent_best"].isna().all():
        return

    bench_df["gap_opt"] = bench_df["incumbent_best"] - float(opt)
    bench_df["gap_opt"] = bench_df["gap_opt"].clip(lower=1e-16)

    noises = sorted(bench_df["noise"].dropna().unique().tolist())
    if not noises:
        return

    nrows = len(noises)
    ncols = min(2, nrows)
    nrows_layout = int(np.ceil(nrows / ncols))
    fig, axs = plt.subplots(
        nrows=nrows_layout, ncols=ncols,
        figsize=(8.0 * ncols, 4.2 * nrows_layout),
        sharex=True,
    )
    axs = np.atleast_1d(axs).ravel()

    legend_handles = []
    legend_labels = []

    for i, noise in enumerate(noises):
        ax = axs[i]
        block = bench_df[bench_df["noise"] == noise]
        if block.empty:
            ax.set_axis_off()
            continue

        grouped = (
            block.groupby(["n_train_current", "model", "sampler"], as_index=False)
            .agg(gap_mean=("gap_opt", "mean"))
            .sort_values("n_train_current")
        )

        for (model, sampler), sb in grouped.groupby(["model", "sampler"]):
            st = style_map[model]
            label = f"{model} | {sampler}"
            line = ax.plot(
                sb["n_train_current"],
                sb["gap_mean"],
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                markersize=5,
                linewidth=1.8,
            )[0]
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.set_yscale("log")
        ax.set_title(f"ruido={noise}")
        ax.set_ylabel("Distancia al optimo (log)")

    for k in range(len(noises), len(axs)):
        axs[k].set_axis_off()

    for ax in axs[-ncols:]:
        ax.set_xlabel("n_train_current")

    if legend_handles:
        n_items = len(legend_labels)
        ncol_legend = min(3, n_items)
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol_legend,
            frameon=True,
        )
    fig.subplots_adjust(hspace=0.55, top=0.92, bottom=0.12)
    fig.suptitle(f"Distancia al optimo -- {benchmark}")
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


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
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(12.5, 3.5 * nrows), sharex=True)
    if nrows == 1:
        axs = [axs]

    legend_handles = []
    legend_labels = []

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
            line = ax.plot(
                sb["n_train_current"],
                sb[metric],
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                label=label,
            )[0]
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.set_title(f"{benchmark} | ruido={noise}")
        ax.set_ylabel(ylabel)
        if metric == "nlpd":
            ax.set_yscale("log")

    axs[-1].set_xlabel("n_train_current")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(3, len(legend_labels)),
            frameon=True,
        )
    fig.subplots_adjust(hspace=0.50, top=0.92, bottom=0.10)
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
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
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

    ax.set_title(f"Evolucion agregada de {ylabel} (media +/- std) - {benchmark}")
    ax.set_xlabel("n_train_current")
    ax.set_ylabel(ylabel)
    if metric == "nlpd":
        ax.set_yscale("log")
    place_legend(ax, outside=True)
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def generate_evolution_plots(
    master_df: pd.DataFrame,
    out_dir: Path,
    tables_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, object]] = None,
    dpi: int = 300,
    save_svg: bool = False,
) -> Dict[str, pd.DataFrame]:
    if master_df.empty:
        return {}
    out_dir = Path(out_dir)
    tables_dir = Path(tables_dir) if tables_dir is not None else out_dir.parent.parent / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    analysis_df = _normalise_active_df(master_df)
    if analysis_df.empty:
        return {}

    analysis_df, range_notes = _infer_y_test_ranges(analysis_df, metadata=metadata)
    style_map = build_model_style_map(analysis_df["model"].unique())
    outputs: Dict[str, pd.DataFrame] = {}
    methodological_warnings: List[str] = []

    if int(analysis_df["step"].min()) > 0:
        methodological_warnings.append(
            "WARNING: active_trajectory.csv no contiene step=0; las mejoras usan el primer paso observado como inicial."
        )

    if "mae" in analysis_df.columns and "range_y_test" in analysis_df.columns:
        denom = pd.to_numeric(analysis_df["range_y_test"], errors="coerce").replace(0.0, np.nan)
        analysis_df["mae_norm"] = analysis_df["mae"] / denom
        grouped_norm = _aggregate_step_metric(analysis_df, "mae_norm")
        if not grouped_norm.empty:
            for benchmark in sorted(grouped_norm["benchmark"].dropna().astype(str).unique()):
                safe_benchmark = sanitize_filename(benchmark)
                _plot_metric_by_step_split_ntrain(
                    grouped=grouped_norm,
                    benchmark=benchmark,
                    metric_label="MAE normalizado",
                    ylabel="MAE / rango de y_test",
                    title="Evolucion normalizada del MAE durante el infill",
                    out_path=out_dir / f"evolution_mae_norm_by_step_split_ntrain_{safe_benchmark}",
                    style_map=style_map,
                    dpi=dpi,
                    save_svg=save_svg,
                )

    mae_summary_for_text = pd.DataFrame()
    mae_runs_for_text = pd.DataFrame()

    for metric, info in PREDICTIVE_METRICS.items():
        if metric not in analysis_df.columns:
            continue

        counts = _build_counts_by_step(analysis_df, metric)
        count_name = f"counts_by_step_{metric}"
        outputs[count_name] = counts
        _save_table(counts, tables_dir / f"{count_name}.csv")
        count_warnings = _count_warnings(counts, metric)
        methodological_warnings.extend(count_warnings)

        grouped = _aggregate_step_metric(analysis_df, metric)
        if not grouped.empty:
            for benchmark in sorted(grouped["benchmark"].dropna().astype(str).unique()):
                safe_benchmark = sanitize_filename(benchmark)
                _plot_metric_by_step_split_ntrain(
                    grouped=grouped,
                    benchmark=benchmark,
                    metric_label=info["label"],
                    ylabel=info["ylabel"],
                    title=f"Evolucion de {info['label']} durante el infill",
                    out_path=out_dir / f"evolution_{metric}_by_step_split_ntrain_{safe_benchmark}",
                    style_map=style_map,
                    dpi=dpi,
                    save_svg=save_svg,
                )

        relative_df, relative_col = _attach_relative_metric(analysis_df, metric)
        grouped_relative = _aggregate_step_metric(relative_df, relative_col)
        if not grouped_relative.empty:
            rel_prefix = "relative" if info["direction"] == "min" else "delta"
            for benchmark in sorted(grouped_relative["benchmark"].dropna().astype(str).unique()):
                safe_benchmark = sanitize_filename(benchmark)
                _plot_metric_by_step_split_ntrain(
                    grouped=grouped_relative,
                    benchmark=benchmark,
                    metric_label=info["label"],
                    ylabel=info["relative_ylabel"],
                    title=f"Evolucion relativa de {info['label']} durante el infill",
                    out_path=out_dir / f"evolution_{rel_prefix}_{metric}_by_step_split_ntrain_{safe_benchmark}",
                    style_map=style_map,
                    dpi=dpi,
                    save_svg=save_svg,
                    horizontal_zero=info["direction"] == "max",
                )

        run_improvements = _build_run_improvements(analysis_df, metric)
        if run_improvements.empty:
            continue

        by_ntrain, aggregated = _summarise_improvements(run_improvements, metric)
        by_ntrain_name = f"summary_{metric}_initial_final_by_benchmark_ntrain_model"
        if metric == "r2":
            aggregated_name = "summary_r2_improvement_by_benchmark_model"
        else:
            aggregated_name = f"summary_{metric}_relative_improvement_by_benchmark_model"

        outputs[by_ntrain_name] = by_ntrain
        outputs[aggregated_name] = aggregated
        _save_table(by_ntrain, tables_dir / f"{by_ntrain_name}.csv")
        _save_table(aggregated, tables_dir / f"{aggregated_name}.csv")

        _plot_final_improvement_bars(
            summary=aggregated,
            metric=metric,
            out_path=out_dir / f"final_relative_improvement_{metric}_by_benchmark",
            dpi=dpi,
            save_svg=save_svg,
        )

        if metric == "mae":
            _plot_best_model_relative_mae_by_benchmark(
                grouped_relative=grouped_relative,
                relative_df=relative_df,
                mae_summary=aggregated,
                out_path=out_dir / "evolution_relative_mae_best_model_by_benchmark",
                dpi=dpi,
                save_svg=save_svg,
            )
            mae_summary_for_text = aggregated
            mae_runs_for_text = run_improvements

    methodological_warnings = list(dict.fromkeys(methodological_warnings))
    for warning in methodological_warnings:
        if warning.startswith("WARNING:"):
            print(warning)

    _write_interpretation(
        path=tables_dir / "interpretacion_evolucion_infill.md",
        mae_summary=mae_summary_for_text,
        run_improvements=mae_runs_for_text,
        warnings=methodological_warnings,
        range_notes=range_notes,
    )

    for benchmark, bench_df in analysis_df.groupby("benchmark", dropna=False):
        bdir = out_dir / str(benchmark)
        bdir.mkdir(parents=True, exist_ok=True)

        _plot_incumbent_gap(
            bench_df=bench_df,
            benchmark=str(benchmark),
            out_path=bdir / "evolucion_gap_optimo_filas_por_ruido",
            style_map=style_map,
            dpi=dpi,
            save_svg=save_svg,
        )

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

    return outputs
