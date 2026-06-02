from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarks import get_benchmark
from src.benchmarks.dataset_generator import generate_benchmark_dataset

from .styling import build_model_style_map, sanitize_filename, save_figure


CONFIG_KEYS = ["benchmark", "sampler", "n_train", "noise", "cv_mode"]
TRAJECTORY_KEYS = CONFIG_KEYS + ["model"]
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


def _noise_label_to_cfg(noise_label: str) -> Dict[str, object]:
    if noise_label == "NoNoise":
        return {"type": "none", "kwargs": {}}
    if noise_label.startswith("Gaussian_s"):
        try:
            sigma = float(noise_label.split("Gaussian_s", 1)[1])
        except Exception:
            sigma = 0.1
        return {"type": "gaussian", "kwargs": {"sigma": sigma}}
    if noise_label == "Heteroscedastic":
        return {"type": "heteroscedastic", "kwargs": {"sigma_base": 0.02, "sigma_scale": 0.15}}
    if noise_label == "Proportional":
        return {"type": "proportional", "kwargs": {"sigma_rel": 0.05, "sigma_base": 0.01}}
    return {"type": "none", "kwargs": {}}


def _parse_x_next_cell(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float).ravel()
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).ravel()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.empty((0,), dtype=float)
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return np.asarray(parsed, dtype=float).ravel()
        except Exception:
            pass
    try:
        return np.asarray([float(value)], dtype=float)
    except Exception:
        return np.empty((0,), dtype=float)


def _normalise_active_df(master_df: pd.DataFrame) -> pd.DataFrame:
    df = master_df.copy()
    rename_map = {
        "mae_test": "mae",
        "rmse_test": "rmse",
        "r2_test": "r2",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "cv_mode" in df.columns:
        df = df[df["cv_mode"].astype(str).str.lower() == "active"].copy()
    if "model" in df.columns:
        df = df[~df["model"].astype(str).str.contains("dummy", case=False, na=False)].copy()

    for col in ["n_train", "step", "n_train_current", "incumbent_best", "y_next"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = set(TRAJECTORY_KEYS + ["step", "x_next"])
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.dropna(subset=["benchmark", "sampler", "n_train", "noise", "cv_mode", "model", "step"]).copy()
    df["n_train"] = df["n_train"].astype(int)
    df["step"] = df["step"].astype(int)
    trajectory_keys = [
        c
        for c in BASE_TRAJECTORY_KEYS + OPTIONAL_TRAJECTORY_KEYS
        if c in df.columns and not df[c].isna().all()
    ]
    df["trajectory_id"] = df[trajectory_keys].astype(str).agg("|".join, axis=1)
    return df.sort_values(TRAJECTORY_KEYS + ["step"]).reset_index(drop=True)


def _get_initial_clean_state(
    cache: Dict[Tuple[str, str, int, str], Tuple[object, np.ndarray, np.ndarray]],
    benchmark: str,
    sampler: str,
    n_train: int,
    noise: str,
    metadata: Optional[Dict[str, object]],
) -> Tuple[object, np.ndarray, np.ndarray]:
    key = (benchmark, sampler, int(n_train), noise)
    if key in cache:
        return cache[key]

    seed = int((metadata or {}).get("seed", 42) or 42)
    n_test = int((metadata or {}).get("n_test", 200) or 200)
    noise_cfg = _noise_label_to_cfg(noise)
    dataset = generate_benchmark_dataset(
        benchmark=benchmark,
        n_train=int(n_train),
        n_test=n_test,
        sampler=sampler,
        noise=str(noise_cfg["type"]),
        noise_kwargs=dict(noise_cfg["kwargs"]),
        n_groups=None,
        seed=seed,
    )
    bench = get_benchmark(benchmark)
    X_initial = np.asarray(dataset.X_train, dtype=float)
    y_initial_clean = np.asarray(bench(X_initial), dtype=float).ravel()
    cache[key] = (bench, X_initial, y_initial_clean)
    return cache[key]


def _safe_relative_reduction(initial: float, current: float) -> float:
    if not np.isfinite(initial) or not np.isfinite(current):
        return np.nan
    denom = abs(float(initial))
    if denom <= 1e-12:
        return np.nan
    return (float(initial) - float(current)) / denom


def _gap_reduction(initial_gap: float, current_gap: float) -> float:
    if not np.isfinite(initial_gap) or not np.isfinite(current_gap) or initial_gap <= 1e-12:
        return np.nan
    return (initial_gap - current_gap) / initial_gap


def _build_clean_incumbent_trajectories(
    active_df: pd.DataFrame,
    metadata: Optional[Dict[str, object]],
) -> pd.DataFrame:
    if active_df.empty:
        return pd.DataFrame()

    cache: Dict[Tuple[str, str, int, str], Tuple[object, np.ndarray, np.ndarray]] = {}
    rows: List[Dict[str, object]] = []

    for trajectory_id, block in active_df.groupby("trajectory_id", dropna=False):
        block = block.sort_values("step").copy()
        first = block.iloc[0]
        benchmark = str(first["benchmark"])
        sampler = str(first["sampler"])
        n_train = int(first["n_train"])
        noise = str(first["noise"])
        model = str(first["model"])
        cv_mode = str(first["cv_mode"])

        bench, _, y_initial_clean = _get_initial_clean_state(
            cache=cache,
            benchmark=benchmark,
            sampler=sampler,
            n_train=n_train,
            noise=noise,
            metadata=metadata,
        )
        opt_raw = getattr(bench, "optimal_value", None)
        optimum_available = opt_raw is not None and np.isfinite(float(opt_raw))
        optimal_value = float(opt_raw) if optimum_available else np.nan

        clean_values = [float(v) for v in y_initial_clean if np.isfinite(v)]
        initial_clean_best = float(np.min(clean_values)) if clean_values else np.nan
        initial_gap_raw = initial_clean_best - optimal_value if optimum_available else np.nan
        initial_gap = max(0.0, initial_gap_raw) if optimum_available and np.isfinite(initial_gap_raw) else np.nan

        additions: List[Tuple[int, float]] = []
        for rec in block.itertuples(index=False):
            x = _parse_x_next_cell(getattr(rec, "x_next"))
            if x.size != int(bench.dim):
                continue
            clean_y = float(np.asarray(bench(x.reshape(1, -1)), dtype=float).ravel()[0])
            additions.append((int(getattr(rec, "step")), clean_y))

        for rec in block.itertuples(index=False):
            step = int(getattr(rec, "step"))
            step_clean_values = clean_values + [y for add_step, y in additions if add_step <= step]
            clean_best = float(np.min(step_clean_values)) if step_clean_values else np.nan
            gap_raw = clean_best - optimal_value if optimum_available else np.nan
            gap = max(0.0, gap_raw) if optimum_available and np.isfinite(gap_raw) else np.nan
            value_reduction = _safe_relative_reduction(initial_clean_best, clean_best)
            gap_reduction = _gap_reduction(initial_gap, gap) if optimum_available else np.nan
            primary_improvement = gap_reduction if optimum_available else value_reduction

            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "benchmark": benchmark,
                    "sampler": sampler,
                    "n_train": n_train,
                    "noise": noise,
                    "cv_mode": cv_mode,
                    "model": model,
                    "step": step,
                    "n_train_current": getattr(rec, "n_train_current", np.nan),
                    "observed_incumbent_best": getattr(rec, "incumbent_best", np.nan),
                    "clean_best_found": clean_best,
                    "initial_clean_best": initial_clean_best,
                    "optimal_value": optimal_value,
                    "optimum_available": bool(optimum_available),
                    "clean_gap_to_opt_raw": gap_raw,
                    "clean_gap_to_opt": gap,
                    "initial_gap_to_opt": initial_gap,
                    "relative_gap_to_initial": gap / initial_gap if optimum_available and initial_gap > 1e-12 else np.nan,
                    "relative_gap_reduction": gap_reduction,
                    "relative_clean_value_reduction": value_reduction,
                    "relative_incumbent_improvement": primary_improvement,
                    "incumbent_metric_note": (
                        "gap_reduction_to_known_optimum"
                        if optimum_available
                        else "clean_value_reduction_no_known_optimum"
                    ),
                }
            )

    return pd.DataFrame(rows)


def _build_counts(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    if trajectory_df.empty:
        return pd.DataFrame(columns=["benchmark", "n_train", "model", "step", "n_runs"])
    return (
        trajectory_df.groupby(["benchmark", "n_train", "model", "step"], as_index=False)
        .agg(n_runs=("trajectory_id", "nunique"))
        .sort_values(["benchmark", "n_train", "model", "step"])
    )


def _counts_warnings(counts: pd.DataFrame) -> List[str]:
    warnings: List[str] = []
    if counts.empty:
        return warnings
    for (benchmark, n_train, model), block in counts.groupby(["benchmark", "n_train", "model"]):
        variants = sorted(block["n_runs"].dropna().astype(int).unique().tolist())
        if len(variants) > 1:
            warnings.append(
                f"WARNING: incumbent tiene n_runs variable en {benchmark}, "
                f"n_train={n_train}, model={model}: {variants}"
            )
    return warnings


def _aggregate_by_step(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    if trajectory_df.empty:
        return pd.DataFrame()
    return (
        trajectory_df.groupby(["benchmark", "n_train", "step", "model"], as_index=False)
        .agg(
            improvement_mean=("relative_incumbent_improvement", "mean"),
            improvement_std=("relative_incumbent_improvement", "std"),
            clean_best_mean=("clean_best_found", "mean"),
            clean_best_std=("clean_best_found", "std"),
            gap_mean=("clean_gap_to_opt", "mean"),
            gap_std=("clean_gap_to_opt", "std"),
            n_runs=("trajectory_id", "nunique"),
            optimum_available=("optimum_available", "all"),
        )
        .sort_values(["benchmark", "n_train", "model", "step"])
    )


def _build_run_improvements(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    if trajectory_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for trajectory_id, block in trajectory_df.sort_values("step").groupby("trajectory_id", dropna=False):
        first = block.iloc[0]
        last = block.iloc[-1]
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "benchmark": first["benchmark"],
                "sampler": first["sampler"],
                "n_train": int(first["n_train"]),
                "noise": first["noise"],
                "cv_mode": first["cv_mode"],
                "model": first["model"],
                "initial_step": int(first["step"]),
                "final_step": int(last["step"]),
                "optimum_available": bool(first["optimum_available"]),
                "optimal_value": first["optimal_value"],
                "clean_initial_best": first["clean_best_found"],
                "clean_final_best": last["clean_best_found"],
                "gap_initial": first["clean_gap_to_opt"],
                "gap_final": last["clean_gap_to_opt"],
                "relative_gap_reduction": last["relative_gap_reduction"],
                "relative_clean_value_reduction": last["relative_clean_value_reduction"],
                "relative_incumbent_improvement": last["relative_incumbent_improvement"],
                "incumbent_metric_note": last["incumbent_metric_note"],
            }
        )
    return pd.DataFrame(rows)


def _classify_improvement(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "sin datos"
    if mean_value <= 0:
        return "sin mejora"
    if mean_value >= 0.50 and (pd.isna(std_value) or std_value <= abs(mean_value)):
        return "mejora clara del minimo"
    if mean_value >= 0.10:
        return "mejora moderada del minimo"
    return "mejora debil del minimo"


def _summarise_run_improvements(run_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if run_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_ntrain = (
        run_df.groupby(["benchmark", "n_train", "model"], as_index=False)
        .agg(
            optimum_available=("optimum_available", "all"),
            optimal_value=("optimal_value", "first"),
            clean_initial_best_mean=("clean_initial_best", "mean"),
            clean_final_best_mean=("clean_final_best", "mean"),
            gap_initial_mean=("gap_initial", "mean"),
            gap_final_mean=("gap_final", "mean"),
            mean_relative_gap_reduction=("relative_gap_reduction", "mean"),
            mean_relative_clean_value_reduction=("relative_clean_value_reduction", "mean"),
            mean_relative_incumbent_improvement=("relative_incumbent_improvement", "mean"),
            std_relative_incumbent_improvement=("relative_incumbent_improvement", "std"),
            n_runs=("relative_incumbent_improvement", "count"),
            initial_step_min=("initial_step", "min"),
            initial_step_max=("initial_step", "max"),
            final_step_min=("final_step", "min"),
            final_step_max=("final_step", "max"),
            metric_note=("incumbent_metric_note", "first"),
        )
        .sort_values(["benchmark", "n_train", "mean_relative_incumbent_improvement"], ascending=[True, True, False])
    )
    by_ntrain["improvement_direction"] = np.where(
        by_ntrain["mean_relative_incumbent_improvement"] > 0,
        "positiva",
        "negativa",
    )

    aggregated = (
        run_df.groupby(["benchmark", "model"], as_index=False)
        .agg(
            optimum_available=("optimum_available", "all"),
            optimal_value=("optimal_value", "first"),
            mean_relative_gap_reduction=("relative_gap_reduction", "mean"),
            mean_relative_clean_value_reduction=("relative_clean_value_reduction", "mean"),
            mean_relative_incumbent_improvement=("relative_incumbent_improvement", "mean"),
            std_relative_incumbent_improvement=("relative_incumbent_improvement", "std"),
            n_runs=("relative_incumbent_improvement", "count"),
            metric_note=("incumbent_metric_note", "first"),
        )
        .sort_values(["benchmark", "mean_relative_incumbent_improvement"], ascending=[True, False])
    )

    best_by_ntrain = (
        run_df.groupby(["benchmark", "model", "n_train"], as_index=False)
        .agg(mean_improvement=("relative_incumbent_improvement", "mean"))
        .dropna(subset=["mean_improvement"])
    )
    best_rows: List[Dict[str, object]] = []
    for (benchmark, model), block in best_by_ntrain.groupby(["benchmark", "model"]):
        best = block.sort_values("mean_improvement", ascending=False).iloc[0]
        best_rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "best_n_train_initial": int(best["n_train"]),
            }
        )
    if best_rows:
        aggregated = aggregated.merge(pd.DataFrame(best_rows), on=["benchmark", "model"], how="left")
    else:
        aggregated["best_n_train_initial"] = np.nan

    aggregated["comentario"] = [
        _classify_improvement(mean_value, std_value)
        for mean_value, std_value in zip(
            aggregated["mean_relative_incumbent_improvement"],
            aggregated["std_relative_incumbent_improvement"],
        )
    ]
    return by_ntrain, aggregated


def _plot_evolution_by_benchmark(
    grouped: pd.DataFrame,
    benchmark: str,
    out_path: Path,
    style_map: Dict[str, Dict[str, str]],
    dpi: int,
    save_svg: bool,
) -> None:
    bench = grouped[grouped["benchmark"].astype(str) == str(benchmark)].copy()
    if bench.empty:
        return

    n_trains = sorted(bench["n_train"].dropna().astype(int).unique().tolist())
    ncols = min(3, len(n_trains))
    nrows = int(np.ceil(len(n_trains) / ncols))
    fig = plt.figure(figsize=(5.2 * ncols + 4.0, 3.8 * nrows))
    grid = fig.add_gridspec(nrows=nrows, ncols=ncols + 1, width_ratios=[1.0] * ncols + [0.85])
    axs = [fig.add_subplot(grid[r, c]) for r in range(nrows) for c in range(ncols)]
    legend_ax = fig.add_subplot(grid[:, -1])
    legend_ax.set_axis_off()

    legend_handles = []
    legend_labels = []
    for i, n_train in enumerate(n_trains):
        ax = axs[i]
        block = bench[bench["n_train"].astype(int) == n_train]
        for model, mb in block.groupby("model"):
            mb = mb.sort_values("step")
            st = style_map.get(model, {})
            color = st.get("color")
            line = ax.plot(
                mb["step"].to_numpy(dtype=float),
                mb["improvement_mean"].to_numpy(dtype=float),
                color=color,
                marker=st.get("marker", "o"),
                linestyle=st.get("linestyle", "-"),
                markersize=4,
                linewidth=1.8,
                label=str(model),
            )[0]
            std = mb["improvement_std"].fillna(0.0).to_numpy(dtype=float)
            mean = mb["improvement_mean"].to_numpy(dtype=float)
            x = mb["step"].to_numpy(dtype=float)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16)
            if model not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(model)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_title(f"n_train inicial = {n_train}")
        ax.set_xlabel("Iteracion de infill")
        ax.set_ylabel("Mejora relativa del incumbent")

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

    optimum_available = bool(bench["optimum_available"].all())
    subtitle = "reduccion del gap al optimo" if optimum_available else "reduccion del mejor valor limpio inicial"
    fig.suptitle(f"Evolucion del minimo encontrado durante el infill - {benchmark}\n({subtitle})", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.78, hspace=0.45, wspace=0.30)
    fig._skip_tight_layout = True
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _plot_final_bars(summary: pd.DataFrame, out_path: Path, dpi: int, save_svg: bool) -> None:
    if summary.empty:
        return
    benchmarks = sorted(summary["benchmark"].dropna().astype(str).unique().tolist())
    ncols = 2
    nrows = int(np.ceil(len(benchmarks) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.5, 4.25 * nrows))
    axs = np.atleast_1d(axs).ravel()

    for i, benchmark in enumerate(benchmarks):
        ax = axs[i]
        block = (
            summary[summary["benchmark"].astype(str) == benchmark]
            .dropna(subset=["mean_relative_incumbent_improvement"])
            .sort_values("mean_relative_incumbent_improvement", ascending=False)
        )
        if block.empty:
            ax.set_axis_off()
            continue
        x = np.arange(len(block))
        y = block["mean_relative_incumbent_improvement"].to_numpy(dtype=float)
        yerr = block["std_relative_incumbent_improvement"].fillna(0.0).to_numpy(dtype=float)
        colors = ["#2a9d8f" if val > 0 else "#c44536" for val in y]
        ax.bar(x, y, yerr=yerr, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(block["model"].astype(str).tolist(), rotation=35, ha="right")
        ax.set_title(str(benchmark))
        ax.set_ylabel("Mejora relativa final del incumbent")

    for k in range(len(benchmarks), len(axs)):
        axs[k].set_axis_off()

    fig.suptitle("Mejora final del minimo encontrado por benchmark")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.90, hspace=0.58, wspace=0.28)
    fig._skip_tight_layout = True
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _select_best_models(summary: pd.DataFrame) -> Dict[str, str]:
    if summary.empty:
        return {}
    ranked = summary.dropna(subset=["mean_relative_incumbent_improvement"]).copy()
    if ranked.empty:
        return {}
    ranked = ranked.sort_values(
        ["benchmark", "mean_relative_incumbent_improvement", "model"],
        ascending=[True, False, True],
    )
    best = ranked.groupby("benchmark", as_index=False).first()
    return {str(row.benchmark): str(row.model) for row in best[["benchmark", "model"]].itertuples(index=False)}


def _plot_best_model_evolution(
    grouped: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path,
    dpi: int,
    save_svg: bool,
) -> None:
    best_models = _select_best_models(summary)
    if grouped.empty or not best_models:
        return

    available = [
        (benchmark, model)
        for benchmark, model in best_models.items()
        if (
            (grouped["benchmark"].astype(str) == benchmark)
            & (grouped["model"].astype(str) == model)
        ).any()
    ]
    if not available:
        return

    plot_mask = pd.Series(False, index=grouped.index)
    for benchmark, model in available:
        plot_mask |= (
            (grouped["benchmark"].astype(str) == benchmark)
            & (grouped["model"].astype(str) == model)
        )
    plot_df = grouped[plot_mask].copy()
    n_trains = sorted(plot_df["n_train"].dropna().astype(int).unique().tolist())
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    ntrain_styles = {
        n_train: {"color": cmap(i % 10), "marker": markers[i % len(markers)]}
        for i, n_train in enumerate(n_trains)
    }

    ncols = 2
    nrows = int(np.ceil(len(available) / ncols))
    fig = plt.figure(figsize=(12.4, 3.35 * nrows + 0.85))
    grid = fig.add_gridspec(nrows=nrows, ncols=ncols + 1, width_ratios=[1.0, 1.0, 0.55])
    axs = [fig.add_subplot(grid[r, c]) for r in range(nrows) for c in range(ncols)]
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
            y = nb["improvement_mean"].to_numpy(dtype=float)
            s = nb["improvement_std"].fillna(0.0).to_numpy(dtype=float)
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
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_title(f"{benchmark}\n{model}", fontsize=10.5)
        ax.set_xlabel("Iteracion de infill")
        ax.set_ylabel("Mejora relativa del incumbent")

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

    fig.suptitle("Evolucion del minimo encontrado del mejor modelo por benchmark", y=0.985)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.105, top=0.88, hspace=0.52, wspace=0.30)
    fig._skip_tight_layout = True
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _write_interpretation(
    path: Path,
    summary: pd.DataFrame,
    warnings: List[str],
) -> None:
    lines: List[str] = [
        "# Interpretacion inicial: evolucion del minimo encontrado durante el infill",
        "",
        "Esta seccion evalua el objetivo de optimizacion del proceso de infill: si las iteraciones encuentran puntos con menor valor de la funcion objetivo.",
        "",
        "La metrica principal se calcula por trayectoria antes de promediar. Para benchmarks con optimo conocido se usa la reduccion relativa del gap al optimo. Para benchmarks sin optimo cerrado en el codigo, como Borehole, se usa la reduccion relativa del mejor valor limpio respecto al diseno inicial.",
        "",
        "Se evalua el valor limpio de la funcion benchmark en los puntos encontrados. En presencia de ruido, esto evita confundir una observacion ruidosa favorable con una mejora real del punto encontrado.",
        "",
    ]

    if warnings:
        lines.append("## Avisos metodologicos")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if summary.empty:
        lines.append("No hay datos suficientes para interpretar el incumbent.")
    else:
        lines.append("## Lectura por benchmark")
        for benchmark, block in summary.groupby("benchmark"):
            block = block.sort_values("mean_relative_incumbent_improvement", ascending=False)
            best = block.iloc[0]
            metric_note = str(best.get("metric_note", ""))
            metric_text = (
                "reduccion del gap al optimo"
                if metric_note == "gap_reduction_to_known_optimum"
                else "reduccion del mejor valor limpio inicial"
            )
            high_std = block[
                block["std_relative_incumbent_improvement"].fillna(0.0)
                > block["mean_relative_incumbent_improvement"].abs().fillna(0.0)
            ]["model"].astype(str).tolist()
            no_improve = block[block["mean_relative_incumbent_improvement"] <= 0]["model"].astype(str).tolist()

            lines.append(f"### {benchmark}")
            lines.append(
                f"- Mejor modelo segun {metric_text}: {best['model']} "
                f"({best['mean_relative_incumbent_improvement']:.3f}, n_runs={int(best['n_runs'])})."
            )
            lines.append(f"- Comentario automatico: {best['comentario']}.")
            if no_improve:
                lines.append(f"- Sin mejora media positiva: {', '.join(no_improve)}.")
            if high_std:
                lines.append(
                    f"- Desviacion tipica alta frente a la media en: {', '.join(high_std)}; interpretar con cautela."
                )
            if metric_note != "gap_reduction_to_known_optimum":
                lines.append(
                    "- No se usa gap al optimo porque el benchmark no define `optimal_value`; se informa mejora del mejor valor limpio encontrado."
                )
            lines.append("")

    lines.extend(
        [
            "## Nota para el TFG",
            "Este analisis no mide error predictivo global del GP. Mide exito de optimizacion: si Expected Improvement mejora el mejor punto encontrado. Por tanto debe presentarse como complementario, y mas central para infill, que MAE/RMSE/R2.",
            "",
            "Dummy se deja fuera porque no propone puntos mediante EI. La baseline relevante aqui es `step=0`, es decir, el mejor punto de la matriz de diseno inicial.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_incumbent_outputs(
    master_df: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    metadata: Optional[Dict[str, object]] = None,
    dpi: int = 300,
    save_svg: bool = False,
) -> Dict[str, pd.DataFrame]:
    active_df = _normalise_active_df(master_df)
    if active_df.empty:
        return {}

    trajectory_df = _build_clean_incumbent_trajectories(active_df=active_df, metadata=metadata)
    if trajectory_df.empty:
        return {}

    counts = _build_counts(trajectory_df)
    warnings = _counts_warnings(counts)
    grouped = _aggregate_by_step(trajectory_df)
    run_df = _build_run_improvements(trajectory_df)
    by_ntrain, aggregated = _summarise_run_improvements(run_df)

    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    counts.to_csv(tables_dir / "counts_by_step_incumbent.csv", index=False)
    by_ntrain.to_csv(tables_dir / "summary_incumbent_initial_final_by_benchmark_ntrain_model.csv", index=False)
    aggregated.to_csv(tables_dir / "summary_incumbent_relative_improvement_by_benchmark_model.csv", index=False)

    style_map = build_model_style_map(active_df["model"].unique())
    for benchmark in sorted(grouped["benchmark"].dropna().astype(str).unique().tolist()):
        safe_benchmark = sanitize_filename(benchmark)
        _plot_evolution_by_benchmark(
            grouped=grouped,
            benchmark=benchmark,
            out_path=figures_dir / f"evolution_incumbent_relative_improvement_by_step_split_ntrain_{safe_benchmark}",
            style_map=style_map,
            dpi=dpi,
            save_svg=save_svg,
        )

    _plot_final_bars(
        summary=aggregated,
        out_path=figures_dir / "final_relative_incumbent_improvement_by_benchmark",
        dpi=dpi,
        save_svg=save_svg,
    )
    _plot_best_model_evolution(
        grouped=grouped,
        summary=aggregated,
        out_path=figures_dir / "evolution_incumbent_best_model_by_benchmark",
        dpi=dpi,
        save_svg=save_svg,
    )
    _write_interpretation(
        path=tables_dir / "interpretacion_incumbent_infill.md",
        summary=aggregated,
        warnings=warnings,
    )

    for warning in warnings:
        if warning.startswith("WARNING:"):
            print(warning)

    return {
        "counts_by_step_incumbent": counts,
        "summary_incumbent_initial_final_by_benchmark_ntrain_model": by_ntrain,
        "summary_incumbent_relative_improvement_by_benchmark_model": aggregated,
    }
