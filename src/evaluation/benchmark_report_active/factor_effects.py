from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .incumbent_analysis import (
    _build_clean_incumbent_trajectories,
    _build_run_improvements,
    _normalise_active_df as _normalise_incumbent_df,
)
from .predictive_diagnostics import _normalise_active_df as _normalise_predictive_df
from .styling import save_figure


PAIR_KEYS_BASE = ["benchmark", "model", "n_train", "sampler", "noise", "cv_mode"]


def _build_mae_run_improvements(master_df: pd.DataFrame) -> pd.DataFrame:
    df = _normalise_predictive_df(master_df)
    if df.empty or "mae" not in df.columns:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for trajectory_id, block in df.sort_values("step").groupby("trajectory_id", dropna=False):
        block = block.dropna(subset=["mae"]).sort_values("step")
        if block.empty:
            continue
        first = block.iloc[0]
        last = block.iloc[-1]
        mae_initial = float(first["mae"])
        mae_final = float(last["mae"])
        rel = np.nan
        if np.isfinite(mae_initial) and abs(mae_initial) > 1e-12:
            rel = (mae_initial - mae_final) / mae_initial
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "benchmark": first["benchmark"],
                "model": first["model"],
                "n_train": int(first["n_train"]),
                "sampler": first["sampler"],
                "noise": first["noise"],
                "cv_mode": first["cv_mode"],
                "mae_initial": mae_initial,
                "mae_final": mae_final,
                "relative_improvement_mae": rel,
                "initial_step": int(first["step"]),
                "final_step": int(last["step"]),
            }
        )
    return pd.DataFrame(rows)


def _build_incumbent_run_improvements(
    master_df: pd.DataFrame,
    metadata: Optional[Dict[str, object]],
) -> pd.DataFrame:
    active_df = _normalise_incumbent_df(master_df)
    if active_df.empty:
        return pd.DataFrame()
    trajectory_df = _build_clean_incumbent_trajectories(active_df=active_df, metadata=metadata)
    run_df = _build_run_improvements(trajectory_df)
    if run_df.empty:
        return pd.DataFrame()
    return run_df.rename(
        columns={
            "relative_incumbent_improvement": "relative_incumbent_improvement",
        }
    )


def _combine_run_metrics(master_df: pd.DataFrame, metadata: Optional[Dict[str, object]]) -> pd.DataFrame:
    mae = _build_mae_run_improvements(master_df)
    inc = _build_incumbent_run_improvements(master_df, metadata=metadata)
    if mae.empty and inc.empty:
        return pd.DataFrame()
    if inc.empty:
        return mae
    if mae.empty:
        return inc

    keep_inc = [
        "trajectory_id",
        "relative_incumbent_improvement",
        "clean_initial_best",
        "clean_final_best",
        "gap_initial",
        "gap_final",
        "metric_note",
    ]
    keep_inc = [c for c in keep_inc if c in inc.columns]
    combined = mae.merge(inc[keep_inc], on="trajectory_id", how="inner")
    return combined


def _paired_delta(
    df: pd.DataFrame,
    metric: str,
    factor: str,
    comparison: str,
    pair_keys: List[str],
    variant_col: str,
    left_value: str,
    right_value: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    cols = pair_keys + [variant_col, metric]
    if df.empty or not set(cols).issubset(df.columns):
        empty_summary = {
            "factor": factor,
            "metric": metric,
            "comparison": comparison,
            "mean_delta": np.nan,
            "std_delta": np.nan,
            "n_pairs": 0,
            "pct_positive": np.nan,
            "interpretation": "sin datos suficientes",
        }
        return pd.DataFrame(), empty_summary

    work = df[cols].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    pivot = (
        work.pivot_table(index=pair_keys, columns=variant_col, values=metric, aggfunc="mean")
        .reset_index()
    )
    if left_value not in pivot.columns or right_value not in pivot.columns:
        empty_summary = {
            "factor": factor,
            "metric": metric,
            "comparison": comparison,
            "mean_delta": np.nan,
            "std_delta": np.nan,
            "n_pairs": 0,
            "pct_positive": np.nan,
            "interpretation": "sin pares comparables",
        }
        return pd.DataFrame(), empty_summary

    paired = pivot.dropna(subset=[left_value, right_value]).copy()
    paired["delta"] = paired[left_value] - paired[right_value]
    paired["factor"] = factor
    paired["metric"] = metric
    paired["comparison"] = comparison
    summary = _summarise_delta(paired["delta"], factor=factor, metric=metric, comparison=comparison)
    return paired, summary


def _summarise_delta(values: pd.Series, factor: str, metric: str, comparison: str) -> Dict[str, object]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return {
            "factor": factor,
            "metric": metric,
            "comparison": comparison,
            "mean_delta": np.nan,
            "std_delta": np.nan,
            "n_pairs": 0,
            "pct_positive": np.nan,
            "interpretation": "sin datos suficientes",
        }
    mean = float(finite.mean())
    std = float(finite.std()) if len(finite) > 1 else 0.0
    pct = float((finite > 0).mean())
    return {
        "factor": factor,
        "metric": metric,
        "comparison": comparison,
        "mean_delta": mean,
        "std_delta": std,
        "n_pairs": int(len(finite)),
        "pct_positive": pct,
        "interpretation": _interpret_delta(mean, std, pct, len(finite)),
    }


def _interpret_delta(mean: float, std: float, pct_positive: float, n_pairs: int) -> str:
    if n_pairs < 3 or not np.isfinite(mean):
        return "evidencia limitada"
    if abs(mean) < 0.03:
        return "efecto medio pequeno"
    if abs(mean) < 0.08 and std > abs(mean):
        return "efecto debil o irregular"
    if mean > 0 and pct_positive >= 0.65:
        return "efecto positivo consistente"
    if mean > 0:
        return "efecto positivo irregular"
    if pct_positive <= 0.35:
        return "efecto negativo consistente"
    return "efecto negativo irregular"


def _ard_paired_delta(df: pd.DataFrame, metric: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    pair_map = {
        "GP_RBF_ARD": "GP_RBF",
        "GP_Matern52_ARD": "GP_Matern52",
    }
    rows: List[pd.DataFrame] = []
    summaries: List[pd.Series] = []
    for ard_model, base_model in pair_map.items():
        work = df[df["model"].astype(str).isin([ard_model, base_model])].copy()
        if work.empty:
            continue
        paired, _ = _paired_delta(
            df=work,
            metric=metric,
            factor="ARD",
            comparison=f"{ard_model} - {base_model}",
            pair_keys=["benchmark", "n_train", "sampler", "noise", "cv_mode"],
            variant_col="model",
            left_value=ard_model,
            right_value=base_model,
        )
        if paired.empty:
            continue
        paired["ard_pair"] = f"{ard_model} - {base_model}"
        rows.append(paired)
        summaries.append(paired["delta"])

    if not rows:
        return pd.DataFrame(), _summarise_delta(pd.Series(dtype=float), "ARD", metric, "ARD - base kernel")

    all_pairs = pd.concat(rows, ignore_index=True)
    summary = _summarise_delta(all_pairs["delta"], "ARD", metric, "ARD - base kernel")
    return all_pairs, summary


def _build_paired_effects(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["relative_incumbent_improvement", "relative_improvement_mae"]
    summaries: List[Dict[str, object]] = []
    detail_rows: List[pd.DataFrame] = []

    for metric in metrics:
        paired, summary = _paired_delta(
            df=df,
            metric=metric,
            factor="sampler",
            comparison="sobol - random",
            pair_keys=["benchmark", "model", "n_train", "noise", "cv_mode"],
            variant_col="sampler",
            left_value="sobol",
            right_value="random",
        )
        summaries.append(summary)
        if not paired.empty:
            detail_rows.append(paired)

        ard_pairs, ard_summary = _ard_paired_delta(df=df, metric=metric)
        summaries.append(ard_summary)
        if not ard_pairs.empty:
            detail_rows.append(ard_pairs)

        for noise_level in ["Gaussian_s0.5", "Gaussian_s1.0"]:
            paired, summary = _paired_delta(
                df=df,
                metric=metric,
                factor="ruido",
                comparison=f"NoNoise - {noise_level}",
                pair_keys=["benchmark", "model", "n_train", "sampler", "cv_mode"],
                variant_col="noise",
                left_value="NoNoise",
                right_value=noise_level,
            )
            summaries.append(summary)
            if not paired.empty:
                detail_rows.append(paired)

    summary_df = pd.DataFrame(summaries)
    summary_df["metric_label"] = summary_df["metric"].map(
        {
            "relative_incumbent_improvement": "incumbent",
            "relative_improvement_mae": "MAE",
        }
    )
    summary_df = summary_df[
        [
            "factor",
            "metric",
            "metric_label",
            "comparison",
            "mean_delta",
            "std_delta",
            "n_pairs",
            "pct_positive",
            "interpretation",
        ]
    ]
    detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    return summary_df, detail_df


def _build_hierarchy_table(
    paired_effects: pd.DataFrame,
    incumbent_summary: pd.DataFrame,
    by_ntrain_summary: pd.DataFrame,
) -> pd.DataFrame:
    def effect_text(factor: str, metric: str, comparison: Optional[str] = None) -> str:
        block = paired_effects[
            (paired_effects["factor"] == factor)
            & (paired_effects["metric"] == metric)
        ].copy()
        if comparison is not None:
            block = block[block["comparison"] == comparison]
        if block.empty:
            return "sin evidencia emparejada suficiente"
        row = block.iloc[0]
        if pd.isna(row["mean_delta"]):
            return "sin evidencia emparejada suficiente"
        return f"delta={row['mean_delta']:.3f}, pct+={row['pct_positive']:.0%}, n={int(row['n_pairs'])}"

    bench_evidence = "diferencias claras en incumbent entre benchmarks"
    if not incumbent_summary.empty and "mean_relative_incumbent_improvement" in incumbent_summary.columns:
        best_by_benchmark = incumbent_summary.groupby("benchmark")[
            "mean_relative_incumbent_improvement"
        ].max()
        if len(best_by_benchmark) > 1:
            bench_evidence = (
                f"mejor incumbent por benchmark entre {best_by_benchmark.min():.3f} "
                f"y {best_by_benchmark.max():.3f}"
            )

    ntrain_evidence = "efecto visible en Forrester/Hartmann6; requiere cautela por margen inicial"
    if not by_ntrain_summary.empty and {"benchmark", "n_train", "mean_relative_incumbent_improvement"}.issubset(
        by_ntrain_summary.columns
    ):
        ranges = (
            by_ntrain_summary.groupby(["benchmark", "n_train"])["mean_relative_incumbent_improvement"]
            .mean()
            .reset_index()
            .groupby("benchmark")["mean_relative_incumbent_improvement"]
            .agg(lambda s: float(s.max() - s.min()) if len(s) > 1 else 0.0)
        )
        if not ranges.empty:
            top = ranges.sort_values(ascending=False).head(2)
            ntrain_evidence = ", ".join(f"{idx}: rango {val:.3f}" for idx, val in top.items())

    kernel_evidence = "no hay ganador universal"
    if not incumbent_summary.empty and {"benchmark", "model", "mean_relative_incumbent_improvement"}.issubset(
        incumbent_summary.columns
    ):
        winners = (
            incumbent_summary.sort_values(
                ["benchmark", "mean_relative_incumbent_improvement", "model"],
                ascending=[True, False, True],
            )
            .groupby("benchmark", as_index=False)
            .first()
        )
        kernel_evidence = "; ".join(f"{r.benchmark}: {r.model}" for r in winners.itertuples(index=False))

    rows = [
        {
            "rank": 1,
            "factor": "benchmark/dimensionalidad",
            "evidence": bench_evidence,
            "interpretation": "factor dominante; condiciona dificultad y margen de mejora",
        },
        {
            "rank": 2,
            "factor": "n_train inicial",
            "evidence": ntrain_evidence,
            "interpretation": "muy relevante; interpretar junto al valor/gap final, no solo mejora relativa",
        },
        {
            "rank": 3,
            "factor": "kernel",
            "evidence": kernel_evidence,
            "interpretation": "importa, pero depende del benchmark; no hay ganador universal",
        },
        {
            "rank": 4,
            "factor": "ARD",
            "evidence": effect_text("ARD", "relative_incumbent_improvement"),
            "interpretation": "ventaja no universal; conviene mantener comparacion por pares",
        },
        {
            "rank": 5,
            "factor": "sampler",
            "evidence": effect_text("sampler", "relative_incumbent_improvement"),
            "interpretation": "efecto secundario frente a benchmark, n_train y kernel",
        },
        {
            "rank": 6,
            "factor": "ruido",
            "evidence": (
                f"G0.5: {effect_text('ruido', 'relative_incumbent_improvement', 'NoNoise - Gaussian_s0.5')}; "
                f"G1.0: {effect_text('ruido', 'relative_incumbent_improvement', 'NoNoise - Gaussian_s1.0')}"
            ),
            "interpretation": "analizar emparejado; afecta estabilidad y puede cambiar por benchmark",
        },
    ]
    return pd.DataFrame(rows)


def _plot_factor_effects(summary: pd.DataFrame, out_path: Path, dpi: int, save_svg: bool) -> None:
    if summary.empty:
        return

    metric_order = [
        ("relative_incumbent_improvement", "Mejora del incumbent"),
        ("relative_improvement_mae", "Mejora relativa del MAE"),
    ]
    order = [
        ("sampler", "sobol - random"),
        ("ARD", "ARD - base kernel"),
        ("ruido", "NoNoise - Gaussian_s0.5"),
        ("ruido", "NoNoise - Gaussian_s1.0"),
    ]
    labels = {
        ("sampler", "sobol - random"): "Sampler: Sobol - Random",
        ("ARD", "ARD - base kernel"): "ARD - kernel base",
        ("ruido", "NoNoise - Gaussian_s0.5"): "NoNoise - Gaussian 0.5",
        ("ruido", "NoNoise - Gaussian_s1.0"): "NoNoise - Gaussian 1.0",
    }

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12.4, 4.8), sharey=True)
    axs = np.atleast_1d(axs).ravel()
    for ax, (metric, title) in zip(axs, metric_order):
        rows = []
        for factor, comparison in order:
            block = summary[
                (summary["metric"] == metric)
                & (summary["factor"] == factor)
                & (summary["comparison"] == comparison)
            ]
            if block.empty:
                rows.append(None)
            else:
                rows.append(block.iloc[0])

        y = np.arange(len(order))
        means = np.asarray([np.nan if row is None else row["mean_delta"] for row in rows], dtype=float)
        stds = np.asarray([0.0 if row is None or pd.isna(row["std_delta"]) else row["std_delta"] for row in rows], dtype=float)
        colors = ["#2a9d8f" if np.isfinite(v) and v >= 0 else "#c44536" for v in means]
        ax.barh(y, means, xerr=stds, color=colors, alpha=0.82, edgecolor="black", linewidth=0.6, capsize=3)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[item] for item in order])
        ax.set_title(title)
        ax.set_xlabel("Delta medio emparejado")
        ax.grid(True, axis="x", alpha=0.25)
        for yy, row, mean in zip(y, rows, means):
            if row is None or not np.isfinite(mean):
                continue
            ax.text(
                mean,
                yy + 0.28,
                f"n={int(row['n_pairs'])}, +={row['pct_positive']:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.78},
            )

    axs[0].invert_yaxis()
    fig.suptitle("Efectos emparejados de condiciones experimentales")
    fig.text(
        0.5,
        0.91,
        "Delta positivo = la primera condicion de la comparacion obtiene mayor mejora",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.15, top=0.82, wspace=0.12)
    fig._skip_tight_layout = True
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def generate_factor_effect_outputs(
    master_df: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    metadata: Optional[Dict[str, object]] = None,
    dpi: int = 300,
    save_svg: bool = False,
) -> Dict[str, pd.DataFrame]:
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_metrics = _combine_run_metrics(master_df=master_df, metadata=metadata)
    paired_summary, _paired_detail = _build_paired_effects(run_metrics)

    incumbent_summary_path = tables_dir / "summary_incumbent_relative_improvement_by_benchmark_model.csv"
    ntrain_summary_path = tables_dir / "summary_incumbent_initial_final_by_benchmark_ntrain_model.csv"
    incumbent_summary = pd.read_csv(incumbent_summary_path) if incumbent_summary_path.exists() else pd.DataFrame()
    by_ntrain_summary = pd.read_csv(ntrain_summary_path) if ntrain_summary_path.exists() else pd.DataFrame()
    hierarchy = _build_hierarchy_table(
        paired_effects=paired_summary,
        incumbent_summary=incumbent_summary,
        by_ntrain_summary=by_ntrain_summary,
    )

    hierarchy_path = tables_dir / "summary_factor_hierarchy_for_tfg.csv"
    paired_path = tables_dir / "summary_factor_effects_paired_for_tfg.csv"
    hierarchy.to_csv(hierarchy_path, index=False)
    paired_summary.to_csv(paired_path, index=False)

    _plot_factor_effects(
        summary=paired_summary,
        out_path=figures_dir / "factor_effects_paired_delta",
        dpi=dpi,
        save_svg=save_svg,
    )

    print("Factor effects outputs")
    print(f"- Hierarchy rows: {len(hierarchy)}")
    print(f"- Paired summary rows: {len(paired_summary)}")
    print(f"- Table: {hierarchy_path}")
    print(f"- Table: {paired_path}")
    print(f"- Figure: {figures_dir / 'factor_effects_paired_delta.png'}")

    return {
        "summary_factor_hierarchy_for_tfg": hierarchy,
        "summary_factor_effects_paired_for_tfg": paired_summary,
    }
