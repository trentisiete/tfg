from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .aggregations import (
    compute_leaderboard,
    compute_robustness_by_noise,
    compute_sampler_effects,
    compute_time_performance,
    compute_top1_table,
    compute_wins_summary,
)


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def generate_phase1_tables(
    master_df: pd.DataFrame,
    final_df: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, pd.DataFrame] = {}

    outputs["master_active_table"] = master_df.copy()
    outputs["final_metrics_active"] = final_df.copy()
    outputs["leaderboard_global_mae"] = compute_leaderboard(final_df, "mae")
    outputs["leaderboard_global_rmse"] = compute_leaderboard(final_df, "rmse")
    outputs["wins_summary_mae"] = compute_wins_summary(final_df, metric="mae")
    outputs["top1_model_mae"] = compute_top1_table(final_df, metric="mae")
    outputs["time_performance"] = compute_time_performance(final_df)
    outputs["robustness_by_noise_mae"] = compute_robustness_by_noise(final_df)
    outputs["sampler_effects_summary"] = compute_sampler_effects(master_df)

    if not final_df.empty:
        pivot = final_df.pivot_table(
            index="benchmark",
            columns="model",
            values="mae",
            aggfunc="mean",
        ).reset_index()
        outputs["pivot_benchmark_model_mae"] = pivot

    if not final_df.empty:
        metrics_summary = (
            final_df.groupby(["benchmark", "noise", "model"], as_index=False)
            .agg(
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                r2=("r2", "mean"),
                nlpd=("nlpd", "mean"),
                coverage_95=("coverage_95", "mean"),
                fit_time=("fit_time", "mean"),
                predict_time=("predict_time", "mean"),
            )
            .sort_values(["benchmark", "noise", "mae"])
        )
        outputs["metrics_summary"] = metrics_summary

    for name, df in outputs.items():
        _save(df, out_dir / f"{name}.csv")

    return outputs
