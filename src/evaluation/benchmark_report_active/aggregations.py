from __future__ import annotations

import ast
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import ACTIVE_MASTER_COLUMNS, FINAL_SCENARIO_KEYS


METRIC_DIRECTIONS: Dict[str, str] = {
    "mae": "min",
    "rmse": "min",
    "nlpd": "min",
    "r2": "max",
    "coverage_95": "target",
}


def _safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _parse_list_cell(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return list(parsed)
        except Exception:
            return [v]
    return [v]


def build_master_active_table(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    if trajectory_df.empty:
        return pd.DataFrame(columns=ACTIVE_MASTER_COLUMNS)

    df = trajectory_df.copy()
    rename_map = {
        "mae_test": "mae",
        "rmse_test": "rmse",
        "r2_test": "r2",
        "nlpd_test": "nlpd",
        "coverage_95_test": "coverage_95",
        "fit_time_step": "fit_time",
        "predict_time_step": "predict_time",
    }
    df = df.rename(columns=rename_map)
    if "cv_mode" in df.columns:
        df = df[df["cv_mode"].astype(str).str.lower() == "active"].copy()

    _safe_numeric(
        df,
        [
            "n_train",
            "step",
            "n_train_current",
            "mae",
            "rmse",
            "r2",
            "nlpd",
            "coverage_95",
            "fit_time",
            "predict_time",
            "incumbent_best",
            "ei_next",
            "y_next",
        ],
    )

    if "x_next" in df.columns:
        df["x_next"] = df["x_next"].map(_parse_list_cell)
    df["total_time"] = df.get("fit_time", 0.0).fillna(0.0) + df.get("predict_time", 0.0).fillna(0.0)

    for col in ACTIVE_MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df[ACTIVE_MASTER_COLUMNS].copy()


def build_final_table(master_df: pd.DataFrame, summary_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if master_df.empty:
        return pd.DataFrame()
    final_from_traj = (
        master_df.sort_values("step")
        .groupby(FINAL_SCENARIO_KEYS, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    if summary_df is None or summary_df.empty:
        return final_from_traj

    sum_active = summary_df.copy()
    if "cv_mode" in sum_active.columns:
        sum_active = sum_active[sum_active["cv_mode"].astype(str).str.lower() == "active"].copy()
    if sum_active.empty:
        return final_from_traj

    keep_cols = ["sampler", "n_train", "benchmark", "noise", "cv_mode", "model", "mae", "rmse", "r2", "nlpd", "coverage_95", "fit_time"]
    for c in keep_cols:
        if c not in sum_active.columns:
            sum_active[c] = np.nan
    sum_active = sum_active[keep_cols].copy()
    sum_active["predict_time"] = np.nan
    sum_active["total_time"] = sum_active["fit_time"]
    sum_active["step"] = np.nan
    sum_active["n_train_current"] = sum_active["n_train"]
    sum_active["incumbent_best"] = np.nan
    sum_active["ei_next"] = np.nan
    sum_active["x_next"] = [[] for _ in range(len(sum_active))]
    sum_active["y_next"] = np.nan

    merged = pd.concat([final_from_traj, sum_active[final_from_traj.columns]], ignore_index=True)
    merged = (
        merged.sort_values(["sampler", "n_train", "benchmark", "noise", "cv_mode", "model", "step"], na_position="last")
        .groupby(FINAL_SCENARIO_KEYS, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return merged


def _best_index(series: pd.Series, metric: str) -> int:
    if metric == "coverage_95":
        return int((series - 0.95).abs().idxmin())
    direction = METRIC_DIRECTIONS.get(metric, "min")
    if direction == "max":
        return int(series.idxmax())
    return int(series.idxmin())


def compute_leaderboard(final_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if final_df.empty or metric not in final_df.columns:
        return pd.DataFrame()
    g = final_df.groupby("model", as_index=False)[metric].agg(["mean", "std", "count"]).reset_index()
    g = g.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": "n"})
    asc = METRIC_DIRECTIONS.get(metric, "min") != "max"
    g = g.sort_values(f"{metric}_mean", ascending=asc).reset_index(drop=True)
    g["rank"] = np.arange(1, len(g) + 1)
    return g


def compute_wins_summary(final_df: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    if final_df.empty or metric not in final_df.columns:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    scenario_cols = ["benchmark", "noise", "sampler", "n_train", "cv_mode"]
    for _, block in final_df.groupby(scenario_cols, dropna=False):
        block = block.dropna(subset=[metric])
        if block.empty:
            continue
        idx = _best_index(block[metric], metric)
        winner = block.loc[idx, "model"]
        rows.append({"winner": winner})
    if not rows:
        return pd.DataFrame()
    s = pd.DataFrame(rows)["winner"].value_counts().rename_axis("model").reset_index(name="wins")
    s["win_rate"] = s["wins"] / s["wins"].sum()
    return s


def compute_top1_table(final_df: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    if final_df.empty or metric not in final_df.columns:
        return pd.DataFrame()
    out_rows: List[Dict[str, object]] = []
    for (benchmark, noise), block in final_df.groupby(["benchmark", "noise"], dropna=False):
        b = block.dropna(subset=[metric]).sort_values(metric, ascending=True)
        if b.empty:
            continue
        best = b.iloc[0]
        second = b.iloc[1] if len(b) > 1 else None
        row = {
            "benchmark": benchmark,
            "noise": noise,
            "best_model": best["model"],
            f"best_{metric}": best[metric],
            "second_model": second["model"] if second is not None else None,
            f"second_{metric}": second[metric] if second is not None else np.nan,
        }
        if second is not None and pd.notna(second[metric]) and second[metric] != 0:
            gap = float(second[metric] - best[metric])
            row["gap"] = gap
            row["gap_pct"] = gap / abs(float(second[metric])) * 100.0
        else:
            row["gap"] = np.nan
            row["gap_pct"] = np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def compute_time_performance(final_df: pd.DataFrame) -> pd.DataFrame:
    if final_df.empty:
        return pd.DataFrame()
    g = (
        final_df.groupby("model", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            fit_time_mean=("fit_time", "mean"),
            predict_time_mean=("predict_time", "mean"),
            total_time_mean=("total_time", "mean"),
            n=("mae", "count"),
        )
        .sort_values(["mae_mean", "total_time_mean"], ascending=[True, True])
    )
    return g


def compute_robustness_by_noise(final_df: pd.DataFrame) -> pd.DataFrame:
    if final_df.empty:
        return pd.DataFrame()
    g = (
        final_df.groupby(["benchmark", "model"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_iqr=("mae", lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
            n=("mae", "count"),
        )
        .sort_values(["benchmark", "mae_mean"], ascending=[True, True])
    )
    return g


def compute_sampler_effects(master_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        return pd.DataFrame()
    gkeys = ["benchmark", "noise", "model", "sampler"]
    grouped = (
        master_df.sort_values("n_train_current")
        .groupby(gkeys, as_index=False)
        .agg(
            mae_initial=("mae", "first"),
            mae_final=("mae", "last"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n_points=("mae", "count"),
        )
    )
    grouped["mae_delta"] = grouped["mae_final"] - grouped["mae_initial"]

    # Pair sobol/random on same benchmark-noise-model where possible.
    piv = grouped.pivot_table(
        index=["benchmark", "noise", "model"],
        columns="sampler",
        values=["mae_initial", "mae_final", "mae_delta"],
        aggfunc="mean",
    )
    if piv.empty:
        return grouped
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    if "mae_final_sobol" in piv.columns and "mae_final_random" in piv.columns:
        piv["final_gap_random_minus_sobol"] = piv["mae_final_random"] - piv["mae_final_sobol"]
    return piv


def summarize_active_coverage(master_df: pd.DataFrame) -> Dict[str, object]:
    if master_df.empty:
        return {
            "n_rows": 0,
            "n_models": 0,
            "n_benchmarks": 0,
            "n_noises": 0,
            "n_samplers": 0,
            "single_model_mode": True,
        }
    return {
        "n_rows": int(len(master_df)),
        "n_models": int(master_df["model"].nunique()),
        "n_benchmarks": int(master_df["benchmark"].nunique()),
        "n_noises": int(master_df["noise"].nunique()),
        "n_samplers": int(master_df["sampler"].nunique()),
        "single_model_mode": bool(master_df["model"].nunique() <= 1),
    }
