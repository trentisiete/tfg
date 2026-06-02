from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.surrogate_metrics import compute_surrogate_metrics
from src.benchmarks.dataset_generator import generate_benchmark_dataset
from src.models.dummy import DummySurrogateRegressor

from .styling import save_figure


CONFIG_KEYS = ["benchmark", "sampler", "n_train", "noise", "cv_mode"]
TRAJECTORY_KEYS = CONFIG_KEYS + ["model"]


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


def _normalise_active_metrics(master_df: pd.DataFrame) -> pd.DataFrame:
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
    for col in ["n_train", "step", "mae", "rmse", "r2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_dummy_baseline_by_config(
    configs: pd.DataFrame,
    metadata: Optional[Dict[str, object]],
) -> pd.DataFrame:
    if configs.empty:
        return pd.DataFrame()

    seed = int((metadata or {}).get("seed", 42) or 42)
    n_test = int((metadata or {}).get("n_test", 200) or 200)
    rows: List[Dict[str, object]] = []

    for cfg in configs.sort_values(CONFIG_KEYS).itertuples(index=False):
        benchmark = str(getattr(cfg, "benchmark"))
        sampler = str(getattr(cfg, "sampler"))
        n_train = int(getattr(cfg, "n_train"))
        noise = str(getattr(cfg, "noise"))
        cv_mode = str(getattr(cfg, "cv_mode"))
        noise_cfg = _noise_label_to_cfg(noise)

        dataset = generate_benchmark_dataset(
            benchmark=benchmark,
            n_train=n_train,
            n_test=n_test,
            sampler=sampler,
            noise=str(noise_cfg["type"]),
            noise_kwargs=dict(noise_cfg["kwargs"]),
            n_groups=None,
            seed=seed,
        )
        model = DummySurrogateRegressor(strategy="mean")
        model.fit(dataset.X_train, dataset.y_train)
        y_pred = model.predict(dataset.X_test)
        metrics = compute_surrogate_metrics(
            y_true=dataset.y_test_clean,
            y_pred=y_pred,
            std_pred=None,
        )

        rows.append(
            {
                "benchmark": benchmark,
                "sampler": sampler,
                "n_train": n_train,
                "noise": noise,
                "cv_mode": cv_mode,
                "model": "Dummy",
                "step": 0,
                "n_train_current": n_train,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "r2": metrics.r2,
                "n_test": metrics.n_samples,
            }
        )

    return pd.DataFrame(rows)


def _build_gp_run_table(active_df: pd.DataFrame) -> pd.DataFrame:
    if active_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for keys, block in active_df.sort_values("step").groupby(TRAJECTORY_KEYS, dropna=False):
        first = block.iloc[0]
        last = block.iloc[-1]
        row = {col: value for col, value in zip(TRAJECTORY_KEYS, keys)}
        row.update(
            {
                "gp_initial_step": int(first["step"]),
                "gp_final_step": int(last["step"]),
                "gp_initial_mae": float(first["mae"]),
                "gp_final_mae": float(last["mae"]),
                "gp_initial_rmse": float(first["rmse"]),
                "gp_final_rmse": float(last["rmse"]),
                "gp_initial_r2": float(first["r2"]),
                "gp_final_r2": float(last["r2"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_gp_vs_dummy_summary(gp_runs: pd.DataFrame, dummy_df: pd.DataFrame) -> pd.DataFrame:
    if gp_runs.empty or dummy_df.empty:
        return pd.DataFrame()

    dummy_cols = CONFIG_KEYS + ["mae", "rmse", "r2"]
    merged = gp_runs.merge(
        dummy_df[dummy_cols].rename(
            columns={
                "mae": "dummy_mae",
                "rmse": "dummy_rmse",
                "r2": "dummy_r2",
            }
        ),
        on=CONFIG_KEYS,
        how="left",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["relative_gain_gp_final_vs_dummy_mae"] = (
        (merged["dummy_mae"] - merged["gp_final_mae"])
        / merged["dummy_mae"].replace(0.0, np.nan).abs()
    )
    merged["relative_gain_gp_initial_vs_dummy_mae"] = (
        (merged["dummy_mae"] - merged["gp_initial_mae"])
        / merged["dummy_mae"].replace(0.0, np.nan).abs()
    )
    merged["gp_final_better_than_dummy_mae"] = merged["gp_final_mae"] < merged["dummy_mae"]
    merged["gp_initial_better_than_dummy_mae"] = merged["gp_initial_mae"] < merged["dummy_mae"]
    merged["r2_gain_gp_final_vs_dummy"] = merged["gp_final_r2"] - merged["dummy_r2"]

    summary = (
        merged.groupby(["benchmark", "model"], as_index=False)
        .agg(
            dummy_mae_mean=("dummy_mae", "mean"),
            gp_initial_mae_mean=("gp_initial_mae", "mean"),
            gp_final_mae_mean=("gp_final_mae", "mean"),
            mean_relative_gain_gp_initial_vs_dummy_mae=("relative_gain_gp_initial_vs_dummy_mae", "mean"),
            mean_relative_gain_gp_final_vs_dummy_mae=("relative_gain_gp_final_vs_dummy_mae", "mean"),
            std_relative_gain_gp_final_vs_dummy_mae=("relative_gain_gp_final_vs_dummy_mae", "std"),
            pct_gp_initial_better_than_dummy_mae=("gp_initial_better_than_dummy_mae", "mean"),
            pct_gp_final_better_than_dummy_mae=("gp_final_better_than_dummy_mae", "mean"),
            mean_r2_gain_gp_final_vs_dummy=("r2_gain_gp_final_vs_dummy", "mean"),
            n_runs=("relative_gain_gp_final_vs_dummy_mae", "count"),
        )
        .sort_values(["benchmark", "mean_relative_gain_gp_final_vs_dummy_mae"], ascending=[True, False])
    )
    summary["comentario_dummy"] = [
        _classify_vs_dummy(mean_gain, pct_better)
        for mean_gain, pct_better in zip(
            summary["mean_relative_gain_gp_final_vs_dummy_mae"],
            summary["pct_gp_final_better_than_dummy_mae"],
        )
    ]
    return summary


def _classify_vs_dummy(mean_gain: float, pct_better: float) -> str:
    if pd.isna(mean_gain):
        return "sin datos"
    if mean_gain > 0.10 and pct_better >= 0.70:
        return "GP claramente mejor que Dummy"
    if mean_gain > 0 and pct_better >= 0.50:
        return "GP mejor que Dummy, efecto moderado"
    if mean_gain > 0:
        return "GP mejor en media, no consistente"
    return "Dummy igual o mejor en media"


def _plot_gp_vs_dummy_bars(summary: pd.DataFrame, out_path: Path, dpi: int, save_svg: bool) -> None:
    if summary.empty:
        return

    benchmarks = sorted(summary["benchmark"].dropna().astype(str).unique().tolist())
    if not benchmarks:
        return

    ncols = 2
    nrows = int(np.ceil(len(benchmarks) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13.0, 4.2 * nrows))
    axs = np.atleast_1d(axs).ravel()

    value_col = "mean_relative_gain_gp_final_vs_dummy_mae"
    std_col = "std_relative_gain_gp_final_vs_dummy_mae"
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

        x = np.arange(len(block))
        y = block[value_col].to_numpy(dtype=float)
        yerr = block[std_col].fillna(0.0).to_numpy(dtype=float)
        colors = ["#2a9d8f" if val > 0 else "#c44536" for val in y]
        ax.bar(x, y, yerr=yerr, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(block["model"].astype(str).tolist(), rotation=35, ha="right")
        ax.set_title(str(benchmark))
        ax.set_ylabel("Mejora relativa MAE final vs Dummy")

    for k in range(len(benchmarks), len(axs)):
        axs[k].set_axis_off()

    fig.suptitle("Comparacion del GP final frente a Dummy")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.90, hspace=0.58, wspace=0.28)
    fig._skip_tight_layout = True
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def generate_dummy_baseline_outputs(
    master_df: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    metadata: Optional[Dict[str, object]] = None,
    dpi: int = 300,
    save_svg: bool = False,
) -> Dict[str, pd.DataFrame]:
    active_df = _normalise_active_metrics(master_df)
    required = set(CONFIG_KEYS + ["model", "step", "mae", "rmse", "r2"])
    if active_df.empty or not required.issubset(active_df.columns):
        return {}

    configs = active_df[CONFIG_KEYS].drop_duplicates().copy()
    dummy_df = _build_dummy_baseline_by_config(configs=configs, metadata=metadata)
    gp_runs = _build_gp_run_table(active_df=active_df)
    summary = _build_gp_vs_dummy_summary(gp_runs=gp_runs, dummy_df=dummy_df)

    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "dummy_baseline_by_config": dummy_df,
        "summary_mae_gp_vs_dummy_by_benchmark_model": summary,
    }
    dummy_df.to_csv(tables_dir / "dummy_baseline_by_config.csv", index=False)
    summary.to_csv(tables_dir / "summary_mae_gp_vs_dummy_by_benchmark_model.csv", index=False)
    _plot_gp_vs_dummy_bars(
        summary=summary,
        out_path=figures_dir / "final_relative_gain_mae_gp_vs_dummy_by_benchmark",
        dpi=dpi,
        save_svg=save_svg,
    )
    return outputs
