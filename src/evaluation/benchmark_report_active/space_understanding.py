from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.analysis.active_learning import _manual_ei_fallback, ei_values_from_model
from src.benchmarks import generate_benchmark_dataset, get_benchmark

from .plots_gp import _build_gp_model, _noise_label_to_cfg, _parse_x_next_cell
from .styling import save_figure


def _select_forrester_block(
    trajectory_df: pd.DataFrame,
    model: str,
    n_train: int,
    sampler: str,
    noise: str,
) -> pd.DataFrame:
    block = trajectory_df[
        (trajectory_df["benchmark"].astype(str).str.lower() == "forrester")
        & (trajectory_df["cv_mode"].astype(str).str.lower() == "active")
        & (trajectory_df["model"].astype(str) == model)
        & (pd.to_numeric(trajectory_df["n_train"], errors="coerce") == int(n_train))
        & (trajectory_df["sampler"].astype(str) == sampler)
        & (trajectory_df["noise"].astype(str) == noise)
    ].copy()
    if block.empty:
        return pd.DataFrame()
    block["step"] = pd.to_numeric(block["step"], errors="coerce").astype(int)
    return block.sort_values("step").reset_index(drop=True)


def _reconstruct_training_before_step(
    block: pd.DataFrame,
    target_step: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    first = block.iloc[0]
    noise_cfg = _noise_label_to_cfg(str(first["noise"]))
    dataset = generate_benchmark_dataset(
        benchmark="forrester",
        n_train=int(first["n_train"]),
        n_test=200,
        sampler=str(first["sampler"]),
        noise=str(noise_cfg["type"]),
        noise_kwargs=dict(noise_cfg["kwargs"]),
        n_groups=None,
        seed=int(seed),
    )
    bench = get_benchmark("forrester")

    X_initial = np.asarray(dataset.X_train, dtype=float)
    y_initial = np.asarray(dataset.y_train, dtype=float).ravel()
    X_train = X_initial.copy()
    y_train = y_initial.copy()

    previous = block[(block["step"] > 0) & (block["step"] < int(target_step))]
    for row in previous.itertuples(index=False):
        x = _parse_x_next_cell(getattr(row, "x_next"))
        y = getattr(row, "y_next")
        if x.size != 1 or pd.isna(y):
            continue
        X_train = np.vstack([X_train, x.reshape(1, 1)])
        y_train = np.concatenate([y_train, [float(y)]])

    return X_initial, y_initial, X_train, y_train, bench


def _expected_improvement_curve(model, x_grid: np.ndarray, y_best: float, xi: float) -> np.ndarray:
    try:
        ei = ei_values_from_model(model, x_grid, y_best=float(y_best), xi=float(xi))
    except Exception:
        ei = _manual_ei_fallback(model, x_grid, y_best=float(y_best), xi=float(xi))
    return np.asarray(ei, dtype=float).ravel()


def generate_forrester_ei_decision_synthesis(
    trajectory_df: pd.DataFrame,
    output_dir: Path,
    *,
    model_name: str = "GP_Matern52",
    n_train: int = 4,
    sampler: str = "sobol",
    noise: str = "NoNoise",
    target_step: int = 5,
    seed: int = 42,
    xi: float = 0.01,
    dpi: int = 300,
    save_svg: bool = False,
) -> Dict[str, Path]:
    """Create a compact explanatory figure for one Forrester EI decision."""
    block = _select_forrester_block(
        trajectory_df=trajectory_df,
        model=model_name,
        n_train=n_train,
        sampler=sampler,
        noise=noise,
    )
    if block.empty:
        return {}

    target = block[block["step"].astype(int) == int(target_step)]
    if target.empty:
        return {}
    target_row = target.iloc[0]
    x_next = _parse_x_next_cell(target_row["x_next"])
    if x_next.size != 1:
        return {}
    x_next_value = float(x_next[0])
    y_next_value = float(target_row["y_next"])
    ei_next_value = float(target_row["ei_next"])

    X_initial, y_initial, X_train, y_train, bench = _reconstruct_training_before_step(
        block=block,
        target_step=target_step,
        seed=seed,
    )
    x_grid = np.linspace(bench.bounds[0][0], bench.bounds[0][1], 900).reshape(-1, 1)
    y_true = np.asarray(bench(x_grid), dtype=float).ravel()

    np.random.seed(seed)
    model = _build_gp_model(model_name)
    model.fit(X_train, y_train)
    y_mean, y_std = model.predict_dist(x_grid)
    y_mean = np.asarray(y_mean, dtype=float).ravel()
    y_std = np.asarray(y_std, dtype=float).ravel()
    y_best_before = float(np.min(y_train))
    x_best_before = float(X_train[int(np.argmin(y_train)), 0])
    y_best_after = min(y_best_before, y_next_value)
    opt_x = float(np.asarray(bench.optimal_location).ravel()[0])
    opt_y = float(bench.optimal_value)

    ei = _expected_improvement_curve(model, x_grid, y_best=y_best_before, xi=xi)
    x_flat = x_grid.ravel()
    mu_next, std_next = model.predict_dist(np.asarray([[x_next_value]], dtype=float))
    mu_next_value = float(np.asarray(mu_next).ravel()[0])
    std_next_value = float(np.asarray(std_next).ravel()[0])

    previous_infill = X_train[len(X_initial) :]
    previous_y = y_train[len(y_initial) :]
    gap_before = max(0.0, y_best_before - opt_y)
    gap_after = max(0.0, y_best_after - opt_y)
    gap_closed = (gap_before - gap_after) / gap_before if gap_before > 1e-12 else np.nan

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / "forrester_ei_decision_synthesis_step5"

    fig = plt.figure(figsize=(12.8, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.34, wspace=0.25)
    ax_surrogate = fig.add_subplot(grid[0, :])
    ax_ei = fig.add_subplot(grid[1, 0])
    ax_gap = fig.add_subplot(grid[1, 1])

    band = ax_surrogate.fill_between(
        x_flat,
        y_mean - 1.96 * y_std,
        y_mean + 1.96 * y_std,
        color="#f4a261",
        alpha=0.22,
        linewidth=0,
        label="incertidumbre 95%",
    )
    true_line = ax_surrogate.plot(x_flat, y_true, color="#264653", linewidth=2.0, label="funcion real")[0]
    mean_line = ax_surrogate.plot(x_flat, y_mean, color="#e76f51", linewidth=1.9, label="media GP")[0]
    init_scatter = ax_surrogate.scatter(
        X_initial[:, 0],
        y_initial,
        color="#111111",
        s=38,
        marker="o",
        zorder=4,
        label="diseno inicial",
    )
    infill_scatter = None
    if len(previous_infill):
        infill_scatter = ax_surrogate.scatter(
            previous_infill[:, 0],
            previous_y,
            color="#6c757d",
            s=42,
            marker="s",
            zorder=4,
            label="infill previo",
        )
    incumbent_scatter = ax_surrogate.scatter(
        [x_best_before],
        [y_best_before],
        color="#2a9d8f",
        marker="D",
        s=72,
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
        label="incumbent antes",
    )
    next_scatter = ax_surrogate.scatter(
        [x_next_value],
        [y_next_value],
        color="#8e44ad",
        marker="*",
        s=190,
        edgecolor="black",
        linewidth=0.8,
        zorder=6,
        label="x_next",
    )
    opt_scatter = ax_surrogate.scatter(
        [opt_x],
        [opt_y],
        color="#f2c94c",
        marker="X",
        s=120,
        edgecolor="black",
        linewidth=0.8,
        zorder=6,
        label="optimo",
    )
    ax_surrogate.axhline(y_best_before, color="#2a9d8f", linestyle=":", linewidth=1.15)
    ax_surrogate.axvline(x_next_value, color="#8e44ad", linestyle="--", linewidth=1.1)
    ax_surrogate.set_xlim(bench.bounds[0][0], bench.bounds[0][1])
    ax_surrogate.annotate(
        f"x_next={x_next_value:.3f}\nf(x_next)={y_next_value:.3f}",
        xy=(x_next_value, y_next_value),
        xytext=(20, 22),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#8e44ad", "linewidth": 1.0},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#8e44ad", "alpha": 0.92},
    )
    ax_surrogate.set_ylabel("f(x)")
    ax_surrogate.set_title("Surrogate antes de decidir el siguiente punto")
    ax_surrogate.grid(True, alpha=0.24)

    handles = [true_line, mean_line, band, init_scatter]
    labels = ["funcion real", "media GP", "incertidumbre 95%", "diseno inicial"]
    if infill_scatter is not None:
        handles.append(infill_scatter)
        labels.append("infill previo")
    handles.extend([incumbent_scatter, next_scatter, opt_scatter])
    labels.extend(["incumbent antes", "x_next", "optimo"])
    ax_surrogate.legend(handles, labels, loc="upper left", ncol=4, frameon=True, fontsize=8.2)

    ax_ei.fill_between(x_flat, 0.0, ei, color="#8e44ad", alpha=0.22, linewidth=0)
    ax_ei.plot(x_flat, ei, color="#8e44ad", linewidth=2.0)
    ax_ei.axvline(x_next_value, color="#8e44ad", linestyle="--", linewidth=1.1)
    for x_obs in X_train[:, 0]:
        ax_ei.axvline(float(x_obs), color="#999999", alpha=0.22, linewidth=0.7)
    ax_ei.scatter([x_next_value], [ei_next_value], color="#8e44ad", marker="*", s=145, edgecolor="black", zorder=5)
    ax_ei.annotate(
        f"EI={ei_next_value:.3f}",
        xy=(x_next_value, ei_next_value),
        xytext=(-78, -34),
        textcoords="offset points",
        fontsize=9,
        ha="right",
        va="top",
        arrowprops={"arrowstyle": "->", "color": "#8e44ad", "linewidth": 1.0},
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#8e44ad", "alpha": 0.92},
    )
    ax_ei.set_title("Criterio Expected Improvement")
    ax_ei.set_xlabel("x")
    ax_ei.set_ylabel("EI(x)")
    ax_ei.set_xlim(bench.bounds[0][0], bench.bounds[0][1])
    ax_ei.grid(True, alpha=0.24)

    gap_labels = ["antes", "despues"]
    gap_values = [gap_before, gap_after]
    colors = ["#adb5bd", "#2a9d8f"]
    y_pos = np.arange(len(gap_labels))
    ax_gap.barh(y_pos, gap_values, color=colors, edgecolor="black", linewidth=0.65)
    ax_gap.set_yticks(y_pos)
    ax_gap.set_yticklabels(gap_labels)
    ax_gap.invert_yaxis()
    ax_gap.set_xlabel("gap al optimo")
    ax_gap.set_title("Efecto sobre el incumbent")
    ax_gap.grid(True, axis="x", alpha=0.24)
    gap_axis_max = max(gap_before * 1.22, 0.2)
    ax_gap.set_xlim(0.0, gap_axis_max)
    ax_gap.set_xticks(np.linspace(0.0, gap_axis_max, 5))
    for y_idx, value in zip(y_pos, gap_values):
        ax_gap.text(value + max(gap_before * 0.025, 0.015), y_idx, f"{value:.3f}", va="center", fontsize=9)

    summary_text = (
        f"incumbent antes = {y_best_before:.3f}\n"
        f"prediccion en x_next: mu={mu_next_value:.3f}, sigma={std_next_value:.3f}\n"
        f"valor observado = {y_next_value:.3f}\n"
        f"gap cerrado = {gap_closed:.1%}"
    )
    ax_gap.text(
        0.98,
        0.08,
        summary_text,
        transform=ax_gap.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#666666", "alpha": 0.92},
    )

    fig.suptitle("Forrester: sintesis visual de una decision de Expected Improvement", y=0.975)
    fig.text(
        0.5,
        0.935,
        f"{model_name}, n_train inicial={n_train}, {sampler}, {noise}, step={target_step}; el GP mostrado es anterior a incorporar x_next",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.88)
    fig._skip_tight_layout = True
    save_figure(fig, base_path, dpi=dpi, save_svg=save_svg)

    metadata = pd.DataFrame(
        [
            {
                "benchmark": "forrester",
                "model": model_name,
                "n_train": int(n_train),
                "sampler": sampler,
                "noise": noise,
                "target_step": int(target_step),
                "x_next": x_next_value,
                "ei_next": ei_next_value,
                "y_best_before": y_best_before,
                "y_next": y_next_value,
                "y_best_after": y_best_after,
                "optimal_x": opt_x,
                "optimal_value": opt_y,
                "gap_before": gap_before,
                "gap_after": gap_after,
                "gap_closed": gap_closed,
                "mu_x_next": mu_next_value,
                "sigma_x_next": std_next_value,
            }
        ]
    )
    metadata_path = output_dir / "forrester_ei_decision_synthesis_step5.csv"
    metadata.to_csv(metadata_path, index=False)

    return {
        "figure_png": base_path.with_suffix(".png"),
        "figure_eps": base_path.with_suffix(".eps"),
        "metadata": metadata_path,
    }
