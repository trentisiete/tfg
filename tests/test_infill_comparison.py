#!/usr/bin/env python
"""
test_infill_comparison.py

Compares two infill strategies on the Branin 2D benchmark:
  - Mode A (active_train_all_models=True):  fixed GP variants run independently.
  - Mode B (active_train_all_models=False): single GP base with CV-audit switching.

Both start from the same initial DoE (n_train=1, sobol, no noise) so the
comparison is fair.  Produces publication-ready figures saved to tests/outputs/.

Usage:
    cd surrogate_models
    python -m tests.test_infill_comparison
"""

import sys
import time
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.base import clone

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmarks import get_benchmark, generate_benchmark_dataset, get_sampler
from src.benchmarks.noise import get_noise_injector
from src.analysis.active_learning import (
    run_active_evaluation,
    select_next_x,
    ei_values_from_model,
    is_active_supported,
)
from src.analysis.surrogate_metrics import compute_surrogate_metrics
from src.configs import (
    get_active_learning_config,
    get_default_models,
    get_base_models,
)

OUTPUT_DIR = ROOT / "tests" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
BENCHMARK_NAME = "branin"
N_TRAIN = 1
N_TEST = 300
SAMPLER = "sobol"
NOISE_TYPE = "none"
NOISE_KWARGS: Dict[str, Any] = {}
SEED = 42

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.4,
})

PALETTE = {
    "GP_Matern32": "#2176AE",
    "GP_Matern52": "#57B8FF",
    "GP_RBF": "#B66D0D",
    "GP": "#D64045",
    "Dummy": "#999999",
}


def _color(name: str) -> str:
    return PALETTE.get(name, "#333333")


# ===================================================================
# 1. Generate shared dataset
# ===================================================================
def build_shared_dataset():
    dataset = generate_benchmark_dataset(
        benchmark=BENCHMARK_NAME,
        n_train=N_TRAIN,
        n_test=N_TEST,
        sampler=SAMPLER,
        noise=NOISE_TYPE,
        noise_kwargs=NOISE_KWARGS,
        seed=SEED,
    )
    return dataset


# ===================================================================
# 2. Run both modes via run_active_evaluation
# ===================================================================
def run_mode(dataset, all_models: bool, audit_scoring: str = "mae",
             verbose: bool = True) -> Dict[str, Any]:
    bench = get_benchmark(BENCHMARK_NAME)
    cfg = get_active_learning_config(dim=bench.dim)

    if all_models:
        models = get_default_models()
    else:
        base = get_base_models()
        models = {"GP": base["GP"]}

    tag = "ALL_MODELS" if all_models else f"SINGLE_GP+SWITCH(scoring={audit_scoring})"
    print(f"\n{'='*70}")
    print(f"  MODE: {tag}")
    print(f"  Models: {list(models.keys())}")
    print(f"  n_infill={cfg['n_infill']}  xi={cfg['ei_xi']}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    results = run_active_evaluation(
        dataset=dataset,
        models=models,
        benchmark_name=BENCHMARK_NAME,
        sampler=SAMPLER,
        noise_type=NOISE_TYPE,
        noise_kwargs=NOISE_KWARGS,
        n_infill=cfg["n_infill"],
        xi=cfg["ei_xi"],
        active_cand_mult=cfg["active_cand_mult"],
        active_cv_check_every=cfg["active_cv_check_every"],
        active_switch_enable=not all_models,
        active_switch_warmup_steps=3,
        active_switch_min_improvement=0.01,
        active_switch_cooldown_steps=3,
        use_default_grids=True,
        seed=SEED,
        verbose=verbose,
        audit_scoring=audit_scoring,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed:.1f}s")
    return results


# ===================================================================
# 3. Extract trajectories
# ===================================================================
def extract_trajectories(results: Dict) -> Dict[str, List[Dict]]:
    trajs = {}
    for model_name, info in results.items():
        if info.get("active_supported") and info.get("trajectory"):
            trajs[model_name] = info["trajectory"]
    return trajs


def best_model_from_results(results: Dict) -> Tuple[str, Dict]:
    best_name, best_info = None, None
    best_mae = float("inf")
    for name, info in results.items():
        if not info.get("active_supported"):
            continue
        mae = info.get("mae")
        if mae is not None and mae < best_mae:
            best_mae = mae
            best_name = name
            best_info = info
    return best_name, best_info


# ===================================================================
# 4. Plotting helpers
# ===================================================================
def _branin_surface(n_grid: int = 200):
    bench = get_benchmark(BENCHMARK_NAME)
    x1 = np.linspace(bench.bounds[0][0], bench.bounds[0][1], n_grid)
    x2 = np.linspace(bench.bounds[1][0], bench.bounds[1][1], n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    grid = np.column_stack([X1.ravel(), X2.ravel()])
    Z = bench(grid).reshape(n_grid, n_grid)
    return X1, X2, Z


def plot_metric_evolution(trajs_a: Dict, trajs_b: Dict, trajs_c: Dict,
                          metric: str, ylabel: str, ax: plt.Axes, title: str):
    for name, traj in trajs_a.items():
        steps = [r["step"] for r in traj]
        vals = [r.get(metric) for r in traj]
        ax.plot(steps, vals, label=f"{name}", color=_color(name),
                linestyle="-", marker="o", markersize=2.5)

    for name, traj in trajs_b.items():
        steps = [r["step"] for r in traj]
        vals = [r.get(metric) for r in traj]
        ax.plot(steps, vals, label=f"{name} (MAE sw.)", color=_color(name),
                linestyle="--", marker="s", markersize=2.5)

    for name, traj in trajs_c.items():
        steps = [r["step"] for r in traj]
        vals = [r.get(metric) for r in traj]
        ax.plot(steps, vals, label=f"{name} (NLPD sw.)", color="#6A0DAD",
                linestyle=":", marker="D", markersize=2.5)

    ax.set_xlabel("Infill step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.8)


def plot_infill_points_on_surface(ax, X1, X2, Z, traj: List[Dict], dataset,
                                  title: str, color: str):
    ax.contourf(X1, X2, Z, levels=30, cmap="viridis", alpha=0.85)
    ax.contour(X1, X2, Z, levels=15, colors="white", linewidths=0.3, alpha=0.5)

    # Initial point(s)
    ax.scatter(dataset.X_train[:, 0], dataset.X_train[:, 1],
               c="white", edgecolors="black", s=60, zorder=5, label="Initial DoE")

    # Infill points with step numbering
    xs = [r["x_next"][0] for r in traj]
    ys = [r["x_next"][1] for r in traj]
    steps = [r["step"] for r in traj]
    scatter = ax.scatter(xs, ys, c=steps, cmap="autumn", edgecolors="black",
                         s=40, zorder=6, linewidths=0.5)

    # Arrow from each point to next
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=0.8, alpha=0.6))

    # Optimal points
    bench = get_benchmark(BENCHMARK_NAME)
    if bench.optimal_location is not None:
        opt = np.atleast_2d(bench.optimal_location)
        ax.scatter(opt[:, 0], opt[:, 1], marker="*", c="red", s=120,
                   edgecolors="black", zorder=7, label="Optima")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Infill step", fontsize=8)


def plot_switch_timeline(audits: List[Dict], ax: plt.Axes, n_infill: int):
    if not audits:
        ax.text(0.5, 0.5, "No audit data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888")
        ax.set_title("CV Audit — Hyperparameter Switches", fontweight="bold")
        return

    steps = []
    best_cv_maes = []
    current_cv_maes = []
    switched = []

    for a in audits:
        if a.get("status") != "ok":
            continue
        steps.append(a["step"])
        best_cv_maes.append(a.get("best_cv_mean"))
        current_cv_maes.append(a.get("current_cv_mean"))
        switched.append(a.get("switch_applied", False))

    ax.plot(steps, best_cv_maes, "o-", color="#2176AE", markersize=4,
            label="Best CV MAE (grid)")
    ax.plot(steps, current_cv_maes, "s--", color="#B66D0D", markersize=4,
            label="Current params CV MAE")

    for i, sw in enumerate(switched):
        if sw:
            ax.axvline(steps[i], color="#D64045", linestyle=":", alpha=0.7, lw=1.5)
            ax.annotate("switch", (steps[i], best_cv_maes[i]),
                        textcoords="offset points", xytext=(4, 8),
                        fontsize=7, color="#D64045", fontweight="bold")

    ax.set_xlabel("Infill step")
    ax.set_ylabel("CV MAE")
    ax.set_title("CV Audit — Hyperparameter Switches", fontweight="bold")
    ax.legend(loc="best", fontsize=8, framealpha=0.8)


# ===================================================================
# 5. Metrics summary table
# ===================================================================
def print_metrics_table(results_a: Dict, results_b: Dict, results_c: Dict):
    header = f"{'Mode':<18} {'Model':<16} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'NLPD':>8} {'Cov95':>7}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("FINAL METRICS COMPARISON")
    print(sep)
    print(header)
    print(sep)

    rows = []
    for name, info in results_a.items():
        if not info.get("active_supported"):
            continue
        rows.append(("AllModels", name, info))
    for name, info in results_b.items():
        if not info.get("active_supported"):
            continue
        rows.append(("Switch(MAE)", name, info))
    for name, info in results_c.items():
        if not info.get("active_supported"):
            continue
        rows.append(("Switch(NLPD)", name, info))

    for mode, name, info in rows:
        mae = info.get("mae")
        rmse = info.get("rmse")
        r2 = info.get("r2")
        nlpd = info.get("nlpd")
        cov = info.get("coverage_95")
        print(
            f"{mode:<18} {name:<16} "
            f"{mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} "
            f"{nlpd:>8.3f} {cov:>7.2%}" if all(v is not None for v in [mae, rmse, r2, nlpd, cov])
            else f"{mode:<18} {name:<16}  (unsupported)"
        )
    print(sep)

    # Best overall
    all_supported = [(m, n, i) for m, n, i in rows if i.get("mae") is not None]
    if all_supported:
        best_mae = min(all_supported, key=lambda x: x[2]["mae"])
        best_nlpd = min(all_supported, key=lambda x: x[2].get("nlpd", float("inf")))
        print(f"\n>> Best by MAE:  {best_mae[0]} / {best_mae[1]}  (MAE={best_mae[2]['mae']:.4f})")
        print(f">> Best by NLPD: {best_nlpd[0]} / {best_nlpd[1]}  (NLPD={best_nlpd[2].get('nlpd', 'N/A')})")


# ===================================================================
# 6. Main figure
# ===================================================================
def create_comparison_figure(dataset, results_a, results_b, results_c):
    trajs_a = extract_trajectories(results_a)
    trajs_b = extract_trajectories(results_b)
    trajs_c = extract_trajectories(results_c)
    best_a_name, best_a_info = best_model_from_results(results_a)

    X1, X2, Z = _branin_surface()
    bench = get_benchmark(BENCHMARK_NAME)
    cfg = get_active_learning_config(dim=bench.dim)

    # ---- Figure 1: Surface + infill points ----
    fig1, axes1 = plt.subplots(1, 3, figsize=(19, 5.5))
    fig1.suptitle(
        f"Infill Comparison on {BENCHMARK_NAME.capitalize()} "
        f"(n_train={N_TRAIN}, n_infill={cfg['n_infill']}, sampler={SAMPLER}, noise={NOISE_TYPE})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Left: Mode A best model
    if best_a_name and best_a_name in trajs_a:
        plot_infill_points_on_surface(
            axes1[0], X1, X2, Z, trajs_a[best_a_name], dataset,
            f"Mode A: Best fixed ({best_a_name})",
            _color(best_a_name),
        )

    # Center: Mode B single GP with MAE switch
    if "GP" in trajs_b:
        plot_infill_points_on_surface(
            axes1[1], X1, X2, Z, trajs_b["GP"], dataset,
            "Mode B: GP + Switch (MAE)",
            _color("GP"),
        )

    # Right: Mode C single GP with NLPD switch
    if "GP" in trajs_c:
        plot_infill_points_on_surface(
            axes1[2], X1, X2, Z, trajs_c["GP"], dataset,
            "Mode C: GP + Switch (NLPD)",
            "#6A0DAD",
        )

    fig1.tight_layout()
    path1 = OUTPUT_DIR / "fig1_infill_surface.png"
    fig1.savefig(path1, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path1}")

    # ---- Figure 2: Metric evolution ----
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
    fig2.suptitle("Metric Evolution During Infill", fontsize=13, fontweight="bold")

    plot_metric_evolution(trajs_a, trajs_b, trajs_c, "mae_test", "MAE", axes2[0, 0], "MAE vs Step")
    plot_metric_evolution(trajs_a, trajs_b, trajs_c, "rmse_test", "RMSE", axes2[0, 1], "RMSE vs Step")
    plot_metric_evolution(trajs_a, trajs_b, trajs_c, "r2_test", "R²", axes2[1, 0], "R² vs Step")
    plot_metric_evolution(trajs_a, trajs_b, trajs_c, "nlpd_test", "NLPD", axes2[1, 1], "NLPD vs Step")

    fig2.tight_layout()
    path2 = OUTPUT_DIR / "fig2_metric_evolution.png"
    fig2.savefig(path2, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path2}")

    # ---- Figure 3: Switch timeline (Mode B and C) ----
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 4.5))
    audits_b = []
    for name, info in results_b.items():
        if info.get("active_supported") and info.get("hyperparam_audit"):
            audits_b = info["hyperparam_audit"]
            break
    plot_switch_timeline(audits_b, ax3a, cfg["n_infill"])
    ax3a.set_title("CV Audit — Switch by MAE", fontweight="bold")

    audits_c = []
    for name, info in results_c.items():
        if info.get("active_supported") and info.get("hyperparam_audit"):
            audits_c = info["hyperparam_audit"]
            break
    plot_switch_timeline(audits_c, ax3b, cfg["n_infill"])
    ax3b.set_title("CV Audit — Switch by NLPD", fontweight="bold")

    fig3.tight_layout()
    path3 = OUTPUT_DIR / "fig3_switch_timeline.png"
    fig3.savefig(path3, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path3}")

    # ---- Figure 4: EI value evolution ----
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    for name, traj in trajs_a.items():
        steps = [r["step"] for r in traj]
        eis = [r["ei_next"] for r in traj]
        ax4.plot(steps, eis, label=f"{name}", color=_color(name), marker="o", markersize=2.5)
    for name, traj in trajs_b.items():
        steps = [r["step"] for r in traj]
        eis = [r["ei_next"] for r in traj]
        ax4.plot(steps, eis, label=f"{name} (MAE sw.)", color=_color(name),
                 linestyle="--", marker="s", markersize=2.5)
    for name, traj in trajs_c.items():
        steps = [r["step"] for r in traj]
        eis = [r["ei_next"] for r in traj]
        ax4.plot(steps, eis, label=f"{name} (NLPD sw.)", color="#6A0DAD",
                 linestyle=":", marker="D", markersize=2.5)
    ax4.set_xlabel("Infill step")
    ax4.set_ylabel("EI value")
    ax4.set_title("Expected Improvement per Step", fontweight="bold")
    ax4.legend(loc="best", fontsize=8, framealpha=0.8)
    fig4.tight_layout()
    path4 = OUTPUT_DIR / "fig4_ei_evolution.png"
    fig4.savefig(path4, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path4}")

    # ---- Figure 5: Summary bar chart ----
    fig5, axes5 = plt.subplots(1, 3, figsize=(14, 4.5))
    fig5.suptitle("Final Metrics — All Models", fontsize=13, fontweight="bold")

    all_entries = []
    for name, info in results_a.items():
        if info.get("active_supported") and info.get("mae") is not None:
            all_entries.append((name, info))
    for name, info in results_b.items():
        if info.get("active_supported") and info.get("mae") is not None:
            all_entries.append((f"{name}\n(MAE sw.)", info))
    for name, info in results_c.items():
        if info.get("active_supported") and info.get("mae") is not None:
            all_entries.append((f"{name}\n(NLPD sw.)", info))

    if all_entries:
        labels = [e[0] for e in all_entries]
        colors = [_color(e[0].split("\n")[0]) for e in all_entries]
        x_pos = np.arange(len(labels))

        for ax, metric, title in [
            (axes5[0], "mae", "MAE (lower is better)"),
            (axes5[1], "rmse", "RMSE (lower is better)"),
            (axes5[2], "r2", "R² (higher is better)"),
        ]:
            vals = [e[1].get(metric, 0) for e in all_entries]
            bars = ax.bar(x_pos, vals, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(title, fontweight="bold")

            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.4f}", ha="center", va="bottom", fontsize=7)

    fig5.tight_layout()
    path5 = OUTPUT_DIR / "fig5_final_metrics_bars.png"
    fig5.savefig(path5, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path5}")

    plt.close("all")


# ===================================================================
# 7. EI Landscape replay
# ===================================================================
EI_GRID_N = 120  # resolution for EI heatmap


def _build_ei_grid():
    """Build a dense grid over the Branin domain for EI evaluation."""
    bench = get_benchmark(BENCHMARK_NAME)
    x1 = np.linspace(bench.bounds[0][0], bench.bounds[0][1], EI_GRID_N)
    x2 = np.linspace(bench.bounds[1][0], bench.bounds[1][1], EI_GRID_N)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    return X1, X2, X_grid


def replay_infill_with_ei(
    dataset,
    model_template,
    trajectory: List[Dict],
    label: str,
) -> List[Dict[str, Any]]:
    """
    Replay the infill sequence recorded in *trajectory*, re-fitting the model
    at each step and capturing the full EI landscape on a dense 2-D grid.

    Returns a list of dicts, one per step, each containing:
        step, X1, X2, EI_grid, x_next, X_train_so_far, y_train_so_far
    """
    from src.analysis.active_learning import _manual_ei_fallback

    bench = get_benchmark(BENCHMARK_NAME)
    cfg = get_active_learning_config(dim=bench.dim)
    X1_g, X2_g, X_grid = _build_ei_grid()

    X_train = np.array(dataset.X_train, dtype=float)
    y_train = np.array(dataset.y_train, dtype=float).ravel()

    m = clone(model_template)
    m.fit(X_train, y_train)

    snapshots: List[Dict[str, Any]] = []

    for rec in trajectory:
        step = rec["step"]
        y_best = float(np.min(y_train))

        # Compute EI on grid
        try:
            ei_grid = ei_values_from_model(m, X_grid, y_best, xi=cfg["ei_xi"])
        except Exception:
            ei_grid = _manual_ei_fallback(m, X_grid, y_best, xi=cfg["ei_xi"])
        ei_grid = np.asarray(ei_grid).ravel()

        x_next = np.asarray(rec["x_next"])
        y_next = float(rec["y_next"])

        snapshots.append({
            "step": step,
            "X1": X1_g,
            "X2": X2_g,
            "EI": ei_grid.reshape(X1_g.shape),
            "x_next": x_next,
            "X_train": X_train.copy(),
            "y_best": y_best,
        })

        # Advance training set exactly as the original run did
        X_train = np.vstack([X_train, x_next.reshape(1, -1)])
        y_train = np.concatenate([y_train, [y_next]])
        m.fit(X_train, y_train)

        print(f"    [EI replay] {label} step={step} y_best={y_best:.4f}")

    return snapshots


def plot_ei_landscapes(snapshots_a: List[Dict], snapshots_b: List[Dict],
                       label_a: str, label_b: str):
    """
    Create a grid figure: rows = [Mode A, Mode B], cols = infill steps.
    Each cell shows the EI heatmap, existing training points, and the chosen x_next.
    """
    bench = get_benchmark(BENCHMARK_NAME)
    n_steps = len(snapshots_a)

    # Steps to show: every 2 steps for readability, always include first and last
    show_steps = sorted(set([0, n_steps - 1] + list(range(0, n_steps, 2))))
    n_cols = len(show_steps)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.4),
                             constrained_layout=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        "EI Landscape Evolution During Infill",
        fontsize=14, fontweight="bold",
    )

    # Optimal locations
    opt_locs = np.atleast_2d(bench.optimal_location) if bench.optimal_location is not None else None

    for row_idx, (snaps, label, color) in enumerate([
        (snapshots_a, label_a, _color(label_a)),
        (snapshots_b, label_b, _color(label_b)),
    ]):
        for col_idx, step_idx in enumerate(show_steps):
            ax = axes[row_idx, col_idx]
            snap = snaps[step_idx]

            # EI heatmap
            ei_data = snap["EI"]
            ei_max = ei_data.max()
            if ei_max > 0:
                ei_norm = ei_data / ei_max
            else:
                ei_norm = ei_data

            im = ax.contourf(snap["X1"], snap["X2"], ei_norm,
                             levels=30, cmap="inferno", alpha=0.9)
            ax.contour(snap["X1"], snap["X2"], ei_norm,
                       levels=10, colors="white", linewidths=0.2, alpha=0.4)

            # Training points so far
            Xt = snap["X_train"]
            ax.scatter(Xt[:, 0], Xt[:, 1], c="white", edgecolors="black",
                       s=20, zorder=5, linewidths=0.5)

            # Chosen next point
            xn = snap["x_next"]
            ax.scatter(xn[0], xn[1], marker="X", c="lime", edgecolors="black",
                       s=70, zorder=6, linewidths=0.8)

            # Optima
            if opt_locs is not None:
                ax.scatter(opt_locs[:, 0], opt_locs[:, 1], marker="*",
                           c="red", s=60, edgecolors="black", zorder=7,
                           linewidths=0.5)

            ax.set_xlim(bench.bounds[0])
            ax.set_ylim(bench.bounds[1])

            # Title: step number + EI max value
            ax.set_title(f"Step {snap['step']}\nEI_max={ei_max:.3f}",
                         fontsize=8, fontweight="bold")

            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold")
            else:
                ax.set_yticklabels([])

            if row_idx == 1:
                ax.set_xlabel("$x_1$", fontsize=8)
            else:
                ax.set_xticklabels([])

            ax.tick_params(labelsize=6)

    # Legend
    from matplotlib.lines import Line2D as L2D
    legend_elems = [
        L2D([0], [0], marker="o", color="w", markerfacecolor="white",
            markeredgecolor="black", markersize=6, label="Training pts"),
        L2D([0], [0], marker="X", color="w", markerfacecolor="lime",
            markeredgecolor="black", markersize=8, label="Next point (x_next)"),
        L2D([0], [0], marker="*", color="w", markerfacecolor="red",
            markeredgecolor="black", markersize=10, label="Optimum"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    path = OUTPUT_DIR / "fig6_ei_landscapes.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=200)
    print(f"Saved: {path}")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print(f"INFILL COMPARISON TEST — Benchmark: {BENCHMARK_NAME.upper()}")
    print(f"n_train={N_TRAIN}  sampler={SAMPLER}  noise={NOISE_TYPE}  seed={SEED}")
    print("=" * 70)

    dataset = build_shared_dataset()
    print(f"Dataset: X_train={dataset.X_train.shape}, X_test={dataset.X_test.shape}")

    # Mode A: all models, fixed hyperparams, no switching
    results_a = run_mode(dataset, all_models=True, verbose=True)

    # Mode B: single GP base, CV-audit switching scored by MAE
    results_b = run_mode(dataset, all_models=False, audit_scoring="mae", verbose=True)

    # Mode C: single GP base, CV-audit switching scored by NLPD
    results_c = run_mode(dataset, all_models=False, audit_scoring="nlpd", verbose=True)

    # Print comparison table
    print_metrics_table(results_a, results_b, results_c)

    # Generate figures 1-5
    create_comparison_figure(dataset, results_a, results_b, results_c)

    # ---- Figure 6: EI landscape replay (Mode A best vs Mode B vs Mode C) ----
    print("\n[EI Landscape Replay]")
    best_a_name, _ = best_model_from_results(results_a)
    trajs_a = extract_trajectories(results_a)
    trajs_c = extract_trajectories(results_c)

    if best_a_name and best_a_name in trajs_a and "GP" in trajs_c:
        models_a = get_default_models()
        models_c = get_base_models()

        snapshots_a = replay_infill_with_ei(
            dataset, models_a[best_a_name], trajs_a[best_a_name], best_a_name,
        )
        snapshots_c = replay_infill_with_ei(
            dataset, models_c["GP"], trajs_c["GP"], "GP (NLPD sw.)",
        )

        plot_ei_landscapes(snapshots_a, snapshots_c, best_a_name, "GP (NLPD sw.)")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
