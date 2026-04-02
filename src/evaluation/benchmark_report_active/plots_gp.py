from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

from src.benchmarks import generate_benchmark_dataset, get_benchmark
from src.models.gp import GPSurrogateRegressor

from .styling import sanitize_filename, save_figure


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


def _build_gp_model(model_name: str) -> GPSurrogateRegressor:
    if model_name == "GP_Matern32":
        kernel = Matern(nu=1.5) + WhiteKernel(noise_level=1e-5)
    elif model_name == "GP_Matern52":
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
    elif model_name == "GP_RBF":
        kernel = RBF() + WhiteKernel(noise_level=1e-5)
    else:
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
    return GPSurrogateRegressor(kernel=kernel, n_restarts_optimizer=3)


def _parse_x_next_cell(v) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v.astype(float).ravel()
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=float).ravel()
    return np.asarray([float(v)], dtype=float)


def _selected_steps(steps: Sequence[int]) -> List[int]:
    if not steps:
        return []
    wanted = [1, 5, 10, 15, 20]
    last = int(max(steps))
    out = sorted(set([s for s in wanted if s in steps] + [last]))
    return out


def _reconstruct_snapshots(
    block: pd.DataFrame,
    metadata: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray]], object]:
    benchmark = str(block["benchmark"].iloc[0])
    sampler = str(block["sampler"].iloc[0])
    noise_label = str(block["noise"].iloc[0])
    n_train = int(block["n_train"].iloc[0])
    seed = int(metadata.get("seed", 42))
    n_test = int(metadata.get("n_test", 200))

    noise_cfg = _noise_label_to_cfg(noise_label)
    ds = generate_benchmark_dataset(
        benchmark=benchmark,
        n_train=n_train,
        n_test=n_test,
        sampler=sampler,
        noise=str(noise_cfg["type"]),
        noise_kwargs=dict(noise_cfg["kwargs"]),
        n_groups=None,
        seed=seed,
    )
    bench = get_benchmark(benchmark)

    x_app = np.vstack([_parse_x_next_cell(v) for v in block["x_next"]]) if len(block) else np.empty((0, ds.X_train.shape[1]))
    y_app = block["y_next"].to_numpy(dtype=float) if "y_next" in block else np.empty((0,))

    snapshots: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    # n=0 means no observed points in the progression plot (pure prior view).
    snapshots[0] = (np.empty((0, bench.dim), dtype=float), np.empty((0,), dtype=float))
    for step in sorted(block["step"].astype(int).unique().tolist()):
        k = int(step)
        Xk = x_app[:k] if len(x_app) >= k else np.empty((0, bench.dim), dtype=float)
        yk = y_app[:k] if len(y_app) >= k else np.empty((0,), dtype=float)
        snapshots[k] = (Xk, yk)
    return ds.X_test, ds.y_test_clean, snapshots, bench


def _plot_1d_snapshot(
    bench,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    step: int,
    out_path: Path,
    dpi: int,
    save_svg: bool,
) -> None:
    xg, y_true, y_pred, y_std = _predict_1d_snapshot_curves(
        bench=bench,
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.fill_between(xg, y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.20, color="#fdb462", label="+/-2sigma")
    ax.fill_between(xg, y_pred - y_std, y_pred + y_std, alpha=0.35, color="#fb8072", label="+/-1sigma")
    ax.plot(xg, y_true, color="#1f78b4", linewidth=2.0, label="f(x) real")
    ax.plot(xg, y_pred, color="#d95f02", linewidth=2.0, label=f"{model_name} prediccion")
    if len(X_train):
        ax.scatter(X_train[:, 0], y_train, color="black", marker="o", s=24, alpha=0.8, label="Puntos train")

    opt_x = getattr(bench, "optimal_location", None)
    opt_y = getattr(bench, "optimal_value", None)
    if opt_x is not None and np.ndim(opt_x) == 1 and len(opt_x) == 1 and opt_y is not None:
        ax.scatter([opt_x[0]], [opt_y], marker="*", s=180, color="#6a3d9a", edgecolor="black", label="Optimo")

    ax.set_title(f"{bench.name} | {model_name} | snapshot paso={step}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _predict_1d_snapshot_curves(
    bench,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lb, ub = bench.bounds[0]
    xg = np.linspace(lb, ub, 450).reshape(-1, 1)
    y_true = np.asarray(bench(xg), dtype=float).ravel()

    # n=0: no observed points yet. Use a prior-like display.
    if X_train is None or len(X_train) == 0:
        y_pred = np.zeros_like(y_true, dtype=float)
        prior_scale = float(np.std(y_true))
        if not np.isfinite(prior_scale) or prior_scale < 1e-6:
            prior_scale = 1.0
        y_std = np.full_like(y_true, prior_scale, dtype=float)
        return xg.ravel(), y_true, y_pred, y_std

    model = _build_gp_model(model_name)
    model.fit(X_train, y_train)
    y_pred, y_std = model.predict_dist(xg)
    return xg.ravel(), y_true, np.asarray(y_pred, dtype=float).ravel(), np.asarray(y_std, dtype=float).ravel()


def _draw_1d_snapshot_on_ax(
    ax,
    bench,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    step: int,
    n_current: Optional[int],
    y_lim: Optional[Tuple[float, float]] = None,
) -> List:
    xg, y_true, y_pred, y_std = _predict_1d_snapshot_curves(
        bench=bench,
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
    )

    h2 = ax.fill_between(xg, y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.18, color="#fdb462", label="+/-2sigma")
    h1 = ax.fill_between(xg, y_pred - y_std, y_pred + y_std, alpha=0.30, color="#fb8072", label="+/-1sigma")
    h_true = ax.plot(xg, y_true, color="#1f78b4", linewidth=1.8, label="f real")[0]
    h_pred = ax.plot(xg, y_pred, color="#d95f02", linewidth=1.7, label="pred GP")[0]
    h_train = ax.scatter(X_train[:, 0], y_train, color="black", marker="o", s=12, alpha=0.85, label="train")

    h_opt = None
    opt_x = getattr(bench, "optimal_location", None)
    opt_y = getattr(bench, "optimal_value", None)
    if opt_x is not None and np.ndim(opt_x) == 1 and len(opt_x) == 1 and opt_y is not None:
        h_opt = ax.scatter([opt_x[0]], [opt_y], marker="*", s=72, color="#6a3d9a", edgecolor="black", linewidth=0.5, label="optimo")

    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.25)
    tag = f"n={int(step)}"
    ax.text(
        0.02,
        0.95,
        tag,
        transform=ax.transAxes,
        fontsize=8.0,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#666666", "alpha": 0.75},
    )

    handles = [h2, h1, h_true, h_pred, h_train]
    labels = ["+/-2sigma", "+/-1sigma", "f real", "pred GP", "train"]
    if h_opt is not None:
        handles.append(h_opt)
        labels.append("optimo")
    return list(zip(handles, labels))


def _plot_1d_noise_progression(
    benchmark: str,
    noise: str,
    model_name: str,
    bench,
    snapshots: Dict[int, Tuple[np.ndarray, np.ndarray]],
    step_to_ntrain: Dict[int, int],
    out_path: Path,
    dpi: int,
    save_svg: bool,
) -> None:
    steps = sorted(int(s) for s in snapshots.keys())
    if not steps:
        return

    # Global y-range for comparability inside this figure.
    y_all = []
    for st in steps:
        Xk, yk = snapshots[st]
        _, y_true, y_pred, y_std = _predict_1d_snapshot_curves(
            bench=bench,
            model_name=model_name,
            X_train=Xk,
            y_train=yk,
        )
        y_all.extend([np.min(y_true), np.max(y_true), np.min(y_pred - 2 * y_std), np.max(y_pred + 2 * y_std)])
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    pad = 0.08 * max(1e-6, y_max - y_min)
    y_lim = (y_min - pad, y_max + pad)

    ncols = min(4, len(steps))
    nrows = int(np.ceil(len(steps) / float(ncols)))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.8 * ncols, 3.2 * nrows), sharex=True, sharey=True)
    axs = np.atleast_1d(axs).reshape(nrows, ncols)

    fig_handles = []
    fig_labels = []
    k = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r, c]
            if k >= len(steps):
                ax.set_axis_off()
                continue
            st = steps[k]
            Xk, yk = snapshots[st]
            entries = _draw_1d_snapshot_on_ax(
                ax=ax,
                bench=bench,
                model_name=model_name,
                X_train=Xk,
                y_train=yk,
                step=st,
                n_current=step_to_ntrain.get(st),
                y_lim=y_lim,
            )
            for h, l in entries:
                if l not in fig_labels:
                    fig_labels.append(l)
                    fig_handles.append(h)
            k += 1

    if fig_handles:
        fig.legend(
            fig_handles,
            fig_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(6, len(fig_labels)),
            frameon=True,
        )
    fig.subplots_adjust(top=0.87, bottom=0.14, hspace=0.30, wspace=0.18)
    fig.suptitle(
        f"{benchmark} | ruido={noise} | modelo={model_name} | evolucion infill n=0..{max(steps)}",
        y=0.98,
    )
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _predict_2d_surfaces(
    bench,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n: int = 48,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1 = np.linspace(bench.bounds[0][0], bench.bounds[0][1], n)
    x2 = np.linspace(bench.bounds[1][0], bench.bounds[1][1], n)
    xx, yy = np.meshgrid(x1, x2)
    Xg = np.column_stack([xx.ravel(), yy.ravel()])
    z_true = bench(Xg).reshape(n, n)

    # n=0 snapshot: no observed points yet.
    if X_train is None or len(X_train) == 0:
        z_pred = np.zeros_like(z_true, dtype=float)
        return xx, yy, z_true, z_pred

    model = _build_gp_model(model_name)
    model.fit(X_train, y_train)
    z_pred, _ = model.predict_dist(Xg)
    z_pred = np.asarray(z_pred, dtype=float).reshape(n, n)
    return xx, yy, z_true, z_pred


def _plot_2d_noise_progression_3d(
    benchmark: str,
    noise: str,
    model_name: str,
    bench,
    snapshots: Dict[int, Tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    dpi: int,
    save_svg: bool,
) -> None:
    steps = sorted(int(s) for s in snapshots.keys())
    if not steps:
        return

    ncols = min(3, len(steps))
    nrows = int(np.ceil(len(steps) / float(ncols)))
    fig = plt.figure(figsize=(5.2 * ncols, 4.4 * nrows))

    for idx, st in enumerate(steps, start=1):
        ax = fig.add_subplot(nrows, ncols, idx, projection="3d")
        Xk, yk = snapshots[st]
        xx, yy, z_true, z_pred = _predict_2d_surfaces(
            bench=bench,
            model_name=model_name,
            X_train=Xk,
            y_train=yk,
            n=42,
        )
        z_floor = float(min(np.min(z_true), np.min(z_pred)))
        z_ceil = float(max(np.max(z_true), np.max(z_pred)))
        z_pad = 0.06 * max(1e-6, z_ceil - z_floor)
        z_floor = z_floor - z_pad
        z_top = z_ceil + z_pad

        ax.plot_surface(
            xx,
            yy,
            z_pred,
            cmap="turbo",
            alpha=0.94,
            linewidth=0.06,
            edgecolor=(0.0, 0.0, 0.0, 0.08),
            antialiased=True,
        )
        # Show real function explicitly as a wireframe overlay (slightly lifted to avoid z-fighting).
        z_eps = 0.004 * max(1e-6, z_top - z_floor)
        ax.plot_wireframe(
            xx,
            yy,
            z_true + z_eps,
            color="#111111",
            linewidth=0.55,
            rstride=2,
            cstride=2,
            alpha=0.82,
        )
        ax.contour(
            xx,
            yy,
            z_true,
            zdir="z",
            offset=z_floor,
            cmap="Greys",
            levels=9,
            linewidths=0.9,
            alpha=0.85,
        )
        if len(Xk):
            ax.scatter(Xk[:, 0], Xk[:, 1], yk, c="black", s=18, alpha=0.9, marker="o", depthshade=False)
        ax.set_title(f"n={int(st)}", fontsize=10)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f")
        ax.set_zlim(z_floor, z_top)
        ax.set_box_aspect((1.0, 1.0, 0.70))
        ax.set_proj_type("persp", focal_length=1.25)
        ax.view_init(elev=24, azim=-132)
        legend_handles = [
            Line2D([0], [0], color="#111111", lw=1.2, label="f real (wireframe)"),
            Line2D([0], [0], color="#666666", lw=1.0, label="f real (contour base)"),
            Line2D([0], [0], color="#1f77b4", lw=2.0, label=f"{model_name} pred"),
            Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=4.5, label="train"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, frameon=True)

    fig.subplots_adjust(top=0.90, hspace=0.28, wspace=0.18)
    fig.suptitle(
        f"{benchmark} 3D | ruido={noise} | modelo={model_name} | evolucion infill n=0..{max(steps)}",
        y=0.98,
    )
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _plot_2d_snapshot(
    bench,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    step: int,
    out_path: Path,
    dpi: int,
    save_svg: bool,
) -> None:
    model = _build_gp_model(model_name)
    model.fit(X_train, y_train)
    n = 75
    x1 = np.linspace(bench.bounds[0][0], bench.bounds[0][1], n)
    x2 = np.linspace(bench.bounds[1][0], bench.bounds[1][1], n)
    xx, yy = np.meshgrid(x1, x2)
    Xg = np.column_stack([xx.ravel(), yy.ravel()])
    z_true = bench(Xg).reshape(n, n)
    z_pred, z_std = model.predict_dist(Xg)
    z_pred = z_pred.reshape(n, n)
    z_std = z_std.reshape(n, n)

    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    c0 = axs[0].contourf(xx, yy, z_true, levels=20, cmap="viridis")
    fig.colorbar(c0, ax=axs[0])
    axs[0].scatter(X_train[:, 0], X_train[:, 1], c="white", edgecolors="black", s=20, label="Train")
    axs[0].set_title("Funcion real")
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("x2")
    axs[0].legend(loc="best")

    c1 = axs[1].contourf(xx, yy, z_pred, levels=20, cmap="magma")
    fig.colorbar(c1, ax=axs[1])
    axs[1].scatter(X_train[:, 0], X_train[:, 1], c="white", edgecolors="black", s=20)
    axs[1].set_title(f"Prediccion {model_name}")
    axs[1].set_xlabel("x1")
    axs[1].set_ylabel("x2")

    c2 = axs[2].contourf(xx, yy, z_std, levels=20, cmap="cividis")
    fig.colorbar(c2, ax=axs[2])
    axs[2].set_title("Desviacion estandar")
    axs[2].set_xlabel("x1")
    axs[2].set_ylabel("x2")

    fig.suptitle(f"{bench.name} | paso={step}")
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _plot_nd_slice_snapshot(
    bench,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    step: int,
    out_path: Path,
    dpi: int,
    save_svg: bool,
) -> None:
    model = _build_gp_model(model_name)
    model.fit(X_train, y_train)
    dim = int(bench.dim)
    centers = np.array([(b[0] + b[1]) * 0.5 for b in bench.bounds], dtype=float)
    n = 240
    x0 = np.linspace(bench.bounds[0][0], bench.bounds[0][1], n)
    Xg = np.tile(centers, (n, 1))
    Xg[:, 0] = x0
    y_true = bench(Xg)
    y_pred, y_std = model.predict_dist(Xg)

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.fill_between(x0, y_pred - y_std, y_pred + y_std, alpha=0.25, color="#fb8072", label="±1σ")
    ax.plot(x0, y_true, color="#1f78b4", linewidth=2.0, label="f real (slice)")
    ax.plot(x0, y_pred, color="#d95f02", linewidth=2.0, label=f"{model_name} (slice)")
    ax.set_title(f"{bench.name} dim={dim} | slice principal | paso={step}")
    ax.set_xlabel("x0 (resto fijo en centro)")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _build_noise_matrix_montage(
    benchmark: str,
    rows: List[Dict[str, object]],
    out_path: Path,
    dpi: int,
    save_svg: bool,
) -> None:
    if not rows:
        return

    step_sets = [set(int(k) for k in row.get("step_to_path", {}).keys()) for row in rows]
    common = sorted(set.intersection(*step_sets)) if step_sets else []
    if common:
        steps_to_show = common
    else:
        steps_to_show = sorted({int(step) for row in rows for step in row.get("step_to_path", {}).keys()})
    if not steps_to_show:
        return

    dim = int(rows[0].get("dim", 0))
    if dim == 1 and all("snapshots" in row and row.get("snapshots") for row in rows):
        nrows = len(rows)
        ncols = len(steps_to_show)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.1 * ncols, 2.9 * nrows), sharex=True, sharey=True)
        if nrows == 1:
            axs = np.array([axs])
        if ncols == 1:
            axs = axs.reshape(nrows, 1)

        # Global y-range for comparability between rows.
        y_all = []
        for row in rows:
            bench = row.get("bench")
            model = str(row.get("model", "GP"))
            snapshots = row.get("snapshots", {})
            for st in steps_to_show:
                if st not in snapshots:
                    continue
                Xk, yk = snapshots[st]
                xg, y_true, y_pred, y_std = _predict_1d_snapshot_curves(
                    bench=bench,
                    model_name=model,
                    X_train=Xk,
                    y_train=yk,
                )
                y_all.append(np.min(y_true))
                y_all.append(np.max(y_true))
                y_all.append(np.min(y_pred - 2 * y_std))
                y_all.append(np.max(y_pred + 2 * y_std))
        if y_all:
            y_min = float(np.min(y_all))
            y_max = float(np.max(y_all))
            pad = 0.08 * max(1e-6, y_max - y_min)
            y_lim = (y_min - pad, y_max + pad)
        else:
            y_lim = None

        fig_handles = []
        fig_labels = []
        for i, row in enumerate(rows):
            noise = str(row.get("noise", "noise"))
            model = str(row.get("model", "GP"))
            snapshots = row.get("snapshots", {})
            step_to_ntrain: Dict[int, int] = row.get("step_to_ntrain", {})
            for j, step in enumerate(steps_to_show):
                ax = axs[i, j]
                if step not in snapshots:
                    ax.axis("off")
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=9)
                    continue

                Xk, yk = snapshots[step]
                entries = _draw_1d_snapshot_on_ax(
                    ax=ax,
                    bench=row.get("bench"),
                    model_name=model,
                    X_train=Xk,
                    y_train=yk,
                    step=step,
                    n_current=step_to_ntrain.get(step),
                    y_lim=y_lim,
                )
                for h, l in entries:
                    if l not in fig_labels:
                        fig_labels.append(l)
                        fig_handles.append(h)

                if i == 0:
                    ax.set_title(f"step {int(step)}", fontsize=11)
                if j == 0:
                    ax.set_ylabel(f"{noise}\n{model}", fontsize=9)

        if fig_handles:
            fig.legend(
                fig_handles,
                fig_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=min(6, len(fig_labels)),
                frameon=True,
            )
        fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.20)
        fig.suptitle(f"{benchmark}: evolucion GP por ruido (matrix interpretable)")
        save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)
        return

    # Generic fallback (image montage) for >1D.
    nrows = len(rows)
    ncols = len(steps_to_show)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.1 * ncols, 2.7 * nrows))
    if nrows == 1:
        axs = np.array([axs])
    if ncols == 1:
        axs = axs.reshape(nrows, 1)

    for i, row in enumerate(rows):
        noise = str(row.get("noise", "noise"))
        model = str(row.get("model", "GP"))
        step_to_path: Dict[int, Path] = row.get("step_to_path", {})
        step_to_ntrain: Dict[int, int] = row.get("step_to_ntrain", {})
        for j, step in enumerate(steps_to_show):
            ax = axs[i, j]
            p = step_to_path.get(step)
            if p is not None and Path(p).exists():
                img = mpimg.imread(p)
                ax.imshow(img)
                ax.axis("off")
                n_current = step_to_ntrain.get(step)
                if n_current is not None:
                    ax.text(
                        0.01,
                        0.01,
                        f"step={int(step)} | n={int(n_current)}",
                        transform=ax.transAxes,
                        fontsize=7.5,
                        va="bottom",
                        ha="left",
                        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "black", "alpha": 0.80},
                    )
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=9)
            if i == 0:
                ax.set_title(f"step {int(step)}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{noise}\n{model}", fontsize=9)

    fig.suptitle(f"GP evolucion simplificada por ruido - {benchmark}")
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def generate_gp_predictions(
    master_df: pd.DataFrame,
    final_df: pd.DataFrame,
    metadata: Dict[str, object],
    out_dir: Path,
    max_gp_benchmarks: int = 3,
    max_gp_steps: int = 20,
    dpi: int = 300,
    save_svg: bool = False,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gp_final = final_df[final_df["model"].astype(str).str.startswith("GP")].copy()
    if gp_final.empty:
        return

    # Pick one canonical GP model per benchmark, then keep that model across noises when possible.
    bench_model_mean = (
        gp_final.groupby(["benchmark", "model"], as_index=False)["mae"]
        .mean()
        .sort_values(["benchmark", "mae"])
    )
    canonical_model_by_bench = (
        bench_model_mean.groupby("benchmark", as_index=False)
        .head(1)
        .set_index("benchmark")["model"]
        .to_dict()
    )

    bench_order = gp_final["benchmark"].drop_duplicates().tolist()[: int(max_gp_benchmarks)]
    rep_rows: List[pd.DataFrame] = []
    for bench in bench_order:
        bench_block = gp_final[gp_final["benchmark"] == bench]
        if bench_block.empty:
            continue
        canonical = canonical_model_by_bench.get(bench)
        for noise in sorted(bench_block["noise"].dropna().unique().tolist()):
            noise_block = bench_block[bench_block["noise"] == noise]
            if noise_block.empty:
                continue
            cand = noise_block[noise_block["model"] == canonical]
            if cand.empty:
                cand = noise_block
            rep_rows.append(cand.sort_values("mae").head(1))

    reps = pd.concat(rep_rows, ignore_index=True) if rep_rows else pd.DataFrame(columns=gp_final.columns)

    for _, row in reps.iterrows():
        benchmark = str(row["benchmark"])
        noise = str(row["noise"])
        sampler = str(row["sampler"])
        n_train = int(row["n_train"])
        model_name = str(row["model"])

        block = master_df[
            (master_df["benchmark"] == benchmark)
            & (master_df["noise"] == noise)
            & (master_df["sampler"] == sampler)
            & (master_df["n_train"] == n_train)
            & (master_df["model"] == model_name)
        ].sort_values("step")
        if block.empty:
            continue
        block = block.head(int(max_gp_steps)).copy()

        _, _, snapshots, bench = _reconstruct_snapshots(block=block, metadata=metadata)
        if not snapshots:
            continue

        full_steps = sorted(snapshots.keys())
        # For 1D matrix visualization, keep all steps from n_add=0..last
        # to show the full active-learning progression.
        if int(bench.dim) == 1:
            simp_steps = full_steps
        else:
            simp_steps = _selected_steps(full_steps)
        base = out_dir / benchmark / noise / f"{sampler}_n{n_train}_{model_name}"
        base.mkdir(parents=True, exist_ok=True)

        # Full evolution only for 1D to keep runtime controlled.
        if int(bench.dim) == 1:
            for st in full_steps:
                Xk, yk = snapshots[st]
                _plot_1d_snapshot(
                    bench=bench,
                    model_name=model_name,
                    X_train=Xk,
                    y_train=yk,
                    step=st,
                    out_path=base / f"full_step_{st:03d}",
                    dpi=dpi,
                    save_svg=save_svg,
                )

        # Simplified snapshots for all dims.
        step_to_path: Dict[int, Path] = {}
        # In progression plots we index by infill count (n=0,1,2,...), not by total train size.
        step_to_ntrain: Dict[int, int] = {int(st): int(st) for st in full_steps}
        for st in simp_steps:
            Xk, yk = snapshots[st]
            target_base = base / f"simplified_step_{st:03d}"
            if int(bench.dim) == 1:
                _plot_1d_snapshot(
                    bench=bench,
                    model_name=model_name,
                    X_train=Xk,
                    y_train=yk,
                    step=st,
                    out_path=target_base,
                    dpi=dpi,
                    save_svg=save_svg,
                )
            elif int(bench.dim) == 2:
                _plot_2d_snapshot(
                    bench=bench,
                    model_name=model_name,
                    X_train=Xk,
                    y_train=yk,
                    step=st,
                    out_path=target_base,
                    dpi=dpi,
                    save_svg=save_svg,
                )
            else:
                _plot_nd_slice_snapshot(
                    bench=bench,
                    model_name=model_name,
                    X_train=Xk,
                    y_train=yk,
                    step=st,
                    out_path=target_base,
                    dpi=dpi,
                    save_svg=save_svg,
                )
            step_to_path[int(st)] = Path(str(target_base) + ".png")

        # Main researcher-facing progression figures: one figure per noise.
        safe_noise = sanitize_filename(noise)
        if int(bench.dim) == 1:
            _plot_1d_noise_progression(
                benchmark=benchmark,
                noise=noise,
                model_name=model_name,
                bench=bench,
                snapshots=snapshots,
                step_to_ntrain=step_to_ntrain,
                out_path=out_dir / benchmark / f"matrix_por_ruido_{benchmark}_{safe_noise}",
                dpi=dpi,
                save_svg=save_svg,
            )
        elif int(bench.dim) == 2 and benchmark.lower() == "branin":
            _plot_2d_noise_progression_3d(
                benchmark=benchmark,
                noise=noise,
                model_name=model_name,
                bench=bench,
                snapshots=snapshots,
                out_path=out_dir / benchmark / f"matrix_por_ruido_{benchmark}_{safe_noise}_3d",
                dpi=dpi,
                save_svg=save_svg,
            )

        # Write small manifest for this GP case.
        (base / "README.md").write_text(
            "\n".join(
                [
                    f"# GP Atlas - {benchmark} | {noise}",
                    "",
                    f"- Modelo: `{model_name}`",
                    f"- Sampler: `{sampler}`",
                    f"- n_train inicial: `{n_train}`",
                    f"- Steps completos (si dim=1): {full_steps}",
                    f"- Steps simplificados: {simp_steps}",
                    f"- Mapeo step->n_infill_visible: {step_to_ntrain}",
                    "",
                    "Nota: reconstruccion basada en dataset inicial regenerado + trayectoria activa.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    # NOTE: combined "all-noises in one matrix" figure intentionally removed.
    # We now export one progression figure per noise type, as requested.
