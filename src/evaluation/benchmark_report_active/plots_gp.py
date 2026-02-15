from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

from src.benchmarks import generate_benchmark_dataset, get_benchmark
from src.models.gp import GPSurrogateRegressor

from .styling import save_figure


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
        n_groups=int(metadata.get("n_groups", 5)),
        seed=seed,
    )
    bench = get_benchmark(benchmark)

    x_app = np.vstack([_parse_x_next_cell(v) for v in block["x_next"]]) if len(block) else np.empty((0, ds.X_train.shape[1]))
    y_app = block["y_next"].to_numpy(dtype=float) if "y_next" in block else np.empty((0,))

    snapshots: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for step in sorted(block["step"].astype(int).unique().tolist()):
        k = int(step)
        Xk = np.vstack([ds.X_train, x_app[:k]]) if len(x_app) >= k else ds.X_train.copy()
        yk = np.concatenate([ds.y_train.ravel(), y_app[:k]]) if len(y_app) >= k else ds.y_train.ravel().copy()
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
    model = _build_gp_model(model_name)
    model.fit(X_train, y_train)
    lb, ub = bench.bounds[0]
    xg = np.linspace(lb, ub, 450).reshape(-1, 1)
    y_true = bench(xg)
    y_pred, y_std = model.predict_dist(xg)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.fill_between(xg.ravel(), y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.20, color="#fdb462", label="±2σ")
    ax.fill_between(xg.ravel(), y_pred - y_std, y_pred + y_std, alpha=0.35, color="#fb8072", label="±1σ")
    ax.plot(xg.ravel(), y_true, color="#1f78b4", linewidth=2.0, label="f(x) real")
    ax.plot(xg.ravel(), y_pred, color="#d95f02", linewidth=2.0, label=f"{model_name} prediccion")
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

    # Keep one representative config per benchmark/noise: best MAE among GP configs.
    reps = (
        gp_final.sort_values("mae")
        .groupby(["benchmark", "noise"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    selected_benchmarks = reps["benchmark"].drop_duplicates().tolist()[: int(max_gp_benchmarks)]
    reps = reps[reps["benchmark"].isin(selected_benchmarks)].copy()

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
        for st in simp_steps:
            Xk, yk = snapshots[st]
            if int(bench.dim) == 1:
                _plot_1d_snapshot(
                    bench=bench,
                    model_name=model_name,
                    X_train=Xk,
                    y_train=yk,
                    step=st,
                    out_path=base / f"simplified_step_{st:03d}",
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
                    out_path=base / f"simplified_step_{st:03d}",
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
                    out_path=base / f"simplified_step_{st:03d}",
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
                    "",
                    "Nota: reconstruccion basada en dataset inicial regenerado + trayectoria activa.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
