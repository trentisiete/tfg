from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone


# ----------------------------
# Utils
# ----------------------------

def safe_name(s: str, maxlen: int = 80) -> str:
    s2 = "".join(ch if ch.isalnum() else "_" for ch in str(s)).strip("_").lower()
    return s2[:maxlen] if len(s2) > maxlen else s2


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def savefig(fig, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def _collect_folds_df(tuning_results: dict) -> pd.DataFrame:
    """
    Convierte resultados nested por modelo a un DataFrame long-form.
    """
    rows = []
    for model_name, payload in tuning_results.items():
        if model_name == "metadata":
            continue
        folds = payload.get("folds", [])
        for f in folds:
            m = f.get("metrics", {})
            rows.append({
                "model": model_name,
                "fold": f.get("fold"),
                "diet": f.get("diet"),
                "mae": m.get("mae"),
                "rmse": m.get("rmse"),
                "coverage95": m.get("coverage95"),
                "inner_best_score": f.get("inner_best_score"),
                "params": f.get("params", {}),
            })
    df = pd.DataFrame(rows)
    return df


def _collect_summary_df(tuning_results: dict) -> pd.DataFrame:
    """
    Extrae summary por modelo (macro mean/std) a un DF.
    """
    rows = []
    for model_name, payload in tuning_results.items():
        if model_name == "metadata":
            continue
        summ = payload.get("summary", {}) or payload.get("results", {}).get("summary", {})
        macro = summ.get("macro", {})

        # soporta dos formatos: el mío (mae_mean) o el tuyo (mae:{mean,std})
        if "mae_mean" in macro:
            mae_mean = macro.get("mae_mean")
            mae_std = macro.get("mae_std")
            rmse_mean = macro.get("rmse_mean")
            rmse_std = macro.get("rmse_std")
            cov_mean = macro.get("coverage95_mean")
            cov_std = macro.get("coverage95_std")
        else:
            mae_mean = (macro.get("mae") or {}).get("mean")
            mae_std = (macro.get("mae") or {}).get("std")
            rmse_mean = (macro.get("rmse") or {}).get("mean")
            rmse_std = (macro.get("rmse") or {}).get("std")
            cov_mean = (macro.get("coverage95") or {}).get("mean")
            cov_std = (macro.get("coverage95") or {}).get("std")

        rows.append({
            "model": model_name,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "coverage95_mean": cov_mean,
            "coverage95_std": cov_std,
        })

    return pd.DataFrame(rows).sort_values("mae_mean", ascending=True)


# ----------------------------
# 1) Comparación de modelos
# ----------------------------

def plot_model_ranking(summary_df: pd.DataFrame, outdir: Path, title: str):
    """
    Barplot de MAE mean por modelo (macro).
    """
    df = summary_df.copy().sort_values("mae_mean", ascending=True)
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()

    x = df["model"].tolist()
    y = df["mae_mean"].to_numpy(dtype=float)
    yerr = df["mae_std"].to_numpy(dtype=float)

    ax.bar(x, y, yerr=yerr)
    ax.set_title(title)
    ax.set_ylabel("MAE (macro mean ± std)")
    ax.tick_params(axis="x", rotation=30)

    savefig(fig, outdir / "ranking_mae.png")


def plot_metric_table(summary_df: pd.DataFrame, outdir: Path):
    """
    Guarda una tabla CSV con métricas resumen por modelo.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(outdir / "summary_by_model.csv", index=False)


def plot_metrics_multi(summary_df: pd.DataFrame, outdir: Path, title: str):
    """
    Barplots separados para MAE y RMSE y Coverage (si existe).
    """
    df = summary_df.copy().sort_values("mae_mean", ascending=True)

    # MAE
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()
    ax.bar(df["model"], df["mae_mean"], yerr=df["mae_std"])
    ax.set_title(f"{title} | MAE")
    ax.set_ylabel("MAE (macro)")
    ax.tick_params(axis="x", rotation=30)
    savefig(fig, outdir / "metric_mae.png")

    # RMSE
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()
    ax.bar(df["model"], df["rmse_mean"], yerr=df["rmse_std"])
    ax.set_title(f"{title} | RMSE")
    ax.set_ylabel("RMSE (macro)")
    ax.tick_params(axis="x", rotation=30)
    savefig(fig, outdir / "metric_rmse.png")

    # Coverage (si hay)
    if df["coverage95_mean"].notna().any():
        fig = plt.figure(figsize=(7, 4))
        ax = plt.gca()
        ax.bar(df["model"], df["coverage95_mean"], yerr=df["coverage95_std"])
        ax.set_title(f"{title} | Coverage95")
        ax.set_ylabel("Coverage95 (macro)")
        ax.tick_params(axis="x", rotation=30)
        savefig(fig, outdir / "metric_coverage95.png")


# ----------------------------
# 2) Diagnóstico por dietas (folds)
# ----------------------------

def plot_fold_mae_boxplot(folds_df: pd.DataFrame, outdir: Path, title: str):
    """
    Boxplot MAE por modelo (distribución por dietas).
    """
    fig = plt.figure(figsize=(8, 4))
    ax = plt.gca()

    models = sorted(folds_df["model"].unique().tolist())
    data = [folds_df.loc[folds_df["model"] == m, "mae"].dropna().to_numpy(dtype=float) for m in models]
    ax.boxplot(data, labels=models, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel("MAE por fold (dieta)")
    ax.tick_params(axis="x", rotation=30)
    savefig(fig, outdir / "fold_mae_boxplot.png")


def plot_mae_by_diet_heatmap(folds_df: pd.DataFrame, outdir: Path, title: str):
    """
    Heatmap (modelos x dietas) con MAE por fold.
    """
    pivot = folds_df.pivot_table(index="model", columns="diet", values="mae", aggfunc="mean")
    pivot = pivot.loc[sorted(pivot.index)]

    fig = plt.figure(figsize=(max(8, 0.6 * pivot.shape[1]), 4))
    ax = plt.gca()
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_ylabel("Modelo")
    ax.set_xlabel("Dieta (fold)")

    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right", fontsize=8)

    savefig(fig, outdir / "heatmap_mae_model_by_diet.png")


def plot_worst_diets(folds_df: pd.DataFrame, outdir: Path, top_k: int = 8):
    """
    Muestra dietas más difíciles (promediando MAE sobre modelos) y por mejor modelo.
    """
    # promedio por dieta y modelo
    g = folds_df.groupby(["diet", "model"])["mae"].mean().reset_index()

    # para cada dieta, cuál es el mejor modelo (menor mae)
    best_per_diet = g.sort_values("mae", ascending=True).groupby("diet").first().reset_index()
    best_per_diet = best_per_diet.sort_values("mae", ascending=False).head(top_k)

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.bar(best_per_diet["diet"], best_per_diet["mae"])
    ax.set_title(f"Top-{top_k} dietas más difíciles (MAE del mejor modelo por dieta)")
    ax.set_ylabel("MAE (menor es mejor, esto muestra dificultad)")
    ax.tick_params(axis="x", rotation=45)
    savefig(fig, outdir / "worst_diets_best_model_mae.png")

    best_per_diet.to_csv(outdir / "worst_diets_table.csv", index=False)


# ----------------------------
# 3) Estabilidad y análisis de hiperparámetros
# ----------------------------

def _explode_params(folds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expande el dict 'params' de cada fold a columnas para analizar frecuencia.
    """
    rows = []
    for _, r in folds_df.iterrows():
        p = r["params"] if isinstance(r["params"], dict) else {}
        base = {k: r[k] for k in ["model", "fold", "diet", "mae", "rmse", "coverage95", "inner_best_score"]}
        for kk, vv in p.items():
            base[f"param__{kk}"] = vv
        rows.append(base)
    return pd.DataFrame(rows)


def plot_param_frequency(folds_df: pd.DataFrame, outdir: Path):
    """
    Para cada modelo, muestra qué hiperparámetros se seleccionan más frecuentemente.
    (Si param es numérico, hace histograma; si es categórico/bool, barras)
    """
    dfp = _explode_params(folds_df)

    for model in sorted(dfp["model"].unique().tolist()):
        sub = dfp[dfp["model"] == model]
        param_cols = [c for c in sub.columns if c.startswith("param__")]
        if not param_cols:
            continue

        mdir = outdir / "params" / safe_name(model)
        mdir.mkdir(parents=True, exist_ok=True)

        for pc in param_cols:
            vals = sub[pc].dropna()
            if vals.empty:
                continue

            # decide tipo
            if np.issubdtype(vals.dtype, np.number):
                fig = plt.figure(figsize=(6, 4))
                ax = plt.gca()
                ax.hist(vals.to_numpy(dtype=float), bins=10)
                ax.set_title(f"{model} | {pc} (hist selección)")
                ax.set_xlabel(pc)
                ax.set_ylabel("Frecuencia")
                savefig(fig, mdir / f"{safe_name(pc)}_hist.png")
            else:
                counts = vals.astype(str).value_counts()
                fig = plt.figure(figsize=(7, 4))
                ax = plt.gca()
                ax.bar(counts.index.tolist(), counts.values)
                ax.set_title(f"{model} | {pc} (frecuencia selección)")
                ax.set_xlabel(pc)
                ax.set_ylabel("Frecuencia")
                ax.tick_params(axis="x", rotation=30)
                savefig(fig, mdir / f"{safe_name(pc)}_bar.png")


def plot_param_vs_mae(folds_df: pd.DataFrame, outdir: Path, only_numeric: bool = True):
    """
    Scatter: valor del parámetro seleccionado vs MAE outer.
    Útil para ver si hay tendencia (p.ej. alpha alto -> peor/better).
    """
    dfp = _explode_params(folds_df)

    for model in sorted(dfp["model"].unique().tolist()):
        sub = dfp[dfp["model"] == model]
        param_cols = [c for c in sub.columns if c.startswith("param__")]
        if not param_cols:
            continue

        mdir = outdir / "param_vs_mae" / safe_name(model)
        mdir.mkdir(parents=True, exist_ok=True)

        for pc in param_cols:
            vals = sub[pc].dropna()
            if vals.empty:
                continue
            if only_numeric and not np.issubdtype(vals.dtype, np.number):
                continue

            fig = plt.figure(figsize=(6, 4))
            ax = plt.gca()
            ax.plot(sub[pc].to_numpy(dtype=float), sub["mae"].to_numpy(dtype=float), "o", markersize=4)
            ax.set_title(f"{model} | {pc} vs MAE (outer)")
            ax.set_xlabel(pc)
            ax.set_ylabel("MAE (outer)")
            savefig(fig, mdir / f"{safe_name(pc)}_vs_mae.png")


# ----------------------------
# 4) “Cómo serían las predicciones” con modelo final
# ----------------------------

def collect_lodo_predictions_fixed_params(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray, fixed_params: dict) -> pd.DataFrame:
    """
    Predicciones LODO usando unos params fijos (los “finales”).
    Esto simula mejor la realidad que entrenar y evaluar en el mismo dataset.
    """
    logo = LeaveOneGroupOut()
    rows = []

    for fold_id, (tr, te) in enumerate(logo.split(X, y, groups)):
        m = clone(model).set_params(**fixed_params)
        m.fit(X[tr], y[tr])

        mean, std = m.predict_dist(X[te])
        mean = np.asarray(mean).ravel()
        std = None if std is None else np.asarray(std).ravel()

        y_true = np.asarray(y[te]).ravel()

        for j in range(len(te)):
            rows.append({
                "fold": fold_id,
                "diet": str(groups[te][j]),
                "y_true": float(y_true[j]),
                "y_pred": float(mean[j]),
                "std": None if std is None else float(std[j]),
                "abs_error": float(abs(y_true[j] - mean[j])),
            })

    return pd.DataFrame(rows)


def plot_parity(pred_df: pd.DataFrame, outdir: Path, title: str, with_errorbars: bool = False):
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = plt.gca()

    x = pred_df["y_true"].to_numpy(dtype=float)
    y = pred_df["y_pred"].to_numpy(dtype=float)

    if with_errorbars and pred_df["std"].notna().any():
        yerr = 1.96 * pred_df["std"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=yerr, fmt="o", markersize=4, capsize=2, linewidth=1)
    else:
        ax.plot(x, y, "o", markersize=4)

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi])

    ax.set_title(title)
    ax.set_xlabel("y real")
    ax.set_ylabel("y predicho")
    savefig(fig, outdir / f"parity_{safe_name(title)}.png")


def plot_error_by_diet(pred_df: pd.DataFrame, outdir: Path, title: str):
    g = pred_df.groupby("diet")["abs_error"].mean().sort_values(ascending=False)
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.bar(g.index.tolist(), g.values)
    ax.set_title(title)
    ax.set_ylabel("MAE por dieta")
    ax.tick_params(axis="x", rotation=45)
    savefig(fig, outdir / f"mae_by_diet_{safe_name(title)}.png")


# ----------------------------
# 5) Orquestador general
# ----------------------------

def visualize_tuning_stage(
    tuning_results: dict,
    outdir: str | Path,
    *,
    title: str = "Tuning exhaustivo (nested LODO)",
):
    """
    Genera un paquete de figuras para analizar tuning:
    - ranking modelos (MAE)
    - métricas MAE/RMSE/Coverage
    - boxplot por dietas y heatmap dietas x modelos
    - dietas más difíciles
    - estabilidad de hiperparámetros y param vs mae
    - exporta tabla summary
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    folds_df = _collect_folds_df(tuning_results)
    summary_df = _collect_summary_df(tuning_results)

    # 1) comparación global
    plot_model_ranking(summary_df, outdir, title=title)
    plot_metrics_multi(summary_df, outdir, title=title)
    plot_metric_table(summary_df, outdir)

    # 2) por dietas
    plot_fold_mae_boxplot(folds_df, outdir, title=f"{title} | MAE por dieta")
    plot_mae_by_diet_heatmap(folds_df, outdir, title=f"{title} | MAE heatmap")
    plot_worst_diets(folds_df, outdir, top_k=8)

    # 3) hiperparámetros
    plot_param_frequency(folds_df, outdir)
    plot_param_vs_mae(folds_df, outdir)

    # guardado DF para inspección
    folds_df.to_csv(outdir / "folds_long.csv", index=False)
