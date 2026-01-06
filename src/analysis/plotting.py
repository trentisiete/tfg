from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone

# Tus modelos
from src.models.dummy import DummySurrogateRegressor
from src.models.ridge import RidgeSurrogateRegressor
from src.models.pls import PLSSurrogateRegressor
from src.models.gp import GPSurrogateRegressor


# -----------------------------
# Data build
# -----------------------------

@dataclass
class TaskData:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]
    df_filtered: pd.DataFrame  # filas usadas (y != NaN)


def build_X_y_groups(
    df: pd.DataFrame,
    target: str,
    *,
    use_tratamiento_numeric: bool = True,  # replicamos tu setup actual (p=15)
) -> TaskData:
    """
    Construye X,y,groups evitando leakage:
    - X: SOLO info de dieta (pre-experimento)
    - y: target a predecir (por lote)
    - groups: dieta para LODO
    """
    df = df.copy()

    # Filtra NaN en y
    df = df.loc[~df[target].isna()].reset_index(drop=True)

    groups = df["diet_name"].astype(str).to_numpy()
    y = df[target].astype(float).to_numpy()

    # Features continuas (pre-experimento)
    base_cols = [
        "inclusion_pct",
        "Proteína (%)_media",
        "Grasa (%)_media",
        "Fibra (%)_media",
        "Cenizas (%)_media",
        "Carbohidratos (%)_media",
        "ratio_P_C",
        "ratio_P_F",
        "ratio_Fibra_Grasa",
        "TPC_dieta_media",
    ]

    if use_tratamiento_numeric:
        # OJO: "Tratamiento" como número puede meter sesgos (ideal: tratarlo como categórico o no usarlo).
        base_cols = ["Tratamiento"] + base_cols

    # One-hot de byproduct_type (categórica real)
    byp = pd.get_dummies(df["byproduct_type"], prefix="byproduct", drop_first=False)

    Xdf = pd.concat([df[base_cols], byp], axis=1)
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")

    # Imputación simple para X (mediana)
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    X = Xdf.to_numpy(dtype=float)
    feature_names = list(Xdf.columns)

    return TaskData(X=X, y=y, groups=groups, feature_names=feature_names, df_filtered=df)


# -----------------------------
# CV predictions collector (para plots)
# -----------------------------

def collect_lodo_predictions(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """
    Corre LODO y guarda predicciones por muestra:
    fold, group(dieta), y_true, y_pred, std(optional), abs_error
    """
    logo = LeaveOneGroupOut()
    rows = []
    for fold_id, (tr, te) in enumerate(logo.split(X, y, groups)):
        m = clone(model)
        m.fit(X[tr], y[tr])

        mean, std = m.predict_dist(X[te])
        mean = np.asarray(mean).ravel()
        std = None if std is None else np.asarray(std).ravel()

        y_true = np.asarray(y[te]).ravel()
        abs_err = np.abs(y_true - mean)

        for j in range(len(te)):
            rows.append({
                "fold": fold_id,
                "diet": str(groups[te][j]),
                "y_true": float(y_true[j]),
                "y_pred": float(mean[j]),
                "std": None if std is None else float(std[j]),
                "abs_error": float(abs_err[j]),
            })

    return pd.DataFrame(rows)


# -----------------------------
# Plot helpers
# -----------------------------

def _savefig(fig, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_target_by_diet(df: pd.DataFrame, target: str, outdir: Path):
    """
    Boxplot por dieta (3 réplicas) + media como punto.
    """
    diets = sorted(df["diet_name"].unique().tolist())
    data = [df.loc[df["diet_name"] == d, target].dropna().values for d in diets]

    fig = plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.boxplot(data, labels=diets, showfliers=True)
    ax.set_title(f"{target} por dieta (réplicas)")
    ax.set_ylabel(target)
    ax.tick_params(axis="x", rotation=45)

    _savefig(fig, outdir / f"data_boxplot_{_safe_name(target)}.png")


def plot_correlation_matrix(X: np.ndarray, feature_names: list[str], outdir: Path, title: str):
    """
    Correlación de features (útil para justificar PLS vs Ridge, colinealidad, etc.)
    """
    corr = np.corrcoef(X, rowvar=False)
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(corr, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)

    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
    ax.set_yticklabels(feature_names, fontsize=7)

    _savefig(fig, outdir / "features_correlation.png")


def plot_parity(pred_df: pd.DataFrame, outdir: Path, title: str, with_errorbars: bool = False):
    """
    Parity plot: y_pred vs y_true.
    Para GP, opcional: error bars con std.
    """
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = plt.gca()

    x = pred_df["y_true"].to_numpy()
    y = pred_df["y_pred"].to_numpy()

    if with_errorbars and pred_df["std"].notna().any():
        yerr = 1.96 * pred_df["std"].to_numpy()
        ax.errorbar(x, y, yerr=yerr, fmt="o", markersize=4, capsize=2, linewidth=1)
    else:
        ax.plot(x, y, "o", markersize=4)

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi])  # línea y=x

    ax.set_title(title)
    ax.set_xlabel("y real")
    ax.set_ylabel("y predicho")

    _savefig(fig, outdir / f"parity_{_safe_name(title)}.png")


def plot_error_by_diet(pred_df: pd.DataFrame, outdir: Path, title: str):
    """
    MAE por dieta (media de réplicas) para ver qué dietas generalizan peor.
    """
    g = pred_df.groupby("diet")["abs_error"].mean().sort_values(ascending=False)

    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.bar(g.index.tolist(), g.values)
    ax.set_title(title)
    ax.set_ylabel("MAE por dieta")
    ax.tick_params(axis="x", rotation=45)

    _savefig(fig, outdir / f"mae_by_diet_{_safe_name(title)}.png")


def plot_gp_uncertainty(pred_df: pd.DataFrame, outdir: Path, title: str):
    """
    Incertidumbre del GP (std) vs predicción: útil para justificar exploración.
    """
    if pred_df["std"].isna().all():
        return

    fig = plt.figure(figsize=(6.5, 4.5))
    ax = plt.gca()
    ax.plot(pred_df["y_pred"].to_numpy(), pred_df["std"].to_numpy(), "o", markersize=4)
    ax.set_title(title)
    ax.set_xlabel("Predicción (media)")
    ax.set_ylabel("Incertidumbre (std)")

    _savefig(fig, outdir / f"gp_uncertainty_{_safe_name(title)}.png")


def plot_linear_coefficients_ridge(ridge_model: RidgeSurrogateRegressor, feature_names: list[str], outdir: Path, title: str, top_k: int = 12):
    """
    Coeficientes (aprox estandarizados) de Ridge: interpretabilidad.
    """
    # necesita estar fitteado
    pipe = ridge_model.model_
    scaler = pipe.named_steps["scaler"]
    reg = pipe.named_steps["model"]

    coef = np.asarray(reg.coef_).ravel()
    scale = np.asarray(scaler.scale_).ravel()
    coef_std = coef / np.where(scale == 0, 1.0, scale)

    idx = np.argsort(np.abs(coef_std))[::-1][:top_k]
    names = [feature_names[i] for i in idx]
    vals = coef_std[idx]

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.bar(names, vals)
    ax.set_title(title)
    ax.set_ylabel("Coeficiente (std approx)")
    ax.tick_params(axis="x", rotation=45)

    _savefig(fig, outdir / f"ridge_coefs_{_safe_name(title)}.png")


def plot_pls_coefficients(pls_model: PLSSurrogateRegressor, feature_names: list[str], outdir: Path, title: str, top_k: int = 12):
    """
    Coeficientes PLS: interpretabilidad (qué features empujan Y).
    """
    # Access the PLS model inside the pipeline
    # Assuming the pipeline step is named "model" as in your PLSSurrogateRegressor implementation
    m = pls_model.model_.named_steps["model"]

    coef = np.asarray(m.coef_).ravel()  # (n_features,)
    idx = np.argsort(np.abs(coef))[::-1][:top_k]
    names = [feature_names[i] for i in idx]
    vals = coef[idx]

    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.bar(names, vals)
    ax.set_title(title)
    ax.set_ylabel("Coeficiente PLS")
    ax.tick_params(axis="x", rotation=45)

    _savefig(fig, outdir / f"pls_coefs_{_safe_name(title)}.png")


# -----------------------------
# Model selection + next investigation point
# -----------------------------

def pick_best_model_by_micro_mae(results: dict) -> str:
    """
    Espera el formato de tu evaluate_model:
    out[model]["results"]["summary"]["micro"]["mae"]
    """
    best_name = None
    best_mae = float("inf")
    for name, payload in results.items():
        if name == "metadata":
            continue
        micro = payload["results"]["summary"]["micro"]
        mae = float(micro["mae"])
        if mae < best_mae:
            best_mae = mae
            best_name = name
    return best_name


def suggest_next_point_gp_ucb(
    gp: GPSurrogateRegressor,
    Xdf: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    beta: float = 1.0,
    n_candidates: int = 2000,
    random_state: int = 0,
) -> dict:
    """
    Propone un 'siguiente punto para investigar' (exploración) usando UCB:
      UCB(x) = mean(x) + beta * std(x)

    - Entrena GP en todo el dataset.
    - Genera candidatos aleatorios dentro del rango de las features actuales.
    - Devuelve el mejor candidato por UCB (y su mean/std).

    Nota: esto opera en el espacio de features nutricionales, no en recetas físicas;
    en el TFG puedes presentarlo como 'siguiente región del espacio a explorar'.
    """
    rng = np.random.default_rng(random_state)

    # Fit GP full
    gp_full = clone(gp)
    gp_full.fit(Xdf.to_numpy(dtype=float), y)

    # Candidate sampling (uniform en min/max por feature)
    Xmin = Xdf.min(axis=0).to_numpy(dtype=float)
    Xmax = Xdf.max(axis=0).to_numpy(dtype=float)
    Xcand = rng.uniform(Xmin, Xmax, size=(n_candidates, Xdf.shape[1]))

    mean, std = gp_full.predict_dist(Xcand)
    mean = np.asarray(mean).ravel()
    std = np.asarray(std).ravel()
    ucb = mean + beta * std

    j = int(np.argmax(ucb))
    cand = Xcand[j]

    return {
        "best_ucb": float(ucb[j]),
        "pred_mean": float(mean[j]),
        "pred_std": float(std[j]),
        "beta": float(beta),
        "candidate_features": {col: float(cand[i]) for i, col in enumerate(Xdf.columns)},
    }


# -----------------------------
# Orchestrator
# -----------------------------

def generate_all_figures_for_target(
    df: pd.DataFrame,
    target: str,
    eval_out: dict,
    task: TaskData,
    outdir: Path,
    models: dict = None,
):
    """
    Genera un set completo de figuras para un target:
    - data: boxplot por dieta
    - features: correlación
    - modelos: parity + error por dieta (best model y GP)
    - interpretabilidad: coeficientes Ridge/PLS (fit full)
    - GP: incertidumbre y coverage interpretabilidad
    """
    outdir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = _make_models()

    # 1) Data overview
    plot_target_by_diet(task.df_filtered, target, outdir)

    # 2) Feature correlations
    plot_correlation_matrix(task.X, task.feature_names, outdir, title=f"Correlación de features | {target}")

    # 3) Predicciones CV para best model y GP (para gráficas)
    best_model_name = pick_best_model_by_micro_mae(eval_out) #TODO: This functions needs to be imported from eval.py

    # A) Best model parity + error by diet
    best_model = models[best_model_name]
    pred_best = collect_lodo_predictions(best_model, task.X, task.y, task.groups) #TODO: This functions needs to be imported from eval.py

    plot_parity(pred_best, outdir, title=f"{best_model_name} | {target}")
    plot_error_by_diet(pred_best, outdir, title=f"{best_model_name} | {target}")

    # B) GP parity + error + uncertainty
    pred_gp = collect_lodo_predictions(models["GP"], task.X, task.y, task.groups) #TODO: This functions needs to be imported from eval.py

    plot_parity(pred_gp, outdir, title=f"GP | {target}", with_errorbars=True)
    plot_error_by_diet(pred_gp, outdir, title=f"GP | {target}")
    plot_gp_uncertainty(pred_gp, outdir, title=f"GP incertidumbre | {target}")

    # 4) Interpretabilidad (fit full)
    ridge = clone(models["Ridge"]).fit(task.X, task.y)
    pls = clone(models["PLS"]).fit(task.X, task.y)

    # Coefs
    plot_linear_coefficients_ridge(ridge, task.feature_names, outdir, title=f"Ridge coefs | {target}")
    plot_pls_coefficients(pls, task.feature_names, outdir, title=f"PLS coefs | {target}")

    # 5) Sugerencia "next point"
    #    (para el TFG: justificar que GP da incertidumbre y por eso sirve para exploración)
    Xdf = pd.DataFrame(task.X, columns=task.feature_names)

    suggestion = suggest_next_point_gp_ucb(models["GP"], Xdf, task.y, task.groups, beta=1.0) #TODO: This functions needs to be imported from gp.py

    # Guarda sugerencia como CSV/JSON sencillo
    sug_path = outdir / f"next_point_ucb_{_safe_name(target)}.csv"
    pd.DataFrame([suggestion["candidate_features"] | {
        "pred_mean": suggestion["pred_mean"],
        "pred_std": suggestion["pred_std"],
        "best_ucb": suggestion["best_ucb"],
        "beta": suggestion["beta"],
    }]).to_csv(sug_path, index=False)


def _safe_name(s: str) -> str: # TODO: move to utils.paths.py

    # Replace non-alphanumeric characters with underscores, but avoid double underscores
    # Also handle unicode characters if necessary, but for filenames simple ascii is safer
    import re
    s = re.sub(r'[^a-zA-Z0-9]', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_').lower()[:50]


def _make_models() -> dict:
    return {
        "Dummy": DummySurrogateRegressor(strategy="mean"),
        "Ridge": RidgeSurrogateRegressor(alpha=1.0, fit_intercept=True),
        "PLS": PLSSurrogateRegressor(n_components=2, scale=True),
        "GP": GPSurrogateRegressor(),
    }
