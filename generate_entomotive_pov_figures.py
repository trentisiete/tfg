from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, r2_score

from src.analysis.tuning_reporter import ModelReconstructor
from src.configs.tuning_specs import FEATURE_COLS_FULL, FEATURE_COLS_REDUCED, TARGET_MAP
from src.utils.paths import ENTOMOTIVE_DATA_DIR, LOGS_DIR, OUTPUTS_DIR, PROJECT_ROOT
from src.utils.tools import slugify


DATA_FILE = ENTOMOTIVE_DATA_DIR / "productivity_hermetia_lote.csv"
LOG_ROOT = LOGS_DIR / "tuning" / "TFG_MAIN_real_case_hermetia_no_tpc_tuning"
REPORT_DIR = OUTPUTS_DIR / "reports" / "TFG_MAIN_real_case_audit_no_tpc"
OUTPUT_DIR = OUTPUTS_DIR / "plots" / "TFG_MAIN_real_case_results_pov_ei_no_tpc"
METRICS_FILE = REPORT_DIR / "active_model_metrics.csv"
RANKING_FILE = REPORT_DIR / "observed_equal_weight_ranking.csv"

FEATURE_MODES = {
    "REDUCED_FEATURES": FEATURE_COLS_REDUCED,
    "FULL_FEATURES": FEATURE_COLS_FULL,
}

TARGET_ORDER = ["FCR", "Quitina", "Proteina"]
TARGET_LABELS = {
    "FCR": "FCR",
    "Quitina": "Quitina",
    "Proteina": "Proteina",
}

TARGET_UNITS = {
    "FCR": "FCR",
    "Quitina": "Quitina (%)",
    "Proteina": "Proteina (%)",
}

TARGET_DIRECTIONS = {
    "FCR": "min",
    "Quitina": "max",
    "Proteina": "max",
}

TARGET_MEAN_COLS = {
    "FCR": "fcr_mean",
    "Quitina": "quitina_mean",
    "Proteina": "proteina_mean",
}

MODEL_LABELS = {
    "Dummy": "Dummy",
    "GP_Linear": "Lineal",
    "GP_RBF_NoARD": "RBF",
    "GP_Matern32_NoARD": "Matern 3/2",
    "GP_Matern52_NoARD": "Matern 5/2",
    "GP_Compuesto_NoARD": "Compuesto",
    "GP_RBF_ARD": "ARD selectivo",
    "GP_Matern32_ARD": "ARD selectivo",
    "GP_Matern52_ARD": "ARD selectivo",
}

BYPRODUCT_COLORS = {
    "control": "#4C78A8",
    "hoja": "#59A14F",
    "orujo": "#E15759",
    "quinoa": "#F28E2B",
}


def style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def build_X_y_groups(df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    data = df.copy()
    data = data.loc[~data[target_col].isna()].reset_index(drop=True)

    groups = data["diet_name"].astype(str).to_numpy()
    y = data[target_col].astype(float).to_numpy()

    byp = pd.get_dummies(data["byproduct_type"], prefix="byproduct", drop_first=False)
    Xdf = data[feature_cols].copy()
    Xdf = pd.concat([Xdf, byp], axis=1)
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    return Xdf.to_numpy(dtype=float), y, groups, Xdf.columns.tolist(), data


def load_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS_FILE)
    metrics["target"] = pd.Categorical(metrics["target"], TARGET_ORDER, ordered=True)
    return metrics.sort_values(["target", "feature_mode", "model"]).reset_index(drop=True)


def select_best_gp_cases(metrics: pd.DataFrame) -> pd.DataFrame:
    gp = metrics[(metrics["model"].str.startswith("GP"))].copy()
    idx = gp.groupby("target", observed=True)["mae_macro_mean"].idxmin()
    best = gp.loc[idx].sort_values("target").reset_index(drop=True)
    best.to_csv(OUTPUT_DIR / "pov_best_gp_cases.csv", index=False)
    return best


def select_model_row(metrics: pd.DataFrame, target: str, model_filter) -> pd.Series | None:
    block = metrics[(metrics["target"].astype(str) == target)].copy()
    block = block[block.apply(model_filter, axis=1)]
    if block.empty:
        return None
    return block.loc[block["mae_macro_mean"].idxmin()]


def plot_lodo_generalization(metrics: pd.DataFrame):
    rows = []
    simple_models = {
        "GP_Linear",
        "GP_RBF_NoARD",
        "GP_Matern32_NoARD",
        "GP_Matern52_NoARD",
    }

    for target in TARGET_ORDER:
        dummy = select_model_row(metrics, target, lambda r: r["model"] == "Dummy")
        simple = select_model_row(metrics, target, lambda r: r["model"] in simple_models)
        comp = select_model_row(metrics, target, lambda r: r["model"] == "GP_Compuesto_NoARD")
        ard = select_model_row(
            metrics,
            target,
            lambda r: str(r["model"]).endswith("_ARD") and "NoARD" not in str(r["model"]),
        )

        for label, row in [
            ("Dummy", dummy),
            ("Mejor GP simple", simple),
            ("GP compuesto", comp),
            ("ARD selectivo", ard),
        ]:
            if row is None:
                continue
            rows.append(
                {
                    "target": target,
                    "comparison": label,
                    "model": row["model"],
                    "feature_mode": row["feature_mode"],
                    "mae": row["mae_macro_mean"],
                    "mae_std": row["mae_macro_std"],
                }
            )

    df = pd.DataFrame(rows)
    dummy_mae = (
        df[df["comparison"] == "Dummy"]
        .set_index("target")["mae"]
        .to_dict()
    )
    df["relative_mae"] = df.apply(lambda r: r["mae"] / dummy_mae[str(r["target"])], axis=1)
    df.to_csv(OUTPUT_DIR / "fig_01_lodo_model_comparison_data.csv", index=False)

    palette = {
        "Dummy": "#8C8C8C",
        "Mejor GP simple": "#4C78A8",
        "GP compuesto": "#2A9D8F",
        "ARD selectivo": "#F28E2B",
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    sns.barplot(
        data=df,
        x="target",
        y="relative_mae",
        hue="comparison",
        palette=palette,
        ax=ax,
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1, alpha=0.8)
    ax.set_title("Generalizacion LODO: rendimiento relativo frente al baseline Dummy")
    ax.set_xlabel("Target evaluado sobre dieta no vista")
    ax.set_ylabel("MAE relativo al Dummy (menor es mejor)")
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=True)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 1.24)

    for target_idx, target in enumerate(TARGET_ORDER):
        target_block = df[df["target"].astype(str) == target]
        d = target_block[target_block["comparison"] == "Dummy"]["mae"]
        c = target_block[target_block["comparison"] == "GP compuesto"]["mae"]
        if not d.empty and not c.empty:
            improvement = 100 * (float(d.iloc[0]) - float(c.iloc[0])) / float(d.iloc[0])
            y = 1.07
            ax.text(
                target_idx,
                y,
                f"Compuesto mejora\n{improvement:.1f}% vs Dummy",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_01_lodo_generalization_vs_dummy.png", bbox_inches="tight")
    plt.close(fig)


def plot_kernel_family_mae(metrics: pd.DataFrame):
    models = [
        "Dummy",
        "GP_Linear",
        "GP_RBF_NoARD",
        "GP_Matern32_NoARD",
        "GP_Matern52_NoARD",
        "GP_Compuesto_NoARD",
    ]
    block = metrics[
        (metrics["feature_mode"] == "FULL_FEATURES")
        & (metrics["model"].isin(models))
    ].copy()
    block["model_label"] = block["model"].map(MODEL_LABELS)
    block["model_label"] = pd.Categorical(
        block["model_label"],
        ["Dummy", "Lineal", "RBF", "Matern 3/2", "Matern 5/2", "Compuesto"],
        ordered=True,
    )
    block.to_csv(OUTPUT_DIR / "fig_04_kernel_family_mae_data.csv", index=False)

    g = sns.catplot(
        data=block,
        kind="bar",
        x="model_label",
        y="mae_macro_mean",
        col="target",
        col_order=TARGET_ORDER,
        palette=["#8C8C8C", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#2A9D8F"],
        sharey=False,
        height=4.1,
        aspect=0.95,
        edgecolor="#333333",
        linewidth=0.5,
    )
    g.set_axis_labels("", "MAE macro LODO")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    g.fig.suptitle(
        "Expresividad geometrica: comparacion de familias de kernel (FULL_FEATURES)",
        y=1.08,
        fontweight="bold",
    )
    g.fig.savefig(OUTPUT_DIR / "fig_04_kernel_family_mae.png", bbox_inches="tight")
    plt.close(g.fig)


def reconstruct_predictions(best_cases: pd.DataFrame) -> pd.DataFrame:
    cached_path = OUTPUT_DIR / "pov_reconstructed_lodo_predictions.csv"
    if cached_path.exists():
        return pd.read_csv(cached_path)

    rows = []
    raw = pd.read_csv(DATA_FILE)

    for _, case in best_cases.iterrows():
        target = str(case["target"])
        target_col = str(case["target_col"])
        feature_mode = str(case["feature_mode"])
        model_name = str(case["model"])
        feature_cols = FEATURE_MODES[feature_mode]
        target_slug = slugify(target)

        X, y, groups, feature_names, data = build_X_y_groups(raw, target_col, feature_cols)
        tuning_path = LOG_ROOT / feature_mode / target_slug / f"{target_slug}_{model_name.lower()}_tuning.json"
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))

        for fold in tuning["folds"]:
            diet = str(fold["diet"])
            test_idx = np.where(groups == diet)[0]
            train_idx = np.where(groups != diet)[0]
            if len(test_idx) == 0:
                continue

            reconstructor = ModelReconstructor(X[train_idx], y[train_idx], feature_names)
            model = reconstructor.retrain_model(model_name, fold["params"])
            pred, std = model.predict_dist(X[test_idx])
            if std is None:
                std = np.full_like(pred, np.nan, dtype=float)

            for local_pos, idx in enumerate(test_idx):
                y_true = float(y[idx])
                y_pred = float(pred[local_pos])
                y_std = float(std[local_pos])
                rows.append(
                    {
                        "target": target,
                        "target_col": target_col,
                        "feature_mode": feature_mode,
                        "model": model_name,
                        "diet_name": data.loc[idx, "diet_name"],
                        "byproduct_type": data.loc[idx, "byproduct_type"],
                        "inclusion_pct": data.loc[idx, "inclusion_pct"],
                        "replica": data.loc[idx, "Replica"] if "Replica" in data.columns else local_pos + 1,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "y_std": y_std,
                        "abs_error": abs(y_true - y_pred),
                        "lower95": y_pred - 1.96 * y_std if np.isfinite(y_std) else np.nan,
                        "upper95": y_pred + 1.96 * y_std if np.isfinite(y_std) else np.nan,
                    }
                )

    pred_df = pd.DataFrame(rows)
    pred_df["inside95"] = (
        (pred_df["y_true"] >= pred_df["lower95"])
        & (pred_df["y_true"] <= pred_df["upper95"])
    )
    pred_df.to_csv(cached_path, index=False)
    return pred_df


def plot_error_by_diet(pred_df: pd.DataFrame):
    raw_error = (
        pred_df.groupby(["target", "diet_name"], observed=True)["abs_error"]
        .mean()
        .reset_index()
    )
    ranges = pred_df.groupby("target", observed=True)["y_true"].agg(lambda s: s.max() - s.min()).to_dict()
    raw_error["normalized_error"] = raw_error.apply(
        lambda r: r["abs_error"] / ranges[str(r["target"])],
        axis=1,
    )
    raw_error.to_csv(OUTPUT_DIR / "fig_02_error_by_left_out_diet_data.csv", index=False)

    pivot_norm = raw_error.pivot(index="diet_name", columns="target", values="normalized_error")
    pivot_raw = raw_error.pivot(index="diet_name", columns="target", values="abs_error")
    pivot_norm = pivot_norm.reindex(columns=TARGET_ORDER)
    pivot_raw = pivot_raw.reindex(columns=TARGET_ORDER)
    diet_order = pivot_norm.mean(axis=1).sort_values(ascending=False).index.tolist()
    pivot_norm = pivot_norm.loc[diet_order]
    pivot_raw = pivot_raw.loc[diet_order]

    annot = pivot_raw.copy()
    for col in annot.columns:
        annot[col] = annot[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    sns.heatmap(
        pivot_norm,
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Error normalizado por rango del target"},
        ax=ax,
    )
    ax.set_title(
        "Dificultad LODO por dieta dejada fuera\n"
        "Color = error normalizado por rango; texto = MAE medio en unidades del target",
        pad=12,
    )
    ax.set_xlabel("Target")
    ax.set_ylabel("Dieta retenida como test")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_02_error_by_left_out_diet.png", bbox_inches="tight")
    plt.close(fig)


def plot_best_gp_parity(pred_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, target in zip(axes, TARGET_ORDER):
        block = pred_df[pred_df["target"] == target].copy()
        vmin = min(block["y_true"].min(), block["y_pred"].min())
        vmax = max(block["y_true"].max(), block["y_pred"].max())
        pad = (vmax - vmin) * 0.08
        vmin -= pad
        vmax += pad

        sns.scatterplot(
            data=block,
            x="y_true",
            y="y_pred",
            hue="byproduct_type",
            palette=BYPRODUCT_COLORS,
            s=62,
            edgecolor="#222222",
            linewidth=0.5,
            ax=ax,
        )
        ax.plot([vmin, vmax], [vmin, vmax], color="#333333", linestyle="--", linewidth=1)
        mae = mean_absolute_error(block["y_true"], block["y_pred"])
        r2 = r2_score(block["y_true"], block["y_pred"])
        case = block[["model", "feature_mode"]].iloc[0]
        ax.set_title(f"{target}\n{case['model']} / {case['feature_mode']}")
        ax.set_xlabel("Observado")
        ax.set_ylabel("Predicho")
        ax.text(
            0.04,
            0.96,
            f"MAE global={mae:.2f}\nR2 global={r2:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
        )
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        if ax is not axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Subproducto", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.suptitle("Prediccion LODO del mejor GP: observado vs predicho", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_03_best_gp_parity_by_target.png", bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_vs_error(pred_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    rows = []
    for ax, target in zip(axes, TARGET_ORDER):
        block = pred_df[pred_df["target"] == target].copy()
        sns.scatterplot(
            data=block,
            x="y_std",
            y="abs_error",
            hue="byproduct_type",
            palette=BYPRODUCT_COLORS,
            s=62,
            edgecolor="#222222",
            linewidth=0.5,
            ax=ax,
        )
        sns.regplot(
            data=block,
            x="y_std",
            y="abs_error",
            scatter=False,
            color="#333333",
            line_kws={"linestyle": "--", "linewidth": 1},
            ax=ax,
        )
        corr = float(block[["y_std", "abs_error"]].corr().iloc[0, 1])
        coverage = float(block["inside95"].mean())
        rows.append({"target": target, "pearson_std_abs_error": corr, "coverage95_sample": coverage})
        ax.set_title(f"{target}\nr={corr:.2f}, coverage95={coverage:.2f}")
        ax.set_xlabel("Desviacion estandar predictiva GP")
        ax.set_ylabel("Error absoluto")
        ax.set_ylim(bottom=0)
        if ax is not axes[-1]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Subproducto", bbox_to_anchor=(1.02, 1), loc="upper left")

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "fig_05_uncertainty_vs_error_data.csv", index=False)
    fig.suptitle("Calibracion practica: incertidumbre del GP frente al error real", fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_05_uncertainty_vs_error.png", bbox_inches="tight")
    plt.close(fig)


def plot_prediction_intervals_by_diet(pred_df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.5))
    interval_rows = []
    for ax, target in zip(axes, TARGET_ORDER):
        block = pred_df[pred_df["target"] == target].copy()
        agg = (
            block.groupby(["diet_name", "byproduct_type"], observed=True)
            .agg(
                observed_mean=("y_true", "mean"),
                predicted_mean=("y_pred", "mean"),
                std_rms=("y_std", lambda s: float(np.sqrt(np.mean(np.square(s))))),
                abs_error_mean=("abs_error", "mean"),
                inside95_rate=("inside95", "mean"),
            )
            .reset_index()
        )
        agg["lower95"] = agg["predicted_mean"] - 1.96 * agg["std_rms"]
        agg["upper95"] = agg["predicted_mean"] + 1.96 * agg["std_rms"]
        ascending = target == "FCR"
        agg = agg.sort_values("observed_mean", ascending=ascending).reset_index(drop=True)
        interval_rows.append(agg.assign(target=target))

        x = np.arange(len(agg))
        colors = agg["byproduct_type"].map(BYPRODUCT_COLORS).fillna("#777777")
        yerr = 1.96 * agg["std_rms"]
        ax.errorbar(
            x,
            agg["predicted_mean"],
            yerr=yerr,
            fmt="o",
            color="#222222",
            ecolor="#4C78A8",
            elinewidth=1.2,
            capsize=3,
            label="Prediccion GP +/- 1.96 std media",
        )
        ax.scatter(
            x,
            agg["observed_mean"],
            s=55,
            c=colors,
            edgecolor="#222222",
            linewidth=0.5,
            zorder=3,
            label="Media observada",
        )
        for i, row in agg.iterrows():
            ax.plot([i, i], [row["observed_mean"], row["predicted_mean"]], color="#999999", linewidth=0.7, alpha=0.7)
        ax.set_title(target)
        ax.set_ylabel(TARGET_UNITS[target])
        ax.set_xticks(x)
        ax.set_xticklabels(agg["diet_name"], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)

    symbol_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="#222222",
            label="Prediccion GP media",
            markerfacecolor="#222222",
            markersize=6,
            linewidth=0,
        ),
        plt.Line2D(
            [0],
            [0],
            color="#4C78A8",
            label="Intervalo GP +/- 1.96 std",
            linewidth=1.5,
        ),
    ]
    product_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=k, markerfacecolor=v, markeredgecolor="#222222", markersize=7)
        for k, v in BYPRODUCT_COLORS.items()
    ]
    axes[0].legend(handles=symbol_handles + product_handles, title="Simbolos y subproducto", bbox_to_anchor=(1.01, 1), loc="upper left")
    pd.concat(interval_rows, ignore_index=True).to_csv(OUTPUT_DIR / "fig_06_prediction_intervals_by_diet_data.csv", index=False)
    fig.suptitle("Prediccion prospectiva por dieta: media LODO e intervalo GP", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_06_prediction_intervals_by_diet.png", bbox_inches="tight")
    plt.close(fig)


def plot_observed_multiobjective_tradeoff():
    ranking = pd.read_csv(RANKING_FILE)
    ranking = ranking.sort_values("exploratory_equal_weight_score", ascending=False).reset_index(drop=True)
    top = ranking.head(11).copy()

    heat = top.set_index("diet_name")[
        ["score_fcr", "score_quitina", "score_proteina", "exploratory_equal_weight_score"]
    ]
    heat = heat.rename(
        columns={
            "score_fcr": "FCR invertido",
            "score_quitina": "Quitina",
            "score_proteina": "Proteina",
            "exploratory_equal_weight_score": "Score medio",
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1, 1.25]})
    sns.heatmap(
        heat,
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Score normalizado"},
        ax=ax1,
    )
    ax1.set_title("Compromiso observado por dieta")
    ax1.set_xlabel("")
    ax1.set_ylabel("")

    sizes = 80 + 260 * (ranking["quitina_mean"] - ranking["quitina_mean"].min()) / (
        ranking["quitina_mean"].max() - ranking["quitina_mean"].min()
    )
    scatter = ax2.scatter(
        ranking["fcr_mean"],
        ranking["proteina_mean"],
        s=sizes,
        c=ranking["exploratory_equal_weight_score"],
        cmap="viridis",
        edgecolor="#222222",
        linewidth=0.6,
        alpha=0.9,
    )
    for _, row in ranking.head(5).iterrows():
        ax2.annotate(
            row["diet_name"],
            (row["fcr_mean"], row["proteina_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax2.set_title("Trade-off FCR / proteina / quitina")
    ax2.set_xlabel("FCR medio observado (menor es mejor)")
    ax2.set_ylabel("Proteina larvaria media (%)")
    ax2.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax2)
    cbar.set_label("Score multiobjetivo observado")

    ax2.text(
        0.02,
        0.02,
        "Tamano del punto = quitina media",
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )
    fig.suptitle("Viabilidad prospectiva: dietas observadas candidatas a exploracion", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_07_observed_multiobjective_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def _risk_label(abs_error: float, coverage: float, normalized_error: float | None = None) -> str:
    if (normalized_error is not None and normalized_error >= 0.25) or coverage < 0.5:
        return "alto"
    if (normalized_error is not None and normalized_error >= 0.12) or coverage < 0.8:
        return "medio"
    return "bajo"


def _fmt_pct(value: float) -> str:
    value = float(value)
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _fmt_ei(value: float) -> str:
    value = float(value)
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    incumbent_value: float,
    direction: str,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement for a minimization or maximization objective."""
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    sigma = np.clip(sigma, 1e-12, None)

    if direction == "min":
        improvement = float(incumbent_value) - mu - float(xi)
    elif direction == "max":
        improvement = mu - float(incumbent_value) - float(xi)
    else:
        raise ValueError(f"Unknown EI direction: {direction}")

    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _tuning_path_for_case(feature_mode: str, target: str, model_name: str) -> Path:
    target_slug = slugify(target)
    return LOG_ROOT / feature_mode / target_slug / f"{target_slug}_{model_name.lower()}_tuning.json"


def _load_representative_params(feature_mode: str, target: str, model_name: str) -> dict:
    tuning_path = _tuning_path_for_case(feature_mode, target, model_name)
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    folds = tuning.get("folds", [])
    if not folds:
        return {}
    return dict(folds[0].get("params", {}))


def build_candidate_grid_for_case(
    raw: pd.DataFrame,
    feature_cols: list[str],
    feature_names: list[str],
    step: int = 1,
) -> pd.DataFrame:
    """
    Feasible pre-infill grid: one byproduct at a time, interpolated inside
    the observed inclusion range. Exact observed percentages are flagged so
    EI can propose genuinely new candidates.
    """
    byproducts = ["hoja", "orujo", "quinoa"]
    source = raw[raw["byproduct_type"].isin(byproducts)].copy()
    interp_feature_cols = [col for col in feature_cols if col != "inclusion_pct"]
    feature_medians = (
        raw[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .median(numeric_only=True)
        .to_dict()
    )

    diet_features = (
        source.groupby(["byproduct_type", "inclusion_pct"], observed=True)[interp_feature_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    diet_features[interp_feature_cols] = diet_features[interp_feature_cols].apply(pd.to_numeric, errors="coerce")
    diet_features = diet_features.fillna(feature_medians)

    rows = []
    for byproduct, block in diet_features.groupby("byproduct_type", observed=True):
        block = block.sort_values("inclusion_pct").reset_index(drop=True)
        if block.empty:
            continue

        x_obs = block["inclusion_pct"].astype(float).to_numpy()
        observed_pcts = np.unique(x_obs)
        start = int(np.ceil(float(x_obs.min()) / step) * step)
        stop = int(np.floor(float(x_obs.max()) / step) * step)

        for pct in np.arange(start, stop + 0.5 * step, step, dtype=float):
            row = {
                "candidate_name": f"{str(byproduct).capitalize()}{_fmt_pct(pct)}",
                "byproduct_type": str(byproduct),
                "inclusion_pct": float(pct),
                "is_observed_pct": bool(np.any(np.isclose(pct, observed_pcts, atol=1e-9))),
            }

            for col in interp_feature_cols:
                row[col] = float(np.interp(pct, x_obs, block[col].astype(float).to_numpy()))

            for col in feature_names:
                if col.startswith("byproduct_"):
                    row[col] = 1.0 if col == f"byproduct_{byproduct}" else 0.0
                elif col not in row:
                    row[col] = float(feature_medians.get(col, 0.0))

            rows.append(row)

    if not rows:
        raise RuntimeError("No EI candidates could be built from the observed diet ranges.")

    meta_cols = ["candidate_name", "byproduct_type", "inclusion_pct", "is_observed_pct"]
    ordered_cols = meta_cols + [col for col in feature_names if col not in meta_cols]
    return pd.DataFrame(rows)[ordered_cols]


def fit_final_gp_case(raw: pd.DataFrame, case: pd.Series):
    target = str(case["target"])
    target_col = str(case["target_col"])
    feature_mode = str(case["feature_mode"])
    model_name = str(case["model"])
    feature_cols = FEATURE_MODES[feature_mode]

    X, y, _, feature_names, _ = build_X_y_groups(raw, target_col, feature_cols)
    params = _load_representative_params(feature_mode, target, model_name)
    reconstructor = ModelReconstructor(X, y, feature_names)
    model = reconstructor.retrain_model(model_name, params)
    return model, feature_cols, feature_names, params


def _final_kernel_text(model) -> str:
    try:
        gpr = model.model_.named_steps["model"]
        return str(getattr(gpr, "kernel_", ""))
    except Exception:
        return ""


def _final_lml(model) -> float | None:
    try:
        gpr = model.model_.named_steps["model"]
        value = getattr(gpr, "log_marginal_likelihood_value_", None)
        return None if value is None else float(value)
    except Exception:
        return None


def compute_ei_pre_infill_candidates(best_cases: pd.DataFrame, xi: float = 0.01, candidate_step_pct: int = 1):
    raw = pd.read_csv(DATA_FILE)
    ranking = pd.read_csv(RANKING_FILE)
    summary_rows = []
    landscape_rows = []

    for _, case in best_cases.iterrows():
        target = str(case["target"])
        direction = TARGET_DIRECTIONS[target]
        target_mean_col = TARGET_MEAN_COLS[target]

        if direction == "min":
            incumbent_row = ranking.loc[ranking[target_mean_col].idxmin()]
            objective = f"Minimizar {target}"
            improvement_label = "reduccion esperada"
        else:
            incumbent_row = ranking.loc[ranking[target_mean_col].idxmax()]
            objective = f"Maximizar {target}"
            improvement_label = "aumento esperado"

        incumbent_value = float(incumbent_row[target_mean_col])
        observed_range = float(ranking[target_mean_col].max() - ranking[target_mean_col].min())
        weak_ei_threshold = max(1e-8, 0.01 * observed_range)
        model, feature_cols, feature_names, params = fit_final_gp_case(raw, case)
        candidates = build_candidate_grid_for_case(raw, feature_cols, feature_names, step=candidate_step_pct)
        X_candidates = candidates[feature_names].to_numpy(dtype=float)
        mu, sigma = model.predict_dist(X_candidates)
        sigma = np.zeros_like(mu, dtype=float) if sigma is None else np.asarray(sigma, dtype=float)

        candidates = candidates.copy()
        candidates["target"] = target
        candidates["objective"] = objective
        candidates["feature_mode"] = str(case["feature_mode"])
        candidates["model"] = str(case["model"])
        candidates["incumbent_diet"] = str(incumbent_row["diet_name"])
        candidates["incumbent_value"] = incumbent_value
        candidates["mu"] = np.asarray(mu, dtype=float).ravel()
        candidates["sigma"] = sigma.ravel()
        candidates["ei"] = expected_improvement(
            candidates["mu"].to_numpy(),
            candidates["sigma"].to_numpy(),
            incumbent_value=incumbent_value,
            direction=direction,
            xi=xi,
        )

        if direction == "min":
            candidates["predicted_improvement"] = incumbent_value - candidates["mu"]
        else:
            candidates["predicted_improvement"] = candidates["mu"] - incumbent_value

        candidate_pool = candidates[~candidates["is_observed_pct"]].copy()
        if candidate_pool.empty:
            candidate_pool = candidates.copy()

        selected = candidate_pool.loc[candidate_pool["ei"].idxmax()]
        selected_ei = float(selected["ei"])
        predicted_improvement = float(selected["predicted_improvement"])

        if selected_ei <= weak_ei_threshold:
            selection_reason = "EI bajo: candidato util solo como pre-infill exploratorio."
        elif predicted_improvement > 0:
            selection_reason = "EI por explotacion: la media GP ya supera al incumbent."
        else:
            selection_reason = "EI por exploracion: la incertidumbre sostiene el ensayo."

        kernel_text = _final_kernel_text(model)
        summary_rows.append(
            {
                "target": target,
                "objective": objective,
                "direction": direction,
                "feature_mode": str(case["feature_mode"]),
                "model": str(case["model"]),
                "xi": float(xi),
                "candidate_grid_step_pct": int(candidate_step_pct),
                "incumbent_diet": str(incumbent_row["diet_name"]),
                "incumbent_byproduct_type": str(incumbent_row["byproduct_type"]),
                "incumbent_inclusion_pct": float(incumbent_row["inclusion_pct"]),
                "incumbent_value": incumbent_value,
                "observed_target_mean_range": observed_range,
                "weak_ei_threshold": weak_ei_threshold,
                "candidate_name": str(selected["candidate_name"]),
                "candidate_byproduct_type": str(selected["byproduct_type"]),
                "candidate_inclusion_pct": float(selected["inclusion_pct"]),
                "candidate_is_observed_pct": bool(selected["is_observed_pct"]),
                "candidate_status": "nuevo_no_ensayado" if not bool(selected["is_observed_pct"]) else "ya_observado",
                "candidate_mu": float(selected["mu"]),
                "candidate_sigma": float(selected["sigma"]),
                "candidate_ei": selected_ei,
                "candidate_predicted_improvement": predicted_improvement,
                "improvement_label": improvement_label,
                "selection_reason": selection_reason,
                "candidate_pool_n": int(len(candidate_pool)),
                "selected_alpha": params.get("alpha"),
                "selected_normalize_y": params.get("normalize_y"),
                "selected_n_restarts_optimizer": params.get("n_restarts_optimizer"),
                "final_log_marginal_likelihood": _final_lml(model),
                "final_optimized_kernel": kernel_text,
            }
        )

        landscape_keep = [
            "target",
            "objective",
            "feature_mode",
            "model",
            "incumbent_diet",
            "incumbent_value",
            "candidate_name",
            "byproduct_type",
            "inclusion_pct",
            "is_observed_pct",
            "mu",
            "sigma",
            "ei",
            "predicted_improvement",
        ]
        landscape_rows.append(candidates[landscape_keep])

    summary_df = pd.DataFrame(summary_rows)
    landscape_df = pd.concat(landscape_rows, ignore_index=True)
    summary_df.to_csv(OUTPUT_DIR / "fig_07_ei_pre_infill_candidates.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "fig_07_incumbent_pre_infill_candidates.csv", index=False)
    landscape_df.to_csv(OUTPUT_DIR / "fig_08_ei_candidate_landscape_data.csv", index=False)
    return summary_df, landscape_df


def plot_ei_pre_infill_table(ei_summary: pd.DataFrame):
    rows = []
    for _, row in ei_summary.iterrows():
        unit = TARGET_UNITS[str(row["target"])]
        incumbent = (
            f"{row['incumbent_diet']}\n"
            f"{row['incumbent_byproduct_type']} {_fmt_pct(row['incumbent_inclusion_pct'])}%\n"
            f"{row['incumbent_value']:.3f} {unit}"
        )
        candidate = (
            f"{row['candidate_name']}\n"
            f"{row['candidate_byproduct_type']} {_fmt_pct(row['candidate_inclusion_pct'])}%\n"
            f"{row['candidate_status'].replace('_', ' ')}"
        )
        posterior = (
            f"mu={row['candidate_mu']:.3f}\n"
            f"sigma={row['candidate_sigma']:.3f}\n"
            f"{row['improvement_label']}={row['candidate_predicted_improvement']:.3f}"
        )
        ei_value = f"EI={_fmt_ei(row['candidate_ei'])}\nxi={row['xi']:.2f}"
        model = (
            f"{row['model']}\n"
            f"{row['feature_mode']}\n"
            f"alpha={row['selected_alpha']}"
        )
        rows.append(
            [
                row["objective"],
                incumbent,
                candidate,
                posterior,
                ei_value,
                model,
                textwrap.fill(str(row["selection_reason"]), width=32),
            ]
        )

    col_labels = [
        "Objetivo",
        "Incumbent observado",
        "Candidato EI",
        "Posterior GP",
        "Adquisicion",
        "Surrogado",
        "Lectura",
    ]

    fig, ax = plt.subplots(figsize=(18.5, 4.1))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.12, 0.16, 0.14, 0.15, 0.10, 0.17, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    table.scale(1, 2.35)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="#222222")
            cell.set_facecolor("#E8EEF7")
        else:
            if col_idx == 2:
                cell.set_facecolor("#E9F7EF")
                cell.set_text_props(weight="bold")
            elif col_idx == 4:
                cell.set_facecolor("#FCF3CF")
            else:
                cell.set_facecolor("#FFFFFF")

    ax.set_title(
        "Pre-infill EI: incumbent observado y mejor candidato no ensayado por el GP\n"
        "EI se calcula sobre porcentajes interpolados no ensayados en rejilla de 1% dentro del rango observado",
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_07_ei_pre_infill_candidates.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_07_incumbent_pre_infill_table.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_07_observed_multiobjective_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def plot_ei_candidate_landscape(ei_landscape: pd.DataFrame, ei_summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), sharey=False)
    palette = {k: v for k, v in BYPRODUCT_COLORS.items() if k != "control"}

    for ax, target in zip(axes, TARGET_ORDER):
        block = ei_landscape[ei_landscape["target"] == target].copy()
        selected = ei_summary[ei_summary["target"] == target].iloc[0]

        sns.lineplot(
            data=block,
            x="inclusion_pct",
            y="ei",
            hue="byproduct_type",
            style="byproduct_type",
            dashes=False,
            palette=palette,
            ax=ax,
            legend=ax is axes[-1],
        )
        observed = block[block["is_observed_pct"]]
        ax.scatter(
            observed["inclusion_pct"],
            observed["ei"],
            s=72,
            facecolors="none",
            edgecolors="#222222",
            linewidths=1.0,
            zorder=4,
        )
        ax.scatter(
            [selected["candidate_inclusion_pct"]],
            [selected["candidate_ei"]],
            marker="*",
            s=250,
            color="#222222",
            edgecolor="#FFFFFF",
            linewidth=0.6,
            zorder=5,
        )
        ax.annotate(
            selected["candidate_name"],
            (selected["candidate_inclusion_pct"], selected["candidate_ei"]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=8.5,
            weight="bold",
        )
        ax.set_title(f"{target}\nincumbent: {selected['incumbent_diet']}")
        ax.set_xlabel("Inclusion de subproducto (%)")
        ax.set_ylabel("Expected Improvement (EI)")
        ax.grid(axis="y", alpha=0.25)
        if ax is axes[-1]:
            ax.legend(title="Subproducto", bbox_to_anchor=(1.02, 1), loc="upper left")

    handles = [
        plt.Line2D([0], [0], marker="o", color="#222222", markerfacecolor="none", linewidth=0, label="porcentaje observado"),
        plt.Line2D([0], [0], marker="*", color="#222222", linewidth=0, markersize=12, label="candidato EI"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.48, -0.03),
        ncol=2,
        frameon=True,
        title="Marcadores",
    )
    fig.suptitle(
        "Paisaje de adquisicion EI para pre-infill: ranking exploratorio de dietas no ensayadas",
        fontweight="bold",
        y=1.05,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUTPUT_DIR / "fig_08_ei_candidate_landscape.png", bbox_inches="tight")
    plt.close(fig)


def plot_incumbent_pre_infill_table(pred_df: pd.DataFrame):
    ranking = pd.read_csv(RANKING_FILE)
    target_ranges = pred_df.groupby("target", observed=True)["y_true"].agg(lambda s: s.max() - s.min()).to_dict()

    objectives = [
        {
            "objective": "Minimizar FCR",
            "selection": ranking.loc[ranking["fcr_mean"].idxmin()],
            "target": "FCR",
            "observed": lambda r: f"FCR={r['fcr_mean']:.3f}",
            "decision": "Incumbent si prima eficiencia alimentaria.",
        },
        {
            "objective": "Maximizar quitina",
            "selection": ranking.loc[ranking["quitina_mean"].idxmax()],
            "target": "Quitina",
            "observed": lambda r: f"Quitina={r['quitina_mean']:.2f}%",
            "decision": "Incumbent si prima valorizacion en quitina.",
        },
        {
            "objective": "Maximizar proteina",
            "selection": ranking.loc[ranking["proteina_mean"].idxmax()],
            "target": "Proteina",
            "observed": lambda r: f"Proteina={r['proteina_mean']:.2f}%",
            "decision": "Incumbent si prima composicion proteica.",
        },
        {
            "objective": "Compromiso igual-peso",
            "selection": ranking.loc[ranking["exploratory_equal_weight_score"].idxmax()],
            "target": None,
            "observed": lambda r: f"Score={r['exploratory_equal_weight_score']:.2f}",
            "decision": "Mejor punto de partida para un infill multiobjetivo.",
        },
    ]

    rows = []
    for item in objectives:
        r = item["selection"]
        diet = str(r["diet_name"])
        candidate = f"{diet}\n{r['byproduct_type']} {int(r['inclusion_pct'])}%"
        values = f"{item['observed'](r)}\nFCR={r['fcr_mean']:.2f}; Q={r['quitina_mean']:.2f}; P={r['proteina_mean']:.2f}"
        scores = (
            f"FCR={r['score_fcr']:.2f}\n"
            f"Q={r['score_quitina']:.2f}; P={r['score_proteina']:.2f}\n"
            f"medio={r['exploratory_equal_weight_score']:.2f}"
        )

        if item["target"] is None:
            block = pred_df[pred_df["diet_name"] == diet].copy()
            agg = (
                block.groupby("target", observed=True)
                .agg(abs_error=("abs_error", "mean"), coverage=("inside95", "mean"))
                .reset_index()
            )
            agg["normalized_error"] = agg.apply(
                lambda row: row["abs_error"] / target_ranges[str(row["target"])],
                axis=1,
            )
            abs_error = float(agg["normalized_error"].mean())
            coverage = float(agg["coverage"].mean())
            risk = _risk_label(abs_error=0.0, coverage=coverage, normalized_error=abs_error)
            gp_support = f"Error norm. medio={abs_error:.2f}\ncoverage95 medio={coverage:.2f}\nriesgo {risk}"
        else:
            block = pred_df[(pred_df["diet_name"] == diet) & (pred_df["target"] == item["target"])]
            abs_error = float(block["abs_error"].mean())
            coverage = float(block["inside95"].mean())
            normalized_error = abs_error / target_ranges[item["target"]]
            risk = _risk_label(abs_error=abs_error, coverage=coverage, normalized_error=normalized_error)
            gp_support = f"MAE LODO={abs_error:.2f}\ncoverage95={coverage:.2f}\nriesgo {risk}"

        rows.append(
            {
                "objective": item["objective"],
                "incumbent": diet,
                "byproduct_type": r["byproduct_type"],
                "inclusion_pct": r["inclusion_pct"],
                "fcr_mean": r["fcr_mean"],
                "quitina_mean": r["quitina_mean"],
                "proteina_mean": r["proteina_mean"],
                "score_fcr": r["score_fcr"],
                "score_quitina": r["score_quitina"],
                "score_proteina": r["score_proteina"],
                "score_mean": r["exploratory_equal_weight_score"],
                "gp_support": gp_support.replace("\n", " | "),
                "risk": risk,
                "decision": item["decision"],
                "table_candidate": candidate,
                "table_values": values,
                "table_scores": scores,
                "table_gp_support": gp_support,
            }
        )

    incumbent_df = pd.DataFrame(rows)
    incumbent_df.drop(
        columns=["table_candidate", "table_values", "table_scores", "table_gp_support"],
    ).to_csv(OUTPUT_DIR / "fig_07_incumbent_pre_infill_candidates.csv", index=False)

    table_data = incumbent_df[
        [
            "objective",
            "table_candidate",
            "table_values",
            "table_scores",
            "table_gp_support",
            "decision",
        ]
    ].values.tolist()

    col_labels = [
        "Sub-objetivo",
        "Incumbent observado",
        "Evidencia observada",
        "Scores observados\n(0-1)",
        "Lectura GP LODO",
        "Uso pre-infill",
    ]

    fig, ax = plt.subplots(figsize=(17.5, 4.2))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.14, 0.14, 0.19, 0.14, 0.17, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.25)

    risk_colors = {"alto": "#FADBD8", "medio": "#FCF3CF", "bajo": "#D5F5E3"}
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="#222222")
            cell.set_facecolor("#E8EEF7")
        else:
            risk = incumbent_df.iloc[row_idx - 1]["risk"]
            cell.set_facecolor(risk_colors[risk] if col_idx == 4 else "#FFFFFF")
            if col_idx == 1:
                cell.set_text_props(weight="bold")

    ax.set_title(
        "Incumbents pre-infill del caso real Hermetia\n"
        "Los scores salen de dietas observadas; el GP aporta lectura de riesgo sobre dietas no vistas",
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_07_incumbent_pre_infill_table.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_07_observed_multiobjective_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style()

    metrics = load_metrics()
    best_cases = select_best_gp_cases(metrics)

    plot_lodo_generalization(metrics)
    plot_kernel_family_mae(metrics)

    pred_df = reconstruct_predictions(best_cases)
    plot_error_by_diet(pred_df)
    plot_best_gp_parity(pred_df)
    plot_uncertainty_vs_error(pred_df)
    plot_prediction_intervals_by_diet(pred_df)
    ei_summary, ei_landscape = compute_ei_pre_infill_candidates(best_cases)
    plot_ei_pre_infill_table(ei_summary)
    plot_ei_candidate_landscape(ei_landscape, ei_summary)

    summary = {
        "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "figures": [
            "fig_01_lodo_generalization_vs_dummy.png",
            "fig_02_error_by_left_out_diet.png",
            "fig_03_best_gp_parity_by_target.png",
            "fig_04_kernel_family_mae.png",
            "fig_05_uncertainty_vs_error.png",
            "fig_06_prediction_intervals_by_diet.png",
            "fig_07_ei_pre_infill_candidates.png",
            "fig_08_ei_candidate_landscape.png",
        ],
        "ei_pre_infill_candidates": ei_summary[
            [
                "target",
                "incumbent_diet",
                "incumbent_value",
                "candidate_name",
                "candidate_mu",
                "candidate_sigma",
                "candidate_ei",
                "candidate_predicted_improvement",
            ]
        ].to_dict(orient="records"),
        "best_cases": best_cases[
            ["target", "feature_mode", "model", "mae_macro_mean", "coverage_95_macro_mean"]
        ].to_dict(orient="records"),
    }
    (OUTPUT_DIR / "pov_figure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
