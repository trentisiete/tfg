from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.configs.tuning_specs import FEATURE_COLS_FULL, FEATURE_COLS_REDUCED, build_named_gp_kernels
from src.models.gp import GPSurrogateRegressor
from src.utils.paths import ENTOMOTIVE_DATA_DIR, OUTPUTS_DIR, PROJECT_ROOT


DATA_FILE = ENTOMOTIVE_DATA_DIR / "productivity_all_lote.csv"
OUT_DIR = OUTPUTS_DIR / "reports" / "quick_productivity_all_protein_lodo"
PLOT_DIR = OUTPUTS_DIR / "plots" / "quick_productivity_all_protein_lodo"
TARGET_COL = "PROTEINA (%)"

FEATURE_SETS = {
    "REDUCED": FEATURE_COLS_REDUCED,
    "FULL": FEATURE_COLS_FULL,
}


def style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
        }
    )


def make_design(df: pd.DataFrame, feature_cols: list[str], include_species: bool) -> tuple[pd.DataFrame, list[str]]:
    parts = [df[feature_cols].copy()]
    parts.append(pd.get_dummies(df["byproduct_type"], prefix="byproduct", drop_first=False))
    if include_species:
        parts.append(pd.get_dummies(df["species"], prefix="species", drop_first=False))
    xdf = pd.concat(parts, axis=1)
    xdf = xdf.apply(pd.to_numeric, errors="coerce")
    return xdf, xdf.columns.tolist()


def fit_predict_gp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kernel = build_named_gp_kernels(X_train.shape[1])["GP_Compuesto_NoARD"]
    model = GPSurrogateRegressor(
        kernel=kernel,
        alpha=1.0,
        normalize_y=True,
        n_restarts_optimizer=5,
    )
    model.fit(X_train, y_train)
    pred, std = model.predict_dist(X_test)
    return np.asarray(pred).ravel(), np.asarray(std).ravel()


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    out = {
        "n": int(len(y_true)),
        "mae_micro": float(mean_absolute_error(y_true, y_pred)),
        "rmse_micro": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2_micro": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "bias_true_minus_pred": float(np.mean(y_true - y_pred)),
        "y_true_mean": float(np.mean(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_true_min": float(np.min(y_true)),
        "y_true_max": float(np.max(y_true)),
    }
    out["nmae_by_range"] = out["mae_micro"] / max(out["y_true_max"] - out["y_true_min"], 1e-12)
    if y_std is not None:
        lower = y_pred - 1.96 * y_std
        upper = y_pred + 1.96 * y_std
        out["coverage95_micro"] = float(np.mean((y_true >= lower) & (y_true <= upper)))
        out["mean_pred_std"] = float(np.mean(y_std))
    else:
        out["coverage95_micro"] = np.nan
        out["mean_pred_std"] = np.nan
    return out


def run_lodo(df: pd.DataFrame, subset_name: str, group_col: str, feature_mode: str, include_species: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = FEATURE_SETS[feature_mode]
    data = df.loc[~df[TARGET_COL].isna()].copy().reset_index(drop=True)
    xdf, feature_names = make_design(data, feature_cols, include_species=include_species)
    y = data[TARGET_COL].astype(float).to_numpy()
    groups = data[group_col].astype(str).to_numpy()

    pred_rows = []
    for group in sorted(pd.unique(groups)):
        test_mask = groups == group
        train_mask = ~test_mask
        train_medians = xdf.loc[train_mask, feature_names].median(numeric_only=True)
        X_train = xdf.loc[train_mask, feature_names].fillna(train_medians).fillna(0.0).to_numpy(dtype=float)
        X_test = xdf.loc[test_mask, feature_names].fillna(train_medians).fillna(0.0).to_numpy(dtype=float)
        y_train = y[train_mask]
        y_test = y[test_mask]

        dummy_global = np.full_like(y_test, float(np.mean(y_train)), dtype=float)
        pred_gp, std_gp = fit_predict_gp(X_train, y_train, X_test)

        train_meta = data.loc[train_mask]
        test_meta = data.loc[test_mask]
        species_mean_map = train_meta.assign(y=y_train).groupby("species", observed=True)["y"].mean().to_dict()
        fallback_mean = float(np.mean(y_train))
        dummy_species = test_meta["species"].map(species_mean_map).fillna(fallback_mean).to_numpy(dtype=float)

        for model_name, pred, std in [
            ("Dummy_GlobalMean", dummy_global, np.full_like(dummy_global, np.nan, dtype=float)),
            ("Dummy_SpeciesMean", dummy_species, np.full_like(dummy_species, np.nan, dtype=float)),
            ("GP_Compuesto_NoARD", pred_gp, std_gp),
        ]:
            lower = pred - 1.96 * std
            upper = pred + 1.96 * std
            inside = (y_test >= lower) & (y_test <= upper)
            for local_idx, row_idx in enumerate(np.where(test_mask)[0]):
                row = data.loc[row_idx]
                pred_rows.append(
                    {
                        "subset": subset_name,
                        "grouping": group_col,
                        "feature_mode": feature_mode,
                        "include_species_feature": include_species,
                        "model": model_name,
                        "heldout_group": group,
                        "species": row["species"],
                        "study_block": row["study_block"],
                        "diet_name": row["diet_name"],
                        "byproduct_type": row["byproduct_type"],
                        "inclusion_pct": row["inclusion_pct"],
                        "replica": row["Replica"],
                        "y_true": float(y_test[local_idx]),
                        "y_pred": float(pred[local_idx]),
                        "y_std": float(std[local_idx]),
                        "abs_error": float(abs(y_test[local_idx] - pred[local_idx])),
                        "inside95": bool(inside[local_idx]) if np.isfinite(std[local_idx]) else np.nan,
                    }
                )

    pred_df = pd.DataFrame(pred_rows)
    metric_rows = []
    for keys, block in pred_df.groupby(
        ["subset", "grouping", "feature_mode", "include_species_feature", "model"],
        observed=True,
    ):
        subset, grouping, fm, species_flag, model_name = keys
        std = None if model_name.startswith("Dummy") else block["y_std"].to_numpy(dtype=float)
        row = metric_dict(block["y_true"].to_numpy(), block["y_pred"].to_numpy(), std)
        per_group_mae = (
            block.groupby("heldout_group", observed=True)
            .apply(lambda g: mean_absolute_error(g["y_true"], g["y_pred"]), include_groups=False)
        )
        row.update(
            {
                "subset": subset,
                "grouping": grouping,
                "feature_mode": fm,
                "include_species_feature": bool(species_flag),
                "model": model_name,
                "n_folds": int(per_group_mae.shape[0]),
                "mae_macro_lodo": float(per_group_mae.mean()),
                "mae_macro_lodo_std": float(per_group_mae.std(ddof=0)),
            }
        )
        metric_rows.append(row)

    return pd.DataFrame(metric_rows), pred_df


def plot_summary(metrics: pd.DataFrame) -> None:
    block = metrics[
        (metrics["subset"] == "all_rows")
        & (metrics["feature_mode"] == "FULL")
        & (metrics["grouping"] == "species_diet")
    ].copy()
    block["feature_variant"] = np.where(block["include_species_feature"], "con species", "sin species")
    block["model_variant"] = block["model"] + "\n" + block["feature_variant"]

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    sns.barplot(
        data=block,
        x="model_variant",
        y="mae_macro_lodo",
        hue="model",
        dodge=False,
        palette={
            "Dummy_GlobalMean": "#8C8C8C",
            "Dummy_SpeciesMean": "#4C78A8",
            "GP_Compuesto_NoARD": "#2A9D8F",
        },
        edgecolor="#333333",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_title("Productivity_all proteína: LODO por especie+dieta")
    ax.set_xlabel("")
    ax.set_ylabel("MAE macro LODO")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "protein_lodo_all_species_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_parity(predictions: pd.DataFrame) -> None:
    block = predictions[
        (predictions["subset"] == "all_rows")
        & (predictions["grouping"] == "species_diet")
        & (predictions["feature_mode"] == "FULL")
        & (predictions["include_species_feature"])
        & (predictions["model"] == "GP_Compuesto_NoARD")
    ].copy()

    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    sns.scatterplot(
        data=block,
        x="y_true",
        y="y_pred",
        hue="species",
        style="study_block",
        s=68,
        edgecolor="#222222",
        linewidth=0.5,
        ax=ax,
    )
    vmin = min(block["y_true"].min(), block["y_pred"].min())
    vmax = max(block["y_true"].max(), block["y_pred"].max())
    pad = (vmax - vmin) * 0.08
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_title("GP compuesto con species: proteína LODO productivity_all")
    ax.set_xlabel("Proteína observada (%)")
    ax.set_ylabel("Proteína predicha (%)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "protein_lodo_gp_species_parity.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    style()

    raw = pd.read_csv(DATA_FILE)
    raw = raw.loc[~raw[TARGET_COL].isna()].copy()
    raw["species_diet"] = raw["species"].astype(str) + "::" + raw["diet_name"].astype(str)

    subsets = {
        "main_only": raw[raw["study_block"] == "main"].copy(),
        "all_rows": raw.copy(),
    }

    all_metrics = []
    all_predictions = []
    for subset_name, subset_df in subsets.items():
        subset_df = subset_df.reset_index(drop=True)
        for grouping in ["diet_name", "species_diet"]:
            for feature_mode in ["REDUCED", "FULL"]:
                for include_species in [False, True]:
                    metrics, predictions = run_lodo(
                        subset_df,
                        subset_name=subset_name,
                        group_col=grouping,
                        feature_mode=feature_mode,
                        include_species=include_species,
                    )
                    all_metrics.append(metrics)
                    all_predictions.append(predictions)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df.to_csv(OUT_DIR / "protein_lodo_metrics.csv", index=False)
    predictions_df.to_csv(OUT_DIR / "protein_lodo_predictions.csv", index=False)

    species_summary = (
        raw.groupby(["species", "study_block"], observed=True)[TARGET_COL]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    species_summary.to_csv(OUT_DIR / "protein_distribution_by_species_block.csv", index=False)

    plot_summary(metrics_df)
    plot_parity(predictions_df)

    best = metrics_df.sort_values(["mae_macro_lodo", "mae_micro"]).head(12)
    summary = {
        "dataset": str(DATA_FILE.relative_to(PROJECT_ROOT)),
        "target": TARGET_COL,
        "out_dir": str(OUT_DIR.relative_to(PROJECT_ROOT)),
        "plot_dir": str(PLOT_DIR.relative_to(PROJECT_ROOT)),
        "n_rows_non_null": int(raw.shape[0]),
        "species_distribution": species_summary.to_dict(orient="records"),
        "best_rows": best[
            [
                "subset",
                "grouping",
                "feature_mode",
                "include_species_feature",
                "model",
                "n_folds",
                "mae_macro_lodo",
                "mae_micro",
                "nmae_by_range",
                "bias_true_minus_pred",
                "coverage95_micro",
            ]
        ].to_dict(orient="records"),
    }
    (OUT_DIR / "protein_lodo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
