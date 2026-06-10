import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut

from src.analysis.tuning_reporter import ModelReconstructor, ResultsLoader, SurrogatePlotter
from src.configs.tuning_specs import FEATURE_COLS_FULL, FEATURE_COLS_REDUCED, TARGET_MAP
from src.utils.paths import ENTOMOTIVE_DATA_DIR, LOGS_DIR, PLOTS_DIR
from src.utils.tools import slugify


BASE_LOG_DIR = LOGS_DIR / "tuning" / "productivity_hermetia_gp_ard_kernels_no_tpc_v1"
DATA_FILE = ENTOMOTIVE_DATA_DIR / "productivity_hermetia_lote.csv"
GLOBAL_OUTPUT_DIR = PLOTS_DIR / "comprehensive_report_gp_ard_kernels_no_tpc_v1"


def build_X_y_groups(df: pd.DataFrame, target_col: str, feature_cols: list):
    """Build X, y and diet groups for the Entomotive real case."""
    data = df.copy()
    data = data.loc[~data[target_col].isna()].reset_index(drop=True)

    groups = data["diet_name"].astype(str).to_numpy()
    y = data[target_col].astype(float).to_numpy()

    byp = pd.get_dummies(data["byproduct_type"], prefix="byproduct", drop_first=False)
    valid_cols = [c for c in feature_cols if c in data.columns]
    Xdf = data[valid_cols].copy()
    Xdf = pd.concat([Xdf, byp], axis=1)
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    return Xdf.to_numpy(dtype=float), y, groups, Xdf.columns.tolist()


def collect_cv_predictions(model, X, y, groups):
    """Collect honest LODO predictions and uncertainty for a refitted model."""
    logo = LeaveOneGroupOut()
    y_pred_cv = np.zeros_like(y)
    y_std_cv = np.zeros_like(y)

    for train_idx, test_idx in logo.split(X, y, groups):
        m_fold = clone(model)
        m_fold.fit(X[train_idx], y[train_idx])

        mean, std = m_fold.predict_dist(X[test_idx])
        y_pred_cv[test_idx] = mean
        if std is not None:
            y_std_cv[test_idx] = std

    return y_pred_cv, y_std_cv


def run_single_experiment_report(
    target_key: str,
    target_col: str,
    experiment_name: str,
    feature_cols: list,
):
    print(f"\n--- Processing {target_key} | {experiment_name} ---")

    log_dir = BASE_LOG_DIR / experiment_name
    output_dir = GLOBAL_OUTPUT_DIR / target_key.lower() / experiment_name
    target_slug = slugify(target_key)

    try:
        loader = ResultsLoader(log_dir)
        summary = loader.load_target_summary(target_slug)
    except FileNotFoundError:
        print(f"Skipping {experiment_name} for {target_key} (log not found)")
        return None

    plotter = SurrogatePlotter(output_dir)

    df_raw = pd.read_csv(DATA_FILE)
    X, y, groups, feature_names = build_X_y_groups(df_raw, target_col, feature_cols)
    reconstructor = ModelReconstructor(X, y, feature_names)

    model_folds = {}
    best_params = {}
    for model_name, info in summary["models"].items():
        folds_csv = info["files"].get("folds_csv")
        if folds_csv:
            model_folds[model_name] = loader.load_fold_results(folds_csv)
            best_params[model_name] = model_folds[model_name].iloc[0]["params"]

    print(f"[{experiment_name}] Plotting stability...")
    for metric in ["mae", "rmse", "coverage95"]:
        plotter.plot_comparative_metrics_box(model_folds, metric=metric)

    cv_preds = {}
    gp_param_rows = []
    for model_name, params in best_params.items():
        print(f"[{experiment_name}] Inspecting {model_name}...")
        try:
            model_base = reconstructor.retrain_model(model_name, params)
            plotter.plot_feature_relevance(model_base, feature_names, model_name)

            y_cv, std_cv = collect_cv_predictions(model_base, X, y, groups)
            cv_preds[model_name] = y_cv

            if model_name.startswith("GP"):
                gpr = model_base.model_.named_steps["model"]
                gp_param_rows.append(
                    {
                        "model": model_name,
                        "alpha": model_base.alpha,
                        "kernel_optimized": gpr.kernel_,
                    }
                )
                plotter.plot_gp_uncertainty_analysis(
                    y, y_cv, std_cv, groups, model_name=model_name
                )
                for feat in feature_names[:3]:
                    plotter.plot_1d_response_slice(
                        model_base,
                        X,
                        y,
                        feature_names,
                        feat,
                        groups,
                        model_name=model_name,
                    )

        except Exception as exc:
            logging.exception("Error inspecting %s: %s", model_name, exc)

    if cv_preds:
        plotter.plot_actual_vs_predicted_cv(y, cv_preds, groups)
    plotter.plot_gp_fitted_parameters(gp_param_rows)

    return summary


def main():
    for target_key, target_col in TARGET_MAP.items():
        print(f"\n{'=' * 60}")
        print(f"GENERATING REPORT FOR TARGET: {target_key}")
        print(f"{'=' * 60}")

        summary_reduced = run_single_experiment_report(
            target_key, target_col, "REDUCED_FEATURES", FEATURE_COLS_REDUCED
        )
        summary_full = run_single_experiment_report(
            target_key, target_col, "FULL_FEATURES", FEATURE_COLS_FULL
        )

        if summary_reduced and summary_full:
            print(f"\n>>> Generating comparison for {target_key}...")
            comp_output_dir = GLOBAL_OUTPUT_DIR / target_key.lower() / "comparison"
            plotter = SurrogatePlotter(comp_output_dir)
            plotter.plot_experiment_comparison(summary_full, summary_reduced, target_key)

    print(f"\nAll reports generated in: {GLOBAL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
