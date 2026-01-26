import json
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning

# Tune Method
from src.analysis.tuning import nested_lodo_tuning

# Paths and Utils
from src.utils.paths import ENTOMOTIVE_DATA_DIR, LOGS_DIR
from src.utils.tools import _to_jsonable, slugify

# Spec Imports
from src.configs.tuning_specs import (
    TARGET_MAP,
    MODELS,
    get_param_grids,
    FEATURE_COLS_FULL,
    FEATURE_COLS_REDUCED
)

# Filter warnings for cleaner logs (optional, enable if you want to see convergence issues)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def build_X_y_groups(df: pd.DataFrame, target_col: str, feature_cols: List[str]):
    """
    Prepares the data matrices X, y and groups.
    Auto-detects categorical columns (like Byproduct type) and one-hot encodes them,
    merging them with the selected numeric feature_cols.
    """
    data = df.copy()
    data = data.loc[~data[target_col].isna()].reset_index(drop=True)

    groups = data["diet_name"].astype(str).to_numpy()
    y = data[target_col].astype(float).to_numpy()

    # Handle Byproduct Type (Categorical) - Always added as it's structural
    if "byproduct_type" in data.columns:
        byp = pd.get_dummies(data["byproduct_type"], prefix="byproduct", drop_first=False)
    else:
        byp = pd.DataFrame()

    # numeric features
    Xdf = data[feature_cols].copy()

    # Concatenate numeric features + dummies
    if not byp.empty:
        Xdf = pd.concat([Xdf, byp], axis=1)

    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    X = Xdf.to_numpy(dtype=float)
    return X, y, groups, Xdf.columns.tolist()


def setup_logging(log_dir: Path, target_slug: str):
    """Initializes logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{target_slug}.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clean existing handlers
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return log_file


def save_model_outputs(target_dir: Path, target_slug: str, model_name: str, result: dict):
    """Saves the tuning results JSON and Fold details CSV."""
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / f"{target_slug}_{model_name.lower()}_tuning.json"
    folds_path = None

    jsonable = _to_jsonable(result)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(jsonable, f, indent=2)

    folds_df = pd.DataFrame(result.get("folds", []))
    if not folds_df.empty:
        folds_path = target_dir / f"{target_slug}_{model_name.lower()}_folds.csv"
        folds_df.to_csv(folds_path, index=False)

    return json_path, folds_path


def run_tuning_session(session_name: str, feature_cols: List[str], base_output_dir: Path):
    """
    Executes a full tuning session for all targets using a specific set of features.

    Args:
        session_name: Name of the experiment (e.g., 'FULL_FEATURES')
        feature_cols: List of column names to use as X.
        base_output_dir: Root directory for logs.
    """

    # Load Data
    df_path = ENTOMOTIVE_DATA_DIR / "productivity_hermetia_lote.csv"
    if not df_path.exists():
        logging.error(f"Data file not found: {df_path}")
        return

    df = pd.read_csv(df_path)

    # Define session specific directory
    session_dir = base_output_dir / session_name

    # Parallel settings
    inner_n_jobs = max(1, (os.cpu_count() or 2) - 1)
    outer_n_jobs = min(4, os.cpu_count() or 1)

    print(f"\n{'='*50}")
    print(f"STARTING SESSION: {session_name}")
    print(f"Output Directory: {session_dir}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"{'='*50}\n")

    for target_label, target_col in TARGET_MAP.items():
        target_slug = slugify(target_label)
        target_dir = session_dir / target_slug

        # Setup Logging
        log_file = setup_logging(target_dir, target_slug)

        logging.info(">>> STARTING TARGET: %s (%s)", target_label, target_col)

        if target_col not in df.columns:
            logging.error("Target column not found: %s", target_col)
            continue

        # Build Matrix
        X, y, groups, feature_names = build_X_y_groups(df, target_col, feature_cols)
        n_features_total = X.shape[1]

        logging.info("Data Shape: Samples=%d | Total Features (with dummies)=%d", len(y), n_features_total)

        # Get Dynamic Grids based on actual feature count (including dummies)
        current_param_grids = get_param_grids(n_features_total)

        summary = {
            "session": session_name,
            "target_label": target_label,
            "target_column": target_col,
            "n_samples": int(len(y)),
            "n_features_input": int(len(feature_cols)),
            "n_features_model": int(n_features_total), # Includes dummies
            "feature_names": feature_names,
            "models": {},
            "log_file": str(log_file),
        }

        # Loop Models
        for model_name, model in MODELS.items():
            param_grid = current_param_grids[model_name]
            grid_size = int(np.prod([len(v) for v in param_grid.values()]))

            logging.info(f"--- Tuning {model_name} (Grid Size: {grid_size}) ---")

            try:
                result = nested_lodo_tuning(
                    model,
                    X,
                    y,
                    groups,
                    param_grid,
                    primary="mae",
                    inner_n_jobs=inner_n_jobs,
                    outer_n_jobs=outer_n_jobs,
                )

                # Save Model Results
                json_path, folds_path = save_model_outputs(target_dir, target_slug, model_name, result)

                summary["models"][model_name] = {
                    "summary": result.get("summary", {}),
                    "chosen_params": result.get("chosen_params", []),
                    "files": {
                        "json": str(json_path),
                        "folds_csv": str(folds_path) if folds_path else None,
                    },
                }

                # New unified metrics structure: macro['metric']['mean']
                macro = result["summary"]["macro"]
                mae_mean = macro.get('mae', {}).get('mean', 'N/A')
                cov95_mean = macro.get('coverage_95', {}).get('mean')
                cov95_str = f"{cov95_mean:.2%}" if cov95_mean is not None else "N/A"

                logging.info(f"    [DONE] {model_name} -> MAE: {mae_mean:.4f} | Coverage95: {cov95_str}")

            except Exception as e:
                logging.error(f"    [ERROR] Failed tuning {model_name}: {e}")

        # Save Summary
        summary_path = target_dir / f"{target_slug}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(summary), f, indent=2)

        logging.info(f"<<< FINISHED TARGET: {target_label}\n")


def main():
    # Base directory for all logs
    BASE_LOG_DIR = LOGS_DIR / "tuning" / "productivity_hermetia_v1"

    # 1. Run Reduced Features Experiment
    run_tuning_session(
        session_name="REDUCED_FEATURES",
        feature_cols=FEATURE_COLS_REDUCED,
        base_output_dir=BASE_LOG_DIR
    )





    # 2. Run Full Features Experiment
    run_tuning_session(
        session_name="FULL_FEATURES",
        feature_cols=FEATURE_COLS_FULL,
        base_output_dir=BASE_LOG_DIR
    )

    print("\n\nAll tuning sessions completed successfully.")

if __name__ == "__main__":
    main()