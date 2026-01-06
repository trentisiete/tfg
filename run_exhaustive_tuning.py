import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Tune Method
from src.analysis.tuning import nested_lodo_tuning

# Paths and Utils
from src.utils.paths import ENTOMOTIVE_DATA_DIR, LOGS_DIR
from src.utils.tools import _to_jsonable, slugify

# Models and Parameters
from src.configs.tuning_specs import TARGET_MAP, FEATURE_COLS, MODELS, PARAM_GRIDS


# Specific helper Functions

def build_X_y_groups(df: pd.DataFrame, target_col: str):
    data = df.copy()
    data = data.loc[~data[target_col].isna()].reset_index(drop=True)

    groups = data["diet_name"].astype(str).to_numpy()
    y = data[target_col].astype(float).to_numpy()

    byp = pd.get_dummies(data["byproduct_type"], prefix="byproduct", drop_first=False)
    Xdf = pd.concat([data[FEATURE_COLS], byp], axis=1)
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    X = Xdf.to_numpy(dtype=float)
    return X, y, groups, Xdf.columns.tolist()


def setup_logging(log_dir: Path, target_slug: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{target_slug}.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Reset handlers to avoid duplicated logs when looping over targets
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info("Logging initialized -> %s", log_file)
    return log_file


def save_model_outputs(target_dir: Path, target_slug: str, model_name: str, result: dict):
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


def main():
    # Path Setup
    # Logging path
    base_dir = LOGS_DIR / "tuning" / "productivity_hermetia"
    # Data path
    df_path = ENTOMOTIVE_DATA_DIR / "productivity_hermetia_lote.csv"
    df = pd.read_csv(df_path)

    # Parallel settings
    inner_n_jobs = max(1, (os.cpu_count() or 2) - 1)
    outer_n_jobs = min(4, os.cpu_count() or 1)

    for target_label, target_col in TARGET_MAP.items():
        # Create target-specific directory and logging
        target_slug = slugify(target_label)
        target_dir = base_dir / target_slug
        log_file = setup_logging(target_dir, target_slug)

        logging.info("Target=%s (%s)", target_label, target_col)

        if target_col not in df.columns:
            logging.error("Target column not found: %s", target_col)
            continue

        # Build data matrices
        X, y, groups, feature_names = build_X_y_groups(df, target_col)
        logging.info("Samples=%d | Features=%d | Groups=%d", len(y), X.shape[1], len(np.unique(groups)))
        logging.info("Inner jobs=%d | Outer jobs=%d", inner_n_jobs, outer_n_jobs)

        summary = {
            "target_label": target_label,
            "target_column": target_col,
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "n_groups": int(np.unique(groups).size),
            "feature_names": feature_names,
            "inner_n_jobs": inner_n_jobs,
            "outer_n_jobs": outer_n_jobs,
            "models": {},
            "log_file": str(log_file),
        }

        # Por cada modelo, ejecutar el tuning anidado LODO
        for model_name, model in MODELS.items():
            grid_size = int(np.prod([len(v) for v in PARAM_GRIDS[model_name].values()]))
            logging.info("[%s] Grid size=%d", model_name, grid_size)
            logging.info("[%s] Starting nested LODO", model_name)

            result = nested_lodo_tuning(
                model,
                X,
                y,
                groups,
                PARAM_GRIDS[model_name],
                primary="mae",
                inner_n_jobs=inner_n_jobs,
                outer_n_jobs=outer_n_jobs,
            )

            # Save results in the specific target/model directory
            json_path, folds_path = save_model_outputs(target_dir, target_slug, model_name, result)

            # Add the model specific results to the summary
            summary["models"][model_name] = {
                "summary": result.get("summary", {}),
                "chosen_params": result.get("chosen_params", []),
                "files": {
                    "json": str(json_path),
                    "folds_csv": str(folds_path) if folds_path else None,
                },
            }

            macro = result["summary"]["macro"]
            logging.info("[%s] Done. Macro MAE=%.4f | Macro RMSE=%.4f", model_name, macro["mae_mean"], macro["rmse_mean"])

        # Save overall summary
        summary_path = target_dir / f"{target_slug}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(summary), f, indent=2)

        logging.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
