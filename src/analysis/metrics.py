from time import time
import sklearn
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
import numpy as np
from sklearn.base import clone
from copy import deepcopy
from multiprocessing import Pool
import logging
import warnings
from ..models.pls import PLSSurrogateRegressor
from ..models.ridge import RidgeSurrogateRegressor
from ..models.gp import GPSurrogateRegressor
import sys
from ..utils.tools import _to_jsonable
import json


def make_splits(groups: np.ndarray, X: np.ndarray, y:np.ndarray):
    # Create splits based on Leave-One-Diet/Group-Out cross-validation
    lodo = LeaveOneGroupOut()
    lodo.get_n_splits(groups=groups)

    for i, (train_index, test_index) in enumerate(lodo.split(X, y, groups)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield (X_train, X_test, y_train, y_test, i)

def _evaluate_single_model(args):
    """
    Worker function to evaluate a single model across all splits.
    Must be at module level for multiprocessing pickling.
    """
    name, model, X, y, groups = args
    unique_groups = np.unique(groups).size

    print(f"[{name}] Evaluating model...")

    base_model = model
    t0 = time()

    # Config
    params = _to_jsonable(base_model.get_params(deep=False))
    scores = {
        "config": {
            "model_name": name,
            "params": params
        },
        "folds": {},
        "summary": {} # Macro and Micro
    }

    # Lists for aggregating metrics across folds
    fold_metrics_list = []
    warnings_list = []

    # Micro (sample-weighted) accumulators
    total_samples = 0
    sum_abs = 0.0
    sum_sq = 0.0
    
    # Coverage counts for micro-averaging
    inside_50_total = 0
    inside_90_total = 0
    inside_95_total = 0
    cov_n_total = 0

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            for X_train, X_test, y_train, y_test, fold_id in make_splits(groups, X, y):
                m = clone(base_model)
                m.fit(X_train, y_train)

                mean, std = m.predict_dist(X_test)
                # Use extended=True to get all metrics
                metrics = m.compute_metrics(y_test, mean, std, extended=True)

                # Store per-fold scores (all metrics)
                scores["folds"][str(fold_id)] = {
                    "n_samples": metrics["n_samples"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics.get("r2"),
                    "max_error": metrics.get("max_error"),
                    "nlpd": metrics.get("nlpd"),
                    "coverage_50": metrics.get("coverage_50"),
                    "coverage_90": metrics.get("coverage_90"),
                    "coverage_95": metrics.get("coverage_95"),
                    "coverage95": metrics.get("coverage95"),  # Legacy alias
                    "mean_interval_width_95": metrics.get("mean_interval_width_95"),
                    "calibration_error_95": metrics.get("calibration_error_95"),
                    "sharpness": metrics.get("sharpness"),
                    "score": -metrics["mae"],  # Negate to make it a score (higher is better)
                }
                
                fold_metrics_list.append(metrics)

                # For micro, accumulators
                n = metrics["n_samples"]
                total_samples += n
                sum_abs += metrics["mae"] * n
                sum_sq += (metrics["rmse"] ** 2) * n

                # Coverage counts
                if metrics.get("_inside_50") is not None:
                    inside_50_total += metrics["_inside_50"]
                if metrics.get("_inside_90") is not None:
                    inside_90_total += metrics["_inside_90"]
                if metrics.get("_inside_95") is not None:
                    inside_95_total += metrics["_inside_95"]
                    cov_n_total += n

            for warning in w:
                logging.warning(f"[{name}] {warning.category.__name__}: {warning.message}")

    except Exception as e:
        logging.error(f"[{name}] {e}")
        return (name, {"error": str(e)})

    # Build comprehensive summary
    scores["summary"] = _build_evaluation_summary(
        fold_metrics_list, 
        unique_groups, 
        total_samples,
        sum_abs, sum_sq,
        inside_50_total, inside_90_total, inside_95_total, cov_n_total
    )

    print(f"[{name}] Finished evaluating model.")

    return (name, {"timestamp": float(time() - t0), "results": scores})


def _build_evaluation_summary(fold_metrics_list, n_folds, total_samples,
                               sum_abs, sum_sq, 
                               inside_50, inside_90, inside_95, cov_n):
    """
    Build comprehensive summary from fold metrics.
    
    Args:
        fold_metrics_list: List of metrics dicts from each fold
        n_folds: Number of folds
        total_samples: Total samples across all folds
        sum_abs, sum_sq: Accumulators for MAE/RMSE micro
        inside_50, inside_90, inside_95, cov_n: Coverage accumulators
        
    Returns:
        Summary dict with macro and micro statistics
    """
    # Define metrics to aggregate
    metric_names = [
        'mae', 'rmse', 'r2', 'max_error', 'nlpd',
        'coverage_50', 'coverage_90', 'coverage_95',
        'mean_interval_width_95', 'calibration_error_95', 'sharpness'
    ]
    
    # Macro: mean and std per metric
    macro = {}
    for name in metric_names:
        values = [m.get(name) for m in fold_metrics_list if m.get(name) is not None]
        if values:
            macro[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            macro[name] = {"mean": None, "std": None, "min": None, "max": None}
    
    # Legacy format aliases
    macro["coverage95"] = macro.get("coverage_95", {"mean": None, "std": None})
    
    # Micro: sample-weighted
    micro = {
        "mae": float(sum_abs / max(1, total_samples)),
        "rmse": float(np.sqrt(sum_sq / max(1, total_samples))),
        "coverage_50": float(inside_50 / max(1, cov_n)) if cov_n > 0 else None,
        "coverage_90": float(inside_90 / max(1, cov_n)) if cov_n > 0 else None,
        "coverage_95": float(inside_95 / max(1, cov_n)) if cov_n > 0 else None,
    }
    micro["coverage95"] = micro.get("coverage_95")  # Legacy alias
    
    return {
        "macro": macro,
        "micro": micro,
        "n_folds": n_folds,
        "total_samples": total_samples,
    }


def evaluate_model(models, X: np.ndarray, y: np.ndarray, groups: np.ndarray, save_path: str | None = None) -> dict:
    """
    Evaluate multiple models in parallel using Leave-One-Group-Out CV
    """
    tasks = [(name, model, X, y, groups) for name, model in models.items()]

    # Multiprocessing pool
    with Pool() as pool:
        results_list = pool.map(_evaluate_single_model, tasks)

    results = dict(results_list)

    # Add metadata and version info
    results["metadata"] = {
        "Num_samples": X.shape[0],
        "Num_features": X.shape[1],
        "Num_groups": np.unique(groups).size,
        "timestamp": time(),
        "versions": {
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "python": sys.version
        }
    }
    output = _to_jsonable(results)

    if save_path is not None:
        save_evaluation_results(output, save_path)

    return output

def save_evaluation_results(results: dict, filepath: str):
    """
    Save evaluation results to a JSON file.
    """
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # Toy example
    groups = np.array([0, 0, 1, 1, 2, 2])
    X = np.array([[1, 2, 3, 4, 5, 6],
                  [1, 2, 3, 4, 5, 6],
                  [1, 2, 3, 4, 5, 6]]).T
    y = np.array([1, 2, 3, 4, 5, 6])

    from ..models.pls import PLSSurrogateRegressor
    from ..models.ridge import RidgeSurrogateRegressor
    from ..models.gp import GPSurrogateRegressor

    models = {
        "PLS": PLSSurrogateRegressor(n_components=2, scale=True),
        "Ridge": RidgeSurrogateRegressor(alpha=1.0, fit_intercept=True),
        "GP": GPSurrogateRegressor(),
    }

    results = evaluate_model(models, X, y, groups, save_path="evaluation_results.json")
    print(results)
