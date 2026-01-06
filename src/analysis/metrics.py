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

    mae_list = []
    rmse_list = []
    cov95_list = []
    warnings_list = []

    # Micro (sample-weighted) in case groups have different sizes
    sum_abs = 0.0
    sum_sq = 0.0
    total_samples = 0

    # Micro coverage95
    inside_total = 0
    cov_n_total = 0

    coverage95_available = True


    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            for X_train, X_test, y_train, y_test, fold_id in make_splits(groups, X, y):
                m = clone(base_model)
                m.fit(X_train, y_train)

                mean, std = m.predict_dist(X_test)
                metrics = m.compute_metrics(y_test, mean, std)

                # Store per-fold scores
                scores["folds"][str(fold_id)] = {
                    "num_samples": metrics["n_samples"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "coverage95": metrics["coverage95"],
                    "score": -metrics["mae"], # Negate to make it a score (higher is better)
                }

                # For macro
                mae_list.append(metrics["mae"])
                rmse_list.append(metrics["rmse"])
                if metrics["coverage95"] is not None:
                    cov95_list.append(metrics["coverage95"])
                else:
                    coverage95_available = False

                # For micro, accumulators
                n = metrics["n_samples"]
                sum_abs += metrics["mae"] * n
                sum_sq += (metrics["rmse"] ** 2) * n
                total_samples += n

                if metrics["_inside95"] is not None:
                    inside_total += metrics["_inside95"]
                    cov_n_total += n


            for warning in w:
                logging.warning(f"[{name}] {warning.category.__name__}: {warning.message}")

    except Exception as e:
        logging.error(f"[{name}] {e}")
        return (name, {"error": str(e)})

    # Compute summary metrics
    macro_mae_mean = float(np.mean(mae_list))
    macro_mae_std = float(np.std(mae_list))

    macro_rmse_mean = float(np.mean(rmse_list))
    macro_rmse_std = float(np.std(rmse_list))

    macro = {
        "mae": {
            "mean": macro_mae_mean,
            "std": macro_mae_std
        },
        "rmse": {
            "mean": macro_rmse_mean,
            "std": macro_rmse_std
        },
        "coverage95": {
            "mean": None,
            "std": None
        }
    }

    micro = {
        "mae": sum_abs / max(1,total_samples),
        "rmse": np.sqrt(sum_sq / total_samples),
        "coverage95": None
    }

    if coverage95_available and len(cov95_list) > 0:
        macro["coverage95"]["mean"] = float(np.mean(cov95_list))
        macro["coverage95"]["std"] = float(np.std(cov95_list))
        if cov_n_total > 0:
            micro["coverage95"] = float(inside_total / cov_n_total)


    scores["summary"] = {
        "macro": macro,
        "micro": micro,
        "n_folds": unique_groups,
    }

    print(f"[{name}] Finished evaluating model.")

    return (name, {"timestamp": float(time() - t0), "results": scores})


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
