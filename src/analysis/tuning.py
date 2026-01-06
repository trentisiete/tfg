from __future__ import annotations

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from joblib import Parallel, delayed


def _evaluate_params_logo(model, Xtr, ytr, gtr, logo, params, primary) -> tuple:
    """
    Evaluate model performance with LOGO

    Args:
        model (SurrogateRegressor): Model to evaluate
        Xtr (np.ndarray): Design matrix for training folds
        ytr (np.ndarray): Target values for training folds
        gtr (np.ndarray): Group labels for LODO
        logo (LeaveOneGroupOut): Validation splitter method
        params (dict): Model parameters
        primary (str): Metric to optimize

    Returns:
        tuple: Best parameters and corresponding score
    """
    fold_scores = []

    for tr2, te2 in logo.split(Xtr, ytr, gtr):
        m = clone(model).set_params(**params)
        m.fit(Xtr[tr2], ytr[tr2])

        mean, std = m.predict_dist(Xtr[te2])
        metrics = m.compute_metrics(ytr[te2], mean, std)

        fold_scores.append(metrics[primary]) # Mean of the metric chosen to optimize

    score = float(np.mean(fold_scores)) # Average primary over folds
    return params, score


def inner_cv_select_params(model, Xtr, ytr, gtr, param_grid,
                           primary="mae", n_jobs: int = 1) -> tuple:
    """
    Returns the best results optimizing the inner model.

    Args:
        model (SurrogateRegressor): Model to tune
        Xtr (np.ndarray): Design matrix for training folds
        ytr (np.ndarray): Target values for training folds
        gtr (np.ndarray): Group labels for LODO
        param_grid (dict): Grid of parameters to search
        primary (str, optional): Metric to optimize. Defaults to "mae".
        n_jobs (int, optional): Number of jobs for parallel processing. Defaults to 1.

    Returns:
        _type_: Best params finded and its corresponding score
    """
    logo = LeaveOneGroupOut()

    evaluations = []

    # Working in parallel
    if n_jobs == 1:
        for params in ParameterGrid(param_grid):
            evaluations.append(_evaluate_params_logo(model, Xtr, ytr, gtr, logo, params, primary))
    else:
        evaluations = Parallel(n_jobs=n_jobs, prefer="processes")( # loky backend
            delayed(_evaluate_params_logo)(model, Xtr, ytr, gtr, logo, params, primary)
            for params in ParameterGrid(param_grid)
        )

    best_params, best_score = min(evaluations, key=lambda x: x[1])

    return best_params, best_score


def _run_outer_fold(model, X, y, groups, fold_id, tr, te,
                    param_grid, primary, inner_n_jobs) -> tuple:
    """
    In this case, we receive a fold, a fold is an X and y split where data
    from one group is left out for testing.This function is an auxiliary
    function to run the P fold combinations (which P means the number of groups)
    in the Nested LODO.

    The output is the results of that specific fold.

    Args:
        model (SurrogateRegressor): The model to tune
        X (np.ndarray): Design matrix
        y (np.ndarray): Target values of the specific group
        groups (np.ndarray): Group labels for LODO
        fold_id (int): Fold identifier of the outer fold
        tr (np.ndarray): Training indices
        te (np.ndarray): Testing indices
        param_grid (dict): Grid of parameters to search
        primary (str): Metric to optimize
        inner_n_jobs (int): Number of jobs for inner parallel processing

    Returns:
        tuple: Fold identifier, fold results, and best parameters
    """
    # Xtr, yrt, gtr: Train data in the specific outer fold
    # Xte, yte,gte: Test data in the specific outer fold
    Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
    Xte, yte, gte = X[te], y[te], groups[te]

    # Inner CV with LODO to select best hyperparameters
    best_params, inner_score = inner_cv_select_params(
        model, Xtr, ytr, gtr, param_grid, primary=primary, n_jobs=inner_n_jobs
    )

    # Fit with best params on outer fold
    m = clone(model).set_params(**best_params)
    # Fit model with Xtr, ytr
    m.fit(Xtr, ytr)

    mean, std = m.predict_dist(Xte)
    metrics = m.compute_metrics(yte, mean, std)

    fold = {
        "fold": fold_id,
        "diet": list(set(map(str, gte)))[0],
        "inner_best_score": float(inner_score),
        "params": best_params,
        "metrics": metrics
    }

    return fold_id, fold, best_params


def nested_lodo_tuning(model, X, y, groups,
                       param_grid, primary="mae",
                       inner_n_jobs: int = 1,
                       outer_n_jobs: int = 1) -> dict:
    """
    This is the main function, it runs the outer LODO function and then, the
    outer LODO function runs the inner LODO function in order to find the best
    parameters of that specific fold.
    
    This functions calculates the metrics for each outer fold with the best parameters
    found in the inner fold.

    Args:
        model (SurrogateRegressor): The model to tune
        X (np.ndarray): Design matrix
        y (np.ndarray): Target values
        groups (np.ndarray): Group labels for LODO
        param_grid (dict): Grid of parameters to search
        primary (str, optional): Metric to optimize. Defaults to "mae".
        inner_n_jobs (int, optional): Number of jobs for inner parallel processing. Defaults to 1.
        outer_n_jobs (int, optional): Number of jobs for outer parallel processing. Defaults to 1.

    Returns:
        dict: Dictionary containing folds, summary, and chosen parameters
    """
    outer = LeaveOneGroupOut()

    tasks = [(fold_id, tr, te) for fold_id, (tr, te) in enumerate(outer.split(X, y, groups))]

    if outer_n_jobs == 1:
        results = [
            _run_outer_fold(model, X, y, groups, fold_id, tr, te, param_grid,
                            primary,
                            inner_n_jobs)
            for fold_id, tr, te in tasks
        ]
    else:
        results = Parallel(n_jobs=outer_n_jobs, prefer="processes")(
            delayed(_run_outer_fold)(model, X, y, groups, fold_id, tr, te, param_grid,
                                     primary,
                                     inner_n_jobs)
            for fold_id, tr, te in tasks
        )

    results = sorted(results, key=lambda x: x[0])
    folds = [r[1] for r in results]
    chosen_params = [r[2] for r in results]

    # summary
    mae_vals = [f["metrics"]["mae"] for f in folds]
    rmse_vals = [f["metrics"]["rmse"] for f in folds]
    cov_vals = [f["metrics"]["coverage95"] for f in folds if f["metrics"]["coverage95"] is not None]

    summary = {
        "macro": {
            "mae_mean": float(np.mean(mae_vals)),
            "mae_std": float(np.std(mae_vals)),
            "rmse_mean": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals)),
            "coverage95_mean": None if len(cov_vals)==0 else float(np.mean(cov_vals)),
            "coverage95_std": None if len(cov_vals)==0 else float(np.std(cov_vals)),
        },
        "n_folds": int(len(folds))
    }

    return {"folds": folds, "summary": summary, "chosen_params": chosen_params}
