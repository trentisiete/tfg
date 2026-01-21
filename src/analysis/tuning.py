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
        # Use extended=False for inner CV (faster, only basic metrics needed)
        metrics = m.compute_metrics(ytr[te2], mean, std, extended=False)

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
    # Use extended=True for outer fold (full metrics for final evaluation)
    metrics = m.compute_metrics(yte, mean, std, extended=True)

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

    # Build comprehensive summary with all metrics
    summary = _build_tuning_summary(folds)

    return {"folds": folds, "summary": summary, "chosen_params": chosen_params}


def _build_tuning_summary(folds: list) -> dict:
    """
    Build comprehensive summary statistics from fold results.
    
    Aggregates all available metrics (basic and extended) across folds.
    
    Args:
        folds: List of fold result dictionaries with 'metrics' key
        
    Returns:
        dict with 'macro' (mean/std per metric) and metadata
    """
    # Define all metrics to aggregate
    # Basic metrics (always present)
    basic_metrics = ['mae', 'rmse']
    
    # Extended metrics (may be None for some models)
    extended_metrics = [
        'r2', 'max_error', 'nlpd',
        'coverage_50', 'coverage_90', 'coverage_95',
        'mean_interval_width_95', 'median_interval_width_95',
        'calibration_error_95', 'sharpness'
    ]
    
    all_metrics = basic_metrics + extended_metrics
    
    # Build macro summary
    macro = {}
    for metric_name in all_metrics:
        values = []
        for f in folds:
            val = f["metrics"].get(metric_name)
            if val is not None:
                values.append(val)
        
        if len(values) > 0:
            macro[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            macro[metric_name] = {"mean": None, "std": None, "min": None, "max": None}
    
    # Legacy format compatibility (coverage95 alias)
    if 'coverage_95' in macro:
        macro['coverage95'] = macro['coverage_95']
    
    # Build micro summary (sample-weighted)
    micro = _build_micro_summary(folds)
    
    return {
        "macro": macro,
        "micro": micro,
        "n_folds": int(len(folds)),
        "total_samples": sum(f["metrics"].get("n_samples", 0) for f in folds),
    }


def _build_micro_summary(folds: list) -> dict:
    """
    Build micro (sample-weighted) summary statistics.
    
    Args:
        folds: List of fold result dictionaries
        
    Returns:
        dict with sample-weighted metrics
    """
    total_samples = 0
    sum_abs = 0.0
    sum_sq = 0.0
    
    # For coverage micro-averaging
    inside_50_total = 0
    inside_90_total = 0
    inside_95_total = 0
    cov_n_total = 0
    
    # For R² micro (weighted)
    r2_weighted_sum = 0.0
    r2_weight_total = 0
    
    for f in folds:
        m = f["metrics"]
        n = m.get("n_samples", 0)
        
        if n > 0:
            total_samples += n
            sum_abs += m.get("mae", 0) * n
            sum_sq += (m.get("rmse", 0) ** 2) * n
            
            # R² weighted
            if m.get("r2") is not None:
                r2_weighted_sum += m["r2"] * n
                r2_weight_total += n
            
            # Coverage counts
            if m.get("_inside_50") is not None:
                inside_50_total += m["_inside_50"]
            if m.get("_inside_90") is not None:
                inside_90_total += m["_inside_90"]
            if m.get("_inside_95") is not None:
                inside_95_total += m["_inside_95"]
                cov_n_total += n
    
    micro = {
        "mae": float(sum_abs / max(1, total_samples)),
        "rmse": float(np.sqrt(sum_sq / max(1, total_samples))),
        "r2": float(r2_weighted_sum / max(1, r2_weight_total)) if r2_weight_total > 0 else None,
        "coverage_50": float(inside_50_total / max(1, cov_n_total)) if cov_n_total > 0 else None,
        "coverage_90": float(inside_90_total / max(1, cov_n_total)) if cov_n_total > 0 else None,
        "coverage_95": float(inside_95_total / max(1, cov_n_total)) if cov_n_total > 0 else None,
    }
    
    # Legacy alias
    micro["coverage95"] = micro.get("coverage_95")
    
    return micro
