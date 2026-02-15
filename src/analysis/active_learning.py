from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, ParameterGrid

from ..benchmarks import get_benchmark, get_sampler
from ..benchmarks.noise import get_noise_injector
from ..configs import get_grid_for_evaluation
from ..models.base import SurrogateRegressor
from .surrogate_metrics import compute_surrogate_metrics
from ..utils.tools import _to_jsonable


class _PredictDistAdapter:
    """Adapter exposing sklearn-like predict(..., return_std=True) for skopt EI."""

    def __init__(self, model: SurrogateRegressor):
        self.model = model

    def predict(self, X: np.ndarray, return_std: bool = False):
        mean, std = self.model.predict_dist(X)
        mean = np.asarray(mean).ravel()

        if not return_std:
            return mean

        if std is None:
            std = np.zeros_like(mean)
        else:
            std = np.asarray(std).ravel()

        return mean, std


@dataclass
class ActiveStepRecord:
    step: int
    n_train_current: int
    incumbent_best: float
    ei_next: float
    x_next: List[float]
    y_next: float
    fit_time_step: float
    predict_time_step: float
    mae_test: Optional[float]
    rmse_test: Optional[float]
    r2_test: Optional[float]
    nlpd_test: Optional[float]
    coverage_95_test: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self.__dict__)


def is_active_supported(model: SurrogateRegressor, X_probe: np.ndarray) -> bool:
    """
    Check if a model supports active-learning uncertainty path.

    Support means predict_dist provides a non-None std vector once fitted.
    """
    # Fast fail for models using base-class predict_dist (std=None by design).
    if model.__class__.predict_dist is SurrogateRegressor.predict_dist:
        return False

    try:
        _, std = model.predict_dist(X_probe)
        return std is not None
    except Exception:
        # If model is not fitted yet or probe fails, this is inconclusive.
        # Caller should fit once before final support decision.
        return True


def ei_values_from_model(
    model: SurrogateRegressor,
    X_candidates: np.ndarray,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Compute Expected Improvement values using scikit-optimize interface."""
    from skopt.acquisition import gaussian_ei

    adapter = _PredictDistAdapter(model)
    ei = gaussian_ei(X_candidates, model=adapter, y_opt=float(y_best), xi=float(xi))
    return np.asarray(ei).ravel()


def _manual_ei_fallback(
    model: SurrogateRegressor,
    X_candidates: np.ndarray,
    y_best: float,
    xi: float,
) -> np.ndarray:
    """Numerically stable fallback if skopt EI call fails."""
    from scipy.stats import norm

    mu, sigma = model.predict_dist(X_candidates)
    mu = np.asarray(mu).ravel()
    sigma = np.asarray(sigma).ravel()
    sigma = np.clip(sigma, 1e-12, None)

    improvement = float(y_best) - mu - float(xi)
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def select_next_x(
    model: SurrogateRegressor,
    bounds: List[tuple],
    y_best: float,
    xi: float,
    n_candidates: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Select next point by continuous EI maximization (differential evolution).

    Returns dict with x_next, ei_next, and info flags.
    """
    from scipy.optimize import differential_evolution

    dim = len(bounds)
    budget = int(max(200, n_candidates))
    popsize = max(8, min(20, int(np.ceil(np.sqrt(budget / max(1, dim * 20))))))
    maxiter = max(10, int(np.ceil(budget / max(1, popsize * dim))) - 1)

    used_fallback = False
    best_x = None
    best_ei = None
    try:
        def objective(x_vec: np.ndarray) -> float:
            x_arr = np.asarray(x_vec, dtype=float).reshape(1, -1)
            try:
                ei_val = ei_values_from_model(model, x_arr, y_best, xi=xi)[0]
            except Exception:
                ei_val = _manual_ei_fallback(model, x_arr, y_best, xi=xi)[0]

            if not np.isfinite(ei_val):
                return 1e12
            return -float(ei_val)

        de_res = differential_evolution(
            objective,
            bounds=bounds,
            strategy="best1bin",
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=seed,
            polish=True,
            updating="deferred",
            workers=1,
        )
        best_x = np.asarray(de_res.x, dtype=float).ravel()
        best_ei = float(-de_res.fun)
    except Exception:
        # Fallback: random-search EI argmax (no Sobol in active selection).
        rng = np.random.default_rng(seed)
        lb = np.asarray([b[0] for b in bounds], dtype=float)
        ub = np.asarray([b[1] for b in bounds], dtype=float)
        X_candidates = rng.uniform(low=lb, high=ub, size=(budget, dim))
        try:
            ei = ei_values_from_model(model, X_candidates, y_best, xi=xi)
        except Exception:
            ei = _manual_ei_fallback(model, X_candidates, y_best, xi=xi)
        used_fallback = True
        best_idx = int(np.argmax(ei))
        best_x = X_candidates[best_idx]
        best_ei = float(ei[best_idx])

    x_next = np.asarray(best_x, dtype=float).ravel()
    ei_next = float(best_ei)

    return {
        "x_next": x_next,
        "ei_next": ei_next,
        "optimizer_budget": int(budget),
        "optimizer_popsize": int(popsize),
        "optimizer_maxiter": int(maxiter),
        "used_ei_fallback": used_fallback,
    }


def _extract_mll_state(model: SurrogateRegressor) -> Dict[str, Any]:
    """Extract fitted GP state to compare with periodic CV audit."""
    state: Dict[str, Any] = {
        "model_params": _to_jsonable(model.get_params(deep=False)),
    }

    # Best-effort extraction for sklearn GPR inside the pipeline.
    try:
        gpr = model.model_.named_steps["model"]
        state["optimized_kernel"] = str(getattr(gpr, "kernel_", None))
        lmll = getattr(gpr, "log_marginal_likelihood_value_", None)
        state["log_marginal_likelihood_value"] = None if lmll is None else float(lmll)
    except Exception:
        pass

    return state


def _grid_model_key(model_name: str) -> str:
    if model_name.upper().startswith("GP"):
        return "GP"
    return model_name


def _score_params_cv_mae(
    model_template: SurrogateRegressor,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: KFold,
) -> Dict[str, Any]:
    """Score one hyperparameter set with KFold CV MAE."""
    fold_mae: List[float] = []
    fold_errors: List[str] = []

    for train_idx, valid_idx in cv.split(X_train):
        m = clone(model_template)
        try:
            m.set_params(**params)
            m.fit(X_train[train_idx], y_train[train_idx])
            y_pred = m.predict(X_train[valid_idx])
            fold_mae.append(float(mean_absolute_error(y_train[valid_idx], y_pred)))
        except Exception as exc:
            fold_errors.append(str(exc))

    return {
        "mean_mae": float(np.mean(fold_mae)) if fold_mae else None,
        "fold_mae": fold_mae,
        "errors": fold_errors,
    }


def run_cv_audit(
    model_template: SurrogateRegressor,
    benchmark_name: str,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    step: int,
    use_default_grids: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Periodic CV-MAE audit for logging only.

    This does not mutate active-learning model selection.
    """
    n_samples = int(len(y_train))
    if n_samples < 3:
        return {
            "step": step,
            "status": "skipped",
            "reason": "insufficient_samples_for_cv",
            "n_samples": n_samples,
        }

    model_key = _grid_model_key(model_name)
    grid = get_grid_for_evaluation(benchmark_name, model_key, use_defaults=use_default_grids)

    if not grid:
        return {
            "step": step,
            "status": "skipped",
            "reason": "no_grid_for_model",
            "model_key": model_key,
        }

    n_splits = min(3, n_samples)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed + step)

    ranking: List[Dict[str, Any]] = []
    for params in ParameterGrid(grid):
        scored = _score_params_cv_mae(
            model_template=model_template,
            params=params,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
        )
        ranking.append(
            {
                "params": _to_jsonable(params),
                "params_raw": dict(params),
                "fold_mae": scored["fold_mae"],
                "mean_mae": scored["mean_mae"],
                "errors": scored["errors"],
            }
        )

    ranking.sort(
        key=lambda rec: (
            rec["mean_mae"] is None,
            float("inf") if rec["mean_mae"] is None else rec["mean_mae"],
        )
    )

    best_cv = ranking[0] if ranking else None
    return {
        "step": step,
        "status": "ok",
        "metric": "mae",
        "model_name": model_name,
        "model_key": model_key,
        "n_splits": n_splits,
        "n_candidates": len(ranking),
        "best_cv": best_cv,
        "ranking": ranking,
    }


def build_active_step_record(
    step: int,
    n_train_current: int,
    incumbent_best: float,
    ei_next: float,
    x_next: np.ndarray,
    y_next: float,
    fit_time_step: float,
    predict_time_step: float,
    metrics,
) -> Dict[str, Any]:
    rec = ActiveStepRecord(
        step=int(step),
        n_train_current=int(n_train_current),
        incumbent_best=float(incumbent_best),
        ei_next=float(ei_next),
        x_next=np.asarray(x_next).ravel().tolist(),
        y_next=float(y_next),
        fit_time_step=float(fit_time_step),
        predict_time_step=float(predict_time_step),
        mae_test=metrics.mae,
        rmse_test=metrics.rmse,
        r2_test=metrics.r2,
        nlpd_test=metrics.nlpd,
        coverage_95_test=metrics.coverage_95,
    )
    return rec.to_dict()


def build_active_final_record(
    model_name: str,
    trajectory: List[Dict[str, Any]],
    hyperparam_audit: List[Dict[str, Any]],
    active_supported: bool = True,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    if not trajectory:
        return {
            "model": model_name,
            "active_supported": active_supported,
            "reason": reason or "empty_trajectory",
            "trajectory": [],
            "hyperparam_audit": hyperparam_audit,
        }

    last = trajectory[-1]
    final_metrics = {
        "mae": last.get("mae_test"),
        "rmse": last.get("rmse_test"),
        "r2": last.get("r2_test"),
        "nlpd": last.get("nlpd_test"),
        "coverage_95": last.get("coverage_95_test"),
    }

    return {
        "model": model_name,
        "active_supported": active_supported,
        "reason": reason,
        "n_steps": len(trajectory),
        "mae": final_metrics["mae"],
        "rmse": final_metrics["rmse"],
        "r2": final_metrics["r2"],
        "nlpd": final_metrics["nlpd"],
        "coverage_95": final_metrics["coverage_95"],
        "fit_time": float(np.sum([r.get("fit_time_step", 0.0) for r in trajectory])),
        "predict_time": float(np.sum([r.get("predict_time_step", 0.0) for r in trajectory])),
        "final_metrics": final_metrics,
        "trajectory": trajectory,
        "hyperparam_audit": hyperparam_audit,
    }


def run_active_evaluation(
    dataset,
    models: Dict[str, SurrogateRegressor],
    benchmark_name: str,
    sampler: str,
    noise_type: str,
    noise_kwargs: Optional[Dict[str, Any]],
    n_infill: int,
    xi: float = 0.01,
    active_cand_mult: int = 500,
    active_cv_check_every: int = 5,
    active_switch_enable: bool = True,
    active_switch_warmup_steps: int = 3,
    active_switch_min_improvement: float = 0.01,
    active_switch_cooldown_steps: int = 3,
    use_default_grids: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run sequential active learning using EI for each model.

    The model fit (MLL-based for GP) is primary. Periodic CV audit can
    optionally switch to better benchmark-grid GP settings with guardrails.
    """
    bench = get_benchmark(benchmark_name)
    noise_kwargs = noise_kwargs or {}
    noise_injector = get_noise_injector(noise_type, seed=seed, **noise_kwargs)
    active_switch_warmup_steps = int(max(0, active_switch_warmup_steps))
    active_switch_cooldown_steps = int(max(0, active_switch_cooldown_steps))
    active_switch_min_improvement = float(max(0.0, active_switch_min_improvement))
    active_switch_enable = bool(active_switch_enable)

    optimizer_budget = max(2000, int(active_cand_mult) * int(bench.dim))
    results: Dict[str, Any] = {}

    if verbose:
        print(
            f"    [active] benchmark={benchmark_name} dim={bench.dim} "
            f"sampler={sampler} noise={noise_type} n_infill={n_infill} "
            f"optimizer_budget={optimizer_budget}"
        )

    for model_name, model in models.items():
        if verbose:
            print(f"    [active] model={model_name} start")

        X_train = np.asarray(dataset.X_train)
        y_train = np.asarray(dataset.y_train).ravel()

        # Safety: active should start with at least one point.
        if len(X_train) == 0:
            init_sampler = get_sampler(sampler, seed=seed)
            x0 = init_sampler.sample_bounds(1, bench.bounds)
            y0_clean = bench(x0)
            y0 = noise_injector.add_noise(y0_clean, x0)
            X_train = x0
            y_train = np.asarray(y0).ravel()

        m = clone(model)
        try:
            m.fit(X_train, y_train)
        except Exception as exc:
            results[model_name] = {
                "model": model_name,
                "active_supported": False,
                "reason": f"fit_failed:{exc}",
                "trajectory": [],
                "hyperparam_audit": [],
            }
            continue

        probe = dataset.X_test[: min(3, len(dataset.X_test))]
        if len(probe) == 0:
            probe = X_train[:1]

        support_flag = is_active_supported(m, probe)
        if support_flag:
            try:
                _, std_probe = m.predict_dist(probe)
                support_flag = std_probe is not None
            except Exception:
                support_flag = False

        if not support_flag:
            if verbose:
                print(f"    [active] model={model_name} unsupported: no predictive std")
            results[model_name] = {
                "model": model_name,
                "active_supported": False,
                "reason": "no_predictive_uncertainty",
                "trajectory": [],
                "hyperparam_audit": [],
            }
            continue

        trajectory: List[Dict[str, Any]] = []
        audits: List[Dict[str, Any]] = []
        last_switch_step = -10**9
        switch_count = 0

        # Track currently selected params over benchmark-specific grid keys.
        model_key = _grid_model_key(model_name)
        model_grid = get_grid_for_evaluation(
            benchmark_name,
            model_key,
            use_defaults=use_default_grids,
        )
        grid_keys = list(model_grid.keys()) if isinstance(model_grid, dict) else []
        all_params = m.get_params(deep=False)
        current_selected_params: Dict[str, Any] = {
            k: all_params[k] for k in grid_keys if k in all_params
        }

        for step in range(1, int(n_infill) + 1):
            incumbent_best = float(np.min(y_train))
            selection = select_next_x(
                model=m,
                bounds=bench.bounds,
                y_best=incumbent_best,
                xi=xi,
                n_candidates=optimizer_budget,
                seed=seed + step * 1009,
            )

            x_next = np.asarray(selection["x_next"]).reshape(1, -1)
            y_next_clean = bench(x_next)
            y_next = noise_injector.add_noise(y_next_clean, x_next)
            y_next_scalar = float(np.asarray(y_next).ravel()[0])

            X_train = np.vstack([X_train, x_next])
            y_train = np.concatenate([y_train, np.asarray(y_next).ravel()])

            t0 = time.perf_counter()
            m.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            mean_test, std_test = m.predict_dist(dataset.X_test)
            predict_time = time.perf_counter() - t0

            metrics = compute_surrogate_metrics(
                y_true=dataset.y_test_clean,
                y_pred=mean_test,
                std_pred=std_test,
            )

            step_rec = build_active_step_record(
                step=step,
                n_train_current=len(y_train),
                incumbent_best=incumbent_best,
                ei_next=selection["ei_next"],
                x_next=x_next[0],
                y_next=y_next_scalar,
                fit_time_step=fit_time,
                predict_time_step=predict_time,
                metrics=metrics,
            )
            step_rec["used_ei_fallback"] = bool(selection.get("used_ei_fallback", False))
            step_rec["optimizer_budget"] = int(selection.get("optimizer_budget", optimizer_budget))
            step_rec["optimizer_popsize"] = int(selection.get("optimizer_popsize", 0))
            step_rec["optimizer_maxiter"] = int(selection.get("optimizer_maxiter", 0))
            trajectory.append(step_rec)

            if verbose:
                print(
                    f"      step={step:03d} best={incumbent_best:.6f} "
                    f"ei={step_rec['ei_next']:.6f} y_next={y_next_scalar:.6f} "
                    f"rmse={step_rec['rmse_test']:.6f}"
                )

            if active_cv_check_every > 0 and step % active_cv_check_every == 0:
                audit = run_cv_audit(
                    model_template=model,
                    benchmark_name=benchmark_name,
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    step=step,
                    use_default_grids=use_default_grids,
                    seed=seed,
                )
                audit["timestamp"] = datetime.now().isoformat()
                audit["current_mll_state"] = _extract_mll_state(m)

                best_cv_params = None
                best_cv_params_raw = None
                best_cv_mean = None
                if audit.get("status") == "ok" and audit.get("best_cv") is not None:
                    best_cv_params = audit["best_cv"].get("params")
                    best_cv_params_raw = audit["best_cv"].get("params_raw")
                    best_cv_mean = audit["best_cv"].get("mean_mae")

                current_params = _to_jsonable(current_selected_params)
                current_cv_mean = None
                if audit.get("status") == "ok":
                    for rec in audit.get("ranking", []):
                        if rec.get("params") == current_params:
                            current_cv_mean = rec.get("mean_mae")
                            break

                # If current params are not present in the configured grid, score them explicitly.
                if (
                    current_cv_mean is None
                    and audit.get("status") == "ok"
                    and current_selected_params
                    and int(audit.get("n_splits", 0)) >= 2
                ):
                    cv = KFold(
                        n_splits=int(audit["n_splits"]),
                        shuffle=True,
                        random_state=seed + step,
                    )
                    scored_current = _score_params_cv_mae(
                        model_template=model,
                        params=current_selected_params,
                        X_train=X_train,
                        y_train=y_train,
                        cv=cv,
                    )
                    current_cv_mean = scored_current.get("mean_mae")

                audit["best_cv_params"] = best_cv_params
                audit["best_cv_mean"] = best_cv_mean
                audit["current_selected_params"] = current_params
                audit["current_cv_mean"] = current_cv_mean
                audit["disagreement"] = (
                    best_cv_params is not None and best_cv_params != current_params
                )

                params_differ = (
                    best_cv_params is not None and best_cv_params != current_params
                )
                cooldown_ok = (step - last_switch_step) >= active_switch_cooldown_steps
                warmup_ok = step >= active_switch_warmup_steps
                switch_improvement = None
                better_enough = False
                if (
                    best_cv_mean is not None
                    and current_cv_mean is not None
                    and np.isfinite(best_cv_mean)
                    and np.isfinite(current_cv_mean)
                ):
                    switch_improvement = float(current_cv_mean - best_cv_mean)
                    better_enough = switch_improvement >= active_switch_min_improvement

                should_switch = (
                    active_switch_enable
                    and params_differ
                    and warmup_ok
                    and cooldown_ok
                    and best_cv_params_raw is not None
                    and (
                        better_enough
                        or (
                            current_cv_mean is None
                            and best_cv_mean is not None
                            and np.isfinite(best_cv_mean)
                            and active_switch_min_improvement == 0.0
                        )
                    )
                )

                audit["switch_applied"] = False
                audit["switch_reason"] = "kept_current_mll"
                audit["switch_improvement"] = switch_improvement
                audit["switch_count"] = switch_count

                if should_switch:
                    t_switch = time.perf_counter()
                    m.set_params(**best_cv_params_raw)
                    m.fit(X_train, y_train)
                    switch_fit_time = time.perf_counter() - t_switch

                    if trajectory:
                        trajectory[-1]["fit_time_step"] = float(
                            trajectory[-1].get("fit_time_step", 0.0) + switch_fit_time
                        )

                    current_selected_params = dict(best_cv_params_raw)
                    last_switch_step = step
                    switch_count += 1

                    audit["switch_applied"] = True
                    audit["switch_reason"] = "cv_better_than_current"
                    audit["switch_fit_time"] = switch_fit_time
                    audit["switch_count"] = switch_count
                    audit["post_switch_mll_state"] = _extract_mll_state(m)

                    if verbose:
                        imp = (
                            f"{switch_improvement:.6f}"
                            if switch_improvement is not None
                            else "n/a"
                        )
                        print(
                            f"      audit step={step:03d} switched_to_cv_best "
                            f"(improvement={imp})"
                        )
                else:
                    if not active_switch_enable:
                        audit["switch_reason"] = "switch_disabled"
                    elif not params_differ:
                        audit["switch_reason"] = "already_best_cv_params"
                    elif not warmup_ok:
                        audit["switch_reason"] = "warmup_not_reached"
                    elif not cooldown_ok:
                        audit["switch_reason"] = "cooldown_not_reached"
                    elif best_cv_params_raw is None:
                        audit["switch_reason"] = "no_cv_best_params"
                    else:
                        audit["switch_reason"] = "improvement_below_threshold"

                # Keep logs JSON-friendly and compact.
                for rec in audit.get("ranking", []):
                    rec.pop("params_raw", None)
                if audit.get("best_cv") is not None:
                    audit["best_cv"].pop("params_raw", None)
                audits.append(audit)

        results[model_name] = build_active_final_record(
            model_name=model_name,
            trajectory=trajectory,
            hyperparam_audit=audits,
            active_supported=True,
        )

    return results
