# @author: José Arbelaez
"""
Benchmark evaluation runner for synthetic datasets.

Provides functions to evaluate surrogate models on benchmark functions
with consistent metrics and reporting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone

from ..benchmarks import (
    SyntheticDataset,
    generate_benchmark_dataset,
    generate_multi_benchmark_suite,
    list_benchmarks,
)
from ..models.base import SurrogateRegressor
from .surrogate_metrics import (
    SurrogateMetrics,
    compute_surrogate_metrics,
    aggregate_metrics,
)
from ..utils.tools import _to_jsonable, slugify
from ..utils.paths import LOGS_DIR


@dataclass
class BenchmarkResult:
    """
    Container for benchmark evaluation results.

    Stores model performance on a single benchmark/noise combination.
    """
    benchmark_name: str
    noise_type: str
    model_name: str

    # Core metrics
    metrics: SurrogateMetrics = None

    # Timing
    fit_time: float = 0.0
    predict_time: float = 0.0

    # Model config
    model_params: Dict = field(default_factory=dict)

    # Optional: predictions for visualization
    y_pred: Optional[np.ndarray] = None
    std_pred: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Export to dictionary."""
        return {
            "benchmark": self.benchmark_name,
            "noise": self.noise_type,
            "model": self.model_name,
            "metrics": self.metrics.to_dict() if self.metrics else {},
            "fit_time_s": self.fit_time,
            "predict_time_s": self.predict_time,
            "model_params": _to_jsonable(self.model_params),
        }


@dataclass
class BenchmarkSuiteResults:
    """
    Container for evaluation results across multiple benchmarks.
    This class is important to maintain the robustness of the tests, each
    benchmark result is stored the same.
    """
    results: List[BenchmarkResult] = field(default_factory=list)

    # Summary
    suite_name: str = ""
    timestamp: str = ""
    total_time: float = 0.0

    def add(self, result: BenchmarkResult):
        """Add a single result in the list.
        """
        self.results.append(result)

    def get_summary_df(self) -> pd.DataFrame:
        """
        Get summary DataFrame of all results.

        Returns:
            DataFrame with one row per (benchmark, noise, model) combination
        """
        rows = []
        for r in self.results: # each element of the list
            row = {
                "benchmark": r.benchmark_name,
                "noise": r.noise_type,
                "model": r.model_name,
                "mae": r.metrics.mae if r.metrics else np.nan,
                "rmse": r.metrics.rmse if r.metrics else np.nan,
                "r2": r.metrics.r2 if r.metrics else np.nan,
                "nlpd": r.metrics.nlpd if r.metrics else np.nan,
                "coverage_95": r.metrics.coverage_95 if r.metrics else np.nan,
                "calibration_error": r.metrics.calibration_error_95 if r.metrics else np.nan,
                "fit_time": r.fit_time,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def get_model_ranking(self, metric: str = "rmse",
                          ascending: bool = True) -> pd.DataFrame:
        """
        Rank models by a specific metric across all benchmarks.

        Args:
            metric: Metric to rank by
            ascending: If True, lower is better (RMSE, MAE)

        Returns:
            DataFrame with average ranks and scores
        """
        df = self.get_summary_df()

        # Compute rank per benchmark/noise
        df["rank"] = df.groupby(["benchmark", "noise"])[metric].rank(ascending=ascending)

        # Average rank per model
        ranking = df.groupby("model").agg({
            "rank": "mean",
            metric: ["mean", "std"],
        }).round(4)

        ranking.columns = ["avg_rank", f"{metric}_mean", f"{metric}_std"]
        return ranking.sort_values("avg_rank")

    def to_json(self, path: Path):
        """Save results to JSON file."""
        data = {
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "total_time_s": self.total_time,
            "n_results": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def evaluate_model_on_dataset(
    model: SurrogateRegressor,
    dataset: SyntheticDataset,
    store_predictions: bool = False,
) -> BenchmarkResult:
    """
    Evaluate a single model on a synthetic dataset.

    Args:
        model: Surrogate model instance (will be cloned)
        dataset: SyntheticDataset with train/test data
        store_predictions: Whether to store y_pred, std_pred in result

    Returns:
        BenchmarkResult with metrics and timing
    """
    m = clone(model)

    # Fit
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(dataset.X_train, dataset.y_train)
    fit_time = time.perf_counter() - t0

    # Predict
    t0 = time.perf_counter()
    mean_pred, std_pred = m.predict_dist(dataset.X_test)
    predict_time = time.perf_counter() - t0

    # Compute metrics (against noisy test values for realistic eval)
    # But also can compare to y_test_clean for "true" error
    metrics = compute_surrogate_metrics(
        y_true=dataset.y_test_clean,  # Compare to clean values
        y_pred=mean_pred,
        std_pred=std_pred,
    )

    result = BenchmarkResult(
        benchmark_name=dataset.benchmark_name,
        noise_type=dataset.noise_type,
        model_name=model.name,
        metrics=metrics,
        fit_time=fit_time,
        predict_time=predict_time,
        model_params=model.get_params(deep=False),
    )

    if store_predictions:
        result.y_pred = mean_pred
        result.std_pred = std_pred

    return result


def evaluate_models_on_suite(
    models: Dict[str, SurrogateRegressor],
    suite: Dict[str, Dict[str, SyntheticDataset]],
    verbose: bool = True,
    store_predictions: bool = False,
) -> BenchmarkSuiteResults:
    """
    Evaluate multiple models on a benchmark suite.

    Args:
        models: Dict of {model_name: model_instance}
        suite: Nested dict from generate_multi_benchmark_suite
        verbose: Print progress
        store_predictions: Store predictions in results

    Returns:
        BenchmarkSuiteResults with all evaluations

    Example:
        >>> models = {
        ...     "GP": GPSurrogateRegressor(),
        ...     "Ridge": RidgeSurrogateRegressor(), # Now ridge is deprecated
        ... }
        >>> suite = generate_multi_benchmark_suite(benchmarks=["forrester", "branin"])
        >>> results = evaluate_models_on_suite(models, suite)
        >>> print(results.get_summary_df())
    """
    from datetime import datetime

    all_results = BenchmarkSuiteResults(
        suite_name="benchmark_evaluation",
        timestamp=datetime.now().isoformat(),
    )

    t_start = time.perf_counter()

    n_total = sum(len(noise_dict) for noise_dict in suite.values()) * len(models)
    n_done = 0

    for bench_name, noise_datasets in suite.items():
        for noise_type, dataset in noise_datasets.items():
            for model_name, model in models.items():
                if verbose:
                    n_done += 1
                    print(f"[{n_done}/{n_total}] {bench_name}/{noise_type}/{model_name}...",
                          end=" ", flush=True)

                try:
                    result = evaluate_model_on_dataset(
                        model=model,
                        dataset=dataset,
                        store_predictions=store_predictions,
                    )
                    all_results.add(result)

                    if verbose:
                        print(f"RMSE={result.metrics.rmse:.4f}, "
                              f"R²={result.metrics.r2:.3f}", flush=True)

                except Exception as e:
                    if verbose:
                        print(f"ERROR: {e}", flush=True)

    all_results.total_time = time.perf_counter() - t_start

    return all_results


def run_quick_benchmark(
    models: Dict[str, SurrogateRegressor],
    benchmarks: List[str] = None,
    n_train: int = 50,
    n_test: int = 200,
    noise_sigma: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> BenchmarkSuiteResults:
    """
    Quick benchmark evaluation with sensible defaults.

    Convenience function for rapid model comparison.

    Args:
        models: Dict of models to evaluate
        benchmarks: List of benchmark names (default: low-dim benchmarks)
        n_train: Training samples per benchmark
        n_test: Test samples per benchmark
        noise_sigma: Gaussian noise level
        seed: Random seed
        verbose: Print progress

    Returns:
        BenchmarkSuiteResults

    Example:
        >>> from src.models.gp import GPSurrogateRegressor
        >>> from src.models.ridge import RidgeSurrogateRegressor
        >>>
        >>> results = run_quick_benchmark(
        ...     models={"GP": GPSurrogateRegressor(), "Ridge": RidgeSurrogateRegressor()},
        ...     benchmarks=["forrester", "branin"],
        ... )
        >>> print(results.get_model_ranking("rmse"))
    """
    if benchmarks is None:
        benchmarks = ["forrester", "branin", "sixhump", "hartmann3"]

    noise_configs = [
        {"type": "none"},
        {"type": "gaussian", "sigma": noise_sigma},
    ]

    if verbose:
        print("=" * 60)
        print("QUICK BENCHMARK EVALUATION")
        print(f"Benchmarks: {benchmarks}")
        print(f"Models: {list(models.keys())}")
        print(f"Noise configs: {noise_configs}")
        print("=" * 60)

    suite = generate_multi_benchmark_suite(
        benchmarks=benchmarks,
        n_train=n_train,
        n_test=n_test,
        noise_configs=noise_configs,
        seed=seed,
    )

    results = evaluate_models_on_suite(
        models=models,
        suite=suite,
        verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("MODEL RANKING (by RMSE)")
        print("=" * 60)
        print(results.get_model_ranking("rmse"))

        print("\n" + "=" * 60)
        print("MODEL RANKING (by R²)")
        print("=" * 60)
        print(results.get_model_ranking("r2", ascending=False))

    return results


def evaluate_model_with_lodo(
    model: SurrogateRegressor,
    dataset: SyntheticDataset,
) -> Dict[str, Any]:
    """
    Evaluate a model using Leave-One-Group-Out CV on a synthetic dataset.

    This mirrors the real data evaluation pipeline, using synthetic groups
    to perform LODO cross-validation on benchmark data.

    Args:
        model: Surrogate model instance (will be cloned for each fold)
        dataset: SyntheticDataset with groups_train assigned

    Returns:
        Dict with 'folds' (per-fold metrics) and 'summary' (macro/micro aggregates)
        Same structure as nested_lodo_tuning results.

    Raises:
        ValueError: If dataset has no groups assigned

    Example:
        >>> dataset = generate_benchmark_dataset(
        ...     benchmark='branin', n_train=100, n_groups=5
        ... )
        >>> results = evaluate_model_with_lodo(GPSurrogateRegressor(), dataset)
        >>> print(results['summary']['macro']['mae'])
    """
    if dataset.groups_train is None:
        raise ValueError(
            "Dataset has no groups. Use n_groups parameter when generating: "
            "generate_benchmark_dataset(..., n_groups=5)"
        )

    from sklearn.model_selection import LeaveOneGroupOut

    X = dataset.X_train
    y = dataset.y_train
    groups = dataset.groups_train

    logo = LeaveOneGroupOut()
    folds = []

    # Accumulators for micro-averaging
    total_samples = 0
    sum_abs = 0.0
    sum_sq = 0.0
    inside_50_total = 0
    inside_90_total = 0
    inside_95_total = 0
    cov_n_total = 0

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone and fit
        m = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X_train, y_train)

        # Predict
        mean_pred, std_pred = m.predict_dist(X_test)

        # Compute extended metrics
        metrics = m.compute_metrics(y_test, mean_pred, std_pred, extended=True)

        folds.append({
            "fold_id": fold_idx,
            "group": groups[test_idx][0] if len(test_idx) > 0 else None,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "metrics": metrics,
        })

        # Accumulate for micro
        n = metrics["n_samples"]
        total_samples += n
        sum_abs += metrics["mae"] * n
        sum_sq += (metrics["rmse"] ** 2) * n

        if metrics.get("_inside_50") is not None:
            inside_50_total += metrics["_inside_50"]
        if metrics.get("_inside_90") is not None:
            inside_90_total += metrics["_inside_90"]
        if metrics.get("_inside_95") is not None:
            inside_95_total += metrics["_inside_95"]
            cov_n_total += n

    # Build summary (same structure as tuning.py)
    summary = _build_lodo_summary(
        folds, total_samples, sum_abs, sum_sq,
        inside_50_total, inside_90_total, inside_95_total, cov_n_total
    )

    return {
        "folds": folds,
        "summary": summary,
        "model_name": model.name,
        "benchmark": dataset.benchmark_name,
        "noise": dataset.noise_type,
    }


def _build_lodo_summary(folds, total_samples, sum_abs, sum_sq,
                        inside_50, inside_90, inside_95, cov_n):
    """Build summary statistics from LODO folds (mirrors tuning.py structure)."""

    metric_names = [
        'mae', 'rmse', 'r2', 'max_error', 'nlpd',
        'coverage_50', 'coverage_90', 'coverage_95',
        'mean_interval_width_95', 'calibration_error_95', 'sharpness'
    ]

    # Macro: mean/std per metric
    macro = {}
    for name in metric_names:
        values = [f["metrics"].get(name) for f in folds if f["metrics"].get(name) is not None]
        if values:
            macro[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        else:
            macro[name] = {"mean": None, "std": None, "min": None, "max": None}

    # Legacy alias
    macro["coverage95"] = macro.get("coverage_95", {"mean": None, "std": None})

    # Micro: sample-weighted
    micro = {
        "mae": float(sum_abs / max(1, total_samples)),
        "rmse": float(np.sqrt(sum_sq / max(1, total_samples))),
        "coverage_50": float(inside_50 / max(1, cov_n)) if cov_n > 0 else None,
        "coverage_90": float(inside_90 / max(1, cov_n)) if cov_n > 0 else None,
        "coverage_95": float(inside_95 / max(1, cov_n)) if cov_n > 0 else None,
    }
    micro["coverage95"] = micro.get("coverage_95")

    return {
        "macro": macro,
        "micro": micro,
        "n_folds": len(folds),
        "total_samples": total_samples,
    }


def nested_lodo_tuning_benchmark(
    base_model: SurrogateRegressor,
    param_grid: Dict[str, List],
    dataset: SyntheticDataset,
    scoring: str = "mae",
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """
    Perform nested LODO tuning on a synthetic benchmark dataset.

    This applies the same nested cross-validation strategy used for real data
    to benchmark functions, enabling fair comparison of tuning results.

    Args:
        base_model: Base model to tune
        param_grid: Hyperparameter grid
        dataset: SyntheticDataset with groups (n_groups > 1)
        scoring: Metric for selecting best params ('mae', 'rmse', 'nlpd')
        n_jobs: Parallel jobs

    Returns:
        Same structure as nested_lodo_tuning from tuning.py

    Example:
        >>> dataset = generate_benchmark_dataset('branin', n_train=100, n_groups=5)
        >>> results = nested_lodo_tuning_benchmark(
        ...     GPSurrogateRegressor(),
        ...     {'alpha': [0.01, 0.1, 1.0]},
        ...     dataset, n_jobs=4
        ... )
    """
    if dataset.groups_train is None:
        raise ValueError(
            "Dataset needs groups for nested LODO. "
            "Use: generate_benchmark_dataset(..., n_groups=5)"
        )

    # Import tuning function and use it directly
    from .tuning import nested_lodo_tuning

    X = dataset.X_train
    y = dataset.y_train
    groups = dataset.groups_train

    results = nested_lodo_tuning(
        model=base_model,
        param_grid=param_grid,
        X=X,
        y=y,
        groups=groups,
        primary=scoring,
        inner_n_jobs=n_jobs,
        outer_n_jobs=1,  # Outer is sequential to avoid nested parallelism issues
    )

    # Add benchmark metadata
    results["benchmark"] = dataset.benchmark_name
    results["noise"] = dataset.noise_type
    results["n_groups"] = len(np.unique(groups))

    return results


def save_benchmark_results(
    results: BenchmarkSuiteResults,
    output_dir: Optional[Path] = None,
    session_name: str = "benchmark_eval",
) -> Path:
    """
    Save benchmark results to files.

    Creates:
        - {session_name}_results.json: Full results
        - {session_name}_summary.csv: Summary DataFrame

    Args:
        results: BenchmarkSuiteResults to save
        output_dir: Output directory (default: LOGS_DIR / "benchmarks")
        session_name: Name prefix for output files

    Returns:
        Path to output directory
    """
    if output_dir is None:
        output_dir = LOGS_DIR / "benchmarks"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / f"{session_name}_results.json"
    results.to_json(json_path)

    # Save summary CSV
    csv_path = output_dir / f"{session_name}_summary.csv"
    results.get_summary_df().to_csv(csv_path, index=False)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - {json_path.name}")
    print(f"  - {csv_path.name}")

    return output_dir


if __name__ == "__main__":
    # Quick test
    from ..models.gp import GPSurrogateRegressor
    from ..models.ridge import RidgeSurrogateRegressor
    from ..models.pls import PLSSurrogateRegressor
    from ..models.dummy import DummySurrogateRegressor

    models = {
        "Dummy": DummySurrogateRegressor(),
        "Ridge": RidgeSurrogateRegressor(),
        "PLS": PLSSurrogateRegressor(n_components=2),
        "GP": GPSurrogateRegressor(),
    }

    results = run_quick_benchmark(
        models=models,
        benchmarks=["forrester", "branin"],
        n_train=30,
        n_test=100,
        seed=42,
    )
