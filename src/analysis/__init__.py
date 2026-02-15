# @author: José Arbelaez
"""
Analysis module for surrogate model evaluation.

This module provides tools for:
    - Cross-validation (LODO) evaluation
    - Nested hyperparameter tuning
    - Comprehensive surrogate metrics
    - Benchmark evaluation
"""

from .metrics import (
    make_splits,
    evaluate_model,
)

from .tuning import (
    nested_lodo_tuning,
)

from .surrogate_metrics import (
    SurrogateMetrics,
    compute_surrogate_metrics,
    metrics_to_dict,
    aggregate_metrics,
    compute_calibration_curve,
)

from .benchmark_runner import (
    BenchmarkResult,
    BenchmarkSuiteResults,
    evaluate_model_on_dataset,
    evaluate_model_with_lodo,
    evaluate_models_on_suite,
    run_quick_benchmark,
    nested_lodo_tuning_benchmark,
)

from .active_learning import (
    run_active_evaluation,
)

__all__ = [
    # Metrics module
    "make_splits",
    "evaluate_model",
    
    # Tuning module
    "nested_lodo_tuning",
    
    # Surrogate metrics
    "SurrogateMetrics",
    "compute_surrogate_metrics",
    "metrics_to_dict",
    "aggregate_metrics",
    "compute_calibration_curve",
    
    # Benchmark runner
    "BenchmarkResult",
    "BenchmarkSuiteResults",
    "evaluate_model_on_dataset",
    "evaluate_model_with_lodo",
    "evaluate_models_on_suite",
    "run_quick_benchmark",
    "nested_lodo_tuning_benchmark",
    "run_active_evaluation",
]
