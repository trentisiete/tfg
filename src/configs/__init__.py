"""
Configuration module for surrogate models project.

Contains:
    - tuning_specs.py: Tuning specifications for real data evaluation
    - benchmark_grids.py: Hyperparameter grids for benchmark evaluation
"""

from .benchmark_grids import (
    BENCHMARK_GRIDS,
    DEFAULT_GRIDS,
    get_benchmark_grid,
    get_default_grid,
    get_grid_for_evaluation,
    list_configured_benchmarks,
    get_all_grids_for_benchmark,
    merge_with_defaults,
)

__all__ = [
    "BENCHMARK_GRIDS",
    "DEFAULT_GRIDS",
    "get_benchmark_grid",
    "get_default_grid",
    "get_grid_for_evaluation",
    "list_configured_benchmarks",
    "get_all_grids_for_benchmark",
    "merge_with_defaults",
]
