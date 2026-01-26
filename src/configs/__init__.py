"""
Configuration module for surrogate models project.

Contains:
    - tuning_specs.py: Tuning specifications for real data evaluation
    - benchmark_grids.py: Hyperparameter grids for benchmark evaluation
    - evaluation_defaults.py: Default configurations for benchmark evaluation
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

from .benchmark_tuning_specs import (
    NOISE_CONFIGS_STANDARD,
    NOISE_CONFIGS_EXTENDED,
    DEFAULT_SAMPLERS,
    N_TRAIN_MULTIPLIERS,
    EVALUATION_DEFAULTS,
    get_noise_configs,
    get_n_train_for_dimension,
    get_default_models,
    get_base_models,
    get_simple_models,
)

__all__ = [
    # benchmark_grids
    "BENCHMARK_GRIDS",
    "DEFAULT_GRIDS",
    "get_benchmark_grid",
    "get_default_grid",
    "get_grid_for_evaluation",
    "list_configured_benchmarks",
    "get_all_grids_for_benchmark",
    "merge_with_defaults",
    # evaluation_defaults
    "NOISE_CONFIGS_STANDARD",
    "NOISE_CONFIGS_EXTENDED",
    "DEFAULT_SAMPLERS",
    "N_TRAIN_MULTIPLIERS",
    "EVALUATION_DEFAULTS",
    "get_noise_configs",
    "get_n_train_for_dimension",
    "get_default_models",
    "get_base_models",
    "get_simple_models",
]
