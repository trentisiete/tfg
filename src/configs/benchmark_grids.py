# @author: José Arbeláez
"""
Hyperparameter grids for benchmark evaluation.

This file contains per-benchmark hyperparameter configurations for model tuning.
Allows fine-grained control over model parameters depending on benchmark characteristics.

Usage:
    from src.configs.benchmark_grids import (
        get_benchmark_grid,
        get_default_grid,
        BENCHMARK_GRIDS,
    )

    # Get grid for specific benchmark
    grid = get_benchmark_grid("forrester", "GP")

    # Get default grid when no specific config exists
    default_grid = get_default_grid("GP")
"""

import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF, Matern, WhiteKernel, DotProduct, ConstantKernel, RationalQuadratic
)
from typing import Dict, List, Any, Optional


# DEFAULT HYPERPARAMETER GRIDS (used when no benchmark-specific grid exists)
# =============================================================================

DEFAULT_GRIDS = {
    "GP": {
        "kernel": [
            Matern(nu=1.5) + WhiteKernel(noise_level=1e-5),
            Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
            RBF() + WhiteKernel(noise_level=1e-5),
        ],
        "n_restarts_optimizer": [3, 5],
        "alpha": [1e-10, 1e-8],
    },

    # These models are not evaluated in the TFG. Nonetheless, they are implemented and
    # ready to use:
    #
    # RandomForest : {}
    # GradientBoosting : {}
    # PLS : {}
    # Ridge : {}

    "Dummy": {
        "strategy": ["mean", "median"],
    },
}


# BENCHMARK-SPECIFIC HYPERPARAMETER GRIDS

# LOW DIMENSIONAL BENCHMARKS (1D - 3D)
# -----------------------------------------------------------------------------

FORRESTER_GRIDS = {
    "GP": {
        "kernel": [
            # 1D functions benefit from smooth kernels
            RBF(length_scale=0.1) + WhiteKernel(noise_level=1e-5),
            RBF(length_scale=0.5) + WhiteKernel(noise_level=1e-5),
            Matern(nu=2.5, length_scale=0.2) + WhiteKernel(noise_level=1e-5),
            Matern(nu=2.5, length_scale=0.5) + WhiteKernel(noise_level=1e-5),
        ],
        "n_restarts_optimizer": [5, 10],
        "alpha": [1e-10],
    },
    # "RandomForest": {
    #     "n_estimators": [50, 100],
    #     "max_depth": [5, 10, None],
    #     "min_samples_leaf": [1, 2],
    # },
    # "GradientBoosting": {
    #     "n_estimators": [50, 100],
    #     "learning_rate": [0.05, 0.1],
    #     "max_depth": [2, 3],
    # },
}

BRANIN_GRIDS = {
    "GP": {
        "kernel": [
            Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
            Matern(nu=1.5) + WhiteKernel(noise_level=1e-5),
            RBF() + WhiteKernel(noise_level=1e-5),
            # ARD for 2D (different length scales per dimension)
            Matern(nu=2.5, length_scale=[1.0, 1.0]) + WhiteKernel(noise_level=1e-5),
        ],
        "n_restarts_optimizer": [5, 10],
        "alpha": [1e-10, 1e-8],
    },
}

SIXHUMPCAMEL_GRIDS = {
    "GP": {
        "kernel": [
            Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
            Matern(nu=1.5) + WhiteKernel(noise_level=1e-4),  # More noise tolerance
            RBF(length_scale=0.5) + WhiteKernel(noise_level=1e-5),
        ],
        "n_restarts_optimizer": [5, 10],
        "alpha": [1e-10, 1e-8],
    },
}

GOLDSTEINPRICE_GRIDS = {
    "GP": {
        "kernel": [
            # High dynamic range function - needs flexible kernels
            Matern(nu=2.5) + WhiteKernel(noise_level=1e-4),
            RBF() + WhiteKernel(noise_level=1e-4),
            ConstantKernel(1e4) * RBF() + WhiteKernel(noise_level=1e-3),
        ],
        "n_restarts_optimizer": [10],
        "alpha": [1e-8, 1e-6],
    },
}

HARTMANN3_GRIDS = {
    "GP": {
        "kernel": [
            Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
            Matern(nu=1.5) + WhiteKernel(noise_level=1e-5),
            RBF() + WhiteKernel(noise_level=1e-5),
            # ARD for 3D
            Matern(nu=2.5, length_scale=[1.0, 1.0, 1.0]) + WhiteKernel(noise_level=1e-5),
        ],
        "n_restarts_optimizer": [5, 10],
        "alpha": [1e-10, 1e-8],
    },
}

ISHIGAMI_GRIDS = {
    "GP": {
        "kernel": [
            # Ishigami has strong nonlinearities
            Matern(nu=1.5) + WhiteKernel(noise_level=1e-5),
            Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
            RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5),
            RBF(length_scale=2.0) + WhiteKernel(noise_level=1e-5),
        ],
        "n_restarts_optimizer": [5, 10],
        "alpha": [1e-10, 1e-8],
    },
}

# MEDIUM/HIGH DIMENSIONAL BENCHMARKS (6D+)
# -----------------------------------------------------------------------------

HARTMANN6_GRIDS = {
    "GP": {
        "kernel": [
            Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
            Matern(nu=1.5) + WhiteKernel(noise_level=1e-5),
            RBF() + WhiteKernel(noise_level=1e-5),
            # ARD with 6 length scales
            Matern(nu=2.5, length_scale=np.ones(6)) + WhiteKernel(noise_level=1e-5),
        ],
        "n_restarts_optimizer": [5, 10],
        "alpha": [1e-10, 1e-8],
    },
    # Example of other models grid
    # "RandomForest": {
    #     "n_estimators": [100, 200, 300],
    #     "max_depth": [10, 20, None],
    #     "min_samples_leaf": [1, 2, 5],
    #     "max_features": ["sqrt", "log2"],
    # },
    # "GradientBoosting": {
    #     "n_estimators": [100, 200, 300],
    #     "learning_rate": [0.01, 0.05, 0.1],
    #     "max_depth": [3, 5, 7],
    #     "subsample": [0.8, 1.0],
    # },
}

BOREHOLE_GRIDS = {
    "GP": {
        "kernel": [
            # 8D function - isotropic and ARD
            Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
            Matern(nu=1.5) + WhiteKernel(noise_level=1e-5),
            RBF() + WhiteKernel(noise_level=1e-5),
            Matern(nu=2.5, length_scale=np.ones(8)) + WhiteKernel(noise_level=1e-5),
        ],
        "n_restarts_optimizer": [5, 10],
        "alpha": [1e-10, 1e-8],
    },
}

WINGWEIGHT_GRIDS = {
    "GP": {
        "kernel": [
            # 10D function
            Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
            Matern(nu=1.5) + WhiteKernel(noise_level=1e-5),
            RBF() + WhiteKernel(noise_level=1e-5),
        ],
        "n_restarts_optimizer": [5, 10],
        "alpha": [1e-10, 1e-8],
    },
}


# MASTER REGISTRY
# =============================================================================

BENCHMARK_GRIDS: Dict[str, Dict[str, Dict[str, Any]]] = {
    # Low-dimensional
    "forrester": FORRESTER_GRIDS,
    "branin": BRANIN_GRIDS,
    "sixhumpcamel": SIXHUMPCAMEL_GRIDS,
    "goldsteinprice": GOLDSTEINPRICE_GRIDS,
    "hartmann3": HARTMANN3_GRIDS,
    "ishigami": ISHIGAMI_GRIDS,

    # Medium/High-dimensional
    "hartmann6": HARTMANN6_GRIDS,
    "borehole": BOREHOLE_GRIDS,
    "wingweight": WINGWEIGHT_GRIDS,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_benchmark_grid(benchmark_name: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get hyperparameter grid for a specific benchmark and model.

    Args:
        benchmark_name: Name of the benchmark (e.g., "forrester", "branin")
        model_name: Name of the model (e.g., "GP", "Ridge")

    Returns:
        Dictionary with hyperparameter grid, or None if not found

    Example:
        >>> grid = get_benchmark_grid("forrester", "GP")
        >>> print(grid.keys())
        dict_keys(['kernel', 'n_restarts_optimizer', 'alpha'])
    """
    benchmark_key = benchmark_name.lower()

    if benchmark_key in BENCHMARK_GRIDS:
        bench_grids = BENCHMARK_GRIDS[benchmark_key]
        # Try exact match first
        if model_name in bench_grids:
            return bench_grids[model_name]
        # Try case-insensitive match
        for key, grid in bench_grids.items():
            if key.lower() == model_name.lower():
                return grid

    return None


def get_default_grid(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get default hyperparameter grid for a model.

    Args:
        model_name: Name of the model (e.g., "GP", "Ridge")

    Returns:
        Dictionary with default hyperparameter grid, or None if not found
    """
    if model_name in DEFAULT_GRIDS:
        return DEFAULT_GRIDS[model_name]

    # Case-insensitive search
    for key, grid in DEFAULT_GRIDS.items():
        if key.lower() == model_name.lower():
            return grid

    return None


def get_grid_for_evaluation(
    benchmark_name: str,
    model_name: str,
    use_defaults: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Get hyperparameter grid for benchmark evaluation.

    Tries benchmark-specific grid first, falls back to default if allowed.

    Args:
        benchmark_name: Name of the benchmark
        model_name: Name of the model
        use_defaults: If True, fall back to default grid when benchmark-specific
                     grid is not found

    Returns:
        Hyperparameter grid dictionary, or None if not found
    """
    # Try benchmark-specific grid
    grid = get_benchmark_grid(benchmark_name, model_name)

    if grid is not None:
        return grid

    # Fall back to defaults
    if use_defaults:
        return get_default_grid(model_name)

    return None


def list_configured_benchmarks() -> List[str]:
    """Return list of benchmarks that have specific hyperparameter configurations."""
    return list(BENCHMARK_GRIDS.keys())


def get_all_grids_for_benchmark(benchmark_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all model grids configured for a specific benchmark.

    Args:
        benchmark_name: Name of the benchmark

    Returns:
        Dictionary mapping model names to their hyperparameter grids
    """
    benchmark_key = benchmark_name.lower()

    if benchmark_key in BENCHMARK_GRIDS:
        return BENCHMARK_GRIDS[benchmark_key]

    return {}


def merge_with_defaults(benchmark_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Get grids for a benchmark, filling in defaults for missing models.

    Args:
        benchmark_name: Name of the benchmark

    Returns:
        Complete dictionary with grids for all models (benchmark-specific + defaults)
    """
    result = dict(DEFAULT_GRIDS)  # Start with defaults

    benchmark_grids = get_all_grids_for_benchmark(benchmark_name)

    # Override with benchmark-specific
    for model_name, grid in benchmark_grids.items():
        result[model_name] = grid

    return result
