# @author: José Arbelaez
"""
evaluation_defaults.py

Default configurations for benchmark evaluation.

Contains:
    - NOISE_CONFIGS: Standard noise configurations for testing
    - DEFAULT_SAMPLERS: Default sampling strategies
    - N_TRAIN_MULTIPLIERS: Multipliers for dynamic n_train calculation
    - get_default_models(): Default model configurations
    - get_base_models(): Base models for hyperparameter tuning
    - get_simple_models(): Minimal set for quick testing
"""

from typing import Dict, List, Any


# =============================================================================
# NOISE CONFIGURATIONS
# =============================================================================

NOISE_CONFIGS_STANDARD = [
    {"type": "none"},                           # Pure interpolation test
    {"type": "gaussian", "sigma": 0.05},        # Low noise
    {"type": "gaussian", "sigma": 0.1},         # Moderate noise
    {"type": "gaussian", "sigma": 0.3},         # High noise
]

NOISE_CONFIGS_EXTENDED = NOISE_CONFIGS_STANDARD + [
    {"type": "heteroscedastic", "sigma_base": 0.02, "sigma_scale": 0.15},
    {"type": "proportional", "sigma_rel": 0.05, "sigma_base": 0.01},
]


def get_noise_configs(include_heteroscedastic: bool = False) -> List[Dict[str, Any]]:
    """
    Get noise configurations for benchmark evaluation.
    
    Args:
        include_heteroscedastic: Include challenging heteroscedastic noise
        
    Returns:
        List of noise configuration dicts
    """
    if include_heteroscedastic:
        return NOISE_CONFIGS_EXTENDED.copy()
    return NOISE_CONFIGS_STANDARD.copy()


# =============================================================================
# SAMPLING CONFIGURATIONS
# =============================================================================

DEFAULT_SAMPLERS = ["sobol", "random"]

# Multipliers for dynamic n_train calculation: n_train = multiplier * dimension
N_TRAIN_MULTIPLIERS = [3, 6, 9, 12]


def get_n_train_for_dimension(dim: int) -> List[int]:
    """
    Calculate training sizes for a given dimension.
    
    Args:
        dim: Benchmark dimension
        
    Returns:
        List of n_train values [3*d, 6*d, 9*d, 12*d]
    """
    return [m * dim for m in N_TRAIN_MULTIPLIERS]


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

def get_default_models() -> Dict[str, Any]:
    """
    Get default model configurations for benchmarking.
    
    Includes GP kernel variants, ensemble methods (Bagging & Boosting), and baseline.
    
    Returns:
        Dict of model_name -> model_instance
    """
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
    from src.models.gp import GPSurrogateRegressor
    from src.models.dummy import DummySurrogateRegressor
    from src.models.bagging import RandomForestSurrogateRegressor
    from src.models.boosting import GradientBoostingSurrogateRegressor
    
    models = {
        # Baseline
        "Dummy": DummySurrogateRegressor(strategy="mean"),
        
        # GP variants
        "GP_Matern32": GPSurrogateRegressor(
            kernel=Matern(nu=1.5) + WhiteKernel(noise_level=1e-5),
            n_restarts_optimizer=3
        ),
        "GP_Matern52": GPSurrogateRegressor(
            kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
            n_restarts_optimizer=3
        ),
        "GP_RBF": GPSurrogateRegressor(
            kernel=RBF() + WhiteKernel(noise_level=1e-5),
            n_restarts_optimizer=3
        ),
        
        # Ensemble methods (Bagging & Boosting)
        "RandomForest": RandomForestSurrogateRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingSurrogateRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
    }
    
    return models


def get_base_models() -> Dict[str, Any]:
    """
    Get base model instances for hyperparameter tuning.
    
    These are untuned models that will be configured via grid search.
    
    Returns:
        Dict of model_name -> base_model_instance
    """
    from src.models.gp import GPSurrogateRegressor
    from src.models.dummy import DummySurrogateRegressor
    from src.models.bagging import RandomForestSurrogateRegressor
    from src.models.boosting import GradientBoostingSurrogateRegressor
    
    return {
        "GP": GPSurrogateRegressor(),
        "Dummy": DummySurrogateRegressor(),
        "RandomForest": RandomForestSurrogateRegressor(),
        "GradientBoosting": GradientBoostingSurrogateRegressor(),
    }


def get_simple_models() -> Dict[str, Any]:
    """
    Get minimal set of models for quick testing.
    
    Returns:
        Dict with Dummy, GP, and RandomForest models
    """
    from src.models.gp import GPSurrogateRegressor
    from src.models.dummy import DummySurrogateRegressor
    from src.models.bagging import RandomForestSurrogateRegressor
    
    return {
        "Dummy": DummySurrogateRegressor(),
        "GP": GPSurrogateRegressor(),
        "RandomForest": RandomForestSurrogateRegressor(),
    }


# =============================================================================
# EVALUATION DEFAULTS
# =============================================================================

EVALUATION_DEFAULTS = {
    "n_test": 300,
    "n_groups": 5,
    "cv_mode": "simple",
    "seed": 42,
    "scoring": "mae",
    "n_jobs": 1,
}
