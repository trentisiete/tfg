# @author: José Arbelaez
"""

Contains:
    - NOISE_CONFIGS: Standard noise configurations for testing
    - DEFAULT_SAMPLERS: Default sampling strategies
    - N_TRAIN_MULTIPLIERS: Multipliers for dynamic n_train calculation
    - ACTIVE_LEARNING_DEFAULTS: Defaults for EI-based active learning
    - get_default_models(): Default model configurations
    - get_base_models(): Base models for hyperparameter tuning
    - get_simple_models(): Minimal set for quick testing
    - get_active_learning_config(): Resolve active-learning runtime config
"""

from typing import Dict, List, Any, Optional


# =============================================================================
# NOISE CONFIGURATIONS
# =============================================================================

NOISE_CONFIGS_STANDARD = [
    {"type": "none"},                           # Normal/base case (no synthetic noise)
    {"type": "gaussian", "sigma": 0.05},        # Low Gaussian noise
    {"type": "gaussian", "sigma": 0.1},         # Moderate Gaussian noise
    {"type": "gaussian", "sigma": 0.3},         # High Gaussian noise
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

# FUTURE_TODO: This samplers can only be used in the initial stage of the benchmark.
#        Therefore, we have to implement Infill criteria samplers to confirm the
#        fact that our strategy relay on Active Learning.

DEFAULT_SAMPLERS = ["sobol", "random"]

# Multipliers for dynamic n_train calculation: n_train = multiplier * dimension

N_TRAIN_MULTIPLIERS = [1, 3, 6, 9]


def get_n_train_for_dimension(dim: int) -> List[int]:
    """
    Calculate training sizes for a given dimension.

    Args:
        dim: Benchmark dimension

    Returns:
        List of n_train values [1*d, 3*d, 6*d, 9*d]

    WARNING: Change number of N_TRAIN_MULTIPLIERS if you want different n_train_samples.
    """
    return [m * dim for m in N_TRAIN_MULTIPLIERS]


# =============================================================================
# ACTIVE LEARNING CONFIGURATIONS
# =============================================================================

ACTIVE_LEARNING_DEFAULTS: Dict[str, Any] = {
    # Optimization setup
    "objective": "minimize",
    "acquisition": "ei",
    "optimizer": "differential_evolution",
    # Initial and sequential budget
    "min_initial_train": 1,
    "n_infill_per_dim": 5,
    # EI behavior
    "ei_xi": 0.01,
    # Continuous optimizer budget rule: max(min_budget, active_cand_mult * dim)
    "optimizer_budget_min": 2000,
    "active_cand_mult": 500,
    # Periodic CV audit and guarded kernel switching
    "active_cv_check_every": 5,
    "cv_audit_metric": "mae",
    "cv_disagreement_policy": "switch_with_guardrails",
    "active_switch_enable": True,
    "active_switch_warmup_steps": 5,
    "active_switch_min_improvement": 0.01,
    "active_switch_cooldown_steps": 5,
    # If True, active mode trains all configured models and compares them.
    # If False, active mode trains a single GP model (more realistic default).
    "active_train_all_models": False,
}


def get_active_learning_config(
    dim: int,
    n_infill: Optional[int] = None,
    ei_xi: Optional[float] = None,
    active_cand_mult: Optional[int] = None,
    active_cv_check_every: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Resolve active-learning config for a benchmark dimension.

    Args:
        dim: Benchmark dimension (>0)
        n_infill: Optional explicit infill budget. If None, uses n_infill_per_dim * dim
        ei_xi: Optional EI exploration parameter override
        active_cand_mult: Optional optimizer budget multiplier override
        active_cv_check_every: Optional CV audit cadence override

    Returns:
        Dict with resolved values and derived optimizer budget.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")

    cfg = ACTIVE_LEARNING_DEFAULTS.copy()

    if n_infill is not None:
        cfg["n_infill"] = int(n_infill)
    else:
        cfg["n_infill"] = int(cfg["n_infill_per_dim"]) * int(dim)

    if cfg["n_infill"] < 1:
        raise ValueError(f"n_infill must be >= 1, got {cfg['n_infill']}")

    if ei_xi is not None:
        cfg["ei_xi"] = float(ei_xi)
    if cfg["ei_xi"] < 0:
        raise ValueError(f"ei_xi must be >= 0, got {cfg['ei_xi']}")

    if active_cand_mult is not None:
        cfg["active_cand_mult"] = int(active_cand_mult)
    if cfg["active_cand_mult"] < 1:
        raise ValueError(f"active_cand_mult must be >= 1, got {cfg['active_cand_mult']}")

    if active_cv_check_every is not None:
        cfg["active_cv_check_every"] = int(active_cv_check_every)
    if cfg["active_cv_check_every"] < 0:
        raise ValueError(
            f"active_cv_check_every must be >= 0, got {cfg['active_cv_check_every']}"
        )

    if int(cfg.get("active_switch_warmup_steps", 0)) < 0:
        raise ValueError(
            f"active_switch_warmup_steps must be >= 0, got {cfg.get('active_switch_warmup_steps')}"
        )
    if float(cfg.get("active_switch_min_improvement", 0.0)) < 0:
        raise ValueError(
            f"active_switch_min_improvement must be >= 0, got {cfg.get('active_switch_min_improvement')}"
        )
    if int(cfg.get("active_switch_cooldown_steps", 0)) < 0:
        raise ValueError(
            f"active_switch_cooldown_steps must be >= 0, got {cfg.get('active_switch_cooldown_steps')}"
        )

    cfg["optimizer_budget"] = max(
        int(cfg["optimizer_budget_min"]),
        int(cfg["active_cand_mult"]) * int(dim),
    )

    return cfg


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

def get_default_models() -> Dict[str, Any]:
    """
    Get default model configurations for benchmarking.

    Includes GP kernel variants and baseline.

    Returns:
        Dict of model_name -> model_instance
    """
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
    from src.models.gp import GPSurrogateRegressor
    from src.models.dummy import DummySurrogateRegressor


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

    return {
        "GP": GPSurrogateRegressor(),
        "Dummy": DummySurrogateRegressor(),
    }


def get_simple_models() -> Dict[str, Any]:
    """
    Get minimal set of models for quick testing.

    Returns:
        Dict with Dummy and GP models
    """
    from src.models.gp import GPSurrogateRegressor
    from src.models.dummy import DummySurrogateRegressor

    return {
        "Dummy": DummySurrogateRegressor(),
        "GP": GPSurrogateRegressor(),
    }


# =============================================================================
# EVALUATION DEFAULTS
# =============================================================================


EVALUATION_DEFAULTS = {
    "n_test": 200,
    "n_groups": 5,
    "cv_mode": "simple_active",
    "seed": 42,
    "scoring": "mae",
    "n_jobs": 1,
    "n_infill_per_dim": ACTIVE_LEARNING_DEFAULTS["n_infill_per_dim"],
    "ei_xi": ACTIVE_LEARNING_DEFAULTS["ei_xi"],
    "active_cand_mult": ACTIVE_LEARNING_DEFAULTS["active_cand_mult"],
    "active_cv_check_every": ACTIVE_LEARNING_DEFAULTS["active_cv_check_every"],
    "active_switch_enable": ACTIVE_LEARNING_DEFAULTS["active_switch_enable"],
    "active_switch_warmup_steps": ACTIVE_LEARNING_DEFAULTS["active_switch_warmup_steps"],
    "active_switch_min_improvement": ACTIVE_LEARNING_DEFAULTS["active_switch_min_improvement"],
    "active_switch_cooldown_steps": ACTIVE_LEARNING_DEFAULTS["active_switch_cooldown_steps"],
    "active_train_all_models": ACTIVE_LEARNING_DEFAULTS["active_train_all_models"],
}
