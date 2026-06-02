# @author: José Arbelaez
"""

Contains:
    - NOISE_CONFIGS: Standard noise configurations for testing
    - DEFAULT_SAMPLERS: Default sampling strategies
    - N_TRAIN_ABSOLUTE: Fixed training sizes (independent of dimension)
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

NOISE_CONFIGS_STANDARD = [ # Reduced from 5 to 3 noise configs to constraint testing runtime
    {"type": "none"},                           # Clean (no synthetic noise)
    {"type": "gaussian", "sigma": 0.5},         # Moderate noise
    {"type": "gaussian", "sigma": 1.0},         # High noise
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

# Fixed training sizes (same count for every benchmark dimension).
N_TRAIN_ABSOLUTE = [1]

# Multipliers for dynamic n_train: n_train = multiplier * dimension
N_TRAIN_MULTIPLIERS = [1, 4]  # Reduced from [1,2,4,8] to constraint testing runtime.


def get_n_train_for_dimension(
    dim: int,
    *,
    extra_absolute: Optional[List[int]] = None,
    include_absolute: bool = True,
    include_multipliers: bool = True,
    exclusive: Optional[List[int]] = None,
) -> List[int]:
    """
    Resolve training sizes for a benchmark dimension.

    Default (all flags on, no extras): {N_TRAIN_ABSOLUTE} ∪ {m * dim | m in N_TRAIN_MULTIPLIERS}.
    Example with N_TRAIN_ABSOLUTE=[1] and N_TRAIN_MULTIPLIERS=[1,4], dim=2 -> [1, 2, 8].

    Args:
        dim: Benchmark dimension
        extra_absolute: Additional literal sizes (e.g. from CLI --n-train)
        include_absolute: Include N_TRAIN_ABSOLUTE from config
        include_multipliers: Include m * dim for each N_TRAIN_MULTIPLIERS entry
        exclusive: If set, return only these sizes (ignores other sources)

    Returns:
        Sorted unique list of n_train values (>= 0)
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")

    if exclusive is not None:
        return sorted({int(v) for v in exclusive if int(v) >= 0})

    values: List[int] = []
    if include_absolute:
        values.extend(N_TRAIN_ABSOLUTE)
    if extra_absolute:
        values.extend(int(v) for v in extra_absolute)
    if include_multipliers:
        values.extend(int(m) * int(dim) for m in N_TRAIN_MULTIPLIERS)

    return sorted({int(v) for v in values if int(v) >= 0})


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
    "max_train_total": 50,
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

    max_train_total = int(cfg.get("max_train_total", 0))
    if max_train_total < 1:
        raise ValueError(f"max_train_total must be >= 1, got {max_train_total}")

    return cfg


# =============================================================================
# MODEL CONFIGURATIONS [USED IN TFG]
# =============================================================================

def get_default_models(dim: Optional[int] = None) -> Dict[str, Any]:
    """
    Get default model configurations for benchmarking.

    Includes GP kernel variants and baseline.
    If dim > 1, includes ARD variants with per-dimension length scales.

    Returns:
        Dict of model_name -> model_instance
    """
    import numpy as np
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF, DotProduct
    from src.models.gp import GPSurrogateRegressor
    from src.models.dummy import DummySurrogateRegressor

    models = {
        # Baseline
        "Dummy": DummySurrogateRegressor(strategy="mean"),

        # Isotropic GP variants
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
        "GP_Linear": GPSurrogateRegressor(
            kernel=DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=1e-5),
            n_restarts_optimizer=3
        ),
    }

    # ARD variants: one length_scale per dimension (only meaningful for dim > 1)
    if dim is not None and dim > 1:
        ls = np.ones(dim)
        models["GP_Matern52_ARD"] = GPSurrogateRegressor(
            kernel=Matern(nu=2.5, length_scale=ls) + WhiteKernel(noise_level=1e-5),
            n_restarts_optimizer=3
        )
        models["GP_RBF_ARD"] = GPSurrogateRegressor(
            kernel=RBF(length_scale=ls) + WhiteKernel(noise_level=1e-5),
            n_restarts_optimizer=3
        )

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
    "n_groups": None,  # Deprecated: synthetic groups not used in benchmarks
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
    "max_train_total": ACTIVE_LEARNING_DEFAULTS["max_train_total"],
}
