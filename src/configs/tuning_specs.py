"""
Tuning specifications for the Entomotive real case.

The real-case comparison is intentionally restricted to:
- Dummy baseline.
- Gaussian Processes split by explicit kernel family.
- ARD and non-ARD variants where the kernel has length-scale parameters.

No tree ensembles or additional regressors are active in this configuration.
"""

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, RBF, WhiteKernel

from src.models.dummy import DummySurrogateRegressor
from src.models.gp import GPSurrogateRegressor


# --- Target Mappings ---
TARGET_MAP = {
    "FCR": "FCR",
    "Quitina": "QUITINA (%)",
    "Proteina": "PROTEINA (%)",
}

# --- Feature Sets ---
FEATURE_COLS_REDUCED = [
    "inclusion_pct",
    "Proteína (%)_media",
    "Fibra (%)_media",
    "Grasa (%)_media",
    "TPC_dieta_media",
]

FEATURE_COLS_FULL = [
    "inclusion_pct",
    "Proteína (%)_media",
    "Grasa (%)_media",
    "Fibra (%)_media",
    "Cenizas (%)_media",
    "Carbohidratos (%)_media",
    "ratio_P_C",
    "ratio_P_F",
    "ratio_Fibra_Grasa",
    "TPC_dieta_media",
]

# --- Model Definitions ---
MODEL_TEMPLATES = {
    "Dummy": DummySurrogateRegressor(),
    "GP_Linear": GPSurrogateRegressor(),
    "GP_RBF_NoARD": GPSurrogateRegressor(),
    "GP_RBF_ARD": GPSurrogateRegressor(),
    "GP_Matern32_NoARD": GPSurrogateRegressor(),
    "GP_Matern32_ARD": GPSurrogateRegressor(),
    "GP_Matern52_NoARD": GPSurrogateRegressor(),
    "GP_Matern52_ARD": GPSurrogateRegressor(),
    "GP_Compuesto_NoARD": GPSurrogateRegressor(),
    "GP_Compuesto_ARD": GPSurrogateRegressor(),
}

BASE_MODEL_NAMES = [
    "Dummy",
    "GP_Linear",
    "GP_RBF_NoARD",
    "GP_Matern32_NoARD",
    "GP_Matern52_NoARD",
    "GP_Compuesto_NoARD",
]

# Backwards-compatible default: base comparison without selective ARD.
MODELS = {name: MODEL_TEMPLATES[name] for name in BASE_MODEL_NAMES}

# ARD is tested only for the previous best non-ARD GP in each target/feature mode.
SELECTED_ARD_BY_CASE = {
    ("REDUCED_FEATURES", "FCR"): "GP_RBF_ARD",
    ("REDUCED_FEATURES", "Quitina"): "GP_RBF_ARD",
    ("REDUCED_FEATURES", "Proteina"): "GP_Matern32_ARD",
    ("FULL_FEATURES", "FCR"): "GP_Matern52_ARD",
    ("FULL_FEATURES", "Quitina"): "GP_Matern52_ARD",
    ("FULL_FEATURES", "Proteina"): "GP_Matern32_ARD",
}


def get_models_for_case(session_name: str, target_label: str) -> dict:
    """Return base models plus the selected ARD variant for this case."""
    names = list(BASE_MODEL_NAMES)
    ard_name = SELECTED_ARD_BY_CASE.get((session_name, target_label))
    if ard_name and ard_name not in names:
        names.append(ard_name)
    return {name: MODEL_TEMPLATES[name] for name in names}


def _white_kernel():
    return WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-7, 0.8))


def _constant_kernel():
    return ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))


def build_named_gp_kernels(n_features: int) -> dict:
    """Return the GP kernel families used in the real-case comparison."""
    length_scale_bounds = (1e-2, 1e5)
    length_scale_ard = np.ones(n_features)

    return {
        "GP_Linear": (
            DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
            + _white_kernel()
        ),
        "GP_RBF_NoARD": (
            RBF(length_scale=1.0, length_scale_bounds=length_scale_bounds)
            + _white_kernel()
        ),
        "GP_RBF_ARD": (
            RBF(length_scale=length_scale_ard, length_scale_bounds=length_scale_bounds)
            + _white_kernel()
        ),
        "GP_Matern32_NoARD": (
            Matern(length_scale=1.0, nu=1.5, length_scale_bounds=length_scale_bounds)
            + _white_kernel()
        ),
        "GP_Matern32_ARD": (
            Matern(length_scale=length_scale_ard, nu=1.5, length_scale_bounds=length_scale_bounds)
            + _white_kernel()
        ),
        "GP_Matern52_NoARD": (
            Matern(length_scale=1.0, nu=2.5, length_scale_bounds=length_scale_bounds)
            + _white_kernel()
        ),
        "GP_Matern52_ARD": (
            Matern(length_scale=length_scale_ard, nu=2.5, length_scale_bounds=length_scale_bounds)
            + _white_kernel()
        ),
        "GP_Compuesto_NoARD": (
            _constant_kernel()
            * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
            + _constant_kernel()
            * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=length_scale_bounds)
            + _white_kernel()
        ),
        "GP_Compuesto_ARD": (
            _constant_kernel()
            * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
            + _constant_kernel()
            * Matern(length_scale=length_scale_ard, nu=2.5, length_scale_bounds=length_scale_bounds)
            + _white_kernel()
        ),
    }


def get_param_grids(n_features: int):
    """Parameter grids for the real-case models."""
    kernels = build_named_gp_kernels(n_features)
    gp_common = {
        "alpha": [1e-10, 1e-5, 1e-2, 1.0],
        "n_restarts_optimizer": [15],
        "normalize_y": [True],
    }

    return {
        "Dummy": {"strategy": ["mean", "median"]},
        "GP_Linear": {**gp_common, "kernel": [kernels["GP_Linear"]]},
        "GP_RBF_NoARD": {**gp_common, "kernel": [kernels["GP_RBF_NoARD"]]},
        "GP_RBF_ARD": {**gp_common, "kernel": [kernels["GP_RBF_ARD"]]},
        "GP_Matern32_NoARD": {**gp_common, "kernel": [kernels["GP_Matern32_NoARD"]]},
        "GP_Matern32_ARD": {**gp_common, "kernel": [kernels["GP_Matern32_ARD"]]},
        "GP_Matern52_NoARD": {**gp_common, "kernel": [kernels["GP_Matern52_NoARD"]]},
        "GP_Matern52_ARD": {**gp_common, "kernel": [kernels["GP_Matern52_ARD"]]},
        "GP_Compuesto_NoARD": {**gp_common, "kernel": [kernels["GP_Compuesto_NoARD"]]},
        "GP_Compuesto_ARD": {**gp_common, "kernel": [kernels["GP_Compuesto_ARD"]]},
    }
