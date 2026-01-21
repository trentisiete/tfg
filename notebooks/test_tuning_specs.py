"""
Tuning specifications and hyperparameter grids for Hermetia illucens analysis.
SUPER EXHAUSTIVE GRID FOR GP (Final Robust Version).
Designed to capture any signal, whether linear, smooth, rugged, or additive.
"""

import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel, DotProduct
)
from src.models.dummy import DummySurrogateRegressor
from src.models.ridge import RidgeSurrogateRegressor
from src.models.pls import PLSSurrogateRegressor
from src.models.gp import GPSurrogateRegressor

# --- Target Mappings ---
TARGET_MAP = {
    "FCR": "FCR",
    "TPC": "TPC_larva_media",
    "Quitina": "QUITINA (%)",
    "Proteina": "PROTEINA (%)",
}

# --- Feature Sets ---

# 1. Reduced / Critical Features
FEATURE_COLS_REDUCED = [
    "inclusion_pct",
    "Proteína (%)_media",
    "Fibra (%)_media",
    "Grasa (%)_media",
    "TPC_dieta_media"
]

# 2. Full Features (Testing ARD capabilities)
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
MODELS = {
    "Dummy": DummySurrogateRegressor(),
    "Ridge": RidgeSurrogateRegressor(),
    "PLS": PLSSurrogateRegressor(),
    "GP": GPSurrogateRegressor(),
}

# --- Dynamic Grid Generation ---

def build_exhaustive_gp_kernels(n_features: int):
    """
    Generates a list of kernels.
    Includes Baselines, Isotropic, ARD, and Composite Kernels.
    IMPORTANT: We instantiate WhiteKernel fresh every time to avoid reference issues.
    """
    kernels = []

    # 1. BOUNDS STRATEGY:
    # We open the bounds significantly.
    # Lower bound 1e-2 allows for sharp changes.
    # Upper bound 1e5 allows for effectively linear/flat behavior (ignoring dimensions).
    ls_bounds = (1e-2)

    # 2. NOISE STRATEGY:
    # We allow the model to assume the data is very clean (1e-7) or very noisy (0.8).
    # High upper bound helps avoid overfitting outliers.
    wn_bounds = (1e-7)
    wn_init = 1e-4

    # =================================================================
    # B. ISOTROPIC KERNELS (Standard assumptions)
    # =================================================================
    # We test different 'smoothness' levels
    # RBF (Infinite smoothness)
    kernels.append(
        RBF(length_scale=1.0, length_scale_bounds=ls_bounds) +
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    return kernels


def get_param_grids(n_features: int):
    """
    Returns the parameter grids.
    GP grid is generated dynamically based on n_features.
    """
    return {
        "Dummy": {
            "strategy": ["mean", "median"]
        },
        "Ridge": {
            # Log-space search for Alpha
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "fit_intercept": [True],
        },
        "PLS": {
            # Components: from 1 up to min(n_features, 8)
            "n_components": list(range(1, min(n_features + 1, 8))),
            "scale": [True],
        },
        "GP": {
            # EXHAUSTIVE ALPHA:
            # 1e-10: Trust WhiteKernel completely.
            # 1e-5: Standard jitter (numerical stability).
            # 1e-2: High jitter (regularization/smoothing).
            "alpha": [1e-2],

            # EXHAUSTIVE RESTARTS:
            # 15 restarts is a good compromise between speed and finding global optima
            "n_restarts_optimizer": [15],

            "normalize_y": [True],

            "kernel": build_exhaustive_gp_kernels(n_features),
        },
    }