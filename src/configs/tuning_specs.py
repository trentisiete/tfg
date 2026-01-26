"""
Tuning specifications and hyperparameter grids for Hermetia illucens analysis.
SUPER EXHAUSTIVE GRID FOR GP (Final Robust Version).

Update:
- Keep Ridge baseline
- Add multiple linear-kernel configurations inside GP (DotProduct variants)
- Add exhaustive grids for RandomForest (Bagging) and GradientBoosting
"""

import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel, DotProduct, ConstantKernel
)

from src.models.dummy import DummySurrogateRegressor
from src.models.ridge import RidgeSurrogateRegressor
from src.models.pls import PLSSurrogateRegressor
from src.models.gp import GPSurrogateRegressor
from src.models.bagging import RandomForestSurrogateRegressor
from src.models.boosting import GradientBoostingSurrogateRegressor


# --- Target Mappings ---
TARGET_MAP = {
    "FCR": "FCR",
    "TPC": "TPC_larva_media",
    "Quitina": "QUITINA (%)",
    "Proteina": "PROTEINA (%)",
}

# --- Feature Sets ---

FEATURE_COLS_REDUCED = [
    "inclusion_pct",
    "Proteína (%)_media",
    "Fibra (%)_media",
    "Grasa (%)_media",
    "TPC_dieta_media"
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
MODELS = {
    "Dummy": DummySurrogateRegressor(),
    "GP": GPSurrogateRegressor(),
    "RandomForest": RandomForestSurrogateRegressor(),
    "GradientBoosting": GradientBoostingSurrogateRegressor(),
}


def build_exhaustive_gp_kernels(n_features: int):
    """
    Generates a SUPER EXHAUSTIVE list of kernels.
    Includes Baselines, Isotropic, ARD, Linear families, and Composite Kernels.
    IMPORTANT: WhiteKernel must be instantiated fresh each time.
    """
    kernels = []

    # Length-scale bounds: allow sharp to very flat behavior
    ls_bounds = (1e-2, 1e5)

    # Noise strategy: from very clean to quite noisy
    wn_bounds = (1e-7, 0.8)
    wn_init = 1e-4

    # =================================================================
    # A. PURE LINEAR FAMILY (GP as “Bayesian linear regression”)
    # =================================================================
    # DotProduct ~ linear kernel. sigma_0 controls bias/intercept scale.
    for sigma0 in [0.01, 0.1, 1.0, 10.0]:
        kernels.append(
            DotProduct(sigma_0=sigma0, sigma_0_bounds=(1e-3, 1e3)) +
            WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
        )

    # ConstantKernel * DotProduct gives a learnable global scale of the linear term.
    for sigma0 in [0.1, 1.0]:
        kernels.append(
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
            DotProduct(sigma_0=sigma0, sigma_0_bounds=(1e-3, 1e3)) +
            WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
        )

    # Optional sanity: noise-only (detect “no signal”)
    kernels.append(
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    # =================================================================
    # B. ISOTROPIC KERNELS (smooth/rough nonlinear)
    # =================================================================
    for nu in [0.5, 1.5, 2.5]:
        kernels.append(
            Matern(length_scale=1.0, nu=nu, length_scale_bounds=ls_bounds) +
            WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
        )

    kernels.append(
        RBF(length_scale=1.0, length_scale_bounds=ls_bounds) +
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    kernels.append(
        RationalQuadratic(
            length_scale=1.0, alpha=1.0,
            length_scale_bounds=ls_bounds,
            alpha_bounds=(1e-2, 1e2)
        ) +
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    # =================================================================
    # C. ARD KERNELS (feature selection)
    # =================================================================
    ls_vec = np.ones(n_features)

    kernels.append(
        Matern(length_scale=ls_vec, nu=1.5, length_scale_bounds=ls_bounds) +
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    kernels.append(
        Matern(length_scale=ls_vec, nu=2.5, length_scale_bounds=ls_bounds) +
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    kernels.append(
        RBF(length_scale=ls_vec, length_scale_bounds=ls_bounds) +
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    # =================================================================
    # D. COMPOSITE KERNELS (linear trend + nonlinear correction)
    # =================================================================
    kernels.append(
        DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e2)) +
        RBF(length_scale=1.0, length_scale_bounds=ls_bounds) +
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    kernels.append(
        DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e2)) +
        Matern(length_scale=1.0, nu=1.5, length_scale_bounds=ls_bounds) +
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    kernels.append(
        DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e2)) +
        RationalQuadratic(length_scale=1.0, alpha=0.5) +
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    return kernels


def get_param_grids(n_features: int):
    """
    Parameter grids for all models. Exhaustive grids for GP, RandomForest, and GradientBoosting.
    """
    return {
        "Dummy": {
            "strategy": ["mean", "median"]
        },
        "GP": {
            "alpha": [1e-10, 1e-5, 1e-2, 1.0],
            "n_restarts_optimizer": [15],
            "normalize_y": [True],
            "kernel": build_exhaustive_gp_kernels(n_features),
        },
        # =====================================================================
        # BAGGING GRID (RandomForest) - Regularizado para pocos datos
        # =====================================================================
        "RandomForest": {
            "n_estimators": [50, 100, 200],          # Menos árboles
            "max_depth": [3, 5, 7],              # Profundidad limitada (sin None)
            "min_samples_leaf": [2, 4, 8],           # Hojas más grandes
            "max_features": ["sqrt", 0.5],     # Menos opciones
            "bootstrap": [True],                     # Solo bootstrap (reduce varianza)
            "random_state": [42],
        },
        # =====================================================================
        # BOOSTING GRID (GradientBoosting) - Regularizado para pocos datos
        # =====================================================================
        "GradientBoosting": {
            "n_estimators": [50, 100, 150],          # Menos iteraciones
            "learning_rate": [0.01, 0.05, 0.1],      # Learning rates conservadores
            "max_depth": [2, 3, 4],               # Árboles poco profundos
            "subsample": [0.7, 0.8, 0.9],            # Stochastic GB para regularizar
            "min_samples_leaf": [2, 4],           # Hojas más grandes
            "random_state": [42],
        },
    }
