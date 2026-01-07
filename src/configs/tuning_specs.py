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
    Generates a SUPER EXHAUSTIVE list of kernels.
    Includes Baselines, Isotropic, ARD, and Composite Kernels.
    IMPORTANT: We instantiate WhiteKernel fresh every time to avoid reference issues.
    """
    kernels = []
    
    # 1. BOUNDS STRATEGY:
    # We open the bounds significantly. 
    # Lower bound 1e-2 allows for sharp changes.
    # Upper bound 1e5 allows for effectively linear/flat behavior (ignoring dimensions).
    ls_bounds = (1e-2, 1e5) 
    
    # 2. NOISE STRATEGY:
    # We allow the model to assume the data is very clean (1e-7) or very noisy (0.8).
    # High upper bound helps avoid overfitting outliers.
    wn_bounds = (1e-7, 0.8)
    wn_init = 1e-4
    
    # =================================================================
    # A. BASELINE / SIMPLE KERNELS (Sanity Checks)
    # =================================================================
    kernels.extend([
        # Pure Linear (Bayesian Ridge equivalent)
        # Checks: "Is the relationship simply linear?"
        DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e3)) + 
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds),
    ])

    # =================================================================
    # B. ISOTROPIC KERNELS (Standard assumptions)
    # =================================================================
    # We test different 'smoothness' levels
    for nu in [0.5, 1.5, 2.5]:
        kernels.append(
            Matern(length_scale=1.0, nu=nu, length_scale_bounds=ls_bounds) + 
            WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
        )
        
    # RBF (Infinite smoothness)
    kernels.append(
        RBF(length_scale=1.0, length_scale_bounds=ls_bounds) + 
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )
    
    # Rational Quadratic (Multi-scale / Heavy tails)
    # Great for when some effects are short-range and others long-range.
    kernels.append(
        RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=ls_bounds, alpha_bounds=(1e-2, 1e2)) + 
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    # =================================================================
    # C. ANISOTROPIC / ARD KERNELS (Feature Selection)
    # =================================================================
    ls_vec = np.ones(n_features)
    
    # ARD Matern 1.5 (Standard physical process + Selection)
    kernels.append(
        Matern(length_scale=ls_vec, nu=1.5, length_scale_bounds=ls_bounds) + 
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )
    
    # ARD Matern 2.5 (Smoother Selection)
    kernels.append(
        Matern(length_scale=ls_vec, nu=2.5, length_scale_bounds=ls_bounds) + 
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )
    
    # ARD RBF (Maximum smoothness Selection)
    kernels.append(
        RBF(length_scale=ls_vec, length_scale_bounds=ls_bounds) + 
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    # =================================================================
    # D. COMPOSITE KERNELS (The "Hail Mary" plays)
    # =================================================================
    
    # 1. Linear Trend + RBF Correction
    # "Generally linear, but with some smooth bumps"
    kernels.append(
        DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2)) + 
        RBF(length_scale=1.0, length_scale_bounds=ls_bounds) + 
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    # 2. Linear Trend + Matern 1.5 Correction
    # "Generally linear, but with some rougher bumps"
    kernels.append(
        DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2)) + 
        Matern(length_scale=1.0, nu=1.5, length_scale_bounds=ls_bounds) + 
        WhiteKernel(noise_level=wn_init, noise_level_bounds=wn_bounds)
    )

    # 3. RationalQuadratic + Linear
    # "Complex multi-scale relations over a linear trend"
    kernels.append(
        DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2)) + 
        RationalQuadratic(length_scale=1.0, alpha=0.5) + 
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
            "alpha": [1e-10, 1e-5, 1e-2, 1], 
            
            # EXHAUSTIVE RESTARTS:
            # 15 restarts is a good compromise between speed and finding global optima
            "n_restarts_optimizer": [15], 
            
            "normalize_y": [True],
            
            "kernel": build_exhaustive_gp_kernels(n_features),
        },
    }