import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel
from src.models.dummy import DummySurrogateRegressor
from src.models.ridge import RidgeSurrogateRegressor
from src.models.pls import PLSSurrogateRegressor
from src.models.gp import GPSurrogateRegressor

def build_gp_kernels(noise_levels):
    """Genera una lista de kernels complejos con componentes de ruido."""
    kernels = []
    for nl in noise_levels:
        wn = WhiteKernel(noise_level=nl, noise_level_bounds=(1e-8, 1e3))
        kernels.extend([
            Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e3)) + wn,
            Matern(length_scale=1.0, nu=0.5, length_scale_bounds=(1e-2, 1e3)) + wn,
            RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + wn,
            RationalQuadratic(length_scale=1.0, alpha=0.5, length_scale_bounds=(1e-2, 1e3), alpha_bounds=(1e-6, 1e6)) + wn,
        ])
    return kernels

# Mapeo de nombre legible -> columna en el CSV
TARGET_MAP = {
    "FCR": "FCR",
    "TPC": "TPC_larva_media",
    "Quitina": "QUITINA (%)",
    "Proteina": "PROTEINA (%)",
}

# Columnas que se usarán como variables independientes (X)
FEATURE_COLS = [
    "inclusion_pct",
    "Tratamiento",
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

# Definición de las instancias base de los modelos
MODELS = {
    "Dummy": DummySurrogateRegressor(),
    "Ridge": RidgeSurrogateRegressor(),
    "PLS": PLSSurrogateRegressor(),
    "GP": GPSurrogateRegressor(),
}

# Espacio de búsqueda de hiperparámetros
PARAM_GRIDS = {
    "Dummy": {"strategy": ["mean", "median"]},
    "Ridge": {
        "alpha": list(np.logspace(-4, 4, 17)),
        "fit_intercept": [True, False],
    },
    "PLS": {
        "n_components": list(range(1, 11)),
        "scale": [True, False],
    },
    "GP": {
        "alpha": [1e-10, 1e-8, 1e-6, 1e-4],
        "n_restarts_optimizer": [0, 3, 8],
        "normalize_y": [True],
        "kernel": build_gp_kernels([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
    },
}