# @author: José Arbelaez
"""
Surrogate models module.

Provides various surrogate model implementations for regression with
uncertainty quantification, all inheriting from SurrogateRegressor base class.

Models:
    - GPSurrogateRegressor: Gaussian Process with kernel-based UQ
    - RidgeSurrogateRegressor: Ridge regression (no native UQ)
    - PLSSurrogateRegressor: Partial Least Squares (no native UQ)
    - DummySurrogateRegressor: Baseline dummy model
"""

from .base import SurrogateRegressor
from .gp import GPSurrogateRegressor
from .ridge import RidgeSurrogateRegressor
from .pls import PLSSurrogateRegressor
from .dummy import DummySurrogateRegressor

__all__ = [
    "SurrogateRegressor",
    "GPSurrogateRegressor",
    "RidgeSurrogateRegressor",
    "PLSSurrogateRegressor",
    "DummySurrogateRegressor",
]
