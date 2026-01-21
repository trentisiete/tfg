# @author: José Arbelaez
"""
Base class for surrogate models.

All surrogate models inherit from SurrogateRegressor and must implement:
    - fit(X, y): Train the model
    - predict(X): Return point predictions
    - predict_dist(X): Return (mean, std) for uncertainty quantification
"""

from __future__ import annotations
from sklearn.base import BaseEstimator, RegressorMixin
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class SurrogateRegressor(ABC, BaseEstimator, RegressorMixin):
    """
    Abstract base class for surrogate models.
    
    Provides a consistent interface for:
        - Training (fit)
        - Point predictions (predict)
        - Probabilistic predictions (predict_dist)
        - Metrics computation (compute_metrics, compute_extended_metrics)
        - Candidate ranking for optimization (rank_candidates)
    
    All models should inherit from this class and implement the abstract methods.
    
    Example:
        >>> class MyModel(SurrogateRegressor):
        ...     name = "MyModel"
        ...     def fit(self, X, y): ...
        ...     def predict(self, X): ...
    """
    name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SurrogateRegressor":
        """Fit the model with X and y.

        Args:
            X (np.ndarray): Features for training.
            y (np.ndarray): Target values for training.
        """
        pass
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values of X matrix

        Args:
            X (np.ndarray): Features for prediction.
        Returns:
            np.ndarray: Predicted target values.
        """
        pass

    def predict_dist(self, X:np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict target values of X matrix along with uncertainty estimates.
        
        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            mean (np.ndarray): Predicted target values.
            std (np.ndarray | None): Uncertainty estimates if available, else None.
        
        Note:
            Models that don't provide uncertainty (e.g., Ridge, PLS) return std=None.
            GP models return actual predictive standard deviation.
        """
        mean = self.predict(X)
        std = None
        return mean, std

    def compute_metrics(self, y_test: np.ndarray, y_pred: np.ndarray, 
                        std_pred: Optional[np.ndarray] = None, 
                        z95: float = 1.96,
                        extended: bool = True) -> Dict[str, Any]:
        """
        Compute evaluation metrics for surrogate model predictions.
        
        This method computes comprehensive metrics including both basic 
        (MAE, RMSE) and extended (NLPD, R², calibration) metrics.
        
        The output format is compatible with both:
            - Real data evaluation (LODO cross-validation)
            - Benchmark synthetic evaluation
        
        Args:
            y_test: True target values
            y_pred: Predicted values (mean)
            std_pred: Predicted standard deviation (optional)
            z95: Z-score for 95% confidence interval (default: 1.96)
            extended: If True, compute all extended metrics (NLPD, R², etc.)
                     If False, only basic metrics for faster inner CV loops

        Returns:
            Dict with all metrics. Always includes:
                - n_samples, mae, rmse, coverage_95, _inside_95
            If extended=True, also includes:
                - r2, max_error, nlpd, coverage_50, coverage_90,
                - mean_interval_width_95, calibration_error_95, sharpness
        """
        from ..analysis.surrogate_metrics import compute_surrogate_metrics, metrics_to_dict
        
        # checkers
        y_test = np.asarray(y_test).ravel()
        y_pred = np.asarray(y_pred).ravel()
        
        if extended:
            # Use full surrogate metrics computation
            metrics_obj = compute_surrogate_metrics(y_test, y_pred, std_pred)
            out = metrics_to_dict(metrics_obj, include_private=True)
            
            # Backwards compatibility: ensure coverage95 key exists (alias)
            out["coverage95"] = out.get("coverage_95")
            out["_inside95"] = out.get("_inside_95")
            
        else:
            # Fast path for inner CV: only basic metrics
            n = int(y_test.shape[0])
            mae_value = float(mean_absolute_error(y_test, y_pred))
            rmse_value = float(root_mean_squared_error(y_test, y_pred))
            
            out = {
                "n_samples": n,
                "mae": mae_value,
                "rmse": rmse_value,
                "coverage95": None,
                "coverage_95": None,
                "_inside95": None,
                "_inside_95": None,
            }
            
            # coverage95 (quick computation for inner CV)
            if std_pred is not None:
                std_pred = np.asarray(std_pred).ravel()
                aux = z95 * std_pred
                lower_bound = y_pred - aux
                upper_bound = y_pred + aux

                inside = (y_test >= lower_bound) & (y_test <= upper_bound)
                inside_count = int(np.sum(inside))

                out["coverage95"] = float(np.mean(inside))
                out["coverage_95"] = out["coverage95"]
                out["_inside95"] = inside_count
                out["_inside_95"] = inside_count

        return out

    def compute_extended_metrics(self, y_test: np.ndarray, y_pred: np.ndarray,
                                  std_pred: Optional[np.ndarray] = None):
        """
        Compute extended metrics including NLPD, R², calibration error, etc.
        
        Args:
            y_test: True target values
            y_pred: Predicted mean values
            std_pred: Predicted standard deviations
            
        Returns:
            SurrogateMetrics dataclass with all computed metrics
            
        Example:
            >>> model.fit(X_train, y_train)
            >>> mean, std = model.predict_dist(X_test)
            >>> metrics = model.compute_extended_metrics(y_test, mean, std)
            >>> print(f"NLPD: {metrics.nlpd:.4f}")
        """
        from ..analysis.surrogate_metrics import compute_surrogate_metrics
        return compute_surrogate_metrics(y_test, y_pred, std_pred)


    def rank_candidates(self, Xcand: np.ndarray, k: int = 5, mode: str = "mean", beta: float = 1.0):
        """
        Rank candidate inputs based on predicted target values.

        Args:
            X (np.ndarray): Candidate features to rank.

        Returns:
            np.ndarray: Indices that would sort the candidates by predicted target values.
        """
        mean, std = self.predict_dist(Xcand)

        # To verify mean and std shapes are 1D:

        mean = np.asarray(mean).ravel()
        std = None if std is None else np.asarray(std).ravel()

        if mode == "mean":
            score = mean
        elif mode == "ucb":
            if std is None:
                raise ValueError("Uncertainty estimates are required for UCB ranking.")
            score = mean + beta*std
        elif mode == "lcb":
            if std is None:
                raise ValueError("Uncertainty estimates are required for LCB ranking.")
            score = mean - beta*std
        else:
            raise ValueError(f"Unknown ranking mode: {mode}")

        # TODO: If FCR is the objective, then we want to minimize it, so we take the lowest scores
        idx = np.argsort(score)[::-1][:k] # Descending order
        return idx, score[idx], mean[idx], std[idx] if std is not None else None