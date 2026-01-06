# @author: José Arbelaez
from __future__ import annotations
from sklearn.base import BaseEstimator, RegressorMixin
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
class SurrogateRegressor(ABC, BaseEstimator, RegressorMixin):
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

    # Do we really need this method?
    def predict_dist(self, X:np.ndarray):
        """
        Predict target values of X matrix along with uncertainty estimates.
        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            mean: Predicted target values.
            std: (optional) Uncertainty estimates if available.
        """
        mean = self.predict(X)

        std = None
        return mean, std


    def compute_metrics(self, y_test:np.ndarray, X_pred:np.ndarray, std_pred: np.ndarray | None = None, z95: float = 1.96) -> dict:

        # checkers
        y_test = np.asarray(y_test).ravel()
        n = int(y_test.shape[0])


        mae_value = float(mean_absolute_error(y_test, X_pred))
        rmse_value = float(root_mean_squared_error(y_test, X_pred))

        out = {
            "n_samples": n,
            "mae": mae_value,
            "rmse": rmse_value,
            "coverage95": None,
            "_inside95": None,  # Number of samples inside 95% CI
        }
        # coverage95
        if std_pred is not None:

            aux = z95 * std_pred
            lower_bound = X_pred - aux
            upper_bound = X_pred + aux

            inside = (y_test >= lower_bound) & (y_test <= upper_bound)

            inside_count = int(np.sum(inside))

            out["coverage95"] = float(np.mean(inside))
            out["_inside95"] = inside_count

        return out


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