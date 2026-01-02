from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
class SurrogateRegressor(ABC):
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
    def predict(self, X: np.ndarray):
        """
        Predict target values of X matrix

        Args:
            X (np.ndarray): Features for prediction.
        Returns:
            _type_: Predicted target values.
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
            std: (optional) Uncertainty estimates.
        """
        mean = self.predict(X)

        # TODO: Handle the case where models return both mean and std
        std = None
        return mean, std

    # TODO: Checck the output types
    def rank_candidates(self, Xcand: np.ndarray, k: int = 5, mode: str = "mean", beta: float = 1.0):
        """
        Rank candidate inputs based on predicted target values.

        Args:
            X (np.ndarray): Candidate features to rank.

        Returns:
            np.ndarray: Indices that would sort the candidates by predicted target values.
        """
        mean, std = self.predict(Xcand)
        
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
        
        idx = np.argsort(score)[::-1][:k] # Descending order
        return idx, score[idx], mean[idx], std[idx] if std is not None else None