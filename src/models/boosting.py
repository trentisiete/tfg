# @author: José Arbelaez
"""
Boosting-based surrogate model using Gradient Boosting.

Gradient Boosting provides:
    - Strong predictive performance through sequential learning
    - Natural handling of non-linear relationships
    - Quantile regression for uncertainty estimation
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional

from .base import SurrogateRegressor


class GradientBoostingSurrogateRegressor(SurrogateRegressor):
    """
    Gradient Boosting surrogate model with uncertainty quantification.
    
    Uses quantile regression to estimate predictive uncertainty by fitting
    separate models for lower and upper quantiles.
    
    Attributes:
        name: Model identifier
        n_estimators: Number of boosting stages
        learning_rate: Shrinkage parameter
        max_depth: Maximum depth of individual trees
        min_samples_leaf: Minimum samples at leaf nodes
        subsample: Fraction of samples for fitting each tree
        
    Example:
        >>> model = GradientBoostingSurrogateRegressor(n_estimators=100)
        >>> model.fit(X_train, y_train)
        >>> mean, std = model.predict_dist(X_test)
    """
    name = "GradientBoostingSurrogateRegressor"

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
        quantile_lower: float = 0.159,  # ~mean - 1*std for normal dist
        quantile_upper: float = 0.841,  # ~mean + 1*std for normal dist
        **kwargs
    ):
        """
        Initialize Gradient Boosting surrogate model.
        
        Args:
            n_estimators: Number of boosting stages. More = better fit but 
                         slower and risk of overfitting.
            learning_rate: Shrinkage factor for each tree's contribution.
                          Lower values need more estimators but often 
                          generalize better.
            max_depth: Maximum depth of individual trees. Shallow trees (3-5)
                      prevent overfitting.
            min_samples_leaf: Minimum samples at leaf nodes.
            subsample: Fraction of samples for each tree (<1.0 adds 
                      stochasticity, like dropout).
            random_state: Random seed for reproducibility.
            quantile_lower: Lower quantile for uncertainty (default: ~16th percentile)
            quantile_upper: Upper quantile for uncertainty (default: ~84th percentile)
            **kwargs: Additional arguments passed to sklearn GradientBoostingRegressor.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.quantile_lower = quantile_lower
        self.quantile_upper = quantile_upper
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingSurrogateRegressor":
        """
        Fit the Gradient Boosting model.
        
        Fits three models:
            1. Mean predictor (squared error loss)
            2. Lower quantile predictor
            3. Upper quantile predictor
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            
        Returns:
            self: Fitted model
        """
        y = np.asarray(y).ravel()
        
        # Scaler (shared)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Main model for mean prediction
        self.model_mean_ = SklearnGBR(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
            loss="squared_error",
            **self.kwargs
        )
        self.model_mean_.fit(X_scaled, y)
        
        # Lower quantile model for uncertainty
        self.model_lower_ = SklearnGBR(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
            loss="quantile",
            alpha=self.quantile_lower,
            **self.kwargs
        )
        self.model_lower_.fit(X_scaled, y)
        
        # Upper quantile model for uncertainty
        self.model_upper_ = SklearnGBR(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
            loss="quantile",
            alpha=self.quantile_upper,
            **self.kwargs
        )
        self.model_upper_.fit(X_scaled, y)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted values of shape (n_samples,)
        """
        X_scaled = self.scaler_.transform(X)
        return self.model_mean_.predict(X_scaled).ravel()

    def predict_dist(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification.
        
        Uncertainty is estimated from the difference between upper and lower
        quantile predictions. For a normal distribution, the interval between
        16th and 84th percentiles covers approximately 1 standard deviation
        on each side.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            mean: Mean predictions of shape (n_samples,)
            std: Estimated standard deviation of shape (n_samples,)
        """
        X_scaled = self.scaler_.transform(X)
        
        # Mean prediction
        mean = self.model_mean_.predict(X_scaled)
        
        # Quantile predictions
        lower = self.model_lower_.predict(X_scaled)
        upper = self.model_upper_.predict(X_scaled)
        
        # Estimate std from quantile range
        # For normal distribution: (84th - 16th) percentile ≈ 2 * std
        std = (upper - lower) / 2.0
        
        # Ensure minimum std for numerical stability
        std = np.maximum(std, 1e-6)
        
        return mean.ravel(), std.ravel()


if __name__ == "__main__":
    # Quick test
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    gb = GradientBoostingSurrogateRegressor(n_estimators=50, random_state=42)
    gb.fit(X, y)
    
    mean, std = gb.predict_dist(X)
    print(f"R² score: {gb.score(X, y):.4f}")
    print(f"Mean prediction range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"Std range: [{std.min():.3f}, {std.max():.3f}]")
