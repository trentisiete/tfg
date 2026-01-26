# @author: José Arbelaez
"""
Bagging-based surrogate model using Random Forest.

Random Forest provides:
    - Natural uncertainty quantification via tree variance
    - Robustness to overfitting through ensemble averaging
    - Good performance on small datasets typical in surrogate modeling
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor as SklearnRF
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional

from .base import SurrogateRegressor


class RandomForestSurrogateRegressor(SurrogateRegressor):
    """
    Random Forest surrogate model with uncertainty quantification.

    Uses the variance across trees to estimate predictive uncertainty,
    which is useful for exploration-exploitation trade-offs in optimization.

    Attributes:
        name: Model identifier
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None = unlimited)
        min_samples_leaf: Minimum samples required at leaf nodes
        max_features: Number of features to consider for best split
        bootstrap: Whether to use bootstrap samples

    Example:
        >>> model = RandomForestSurrogateRegressor(n_estimators=100)
        >>> model.fit(X_train, y_train)
        >>> mean, std = model.predict_dist(X_test)
    """
    name = "RandomForestSurrogateRegressor"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        **kwargs
    ):
        """
        Initialize Random Forest surrogate model.

        Args:
            n_estimators: Number of trees in the forest. More trees = smoother
                         uncertainty estimates but slower training.
            max_depth: Maximum depth of trees. None means unlimited depth.
                      Shallower trees can prevent overfitting on small datasets.
            min_samples_leaf: Minimum samples at leaf nodes. Higher values
                             create smoother predictions.
            max_features: Features to consider per split. "sqrt" or "log2"
                         recommended for decorrelating trees.
            bootstrap: Use bootstrap sampling. True recommended for
                      uncertainty quantification.
            random_state: Random seed for reproducibility.
            n_jobs: Parallel jobs (-1 = all cores).
            **kwargs: Additional arguments passed to sklearn RandomForestRegressor.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestSurrogateRegressor":
        """
        Fit the Random Forest model.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)

        Returns:
            self: Fitted model
        """
        self.model_ = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SklearnRF(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self.kwargs
            ))
        ])

        y = np.asarray(y).ravel()
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predicted values of shape (n_samples,)
        """
        return self.model_.predict(X).ravel()

    def predict_dist(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty quantification.

        Uncertainty is computed as the standard deviation of predictions
        across all trees in the forest.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            mean: Mean predictions of shape (n_samples,)
            std: Standard deviation across trees of shape (n_samples,)
        """
        # Transform features
        X_scaled = self.model_.named_steps["scaler"].transform(X)
        rf = self.model_.named_steps["model"]

        # Get predictions from all trees
        all_preds = np.array([tree.predict(X_scaled) for tree in rf.estimators_])

        # Mean and std across trees
        mean = np.mean(all_preds, axis=0)
        std = np.std(all_preds, axis=0)

        # Ensure minimum std for numerical stability
        std = np.maximum(std, 1e-6)

        return mean.ravel(), std.ravel()


if __name__ == "__main__":
    # Quick test
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    rf = RandomForestSurrogateRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)

    mean, std = rf.predict_dist(X)
    print(f"R² score: {rf.score(X, y):.4f}")
    print(f"Mean prediction range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"Std range: [{std.min():.3f}, {std.max():.3f}]")
