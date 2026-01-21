# @author: José Arbelaez
from sklearn.cross_decomposition import PLSRegression as SklearnPLS
import numpy as np
from .base import SurrogateRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class PLSSurrogateRegressor(SurrogateRegressor):
    name = "PLSSurrogateRegressor"
    def __init__(self, n_components: int = 2, scale=True, auto_adjust_components: bool = True, **kwargs):
        """
        PLS Surrogate Regressor.
        
        Args:
            n_components: Number of PLS components (default: 2)
            scale: Whether to scale X and y (default: True)
            auto_adjust_components: If True, automatically reduce n_components 
                                   to max(1, min(n_samples, n_features)) when needed.
                                   This prevents errors on low-dimensional datasets.
            **kwargs: Additional arguments passed to sklearn PLSRegression
        """
        self.n_components = n_components
        self.scale = scale
        self.auto_adjust_components = auto_adjust_components
        self.kwargs = kwargs
        self._effective_n_components = None  # Actual components used after adjustment

    def fit(self, X:np.ndarray, y:np.ndarray) -> "SurrogateRegressor":
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        n_samples, n_features = X.shape
        
        # Auto-adjust n_components if needed
        max_components = min(n_samples, n_features)
        
        if self.auto_adjust_components and self.n_components > max_components:
            self._effective_n_components = max(1, max_components)
        else:
            self._effective_n_components = self.n_components

        self.model_ = Pipeline([
                    ("model", SklearnPLS(n_components=self._effective_n_components, scale=self.scale, **self.kwargs))
                    ]) # PLS also scale by default X and y

        self.model_.fit(X, y)
        return self

    def predict(self, X:np.ndarray) -> np.ndarray:
        # Ravel return tu 1D array instead of 2D with one column
        return self.model_.predict(X).ravel()
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'n_components': self.n_components,
            'scale': self.scale,
            'auto_adjust_components': self.auto_adjust_components,
        }
        params.update(self.kwargs)
        return params
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if key in ['n_components', 'scale', 'auto_adjust_components']:
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self

if __name__ == "__main__":
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    pls = PLSSurrogateRegressor(n_components = 5)
    pls.fit(X,y)
    print(pls.predict(X))

    print(pls.score(X,y))