# @author: José Arbelaez
from sklearn.linear_model import Ridge as SklearnRidge
import numpy as np
from .base import SurrogateRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class RidgeSurrogateRegressor(SurrogateRegressor):
    name = "RidgeSurrogateRegressor"

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True, **kwargs):
        """
        initialize Ridge model with the parameters passed as arguments using a pipeline
        Args:
            alpha (float, optional): _description_. Defaults to 1.0.
            fit_intercept (bool, optional): _description_. Defaults to True.
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.kwargs = kwargs


    def fit(self, X:np.ndarray, y:np.ndarray) -> "SurrogateRegressor":
        self.model_ = Pipeline([
                    ("scaler",StandardScaler()),
                    ("model", SklearnRidge(alpha=self.alpha, fit_intercept=self.fit_intercept, **self.kwargs))
                    ])

        y = np.asarray(y).ravel()
        self.model_.fit(X,y)
        return self

    def predict(self, X:np.ndarray) -> np.ndarray:
        return self.model_.predict(X).ravel()


if __name__ == "__main__":
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    ridge = RidgeSurrogateRegressor(alpha = 1.0, fit_intercept=True)
    ridge.fit(X,y)
    print(ridge.score(X,y))