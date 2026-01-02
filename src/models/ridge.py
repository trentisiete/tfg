# @author: José Arbelaez
from sklearn.linear_model import Ridge as SklearnRidge
import numpy as np
from .base import SurrogateRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class RidgeSurrogateRegressor(SurrogateRegressor):
    name = "RidgeSurrogateRegressor"

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        """
        initialize Ridge model with the parameters passed as arguments using a pipeline
        Args:
            alpha (float, optional): _description_. Defaults to 1.0.
            fit_intercept (bool, optional): _description_. Defaults to True.
        """
        self.model = Pipeline([
            ("scaler",StandardScaler()),
            ("model", SklearnRidge(alpha=alpha, fit_intercept=fit_intercept))
            ])

    def fit(self, X:np.ndarray, y:np.ndarray) -> "SurrogateRegressor":

        y = np.asarray(y).ravel()
        self.model.fit(X,y)
        return self

    def predict(self, X:np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# TODO: Add predict_dist and rank_candidates methods
# TODO: Tests
if __name__ == "__main__":
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    
    ridge = RidgeSurrogateRegressor(alpha = 1.0, fit_intercept=True)
    ridge.fit(X,y)
    print(ridge.predict(X))