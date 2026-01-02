# @author: José Arbelaez
import numpy as np
from sklearn.dummy import DummyRegressor
from .base import SurrogateRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

class DummySurrogateRegressor(SurrogateRegressor):
    name = "DummySurrogateRegressor"

    def __init__(self, strategy:str="mean"):
        """
        initialize Dummy model with the strategy passed as argument

        Args:
            strategy (str, optional): Defaults to "mean".
        """
        self.strategy = strategy

    def fit(self, X:np.ndarray, y:np.ndarray) -> "SurrogateRegressor":

        self.model_ = Pipeline([
            ("scaler",StandardScaler()),
            ("model", DummyRegressor(strategy=self.strategy))
        ])

        y = np.asarray(y).ravel()
        self.model_.fit(X, y)
        return self

    def predict(self, X:np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        mean_absolute_error_value = mean_absolute_error(y, self.predict(X))
        return -mean_absolute_error_value  # Negate to make it a score (higher is better


# TODO: Add predict_dist and rank_candidates methods
# TODO: Tests
if __name__ == "__main__":
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    dummy = DummySurrogateRegressor(strategy="mean")
    dummy.fit(X,y)
    print(dummy.predict(X))
