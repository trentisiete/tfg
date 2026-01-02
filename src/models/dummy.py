# @author: José Arbelaez
import numpy as np
from sklearn.dummy import DummyRegressor
from .base import SurrogateRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class DummySurrogateRegressor(SurrogateRegressor):
    name = "DummySurrogateRegressor"

    def __init__(self, strategy:str="mean"):
        """
        initialize Dummy model with the strategy passed as argument

        Args:
            strategy (str, optional): Defaults to "mean".
        """
        self.strategy = strategy
        self.model = Pipeline([
            ("scaler",StandardScaler()),
            ("model", DummyRegressor(strategy=self.strategy))
        ])
    def fit(self, X:np.ndarray, y:np.ndarray) -> "SurrogateRegressor":

        y = np.asarray(y).ravel()
        self.model.fit(X, y)
        return self

    def predict(self, X:np.ndarray) -> np.ndarray:
        return self.model.predict(X)

# TODO: Add predict_dist and rank_candidates methods
# TODO: Tests
if __name__ == "__main__":
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    dummy = DummySurrogateRegressor(strategy="mean")
    dummy.fit(X,y)
    print(dummy.predict(X))
