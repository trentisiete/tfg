# @author: José Arbelaez
from sklearn.cross_decomposition import PLSRegression as SklearnPLS
import numpy as np
from .base import SurrogateRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

class PLSSurrogateRegressor(SurrogateRegressor):
    name = "PLSSurrogateRegressor"
    def __init__(self, n_components: int = 2, scale=True):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X:np.ndarray, y:np.ndarray) -> "SurrogateRegressor":

        self.model_ = Pipeline([
                    ("model", SklearnPLS(n_components=self.n_components, scale=self.scale))
                    ]) # PLS also scale by default X and y

        y = np.asarray(y).ravel()
        self.model_.fit(X,y)
        return self

    def predict(self, X:np.ndarray) -> np.ndarray:
        # Ravel return tu 1D array instead of 2D with one column
        return self.model_.predict(X).ravel()

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        mean_absolute_error_value = mean_absolute_error(y, self.predict(X))
        return -mean_absolute_error_value  # Negate to make it a score (higher is better

if __name__ == "__main__":
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    pls = PLSSurrogateRegressor(n_components = 5)
    pls.fit(X,y)
    print(pls.predict(X))