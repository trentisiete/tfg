# @author: José Arbelaez
from sklearn.cross_decomposition import PLSRegression as SklearnPLS
import numpy as np
from .base import SurrogateRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class PLSSurrogateRegressor(SurrogateRegressor):
    name = "PLSSurrogateRegressor"
    def __init__(self, n_components: int = 2, scale=True):
        self.model = Pipeline([
            ("model", SklearnPLS(n_components=n_components, scale=scale)
            )]) # PLS also scale by default X and y

    def fit(self, X:np.ndarray, y:np.ndarray) -> "SurrogateRegressor":

        y = np.asarray(y).ravel()
        self.model.fit(X,y)
        return self

    def predict(self, X:np.ndarray) -> np.ndarray:
        # Ravel return tu 1D array instead of 2D with one column
        return self.model.predict(X).ravel()

if __name__ == "__main__":
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    pls = PLSSurrogateRegressor(n_components = 5)
    pls.fit(X,y)
    print(pls.predict(X))