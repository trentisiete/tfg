# @author: José Arbelaez
from sklearn.cross_decomposition import PLSRegression as SklearnPLS
import numpy as np
from .base import SurrogateRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class PLSSurrogateRegressor(SurrogateRegressor):
    name = "PLSSurrogateRegressor"
    def __init__(self, n_components: int = 2, scale=True, **kwargs):
        self.n_components = n_components
        self.scale = scale
        self.kwargs = kwargs

    def fit(self, X:np.ndarray, y:np.ndarray) -> "SurrogateRegressor":

        self.model_ = Pipeline([
                    ("model", SklearnPLS(n_components=self.n_components, scale=self.scale, **self.kwargs))
                    ]) # PLS also scale by default X and y

        y = np.asarray(y).ravel()
        self.model_.fit(X,y)
        return self

    def predict(self, X:np.ndarray) -> np.ndarray:
        # Ravel return tu 1D array instead of 2D with one column
        return self.model_.predict(X).ravel()

if __name__ == "__main__":
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    pls = PLSSurrogateRegressor(n_components = 5)
    pls.fit(X,y)
    print(pls.predict(X))

    print(pls.score(X,y))