# @author: José Arbelaez
from sklearn.gaussian_process import GaussianProcessRegressor as SKLearnGPR
import numpy as np
from .base import SurrogateRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel


class GPSurrogateRegressor(SurrogateRegressor):
    name = "GPSurrogateRegressor"

    def __init__(self, kernel=None, alpha: float = 1e-10, normalize_y: bool = True, n_restarts_optimizer: int = 0):
        """
        initialize Gaussian Process model with the parameters passed as arguments using a pipeline
        Args:
            kernel (_type_, optional): Kernel to use. Defaults to Matern(nu = 3/2) + WhiteKernel().
            alpha (float, optional): Value added to the diagonal of the kernel matrix during fitting. Defaults to 1e-10.
            normalize_y (bool, optional): Whether to normalize the target values. Defaults to True.
        """
        if kernel is None:
            kernel = Matern(nu = 3/2) + WhiteKernel()

        self.model = Pipeline([
            ("scaler",StandardScaler()),
            ("model", SKLearnGPR(kernel=kernel, alpha=alpha, normalize_y=normalize_y, n_restarts_optimizer=n_restarts_optimizer))
            ])

    def fit(self, X:np.ndarray, y:np.ndarray) -> "SurrogateRegressor":

        y = np.asarray(y).ravel()
        self.model.fit(X,y)
        return self

    def _predict_steps(self, X):
        Xs = self.model.named_steps["scaler"].transform(X)

        gpr = self.model.named_steps["model"]
        mean, std = gpr.predict(Xs, return_std=True)

        return mean, std

    def predict(self, X):
        mean, _ = self._predict_steps(X)
        return mean

    def predict_dist(self, X):
        mean, std = self._predict_steps(X)
        return mean, std

