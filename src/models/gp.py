# @author: José Arbelaez
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor as SKLearnGPR
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import SurrogateRegressor

# Silence bound-hitting GP hyperparameter warnings from sklearn
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.gaussian_process.kernels")

class GPSurrogateRegressor(SurrogateRegressor):
    name = "GPSurrogateRegressor"

    def __init__(self, kernel=None, alpha: float = 1e-10, normalize_y: bool = True, n_restarts_optimizer: int = 0, noise_level: float = 1e-05, **kwargs):
        """
        initialize Gaussian Process model with the parameters passed as arguments using a pipeline
        Args:
            kernel (_type_, optional): Kernel to use. Defaults to Matern(nu = 3/2) + WhiteKernel().
            alpha (float, optional): Value added to the diagonal of the kernel matrix during fitting. Defaults to 1e-10.
            normalize_y (bool, optional): Whether to normalize the target values. Defaults to True.
        """

        if kernel is None:
            kernel = Matern(nu = 3/2) + WhiteKernel(noise_level=noise_level)

        self.kernel = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.noise_level = noise_level
        self.kwargs = kwargs

    def fit(self, X:np.ndarray, y:np.ndarray) -> "SurrogateRegressor":

        self.model_ = Pipeline([
            ("scaler",StandardScaler()),
            ("model", SKLearnGPR(kernel=self.kernel, alpha=self.alpha, normalize_y=self.normalize_y, n_restarts_optimizer=self.n_restarts_optimizer, **self.kwargs))
            ])

        y = np.asarray(y).ravel()
        self.model_.fit(X,y)
        return self

    def _predict_steps(self, X):
        Xs = self.model_.named_steps["scaler"].transform(X)

        gpr = self.model_.named_steps["model"]
        mean, std = gpr.predict(Xs, return_std=True)

        return np.asarray(mean).ravel(), np.asarray(std).ravel()

    def predict(self, X):
        mean, _ = self._predict_steps(X)
        return mean

    def predict_dist(self, X):
        mean, std = self._predict_steps(X)
        return mean, std

if __name__ == "__main__":
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    gp = GPSurrogateRegressor()
    gp.fit(X,y)
    print(gp.predict(X))
    print(gp.score(X,y))