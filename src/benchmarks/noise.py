# @author: José Arbelaez
"""
Noise injection utilities for synthetic benchmark evaluation.

Provides different noise models:
    - NoNoise: Pure function evaluation (interpolation test)
    - GaussianNoise: Homoscedastic Gaussian noise
    - HeteroscedasticNoise: Input-dependent noise variance
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np


@dataclass
class NoiseInjector(ABC):
    """
    Abstract base class for noise injection strategies.

    Noise is added to function evaluations y = f(X) to simulate
    measurement uncertainty or stochastic objectives.
    """
    seed: Optional[int] = None

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    @abstractmethod
    def add_noise(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Add noise to function evaluations.

        Args:
            y: Clean function values of shape (n_samples,)
            X: Input points (optional, needed for heteroscedastic noise)

        Returns:
            np.ndarray: Noisy function values
        """
        pass

    @abstractmethod
    def get_noise_std(self, X: Optional[np.ndarray] = None,
                      y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the standard deviation of noise at each point.

        Useful for calculating theoretical coverage.

        Args:
            X: Input points
            y: Function values (needed for some noise models)

        Returns:
            np.ndarray: Noise std at each point
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}(seed={self.seed})"


@dataclass
class NoNoise(NoiseInjector):
    """
    No noise injection (pure function evaluation).

    Use for testing interpolation capabilities of surrogates.
    A good GP should achieve near-zero error on training points
    when there's no noise.
    """

    def add_noise(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        return y.copy()

    def get_noise_std(self, X: Optional[np.ndarray] = None,
                      y: Optional[np.ndarray] = None) -> np.ndarray:
        n = len(y) if y is not None else (len(X) if X is not None else 1)
        return np.zeros(n)


@dataclass
class GaussianNoise(NoiseInjector):
    """
    Homoscedastic Gaussian noise (constant variance).

    Adds ε ~ N(0, σ²) to function values.

    Args:
        sigma: Standard deviation of noise
        seed: Random seed for reproducibility

    Example:
        >>> noise = GaussianNoise(sigma=0.1, seed=42)
        >>> y_noisy = noise.add_noise(y_clean)
    """
    sigma: float = 0.1

    def add_noise(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        y = np.asarray(y).ravel()
        eps = self._rng.normal(0, self.sigma, size=len(y))
        return y + eps

    def get_noise_std(self, X: Optional[np.ndarray] = None,
                      y: Optional[np.ndarray] = None) -> np.ndarray:
        n = len(y) if y is not None else (len(X) if X is not None else 1)
        return np.full(n, self.sigma)

    def __repr__(self) -> str:
        return f"GaussianNoise(sigma={self.sigma}, seed={self.seed})"


@dataclass
class HeteroscedasticNoise(NoiseInjector):
    """
    Heteroscedastic noise (input-dependent variance).

    Noise variance changes based on input location.
    Many models (especially standard GPs) struggle with this.

    Supports two modes:
        1. 'linear': σ(x) = sigma_base + sigma_scale * ||x||
        2. 'custom': User-provided function σ(X) -> sigmas

    Args:
        sigma_base: Base noise level (minimum)
        sigma_scale: Scale factor for position-dependent component
        mode: 'linear' or 'custom'
        sigma_func: Custom function X -> sigma (if mode='custom')
        seed: Random seed

    Example:
        # Linear mode: noise increases with distance from origin
        >>> noise = HeteroscedasticNoise(sigma_base=0.05, sigma_scale=0.2)

        # Custom mode: noise based on first coordinate
        >>> noise = HeteroscedasticNoise(
        ...     mode='custom',
        ...     sigma_func=lambda X: 0.1 + 0.3 * np.abs(X[:, 0])
        ... )
    """
    sigma_base: float = 0.05
    sigma_scale: float = 0.2
    mode: str = "linear"
    sigma_func: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def _compute_sigma(self, X: np.ndarray) -> np.ndarray:
        """Compute noise std at each input point."""
        X = np.atleast_2d(X)

        if self.mode == "custom" and self.sigma_func is not None:
            return np.asarray(self.sigma_func(X)).ravel()

        # Default: linear mode - noise increases with ||x||
        # Normalize X to [0,1] range effect
        norms = np.linalg.norm(X, axis=1)
        max_norm = np.sqrt(X.shape[1])  # Max norm in unit hypercube
        normalized = norms / max_norm

        return self.sigma_base + self.sigma_scale * normalized

    def add_noise(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        y = np.asarray(y).ravel()

        if X is None:
            raise ValueError("HeteroscedasticNoise requires X to compute noise levels")

        sigmas = self._compute_sigma(X)
        eps = self._rng.normal(0, 1, size=len(y)) * sigmas

        return y + eps

    def get_noise_std(self, X: Optional[np.ndarray] = None,
                      y: Optional[np.ndarray] = None) -> np.ndarray:
        if X is None:
            raise ValueError("HeteroscedasticNoise requires X to compute noise levels")
        return self._compute_sigma(X)

    def __repr__(self) -> str:
        if self.mode == "custom":
            return f"HeteroscedasticNoise(mode='custom', seed={self.seed})"
        return f"HeteroscedasticNoise(base={self.sigma_base}, scale={self.sigma_scale}, seed={self.seed})"


@dataclass
class ProportionalNoise(NoiseInjector):
    """
    Noise proportional to function value magnitude.

    σ(x) = sigma_rel * |y(x)| + sigma_base

    Useful for simulating measurement noise that scales with signal.

    Args:
        sigma_rel: Relative noise (fraction of |y|)
        sigma_base: Base noise level (minimum)
        seed: Random seed
    """
    sigma_rel: float = 0.05
    sigma_base: float = 0.01

    def add_noise(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> np.ndarray:
        y = np.asarray(y).ravel()
        sigmas = self.sigma_base + self.sigma_rel * np.abs(y)
        eps = self._rng.normal(0, 1, size=len(y)) * sigmas
        return y + eps

    def get_noise_std(self, X: Optional[np.ndarray] = None,
                      y: Optional[np.ndarray] = None) -> np.ndarray:
        if y is None:
            raise ValueError("ProportionalNoise requires y to compute noise levels")
        y = np.asarray(y).ravel()
        return self.sigma_base + self.sigma_rel * np.abs(y)

    def __repr__(self) -> str:
        return f"ProportionalNoise(rel={self.sigma_rel}, base={self.sigma_base})"


# =============================================================================
# NOISE FACTORY
# =============================================================================

NOISE_REGISTRY = {
    "none": NoNoise,
    "zero": NoNoise,  # alias
    "gaussian": GaussianNoise,
    "normal": GaussianNoise,  # alias
    "heteroscedastic": HeteroscedasticNoise,
    "hetero": HeteroscedasticNoise,  # alias
    "proportional": ProportionalNoise,
}


def get_noise_injector(name: str, seed: Optional[int] = None, **kwargs) -> NoiseInjector:
    """
    Factory function to create noise injector instances by name.

    Args:
        name: Noise type identifier (case-insensitive)
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to noise constructor

    Returns:
        NoiseInjector instance

    Example:
        >>> noise = get_noise_injector("gaussian", sigma=0.1, seed=42)
        >>> y_noisy = noise.add_noise(y_clean)
    """
    name_lower = name.lower()
    if name_lower not in NOISE_REGISTRY:
        available = ", ".join(set(NOISE_REGISTRY.keys()))
        raise ValueError(f"Unknown noise type '{name}'. Available: {available}")

    return NOISE_REGISTRY[name_lower](seed=seed, **kwargs)


if __name__ == "__main__":
    # Test noise injectors
    print("Testing noise injectors...")

    np.random.seed(42)
    X = np.random.rand(100, 2)
    y_clean = np.sin(X[:, 0] * 2 * np.pi)

    for name in ["none", "gaussian", "heteroscedastic", "proportional"]:
        if name == "gaussian":
            noise = get_noise_injector(name, sigma=0.1, seed=42)
        elif name == "proportional":
            noise = get_noise_injector(name, sigma_rel=0.05, seed=42)
        else:
            noise = get_noise_injector(name, seed=42)

        y_noisy = noise.add_noise(y_clean, X)
        diff = np.abs(y_noisy - y_clean)
        print(f"  {name:15s} | mean_diff={diff.mean():.4f} | max_diff={diff.max():.4f}")
