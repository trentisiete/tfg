# @author: José Arbelaez
"""
Sampling strategies for Design of Experiments (DOE).

Provides different methods for generating space-filling designs:
    - Sobol sequences (quasi-random, best default for BO)
    - Latin Hypercube Sampling (classic in engineering)

    # These two will not be used in the TFG, but maintained for completeness and consistency.
    - Grid sampling (for visualization in low dimensions)
    - Random sampling (baseline comparison)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class SamplingStrategy(ABC):
    """
    Abstract base class for sampling strategies.

    All samplers generate points in [0, 1]^d by default.
    Use scale_to_bounds() to transform to actual domain.
    """
    seed: Optional[int] = None

    @abstractmethod
    def sample(self, n_samples: int, dim: int) -> np.ndarray:
        """
        Generate n_samples in [0, 1]^d.

        Args:
            n_samples: Number of points to generate
            dim: Dimensionality of the space

        Returns:
            np.ndarray: Array of shape (n_samples, dim) in [0,1]^d

        Example:
            >>> sampler = SobolSampler(seed=42)
            >>> X = sampler.sample(100, dim=3) -> shape (100, 3) with values in [0, 1]
        """
        pass

    def sample_bounds(self, n_samples: int,
                      bounds: List[Tuple[float, float]]) -> np.ndarray:
        """
        Generate samples directly scaled to given bounds.

        Args:
            n_samples: Number of points to generate
            bounds: List of (min, max) tuples for each dimension

        Returns:
            np.ndarray: Points scaled to specified bounds
        """
        dim = len(bounds)
        X_unit = self.sample(n_samples, dim)

        bounds_arr = np.array(bounds)
        lb, ub = bounds_arr[:, 0], bounds_arr[:, 1]

        return X_unit * (ub - lb) + lb

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seed={self.seed})"


@dataclass
class SobolSampler(SamplingStrategy):
    """
    Sobol quasi-random sequence sampler.

    Provides excellent space-filling properties and is the recommended
    default for surrogate modeling and Bayesian optimization.

    Uses scipy.stats.qmc.Sobol for generation.

    Args:
        seed: Random seed for reproducibility
        scramble: Whether to apply Owen scrambling (recommended)
    """
    scramble: bool = True

    def sample(self, n_samples: int, dim: int) -> np.ndarray:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=dim, scramble=self.scramble, seed=self.seed)

        # EXPLAIN_AT:
        # Sobol sequences work best with powers of 2
        # We generate enough and take first n_samples
        return sampler.random(n=n_samples)


@dataclass
class LatinHypercubeSampler(SamplingStrategy):
    """
    Latin Hypercube Sampling (LHS).

    Classic space-filling design used extensively in engineering.
    Ensures each row and column in the design matrix has exactly
    one sample.

    Args:
        seed: Random seed for reproducibility
        optimization: Whether to optimize the LHS design ("random" or "random-cd")
        strength: Strength of the LHS (1 for standard, 2 for orthogonal)
    
    Note: This sampler is not part of the TFG's experiments.
    """
    optimization: Optional[str] = "random-cd"
    strength: int = 1

    def sample(self, n_samples: int, dim: int) -> np.ndarray:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(
            d=dim,
            seed=self.seed,
            optimization=self.optimization,
            strength=self.strength
        )

        return sampler.random(n=n_samples)


@dataclass
class GridSampler(SamplingStrategy):
    """
    Regular grid sampling.

    Best used for visualization in 1D or 2D.
    WARNING: Scales poorly with dimension (curse of dimensionality).

    For n_samples points in d dimensions, uses approximately
    n_samples^(1/d) points per dimension.

    Args:
        seed: Ignored for grid sampling (deterministic)

    Note: This sampler is not part of the TFG's experiments.
    """

    def sample(self, n_samples: int, dim: int) -> np.ndarray:
        # Calculate points per dimension
        points_per_dim = max(2, int(np.ceil(n_samples ** (1 / dim))))

        # Create 1D grids for each dimension
        grids = [np.linspace(0, 1, points_per_dim) for _ in range(dim)]

        # Create meshgrid and reshape
        mesh = np.meshgrid(*grids, indexing='ij')
        X = np.stack([m.ravel() for m in mesh], axis=-1)

        # If we have more points than requested, subsample uniformly
        if len(X) > n_samples:
            indices = np.linspace(0, len(X) - 1, n_samples, dtype=int)
            X = X[indices]

        return X

    def sample_1d(self, n_samples: int) -> np.ndarray:
        """Convenience method for 1D grids."""
        return np.linspace(0, 1, n_samples).reshape(-1, 1)

    def sample_2d(self, n_per_dim: int) -> np.ndarray:
        """
        Convenience method for 2D grids with explicit points per dimension.

        Returns:
            np.ndarray: Grid of shape (n_per_dim^2, 2)
        """
        x = np.linspace(0, 1, n_per_dim)
        X1, X2 = np.meshgrid(x, x)
        return np.column_stack([X1.ravel(), X2.ravel()])


@dataclass
class RandomSampler(SamplingStrategy):
    """
    Uniform random sampling.

    Simple baseline for comparison. Generally inferior to
    Sobol or LHS for space-filling.

    Args:
        seed: Random seed for reproducibility
    """

    def sample(self, n_samples: int, dim: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.random((n_samples, dim))


# SAMPLER FACTORY

SAMPLER_REGISTRY = {
    "sobol": SobolSampler,
    "lhs": LatinHypercubeSampler,
    "latin": LatinHypercubeSampler,  # alias
    "grid": GridSampler,
    "random": RandomSampler,
    "uniform": RandomSampler,  # alias
}


def get_sampler(name: str, seed: Optional[int] = None, **kwargs) -> SamplingStrategy:
    """
    Factory function to create sampler instances by name.

    Args:
        name: Sampler identifier (case-insensitive)
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to sampler constructor

    Returns:
        SamplingStrategy instance

    Example:
        >>> sampler = get_sampler("sobol", seed=42)
        >>> X = sampler.sample(100, dim=3)
    """
    name_lower = name.lower()
    if name_lower not in SAMPLER_REGISTRY:
        available = ", ".join(set(SAMPLER_REGISTRY.keys()))
        raise ValueError(f"Unknown sampler '{name}'. Available: {available}")

    return SAMPLER_REGISTRY[name_lower](seed=seed, **kwargs)


if __name__ == "__main__":
    # Visual test of sampling strategies
    print("Testing sampling strategies...")

    for name in ["sobol", "lhs", "grid", "random"]:
        sampler = get_sampler(name, seed=42)
        X = sampler.sample(100, dim=2)
        print(f"  {name:10s} | shape={X.shape} | range=[{X.min():.3f}, {X.max():.3f}]")

    # Test with bounds
    bounds = [(0, 10), (-5, 5)]
    sampler = get_sampler("sobol", seed=42)
    X_scaled = sampler.sample_bounds(50, bounds)
    print(f"\nSobol with bounds {bounds}:")
    print(f"  X[:,0] range: [{X_scaled[:,0].min():.2f}, {X_scaled[:,0].max():.2f}]")
    print(f"  X[:,1] range: [{X_scaled[:,1].min():.2f}, {X_scaled[:,1].max():.2f}]")
