# @author: José Arbelaez
"""
Classical benchmark functions for surrogate model evaluation.

All functions follow a consistent interface through the BenchmarkFunction ABC.
Each function defines its bounds, optimal value, and evaluation method.

References:
    - Forrester et al. (2008) "Engineering Design via Surrogate Modelling"
    - Surjanovic & Bingham: https://www.sfu.ca/~ssurjano/optimization.html
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Type
import numpy as np


@dataclass
class BenchmarkFunction(ABC):
    """
    Abstract base class for benchmark functions.

    All benchmark functions must define:
        - name: Human-readable identifier
        - dim: Input dimensionality
        - bounds: List of (min, max) tuples for each dimension
        - optimal_value: Known global minimum (for reference)

    Usage:
        >>> bench = Forrester1D()
        >>> X = np.random.rand(100, 1) * 1  # Scale to bounds
        >>> y = bench(X)
    """
    name: str = field(init=False)
    dim: int = field(init=False)
    bounds: List[Tuple[float, float]] = field(init=False)
    optimal_value: float = field(init=False)
    optimal_location: Optional[np.ndarray] = field(init=False, default=None)

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the function at points X.

        Args:
            X: Array of shape (n_samples, dim) with input points.
               Each row is a point, columns are dimensions.

        Returns:
            np.ndarray: Function values of shape (n_samples,)
        """
        pass

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Alias for __call__ for explicit evaluation."""
        return self(X)

    def get_bounds_array(self) -> np.ndarray:
        """Returns bounds as numpy array of shape (dim, 2)."""
        return np.array(self.bounds)

    def scale_to_bounds(self, X_unit: np.ndarray) -> np.ndarray:
        """
        Scale points from [0,1]^d to actual bounds.

        Args:
            X_unit: Points in unit hypercube [0,1]^d

        Returns:
            X scaled to the function's actual bounds
        """
        bounds = self.get_bounds_array()
        lb, ub = bounds[:, 0], bounds[:, 1]
        return X_unit * (ub - lb) + lb

    def scale_from_bounds(self, X: np.ndarray) -> np.ndarray:
        """
        Scale points from actual bounds to [0,1]^d.

        Args:
            X: Points in actual bounds

        Returns:
            X scaled to unit hypercube [0,1]^d
        """
        bounds = self.get_bounds_array()
        lb, ub = bounds[:, 0], bounds[:, 1]
        return (X - lb) / (ub - lb)

    def __repr__(self) -> str:
        return f"{self.name}(dim={self.dim}, optimal={self.optimal_value:.6f})"


# =============================================================================
# LOW DIMENSIONAL BENCHMARKS (1D - 3D)
# =============================================================================

@dataclass
class Forrester1D(BenchmarkFunction):
    """
    Forrester function (1D).

    A simple 1D function commonly used in BO/GP literature.
    Has one global minimum and one local minimum.

    f(x) = (6x - 2)² sin(12x - 4)

    Domain: x ∈ [0, 1]
    Global minimum: f(x*) ≈ -6.0207 at x* ≈ 0.7572
    """

    def __post_init__(self):
        self.name = "Forrester1D"
        self.dim = 1
        self.bounds = [(0.0, 1.0)]
        self.optimal_value = -6.020740
        self.optimal_location = np.array([0.757249])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        x = X[:, 0]
        return ((6 * x - 2) ** 2) * np.sin(12 * x - 4)


@dataclass
class Branin2D(BenchmarkFunction):
    """
    Branin (Branin-Hoo) function (2D).

    A multimodal function with 3 global minima.
    Commonly used as a benchmark in Bayesian optimization.

    f(x) = a(x₂ - bx₁² + cx₁ - r)² + s(1-t)cos(x₁) + s

    With: a=1, b=5.1/(4π²), c=5/π, r=6, s=10, t=1/(8π)

    Domain: x₁ ∈ [-5, 10], x₂ ∈ [0, 15]
    Global minimum: f(x*) = 0.397887 at three locations
    """

    def __post_init__(self):
        self.name = "Branin2D"
        self.dim = 2
        self.bounds = [(-5.0, 10.0), (0.0, 15.0)]
        self.optimal_value = 0.397887
        self.optimal_location = np.array([
            [-np.pi, 12.275],
            [np.pi, 2.275],
            [9.42478, 2.475]
        ])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        x1, x2 = X[:, 0], X[:, 1]

        a = 1.0
        b = 5.1 / (4 * np.pi ** 2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8 * np.pi)

        term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
        term2 = s * (1 - t) * np.cos(x1)

        return term1 + term2 + s


@dataclass
class SixHumpCamel2D(BenchmarkFunction):
    """
    Six-Hump Camel function (2D).

    A multimodal function with 6 local minima, 2 of which are global.

    f(x) = (4 - 2.1x₁² + x₁⁴/3)x₁² + x₁x₂ + (-4 + 4x₂²)x₂²

    Domain: x₁ ∈ [-3, 3], x₂ ∈ [-2, 2]
    Global minimum: f(x*) = -1.0316 at (±0.0898, ∓0.7126)
    """

    def __post_init__(self):
        self.name = "SixHumpCamel2D"
        self.dim = 2
        self.bounds = [(-3.0, 3.0), (-2.0, 2.0)]
        self.optimal_value = -1.031628
        self.optimal_location = np.array([
            [0.0898, -0.7126],
            [-0.0898, 0.7126]
        ])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        x1, x2 = X[:, 0], X[:, 1]

        term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2 ** 2) * x2 ** 2

        return term1 + term2 + term3


@dataclass
class GoldsteinPrice2D(BenchmarkFunction):
    """
    Goldstein-Price function (2D).

    A highly nonlinear function used as a challenging optimization benchmark.

    Domain: x₁, x₂ ∈ [-2, 2]
    Global minimum: f(x*) = 3 at (0, -1)
    """

    def __post_init__(self):
        self.name = "GoldsteinPrice2D"
        self.dim = 2
        self.bounds = [(-2.0, 2.0), (-2.0, 2.0)]
        self.optimal_value = 3.0
        self.optimal_location = np.array([0.0, -1.0])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        x1, x2 = X[:, 0], X[:, 1]

        term1 = 1 + (x1 + x2 + 1) ** 2 * (
            19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
        )
        term2 = 30 + (2 * x1 - 3 * x2) ** 2 * (
            18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2
        )

        return term1 * term2


@dataclass
class Hartmann3D(BenchmarkFunction):
    """
    Hartmann 3D function.

    A multimodal function with 4 local minima.
    Commonly used to test optimization and surrogate algorithms.

    Domain: xᵢ ∈ [0, 1] for i = 1, 2, 3
    Global minimum: f(x*) ≈ -3.86278 at (0.114614, 0.555649, 0.852547)
    """

    def __post_init__(self):
        self.name = "Hartmann3D"
        self.dim = 3
        self.bounds = [(0.0, 1.0)] * 3
        self.optimal_value = -3.86278
        self.optimal_location = np.array([0.114614, 0.555649, 0.852547])

        # Hartmann parameters
        self._alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self._A = np.array([
            [3.0, 10, 30],
            [0.1, 10, 35],
            [3.0, 10, 30],
            [0.1, 10, 35]
        ])
        self._P = np.array([
            [0.3689, 0.1170, 0.2673],
            [0.4699, 0.4387, 0.7470],
            [0.1091, 0.8732, 0.5547],
            [0.0381, 0.5743, 0.8828]
        ])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        n = X.shape[0]
        result = np.zeros(n)

        for i in range(n):
            outer_sum = 0.0
            for j in range(4):
                inner_sum = np.sum(self._A[j] * (X[i] - self._P[j]) ** 2)
                outer_sum += self._alpha[j] * np.exp(-inner_sum)
            result[i] = -outer_sum

        return result


@dataclass
class Ishigami3D(BenchmarkFunction):
    """
    Ishigami function (3D).

    Commonly used in sensitivity analysis and uncertainty quantification.
    Has strong nonlinearity and variable interactions.

    f(x) = sin(x₁) + a·sin²(x₂) + b·x₃⁴·sin(x₁)

    With: a = 7, b = 0.1

    Domain: xᵢ ∈ [-π, π] for i = 1, 2, 3
    """

    a: float = 7.0
    b: float = 0.1

    def __post_init__(self):
        self.name = "Ishigami3D"
        self.dim = 3
        self.bounds = [(-np.pi, np.pi)] * 3
        self.optimal_value = None  # Not a simple optimization target
        self.optimal_location = None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]

        term1 = np.sin(x1)
        term2 = self.a * np.sin(x2) ** 2
        term3 = self.b * (x3 ** 4) * np.sin(x1)

        return term1 + term2 + term3


# =============================================================================
# MEDIUM DIMENSIONAL BENCHMARKS (6D - 10D)
# =============================================================================

@dataclass
class Hartmann6D(BenchmarkFunction):
    """
    Hartmann 6D function.

    A multimodal function widely used as a benchmark for optimization
    algorithms in moderate dimensions.

    Domain: xᵢ ∈ [0, 1] for i = 1, ..., 6
    Global minimum: f(x*) ≈ -3.32237
    """

    def __post_init__(self):
        self.name = "Hartmann6D"
        self.dim = 6
        self.bounds = [(0.0, 1.0)] * 6
        self.optimal_value = -3.32237
        self.optimal_location = np.array([
            0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573
        ])

        self._alpha = np.array([1.0, 1.2, 3.0, 3.2])
        self._A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ])
        self._P = np.array([
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
        ])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        n = X.shape[0]
        result = np.zeros(n)

        for i in range(n):
            outer_sum = 0.0
            for j in range(4):
                inner_sum = np.sum(self._A[j] * (X[i] - self._P[j]) ** 2)
                outer_sum += self._alpha[j] * np.exp(-inner_sum)
            result[i] = -outer_sum

        return result


@dataclass
class Borehole8D(BenchmarkFunction):
    """
    Borehole function (8D).

    A physically-motivated benchmark representing water flow through a borehole.
    Commonly used in engineering surrogate modeling.

    Models water flow rate through a borehole drilled from an upper aquifer
    to a lower aquifer.

    Domain:
        rw ∈ [0.05, 0.15]    - radius of borehole (m)
        r  ∈ [100, 50000]    - radius of influence (m)
        Tu ∈ [63070, 115600] - transmissivity of upper aquifer (m²/yr)
        Hu ∈ [990, 1110]     - potentiometric head of upper aquifer (m)
        Tl ∈ [63.1, 116]     - transmissivity of lower aquifer (m²/yr)
        Hl ∈ [700, 820]      - potentiometric head of lower aquifer (m)
        L  ∈ [1120, 1680]    - length of borehole (m)
        Kw ∈ [9855, 12045]   - hydraulic conductivity of borehole (m/yr)
    """

    def __post_init__(self):
        self.name = "Borehole8D"
        self.dim = 8
        self.bounds = [
            (0.05, 0.15),       # rw
            (100.0, 50000.0),   # r
            (63070.0, 115600.0),# Tu
            (990.0, 1110.0),    # Hu
            (63.1, 116.0),      # Tl
            (700.0, 820.0),     # Hl
            (1120.0, 1680.0),   # L
            (9855.0, 12045.0)   # Kw
        ]
        self.optimal_value = None  # Physical model, no simple optimum
        self.optimal_location = None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)

        rw = X[:, 0]  # radius of borehole
        r = X[:, 1]   # radius of influence
        Tu = X[:, 2]  # transmissivity upper aquifer
        Hu = X[:, 3]  # potentiometric head upper
        Tl = X[:, 4]  # transmissivity lower aquifer
        Hl = X[:, 5]  # potentiometric head lower
        L = X[:, 6]   # length of borehole
        Kw = X[:, 7]  # hydraulic conductivity

        numerator = 2 * np.pi * Tu * (Hu - Hl)

        log_term = np.log(r / rw)
        denom1 = 1 + (2 * L * Tu) / (log_term * rw ** 2 * Kw)
        denom2 = Tu / Tl

        return numerator / (log_term * (denom1 + denom2))


@dataclass
class WingWeight10D(BenchmarkFunction):
    """
    Wing Weight function (10D).

    Models the weight of a light aircraft wing as a function of design parameters.

    Domain:
        Sw  ∈ [150, 200]    - wing area (ft²)
        Wfw ∈ [220, 300]    - weight of fuel in wing (lb)
        A   ∈ [6, 10]       - aspect ratio
        Λ   ∈ [-10, 10]     - quarter-chord sweep (deg)
        q   ∈ [16, 45]      - dynamic pressure at cruise (lb/ft²)
        λ   ∈ [0.5, 1]      - taper ratio
        tc  ∈ [0.08, 0.18]  - thickness to chord ratio
        Nz  ∈ [2.5, 6]      - ultimate load factor
        Wdg ∈ [1700, 2500]  - flight design gross weight (lb)
        Wp  ∈ [0.025, 0.08] - paint weight (lb/ft²)
    """

    def __post_init__(self):
        self.name = "WingWeight10D"
        self.dim = 10
        self.bounds = [
            (150.0, 200.0),    # Sw
            (220.0, 300.0),    # Wfw
            (6.0, 10.0),       # A
            (-10.0, 10.0),     # Lambda (sweep)
            (16.0, 45.0),      # q
            (0.5, 1.0),        # lambda (taper)
            (0.08, 0.18),      # tc
            (2.5, 6.0),        # Nz
            (1700.0, 2500.0),  # Wdg
            (0.025, 0.08)      # Wp
        ]
        self.optimal_value = None  # Engineering model
        self.optimal_location = None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)

        Sw = X[:, 0]
        Wfw = X[:, 1]
        A = X[:, 2]
        Lambda = X[:, 3] * np.pi / 180  # Convert to radians
        q = X[:, 4]
        lam = X[:, 5]  # taper ratio
        tc = X[:, 6]
        Nz = X[:, 7]
        Wdg = X[:, 8]
        Wp = X[:, 9]

        term1 = 0.036 * Sw ** 0.758 * Wfw ** 0.0035
        term2 = (A / np.cos(Lambda) ** 2) ** 0.6
        term3 = q ** 0.006 * lam ** 0.04
        term4 = (100 * tc / np.cos(Lambda)) ** (-0.3)
        term5 = (Nz * Wdg) ** 0.49

        return term1 * term2 * term3 * term4 * term5 + Sw * Wp


# =============================================================================
# BENCHMARK REGISTRY
# =============================================================================

BENCHMARK_REGISTRY: Dict[str, Type[BenchmarkFunction]] = {
    # Low dimensional
    "forrester": Forrester1D,
    "branin": Branin2D,
    "sixhump": SixHumpCamel2D,
    "goldstein": GoldsteinPrice2D,
    "hartmann3": Hartmann3D,
    "ishigami": Ishigami3D,
    # Medium dimensional
    "hartmann6": Hartmann6D,
    "borehole": Borehole8D,
    "wingweight": WingWeight10D,
}

# Grouped by difficulty / dimensionality
BENCHMARKS_LOW_DIM = ["forrester", "branin", "sixhump", "goldstein", "hartmann3", "ishigami"]
BENCHMARKS_MEDIUM_DIM = ["hartmann6", "borehole", "wingweight"]


def get_benchmark(name: str, **kwargs) -> BenchmarkFunction:
    """
    Factory function to create benchmark instances by name.

    Args:
        name: Benchmark identifier (case-insensitive)
        **kwargs: Additional arguments passed to benchmark constructor

    Returns:
        BenchmarkFunction instance

    Example:
        >>> bench = get_benchmark("forrester")
        >>> bench = get_benchmark("ishigami", a=7, b=0.05)
    """
    name_lower = name.lower()
    if name_lower not in BENCHMARK_REGISTRY:
        available = ", ".join(BENCHMARK_REGISTRY.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")

    return BENCHMARK_REGISTRY[name_lower](**kwargs)


def list_benchmarks(include_info: bool = False) -> List[str] | List[Dict]:
    """
    List all available benchmark functions.

    Args:
        include_info: If True, return list of dicts with detailed info

    Returns:
        List of benchmark names or detailed info dicts
    """
    if not include_info:
        return list(BENCHMARK_REGISTRY.keys())

    info = []
    for name, cls in BENCHMARK_REGISTRY.items():
        instance = cls()
        info.append({
            "name": name,
            "full_name": instance.name,
            "dim": instance.dim,
            "bounds": instance.bounds,
            "optimal_value": instance.optimal_value,
        })
    return info


if __name__ == "__main__":
    # Quick test of all benchmarks
    print("Testing all benchmark functions...")

    for name in list_benchmarks():
        bench = get_benchmark(name)

        # Generate random test points
        np.random.seed(42)
        X_unit = np.random.rand(10, bench.dim)
        X = bench.scale_to_bounds(X_unit)
        y = bench(X)

        print(f"  {bench.name:20s} | dim={bench.dim:2d} | y_range=[{y.min():.3f}, {y.max():.3f}]")

    print("\nAll benchmarks passed basic evaluation test!")
