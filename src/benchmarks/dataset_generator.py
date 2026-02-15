# @author: José Arbelaez
"""
Dataset generator for synthetic benchmark evaluation.

Combines benchmark functions, sampling strategies, and noise models
to create reproducible synthetic datasets for surrogate model testing.
"""

# EXPLAIN_AT: When we talk about train and test, What is train and test in the
# context of active learning? In active learning,
# we typically start with a small initial training set and then iteratively
# select new samples to add to the training set based on some acquisition function.
# The "test" set in this context is often a fixed set of points that we use to
# evaluate the performance of our surrogate model at each iteration.


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd

from .functions import BenchmarkFunction, get_benchmark, list_benchmarks
from .sampling import SamplingStrategy, get_sampler
from .noise import NoiseInjector, get_noise_injector


@dataclass
class SyntheticDataset:
    """
    Container for synthetic benchmark datasets.

    Stores all information needed for reproducible experiments:
        - X_train, y_train: Training data
        - X_test, y_test: Test data
        - y_test_clean: Noise-free test values (for reference)
        - Metadata about generation parameters
    """
    # Core data
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_test_clean: np.ndarray  # Noise-free for calculating true error

    # For coverage calculations with known noise
    noise_std_train: Optional[np.ndarray] = None
    noise_std_test: Optional[np.ndarray] = None

    # Metadata
    benchmark_name: str = ""
    dim: int = 0
    n_train: int = 0
    n_test: int = 0
    sampler_name: str = ""
    noise_type: str = ""
    seed: Optional[int] = None
    bounds: List[Tuple[float, float]] = field(default_factory=list)

    # Optional: groups for LODO-style CV (simulated batches)
    # We are not considering this part in the TFG
    groups_train: Optional[np.ndarray] = None
    groups_test: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Export dataset to dictionary (for serialization)."""
        return {
            "X_train": self.X_train.tolist(),
            "y_train": self.y_train.tolist(),
            "X_test": self.X_test.tolist(),
            "y_test": self.y_test.tolist(),
            "y_test_clean": self.y_test_clean.tolist(),
            "metadata": {
                "benchmark": self.benchmark_name,
                "dim": self.dim,
                "n_train": self.n_train,
                "n_test": self.n_test,
                "sampler": self.sampler_name,
                "noise": self.noise_type,
                "seed": self.seed,
                "bounds": self.bounds,
            }
        }

    def get_train_df(self) -> pd.DataFrame:
        """Return training data as DataFrame."""
        cols = [f"x{i}" for i in range(self.dim)]
        df = pd.DataFrame(self.X_train, columns=cols)
        df["y"] = self.y_train
        if self.groups_train is not None:
            df["group"] = self.groups_train
        return df

    def get_test_df(self) -> pd.DataFrame:
        """Return test data as DataFrame."""
        cols = [f"x{i}" for i in range(self.dim)]
        df = pd.DataFrame(self.X_test, columns=cols)
        df["y"] = self.y_test
        df["y_clean"] = self.y_test_clean
        if self.groups_test is not None:
            df["group"] = self.groups_test
        return df

    def __repr__(self) -> str:
        return (f"SyntheticDataset(benchmark='{self.benchmark_name}', "
                f"dim={self.dim}, n_train={self.n_train}, n_test={self.n_test}, "
                f"noise='{self.noise_type}')")


def generate_benchmark_dataset(
    benchmark: Union[str, BenchmarkFunction],
    n_train: int = 50,
    n_test: int = 200,
    sampler: Union[str, SamplingStrategy] = "sobol",
    noise: Union[str, NoiseInjector] = "none",
    noise_kwargs: Optional[Dict] = None,
    n_groups: Optional[int] = None,
    seed: int = 42,
) -> SyntheticDataset:
    """
    Generate a synthetic dataset from a benchmark function.

    This is the main function for creating reproducible benchmark datasets.

    Args:
        benchmark: Benchmark function name or instance
        n_train: Number of training samples
        n_test: Number of test samples
        sampler: Sampling strategy name or instance ("sobol", "lhs", "random", "grid")
        noise: Noise type or instance ("none", "gaussian", "heteroscedastic")
        noise_kwargs: Additional arguments for noise (e.g., {"sigma": 0.1})
        n_groups: If provided, assign synthetic group labels for LODO-style CV
        seed: Random seed for reproducibility

    Returns:
        SyntheticDataset with all generated data

    Example:
        >>> dataset = generate_benchmark_dataset(
        ...     benchmark="forrester",
        ...     n_train=30,
        ...     n_test=100,
        ...     sampler="sobol",
        ...     noise="gaussian",
        ...     noise_kwargs={"sigma": 0.1},
        ...     seed=42
        ... )
        >>> print(dataset)
        >>> X_train, y_train = dataset.X_train, dataset.y_train
    """
    # Resolve benchmark
    if isinstance(benchmark, str):
        bench = get_benchmark(benchmark)
    else:
        bench = benchmark

    # Resolve sampler
    if isinstance(sampler, str):
        train_sampler = get_sampler(sampler, seed=seed)
        test_sampler = get_sampler(sampler, seed=seed + 1000)  # Different seed for test
    else:
        train_sampler = sampler
        test_sampler = get_sampler("sobol", seed=seed + 1000)

    # Resolve noise
    noise_kwargs = noise_kwargs or {}
    if isinstance(noise, str):
        noise_injector = get_noise_injector(noise, seed=seed, **noise_kwargs)
    else:
        noise_injector = noise

    # Generate train samples
    # FUTURE: We have to make sure that this dataset generator works correctly when active learning is implemented.
    X_train = train_sampler.sample_bounds(n_train, bench.bounds)
    y_train_clean = bench(X_train)
    y_train = noise_injector.add_noise(y_train_clean, X_train)

    # Generate test samples (use different seed for independence)
    X_test = test_sampler.sample_bounds(n_test, bench.bounds)
    y_test_clean = bench(X_test)
    y_test = noise_injector.add_noise(y_test_clean, X_test)

    # Get noise std if available
    noise_std_train = noise_injector.get_noise_std(X_train, y_train_clean)
    noise_std_test = noise_injector.get_noise_std(X_test, y_test_clean)

    # Generate synthetic groups if requested (for LODO-style evaluation)
    # This is not being used, create LODO sinthetically is not easy and this is incorrect.
    # Mainly because the groups are generated randomly and not based on spatial clustering.
    # Therefore, we are not considering this part in the TFG.
    groups_train = None
    groups_test = None
    if n_groups is not None and n_groups > 1:
        # Assign groups based on spatial clustering or simple split
        rng = np.random.default_rng(seed)
        groups_train = rng.integers(0, n_groups, size=n_train)
        groups_test = rng.integers(0, n_groups, size=n_test)

    return SyntheticDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_test_clean=y_test_clean,
        noise_std_train=noise_std_train,
        noise_std_test=noise_std_test,
        benchmark_name=bench.name,
        dim=bench.dim,
        n_train=n_train,
        n_test=n_test,
        sampler_name=train_sampler.__class__.__name__,
        noise_type=noise_injector.name,
        seed=seed,
        bounds=bench.bounds,
        groups_train=groups_train,
        groups_test=groups_test,
    )


def generate_multi_benchmark_suite(
    benchmarks: Optional[List[str]] = None,
    n_train: int = 50,
    n_test: int = 200,
    sampler: str = "sobol",
    noise_configs: Optional[List[Dict]] = None,
    n_groups: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, SyntheticDataset]]:
    """
    Generate datasets for multiple benchmarks and noise configurations.

    Creates a comprehensive test suite for surrogate model evaluation.
    Uses the generate_benchmark_dataset function internally.

    Args:
        benchmarks: List of benchmark names (default: all available)
        n_train: Number of training samples per dataset
        n_test: Number of test samples per dataset
        sampler: Sampling strategy name
        noise_configs: List of noise configurations, e.g.:
            [
                {"type": "none"},
                {"type": "gaussian", "sigma": 0.1},
                {"type": "heteroscedastic", "sigma_base": 0.05}
            ]
            Default: [{"type": "none"}, {"type": "gaussian", "sigma": 0.1}]
        n_groups: Number of groups for LODO-style CV
        seed: Base random seed

    Returns:
        Nested dict: {benchmark_name: {noise_type: SyntheticDataset}}

    Example:
        >>> suite = generate_multi_benchmark_suite(
        ...     benchmarks=["forrester", "branin", "hartmann3"],
        ...     n_train=50,
        ...     noise_configs=[
        ...         {"type": "none"},
        ...         {"type": "gaussian", "sigma": 0.1}
        ...     ]
        ... )
        >>> for bench_name, noise_datasets in suite.items():
        ...     for noise_name, dataset in noise_datasets.items():
        ...         print(f"{bench_name} / {noise_name}: {dataset.n_train} train samples")
    """
    # Default benchmarks
    if benchmarks is None:
        benchmarks = list_benchmarks()

    # Default noise configurations
    if noise_configs is None:
        noise_configs = [
            {"type": "none"},
            {"type": "gaussian", "sigma": 0.1},
        ]

    suite = {}
    seed_counter = seed

    for bench_name in benchmarks:
        suite[bench_name] = {}

        for noise_cfg in noise_configs:
            noise_type = noise_cfg.pop("type", "none")
            noise_label = noise_type
            if noise_type == "gaussian" and "sigma" in noise_cfg:
                noise_label = f"gaussian_s{noise_cfg['sigma']}"

            dataset = generate_benchmark_dataset(
                benchmark=bench_name,
                n_train=n_train,
                n_test=n_test,
                sampler=sampler,
                noise=noise_type,
                noise_kwargs=noise_cfg,
                n_groups=n_groups,
                seed=seed_counter,
            )

            suite[bench_name][noise_label] = dataset
            seed_counter += 1

            # Restore type for next iteration
            noise_cfg["type"] = noise_type

    return suite


if __name__ == "__main__":
    # Demo: Generate a single dataset
    print("=" * 60)
    print("SINGLE DATASET GENERATION")
    print("=" * 60)

    dataset = generate_benchmark_dataset(
        benchmark="forrester",
        n_train=30,
        n_test=100,
        sampler="sobol",
        noise="gaussian",
        noise_kwargs={"sigma": 0.1},
        seed=42
    )

    print(f"\nGenerated: {dataset}")
    print(f"X_train shape: {dataset.X_train.shape}")
    print(f"y_train range: [{dataset.y_train.min():.3f}, {dataset.y_train.max():.3f}]")

    # Demo: Generate multi-benchmark suite
    print("\n" + "=" * 60)
    print("MULTI-BENCHMARK SUITE")
    print("=" * 60)

    suite = generate_multi_benchmark_suite(
        benchmarks=["forrester", "branin", "hartmann3"],
        n_train=50,
        n_test=100,
        noise_configs=[
            {"type": "none"},
            {"type": "gaussian", "sigma": 0.1},
            {"type": "gaussian", "sigma": 0.3},
        ],
        seed=42
    )

    print(f"\nGenerated suite with {len(suite)} benchmarks:")
    for bench_name, noise_datasets in suite.items():
        print(f"\n  {bench_name}:")
        for noise_name, ds in noise_datasets.items():
            print(f"    {noise_name:20s} -> train={ds.n_train}, test={ds.n_test}")
