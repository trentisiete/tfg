# @author: José Arbelaez
"""
Benchmarks module for synthetic function evaluation.

Provides:
    - Classical benchmark functions (Forrester, Branin, Hartmann, etc.)
    - Sampling strategies (Sobol, LHS, Grid)
    - Noise injection utilities
    - Dataset generation for surrogate model evaluation
"""

from .functions import (
    BenchmarkFunction,
    Forrester1D,
    Branin2D,
    SixHumpCamel2D,
    GoldsteinPrice2D,
    Hartmann3D,
    Hartmann6D,
    Borehole8D,
    WingWeight10D,
    Ishigami3D,
    BENCHMARK_REGISTRY,
    BENCHMARKS_LOW_DIM,
    BENCHMARKS_MEDIUM_DIM,
    get_benchmark,
    list_benchmarks,
)

from .sampling import (
    SamplingStrategy,
    SobolSampler,
    LatinHypercubeSampler,
    GridSampler,
    RandomSampler,
    get_sampler,
)

from .noise import (
    NoiseInjector,
    GaussianNoise,
    HeteroscedasticNoise,
    NoNoise,
    get_noise_injector,
)

from .dataset_generator import (
    SyntheticDataset,
    generate_benchmark_dataset,
    generate_multi_benchmark_suite,
)

__all__ = [
    # Functions
    "BenchmarkFunction",
    "Forrester1D",
    "Branin2D", 
    "SixHumpCamel2D",
    "GoldsteinPrice2D",
    "Hartmann3D",
    "Hartmann6D",
    "Borehole8D",
    "WingWeight10D",
    "Ishigami3D",
    "BENCHMARK_REGISTRY",
    "BENCHMARKS_LOW_DIM",
    "BENCHMARKS_MEDIUM_DIM",
    "get_benchmark",
    "list_benchmarks",
    # Sampling
    "SamplingStrategy",
    "SobolSampler",
    "LatinHypercubeSampler",
    "GridSampler",
    "RandomSampler",
    "get_sampler",
    # Noise
    "NoiseInjector",
    "GaussianNoise",
    "HeteroscedasticNoise",
    "NoNoise",
    "get_noise_injector",
    # Dataset Generation
    "SyntheticDataset",
    "generate_benchmark_dataset",
    "generate_multi_benchmark_suite",
]
