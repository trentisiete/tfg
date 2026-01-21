# @author: José Arbelaez & Copilot
"""
Gaussian Process Visualization Module.

Generates professional visualizations showing:
- GP mean predictions vs true function
- Uncertainty bands (±1σ, ±2σ)
- Training points
- Comparison across different kernels

Essential for understanding GP behavior and uncertainty quantification.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch generation

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

# Sklearn GP imports
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

# Local imports
from ..benchmarks import (
    get_benchmark, 
    generate_benchmark_dataset,
    BenchmarkFunction,
    SyntheticDataset,
    BENCHMARK_REGISTRY,
)
from ..benchmarks.noise import get_noise_injector


# =============================================================================
# CONFIGURATION
# =============================================================================

# Professional color palette
COLORS = {
    'true_function': '#1f77b4',      # Blue
    'gp_mean': '#ff7f0e',            # Orange  
    'gp_band_1sigma': '#ffbb78',     # Light orange
    'gp_band_2sigma': '#ffeedd',     # Very light orange
    'train_points': '#2ca02c',       # Green
    'test_points': '#d62728',        # Red
    'optimal': '#9467bd',            # Purple
}

# Kernel configurations for comparison
GP_KERNELS = {
    'Matern32': Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-5),
    'Matern52': Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5),
    'RBF': RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5),
}

# Figure style
plt.style.use('seaborn-v0_8-whitegrid')


@dataclass
class GPModel:
    """Container for trained GP with metadata."""
    name: str
    kernel_name: str
    gpr: GaussianProcessRegressor
    scaler: StandardScaler
    X_train: np.ndarray
    y_train: np.ndarray
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty."""
        X_scaled = self.scaler.transform(X)
        mean, std = self.gpr.predict(X_scaled, return_std=True)
        return mean, std


# =============================================================================
# GP TRAINING UTILITIES
# =============================================================================

def train_gp(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    kernel_name: str = 'Matern52',
    n_restarts: int = 5,
) -> GPModel:
    """
    Train a GP model with specified kernel.
    
    Args:
        X_train: Training features
        y_train: Training targets
        kernel_name: One of 'Matern32', 'Matern52', 'RBF'
        n_restarts: Number of optimizer restarts
        
    Returns:
        Trained GPModel instance
    """
    if kernel_name not in GP_KERNELS:
        raise ValueError(f"Unknown kernel: {kernel_name}. Available: {list(GP_KERNELS.keys())}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Train GP
    kernel = GP_KERNELS[kernel_name]
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr.fit(X_scaled, y_train.ravel())
    
    return GPModel(
        name=f"GP_{kernel_name}",
        kernel_name=kernel_name,
        gpr=gpr,
        scaler=scaler,
        X_train=X_train,
        y_train=y_train,
    )


def train_all_kernels(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    n_restarts: int = 5,
) -> Dict[str, GPModel]:
    """Train GPs with all available kernels."""
    models = {}
    for kernel_name in GP_KERNELS.keys():
        models[kernel_name] = train_gp(X_train, y_train, kernel_name, n_restarts)
    return models


# =============================================================================
# 1D VISUALIZATION
# =============================================================================

def plot_gp_1d(
    benchmark: BenchmarkFunction,
    gp_model: GPModel,
    n_grid: int = 500,
    show_training: bool = True,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot GP fit for a 1D function.
    
    Classic visualization showing:
    - True function
    - GP mean prediction
    - ±1σ and ±2σ uncertainty bands
    - Training points
    
    Args:
        benchmark: 1D benchmark function
        gp_model: Trained GP model
        n_grid: Number of points for plotting
        show_training: Whether to show training points
        ax: Matplotlib axes (created if None)
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if benchmark.dim != 1:
        raise ValueError(f"plot_gp_1d requires 1D function, got dim={benchmark.dim}")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create dense grid
    bounds = benchmark.bounds[0]
    X_grid = np.linspace(bounds[0], bounds[1], n_grid).reshape(-1, 1)
    
    # True function values
    y_true = benchmark(X_grid)
    
    # GP predictions
    y_mean, y_std = gp_model.predict(X_grid)
    
    # Plot uncertainty bands (2σ first, then 1σ on top)
    ax.fill_between(
        X_grid.ravel(), 
        y_mean - 2*y_std, 
        y_mean + 2*y_std,
        alpha=0.3, 
        color=COLORS['gp_band_2sigma'],
        label='±2σ',
        edgecolor='none',
    )
    ax.fill_between(
        X_grid.ravel(), 
        y_mean - y_std, 
        y_mean + y_std,
        alpha=0.5, 
        color=COLORS['gp_band_1sigma'],
        label='±1σ',
        edgecolor='none',
    )
    
    # Plot true function
    ax.plot(
        X_grid, y_true, 
        color=COLORS['true_function'], 
        linewidth=2.5, 
        label='True function',
        linestyle='--',
    )
    
    # Plot GP mean
    ax.plot(
        X_grid, y_mean, 
        color=COLORS['gp_mean'], 
        linewidth=2, 
        label=f'GP mean ({gp_model.kernel_name})',
    )
    
    # Plot training points
    if show_training:
        ax.scatter(
            gp_model.X_train, gp_model.y_train,
            color=COLORS['train_points'],
            s=80, 
            marker='o',
            edgecolors='white',
            linewidths=1.5,
            label=f'Training points (n={len(gp_model.X_train)})',
            zorder=10,
        )
    
    # Mark optimal point if known
    if benchmark.optimal_location is not None:
        ax.axvline(
            x=benchmark.optimal_location[0],
            color=COLORS['optimal'],
            linestyle=':',
            alpha=0.7,
            label=f'Optimal (x={benchmark.optimal_location[0]:.3f})',
        )
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(bounds)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(
            f'{benchmark.name}: GP Regression with {gp_model.kernel_name} Kernel',
            fontsize=14, fontweight='bold'
        )
    
    return ax


def plot_gp_1d_comparison(
    benchmark: BenchmarkFunction,
    gp_models: Dict[str, GPModel],
    n_grid: int = 500,
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compare multiple GP kernels on a 1D function side-by-side.
    
    Args:
        benchmark: 1D benchmark function
        gp_models: Dict of {kernel_name: trained_model}
        n_grid: Points for plotting
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    n_models = len(gp_models)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (kernel_name, model) in zip(axes, gp_models.items()):
        plot_gp_1d(benchmark, model, n_grid=n_grid, ax=ax, 
                   title=f'{kernel_name} Kernel')
        if ax != axes[0]:
            ax.set_ylabel('')
    
    fig.suptitle(
        f'{benchmark.name}: Kernel Comparison',
        fontsize=16, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    return fig


# =============================================================================
# 2D VISUALIZATION
# =============================================================================

def plot_gp_2d_contour(
    benchmark: BenchmarkFunction,
    gp_model: GPModel,
    n_grid: int = 100,
    show_training: bool = True,
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot GP fit for 2D function with contour plots.
    
    Creates three panels:
    1. True function contour
    2. GP mean prediction contour
    3. GP uncertainty (std) contour
    
    Args:
        benchmark: 2D benchmark function
        gp_model: Trained GP model
        n_grid: Grid resolution per dimension
        show_training: Show training points
        figsize: Figure size
        save_path: Path to save
        
    Returns:
        Figure
    """
    if benchmark.dim != 2:
        raise ValueError(f"plot_gp_2d_contour requires 2D function, got dim={benchmark.dim}")
    
    # Create grid
    b0, b1 = benchmark.bounds[0], benchmark.bounds[1]
    x0 = np.linspace(b0[0], b0[1], n_grid)
    x1 = np.linspace(b1[0], b1[1], n_grid)
    X0, X1 = np.meshgrid(x0, x1)
    X_grid = np.column_stack([X0.ravel(), X1.ravel()])
    
    # True function
    y_true = benchmark(X_grid).reshape(n_grid, n_grid)
    
    # GP predictions
    y_mean, y_std = gp_model.predict(X_grid)
    y_mean = y_mean.reshape(n_grid, n_grid)
    y_std = y_std.reshape(n_grid, n_grid)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Common colormap range for function values
    vmin = min(y_true.min(), y_mean.min())
    vmax = max(y_true.max(), y_mean.max())
    
    # Panel 1: True function
    cf1 = axes[0].contourf(X0, X1, y_true, levels=30, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].contour(X0, X1, y_true, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    axes[0].set_title('True Function', fontsize=12, fontweight='bold')
    fig.colorbar(cf1, ax=axes[0], label='f(x)')
    
    # Panel 2: GP Mean
    cf2 = axes[1].contourf(X0, X1, y_mean, levels=30, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].contour(X0, X1, y_mean, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    axes[1].set_title(f'GP Mean ({gp_model.kernel_name})', fontsize=12, fontweight='bold')
    fig.colorbar(cf2, ax=axes[1], label='μ(x)')
    
    # Panel 3: GP Uncertainty
    cf3 = axes[2].contourf(X0, X1, y_std, levels=30, cmap='Oranges')
    axes[2].contour(X0, X1, y_std, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    axes[2].set_title('GP Uncertainty (σ)', fontsize=12, fontweight='bold')
    fig.colorbar(cf3, ax=axes[2], label='σ(x)')
    
    # Add training points to all panels
    if show_training:
        for ax in axes:
            ax.scatter(
                gp_model.X_train[:, 0], gp_model.X_train[:, 1],
                c='red', s=40, marker='o', edgecolors='white', linewidths=1,
                label=f'Train (n={len(gp_model.X_train)})', zorder=10
            )
            ax.legend(loc='upper right', fontsize=9)
    
    # Mark optimal if known
    if benchmark.optimal_location is not None:
        opt = np.atleast_2d(benchmark.optimal_location)
        for ax in axes:
            ax.scatter(opt[:, 0], opt[:, 1], c='purple', s=100, marker='*',
                      edgecolors='white', linewidths=1, label='Optimal')
    
    for ax in axes:
        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        ax.set_xlim(b0)
        ax.set_ylim(b1)
    
    fig.suptitle(
        f'{benchmark.name}: GP Regression ({gp_model.kernel_name})',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    return fig


def plot_gp_2d_slices(
    benchmark: BenchmarkFunction,
    gp_model: GPModel,
    slice_values: Optional[List[float]] = None,
    n_grid: int = 200,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot 1D slices through a 2D GP.
    
    Shows GP behavior along each dimension while fixing the other.
    
    Args:
        benchmark: 2D benchmark function
        gp_model: Trained GP
        slice_values: Values at which to take slices (default: midpoint of bounds)
        n_grid: Points per slice
        figsize: Figure size
        save_path: Path to save
        
    Returns:
        Figure
    """
    if benchmark.dim != 2:
        raise ValueError(f"plot_gp_2d_slices requires 2D function, got dim={benchmark.dim}")
    
    b0, b1 = benchmark.bounds[0], benchmark.bounds[1]
    
    # Default slice values: midpoint
    if slice_values is None:
        slice_values = [
            (b1[0] + b1[1]) / 2,  # Fix x2 at midpoint
            (b0[0] + b0[1]) / 2,  # Fix x1 at midpoint
        ]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Slice 1: Vary x1, fix x2
    x1_range = np.linspace(b0[0], b0[1], n_grid)
    X_slice1 = np.column_stack([x1_range, np.full(n_grid, slice_values[0])])
    y_true1 = benchmark(X_slice1)
    y_mean1, y_std1 = gp_model.predict(X_slice1)
    
    axes[0].fill_between(x1_range, y_mean1 - 2*y_std1, y_mean1 + 2*y_std1,
                         alpha=0.3, color=COLORS['gp_band_2sigma'], label='±2σ')
    axes[0].fill_between(x1_range, y_mean1 - y_std1, y_mean1 + y_std1,
                         alpha=0.5, color=COLORS['gp_band_1sigma'], label='±1σ')
    axes[0].plot(x1_range, y_true1, '--', color=COLORS['true_function'], 
                 linewidth=2, label='True')
    axes[0].plot(x1_range, y_mean1, color=COLORS['gp_mean'], 
                 linewidth=2, label='GP mean')
    
    # Add training points near this slice
    mask1 = np.abs(gp_model.X_train[:, 1] - slice_values[0]) < (b1[1] - b1[0]) * 0.1
    if mask1.any():
        axes[0].scatter(gp_model.X_train[mask1, 0], gp_model.y_train[mask1],
                       color=COLORS['train_points'], s=60, marker='o',
                       edgecolors='white', label='Nearby train', zorder=10)
    
    axes[0].set_xlabel('$x_1$', fontsize=11)
    axes[0].set_ylabel('f(x)', fontsize=11)
    axes[0].set_title(f'Slice: $x_2$ = {slice_values[0]:.2f}', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(b0)
    
    # Slice 2: Vary x2, fix x1
    x2_range = np.linspace(b1[0], b1[1], n_grid)
    X_slice2 = np.column_stack([np.full(n_grid, slice_values[1]), x2_range])
    y_true2 = benchmark(X_slice2)
    y_mean2, y_std2 = gp_model.predict(X_slice2)
    
    axes[1].fill_between(x2_range, y_mean2 - 2*y_std2, y_mean2 + 2*y_std2,
                         alpha=0.3, color=COLORS['gp_band_2sigma'], label='±2σ')
    axes[1].fill_between(x2_range, y_mean2 - y_std2, y_mean2 + y_std2,
                         alpha=0.5, color=COLORS['gp_band_1sigma'], label='±1σ')
    axes[1].plot(x2_range, y_true2, '--', color=COLORS['true_function'],
                 linewidth=2, label='True')
    axes[1].plot(x2_range, y_mean2, color=COLORS['gp_mean'],
                 linewidth=2, label='GP mean')
    
    mask2 = np.abs(gp_model.X_train[:, 0] - slice_values[1]) < (b0[1] - b0[0]) * 0.1
    if mask2.any():
        axes[1].scatter(gp_model.X_train[mask2, 1], gp_model.y_train[mask2],
                       color=COLORS['train_points'], s=60, marker='o',
                       edgecolors='white', label='Nearby train', zorder=10)
    
    axes[1].set_xlabel('$x_2$', fontsize=11)
    axes[1].set_ylabel('f(x)', fontsize=11)
    axes[1].set_title(f'Slice: $x_1$ = {slice_values[1]:.2f}', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(b1)
    
    fig.suptitle(
        f'{benchmark.name}: GP Slices ({gp_model.kernel_name})',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    return fig


# =============================================================================
# HIGH-DIMENSIONAL VISUALIZATION (3D+)
# =============================================================================

def plot_gp_nd_slices(
    benchmark: BenchmarkFunction,
    gp_model: GPModel,
    n_grid: int = 200,
    n_slices_per_dim: int = 1,
    figsize_per_dim: Tuple[int, int] = (6, 4),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot 1D slices through high-dimensional GP.
    
    For each dimension, creates a slice plot varying that dimension
    while fixing all others at their midpoint.
    
    Args:
        benchmark: nD benchmark function
        gp_model: Trained GP
        n_grid: Points per slice
        n_slices_per_dim: Number of slices per dimension
        figsize_per_dim: Size per subplot
        save_path: Path to save
        
    Returns:
        Figure
    """
    dim = benchmark.dim
    bounds = np.array(benchmark.bounds)
    midpoints = (bounds[:, 0] + bounds[:, 1]) / 2
    
    # Create figure grid
    n_cols = min(3, dim)
    n_rows = (dim + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_dim[0] * n_cols, figsize_per_dim[1] * n_rows)
    )
    axes = np.atleast_2d(axes)
    
    for d in range(dim):
        row, col = d // n_cols, d % n_cols
        ax = axes[row, col]
        
        # Vary dimension d, fix others at midpoint
        x_range = np.linspace(bounds[d, 0], bounds[d, 1], n_grid)
        X_slice = np.tile(midpoints, (n_grid, 1))
        X_slice[:, d] = x_range
        
        # True function
        y_true = benchmark(X_slice)
        
        # GP predictions
        y_mean, y_std = gp_model.predict(X_slice)
        
        # Plot
        ax.fill_between(x_range, y_mean - 2*y_std, y_mean + 2*y_std,
                       alpha=0.3, color=COLORS['gp_band_2sigma'])
        ax.fill_between(x_range, y_mean - y_std, y_mean + y_std,
                       alpha=0.5, color=COLORS['gp_band_1sigma'])
        ax.plot(x_range, y_true, '--', color=COLORS['true_function'], linewidth=2)
        ax.plot(x_range, y_mean, color=COLORS['gp_mean'], linewidth=2)
        
        # Project training points onto this dimension
        # Show points where all other dims are near midpoint
        dist_to_mid = np.abs(gp_model.X_train - midpoints)
        dist_to_mid[:, d] = 0  # Don't count current dimension
        rel_dist = dist_to_mid / (bounds[:, 1] - bounds[:, 0])
        mask = rel_dist.max(axis=1) < 0.2
        
        if mask.any():
            ax.scatter(gp_model.X_train[mask, d], gp_model.y_train[mask],
                      color=COLORS['train_points'], s=50, marker='o',
                      edgecolors='white', alpha=0.8, zorder=10)
        
        ax.set_xlabel(f'$x_{{{d+1}}}$', fontsize=10)
        ax.set_ylabel('f(x)', fontsize=10)
        ax.set_title(f'Dim {d+1} slice', fontsize=11)
        ax.set_xlim(bounds[d])
    
    # Hide empty subplots
    for d in range(dim, n_rows * n_cols):
        row, col = d // n_cols, d % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle(
        f'{benchmark.name} ({dim}D): GP Slices ({gp_model.kernel_name})',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    try:
        plt.tight_layout()
    except Exception:
        pass  # Ignore tight_layout errors for complex figures
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    return fig


# =============================================================================
# KERNEL COMPARISON DASHBOARD
# =============================================================================

def generate_kernel_comparison_dashboard(
    benchmark: BenchmarkFunction,
    X_train: np.ndarray,
    y_train: np.ndarray,
    noise_type: str = "NoNoise",
    output_dir: Optional[Path] = None,
    n_restarts: int = 5,
) -> Dict[str, plt.Figure]:
    """
    Generate comprehensive kernel comparison dashboard for a benchmark.
    
    Creates separate figures for each kernel showing GP fit quality.
    
    Args:
        benchmark: Benchmark function
        X_train: Training features
        y_train: Training targets
        noise_type: Noise type label
        output_dir: Directory to save figures
        n_restarts: GP optimizer restarts
        
    Returns:
        Dict of {figure_name: Figure}
    """
    # Train all kernels
    print(f"    Training GPs for {benchmark.name}...")
    models = train_all_kernels(X_train, y_train, n_restarts=n_restarts)
    
    figures = {}
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    dim = benchmark.dim
    
    if dim == 1:
        # 1D: Side-by-side comparison
        save_path = output_dir / f"{benchmark.name}_{noise_type}_kernel_comparison.png" if output_dir else None
        fig = plot_gp_1d_comparison(benchmark, models, save_path=save_path)
        figures['kernel_comparison'] = fig
        
        # Individual detailed plots
        for kernel_name, model in models.items():
            save_path = output_dir / f"{benchmark.name}_{noise_type}_{kernel_name}.png" if output_dir else None
            fig_single, ax = plt.subplots(figsize=(10, 6))
            plot_gp_1d(benchmark, model, ax=ax)
            fig_single.tight_layout()
            if save_path:
                fig_single.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig_single)
            figures[f'{kernel_name}_detail'] = fig_single
            
    elif dim == 2:
        # 2D: Contour plots for each kernel
        for kernel_name, model in models.items():
            # Contour plot
            save_path = output_dir / f"{benchmark.name}_{noise_type}_{kernel_name}_contour.png" if output_dir else None
            fig = plot_gp_2d_contour(benchmark, model, save_path=save_path)
            figures[f'{kernel_name}_contour'] = fig
            
            # Slice plot
            save_path = output_dir / f"{benchmark.name}_{noise_type}_{kernel_name}_slices.png" if output_dir else None
            fig = plot_gp_2d_slices(benchmark, model, save_path=save_path)
            figures[f'{kernel_name}_slices'] = fig
    
    else:
        # nD: Slice plots
        for kernel_name, model in models.items():
            save_path = output_dir / f"{benchmark.name}_{noise_type}_{kernel_name}_slices.png" if output_dir else None
            fig = plot_gp_nd_slices(benchmark, model, save_path=save_path)
            figures[f'{kernel_name}_slices'] = fig
    
    return figures


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class GPVisualizationGenerator:
    """
    Generates comprehensive GP visualizations for all benchmarks.
    
    Usage:
        >>> gen = GPVisualizationGenerator(output_dir="outputs/gp_viz")
        >>> gen.generate_all_benchmarks()
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        n_train: int = 50,
        seed: int = 42,
        n_restarts: int = 5,
    ):
        """
        Initialize generator.
        
        Args:
            output_dir: Directory for output figures
            n_train: Number of training points
            seed: Random seed
            n_restarts: GP optimizer restarts
        """
        self.output_dir = Path(output_dir)
        self.n_train = n_train
        self.seed = seed
        self.n_restarts = n_restarts
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_for_benchmark(
        self,
        benchmark_name: str,
        noise_types: List[str] = None,
    ) -> Dict[str, Dict[str, plt.Figure]]:
        """
        Generate visualizations for a single benchmark with multiple noise types.
        
        Args:
            benchmark_name: Benchmark identifier
            noise_types: List of noise types to use
            
        Returns:
            Nested dict {noise_type: {fig_name: Figure}}
        """
        if noise_types is None:
            noise_types = ['NoNoise', 'GaussianNoise']
        
        benchmark = get_benchmark(benchmark_name)
        print(f"\n{'='*60}")
        print(f"Generating GP visualizations for: {benchmark.name} (dim={benchmark.dim})")
        print(f"{'='*60}")
        
        all_figures = {}
        
        for noise_type in noise_types:
            print(f"\n  Noise type: {noise_type}")
            
            # Generate dataset
            noise_map = {
                'NoNoise': 'none',
                'GaussianNoise': 'gaussian',
                'HeteroscedasticNoise': 'heteroscedastic',
            }
            noise_key = noise_map.get(noise_type, 'none')
            
            noise_kwargs = {'sigma': 0.1} if noise_key == 'gaussian' else {}
            
            dataset = generate_benchmark_dataset(
                benchmark=benchmark_name,
                n_train=self.n_train,
                n_test=200,
                sampler='sobol',
                noise=noise_key,
                noise_kwargs=noise_kwargs,
                seed=self.seed,
            )
            
            # Create subdirectory
            bench_dir = self.output_dir / benchmark.name / noise_type
            bench_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate figures
            figures = generate_kernel_comparison_dashboard(
                benchmark=benchmark,
                X_train=dataset.X_train,
                y_train=dataset.y_train,
                noise_type=noise_type,
                output_dir=bench_dir,
                n_restarts=self.n_restarts,
            )
            
            all_figures[noise_type] = figures
            print(f"    Generated {len(figures)} figures")
        
        return all_figures
    
    def generate_all_benchmarks(
        self,
        benchmarks: Optional[List[str]] = None,
        noise_types: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Dict[str, plt.Figure]]]:
        """
        Generate visualizations for all benchmarks.
        
        Args:
            benchmarks: List of benchmark names (default: all)
            noise_types: List of noise types
            
        Returns:
            Nested dict {benchmark: {noise: {fig_name: Figure}}}
        """
        if benchmarks is None:
            benchmarks = list(BENCHMARK_REGISTRY.keys())
        
        if noise_types is None:
            noise_types = ['NoNoise', 'GaussianNoise']
        
        all_results = {}
        
        print(f"\n{'#'*60}")
        print(f"# GP VISUALIZATION GENERATOR")
        print(f"# Benchmarks: {len(benchmarks)}")
        print(f"# Noise types: {noise_types}")
        print(f"# Output: {self.output_dir}")
        print(f"{'#'*60}")
        
        for bench_name in benchmarks:
            all_results[bench_name] = self.generate_for_benchmark(
                bench_name, noise_types
            )
        
        # Generate summary index
        self._generate_index(benchmarks, noise_types)
        
        return all_results
    
    def _generate_index(
        self,
        benchmarks: List[str],
        noise_types: List[str],
    ):
        """Generate markdown index of all figures."""
        index_path = self.output_dir / "GP_VISUALIZATION_INDEX.md"
        
        lines = [
            "# GP Visualization Index",
            "",
            f"Generated: {np.datetime64('now')}",
            f"Training points: {self.n_train}",
            f"Seed: {self.seed}",
            "",
            "## Benchmarks",
            "",
        ]
        
        for bench_name in benchmarks:
            bench = get_benchmark(bench_name)
            lines.append(f"### {bench.name} ({bench.dim}D)")
            lines.append("")
            
            for noise in noise_types:
                bench_dir = f"{bench.name}/{noise}"
                lines.append(f"**{noise}:**")
                lines.append("")
                
                # List expected figures
                if bench.dim == 1:
                    lines.append(f"- [Kernel Comparison]({bench_dir}/{bench.name}_{noise}_kernel_comparison.png)")
                    for k in GP_KERNELS.keys():
                        lines.append(f"- [{k} Detail]({bench_dir}/{bench.name}_{noise}_{k}.png)")
                elif bench.dim == 2:
                    for k in GP_KERNELS.keys():
                        lines.append(f"- [{k} Contour]({bench_dir}/{bench.name}_{noise}_{k}_contour.png)")
                        lines.append(f"- [{k} Slices]({bench_dir}/{bench.name}_{noise}_{k}_slices.png)")
                else:
                    for k in GP_KERNELS.keys():
                        lines.append(f"- [{k} Slices]({bench_dir}/{bench.name}_{noise}_{k}_slices.png)")
                
                lines.append("")
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"\n✓ Index generated: {index_path}")


# =============================================================================
# INTEGRATION WITH BENCHMARK VISUAL REPORTER
# =============================================================================

def add_gp_visualizations_to_report(
    report_dir: Path,
    benchmarks: Optional[List[str]] = None,
    noise_types: Optional[List[str]] = None,
    n_train: int = 50,
    seed: int = 42,
):
    """
    Add GP visualizations to an existing benchmark report.
    
    Args:
        report_dir: Path to existing report directory
        benchmarks: Benchmarks to visualize
        noise_types: Noise types
        n_train: Training points
        seed: Random seed
    """
    gp_viz_dir = report_dir / "figures" / "gp_predictions"
    
    generator = GPVisualizationGenerator(
        output_dir=gp_viz_dir,
        n_train=n_train,
        seed=seed,
    )
    
    generator.generate_all_benchmarks(
        benchmarks=benchmarks,
        noise_types=noise_types,
    )


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate GP prediction visualizations for benchmarks"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/plots/gp_predictions',
        help='Output directory'
    )
    parser.add_argument(
        '--benchmarks', '-b',
        nargs='+',
        default=None,
        help='Specific benchmarks to generate (default: all)'
    )
    parser.add_argument(
        '--noise',
        nargs='+',
        default=['NoNoise', 'GaussianNoise'],
        help='Noise types to include'
    )
    parser.add_argument(
        '--n-train',
        type=int,
        default=50,
        help='Number of training points'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--report-dir',
        type=str,
        default=None,
        help='If provided, add visualizations to existing report'
    )
    
    args = parser.parse_args()
    
    if args.report_dir:
        add_gp_visualizations_to_report(
            report_dir=Path(args.report_dir),
            benchmarks=args.benchmarks,
            noise_types=args.noise,
            n_train=args.n_train,
            seed=args.seed,
        )
    else:
        generator = GPVisualizationGenerator(
            output_dir=args.output,
            n_train=args.n_train,
            seed=args.seed,
        )
        generator.generate_all_benchmarks(
            benchmarks=args.benchmarks,
            noise_types=args.noise,
        )


if __name__ == "__main__":
    main()
