# @author: José Arbelaez
"""
Extended metrics for surrogate model evaluation.

Provides metrics beyond RMSE/MAE that are critical for surrogate modeling:
    - NLPD (Negative Log Predictive Density): Calibration measure
    - Coverage: Confidence interval accuracy
    - Interval Width: Uncertainty "usefulness"
    - R² / Coefficient of determination

These metrics help assess not just prediction accuracy but also
uncertainty quantification quality.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
from scipy import stats


@dataclass
class SurrogateMetrics:
    """
    Container for comprehensive surrogate model evaluation metrics.

    Attributes:
        # Accuracy metrics
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        r2: Coefficient of determination
        max_error: Maximum absolute error

        # Uncertainty metrics (if std available)
        nlpd: Negative Log Predictive Density
        coverage_50: Fraction of test points in 50% CI
        coverage_90: Fraction of test points in 90% CI
        coverage_95: Fraction of test points in 95% CI
        mean_interval_width_95: Average width of 95% CI
        median_interval_width_95: Median width of 95% CI

        # Derived
        calibration_error_95: |coverage_95 - 0.95| (lower is better)
        sharpness: Mean std (lower is better if well-calibrated)
    """
    # Sample info
    n_samples: int = 0

    # Accuracy
    mae: float = np.nan
    rmse: float = np.nan
    r2: float = np.nan
    max_error: float = np.nan

    # Uncertainty (None if no std provided)
    nlpd: Optional[float] = None
    coverage_50: Optional[float] = None
    coverage_90: Optional[float] = None
    coverage_95: Optional[float] = None
    mean_interval_width_95: Optional[float] = None
    median_interval_width_95: Optional[float] = None
    calibration_error_95: Optional[float] = None
    sharpness: Optional[float] = None

    # Raw counts for micro-averaging
    _inside_50: Optional[int] = None
    _inside_90: Optional[int] = None
    _inside_95: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding private fields."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def summary_str(self, include_uncertainty: bool = True) -> str:
        """Human-readable summary string."""
        lines = [
            f"n={self.n_samples}",
            f"MAE={self.mae:.4f}",
            f"RMSE={self.rmse:.4f}",
            f"R²={self.r2:.4f}",
        ]
        if include_uncertainty and self.coverage_95 is not None:
            lines.extend([
                f"Cov95={self.coverage_95:.2%}",
                f"NLPD={self.nlpd:.3f}" if self.nlpd is not None else "NLPD=N/A",
                f"CalErr={self.calibration_error_95:.3f}",
            ])
        return " | ".join(lines)


def compute_surrogate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    std_pred: Optional[np.ndarray] = None,
    z_values: Dict[str, float] = None,
) -> SurrogateMetrics:
    """
    Compute comprehensive metrics for surrogate model evaluation.

    Args:
        y_true: True target values (n_samples,)
        y_pred: Predicted mean values (n_samples,)
        std_pred: Predicted standard deviations (n_samples,) or None
        z_values: Dict of confidence level -> z-score for coverage
                  Default: {50: 0.6745, 90: 1.645, 95: 1.96}

    Returns:
        SurrogateMetrics dataclass with all computed metrics

    Example:
        >>> metrics = compute_surrogate_metrics(y_test, mean_pred, std_pred)
        >>> print(f"RMSE: {metrics.rmse:.4f}, Coverage95: {metrics.coverage_95:.2%}")
    """
    # Default z-values for confidence intervals
    if z_values is None:
        z_values = {50: 0.6745, 90: 1.645, 95: 1.96}

    # Ensure arrays
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)

    # Basic accuracy metrics
    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_error = float(np.max(abs_errors))

    # R² calculation
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = SurrogateMetrics(
        n_samples=n,
        mae=mae,
        rmse=rmse,
        r2=r2,
        max_error=max_error,
    )

    # Uncertainty metrics (if std provided)
    if std_pred is not None:
        std_pred = np.asarray(std_pred).ravel()

        # Clip std to avoid numerical issues
        std_pred = np.clip(std_pred, 1e-10, None)

        # NLPD: Negative Log Predictive Density
        # Assumes Gaussian predictive distribution
        nlpd = _compute_nlpd(y_true, y_pred, std_pred)
        metrics.nlpd = nlpd

        # Coverage at different confidence levels
        for level, z in z_values.items():
            lower = y_pred - z * std_pred
            upper = y_pred + z * std_pred
            inside = (y_true >= lower) & (y_true <= upper)
            coverage = float(np.mean(inside))
            inside_count = int(np.sum(inside))

            if level == 50:
                metrics.coverage_50 = coverage
                metrics._inside_50 = inside_count
            elif level == 90:
                metrics.coverage_90 = coverage
                metrics._inside_90 = inside_count
            elif level == 95:
                metrics.coverage_95 = coverage
                metrics._inside_95 = inside_count

        # Interval width at 95%
        z95 = z_values.get(95, 1.96)
        widths_95 = 2 * z95 * std_pred
        metrics.mean_interval_width_95 = float(np.mean(widths_95))
        metrics.median_interval_width_95 = float(np.median(widths_95))

        # Calibration error (ideal coverage should be 95%)
        if metrics.coverage_95 is not None:
            metrics.calibration_error_95 = float(abs(metrics.coverage_95 - 0.95))

        # Sharpness (mean predicted std)
        metrics.sharpness = float(np.mean(std_pred))

    return metrics


def _compute_nlpd(y_true: np.ndarray, y_pred: np.ndarray,
                  std_pred: np.ndarray) -> float:
    """
    Compute Negative Log Predictive Density.

    NLPD = -1/n * Σ log p(y_i | μ_i, σ_i)

    For Gaussian: NLPD = 1/n * Σ [0.5*log(2πσ²) + (y-μ)²/(2σ²)]

    Lower NLPD is better. Penalizes both:
        - Wrong predictions (high error)
        - Overconfident predictions (low std when error is high)
        - Underconfident predictions (high std when error is low)

    Args:
        y_true: True values
        y_pred: Predicted means
        std_pred: Predicted stds

    Returns:
        NLPD value (float)
    """
    var_pred = std_pred ** 2

    # Gaussian log-likelihood
    # log p(y|μ,σ) = -0.5 * log(2πσ²) - (y-μ)²/(2σ²)
    log_probs = -0.5 * np.log(2 * np.pi * var_pred) - \
                (y_true - y_pred) ** 2 / (2 * var_pred)

    # Negative mean log probability
    nlpd = -float(np.mean(log_probs))

    return nlpd


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    std_pred: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve for uncertainty quantification.

    For a well-calibrated model, the empirical coverage should match
    the expected coverage at each confidence level.

    Args:
        y_true: True target values
        y_pred: Predicted means
        std_pred: Predicted standard deviations
        n_bins: Number of confidence levels to evaluate

    Returns:
        expected_coverage: Array of expected coverages (e.g., [0.1, 0.2, ..., 0.9])
        empirical_coverage: Array of observed coverages

    Example:
        >>> expected, empirical = compute_calibration_curve(y_test, mu, std)
        >>> # Perfect calibration: empirical ≈ expected
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    std_pred = np.asarray(std_pred).ravel()

    # Expected coverage levels
    expected = np.linspace(0.1, 0.9, n_bins)
    empirical = np.zeros(n_bins)

    for i, p in enumerate(expected):
        # Compute z-score for this confidence level
        z = stats.norm.ppf(0.5 + p / 2)

        lower = y_pred - z * std_pred
        upper = y_pred + z * std_pred

        inside = (y_true >= lower) & (y_true <= upper)
        empirical[i] = np.mean(inside)

    return expected, empirical


def metrics_to_dict(metrics: SurrogateMetrics, include_private: bool = False) -> Dict[str, Any]:
    """
    Convert SurrogateMetrics to a flat dictionary compatible with existing code.

    This function ensures compatibility between:
        - Benchmark synthetic evaluation
        - Real data evaluation (LODO)
        - Nested tuning results

    Args:
        metrics: SurrogateMetrics instance
        include_private: Whether to include _inside* fields (for micro-averaging)

    Returns:
        Dict with all metrics in a flat structure

    Example:
        >>> metrics = compute_surrogate_metrics(y_true, y_pred, std_pred)
        >>> d = metrics_to_dict(metrics)
        >>> d['mae'], d['rmse'], d['nlpd']  # All accessible
    """
    d = {
        # Core (always present)
        "n_samples": metrics.n_samples,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "r2": metrics.r2,
        "max_error": metrics.max_error,

        # Uncertainty metrics (None if not available)
        "nlpd": metrics.nlpd,
        "coverage_50": metrics.coverage_50,
        "coverage_90": metrics.coverage_90,
        "coverage_95": metrics.coverage_95,
        "mean_interval_width_95": metrics.mean_interval_width_95,
        "median_interval_width_95": metrics.median_interval_width_95,
        "calibration_error_95": metrics.calibration_error_95,
        "sharpness": metrics.sharpness,
    }

    if include_private:
        d["_inside_50"] = metrics._inside_50
        d["_inside_90"] = metrics._inside_90
        d["_inside_95"] = metrics._inside_95

    return d


def aggregate_metrics(
    metrics_list: list[SurrogateMetrics],
    method: str = "macro"
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple folds/datasets.

    Args:
        metrics_list: List of SurrogateMetrics from each fold
        method: "macro" (average of metrics) or "micro" (sample-weighted)

    Returns:
        Dict with mean and std for each metric

    Example:
        >>> fold_metrics = [compute_surrogate_metrics(...) for fold in folds]
        >>> summary = aggregate_metrics(fold_metrics)
        >>> print(f"MAE: {summary['mae']['mean']:.4f} ± {summary['mae']['std']:.4f}")
    """
    if not metrics_list:
        return {}

    result = {}

    # Metrics to aggregate
    metric_names = ['mae', 'rmse', 'r2', 'max_error', 'nlpd',
                    'coverage_50', 'coverage_90', 'coverage_95',
                    'mean_interval_width_95', 'calibration_error_95', 'sharpness']

    for name in metric_names:
        values = [getattr(m, name) for m in metrics_list if getattr(m, name) is not None]

        if values:
            if method == "macro":
                result[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            else:  # micro
                # Sample-weighted average
                weights = [m.n_samples for m in metrics_list if getattr(m, name) is not None]
                weighted_mean = float(np.average(values, weights=weights))
                result[name] = {
                    "mean": weighted_mean,
                    "std": float(np.std(values)),  # Still report std
                }

    # Total samples
    result["n_total"] = sum(m.n_samples for m in metrics_list)
    result["n_folds"] = len(metrics_list)

    return result


if __name__ == "__main__":
    # Demo of metrics computation
    print("=" * 60)
    print("SURROGATE METRICS DEMO")
    print("=" * 60)

    np.random.seed(42)
    n = 100

    # Simulate predictions
    y_true = np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 0.1, n)
    y_pred = np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 0.05, n)
    std_pred = np.abs(np.random.normal(0.15, 0.05, n))  # Estimated uncertainty

    # Compute metrics
    metrics = compute_surrogate_metrics(y_true, y_pred, std_pred)

    print("\nComputed metrics:")
    print(f"  Accuracy:")
    print(f"    MAE      = {metrics.mae:.4f}")
    print(f"    RMSE     = {metrics.rmse:.4f}")
    print(f"    R²       = {metrics.r2:.4f}")
    print(f"    Max Err  = {metrics.max_error:.4f}")

    print(f"\n  Uncertainty:")
    print(f"    NLPD     = {metrics.nlpd:.4f}")
    print(f"    Cov 50%  = {metrics.coverage_50:.2%}")
    print(f"    Cov 90%  = {metrics.coverage_90:.2%}")
    print(f"    Cov 95%  = {metrics.coverage_95:.2%}")
    print(f"    Cal Err  = {metrics.calibration_error_95:.4f}")
    print(f"    Width 95 = {metrics.mean_interval_width_95:.4f}")
    print(f"    Sharpness= {metrics.sharpness:.4f}")

    # Calibration curve
    print("\nCalibration curve:")
    expected, empirical = compute_calibration_curve(y_true, y_pred, std_pred)
    for exp, emp in zip(expected, empirical):
        bar = "█" * int(emp * 20)
        print(f"  {exp:.0%} -> {emp:.0%} {bar}")
