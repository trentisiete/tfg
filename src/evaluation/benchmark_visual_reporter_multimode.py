# @author: José Arbelaez
"""
benchmark_visual_reporter_multimode.py

Extended Benchmark Visual Evaluation Pack for multi-mode benchmark results.
Generates comprehensive figures, tables, and reports considering:
- CV modes: simple vs nested
- Samplers: Sobol vs LHS
- Training sizes: multiple n_train values
- GP predictions with uncertainty visualization

This module EXTENDS (does NOT replace) benchmark_visual_reporter.py.

Usage:
    python -m src.evaluation.benchmark_visual_reporter_multimode --session comprehensive_20260121_...
    
    Or programmatically:
        from src.evaluation.benchmark_visual_reporter_multimode import generate_benchmark_report_multimode
        generate_benchmark_report_multimode("comprehensive_20260121_...")

Requirements:
    The input JSON must contain data for BOTH samplers (sobol, lhs) and BOTH cv_modes (simple, tuning/nested).
    If requirements are not met, the script will abort with a clear error message.
    
OUTPUT FORMAT:
    - ONLY PNG files (no PDF generation for performance)
"""

from __future__ import annotations

import ast
import json
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter, defaultdict
from itertools import combinations, product
import traceback

# Use non-interactive backend for batch generation
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde

# Project imports
from src.utils.paths import LOGS_DIR, PLOTS_DIR

# Import from existing module for reuse
from src.evaluation.benchmark_visual_reporter import (
    MODEL_FAMILY_PALETTE,
    GP_KERNEL_PALETTE,
    NOISE_PALETTE,
    MODEL_VARIANT_PALETTE,
    BenchmarkGlobalPlotter,
)

# =============================================================================
# PUBLICATION-READY STYLE CONFIGURATION
# =============================================================================

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'font.family': 'sans-serif',
})

# Extended color palettes
SAMPLER_PALETTE = {
    "sobol": "#e74c3c",   # Red
    "lhs": "#3498db",     # Blue
}

CV_MODE_PALETTE = {
    "simple": "#2ecc71",   # Green
    "tuning": "#9b59b6",   # Purple
    "nested": "#9b59b6",   # Purple (alias)
}

NTRAIN_CMAP = plt.cm.viridis

# Model family colors
MODEL_FAMILY_COLORS = {
    "Dummy": "#95a5a6",
    "Ridge": "#3498db",
    "PLS": "#9b59b6",
    "GP": "#2ecc71",
}


# =============================================================================
# HELPER: SAVE FIGURE (ONLY PNG)
# =============================================================================

def save_figure(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    """
    Save figure as PNG only (no PDF for performance).
    
    Args:
        fig: Matplotlib figure
        path: Output path (will ensure .png extension)
        dpi: Resolution
    """
    path = Path(path)
    if path.suffix.lower() != '.png':
        path = path.with_suffix('.png')
    
    fig.savefig(path, bbox_inches='tight', dpi=dpi, facecolor='white')
    plt.close(fig)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MultiModeRequirements:
    """Requirements verification result."""
    has_sobol: bool = False
    has_lhs: bool = False
    has_simple: bool = False
    has_tuning: bool = False
    samplers_found: List[str] = field(default_factory=list)
    cv_modes_found: List[str] = field(default_factory=list)
    n_train_values: List[int] = field(default_factory=list)
    noise_types: List[str] = field(default_factory=list)
    benchmarks: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        # Valid if we have at least one sampler and one cv_mode
        has_any_sampler = self.has_sobol or self.has_lhs or len(self.samplers_found) > 0
        has_any_cv_mode = self.has_simple or self.has_tuning or len(self.cv_modes_found) > 0
        return has_any_sampler and has_any_cv_mode
    
    def get_error_message(self) -> str:
        if self.is_valid:
            return ""
        missing = []
        if not self.samplers_found:
            missing.append("at least one sampler (sobol, lhs, random, etc.)")
        if not self.cv_modes_found:
            missing.append("at least one cv_mode (simple, tuning, nested)")
        return f"No se puede ejecutar benchmark_visual_reporter_multimode: faltan {{{', '.join(missing)}}}"


@dataclass
class DataQualityReportMultiMode:
    """Extended data quality report for multi-mode evaluation."""
    total_records: int
    unique_configs: int
    samplers: List[str]
    cv_modes: List[str]
    n_train_values: List[int]
    benchmarks: List[str]
    noise_types: List[str]
    models: List[str]
    missing_fields: Dict[str, int]
    completeness_by_factor: Dict[str, float]
    recommendations: List[str]


@dataclass
class BenchmarkInventoryMultiMode:
    """Extended inventory for multi-mode benchmark session."""
    session_name: str
    timestamp: str
    total_time_s: float
    samplers: List[str]
    cv_modes: List[str]
    n_train_values: List[int]
    benchmarks: List[str]
    noise_types: List[str]
    models: List[str]
    n_results: int


# =============================================================================
# REQUIREMENTS VERIFIER
# =============================================================================

class MultiModeReportVerifier:
    """
    Verifies that the benchmark results contain all required factors
    for multi-mode analysis (sobol+lhs, simple+nested).
    """
    
    REQUIRED_SAMPLERS = {"sobol", "lhs"}
    REQUIRED_CV_MODES = {"simple", "tuning"}  # tuning = nested
    
    def __init__(self, results_data: Dict, strict: bool = True):
        """
        Args:
            results_data: Loaded JSON data (comprehensive_results.json)
            strict: If True, abort if requirements not met
        """
        self.data = results_data
        self.strict = strict
        self.requirements = MultiModeRequirements()
        
    def verify(self) -> MultiModeRequirements:
        """
        Verify that all required factors are present.
        
        Returns:
            MultiModeRequirements with verification results
            
        Raises:
            ValueError if strict=True and requirements not met
        """
        # Extract metadata
        metadata = self.data.get("metadata", {})
        results = self.data.get("results", {})
        
        # Check samplers
        samplers = set(metadata.get("samplers", []))
        if not samplers and results:
            samplers = set(results.keys())
        
        self.requirements.samplers_found = sorted(samplers)
        self.requirements.has_sobol = "sobol" in samplers
        self.requirements.has_lhs = "lhs" in samplers
        
        # Check cv_modes
        cv_modes = set()
        if metadata.get("cv_mode") == "both":
            cv_modes = {"simple", "tuning"}
        elif metadata.get("cv_mode"):
            cv_modes = {metadata["cv_mode"]}
        else:
            # Scan results to find cv_modes
            for sampler_data in results.values():
                if isinstance(sampler_data, dict):
                    for ntrain_data in sampler_data.values():
                        if isinstance(ntrain_data, dict):
                            for bench_data in ntrain_data.values():
                                if isinstance(bench_data, dict):
                                    for noise_data in bench_data.values():
                                        if isinstance(noise_data, dict):
                                            cv_modes.update(noise_data.keys())
        
        self.requirements.cv_modes_found = sorted(cv_modes)
        self.requirements.has_simple = "simple" in cv_modes
        self.requirements.has_tuning = "tuning" in cv_modes or "nested" in cv_modes
        
        # Extract other factors
        self.requirements.n_train_values = sorted(metadata.get("n_train_list", []))
        self.requirements.noise_types = [n.get("type", "unknown") for n in metadata.get("noise_configs", [])]
        self.requirements.benchmarks = metadata.get("benchmarks", [])
        self.requirements.models = metadata.get("models", [])
        
        # Build missing list only if no data at all
        if not self.requirements.samplers_found:
            self.requirements.missing.append("at least one sampler")
        if not self.requirements.cv_modes_found:
            self.requirements.missing.append("at least one cv_mode")
        
        # Strict mode: raise error if invalid
        if self.strict and not self.requirements.is_valid:
            raise ValueError(self.requirements.get_error_message())
        
        return self.requirements


# =============================================================================
# EXTENDED DATA LOADER
# =============================================================================

class MultiModeResultsLoader:
    """
    Loads and normalizes comprehensive benchmark results with multi-mode support.
    
    Handles the nested structure:
        results[sampler][n_train][benchmark][noise][cv_mode][model] = metrics
    """
    
    def __init__(self, results_json_path: Path, strict: bool = True):
        """
        Args:
            results_json_path: Path to comprehensive_results.json
            strict: If True, abort if requirements not met
        """
        self.json_path = Path(results_json_path)
        self.session_name = self.json_path.parent.name
        self.strict = strict
        
        self._raw_data = None
        self._master_df = None
        self._requirements = None
        
    def load_raw(self) -> Dict:
        """Load raw JSON data."""
        if self._raw_data is None:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self._raw_data = json.load(f)
        return self._raw_data
    
    def verify_requirements(self) -> MultiModeRequirements:
        """Verify multi-mode requirements."""
        if self._requirements is None:
            raw = self.load_raw()
            verifier = MultiModeReportVerifier(raw, strict=self.strict)
            self._requirements = verifier.verify()
        return self._requirements
    
    def _parse_kernel_variant(self, kernel_str: str) -> str:
        """Parse GP kernel string to variant name."""
        if not kernel_str:
            return "Unknown"
        
        kernel_lower = kernel_str.lower()
        
        if "matern" in kernel_lower:
            nu_match = re.search(r'nu\s*=\s*([\d.]+)', kernel_str)
            if nu_match:
                nu_val = float(nu_match.group(1))
                if abs(nu_val - 1.5) < 0.1:
                    return "Matern32"
                elif abs(nu_val - 2.5) < 0.1:
                    return "Matern52"
            return "Matern"
        elif "rbf" in kernel_lower:
            return "RBF"
        else:
            return "Custom"
    
    def _extract_model_variant(self, model_name: str, model_params: Dict) -> Tuple[str, str]:
        """Extract model_family and model_variant."""
        # Simplified model name extraction
        name_lower = model_name.lower()
        
        if "dummy" in name_lower:
            return "Dummy", "Dummy"
        elif "ridge" in name_lower:
            return "Ridge", "Ridge"
        elif "pls" in name_lower:
            n_comp = model_params.get("n_components", "?")
            return "PLS", f"PLS_n{n_comp}"
        elif "gp" in name_lower:
            kernel_str = str(model_params.get("kernel", ""))
            kernel_variant = self._parse_kernel_variant(kernel_str)
            return "GP", f"GP_{kernel_variant}"
        else:
            return model_name, model_name
    
    def build_master_table(self) -> pd.DataFrame:
        """
        Build normalized master table from nested results structure.
        
        Returns:
            DataFrame with columns:
                sampler, n_train, benchmark, noise, cv_mode, model, model_family, model_variant,
                mae, rmse, r2, nlpd, coverage_95, calibration_error_95, sharpness, fit_time, ...
        """
        if self._master_df is not None:
            return self._master_df
        
        # Verify requirements first
        self.verify_requirements()
        
        raw = self.load_raw()
        metadata = raw.get("metadata", {})
        results = raw.get("results", {})
        
        records = []
        
        # Navigate nested structure
        for sampler, sampler_data in results.items():
            if not isinstance(sampler_data, dict):
                continue
                
            for n_train_str, ntrain_data in sampler_data.items():
                if not isinstance(ntrain_data, dict):
                    continue
                try:
                    n_train = int(n_train_str)
                except (ValueError, TypeError):
                    continue
                
                for benchmark, bench_data in ntrain_data.items():
                    if not isinstance(bench_data, dict):
                        continue
                    
                    for noise, noise_data in bench_data.items():
                        if not isinstance(noise_data, dict):
                            continue
                        
                        for cv_mode, cv_data in noise_data.items():
                            if not isinstance(cv_data, dict):
                                continue
                            
                            for model_name, model_results in cv_data.items():
                                if not isinstance(model_results, dict):
                                    continue
                                
                                # Skip error entries
                                if "error" in model_results:
                                    continue
                                
                                # Extract model params
                                model_params = model_results.get("model_params", {})
                                if isinstance(model_params, str):
                                    try:
                                        model_params = json.loads(model_params)
                                    except:
                                        model_params = {}
                                
                                family, variant = self._extract_model_variant(model_name, model_params)
                                
                                record = {
                                    "sampler": sampler,
                                    "n_train": n_train,
                                    "benchmark": benchmark,
                                    "noise": noise,
                                    "cv_mode": cv_mode,
                                    "model": model_name,
                                    "model_family": family,
                                    "model_variant": variant,
                                    # Metrics
                                    "mae": model_results.get("mae"),
                                    "rmse": model_results.get("rmse"),
                                    "r2": model_results.get("r2"),
                                    "nlpd": model_results.get("nlpd"),
                                    "coverage_50": model_results.get("coverage_50"),
                                    "coverage_90": model_results.get("coverage_90"),
                                    "coverage_95": model_results.get("coverage_95"),
                                    "calibration_error_95": model_results.get("calibration_error_95"),
                                    "sharpness": model_results.get("sharpness"),
                                    "fit_time": model_results.get("fit_time"),
                                    "predict_time": model_results.get("predict_time"),
                                    # For tuning mode
                                    "mae_std": model_results.get("mae_std"),
                                    "rmse_std": model_results.get("rmse_std"),
                                    "r2_std": model_results.get("r2_std"),
                                    "n_folds": model_results.get("n_folds"),
                                }
                                
                                records.append(record)
        
        self._master_df = pd.DataFrame(records)
        
        # Ensure proper column order
        primary_cols = [
            "sampler", "n_train", "benchmark", "noise", "cv_mode",
            "model", "model_family", "model_variant",
            "mae", "rmse", "r2", "nlpd", "coverage_95",
            "calibration_error_95", "sharpness", "fit_time"
        ]
        other_cols = [c for c in self._master_df.columns if c not in primary_cols]
        ordered_cols = [c for c in primary_cols if c in self._master_df.columns] + other_cols
        self._master_df = self._master_df[ordered_cols]
        
        logging.info(f"Built master table with {len(self._master_df)} records")
        return self._master_df
    
    def get_inventory(self) -> BenchmarkInventoryMultiMode:
        """Get inventory of the multi-mode benchmark session."""
        raw = self.load_raw()
        df = self.build_master_table()
        metadata = raw.get("metadata", {})
        
        return BenchmarkInventoryMultiMode(
            session_name=self.session_name,
            timestamp=metadata.get("timestamp", ""),
            total_time_s=metadata.get("total_time_s", 0),
            samplers=sorted(df["sampler"].unique()),
            cv_modes=sorted(df["cv_mode"].unique()),
            n_train_values=sorted(df["n_train"].unique()),
            benchmarks=sorted(df["benchmark"].unique()),
            noise_types=sorted(df["noise"].unique()),
            models=sorted(df["model_variant"].unique()),
            n_results=len(df),
        )
    
    def export_master_table(self, output_dir: Path) -> Path:
        """Export master table to CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        path = output_dir / "master_table.csv"
        self.build_master_table().to_csv(path, index=False)
        logging.info(f"Exported master table: {path}")
        return path


# =============================================================================
# EXTENDED TABLE GENERATOR
# =============================================================================

class MultiModeTableGenerator:
    """
    Generates tables for multi-mode benchmark analysis.
    """
    
    def __init__(self, master_df: pd.DataFrame, output_dir: Path):
        self.df = master_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_leaderboards_disaggregated(self, 
                                            metrics: List[str] = None,
                                            top_n: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Generate leaderboards disaggregated by sampler, cv_mode, n_train.
        """
        if metrics is None:
            metrics = ["mae", "rmse", "r2"]
        
        leaderboards = {}
        
        for metric in metrics:
            if metric not in self.df.columns:
                continue
            
            ascending = metric != "r2"
            
            # Global leaderboard (aggregated)
            global_lb = (self.df.groupby("model_variant")[metric]
                         .agg(["mean", "std", "count"])
                         .reset_index()
                         .sort_values("mean", ascending=ascending)
                         .head(top_n))
            global_lb["rank"] = range(1, len(global_lb) + 1)
            global_lb.to_csv(self.output_dir / f"leaderboard_global_{metric}.csv", index=False)
            leaderboards[f"global_{metric}"] = global_lb
            
            # By sampler
            for sampler in self.df["sampler"].unique():
                sampler_df = self.df[self.df["sampler"] == sampler]
                lb = (sampler_df.groupby("model_variant")[metric]
                      .agg(["mean", "std", "count"])
                      .reset_index()
                      .sort_values("mean", ascending=ascending)
                      .head(top_n))
                lb["rank"] = range(1, len(lb) + 1)
                lb.to_csv(self.output_dir / f"leaderboard_{sampler}_{metric}.csv", index=False)
                leaderboards[f"{sampler}_{metric}"] = lb
            
            # By cv_mode
            for cv_mode in self.df["cv_mode"].unique():
                cv_df = self.df[self.df["cv_mode"] == cv_mode]
                lb = (cv_df.groupby("model_variant")[metric]
                      .agg(["mean", "std", "count"])
                      .reset_index()
                      .sort_values("mean", ascending=ascending)
                      .head(top_n))
                lb["rank"] = range(1, len(lb) + 1)
                lb.to_csv(self.output_dir / f"leaderboard_{cv_mode}_{metric}.csv", index=False)
                leaderboards[f"{cv_mode}_{metric}"] = lb
            
            # By n_train
            for n_train in sorted(self.df["n_train"].unique()):
                ntrain_df = self.df[self.df["n_train"] == n_train]
                lb = (ntrain_df.groupby("model_variant")[metric]
                      .agg(["mean", "std", "count"])
                      .reset_index()
                      .sort_values("mean", ascending=ascending)
                      .head(top_n))
                lb["rank"] = range(1, len(lb) + 1)
                lb.to_csv(self.output_dir / f"leaderboard_ntrain{n_train}_{metric}.csv", index=False)
                leaderboards[f"ntrain{n_train}_{metric}"] = lb
        
        logging.info(f"Generated {len(leaderboards)} disaggregated leaderboards")
        return leaderboards
    
    def generate_stability_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Generate stability analysis tables (std, IQR, CV).
        """
        tables = {}
        
        for metric in ["mae", "rmse", "r2"]:
            if metric not in self.df.columns:
                continue
            
            # Group by model_variant and compute stability metrics
            stability = (self.df.groupby("model_variant")[metric]
                         .agg([
                             "mean", "std", "median",
                             lambda x: x.quantile(0.25),
                             lambda x: x.quantile(0.75),
                             lambda x: x.std() / x.mean() if x.mean() != 0 else np.nan  # CV
                         ])
                         .reset_index())
            stability.columns = ["model_variant", "mean", "std", "median", "p25", "p75", "cv"]
            stability["iqr"] = stability["p75"] - stability["p25"]
            stability = stability.sort_values("cv", ascending=True)
            
            path = self.output_dir / f"stability_{metric}.csv"
            stability.to_csv(path, index=False)
            tables[metric] = stability
        
        # Stability by group (sampler, cv_mode, n_train)
        group_stability = (self.df.groupby(["sampler", "cv_mode", "n_train", "model_variant"])["mae"]
                          .agg(["mean", "std"])
                          .reset_index())
        group_stability.to_csv(self.output_dir / "stability_by_group.csv", index=False)
        tables["by_group"] = group_stability
        
        logging.info(f"Generated {len(tables)} stability tables")
        return tables
    
    def generate_topx_tables(self, x: int = 10, metric: str = "mae") -> Dict[str, pd.DataFrame]:
        """
        Generate Top-X tables for each benchmark.
        """
        ascending = metric != "r2"
        tables = {}
        
        topx_dir = self.output_dir / "topX_by_problem"
        topx_dir.mkdir(exist_ok=True)
        
        for benchmark in self.df["benchmark"].unique():
            bench_df = self.df[self.df["benchmark"] == benchmark]
            
            # For each combination of factors
            for sampler in bench_df["sampler"].unique():
                for cv_mode in bench_df["cv_mode"].unique():
                    for noise in bench_df["noise"].unique():
                        subset = bench_df[
                            (bench_df["sampler"] == sampler) &
                            (bench_df["cv_mode"] == cv_mode) &
                            (bench_df["noise"] == noise)
                        ]
                        
                        if subset.empty:
                            continue
                        
                        # Aggregate by model_variant (across n_train)
                        topx = (subset.groupby("model_variant")[metric]
                                .agg(["mean", "std", "count"])
                                .reset_index()
                                .sort_values("mean", ascending=ascending)
                                .head(x))
                        topx["rank"] = range(1, len(topx) + 1)
                        
                        filename = f"top{x}_{benchmark}_{metric}_{sampler}_{cv_mode}_{noise}.csv"
                        topx.to_csv(topx_dir / filename, index=False)
                        tables[f"{benchmark}_{sampler}_{cv_mode}_{noise}"] = topx
        
        logging.info(f"Generated {len(tables)} Top-{x} tables")
        return tables
    
    def generate_best_by_factor(self, metric: str = "mae") -> pd.DataFrame:
        """
        Generate table of best model per (benchmark, sampler, cv_mode, n_train).
        """
        ascending = metric != "r2"
        
        results = []
        
        for (bench, sampler, cv_mode, n_train), group in self.df.groupby(
            ["benchmark", "sampler", "cv_mode", "n_train"]
        ):
            if group[metric].isna().all():
                continue
            
            sorted_group = group.sort_values(metric, ascending=ascending)
            best = sorted_group.iloc[0]
            
            results.append({
                "benchmark": bench,
                "sampler": sampler,
                "cv_mode": cv_mode,
                "n_train": n_train,
                "best_model": best["model_variant"],
                f"best_{metric}": best[metric],
            })
        
        df_best = pd.DataFrame(results)
        df_best.to_csv(self.output_dir / f"best_model_per_benchmark_by_sampler_cv_ntrain_{metric}.csv", index=False)
        
        return df_best
    
    def generate_all_tables(self) -> Dict[str, Any]:
        """Generate all multi-mode tables."""
        logging.info("Generating all multi-mode tables...")
        
        results = {
            "leaderboards": self.generate_leaderboards_disaggregated(),
            "stability": self.generate_stability_tables(),
            "topx_mae": self.generate_topx_tables(x=10, metric="mae"),
            "topx_rmse": self.generate_topx_tables(x=10, metric="rmse"),
            "best_by_factor_mae": self.generate_best_by_factor("mae"),
            "best_by_factor_rmse": self.generate_best_by_factor("rmse"),
        }
        
        logging.info(f"All tables saved to: {self.output_dir}")
        return results


# =============================================================================
# SAMPLING EFFECTS PLOTTER
# =============================================================================

class SamplingEffectsPlotter:
    """
    Generates plots showing effects of sampling strategy and n_train.
    """
    
    def __init__(self, master_df: pd.DataFrame, output_dir: Path):
        self.df = master_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_model_color(self, model_variant: str) -> str:
        """Get consistent color for model variant."""
        if model_variant in MODEL_VARIANT_PALETTE:
            return MODEL_VARIANT_PALETTE[model_variant]
        if model_variant.startswith("GP"):
            return "#2ecc71"
        elif model_variant.startswith("PLS"):
            return "#9b59b6"
        elif model_variant.startswith("Ridge"):
            return "#3498db"
        return "#95a5a6"
    
    def plot_metric_vs_ntrain(self, metric: str = "mae", 
                              benchmark: str = None,
                              save: bool = True) -> plt.Figure:
        """
        Plot metric vs n_train (learning curves) for each model.
        """
        if benchmark:
            df = self.df[self.df["benchmark"] == benchmark]
            title_suffix = f" - {benchmark}"
        else:
            df = self.df
            title_suffix = " (Global Average)"
        
        if df.empty:
            return None
        
        # Average by (n_train, model_variant, sampler)
        agg = (df.groupby(["n_train", "model_variant", "sampler"])[metric]
               .agg(["mean", "std"])
               .reset_index())
        
        n_samplers = agg["sampler"].nunique()
        fig, axes = plt.subplots(1, n_samplers, figsize=(7 * n_samplers, 6), sharey=True)
        
        if n_samplers == 1:
            axes = [axes]
        
        for ax, sampler in zip(axes, sorted(agg["sampler"].unique())):
            sampler_data = agg[agg["sampler"] == sampler]
            
            for model in sorted(sampler_data["model_variant"].unique()):
                model_data = sampler_data[sampler_data["model_variant"] == model]
                model_data = model_data.sort_values("n_train")
                
                color = self._get_model_color(model)
                ax.plot(model_data["n_train"], model_data["mean"], 
                        'o-', color=color, label=model, linewidth=2, markersize=6)
                
                if model_data["std"].notna().any():
                    ax.fill_between(
                        model_data["n_train"],
                        model_data["mean"] - model_data["std"],
                        model_data["mean"] + model_data["std"],
                        color=color, alpha=0.2
                    )
            
            ax.set_xlabel("n_train", fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f"Sampler: {sampler.upper()}", fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f"{metric.upper()} vs Training Size{title_suffix}", 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            suffix = f"_{benchmark}" if benchmark else "_global"
            path = self.output_dir / f"metric_vs_ntrain_{metric}{suffix}.png"
            save_figure(fig, path)
        
        return fig
    
    def plot_sobol_vs_lhs_comparison(self, metric: str = "mae",
                                      save: bool = True) -> plt.Figure:
        """
        Side-by-side comparison of Sobol vs LHS performance.
        """
        # Average by (sampler, model_variant)
        agg = (self.df.groupby(["sampler", "model_variant"])[metric]
               .agg(["mean", "std"])
               .reset_index())
        
        # Pivot for comparison
        pivot = agg.pivot(index="model_variant", columns="sampler", values="mean")
        
        if pivot.empty or "sobol" not in pivot.columns or "lhs" not in pivot.columns:
            logging.warning("Cannot create Sobol vs LHS comparison - missing data")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Scatter plot (sobol vs lhs)
        ax = axes[0]
        colors = [self._get_model_color(m) for m in pivot.index]
        ax.scatter(pivot["sobol"], pivot["lhs"], c=colors, s=100, alpha=0.8, edgecolors='black')
        
        # Add diagonal (equal performance)
        lims = [min(pivot["sobol"].min(), pivot["lhs"].min()),
                max(pivot["sobol"].max(), pivot["lhs"].max())]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Equal performance')
        
        # Annotate points
        for i, model in enumerate(pivot.index):
            ax.annotate(model, (pivot.loc[model, "sobol"], pivot.loc[model, "lhs"]),
                        fontsize=8, xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel(f"Sobol {metric.upper()}", fontsize=12)
        ax.set_ylabel(f"LHS {metric.upper()}", fontsize=12)
        ax.set_title("Sobol vs LHS Performance", fontsize=14, fontweight='bold')
        ax.legend()
        
        # Right: Bar plot showing difference
        ax = axes[1]
        diff = pivot["lhs"] - pivot["sobol"]  # Positive = Sobol better
        diff = diff.sort_values()
        
        colors_bar = ['#2ecc71' if d > 0 else '#e74c3c' for d in diff.values]
        ax.barh(range(len(diff)), diff.values, color=colors_bar, edgecolor='black')
        ax.set_yticks(range(len(diff)))
        ax.set_yticklabels(diff.index)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel(f"LHS - Sobol ({metric.upper()})", fontsize=12)
        ax.set_title("Performance Difference\n(Positive = Sobol better)", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / f"sobol_vs_lhs_{metric}.png"
            save_figure(fig, path)
        
        return fig
    
    def plot_ntrain_effect_heatmap(self, metric: str = "mae",
                                    save: bool = True) -> plt.Figure:
        """
        Heatmap showing metric by (n_train, model_variant).
        """
        # Average by (n_train, model_variant)
        pivot = (self.df.groupby(["n_train", "model_variant"])[metric]
                 .mean()
                 .unstack())
        
        if pivot.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        cmap = "RdYlGn" if metric == "r2" else "RdYlGn_r"
        sns.heatmap(pivot, annot=True, fmt=".3g", cmap=cmap,
                    ax=ax, linewidths=0.5, cbar_kws={"label": metric.upper()})
        
        ax.set_xlabel("Model Variant", fontsize=12)
        ax.set_ylabel("n_train", fontsize=12)
        ax.set_title(f"Effect of Training Size on {metric.upper()}", 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / f"ntrain_effect_heatmap_{metric}.png"
            save_figure(fig, path)
        
        return fig
    
    def plot_learning_curves_by_benchmark(self, metric: str = "mae",
                                           save: bool = True) -> Dict[str, plt.Figure]:
        """
        Generate learning curves for each benchmark (small multiples).
        """
        benchmarks = sorted(self.df["benchmark"].unique())
        figures = {}
        
        for benchmark in benchmarks:
            fig = self.plot_metric_vs_ntrain(metric=metric, benchmark=benchmark, save=save)
            if fig:
                figures[benchmark] = fig
        
        return figures
    
    def generate_all_plots(self) -> Dict[str, Any]:
        """Generate all sampling effects plots."""
        logging.info("Generating sampling effects plots...")
        
        results = {}
        
        # Metric vs n_train (global)
        for metric in ["mae", "rmse", "r2"]:
            results[f"vs_ntrain_{metric}"] = self.plot_metric_vs_ntrain(metric)
        
        # Sobol vs LHS comparison
        for metric in ["mae", "rmse"]:
            results[f"sobol_vs_lhs_{metric}"] = self.plot_sobol_vs_lhs_comparison(metric)
        
        # n_train effect heatmap
        results["ntrain_heatmap_mae"] = self.plot_ntrain_effect_heatmap("mae")
        results["ntrain_heatmap_rmse"] = self.plot_ntrain_effect_heatmap("rmse")
        
        # Learning curves by benchmark
        results["learning_curves_mae"] = self.plot_learning_curves_by_benchmark("mae")
        
        plt.close('all')
        logging.info(f"Sampling effects plots saved to: {self.output_dir}")
        return results


# =============================================================================
# CV DIAGNOSTICS PLOTTER
# =============================================================================

class CVDiagnosticsPlotter:
    """
    Generates plots comparing simple vs nested CV.
    """
    
    def __init__(self, master_df: pd.DataFrame, output_dir: Path):
        self.df = master_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_model_color(self, model_variant: str) -> str:
        """Get consistent color for model variant."""
        if model_variant in MODEL_VARIANT_PALETTE:
            return MODEL_VARIANT_PALETTE[model_variant]
        return "#7f8c8d"
    
    def plot_simple_vs_nested_distribution(self, metric: str = "mae",
                                            benchmark: str = None,
                                            save: bool = True) -> plt.Figure:
        """
        Distribution comparison: simple vs nested CV.
        """
        df = self.df.copy()
        if benchmark:
            df = df[df["benchmark"] == benchmark]
            title_suffix = f" - {benchmark}"
        else:
            title_suffix = " (All Benchmarks)"
        
        if df.empty:
            return None
        
        # Check we have both cv_modes
        cv_modes = df["cv_mode"].unique()
        if len(cv_modes) < 2:
            logging.warning(f"Only one CV mode found: {cv_modes}")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Violin/box plot by cv_mode
        ax = axes[0]
        models = sorted(df["model_variant"].unique())
        positions = np.arange(len(models))
        width = 0.35
        
        for i, cv_mode in enumerate(["simple", "tuning"]):
            cv_df = df[df["cv_mode"] == cv_mode]
            if cv_df.empty:
                continue
            
            data = [cv_df[cv_df["model_variant"] == m][metric].values for m in models]
            bp = ax.boxplot(data, positions=positions + (i - 0.5) * width,
                           widths=width * 0.8, patch_artist=True)
            
            color = CV_MODE_PALETTE.get(cv_mode, "#7f8c8d")
            for box in bp['boxes']:
                box.set_facecolor(color)
                box.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} Distribution by CV Mode{title_suffix}", fontweight='bold')
        
        # Legend
        handles = [mpatches.Patch(color=CV_MODE_PALETTE.get(cv, "#7f8c8d"), label=cv)
                   for cv in ["simple", "tuning"]]
        ax.legend(handles=handles, loc='best')
        
        # Right: Gap plot (nested - simple)
        ax = axes[1]
        
        # Compute gap
        gaps = []
        for model in models:
            simple_val = df[(df["cv_mode"] == "simple") & (df["model_variant"] == model)][metric].mean()
            tuning_val = df[(df["cv_mode"] == "tuning") & (df["model_variant"] == model)][metric].mean()
            if pd.notna(simple_val) and pd.notna(tuning_val):
                gaps.append({
                    "model": model,
                    "gap": tuning_val - simple_val,
                    "simple": simple_val,
                    "tuning": tuning_val,
                })
        
        if gaps:
            gaps_df = pd.DataFrame(gaps).sort_values("gap")
            colors_bar = ['#2ecc71' if g < 0 else '#e74c3c' for g in gaps_df["gap"]]
            ax.barh(range(len(gaps_df)), gaps_df["gap"].values, color=colors_bar, edgecolor='black')
            ax.set_yticks(range(len(gaps_df)))
            ax.set_yticklabels(gaps_df["model"])
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel(f"Tuning - Simple ({metric.upper()})")
            ax.set_title(f"CV Mode Gap\n(Negative = Tuning better)", fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            suffix = f"_{benchmark}" if benchmark else "_global"
            path = self.output_dir / f"simple_vs_nested_{metric}{suffix}.png"
            save_figure(fig, path)
        
        return fig
    
    def plot_cv_stability_comparison(self, metric: str = "mae",
                                      save: bool = True) -> plt.Figure:
        """
        Compare stability (std) between CV modes.
        """
        # Only consider tuning mode (which has std from folds)
        tuning_df = self.df[self.df["cv_mode"] == "tuning"]
        
        if tuning_df.empty or f"{metric}_std" not in tuning_df.columns:
            logging.warning("No tuning data with std available")
            return None
        
        # Use the std from nested CV folds
        std_col = f"{metric}_std"
        
        agg = (tuning_df.groupby("model_variant")
               .agg({metric: "mean", std_col: "mean"})
               .reset_index())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = [self._get_model_color(m) for m in agg["model_variant"]]
        ax.scatter(agg[metric], agg[std_col], c=colors, s=150, alpha=0.8, edgecolors='black')
        
        for i, row in agg.iterrows():
            ax.annotate(row["model_variant"], (row[metric], row[std_col]),
                        fontsize=9, xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel(f"Mean {metric.upper()}", fontsize=12)
        ax.set_ylabel(f"Std {metric.upper()} (across folds)", fontsize=12)
        ax.set_title("Performance vs Stability (Nested CV)", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / f"cv_stability_{metric}.png"
            save_figure(fig, path)
        
        return fig
    
    def generate_all_plots(self) -> Dict[str, Any]:
        """Generate all CV diagnostics plots."""
        logging.info("Generating CV diagnostics plots...")
        
        results = {}
        
        # Simple vs nested distribution
        for metric in ["mae", "rmse"]:
            results[f"simple_vs_nested_{metric}"] = self.plot_simple_vs_nested_distribution(metric)
        
        # By benchmark
        for benchmark in self.df["benchmark"].unique():
            results[f"simple_vs_nested_mae_{benchmark}"] = self.plot_simple_vs_nested_distribution(
                "mae", benchmark=benchmark
            )
        
        # Stability comparison
        results["cv_stability_mae"] = self.plot_cv_stability_comparison("mae")
        
        plt.close('all')
        logging.info(f"CV diagnostics plots saved to: {self.output_dir}")
        return results


# =============================================================================
# TOP-X COMPARATOR
# =============================================================================

class TopXComparator:
    """
    Generates Top-X comparisons and cross-problem analysis.
    """
    
    def __init__(self, master_df: pd.DataFrame, output_dir: Path, top_x: int = 10):
        self.df = master_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_x = top_x
    
    def _get_model_color(self, model_variant: str) -> str:
        """Get consistent color for model variant."""
        if model_variant in MODEL_VARIANT_PALETTE:
            return MODEL_VARIANT_PALETTE[model_variant]
        return "#7f8c8d"
    
    def plot_topx_bars(self, benchmark: str, metric: str = "mae",
                        sampler: str = None, cv_mode: str = None,
                        save: bool = True) -> plt.Figure:
        """
        Bar plot of Top-X models for a benchmark.
        """
        ascending = metric != "r2"
        
        df = self.df[self.df["benchmark"] == benchmark]
        
        if sampler:
            df = df[df["sampler"] == sampler]
        if cv_mode:
            df = df[df["cv_mode"] == cv_mode]
        
        if df.empty:
            return None
        
        # Aggregate
        agg = (df.groupby("model_variant")[metric]
               .agg(["mean", "std", "count"])
               .reset_index()
               .sort_values("mean", ascending=ascending)
               .head(self.top_x))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = [self._get_model_color(m) for m in agg["model_variant"]]
        bars = ax.bar(range(len(agg)), agg["mean"], color=colors, edgecolor='black', alpha=0.8)
        
        # Error bars
        if agg["std"].notna().any():
            ax.errorbar(range(len(agg)), agg["mean"], yerr=agg["std"],
                        fmt='none', color='black', capsize=4)
        
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg["model_variant"], rotation=45, ha='right')
        ax.set_ylabel(metric.upper())
        
        # Title with all factors
        title_parts = [f"Top-{self.top_x} by {metric.upper()}", benchmark]
        if sampler:
            title_parts.append(f"sampler={sampler}")
        if cv_mode:
            title_parts.append(f"cv={cv_mode}")
        ax.set_title(" | ".join(title_parts), fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            suffix = f"_{sampler or 'all'}_{cv_mode or 'all'}"
            path = self.output_dir / f"top{self.top_x}_{benchmark}_{metric}{suffix}.png"
            save_figure(fig, path)
        
        return fig
    
    def plot_cross_problem_top1_heatmap(self, metric: str = "mae",
                                         sampler: str = None,
                                         cv_mode: str = None,
                                         save: bool = True) -> plt.Figure:
        """
        Heatmap showing best model per benchmark.
        """
        ascending = metric != "r2"
        
        df = self.df.copy()
        if sampler:
            df = df[df["sampler"] == sampler]
        if cv_mode:
            df = df[df["cv_mode"] == cv_mode]
        
        if df.empty:
            return None
        
        # Find best model per benchmark
        best_models = []
        for benchmark in df["benchmark"].unique():
            bench_df = df[df["benchmark"] == benchmark]
            avg = bench_df.groupby("model_variant")[metric].mean()
            best = avg.idxmin() if ascending else avg.idxmax()
            best_models.append({"benchmark": benchmark, "best_model": best, "score": avg[best]})
        
        best_df = pd.DataFrame(best_models)
        
        # Create pivot for heatmap
        models = sorted(df["model_variant"].unique())
        benchmarks = sorted(df["benchmark"].unique())
        
        # Create matrix where 1 = best model for that benchmark
        matrix = np.zeros((len(benchmarks), len(models)))
        for i, bench in enumerate(benchmarks):
            for j, model in enumerate(models):
                if best_df[best_df["benchmark"] == bench]["best_model"].values[0] == model:
                    matrix[i, j] = 1
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(matrix, annot=False, cmap="Greens",
                    xticklabels=models, yticklabels=benchmarks,
                    ax=ax, linewidths=0.5, cbar=False)
        
        # Mark best with star
        for i, bench in enumerate(benchmarks):
            best = best_df[best_df["benchmark"] == bench]["best_model"].values[0]
            j = models.index(best)
            ax.text(j + 0.5, i + 0.5, "★", ha='center', va='center', fontsize=16, color='gold')
        
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Benchmark", fontsize=12)
        
        title_parts = [f"Best Model per Benchmark ({metric.upper()})"]
        if sampler:
            title_parts.append(f"sampler={sampler}")
        if cv_mode:
            title_parts.append(f"cv={cv_mode}")
        ax.set_title(" | ".join(title_parts), fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            suffix = f"_{sampler or 'all'}_{cv_mode or 'all'}"
            path = self.output_dir / f"cross_problem_top1_{metric}{suffix}.png"
            save_figure(fig, path)
        
        return fig
    
    def plot_topx_stability(self, benchmark: str, metric: str = "mae",
                            save: bool = True) -> plt.Figure:
        """
        Violin/box plot showing stability of Top-X models.
        """
        ascending = metric != "r2"
        
        df = self.df[self.df["benchmark"] == benchmark]
        
        if df.empty:
            return None
        
        # Get top-X models
        avg = df.groupby("model_variant")[metric].mean()
        top_models = avg.sort_values(ascending=ascending).head(self.top_x).index.tolist()
        
        top_df = df[df["model_variant"].isin(top_models)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Order by mean
        order = top_df.groupby("model_variant")[metric].mean().sort_values(ascending=ascending).index.tolist()
        
        colors = [self._get_model_color(m) for m in order]
        
        # Violin plot
        parts = ax.violinplot(
            [top_df[top_df["model_variant"] == m][metric].values for m in order],
            positions=range(len(order)),
            showmeans=False, showmedians=False, showextrema=False
        )
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        
        # Overlay box
        bp = ax.boxplot(
            [top_df[top_df["model_variant"] == m][metric].values for m in order],
            positions=range(len(order)),
            widths=0.3, patch_artist=True, showfliers=True
        )
        
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i])
            box.set_alpha(0.8)
        
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha='right')
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Top-{self.top_x} Stability - {benchmark}", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / f"top{self.top_x}_stability_{benchmark}_{metric}.png"
            save_figure(fig, path)
        
        return fig
    
    def generate_all_plots(self) -> Dict[str, Any]:
        """Generate all Top-X comparison plots."""
        logging.info("Generating Top-X comparison plots...")
        
        results = {}
        
        # Top-X bars for each benchmark
        for benchmark in self.df["benchmark"].unique():
            for metric in ["mae", "rmse"]:
                results[f"topx_{benchmark}_{metric}"] = self.plot_topx_bars(benchmark, metric)
                results[f"topx_stability_{benchmark}_{metric}"] = self.plot_topx_stability(benchmark, metric)
        
        # Cross-problem heatmaps
        for metric in ["mae", "rmse"]:
            results[f"cross_top1_{metric}"] = self.plot_cross_problem_top1_heatmap(metric)
            
            # By sampler
            for sampler in self.df["sampler"].unique():
                results[f"cross_top1_{metric}_{sampler}"] = self.plot_cross_problem_top1_heatmap(
                    metric, sampler=sampler
                )
            
            # By cv_mode
            for cv_mode in self.df["cv_mode"].unique():
                results[f"cross_top1_{metric}_{cv_mode}"] = self.plot_cross_problem_top1_heatmap(
                    metric, cv_mode=cv_mode
                )
        
        plt.close('all')
        logging.info(f"Top-X plots saved to: {self.output_dir}")
        return results


# =============================================================================
# SUPER AGGREGATED PLOTTER (NEW - MULTI-PANEL OVERVIEW FIGURES)
# =============================================================================

class SuperAggregatedPlotter:
    """
    Generates powerful multi-panel overview figures.
    
    Required outputs (minimum 4):
    1. Heatmap + Ranking Dashboard (2-3 panels)
    2. Trade-off Plot (Quality vs Stability)
    3. Small Multiples / FacetGrid (metric vs n_train)
    4. Ridgeline / KDE distributions
    """
    
    def __init__(self, master_df: pd.DataFrame, output_dir: Path):
        self.df = master_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_family_color(self, family: str) -> str:
        """Get color for model family."""
        return MODEL_FAMILY_COLORS.get(family, "#7f8c8d")
    
    def plot_dashboard_heatmap_ranking(self, metric: str = "mae",
                                        save: bool = True) -> plt.Figure:
        """
        Dashboard with 3 panels:
        A) Heatmap: benchmark × model_variant (mean metric)
        B) Mean Rank with error bars
        C) Win count (nº times top-1)
        """
        ascending = metric != "r2"
        
        fig = plt.figure(figsize=(20, 8))
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1], wspace=0.3)
        
        # =====================================================================
        # Panel A: Heatmap
        # =====================================================================
        ax_heat = fig.add_subplot(gs[0])
        
        pivot = self.df.pivot_table(
            values=metric,
            index="benchmark",
            columns="model_variant",
            aggfunc="mean"
        )
        
        cmap = "RdYlGn" if metric == "r2" else "RdYlGn_r"
        sns.heatmap(pivot, annot=True, fmt=".3g", cmap=cmap,
                    ax=ax_heat, linewidths=0.5,
                    cbar_kws={"label": f"Mean {metric.upper()}", "shrink": 0.8})
        
        ax_heat.set_title(f"A) {metric.upper()} by Benchmark × Model", fontsize=12, fontweight='bold')
        ax_heat.set_xlabel("Model Variant", fontsize=10)
        ax_heat.set_ylabel("Benchmark", fontsize=10)
        plt.setp(ax_heat.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        
        # =====================================================================
        # Panel B: Mean Rank with error bars
        # =====================================================================
        ax_rank = fig.add_subplot(gs[1])
        
        # Compute ranks per benchmark
        ranks_list = []
        for benchmark in self.df["benchmark"].unique():
            bench_df = self.df[self.df["benchmark"] == benchmark]
            avg = bench_df.groupby("model_variant")[metric].mean()
            
            if ascending:
                rank_series = avg.rank(ascending=True)
            else:
                rank_series = avg.rank(ascending=False)
            
            for model, rank in rank_series.items():
                ranks_list.append({"model_variant": model, "rank": rank, "benchmark": benchmark})
        
        ranks_df = pd.DataFrame(ranks_list)
        mean_ranks = ranks_df.groupby("model_variant")["rank"].agg(["mean", "std"]).reset_index()
        mean_ranks = mean_ranks.sort_values("mean")
        
        colors = [self._get_family_color(
            self.df[self.df["model_variant"] == m]["model_family"].iloc[0] 
            if len(self.df[self.df["model_variant"] == m]) > 0 else "Unknown"
        ) for m in mean_ranks["model_variant"]]
        
        ax_rank.barh(range(len(mean_ranks)), mean_ranks["mean"], 
                     xerr=mean_ranks["std"].fillna(0), 
                     color=colors, edgecolor='black', alpha=0.8, capsize=3)
        ax_rank.set_yticks(range(len(mean_ranks)))
        ax_rank.set_yticklabels(mean_ranks["model_variant"], fontsize=9)
        ax_rank.set_xlabel("Mean Rank (lower = better)", fontsize=10)
        ax_rank.set_title("B) Mean Rank ± Std", fontsize=12, fontweight='bold')
        ax_rank.invert_yaxis()
        
        # =====================================================================
        # Panel C: Win count
        # =====================================================================
        ax_wins = fig.add_subplot(gs[2])
        
        # Count wins (top-1) per model
        wins = {}
        for benchmark in self.df["benchmark"].unique():
            bench_df = self.df[self.df["benchmark"] == benchmark]
            avg = bench_df.groupby("model_variant")[metric].mean()
            
            if ascending:
                winner = avg.idxmin()
            else:
                winner = avg.idxmax()
            
            wins[winner] = wins.get(winner, 0) + 1
        
        wins_df = pd.DataFrame([
            {"model_variant": m, "wins": w} for m, w in wins.items()
        ]).sort_values("wins", ascending=True)
        
        colors_wins = [self._get_family_color(
            self.df[self.df["model_variant"] == m]["model_family"].iloc[0]
            if len(self.df[self.df["model_variant"] == m]) > 0 else "Unknown"
        ) for m in wins_df["model_variant"]]
        
        ax_wins.barh(range(len(wins_df)), wins_df["wins"], color=colors_wins, edgecolor='black', alpha=0.8)
        ax_wins.set_yticks(range(len(wins_df)))
        ax_wins.set_yticklabels(wins_df["model_variant"], fontsize=9)
        ax_wins.set_xlabel("Win Count (Top-1)", fontsize=10)
        ax_wins.set_title("C) Wins per Model", fontsize=12, fontweight='bold')
        
        # Add legend for model families
        family_handles = [mpatches.Patch(color=self._get_family_color(f), label=f) 
                          for f in MODEL_FAMILY_COLORS.keys()]
        fig.legend(handles=family_handles, loc='upper right', fontsize=9, title="Model Family")
        
        fig.suptitle(f"Dashboard: {metric.upper()} Overview (All Factors)", 
                     fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / f"dashboard_heatmap_ranking_{metric}.png"
            save_figure(fig, path)
        
        return fig
    
    def plot_tradeoff_quality_stability(self, metric: str = "mae",
                                         save: bool = True) -> plt.Figure:
        """
        Trade-off scatter:
        - X: mean metric
        - Y: stability (std or IQR)
        - Color: sampler
        - Marker: cv_mode
        - Size: n_train (inverse - smaller n_train = bigger marker)
        """
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Aggregate by (model_variant, sampler, cv_mode, n_train)
        agg = (self.df.groupby(["model_variant", "sampler", "cv_mode", "n_train"])[metric]
               .agg(["mean", "std"])
               .reset_index())
        agg.columns = ["model_variant", "sampler", "cv_mode", "n_train", "mean", "std"]
        
        # Replace NaN std with 0
        agg["std"] = agg["std"].fillna(0)
        
        # Size based on n_train (inverse: smaller n_train = bigger marker)
        n_train_values = sorted(agg["n_train"].unique())
        size_map = {nt: 200 - i * 30 for i, nt in enumerate(n_train_values)}
        agg["size"] = agg["n_train"].map(size_map)
        
        # Markers for cv_mode
        marker_map = {"simple": "o", "tuning": "s", "nested": "s"}
        
        # Plot each group
        for sampler in agg["sampler"].unique():
            for cv_mode in agg["cv_mode"].unique():
                subset = agg[(agg["sampler"] == sampler) & (agg["cv_mode"] == cv_mode)]
                
                if subset.empty:
                    continue
                
                ax.scatter(
                    subset["mean"], subset["std"],
                    c=SAMPLER_PALETTE.get(sampler, "#7f8c8d"),
                    s=subset["size"],
                    marker=marker_map.get(cv_mode, "o"),
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f"{sampler}/{cv_mode}"
                )
        
        # Annotate top models (best mean with low std)
        # Pareto frontier approximation
        agg_sorted = agg.sort_values(["mean", "std"])
        top_n = min(5, len(agg_sorted))
        for i, row in agg_sorted.head(top_n).iterrows():
            ax.annotate(
                f"{row['model_variant']}\nn={row['n_train']}",
                (row["mean"], row["std"]),
                fontsize=8, xytext=(5, 5), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
            )
        
        ax.set_xlabel(f"Mean {metric.upper()} (Quality)", fontsize=12)
        ax.set_ylabel(f"Std {metric.upper()} (Stability)", fontsize=12)
        ax.set_title(f"Quality vs Stability Trade-off\n(Size ∝ 1/n_train)", fontsize=14, fontweight='bold')
        
        # Legend
        ax.legend(loc='upper right', fontsize=9, title="Sampler/CV")
        
        # Size legend
        handles_size = [plt.scatter([], [], s=size_map[nt], c='gray', alpha=0.5, label=f"n={nt}")
                        for nt in n_train_values[:4]]  # Limit to 4
        ax.legend(handles=handles_size, loc='lower right', title="n_train", fontsize=8)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / f"tradeoff_quality_stability_{metric}.png"
            save_figure(fig, path)
        
        return fig
    
    def plot_facet_learning_curves(self, metric: str = "mae",
                                    by: str = "model_family",
                                    save: bool = True) -> plt.Figure:
        """
        FacetGrid: metric vs n_train, faceted by model_family (or benchmark).
        One panel per family, lines for each sampler, with error bands.
        """
        families = sorted(self.df["model_family"].unique())
        n_families = len(families)
        
        n_cols = min(2, n_families)
        n_rows = (n_families + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), sharey=True)
        axes = np.atleast_2d(axes)
        
        for idx, family in enumerate(families):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            family_df = self.df[self.df["model_family"] == family]
            
            # Aggregate by (n_train, sampler)
            agg = (family_df.groupby(["n_train", "sampler"])[metric]
                   .agg(["mean", "std"])
                   .reset_index())
            
            for sampler in agg["sampler"].unique():
                sampler_data = agg[agg["sampler"] == sampler].sort_values("n_train")
                color = SAMPLER_PALETTE.get(sampler, "#7f8c8d")
                
                ax.plot(sampler_data["n_train"], sampler_data["mean"],
                        'o-', color=color, label=sampler.upper(), linewidth=2, markersize=6)
                
                if sampler_data["std"].notna().any():
                    ax.fill_between(
                        sampler_data["n_train"],
                        sampler_data["mean"] - sampler_data["std"],
                        sampler_data["mean"] + sampler_data["std"],
                        color=color, alpha=0.2
                    )
            
            ax.set_xlabel("n_train", fontsize=10)
            ax.set_ylabel(metric.upper(), fontsize=10)
            ax.set_title(f"{family}", fontsize=12, fontweight='bold', color=self._get_family_color(family))
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_families, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)
        
        fig.suptitle(f"Learning Curves by Model Family ({metric.upper()})",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            path = self.output_dir / f"facet_learning_curves_{metric}_by_{by}.png"
            save_figure(fig, path)
        
        return fig
    
    def plot_ridgeline_kde(self, metric: str = "mae",
                           save: bool = True) -> plt.Figure:
        """
        Ridgeline/KDE plot comparing simple vs nested CV distributions.
        Separated by sampler.
        """
        samplers = sorted(self.df["sampler"].unique())
        cv_modes = ["simple", "tuning"]
        
        fig, axes = plt.subplots(1, len(samplers), figsize=(8 * len(samplers), 8), sharey=True)
        
        if len(samplers) == 1:
            axes = [axes]
        
        models = sorted(self.df["model_variant"].unique())
        y_positions = np.arange(len(models))
        
        for ax, sampler in zip(axes, samplers):
            ax.set_title(f"Sampler: {sampler.upper()}", fontsize=12, fontweight='bold')
            
            for i, model in enumerate(models):
                for cv_mode in cv_modes:
                    subset = self.df[
                        (self.df["model_variant"] == model) &
                        (self.df["sampler"] == sampler) &
                        (self.df["cv_mode"] == cv_mode)
                    ][metric].dropna()
                    
                    if len(subset) < 3:
                        continue
                    
                    try:
                        kde = gaussian_kde(subset)
                        x_range = np.linspace(subset.min(), subset.max(), 100)
                        density = kde(x_range)
                        
                        # Normalize density for visibility
                        density = density / density.max() * 0.35
                        
                        color = CV_MODE_PALETTE.get(cv_mode, "#7f8c8d")
                        alpha = 0.6 if cv_mode == "simple" else 0.4
                        
                        ax.fill_betweenx(
                            i + density, x_range, i,
                            alpha=alpha, color=color,
                            label=cv_mode if i == 0 else None
                        )
                        ax.plot(x_range, i + density, color=color, linewidth=1)
                    except Exception:
                        continue
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels(models, fontsize=9)
            ax.set_xlabel(metric.upper(), fontsize=10)
            ax.set_ylabel("Model", fontsize=10)
            ax.legend(loc='upper right', fontsize=9, title="CV Mode")
            ax.grid(True, axis='x', alpha=0.3)
        
        fig.suptitle(f"Distribution Ridgeline: Simple vs Nested ({metric.upper()})",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            path = self.output_dir / f"ridgeline_kde_{metric}.png"
            save_figure(fig, path)
        
        return fig
    
    def generate_all_plots(self) -> Dict[str, Any]:
        """Generate all super aggregated plots (minimum 4 required)."""
        logging.info("Generating super aggregated overview plots...")
        
        results = {}
        
        # 1. Dashboard: Heatmap + Ranking + Wins
        for metric in ["mae", "rmse"]:
            results[f"dashboard_{metric}"] = self.plot_dashboard_heatmap_ranking(metric)
        
        # 2. Trade-off: Quality vs Stability
        for metric in ["mae", "rmse"]:
            results[f"tradeoff_{metric}"] = self.plot_tradeoff_quality_stability(metric)
        
        # 3. FacetGrid: Learning curves by model family
        for metric in ["mae", "rmse"]:
            results[f"facet_learning_{metric}"] = self.plot_facet_learning_curves(metric)
        
        # 4. Ridgeline/KDE: simple vs nested
        for metric in ["mae", "rmse"]:
            results[f"ridgeline_{metric}"] = self.plot_ridgeline_kde(metric)
        
        plt.close('all')
        logging.info(f"Super aggregated plots saved to: {self.output_dir}")
        return results


# =============================================================================
# GP PREDICTIONS GENERATOR (WITH SAMPLER POINTS + UNCERTAINTY BANDS)
# =============================================================================

class GPPredictionsGenerator:
    """
    Generates GP prediction visualizations with:
    - Training points (sampler overlay)
    - Uncertainty bands (±1σ, ±2σ)
    - Multiple n_train values in subplots
    - Comparison across samplers
    
    This integrates with gp_visualization.py but is adapted for batch reporting.
    """
    
    def __init__(self, master_df: pd.DataFrame, output_dir: Path,
                 session_dir: Path = None, top_k_gp: int = 3):
        self.df = master_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = session_dir
        self.top_k_gp = top_k_gp
        
        # Filter to GP models only
        self.gp_df = self.df[self.df["model_family"] == "GP"].copy()
        
        # Try to import GP training functions
        self._gp_available = False
        try:
            from src.evaluation.gp_visualization import train_gp, GPModel
            from src.benchmarks import get_benchmark, generate_benchmark_dataset
            self._train_gp = train_gp
            self._get_benchmark = get_benchmark
            self._generate_dataset = generate_benchmark_dataset
            self._gp_available = True
        except ImportError as e:
            logging.warning(f"GP visualization dependencies not available: {e}")
    
    def _get_top_gp_configs(self, benchmark: str, sampler: str,
                            metric: str = "mae") -> List[str]:
        """Get top-k GP configurations by metric."""
        subset = self.gp_df[
            (self.gp_df["benchmark"] == benchmark) &
            (self.gp_df["sampler"] == sampler)
        ]
        
        if subset.empty:
            return []
        
        ascending = metric != "r2"
        avg = subset.groupby("model_variant")[metric].mean()
        return avg.sort_values(ascending=ascending).head(self.top_k_gp).index.tolist()
    
    def plot_gp_ntrain_sweep_with_predictions(self, benchmark: str, sampler: str,
                                               model_variant: str = "GP_Matern52",
                                               noise: str = "Gaussian_s0.1",
                                               seed: int = 42,
                                               save: bool = True) -> plt.Figure:
        """
        Generate GP prediction plot with n_train sweep.
        
        Shows multiple subplots, one per n_train value, each with:
        - True function (if 1D/2D)
        - GP mean prediction
        - ±1σ, ±2σ uncertainty bands
        - Training points (X_train) as scatter
        """
        if not self._gp_available:
            logging.warning("GP visualization not available, generating metrics-only plot")
            return self._plot_gp_metrics_sweep(benchmark, sampler, model_variant, save)
        
        # Get n_train values
        n_train_values = sorted(self.gp_df["n_train"].unique())
        
        if len(n_train_values) == 0:
            return None
        
        try:
            bench_func = self._get_benchmark(benchmark)
        except Exception as e:
            logging.warning(f"Cannot get benchmark {benchmark}: {e}")
            return self._plot_gp_metrics_sweep(benchmark, sampler, model_variant, save)
        
        dim = bench_func.dim
        
        # Only handle 1D functions with full visualization
        if dim != 1:
            return self._plot_gp_metrics_sweep(benchmark, sampler, model_variant, save)
        
        # Create subplots for each n_train
        n_plots = len(n_train_values)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharey=True)
        axes = np.atleast_2d(axes).flatten()
        
        # Parse kernel from model_variant
        kernel_name = model_variant.replace("GP_", "") if model_variant.startswith("GP_") else "Matern52"
        
        # Dense grid for plotting
        bounds = bench_func.bounds[0]
        X_grid = np.linspace(bounds[0], bounds[1], 300).reshape(-1, 1)
        y_true = bench_func(X_grid)
        
        for idx, n_train in enumerate(n_train_values):
            ax = axes[idx]
            
            try:
                # Generate dataset with specific sampler and n_train
                dataset = self._generate_dataset(
                    benchmark=benchmark,
                    n_train=n_train,
                    n_test=100,
                    sampler=sampler,
                    noise='gaussian' if 'Gaussian' in noise else 'none',
                    noise_kwargs={'sigma': 0.1},
                    seed=seed,
                )
                
                # Train GP
                gp_model = self._train_gp(
                    dataset.X_train, dataset.y_train,
                    kernel_name=kernel_name,
                    n_restarts=3
                )
                
                # Predict
                y_mean, y_std = gp_model.predict(X_grid)
                
                # Plot uncertainty bands
                ax.fill_between(
                    X_grid.ravel(),
                    y_mean - 2 * y_std,
                    y_mean + 2 * y_std,
                    alpha=0.2, color='#ff7f0e', label='±2σ'
                )
                ax.fill_between(
                    X_grid.ravel(),
                    y_mean - y_std,
                    y_mean + y_std,
                    alpha=0.4, color='#ffbb78', label='±1σ'
                )
                
                # Plot true function
                ax.plot(X_grid, y_true, '--', color='#1f77b4', linewidth=2, label='True')
                
                # Plot GP mean
                ax.plot(X_grid, y_mean, color='#ff7f0e', linewidth=2, label='GP mean')
                
                # Plot training points
                sampler_color = SAMPLER_PALETTE.get(sampler, '#2ca02c')
                ax.scatter(
                    dataset.X_train, dataset.y_train,
                    c=sampler_color, s=60, marker='o',
                    edgecolors='white', linewidths=1,
                    label=f'Train ({sampler})', zorder=10
                )
                
            except Exception as e:
                logging.warning(f"Failed to generate GP plot for n={n_train}: {e}")
                ax.text(0.5, 0.5, f"Error: {str(e)[:30]}...",
                        transform=ax.transAxes, ha='center', va='center')
            
            ax.set_xlabel("x", fontsize=10)
            ax.set_ylabel("f(x)", fontsize=10)
            ax.set_title(f"n_train = {n_train}", fontsize=11, fontweight='bold')
            ax.set_xlim(bounds)
            
            if idx == 0:
                ax.legend(loc='best', fontsize=8)
            
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(
            f"{benchmark} | {model_variant} | {sampler.upper()}\nGP Predictions across n_train",
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        
        if save:
            path = self.output_dir / f"gp_ntrain_sweep_{benchmark}_{sampler}_{model_variant}.png"
            save_figure(fig, path)
        
        return fig
    
    def _plot_gp_metrics_sweep(self, benchmark: str, sampler: str,
                                model_variant: str,
                                save: bool = True) -> plt.Figure:
        """
        Fallback: Plot GP metrics evolution across n_train (no actual predictions).
        """
        subset = self.gp_df[
            (self.gp_df["benchmark"] == benchmark) &
            (self.gp_df["sampler"] == sampler) &
            (self.gp_df["model_variant"] == model_variant)
        ]
        
        if subset.empty:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MAE vs n_train
        agg = subset.groupby("n_train")["mae"].agg(["mean", "std"]).reset_index()
        axes[0].errorbar(agg["n_train"], agg["mean"], yerr=agg["std"],
                         fmt='o-', capsize=4, color='#e74c3c', linewidth=2)
        axes[0].set_xlabel("n_train")
        axes[0].set_ylabel("MAE")
        axes[0].set_title("MAE Evolution")
        axes[0].grid(True, alpha=0.3)
        
        # Coverage vs n_train
        if "coverage_95" in subset.columns:
            agg = subset.groupby("n_train")["coverage_95"].agg(["mean", "std"]).reset_index()
            axes[1].errorbar(agg["n_train"], agg["mean"], yerr=agg["std"],
                             fmt='o-', capsize=4, color='#3498db', linewidth=2)
            axes[1].axhline(y=0.95, color='green', linestyle='--', label='Target 95%')
            axes[1].set_xlabel("n_train")
            axes[1].set_ylabel("Coverage 95%")
            axes[1].set_title("Coverage Evolution")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # NLPD vs n_train
        if "nlpd" in subset.columns:
            agg = subset.groupby("n_train")["nlpd"].agg(["mean", "std"]).reset_index()
            axes[2].errorbar(agg["n_train"], agg["mean"], yerr=agg["std"],
                             fmt='o-', capsize=4, color='#9b59b6', linewidth=2)
            axes[2].set_xlabel("n_train")
            axes[2].set_ylabel("NLPD")
            axes[2].set_title("NLPD Evolution")
            axes[2].grid(True, alpha=0.3)
        
        fig.suptitle(f"{model_variant} | {benchmark} | {sampler}",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            path = self.output_dir / f"gp_metrics_sweep_{benchmark}_{sampler}_{model_variant}.png"
            save_figure(fig, path)
        
        return fig
    
    def plot_sampler_comparison_panel(self, benchmark: str,
                                       model_variant: str = "GP_Matern52",
                                       n_train: int = None,
                                       save: bool = True) -> plt.Figure:
        """
        Side-by-side comparison of Sobol vs LHS for GP.
        Shows 4-panel: MAE, Coverage, NLPD, Sharpness.
        """
        subset = self.gp_df[
            (self.gp_df["benchmark"] == benchmark) &
            (self.gp_df["model_variant"] == model_variant)
        ]
        
        if n_train:
            subset = subset[subset["n_train"] == n_train]
        
        if subset.empty:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ["mae", "coverage_95", "nlpd", "sharpness"]
        titles = ["MAE", "Coverage 95%", "NLPD", "Sharpness"]
        
        for ax, metric, title in zip(axes.ravel(), metrics, titles):
            if metric not in subset.columns or subset[metric].isna().all():
                ax.set_visible(False)
                continue
            
            # Aggregate by sampler
            agg = subset.groupby("sampler")[metric].agg(["mean", "std"]).reset_index()
            
            colors = [SAMPLER_PALETTE.get(s, "#7f8c8d") for s in agg["sampler"]]
            bars = ax.bar(agg["sampler"], agg["mean"], color=colors, edgecolor='black', alpha=0.8)
            
            if agg["std"].notna().any():
                ax.errorbar(range(len(agg)), agg["mean"], yerr=agg["std"],
                            fmt='none', color='black', capsize=6)
            
            ax.set_ylabel(metric.upper())
            ax.set_title(title, fontweight='bold')
        
        fig.suptitle(f"{model_variant} | {benchmark} | Sobol vs LHS",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            suffix = f"_n{n_train}" if n_train else ""
            path = self.output_dir / f"gp_sampler_comparison_{benchmark}_{model_variant}{suffix}.png"
            save_figure(fig, path)
        
        return fig
    
    def generate_gp_atlas_index(self) -> Path:
        """Generate comprehensive GP atlas index."""
        index_path = self.output_dir.parent / "indices" / "GP_ATLAS_INDEX.md"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = [
            "# GP Hyperparam Atlas Index",
            "",
            "## Overview",
            "",
            "This atlas contains GP prediction visualizations showing:",
            "- Training points (sampler overlay: Sobol vs LHS)",
            "- Uncertainty bands (±1σ, ±2σ)",
            "- Effect of n_train on predictions",
            "",
            "## n_train Sweep Plots",
            "",
            "Shows how GP predictions and uncertainty evolve with training size.",
            "",
        ]
        
        # List n_train sweep files
        for f in sorted(self.output_dir.glob("gp_ntrain_sweep_*.png")):
            lines.append(f"- [{f.stem}](../figures/gp_predictions/{f.name})")
        
        for f in sorted(self.output_dir.glob("gp_metrics_sweep_*.png")):
            lines.append(f"- [{f.stem}](../figures/gp_predictions/{f.name})")
        
        lines.extend([
            "",
            "## Sampler Comparison Plots",
            "",
            "Side-by-side comparison of Sobol vs LHS sampling for GP models.",
            "",
        ])
        
        for f in sorted(self.output_dir.glob("gp_sampler_comparison_*.png")):
            lines.append(f"- [{f.stem}](../figures/gp_predictions/{f.name})")
        
        lines.extend([
            "",
            "## Interpretation Guide",
            "",
            "- **Uncertainty bands**: Wider bands indicate less confident predictions",
            "- **Sobol vs LHS**: Sobol provides more uniform coverage, LHS better stratification",
            "- **n_train effect**: More training points → narrower uncertainty bands",
            "",
        ])
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logging.info(f"Generated GP atlas index: {index_path}")
        return index_path
    
    def generate_all_plots(self) -> Dict[str, Any]:
        """Generate all GP prediction plots."""
        logging.info("Generating GP prediction plots with sampler overlay...")
        
        if self.gp_df.empty:
            logging.warning("No GP data found in results")
            return {}
        
        results = {}
        
        benchmarks = self.gp_df["benchmark"].unique()
        samplers = self.gp_df["sampler"].unique()
        
        # For each benchmark and sampler, generate n_train sweep for top-k GPs
        for benchmark in benchmarks:
            for sampler in samplers:
                top_gp_variants = self._get_top_gp_configs(benchmark, sampler)
                
                for variant in top_gp_variants:
                    key = f"ntrain_sweep_{benchmark}_{sampler}_{variant}"
                    results[key] = self.plot_gp_ntrain_sweep_with_predictions(
                        benchmark, sampler, variant
                    )
        
        # Sampler comparison for each benchmark
        for benchmark in benchmarks:
            for variant in self.gp_df["model_variant"].unique():
                key = f"sampler_comp_{benchmark}_{variant}"
                results[key] = self.plot_sampler_comparison_panel(benchmark, variant)
        
        # Generate atlas index
        self.generate_gp_atlas_index()
        
        plt.close('all')
        logging.info(f"GP prediction plots saved to: {self.output_dir}")
        return results


# =============================================================================
# REPORT INDEX GENERATOR (UPDATED)
# =============================================================================

def generate_report_index(output_dir: Path, inventory: BenchmarkInventoryMultiMode) -> Path:
    """Generate main report index markdown."""
    index_path = output_dir / "indices" / "REPORT_INDEX.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# Benchmark Visual Evaluation Report (Multi-Mode)",
        "",
        f"**Session**: {inventory.session_name}",
        f"**Timestamp**: {inventory.timestamp}",
        f"**Total Time**: {inventory.total_time_s:.1f}s",
        "",
        "## Configuration",
        "",
        f"- **Samplers**: {', '.join(inventory.samplers)}",
        f"- **CV Modes**: {', '.join(inventory.cv_modes)}",
        f"- **Training Sizes**: {inventory.n_train_values}",
        f"- **Benchmarks**: {', '.join(inventory.benchmarks)}",
        f"- **Noise Types**: {', '.join(inventory.noise_types)}",
        f"- **Models**: {', '.join(inventory.models)}",
        f"- **Total Results**: {inventory.n_results}",
        "",
        "## Directory Structure",
        "",
        "```",
        "report_multimode/",
        "├── tables/",
        "│   ├── master_table.csv",
        "│   ├── leaderboard_*.csv",
        "│   ├── stability_*.csv",
        "│   └── topX_by_problem/",
        "├── figures/",
        "│   ├── global/                    (original global plots)",
        "│   ├── overview_super_aggregated/ (NEW: multi-panel dashboards)",
        "│   ├── sampling_effects/          (sobol vs lhs, n_train effects)",
        "│   ├── cv_diagnostics/            (simple vs nested comparison)",
        "│   ├── comparisons_topX/          (top-X analysis)",
        "│   └── gp_predictions/            (NEW: GP with sampler points + bands)",
        "└── indices/",
        "    ├── REPORT_INDEX.md            (this file)",
        "    └── GP_ATLAS_INDEX.md          (GP visualization guide)",
        "```",
        "",
        "## Navigation",
        "",
        "### Tables",
        "- [Master Table](../tables/master_table.csv)",
        "- [Global Leaderboard MAE](../tables/leaderboard_global_mae.csv)",
        "- [Stability Analysis](../tables/stability_mae.csv)",
        "- [Best Model by Factor](../tables/best_model_per_benchmark_by_sampler_cv_ntrain_mae.csv)",
        "",
        "### Super Aggregated Overview (NEW)",
        "",
        "Multi-panel dashboards for global insights:",
        "",
        "- [Dashboard: Heatmap + Ranking + Wins (MAE)](../figures/overview_super_aggregated/dashboard_heatmap_ranking_mae.png)",
        "- [Trade-off: Quality vs Stability](../figures/overview_super_aggregated/tradeoff_quality_stability_mae.png)",
        "- [FacetGrid: Learning Curves by Family](../figures/overview_super_aggregated/facet_learning_curves_mae_by_model_family.png)",
        "- [Ridgeline/KDE: CV Mode Distributions](../figures/overview_super_aggregated/ridgeline_kde_mae.png)",
        "",
        "### Sampling Effects",
        "- [Sobol vs LHS Comparison](../figures/sampling_effects/sobol_vs_lhs_mae.png)",
        "- [n_train Effect Heatmap](../figures/sampling_effects/ntrain_effect_heatmap_mae.png)",
        "",
        "### CV Diagnostics",
        "- [Simple vs Nested Distribution](../figures/cv_diagnostics/simple_vs_nested_mae_global.png)",
        "",
        "### Top-X Comparisons",
        "- [Cross-Problem Top-1 Heatmap](../figures/comparisons_topX/cross_problem_top1_mae_all_all.png)",
        "",
        "### GP Predictions (NEW)",
        "",
        "GP visualizations with training points and uncertainty bands:",
        "",
        "- [GP Atlas Index](GP_ATLAS_INDEX.md)",
        "",
        "Key features:",
        "- Training points (X_train) as scatter with sampler color",
        "- Uncertainty bands (±1σ, ±2σ shading)",
        "- n_train sweep showing how predictions improve",
        "- Sobol vs LHS comparison",
        "",
        "---",
        "",
        "*Output format: PNG only (no PDF for performance)*",
        "",
    ]
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logging.info(f"Generated report index: {index_path}")
    return index_path


# =============================================================================
# MAIN REPORT GENERATOR (UPDATED)
# =============================================================================

def generate_benchmark_report_multimode(
    session_name: str,
    results_json_path: Path = None,
    output_dir: Path = None,
    strict: bool = True,
    top_x: int = 10,
    max_atlas_configs: int = 25,
) -> Path:
    """
    Generate complete multi-mode benchmark visual evaluation report.
    
    Args:
        session_name: Name or partial name of benchmark session
        results_json_path: Explicit path to results JSON (auto-detected if None)
        output_dir: Output directory (auto-generated if None)
        strict: If True, abort if requirements (sobol+lhs, simple+nested) not met
        top_x: Number of top configurations to analyze
        max_atlas_configs: Max GP configs for atlas (anti-explosion)
        
    Returns:
        Path to report directory
        
    Raises:
        ValueError: If strict=True and requirements not met
        FileNotFoundError: If results JSON not found
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    print("=" * 70)
    print(f"BENCHMARK VISUAL EVALUATION (MULTI-MODE): {session_name}")
    print("=" * 70)
    
    # Auto-detect results file
    if results_json_path is None:
        # Search in LOGS_DIR/benchmarks/
        search_dir = LOGS_DIR / "benchmarks"
        candidates = list(search_dir.glob(f"*{session_name}*"))
        
        if not candidates:
            raise FileNotFoundError(f"No session found matching '{session_name}' in {search_dir}")
        
        # Look for comprehensive_results.json
        for candidate in candidates:
            if candidate.is_dir():
                json_path = candidate / "comprehensive_results.json"
                if json_path.exists():
                    results_json_path = json_path
                    break
        
        if results_json_path is None:
            raise FileNotFoundError(f"No comprehensive_results.json found in candidates: {candidates}")
    
    results_json_path = Path(results_json_path)
    logging.info(f"Loading results from: {results_json_path}")
    
    # Load and verify data (strict=False to allow partial data)
    loader = MultiModeResultsLoader(results_json_path, strict=False)
    requirements = loader.verify_requirements()
    
    if not requirements.is_valid:
        print(f"\n❌ {requirements.get_error_message()}")
        print("\nEncontrado en este JSON:")
        print(f"  - samplers: {requirements.samplers_found}")
        print(f"  - cv_modes: {requirements.cv_modes_found}")
        raise ValueError(requirements.get_error_message())
    
    logging.info(f"Data found: samplers={requirements.samplers_found}, cv_modes={requirements.cv_modes_found}")
    
    df = loader.build_master_table()
    inventory = loader.get_inventory()
    
    # Setup output directory
    if output_dir is None:
        output_dir = results_json_path.parent / "report_multimode"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Output directory: {output_dir}")
    
    # Create directory structure (UPDATED)
    dirs = {
        "tables": output_dir / "tables",
        "global": output_dir / "figures" / "global",
        "super_aggregated": output_dir / "figures" / "overview_super_aggregated",
        "sampling_effects": output_dir / "figures" / "sampling_effects",
        "cv_diagnostics": output_dir / "figures" / "cv_diagnostics",
        "comparisons_topX": output_dir / "figures" / "comparisons_topX",
        "gp_predictions": output_dir / "figures" / "gp_predictions",
        "indices": output_dir / "indices",
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Export Master Table
    # =========================================================================
    print("\n[1/8] Exporting master table...")
    loader.export_master_table(dirs["tables"])
    
    # =========================================================================
    # STEP 2: Generate Tables
    # =========================================================================
    print("\n[2/8] Generating tables...")
    table_gen = MultiModeTableGenerator(df, dirs["tables"])
    table_gen.generate_all_tables()
    
    # =========================================================================
    # STEP 3: Generate Original Global Plots (reuse existing)
    # =========================================================================
    print("\n[3/8] Generating global plots (original style)...")
    
    # Transform df to match original format expected by BenchmarkGlobalPlotter
    column_mapping = {
        "noise": "noise_type",
        "fit_time": "fit_time_s",
        "predict_time": "predict_time_s",
    }
    df_original = df.rename(columns=column_mapping)
    
    global_plotter = BenchmarkGlobalPlotter(df_original, dirs["global"])
    global_plotter.generate_all_plots()
    
    # =========================================================================
    # STEP 4: Super Aggregated Overview Plots (NEW)
    # =========================================================================
    print("\n[4/8] Generating super aggregated overview plots...")
    super_plotter = SuperAggregatedPlotter(df, dirs["super_aggregated"])
    super_plotter.generate_all_plots()
    
    # =========================================================================
    # STEP 5: Sampling Effects Plots
    # =========================================================================
    print("\n[5/8] Generating sampling effects plots...")
    sampling_plotter = SamplingEffectsPlotter(df, dirs["sampling_effects"])
    sampling_plotter.generate_all_plots()
    
    # =========================================================================
    # STEP 6: CV Diagnostics Plots
    # =========================================================================
    print("\n[6/8] Generating CV diagnostics plots...")
    cv_plotter = CVDiagnosticsPlotter(df, dirs["cv_diagnostics"])
    cv_plotter.generate_all_plots()
    
    # =========================================================================
    # STEP 7: Top-X Comparison Plots
    # =========================================================================
    print("\n[7/8] Generating Top-X comparison plots...")
    topx_comparator = TopXComparator(df, dirs["comparisons_topX"], top_x=top_x)
    topx_comparator.generate_all_plots()
    
    # =========================================================================
    # STEP 8: GP Predictions with Sampler Points (NEW - CRITICAL)
    # =========================================================================
    print("\n[8/8] Generating GP predictions with sampler points + uncertainty bands...")
    gp_generator = GPPredictionsGenerator(
        df, dirs["gp_predictions"],
        session_dir=results_json_path.parent,
        top_k_gp=3
    )
    gp_generator.generate_all_plots()
    
    # =========================================================================
    # Generate Report Index
    # =========================================================================
    print("\nGenerating report index...")
    generate_report_index(output_dir, inventory)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n✓ Output directory: {output_dir}")
    print(f"✓ Total records processed: {len(df)}")
    print(f"✓ Samplers: {inventory.samplers}")
    print(f"✓ CV modes: {inventory.cv_modes}")
    print(f"✓ Training sizes: {inventory.n_train_values}")
    print(f"✓ Benchmarks: {len(inventory.benchmarks)}")
    print(f"✓ Models: {len(inventory.models)}")
    print(f"\n📄 Report Index: {output_dir / 'indices' / 'REPORT_INDEX.md'}")
    print(f"📄 GP Atlas Index: {output_dir / 'indices' / 'GP_ATLAS_INDEX.md'}")
    print(f"\n📂 Super Aggregated: {dirs['super_aggregated']}")
    print(f"📂 GP Predictions: {dirs['gp_predictions']}")
    print(f"\n⚠️  Output format: PNG only (no PDF)")
    
    return output_dir


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate multi-mode benchmark visual evaluation report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect session by name
    python -m src.evaluation.benchmark_visual_reporter_multimode --session comprehensive_20260121
    
    # Explicit path
    python -m src.evaluation.benchmark_visual_reporter_multimode --json path/to/comprehensive_results.json
    
    # Custom options
    python -m src.evaluation.benchmark_visual_reporter_multimode --session my_session --topx 15 --max_atlas 30
    
    # Non-strict mode (continue even if requirements not fully met)
    python -m src.evaluation.benchmark_visual_reporter_multimode --session partial_run --no-strict
        """
    )
    
    parser.add_argument(
        "--session", "-s",
        type=str,
        default=None,
        help="Session name (or partial match) to process"
    )
    
    parser.add_argument(
        "--json", "-j",
        type=str,
        default=None,
        help="Explicit path to comprehensive_results.json"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: auto-generated in session folder)"
    )
    
    parser.add_argument(
        "--topx",
        type=int,
        default=10,
        help="Number of top configurations to analyze (default: 10)"
    )
    
    parser.add_argument(
        "--max_atlas_configs",
        type=int,
        default=25,
        help="Max GP configs for atlas to avoid explosion (default: 25)"
    )
    
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Continue even if not all requirements (sobol+lhs, simple+nested) are met"
    )
    
    args = parser.parse_args()
    
    if not args.session and not args.json:
        parser.error("Either --session or --json must be provided")
    
    try:
        output_dir = generate_benchmark_report_multimode(
            session_name=args.session or "",
            results_json_path=Path(args.json) if args.json else None,
            output_dir=Path(args.output) if args.output else None,
            strict=not args.no_strict,
            top_x=args.topx,
            max_atlas_configs=args.max_atlas_configs,
        )
        print(f"\n✅ Report generated successfully: {output_dir}")
        
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"\n❌ File Not Found: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
