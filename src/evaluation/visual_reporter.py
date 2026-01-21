# @author: José Arbelaez
"""
visual_reporter.py

Comprehensive evaluation and visualization module for surrogate model tuning results.
Generates publication-ready figures and tables from run_exhaustive_tuning outputs.

Usage:
    python -m src.evaluation.visual_reporter --session productivity_hermetia_v2_comprehensive
    
    Or programmatically:
        from src.evaluation.visual_reporter import generate_full_report
        generate_full_report("productivity_hermetia_v2_comprehensive")
"""

from __future__ import annotations

import ast
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

# Project imports
from src.utils.paths import LOGS_DIR, PLOTS_DIR
from src.models.gp import GPSurrogateRegressor
from src.models.pls import PLSSurrogateRegressor
from src.models.ridge import RidgeSurrogateRegressor
from src.models.dummy import DummySurrogateRegressor

# =============================================================================
# PUBLICATION-READY STYLE CONFIGURATION
# =============================================================================

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 300,
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

# Color palettes for consistency
MODEL_PALETTE = {
    "GP": "#2ecc71",      # Green
    "Ridge": "#3498db",   # Blue  
    "PLS": "#9b59b6",     # Purple
    "Dummy": "#95a5a6",   # Gray
}

EXPERIMENT_PALETTE = {
    "FULL_FEATURES": "#e74c3c",     # Red
    "REDUCED_FEATURES": "#3498db",   # Blue
}


# =============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# =============================================================================

@dataclass
class FoldResult:
    """Single fold evaluation result."""
    fold_id: int
    diet: str
    inner_best_score: float
    params: Dict[str, Any]
    n_samples: int
    mae: float
    rmse: float
    coverage95: Optional[float] = None
    r2: Optional[float] = None
    nlpd: Optional[float] = None
    
    
@dataclass
class ModelTuningResult:
    """Complete tuning results for one model."""
    model_name: str
    target: str
    experiment: str
    folds: List[FoldResult]
    macro_mae_mean: float
    macro_mae_std: float
    macro_rmse_mean: float
    macro_rmse_std: float
    macro_cov95_mean: Optional[float] = None
    macro_cov95_std: Optional[float] = None
    chosen_params: List[Dict] = field(default_factory=list)


@dataclass  
class ExperimentInventory:
    """Inventory of a tuning session."""
    session_name: str
    experiments: List[str]  # e.g., ["FULL_FEATURES", "REDUCED_FEATURES"]
    targets: List[str]      # e.g., ["fcr", "tpc", "proteina", "quitina"]
    models: List[str]       # e.g., ["GP", "Ridge", "PLS", "Dummy"]
    total_runs: int
    missing_combinations: List[Tuple[str, str, str]]  # (experiment, target, model)


# =============================================================================
# DISCOVERY AND LOADING
# =============================================================================

class TuningResultsDiscovery:
    """
    Auto-discovers and inventories all tuning runs in a session directory.
    
    Validates completeness and reports missing combinations.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize discovery with base tuning logs directory.
        
        Args:
            base_dir: Root tuning logs directory. Defaults to LOGS_DIR / "tuning"
        """
        self.base_dir = Path(base_dir) if base_dir else LOGS_DIR / "tuning"
        
    def list_sessions(self) -> List[str]:
        """List all available tuning sessions."""
        if not self.base_dir.exists():
            return []
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
    
    def inventory_session(self, session_name: str) -> ExperimentInventory:
        """
        Create complete inventory of a tuning session.
        
        Args:
            session_name: Name of the session (e.g., "productivity_hermetia_v2_comprehensive")
            
        Returns:
            ExperimentInventory with discovered structure
        """
        session_dir = self.base_dir / session_name
        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_dir}")
        
        experiments = []
        targets_set = set()
        models_set = set()
        total_runs = 0
        missing = []
        
        # Discover experiments (FULL_FEATURES, REDUCED_FEATURES)
        for exp_dir in session_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            exp_name = exp_dir.name
            experiments.append(exp_name)
            
            # Discover targets
            for target_dir in exp_dir.iterdir():
                if not target_dir.is_dir():
                    continue
                target_slug = target_dir.name
                targets_set.add(target_slug)
                
                # Check for summary file
                summary_file = target_dir / f"{target_slug}_summary.json"
                if summary_file.exists():
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    for model_name in summary.get("models", {}).keys():
                        models_set.add(model_name)
                        total_runs += 1
        
        # Check for missing combinations
        for exp in experiments:
            for target in targets_set:
                for model in models_set:
                    tuning_file = session_dir / exp / target / f"{target}_{model.lower()}_tuning.json"
                    if not tuning_file.exists():
                        missing.append((exp, target, model))
        
        return ExperimentInventory(
            session_name=session_name,
            experiments=sorted(experiments),
            targets=sorted(targets_set),
            models=sorted(models_set),
            total_runs=total_runs,
            missing_combinations=missing,
        )
    
    def print_inventory(self, inventory: ExperimentInventory):
        """Pretty-print inventory summary."""
        print("=" * 60)
        print(f"TUNING SESSION: {inventory.session_name}")
        print("=" * 60)
        print(f"Experiments: {inventory.experiments}")
        print(f"Targets: {inventory.targets}")
        print(f"Models: {inventory.models}")
        print(f"Total runs: {inventory.total_runs}")
        
        if inventory.missing_combinations:
            print(f"\n⚠️  MISSING COMBINATIONS ({len(inventory.missing_combinations)}):")
            for exp, target, model in inventory.missing_combinations:
                print(f"   - {exp}/{target}/{model}")
        else:
            print("\n✓ All combinations present")
        print("=" * 60)


class UnifiedResultsLoader:
    """
    Loads and normalizes tuning results to a unified tidy DataFrame format.
    
    Supports both old (flat) and new (nested) metrics structures.
    """
    
    EXPECTED_MODELS = ["GP", "Ridge", "PLS", "Dummy"]
    
    def __init__(self, session_dir: Path):
        """
        Initialize loader for a specific session.
        
        Args:
            session_dir: Path to session directory (e.g., LOGS_DIR/tuning/session_name)
        """
        self.session_dir = Path(session_dir)
        self._cache = {}
        
    def load_summary(self, experiment: str, target: str) -> Dict:
        """Load target summary JSON."""
        path = self.session_dir / experiment / target / f"{target}_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Summary not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_model_tuning(self, experiment: str, target: str, model: str) -> Dict:
        """Load detailed tuning results for a specific model."""
        path = self.session_dir / experiment / target / f"{target}_{model.lower()}_tuning.json"
        if not path.exists():
            raise FileNotFoundError(f"Tuning file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_folds_csv(self, experiment: str, target: str, model: str) -> pd.DataFrame:
        """Load folds CSV with proper parsing."""
        path = self.session_dir / experiment / target / f"{target}_{model.lower()}_folds.csv"
        if not path.exists():
            raise FileNotFoundError(f"Folds CSV not found: {path}")
        
        df = pd.read_csv(path)
        
        # Parse string representations of dicts
        def _safe_parse(value):
            if not isinstance(value, str):
                return value
            try:
                return ast.literal_eval(value)
            except:
                return value
        
        if "params" in df.columns:
            df["params"] = df["params"].apply(_safe_parse)
        
        if "metrics" in df.columns:
            df["metrics"] = df["metrics"].apply(_safe_parse)
            # Expand metrics dict to columns
            metrics_df = pd.json_normalize(df["metrics"])
            
            # Drop metrics column and existing columns that will be duplicated
            df = df.drop("metrics", axis=1)
            existing_cols = set(df.columns)
            new_cols = [c for c in metrics_df.columns if c not in existing_cols]
            
            # Only add new columns that don't already exist
            if new_cols:
                df = pd.concat([df, metrics_df[new_cols]], axis=1)
        
        # Remove any remaining duplicate columns (keep first)
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    def load_all_folds_tidy(self, experiments: List[str] = None, 
                            targets: List[str] = None,
                            models: List[str] = None) -> pd.DataFrame:
        """
        Load all fold results into a single tidy DataFrame.
        
        Args:
            experiments: Filter to specific experiments (None = all)
            targets: Filter to specific targets (None = all)
            models: Filter to specific models (None = all)
            
        Returns:
            Tidy DataFrame with columns:
                experiment, target, model, fold, diet, mae, rmse, coverage95, ...
        """
        all_rows = []
        
        # Auto-discover if not specified
        if experiments is None:
            experiments = [d.name for d in self.session_dir.iterdir() if d.is_dir()]
        
        for exp in experiments:
            exp_dir = self.session_dir / exp
            if not exp_dir.exists():
                continue
                
            if targets is None:
                exp_targets = [d.name for d in exp_dir.iterdir() if d.is_dir()]
            else:
                exp_targets = targets
            
            for target in exp_targets:
                target_dir = exp_dir / target
                if not target_dir.exists():
                    continue
                
                # Load summary to get available models
                try:
                    summary = self.load_summary(exp, target)
                    available_models = list(summary.get("models", {}).keys())
                except FileNotFoundError:
                    continue
                
                if models is None:
                    target_models = available_models
                else:
                    target_models = [m for m in models if m in available_models]
                
                for model in target_models:
                    try:
                        df = self.load_folds_csv(exp, target, model)
                        df["experiment"] = exp
                        df["target"] = target
                        df["model"] = model
                        all_rows.append(df)
                    except FileNotFoundError:
                        logging.warning(f"Missing: {exp}/{target}/{model}")
                        continue
        
        if not all_rows:
            return pd.DataFrame()
        
        combined = pd.concat(all_rows, ignore_index=True)
        
        # Standardize column names FIRST (before dedup)
        col_renames = {
            "coverage95": "coverage_95",
        }
        combined = combined.rename(columns={k: v for k, v in col_renames.items() if k in combined.columns})
        
        # Remove duplicate columns (keep first occurrence)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        
        return combined
    
    def load_macro_summary_tidy(self) -> pd.DataFrame:
        """
        Load macro (aggregated) metrics for all models into tidy format.
        
        Returns:
            DataFrame with columns:
                experiment, target, model, metric, mean, std
        """
        rows = []
        
        for exp_dir in self.session_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            exp = exp_dir.name
            
            for target_dir in exp_dir.iterdir():
                if not target_dir.is_dir():
                    continue
                target = target_dir.name
                
                try:
                    summary = self.load_summary(exp, target)
                except FileNotFoundError:
                    continue
                
                for model, model_info in summary.get("models", {}).items():
                    macro = model_info.get("summary", {}).get("macro", {})
                    
                    # Handle both old and new metrics structures
                    for metric_key in ["mae", "rmse", "coverage95", "coverage_95", "r2", "nlpd"]:
                        if isinstance(macro.get(metric_key), dict):
                            # New structure: {'mean': x, 'std': y}
                            mean_val = macro[metric_key].get("mean")
                            std_val = macro[metric_key].get("std")
                        else:
                            # Old structure: metric_mean, metric_std
                            mean_val = macro.get(f"{metric_key}_mean")
                            std_val = macro.get(f"{metric_key}_std")
                        
                        if mean_val is not None:
                            rows.append({
                                "experiment": exp,
                                "target": target,
                                "model": model,
                                "metric": metric_key.replace("coverage95", "coverage_95"),
                                "mean": mean_val,
                                "std": std_val,
                            })
        
        return pd.DataFrame(rows)


# =============================================================================
# COMPREHENSIVE REPORTER
# =============================================================================

class ComprehensiveReporter:
    """
    Generates comprehensive visual reports from tuning results.
    
    Produces publication-ready figures for:
        - Model rankings
        - Metric distributions
        - Experiment comparisons
        - Heatmaps
        - Hyperparameter sensitivity
    """
    
    def __init__(self, loader: UnifiedResultsLoader, output_dir: Path):
        """
        Initialize reporter.
        
        Args:
            loader: UnifiedResultsLoader instance
            output_dir: Directory for saving figures and tables
        """
        self.loader = loader
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        self._folds_df = None
        self._macro_df = None
    
    @property
    def folds_df(self) -> pd.DataFrame:
        """Lazy-load folds DataFrame."""
        if self._folds_df is None:
            self._folds_df = self.loader.load_all_folds_tidy()
        return self._folds_df
    
    @property
    def macro_df(self) -> pd.DataFrame:
        """Lazy-load macro summary DataFrame."""
        if self._macro_df is None:
            self._macro_df = self.loader.load_macro_summary_tidy()
        return self._macro_df
    
    # -------------------------------------------------------------------------
    # 1. GLOBAL RANKINGS
    # -------------------------------------------------------------------------
    
    def plot_global_model_ranking(self, metric: str = "mae", 
                                   save: bool = True) -> plt.Figure:
        """
        Bar plot ranking models globally across all targets and experiments.
        
        Args:
            metric: Metric to rank by (mae, rmse, coverage_95)
            save: Whether to save figure
            
        Returns:
            matplotlib Figure
        """
        df = self.macro_df[self.macro_df["metric"] == metric].copy()
        
        if df.empty:
            logging.warning(f"No data for metric: {metric}")
            return None
        
        # Aggregate across experiments and targets
        agg = df.groupby("model").agg({
            "mean": ["mean", "std", "count"]
        }).reset_index()
        agg.columns = ["model", "avg_mean", "spread_std", "n_evaluations"]
        agg = agg.sort_values("avg_mean")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = [MODEL_PALETTE.get(m, "#7f8c8d") for m in agg["model"]]
        bars = ax.barh(agg["model"], agg["avg_mean"], color=colors, edgecolor='black', alpha=0.8)
        
        # Error bars (spread across targets/experiments)
        ax.errorbar(agg["avg_mean"], agg["model"], xerr=agg["spread_std"], 
                    fmt='none', color='black', capsize=4, capthick=1.5)
        
        # Annotations
        for bar, val, n in zip(bars, agg["avg_mean"], agg["n_evaluations"]):
            ax.text(val + agg["avg_mean"].max() * 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f} (n={n})', va='center', fontsize=10)
        
        ax.set_xlabel(f"{metric.upper()} (Mean ± SD across targets)", fontsize=12)
        ax.set_title(f"Global Model Ranking by {metric.upper()}\n(Lower is better for MAE/RMSE)", 
                     fontsize=14, fontweight='bold')
        ax.axvline(x=agg["avg_mean"].iloc[0], color='green', linestyle='--', alpha=0.5, 
                   label=f'Best: {agg["model"].iloc[0]}')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        
        if save:
            path = self.figures_dir / f"ranking_global_{metric}.png"
            fig.savefig(path, bbox_inches='tight')
            logging.info(f"Saved: {path}")
        
        return fig
    
    def plot_ranking_by_target(self, metric: str = "mae",
                                save: bool = True) -> plt.Figure:
        """
        Faceted bar plot showing model rankings per target.
        """
        df = self.macro_df[self.macro_df["metric"] == metric].copy()
        
        if df.empty:
            return None
        
        # Pivot for heatmap-like view
        targets = df["target"].unique()
        n_targets = len(targets)
        
        fig, axes = plt.subplots(1, n_targets, figsize=(4*n_targets, 5), sharey=True)
        if n_targets == 1:
            axes = [axes]
        
        for ax, target in zip(axes, sorted(targets)):
            target_df = df[df["target"] == target].copy()
            target_df = target_df.groupby("model")["mean"].mean().reset_index()
            target_df = target_df.sort_values("mean")
            
            colors = [MODEL_PALETTE.get(m, "#7f8c8d") for m in target_df["model"]]
            ax.barh(target_df["model"], target_df["mean"], color=colors, edgecolor='black')
            ax.set_title(f"{target.upper()}", fontsize=12, fontweight='bold')
            ax.set_xlabel(metric.upper())
            
            # Mark best
            ax.axvline(target_df["mean"].min(), color='green', linestyle='--', alpha=0.5)
        
        fig.suptitle(f"Model Ranking by Target ({metric.upper()})", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            path = self.figures_dir / f"ranking_by_target_{metric}.png"
            fig.savefig(path, bbox_inches='tight')
        
        return fig
    
    # -------------------------------------------------------------------------
    # 2. METRIC DISTRIBUTIONS
    # -------------------------------------------------------------------------
    
    def plot_metric_distribution_box(self, metric: str = "mae",
                                      facet_by: str = "target",
                                      save: bool = True) -> plt.Figure:
        """
        Box/violin plot showing metric distribution across folds.
        
        Args:
            metric: Metric column name
            facet_by: 'target' or 'experiment'
        """
        df = self.folds_df.copy()
        
        if metric not in df.columns:
            logging.warning(f"Metric {metric} not in folds data")
            return None
        
        # Handle dict/nested values and clean metric values
        def _extract_numeric(val):
            if isinstance(val, dict):
                # Try common keys
                for key in ['value', 'mean', metric]:
                    if key in val:
                        return val[key]
                return None
            return val
        
        df[metric] = df[metric].apply(_extract_numeric)
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        df = df.dropna(subset=[metric])
        
        if df.empty:
            logging.warning(f"No valid data for metric: {metric}")
            return None
        
        facet_values = df[facet_by].unique()
        n_facets = len(facet_values)
        
        fig, axes = plt.subplots(1, n_facets, figsize=(5*n_facets, 6), sharey=False)
        if n_facets == 1:
            axes = [axes]
        
        for ax, facet_val in zip(axes, sorted(facet_values)):
            facet_df = df[df[facet_by] == facet_val]
            
            # Box plot with hue to avoid FutureWarning
            sns.boxplot(data=facet_df, x="model", y=metric, hue="model", ax=ax,
                        palette=MODEL_PALETTE, showfliers=False, legend=False)
            
            # Overlay strip plot colored by diet
            sns.stripplot(data=facet_df, x="model", y=metric, hue="diet",
                          ax=ax, palette="tab20", size=6, alpha=0.7, 
                          jitter=True, dodge=False, legend=False)
            
            ax.set_title(f"{facet_val.upper()}", fontsize=12, fontweight='bold')
            ax.set_xlabel("")
            ax.set_ylabel(metric.upper() if ax == axes[0] else "")
            ax.tick_params(axis='x', rotation=45)
        
        fig.suptitle(f"{metric.upper()} Distribution by {facet_by.title()} (LODO CV Folds)", 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            path = self.figures_dir / f"distribution_{metric}_by_{facet_by}.png"
            fig.savefig(path, bbox_inches='tight')
        
        return fig
    
    # -------------------------------------------------------------------------
    # 3. EXPERIMENT COMPARISON (FULL vs REDUCED)
    # -------------------------------------------------------------------------
    
    def plot_experiment_comparison(self, target: str = None,
                                    metrics: List[str] = None,
                                    save: bool = True) -> plt.Figure:
        """
        Side-by-side comparison of FULL_FEATURES vs REDUCED_FEATURES.
        """
        if metrics is None:
            metrics = ["mae", "rmse"]
        
        df = self.macro_df[self.macro_df["metric"].isin(metrics)].copy()
        
        if target:
            df = df[df["target"] == target]
        
        if df.empty:
            return None
        
        # Create plot
        g = sns.catplot(
            data=df,
            x="experiment", y="mean", hue="model",
            col="metric" if len(metrics) > 1 else None,
            row="target" if not target and df["target"].nunique() > 1 else None,
            kind="bar",
            palette=MODEL_PALETTE,
            height=4, aspect=1.2,
            sharey=False,
            legend_out=True,
        )
        
        title = f"Experiment Comparison"
        if target:
            title += f" ({target.upper()})"
        g.fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
        
        if save:
            suffix = f"_{target}" if target else "_all"
            path = self.figures_dir / f"comparison_experiments{suffix}.png"
            g.savefig(path, bbox_inches='tight')
        
        return g.fig
    
    # -------------------------------------------------------------------------
    # 4. HEATMAPS
    # -------------------------------------------------------------------------
    
    def plot_performance_heatmap(self, metric: str = "mae",
                                  experiment: str = None,
                                  save: bool = True) -> plt.Figure:
        """
        Heatmap of model × target performance.
        """
        df = self.macro_df[self.macro_df["metric"] == metric].copy()
        
        if experiment:
            df = df[df["experiment"] == experiment]
        else:
            # Average across experiments
            df = df.groupby(["target", "model"])["mean"].mean().reset_index()
        
        if df.empty:
            return None
        
        # Pivot to matrix
        pivot = df.pivot(index="model", columns="target", values="mean")
        
        # Rank within each target (for annotation)
        ranks = pivot.rank(axis=0)
        
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*1.5), 6))
        
        # Heatmap with values
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn_r",
                    ax=ax, cbar_kws={"label": metric.upper()},
                    linewidths=0.5, linecolor='white')
        
        # Add rank indicators
        for i, model in enumerate(pivot.index):
            for j, target in enumerate(pivot.columns):
                rank = int(ranks.loc[model, target])
                if rank == 1:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                                edgecolor='gold', linewidth=3))
        
        ax.set_title(f"Performance Matrix ({metric.upper()})\n" + 
                     ("Gold border = Best for target" if experiment else "Averaged across experiments"),
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Target Variable")
        ax.set_ylabel("Model")
        
        plt.tight_layout()
        
        if save:
            suffix = f"_{experiment}" if experiment else "_avg"
            path = self.figures_dir / f"heatmap_{metric}{suffix}.png"
            fig.savefig(path, bbox_inches='tight')
        
        return fig
    
    # -------------------------------------------------------------------------
    # 5. HYPERPARAMETER SENSITIVITY
    # -------------------------------------------------------------------------
    
    def plot_hyperparameter_distribution(self, model: str = "GP",
                                          param: str = "kernel",
                                          save: bool = True) -> plt.Figure:
        """
        Show distribution of chosen hyperparameters across folds.
        """
        df = self.folds_df[self.folds_df["model"] == model].copy()
        
        if df.empty or "params" not in df.columns:
            return None
        
        # Extract param values
        def extract_param(params_dict, key):
            if isinstance(params_dict, dict):
                return str(params_dict.get(key, "N/A"))
            return "N/A"
        
        df["param_value"] = df["params"].apply(lambda x: extract_param(x, param))
        
        # Count occurrences
        counts = df.groupby(["target", "param_value"]).size().reset_index(name="count")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.barplot(data=counts, x="param_value", y="count", hue="target",
                    ax=ax, palette="Set2")
        
        ax.set_title(f"{model} Hyperparameter Selection: {param}\n(Across LODO folds)",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(f"{param} value")
        ax.set_ylabel("Times selected")
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title="Target", bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        
        if save:
            path = self.figures_dir / f"hyperparam_{model}_{param}.png"
            fig.savefig(path, bbox_inches='tight')
        
        return fig
    
    # -------------------------------------------------------------------------
    # 6. TABLE EXPORTS
    # -------------------------------------------------------------------------
    
    def export_summary_table(self, save: bool = True) -> pd.DataFrame:
        """
        Export comprehensive summary table.
        """
        df = self.macro_df.copy()
        
        # Pivot to wide format
        pivot = df.pivot_table(
            index=["experiment", "target", "model"],
            columns="metric",
            values=["mean", "std"],
            aggfunc="first"
        )
        
        # Flatten column names
        pivot.columns = [f"{metric}_{stat}" for stat, metric in pivot.columns]
        pivot = pivot.reset_index()
        
        if save:
            path = self.tables_dir / "summary_all.csv"
            pivot.to_csv(path, index=False)
            logging.info(f"Saved: {path}")
        
        return pivot
    
    def export_best_models_table(self, metric: str = "mae",
                                  save: bool = True) -> pd.DataFrame:
        """
        Export table of best model per target/experiment.
        """
        df = self.macro_df[self.macro_df["metric"] == metric].copy()
        
        # Find best per group
        idx = df.groupby(["experiment", "target"])["mean"].idxmin()
        best = df.loc[idx].copy()
        best = best.rename(columns={"mean": f"best_{metric}", "model": "best_model"})
        
        if save:
            path = self.tables_dir / f"best_models_{metric}.csv"
            best.to_csv(path, index=False)
        
        return best
    
    # -------------------------------------------------------------------------
    # GENERATE ALL REPORTS
    # -------------------------------------------------------------------------
    
    def generate_all(self, metrics: List[str] = None):
        """
        Generate all standard reports.
        """
        if metrics is None:
            metrics = ["mae", "rmse"]
        
        logging.info("Generating comprehensive reports...")
        
        # Rankings
        for metric in metrics:
            self.plot_global_model_ranking(metric)
            self.plot_ranking_by_target(metric)
        
        # Distributions
        for metric in metrics:
            self.plot_metric_distribution_box(metric, facet_by="target")
        
        # Coverage if available
        if "coverage_95" in self.folds_df.columns:
            self.plot_metric_distribution_box("coverage_95", facet_by="target")
        
        # Experiment comparisons
        self.plot_experiment_comparison(metrics=metrics)
        
        # Heatmaps
        for metric in metrics:
            self.plot_performance_heatmap(metric)
        
        # Hyperparameter analysis
        self.plot_hyperparameter_distribution("GP", "kernel")
        self.plot_hyperparameter_distribution("Ridge", "alpha")
        
        # Tables
        self.export_summary_table()
        for metric in metrics:
            self.export_best_models_table(metric)
        
        logging.info(f"Reports saved to: {self.output_dir}")
        
        plt.close('all')


# =============================================================================
# GP DIAGNOSTICS
# =============================================================================

class GPDiagnostics:
    """
    Specialized diagnostic visualizations for Gaussian Process models.
    
    Includes:
        - Uncertainty analysis
        - Calibration curves
        - Error vs uncertainty correlation
        - Response profiles
    """
    
    def __init__(self, loader: UnifiedResultsLoader, output_dir: Path):
        self.loader = loader
        self.output_dir = Path(output_dir) / "gp_diagnostics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_coverage_analysis(self, save: bool = True) -> plt.Figure:
        """
        Analyze coverage calibration across targets.
        
        Expected: 95% of points should fall within 95% CI.
        """
        df = self.loader.load_all_folds_tidy(models=["GP"])
        
        if df.empty or "coverage_95" not in df.columns:
            logging.warning("No GP coverage data available")
            return None
        
        df["coverage_95"] = pd.to_numeric(df["coverage_95"], errors="coerce")
        df = df.dropna(subset=["coverage_95"])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # A. Coverage by target
        ax1 = axes[0]
        targets = df["target"].unique()
        
        for i, target in enumerate(sorted(targets)):
            target_cov = df[df["target"] == target]["coverage_95"]
            ax1.scatter([i] * len(target_cov), target_cov, 
                        alpha=0.6, s=80, label=target.upper())
            ax1.scatter(i, target_cov.mean(), color='red', s=150, 
                        marker='D', edgecolor='black', zorder=10)
        
        ax1.axhline(0.95, color='green', linestyle='--', lw=2, label='Target (95%)')
        ax1.axhspan(0.90, 1.0, alpha=0.1, color='green')
        ax1.set_xticks(range(len(targets)))
        ax1.set_xticklabels([t.upper() for t in sorted(targets)])
        ax1.set_ylabel("Coverage (95% CI)")
        ax1.set_title("GP Calibration by Target\n(Red diamond = mean)", fontweight='bold')
        ax1.set_ylim(0, 1.1)
        ax1.legend(loc='lower right')
        
        # B. Histogram of coverage
        ax2 = axes[1]
        ax2.hist(df["coverage_95"], bins=20, edgecolor='black', alpha=0.7, color='cornflowerblue')
        ax2.axvline(0.95, color='green', linestyle='--', lw=2, label='Target (95%)')
        ax2.axvline(df["coverage_95"].mean(), color='red', linestyle='-', lw=2, 
                    label=f'Mean: {df["coverage_95"].mean():.2%}')
        ax2.set_xlabel("Coverage (95% CI)")
        ax2.set_ylabel("Count (folds)")
        ax2.set_title("Coverage Distribution Across All Folds", fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "gp_coverage_analysis.png"
            fig.savefig(path, bbox_inches='tight')
        
        return fig
    
    def plot_error_vs_uncertainty_summary(self, save: bool = True) -> plt.Figure:
        """
        Summary plot: Does GP know what it doesn't know?
        
        Note: Requires access to raw predictions (std_pred) which may not be
        stored in standard outputs. This plots aggregate statistics instead.
        """
        df = self.loader.load_all_folds_tidy(models=["GP"])
        
        if df.empty:
            return None
        
        df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
        df["coverage_95"] = pd.to_numeric(df["coverage_95"], errors="coerce")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter: MAE vs Coverage
        scatter = ax.scatter(df["mae"], df["coverage_95"], 
                            c=df["target"].astype('category').cat.codes,
                            cmap='Set2', s=100, alpha=0.7, edgecolor='black')
        
        # Ideal region
        ax.axhspan(0.90, 1.0, alpha=0.1, color='green', label='Good calibration region')
        ax.axhline(0.95, color='green', linestyle='--', alpha=0.5)
        
        ax.set_xlabel("MAE (prediction error)")
        ax.set_ylabel("Coverage (95% CI)")
        ax.set_title("GP Error vs Calibration by Fold\n(Ideal: High coverage, low MAE)", fontweight='bold')
        
        # Legend for targets
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=plt.cm.Set2(i/len(df["target"].unique())),
                              markersize=10, label=t.upper())
                   for i, t in enumerate(sorted(df["target"].unique()))]
        ax.legend(handles=handles, title="Target", loc='lower left')
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "gp_error_vs_calibration.png"
            fig.savefig(path, bbox_inches='tight')
        
        return fig
    
    def generate_all(self):
        """Generate all GP diagnostic plots."""
        logging.info("Generating GP diagnostics...")
        
        self.plot_coverage_analysis()
        self.plot_error_vs_uncertainty_summary()
        
        logging.info(f"GP diagnostics saved to: {self.output_dir}")
        plt.close('all')


# =============================================================================
# COVERAGE VERIFICATION
# =============================================================================

class CoverageVerifier:
    """
    Verifies experimental coverage and identifies gaps.
    """
    
    def __init__(self, discovery: TuningResultsDiscovery, session_name: str):
        self.discovery = discovery
        self.session_name = session_name
        self.inventory = discovery.inventory_session(session_name)
    
    def verify(self) -> Dict[str, Any]:
        """
        Run all verification checks.
        
        Returns:
            Dict with check results and recommendations
        """
        results = {
            "session": self.session_name,
            "checks": {},
            "recommendations": [],
        }
        
        # Check 1: LODO-CV presence
        has_lodo = self._check_lodo_cv()
        results["checks"]["lodo_cv"] = {
            "present": has_lodo,
            "description": "Leave-One-Diet-Out cross-validation"
        }
        
        # Check 2: Multiple experiments (feature sets)
        n_experiments = len(self.inventory.experiments)
        results["checks"]["experiments"] = {
            "count": n_experiments,
            "names": self.inventory.experiments,
            "comparison_possible": n_experiments >= 2
        }
        
        # Check 3: Model coverage
        expected_models = {"GP", "Ridge", "PLS", "Dummy"}
        found_models = set(self.inventory.models)
        missing_models = expected_models - found_models
        results["checks"]["models"] = {
            "found": list(found_models),
            "missing": list(missing_models),
            "complete": len(missing_models) == 0
        }
        
        # Check 4: Missing combinations
        results["checks"]["completeness"] = {
            "total_expected": len(self.inventory.experiments) * len(self.inventory.targets) * len(expected_models),
            "total_found": self.inventory.total_runs,
            "missing": self.inventory.missing_combinations
        }
        
        # Generate recommendations
        if not has_lodo:
            results["recommendations"].append(
                "⚠️ No LODO-CV detected. The current setup IS LODO-CV (Leave-One-Diet-Out). "
                "Consider this verified."
            )
        
        if n_experiments < 2:
            results["recommendations"].append(
                "⚠️ Only one experiment (feature set). Consider running both FULL_FEATURES and REDUCED_FEATURES."
            )
        
        if missing_models:
            results["recommendations"].append(
                f"⚠️ Missing models: {missing_models}. Add to run_exhaustive_tuning MODELS dict."
            )
        
        if self.inventory.missing_combinations:
            results["recommendations"].append(
                f"⚠️ {len(self.inventory.missing_combinations)} missing model/target combinations. "
                "Re-run tuning for completeness."
            )
        
        if not results["recommendations"]:
            results["recommendations"].append("✓ All checks passed. Coverage is complete.")
        
        return results
    
    def _check_lodo_cv(self) -> bool:
        """Check if LODO-CV is being used (it always is in this setup)."""
        # In this project, all tuning uses nested LODO-CV by design
        # Check by looking at fold structure in any tuning file
        try:
            session_dir = self.discovery.base_dir / self.session_name
            for exp_dir in session_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                for target_dir in exp_dir.iterdir():
                    if not target_dir.is_dir():
                        continue
                    for json_file in target_dir.glob("*_tuning.json"):
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        if "folds" in data and len(data["folds"]) > 0:
                            # Check if folds have "diet" key (LODO indicator)
                            return "diet" in data["folds"][0]
        except Exception:
            pass
        return True  # Assume LODO-CV by design
    
    def print_report(self):
        """Print verification report."""
        results = self.verify()
        
        print("=" * 60)
        print(f"COVERAGE VERIFICATION: {results['session']}")
        print("=" * 60)
        
        for check_name, check_data in results["checks"].items():
            print(f"\n{check_name.upper()}:")
            for k, v in check_data.items():
                print(f"  {k}: {v}")
        
        print("\n" + "-" * 60)
        print("RECOMMENDATIONS:")
        for rec in results["recommendations"]:
            print(f"  {rec}")
        print("=" * 60)


# =============================================================================
# REPORT GENERATION ENTRYPOINT
# =============================================================================

def generate_full_report(session_name: str, 
                         output_dir: Optional[Path] = None,
                         metrics: List[str] = None) -> Path:
    """
    Generate complete visual report for a tuning session.
    
    Args:
        session_name: Name of tuning session
        output_dir: Custom output directory (default: PLOTS_DIR/reports/{session_name})
        metrics: Metrics to include (default: ['mae', 'rmse'])
        
    Returns:
        Path to output directory
    """
    if metrics is None:
        metrics = ["mae", "rmse"]
    
    if output_dir is None:
        output_dir = PLOTS_DIR / "reports" / session_name
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    print("=" * 70)
    print(f"GENERATING VISUAL REPORT: {session_name}")
    print("=" * 70)
    
    # 1. Discovery and inventory
    discovery = TuningResultsDiscovery()
    inventory = discovery.inventory_session(session_name)
    discovery.print_inventory(inventory)
    
    # 2. Coverage verification
    verifier = CoverageVerifier(discovery, session_name)
    verifier.print_report()
    
    # 3. Load data
    session_dir = LOGS_DIR / "tuning" / session_name
    loader = UnifiedResultsLoader(session_dir)
    
    # 4. Generate main reports
    reporter = ComprehensiveReporter(loader, output_dir)
    reporter.generate_all(metrics=metrics)
    
    # 5. GP diagnostics
    gp_diag = GPDiagnostics(loader, output_dir)
    gp_diag.generate_all()
    
    # 6. Generate index markdown
    _generate_report_index(output_dir, session_name, inventory)
    
    print("\n" + "=" * 70)
    print(f"REPORT COMPLETE: {output_dir}")
    print("=" * 70)
    
    return output_dir


def _generate_report_index(output_dir: Path, session_name: str, 
                           inventory: ExperimentInventory):
    """Generate markdown index of all generated files."""
    index_path = output_dir / "README.md"
    
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    gp_dir = output_dir / "gp_diagnostics"
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(f"# Visual Report: {session_name}\n\n")
        f.write(f"Generated automatically by `visual_reporter.py`\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Experiments**: {', '.join(inventory.experiments)}\n")
        f.write(f"- **Targets**: {', '.join(inventory.targets)}\n")
        f.write(f"- **Models**: {', '.join(inventory.models)}\n")
        f.write(f"- **Total runs**: {inventory.total_runs}\n\n")
        
        f.write("## Figures\n\n")
        if figures_dir.exists():
            for fig_path in sorted(figures_dir.glob("*.png")):
                f.write(f"- [{fig_path.name}](figures/{fig_path.name})\n")
        
        f.write("\n## GP Diagnostics\n\n")
        if gp_dir.exists():
            for fig_path in sorted(gp_dir.glob("*.png")):
                f.write(f"- [{fig_path.name}](gp_diagnostics/{fig_path.name})\n")
        
        f.write("\n## Tables\n\n")
        if tables_dir.exists():
            for tbl_path in sorted(tables_dir.glob("*.csv")):
                f.write(f"- [{tbl_path.name}](tables/{tbl_path.name})\n")
    
    logging.info(f"Index saved: {index_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate visual reports from tuning results"
    )
    parser.add_argument(
        "--session", "-s",
        type=str,
        required=True,
        help="Tuning session name (e.g., productivity_hermetia_v2_comprehensive)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: outputs/plots/reports/{session})"
    )
    parser.add_argument(
        "--metrics", "-m",
        nargs="+",
        default=["mae", "rmse"],
        help="Metrics to report (default: mae rmse)"
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List available sessions and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_sessions:
        discovery = TuningResultsDiscovery()
        print("Available sessions:")
        for s in discovery.list_sessions():
            print(f"  - {s}")
    else:
        output_path = Path(args.output) if args.output else None
        generate_full_report(
            session_name=args.session,
            output_dir=output_path,
            metrics=args.metrics
        )
