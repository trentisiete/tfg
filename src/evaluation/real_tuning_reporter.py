# @author: José Arbelaez & Copilot
"""
Real Tuning Visual Evaluation Pack.

Comprehensive analysis and visualization module for real-world surrogate model
tuning results. Generates publication-ready figures, tables, and reports.

Features:
    - Master table normalization (tidy format)
    - FULL vs REDUCED feature comparison
    - Model comparison (heatmaps, rankings, WTL)
    - GP Deep Dive (kernels, uncertainty, calibration)
    - GP uncertainty visualizations for real data

Usage:
    python -m src.evaluation.real_tuning_reporter --session productivity_hermetia_v2_comprehensive
    
    Or programmatically:
        from src.evaluation.real_tuning_reporter import generate_real_tuning_report
        generate_real_tuning_report("productivity_hermetia_v2_comprehensive")
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
from collections import Counter
from datetime import datetime
import hashlib

# Use non-interactive backend for batch generation
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Project imports
from src.utils.paths import LOGS_DIR, PLOTS_DIR

# =============================================================================
# CONFIGURATION & STYLE
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

# Color palettes
MODEL_FAMILY_PALETTE = {
    "Dummy": "#95a5a6",
    "Ridge": "#3498db",
    "PLS": "#9b59b6",
    "GP": "#2ecc71",
}

GP_KERNEL_PALETTE = {
    "Matern32": "#27ae60",
    "Matern52": "#2ecc71",
    "RBF": "#1abc9c",
    "DotProduct": "#16a085",
    "RationalQuadratic": "#48c9b0",
}

FEATURE_MODE_PALETTE = {
    "FULL_FEATURES": "#e74c3c",
    "REDUCED_FEATURES": "#3498db",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TuningInventory:
    """Inventory of available tuning results."""
    session_name: str
    feature_modes: List[str]
    targets: List[str]
    models: List[str]
    n_folds: int = 0
    total_configs: int = 0


@dataclass
class DataQualityReport:
    """Data quality assessment."""
    missing_files: List[str] = field(default_factory=list)
    missing_metrics: Dict[str, List[str]] = field(default_factory=dict)
    inconsistencies: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        lines = ["# Data Quality Report", ""]
        
        if self.missing_files:
            lines.append("## Missing Files")
            for f in self.missing_files:
                lines.append(f"- {f}")
            lines.append("")
        
        if self.missing_metrics:
            lines.append("## Missing Metrics by Model")
            for model, metrics in self.missing_metrics.items():
                lines.append(f"- **{model}**: {', '.join(metrics)}")
            lines.append("")
        
        if self.inconsistencies:
            lines.append("## Inconsistencies")
            for i in self.inconsistencies:
                lines.append(f"- {i}")
            lines.append("")
        
        if self.recommendations:
            lines.append("## Recommendations for Pipeline")
            for r in self.recommendations:
                lines.append(f"- {r}")
            lines.append("")
        
        return '\n'.join(lines)


# =============================================================================
# MASTER TABLE BUILDER
# =============================================================================

class MasterTableBuilder:
    """
    Builds normalized tidy master table from tuning results.
    
    Columns:
        - feature_mode: FULL_FEATURES / REDUCED_FEATURES
        - target: fcr, proteina, quitina, tpc
        - split_type: LODO (Leave-One-Diet-Out)
        - fold_id: 0, 1, 2, ...
        - group_id: diet name (Control, Hoja15, etc.)
        - model_family: GP, Ridge, PLS, Dummy
        - model_variant: kernel/params hash
        - params: JSON string of params
        - Metrics: mae, rmse, r2, coverage95, nlpd, etc.
    """
    
    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.quality_report = DataQualityReport()
        
    def build(self) -> pd.DataFrame:
        """Build master table from all available results."""
        rows = []
        
        for mode_dir in self.session_dir.iterdir():
            if not mode_dir.is_dir():
                continue
            
            feature_mode = mode_dir.name
            if feature_mode not in ['FULL_FEATURES', 'REDUCED_FEATURES']:
                continue
            
            for target_dir in mode_dir.iterdir():
                if not target_dir.is_dir():
                    continue
                
                target = target_dir.name
                
                # Process each model's tuning JSON
                for json_file in target_dir.glob('*_tuning.json'):
                    model_name = json_file.stem.replace(f'{target}_', '').replace('_tuning', '')
                    model_family = self._get_model_family(model_name)
                    
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        for fold_data in data.get('folds', []):
                            row = self._process_fold(
                                fold_data, 
                                feature_mode, 
                                target, 
                                model_family
                            )
                            rows.append(row)
                            
                    except Exception as e:
                        self.quality_report.missing_files.append(f"{json_file}: {e}")
        
        df = pd.DataFrame(rows)
        
        # Add computed columns
        if not df.empty:
            df = self._add_computed_columns(df)
        
        return df
    
    def _get_model_family(self, model_name: str) -> str:
        """Map model name to family."""
        mapping = {
            'gp': 'GP',
            'ridge': 'Ridge',
            'pls': 'PLS',
            'dummy': 'Dummy',
        }
        return mapping.get(model_name.lower(), model_name)
    
    def _process_fold(self, fold_data: Dict, feature_mode: str, 
                      target: str, model_family: str) -> Dict:
        """Process a single fold into a row."""
        params = fold_data.get('params', {})
        metrics = fold_data.get('metrics', {})
        
        # Extract model variant
        model_variant = self._extract_model_variant(model_family, params)
        
        # Build row
        row = {
            'feature_mode': feature_mode,
            'target': target,
            'split_type': 'LODO',  # Leave-One-Diet-Out
            'fold_id': fold_data.get('fold', -1),
            'group_id': fold_data.get('diet', 'unknown'),
            'model_family': model_family,
            'model_variant': model_variant,
            'params': json.dumps(params, default=str),
            'inner_best_score': fold_data.get('inner_best_score'),
            # Standard metrics
            'n_samples': metrics.get('n_samples'),
            'mae': metrics.get('mae'),
            'rmse': metrics.get('rmse'),
            'r2': metrics.get('r2'),
            'max_error': metrics.get('max_error'),
            # Probabilistic metrics (GP only)
            'coverage95': metrics.get('coverage95'),
            'coverage90': metrics.get('coverage90'),
            'coverage50': metrics.get('coverage50'),
            'nlpd': metrics.get('nlpd'),
            'calibration_error_95': metrics.get('calibration_error_95'),
            'sharpness': metrics.get('sharpness'),
            'mean_interval_width_95': metrics.get('mean_interval_width_95'),
            # Timing
            'fit_time': metrics.get('fit_time'),
            'predict_time': metrics.get('predict_time'),
        }
        
        return row
    
    def _extract_model_variant(self, model_family: str, params: Dict) -> str:
        """Extract human-readable model variant identifier."""
        if model_family == 'GP':
            kernel_str = str(params.get('kernel', 'default'))
            kernel_type = self._parse_kernel_type(kernel_str)
            alpha = params.get('alpha', 'default')
            return f"GP_{kernel_type}_a{alpha}"
        
        elif model_family == 'Ridge':
            alpha = params.get('alpha', 'default')
            return f"Ridge_a{alpha}"
        
        elif model_family == 'PLS':
            n_components = params.get('n_components', 'default')
            return f"PLS_n{n_components}"
        
        elif model_family == 'Dummy':
            strategy = params.get('strategy', 'mean')
            return f"Dummy_{strategy}"
        
        return f"{model_family}_default"
    
    def _parse_kernel_type(self, kernel_str: str) -> str:
        """Parse kernel type from string representation."""
        if 'Matern' in kernel_str:
            if 'nu=2.5' in kernel_str:
                return 'Matern52'
            elif 'nu=1.5' in kernel_str:
                return 'Matern32'
            elif 'nu=0.5' in kernel_str:
                return 'Matern12'
            return 'Matern'
        elif 'RBF' in kernel_str:
            return 'RBF'
        elif 'DotProduct' in kernel_str:
            return 'DotProduct'
        elif 'RationalQuadratic' in kernel_str:
            return 'RQ'
        elif 'Periodic' in kernel_str:
            return 'Periodic'
        return 'Unknown'
    
    def _add_computed_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed/derived columns."""
        # Kernel type for GP
        df['kernel_type'] = df.apply(
            lambda r: self._parse_kernel_type(r['params']) if r['model_family'] == 'GP' else None,
            axis=1
        )
        
        return df
    
    def get_quality_report(self) -> DataQualityReport:
        """Get data quality report."""
        return self.quality_report


# =============================================================================
# FULL VS REDUCED COMPARISON
# =============================================================================

class FullVsReducedAnalyzer:
    """
    Analyzes differences between FULL_FEATURES and REDUCED_FEATURES experiments.
    """
    
    def __init__(self, master_df: pd.DataFrame, output_dir: Path):
        self.df = master_df
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / 'tables'
        self.figures_dir = self.output_dir / 'figures' / 'full_vs_reduced'
        
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self):
        """Generate all FULL vs REDUCED comparisons."""
        self._generate_global_comparison_table()
        self._generate_by_target_comparison_table()
        self._generate_delta_plots()
        self._generate_heatmaps()
        self._generate_ranking_comparison()
    
    def _generate_global_comparison_table(self):
        """Generate global comparison table."""
        metrics = ['mae', 'rmse', 'coverage95', 'nlpd']
        
        rows = []
        for model in self.df['model_family'].unique():
            for mode in ['FULL_FEATURES', 'REDUCED_FEATURES']:
                subset = self.df[
                    (self.df['model_family'] == model) & 
                    (self.df['feature_mode'] == mode)
                ]
                
                for metric in metrics:
                    if metric in subset.columns:
                        values = subset[metric].dropna()
                        if len(values) > 0:
                            rows.append({
                                'model_family': model,
                                'feature_mode': mode,
                                'metric': metric,
                                'mean': values.mean(),
                                'std': values.std(),
                                'median': values.median(),
                                'iqr': values.quantile(0.75) - values.quantile(0.25),
                                'n': len(values),
                            })
        
        df_global = pd.DataFrame(rows)
        df_global.to_csv(self.tables_dir / 'full_vs_reduced_global.csv', index=False)
        
        # Pivot for easier reading
        if not df_global.empty:
            pivot = df_global.pivot_table(
                index=['model_family', 'metric'],
                columns='feature_mode',
                values='mean'
            )
            if 'FULL_FEATURES' in pivot.columns and 'REDUCED_FEATURES' in pivot.columns:
                pivot['delta'] = pivot['FULL_FEATURES'] - pivot['REDUCED_FEATURES']
                pivot.to_csv(self.tables_dir / 'full_vs_reduced_pivot.csv')
        
        return df_global
    
    def _generate_by_target_comparison_table(self):
        """Generate per-target comparison table."""
        metrics = ['mae', 'rmse', 'coverage95']
        
        rows = []
        for target in self.df['target'].unique():
            for model in self.df['model_family'].unique():
                for metric in metrics:
                    full_vals = self.df[
                        (self.df['target'] == target) &
                        (self.df['model_family'] == model) &
                        (self.df['feature_mode'] == 'FULL_FEATURES')
                    ][metric].dropna()
                    
                    reduced_vals = self.df[
                        (self.df['target'] == target) &
                        (self.df['model_family'] == model) &
                        (self.df['feature_mode'] == 'REDUCED_FEATURES')
                    ][metric].dropna()
                    
                    if len(full_vals) > 0 and len(reduced_vals) > 0:
                        full_mean = full_vals.mean()
                        reduced_mean = reduced_vals.mean()
                        rows.append({
                            'target': target,
                            'model_family': model,
                            'metric': metric,
                            'full_mean': full_mean,
                            'reduced_mean': reduced_mean,
                            'delta': full_mean - reduced_mean,
                            'ratio': full_mean / reduced_mean if reduced_mean != 0 else np.nan,
                        })
        
        df_by_target = pd.DataFrame(rows)
        df_by_target.to_csv(self.tables_dir / 'full_vs_reduced_by_target.csv', index=False)
        
        return df_by_target
    
    def _generate_delta_plots(self):
        """Generate delta plots (FULL - REDUCED)."""
        for metric in ['rmse', 'mae']:
            self._plot_delta_by_target(metric)
    
    def _plot_delta_by_target(self, metric: str):
        """Plot delta for a specific metric."""
        data = []
        
        for target in self.df['target'].unique():
            for model in self.df['model_family'].unique():
                full_vals = self.df[
                    (self.df['target'] == target) &
                    (self.df['model_family'] == model) &
                    (self.df['feature_mode'] == 'FULL_FEATURES')
                ][metric].dropna()
                
                reduced_vals = self.df[
                    (self.df['target'] == target) &
                    (self.df['model_family'] == model) &
                    (self.df['feature_mode'] == 'REDUCED_FEATURES')
                ][metric].dropna()
                
                if len(full_vals) > 0 and len(reduced_vals) > 0:
                    delta = full_vals.mean() - reduced_vals.mean()
                    data.append({
                        'target': target,
                        'model': model,
                        'delta': delta,
                    })
        
        if not data:
            return
        
        df_plot = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        targets = df_plot['target'].unique()
        models = df_plot['model'].unique()
        x = np.arange(len(targets))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            model_data = df_plot[df_plot['model'] == model]
            deltas = [model_data[model_data['target'] == t]['delta'].values[0] 
                     if len(model_data[model_data['target'] == t]) > 0 else 0 
                     for t in targets]
            
            color = MODEL_FAMILY_PALETTE.get(model, '#333333')
            bars = ax.bar(x + i * width, deltas, width, label=model, color=color, alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, deltas):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3 if height >= 0 else -10),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top',
                           fontsize=8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Target')
        ax.set_ylabel(f'Δ{metric.upper()} (FULL - REDUCED)')
        ax.set_title(f'{metric.upper()} Delta: Positive = FULL worse, Negative = FULL better')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(targets)
        ax.legend(loc='best')
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / f'delta_{metric}_by_target.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_heatmaps(self):
        """Generate heatmaps for FULL and REDUCED."""
        for mode in ['FULL_FEATURES', 'REDUCED_FEATURES']:
            self._plot_heatmap(mode, 'rmse')
            self._plot_heatmap(mode, 'mae')
        
        # Delta heatmap
        self._plot_delta_heatmap('rmse')
    
    def _plot_heatmap(self, mode: str, metric: str):
        """Plot heatmap for targets × models."""
        subset = self.df[self.df['feature_mode'] == mode]
        
        pivot = subset.pivot_table(
            index='target',
            columns='model_family',
            values=metric,
            aggfunc='mean'
        )
        
        if pivot.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, 
                   linewidths=0.5, cbar_kws={'label': metric.upper()})
        
        ax.set_title(f'{mode}: Mean {metric.upper()} by Target × Model')
        ax.set_xlabel('Model')
        ax.set_ylabel('Target')
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / f'heatmap_{metric}_{mode}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_delta_heatmap(self, metric: str):
        """Plot heatmap of deltas."""
        full = self.df[self.df['feature_mode'] == 'FULL_FEATURES'].pivot_table(
            index='target', columns='model_family', values=metric, aggfunc='mean'
        )
        reduced = self.df[self.df['feature_mode'] == 'REDUCED_FEATURES'].pivot_table(
            index='target', columns='model_family', values=metric, aggfunc='mean'
        )
        
        if full.empty or reduced.empty:
            return
        
        delta = full - reduced
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use diverging colormap centered at 0
        vmax = max(abs(delta.min().min()), abs(delta.max().max()))
        
        sns.heatmap(delta, annot=True, fmt='.2f', cmap='RdBu_r', ax=ax,
                   center=0, vmin=-vmax, vmax=vmax,
                   linewidths=0.5, cbar_kws={'label': f'Δ{metric.upper()}'})
        
        ax.set_title(f'Delta {metric.upper()}: FULL - REDUCED\n(Red = FULL worse, Blue = FULL better)')
        ax.set_xlabel('Model')
        ax.set_ylabel('Target')
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / f'heatmap_delta_{metric}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_ranking_comparison(self):
        """Generate ranking comparison plots."""
        for metric in ['rmse', 'mae']:
            self._plot_ranking_comparison(metric)
    
    def _plot_ranking_comparison(self, metric: str):
        """Plot average rank by feature mode."""
        # Compute ranks within each (target, feature_mode) group
        df_rank = self.df.copy()
        df_rank['rank'] = df_rank.groupby(['target', 'feature_mode'])[metric].rank()
        
        # Average rank per model × mode
        avg_ranks = df_rank.groupby(['model_family', 'feature_mode'])['rank'].mean().reset_index()
        
        if avg_ranks.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot grouped bars
        models = avg_ranks['model_family'].unique()
        modes = ['FULL_FEATURES', 'REDUCED_FEATURES']
        x = np.arange(len(models))
        width = 0.35
        
        for i, mode in enumerate(modes):
            mode_data = avg_ranks[avg_ranks['feature_mode'] == mode]
            ranks = [mode_data[mode_data['model_family'] == m]['rank'].values[0]
                    if len(mode_data[mode_data['model_family'] == m]) > 0 else np.nan
                    for m in models]
            
            color = FEATURE_MODE_PALETTE.get(mode, '#333333')
            ax.bar(x + i * width, ranks, width, label=mode, color=color, alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel(f'Average Rank ({metric.upper()}, lower = better)')
        ax.set_title(f'Model Ranking Comparison: FULL vs REDUCED')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(models)
        ax.legend()
        ax.invert_yaxis()  # Lower rank = better
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / f'ranking_comparison_{metric}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


# =============================================================================
# MODEL COMPARISON
# =============================================================================

class ModelComparisonAnalyzer:
    """
    Comprehensive model comparison analysis.
    """
    
    def __init__(self, master_df: pd.DataFrame, output_dir: Path):
        self.df = master_df
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / 'tables'
        self.figures_dir = self.output_dir / 'figures' / 'model_comparison'
        
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self):
        """Generate all model comparisons."""
        self._generate_global_heatmaps()
        self._generate_distribution_plots()
        self._generate_pairwise_wins()
        self._generate_pareto_plot()
        self._generate_per_target_analysis()
    
    def _generate_global_heatmaps(self):
        """Generate target × model heatmaps."""
        for metric in ['rmse', 'mae', 'coverage95']:
            if metric in self.df.columns:
                self._plot_global_heatmap(metric)
    
    def _plot_global_heatmap(self, metric: str):
        """Plot global heatmap."""
        pivot = self.df.pivot_table(
            index='target',
            columns='model_variant',
            values=metric,
            aggfunc='mean'
        )
        
        if pivot.empty:
            return
        
        # Sort columns by mean value
        col_order = pivot.mean().sort_values().index
        pivot = pivot[col_order]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        cmap = 'YlOrRd' if metric != 'coverage95' else 'YlGn'
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap, ax=ax,
                   linewidths=0.5, cbar_kws={'label': metric.upper()})
        
        ax.set_title(f'Mean {metric.upper()} by Target × Model Variant')
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('Target')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / f'global_heatmap_{metric}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_distribution_plots(self):
        """Generate distribution plots."""
        for metric in ['rmse', 'mae']:
            self._plot_distribution(metric)
    
    def _plot_distribution(self, metric: str):
        """Plot metric distribution by model family."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot
        ax1 = axes[0]
        palette = [MODEL_FAMILY_PALETTE.get(m, '#333333') for m in self.df['model_family'].unique()]
        sns.boxplot(data=self.df, x='model_family', y=metric, ax=ax1, palette=palette)
        ax1.set_title(f'{metric.upper()} Distribution by Model Family')
        ax1.set_xlabel('Model Family')
        ax1.set_ylabel(metric.upper())
        
        # Violin plot faceted by feature mode
        ax2 = axes[1]
        sns.violinplot(data=self.df, x='model_family', y=metric, hue='feature_mode',
                      split=True, ax=ax2, palette=FEATURE_MODE_PALETTE)
        ax2.set_title(f'{metric.upper()} by Model × Feature Mode')
        ax2.set_xlabel('Model Family')
        ax2.set_ylabel(metric.upper())
        ax2.legend(title='Feature Mode', loc='upper right')
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / f'distribution_{metric}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_pairwise_wins(self):
        """Generate pairwise win/tie/loss matrix."""
        for metric in ['rmse', 'mae']:
            self._compute_wtl_matrix(metric)
    
    def _compute_wtl_matrix(self, metric: str):
        """Compute win/tie/loss matrix."""
        models = sorted(self.df['model_family'].unique())
        
        # Initialize matrices
        wins = pd.DataFrame(0, index=models, columns=models)
        
        # For each (target, feature_mode, fold) combination
        for (target, mode, fold), group in self.df.groupby(['target', 'feature_mode', 'fold_id']):
            if len(group) < 2:
                continue
            
            values = group.set_index('model_family')[metric].dropna()
            
            for m1 in models:
                for m2 in models:
                    if m1 == m2:
                        continue
                    if m1 in values.index and m2 in values.index:
                        # Lower is better for RMSE/MAE
                        if values[m1] < values[m2]:
                            wins.loc[m1, m2] += 1
        
        # Save
        wins.to_csv(self.tables_dir / f'wins_matrix_{metric}.csv')
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(wins, annot=True, fmt='d', cmap='Greens', ax=ax,
                   linewidths=0.5, cbar_kws={'label': 'Wins'})
        ax.set_title(f'Pairwise Wins ({metric.upper()})\nRow beats Column')
        ax.set_xlabel('Loser')
        ax.set_ylabel('Winner')
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / f'wins_matrix_{metric}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Summary
        wins_summary = pd.DataFrame({
            'model': models,
            'total_wins': wins.sum(axis=1).values,
            'total_losses': wins.sum(axis=0).values,
        })
        wins_summary['win_rate'] = wins_summary['total_wins'] / (wins_summary['total_wins'] + wins_summary['total_losses'])
        wins_summary = wins_summary.sort_values('total_wins', ascending=False)
        wins_summary.to_csv(self.tables_dir / f'wins_summary_{metric}.csv', index=False)
    
    def _generate_pareto_plot(self):
        """Generate Pareto plot of fit_time vs RMSE."""
        if 'fit_time' not in self.df.columns:
            return
        
        # Aggregate by model
        agg = self.df.groupby('model_family').agg({
            'rmse': 'mean',
            'fit_time': 'mean'
        }).dropna()
        
        if agg.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model in agg.index:
            color = MODEL_FAMILY_PALETTE.get(model, '#333333')
            ax.scatter(agg.loc[model, 'fit_time'], agg.loc[model, 'rmse'],
                      s=200, c=color, label=model, edgecolors='black', linewidths=1.5)
            ax.annotate(model, (agg.loc[model, 'fit_time'], agg.loc[model, 'rmse']),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Mean Fit Time (s)')
        ax.set_ylabel('Mean RMSE')
        ax.set_title('Pareto Front: Fit Time vs RMSE')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'pareto_time_vs_rmse.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _generate_per_target_analysis(self):
        """Generate per-target analysis."""
        for target in self.df['target'].unique():
            target_dir = self.figures_dir / target
            target_dir.mkdir(exist_ok=True)
            
            subset = self.df[self.df['target'] == target]
            
            # Ranking bar plot
            self._plot_target_ranking(subset, target, target_dir)
    
    def _plot_target_ranking(self, df: pd.DataFrame, target: str, output_dir: Path):
        """Plot ranking for a specific target."""
        agg = df.groupby('model_family')['rmse'].agg(['mean', 'std']).sort_values('mean')
        
        if agg.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        colors = [MODEL_FAMILY_PALETTE.get(m, '#333333') for m in agg.index]
        bars = ax.barh(agg.index, agg['mean'], xerr=agg['std'], color=colors, 
                      alpha=0.8, capsize=5, edgecolor='black')
        
        ax.set_xlabel('RMSE (mean ± std)')
        ax.set_title(f'{target}: Model Ranking by RMSE')
        ax.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, agg['mean']):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'ranking_rmse.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


# =============================================================================
# GP DEEP DIVE
# =============================================================================

class GPDeepDiveAnalyzer:
    """
    Specialized GP analysis: kernels, uncertainty, calibration.
    """
    
    def __init__(self, master_df: pd.DataFrame, output_dir: Path):
        self.df = master_df[master_df['model_family'] == 'GP'].copy()
        self.output_dir = Path(output_dir)
        self.tables_dir = self.output_dir / 'tables'
        self.figures_dir = self.output_dir / 'figures' / 'gp_deep_dive'
        
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self):
        """Generate all GP deep dive analysis."""
        if self.df.empty:
            logging.warning("No GP results found for deep dive analysis")
            return
        
        self._generate_top_k_selection()
        self._analyze_kernels()
        self._analyze_hyperparameters()
        self._generate_calibration_plots()
    
    def _generate_top_k_selection(self, k: int = 5):
        """Select Top-K GPs by primary metric."""
        # Rank by RMSE (lower is better)
        ranked = self.df.sort_values('rmse')
        
        # Get unique configurations
        top_k = ranked.drop_duplicates(subset=['kernel_type', 'model_variant']).head(k)
        
        top_k_summary = top_k[['target', 'feature_mode', 'model_variant', 'kernel_type', 
                               'rmse', 'mae', 'coverage95', 'nlpd']].copy()
        top_k_summary['rank'] = range(1, len(top_k_summary) + 1)
        top_k_summary['reason'] = 'Lowest RMSE'
        
        top_k_summary.to_csv(self.tables_dir / 'top5_gp_selection.csv', index=False)
        
        return top_k_summary
    
    def _analyze_kernels(self):
        """Analyze kernel performance."""
        if 'kernel_type' not in self.df.columns:
            return
        
        # Kernel frequency
        kernel_counts = self.df['kernel_type'].value_counts()
        kernel_counts.to_csv(self.tables_dir / 'kernel_frequency.csv')
        
        # Kernel vs RMSE
        fig, ax = plt.subplots(figsize=(10, 6))
        
        kernel_order = self.df.groupby('kernel_type')['rmse'].mean().sort_values().index
        
        sns.boxplot(data=self.df, x='kernel_type', y='rmse', order=kernel_order,
                   palette='viridis', ax=ax)
        sns.stripplot(data=self.df, x='kernel_type', y='rmse', order=kernel_order,
                     color='black', alpha=0.5, size=4, ax=ax)
        
        ax.set_title('GP Kernel Performance: RMSE Distribution')
        ax.set_xlabel('Kernel Type')
        ax.set_ylabel('RMSE')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'kernel_vs_rmse.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Kernel comparison heatmap
        pivot = self.df.pivot_table(
            index='target',
            columns='kernel_type',
            values='rmse',
            aggfunc='mean'
        )
        
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                       linewidths=0.5)
            ax.set_title('Kernel × Target: Mean RMSE')
            plt.tight_layout()
            fig.savefig(self.figures_dir / 'kernel_target_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    def _analyze_hyperparameters(self):
        """Analyze hyperparameter distributions."""
        # Extract alpha from params
        def extract_alpha(params_str):
            try:
                params = json.loads(params_str) if isinstance(params_str, str) else params_str
                return params.get('alpha')
            except:
                return None
        
        self.df['alpha'] = self.df['params'].apply(extract_alpha)
        
        # Alpha vs RMSE
        alpha_data = self.df.dropna(subset=['alpha'])
        
        if not alpha_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.scatterplot(data=alpha_data, x='alpha', y='rmse', hue='kernel_type',
                           style='target', s=100, ax=ax)
            
            ax.set_xscale('log')
            ax.set_title('GP Alpha vs RMSE')
            ax.set_xlabel('Alpha (log scale)')
            ax.set_ylabel('RMSE')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            fig.savefig(self.figures_dir / 'alpha_vs_rmse.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    def _generate_calibration_plots(self):
        """Generate calibration and uncertainty plots."""
        # Coverage vs nominal
        if 'coverage95' in self.df.columns:
            self._plot_calibration_curve()
        
        # Sharpness vs calibration error
        if 'sharpness' in self.df.columns and 'calibration_error_95' in self.df.columns:
            self._plot_sharpness_vs_calibration()
    
    def _plot_calibration_curve(self):
        """Plot observed vs nominal coverage."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Nominal coverage levels
        nominals = [50, 90, 95]
        observed = {}
        
        for nom in nominals:
            col = f'coverage{nom}'
            if col in self.df.columns:
                observed[nom] = self.df[col].dropna().mean() * 100  # Convert to percentage
        
        if observed:
            ax.plot([0, 100], [0, 100], 'k--', label='Perfect calibration')
            
            for nom, obs in observed.items():
                ax.scatter(nom, obs, s=200, zorder=10, label=f'{nom}% coverage')
                ax.annotate(f'{obs:.1f}%', (nom, obs), xytext=(5, 5),
                           textcoords='offset points', fontsize=10)
            
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xlabel('Nominal Coverage (%)')
            ax.set_ylabel('Observed Coverage (%)')
            ax.set_title('GP Calibration: Observed vs Nominal Coverage')
            ax.legend(loc='lower right')
            ax.set_aspect('equal')
            
            plt.tight_layout()
            fig.savefig(self.figures_dir / 'calibration_curve.png', dpi=150, bbox_inches='tight')
        
        plt.close(fig)
    
    def _plot_sharpness_vs_calibration(self):
        """Plot sharpness vs calibration error trade-off."""
        data = self.df.dropna(subset=['sharpness', 'calibration_error_95'])
        
        if data.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.scatterplot(data=data, x='sharpness', y='calibration_error_95',
                       hue='kernel_type', style='target', s=100, ax=ax)
        
        ax.set_xlabel('Sharpness (Mean Interval Width)')
        ax.set_ylabel('Calibration Error (95%)')
        ax.set_title('GP Trade-off: Sharpness vs Calibration Error')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        fig.savefig(self.figures_dir / 'sharpness_vs_calibration.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


# =============================================================================
# GP UNCERTAINTY VISUALIZATION FOR REAL DATA
# =============================================================================

class GPRealDataVisualizer:
    """
    Generates GP uncertainty visualizations for real data.
    
    Since real data is high-dimensional, creates:
    - 1D slices along top features
    - PCA uncertainty maps
    - Sorted "mythical GP plot"
    """
    
    def __init__(self, master_df: pd.DataFrame, session_dir: Path, output_dir: Path):
        self.master_df = master_df
        self.session_dir = Path(session_dir)
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'figures' / 'gp_uncertainty'
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Track what we can and cannot do
        self.quality_notes = []
    
    def generate_all(self):
        """Generate GP uncertainty visualizations."""
        # For each target and feature mode with GP results
        gp_configs = self.master_df[
            self.master_df['model_family'] == 'GP'
        ][['feature_mode', 'target']].drop_duplicates()
        
        for _, row in gp_configs.iterrows():
            self._generate_for_config(row['feature_mode'], row['target'])
    
    def _generate_for_config(self, feature_mode: str, target: str):
        """Generate visualizations for a specific config."""
        config_dir = self.figures_dir / feature_mode / target
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data and retrain GP
        data_info = self._load_data_and_train(feature_mode, target)
        
        if data_info is None:
            self.quality_notes.append(
                f"Could not load data for {feature_mode}/{target}. "
                "Recommendation: Save X_train, y_train in tuning output."
            )
            return
        
        X, y, model, feature_names, groups = data_info
        
        # Generate visualizations
        self._plot_mythical_gp(X, y, model, groups, config_dir, target)
        self._plot_uncertainty_vs_error(X, y, model, groups, config_dir, target)
        self._plot_1d_slices(X, y, model, feature_names, groups, config_dir, target)
        self._plot_pca_uncertainty_map(X, y, model, groups, config_dir, target)
    
    def _load_data_and_train(self, feature_mode: str, target: str):
        """Load data and retrain GP model."""
        try:
            # Try to load the dataset
            from src.utils.paths import DATA_DIR
            
            # Map target to column name
            target_column_map = {
                'proteina': 'PROTEINA (%)',
                'fcr': 'FCR',
                'quitina': 'QUITINA (%)',
                'tpc': 'TPC (mg GAE/g)',
            }
            
            # Load dataset
            dataset_path = DATA_DIR / 'entomotive_datasets' / 'productivity_hermetia_lote.csv'
            
            if not dataset_path.exists():
                return None
            
            df = pd.read_csv(dataset_path)
            
            # Get target column
            target_col = target_column_map.get(target.lower())
            if target_col not in df.columns:
                return None
            
            # Get features (exclude target and group columns)
            exclude_cols = ['Lote', 'diet', target_col] + list(target_column_map.values())
            feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
            
            # Clean data
            df_clean = df.dropna(subset=[target_col] + feature_cols)
            
            X = df_clean[feature_cols].values
            y = df_clean[target_col].values
            groups = df_clean['diet'].values if 'diet' in df_clean.columns else None
            
            # Load best GP params
            gp_json_path = self.session_dir / feature_mode / target / f'{target}_gp_tuning.json'
            
            if not gp_json_path.exists():
                return None
            
            with open(gp_json_path) as f:
                gp_data = json.load(f)
            
            # Use first fold's params (or best)
            best_params = gp_data['chosen_params'][0] if gp_data['chosen_params'] else {}
            
            # Train GP
            from src.models.gp import GPSurrogateRegressor
            
            # Clean params (kernel string issue)
            clean_params = {k: v for k, v in best_params.items() if k != 'kernel'}
            
            model = GPSurrogateRegressor(**clean_params)
            model.fit(X, y)
            
            return X, y, model, feature_cols, groups
            
        except Exception as e:
            logging.error(f"Error loading data for {feature_mode}/{target}: {e}")
            return None
    
    def _plot_mythical_gp(self, X, y, model, groups, output_dir, target):
        """
        The classic "mythical GP plot": sorted by true value with confidence bands.
        """
        try:
            mean, std = model.predict_dist(X)
        except:
            mean = model.predict(X)
            std = np.zeros_like(mean)
        
        # Sort by true value
        sort_idx = np.argsort(y)
        x_idx = np.arange(len(y))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Confidence bands
        if np.any(std > 0):
            ax.fill_between(x_idx, 
                           mean[sort_idx] - 1.96 * std[sort_idx],
                           mean[sort_idx] + 1.96 * std[sort_idx],
                           color='cornflowerblue', alpha=0.3, label='95% Confidence')
        
        # Mean prediction
        ax.plot(x_idx, mean[sort_idx], 'b-', lw=1.5, label='GP Mean', zorder=5)
        
        # True values
        if groups is not None:
            for g in np.unique(groups):
                mask = groups[sort_idx] == g
                ax.scatter(x_idx[mask], y[sort_idx][mask], label=g, s=60, 
                          edgecolors='k', alpha=0.8, zorder=10)
        else:
            ax.scatter(x_idx, y[sort_idx], color='red', s=60, 
                      edgecolors='k', label='True values', zorder=10)
        
        ax.set_xlabel('Sample Index (sorted by true value)')
        ax.set_ylabel(target.upper())
        ax.set_title(f'{target.upper()}: GP Calibration - Confidence vs Real Data')
        ax.legend(loc='upper left', ncol=2, fontsize=9)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'mythical_gp_plot.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_uncertainty_vs_error(self, X, y, model, groups, output_dir, target):
        """Plot predicted uncertainty vs actual error."""
        try:
            mean, std = model.predict_dist(X)
        except:
            return
        
        abs_error = np.abs(y - mean)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if groups is not None:
            for g in np.unique(groups):
                mask = groups == g
                ax.scatter(std[mask], abs_error[mask], label=g, s=60, alpha=0.7)
        else:
            ax.scatter(std, abs_error, color='steelblue', s=60, alpha=0.7)
        
        # Trend line
        if len(std) > 2:
            z = np.polyfit(std, abs_error, 1)
            p = np.poly1d(z)
            x_line = np.linspace(std.min(), std.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', lw=2, label='Trend')
            
            # Correlation
            corr = np.corrcoef(std, abs_error)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   fontsize=12, verticalalignment='top')
        
        ax.set_xlabel('Predicted Uncertainty (σ)')
        ax.set_ylabel('Absolute Error')
        ax.set_title(f'{target.upper()}: Uncertainty vs Error\n(Good GP: positive correlation)')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        fig.savefig(output_dir / 'uncertainty_vs_error.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_1d_slices(self, X, y, model, feature_names, groups, output_dir, target):
        """Plot 1D response slices for top features."""
        # Select top-2 features by variance
        variances = X.var(axis=0)
        top_2_idx = np.argsort(variances)[-2:]
        
        for idx in top_2_idx:
            feature_name = feature_names[idx] if idx < len(feature_names) else f'Feature {idx}'
            self._plot_single_slice(X, y, model, idx, feature_name, groups, output_dir, target)
    
    def _plot_single_slice(self, X, y, model, feature_idx, feature_name, groups, output_dir, target):
        """Plot a single 1D slice."""
        # Create grid along this feature
        x_min, x_max = X[:, feature_idx].min(), X[:, feature_idx].max()
        padding = (x_max - x_min) * 0.1
        x_grid = np.linspace(x_min - padding, x_max + padding, 200)
        
        # Fix other features at median
        X_eval = np.tile(np.median(X, axis=0), (200, 1))
        X_eval[:, feature_idx] = x_grid
        
        try:
            mean, std = model.predict_dist(X_eval)
        except:
            mean = model.predict(X_eval)
            std = np.zeros_like(mean)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Uncertainty band
        if np.any(std > 0):
            ax.fill_between(x_grid, mean - 1.96*std, mean + 1.96*std,
                           color='cornflowerblue', alpha=0.2, label='95% CI')
        
        # Mean
        ax.plot(x_grid, mean, 'b-', lw=2, label='GP Response')
        
        # Real data points
        if groups is not None:
            for g in np.unique(groups):
                mask = groups == g
                ax.scatter(X[mask, feature_idx], y[mask], label=g, s=60, 
                          edgecolors='k', alpha=0.7, zorder=10)
        else:
            ax.scatter(X[:, feature_idx], y, color='red', s=60,
                      edgecolors='k', label='Data', zorder=10)
        
        # Rug plot
        ax.plot(X[:, feature_idx], np.full(len(X), ax.get_ylim()[0]), '|k', 
               markersize=10, alpha=0.3)
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel(target.upper())
        ax.set_title(f'{target.upper()}: Response Profile - {feature_name}\n(others fixed at median)')
        ax.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        
        # Clean filename
        safe_name = re.sub(r'[^\w\-]', '_', feature_name)
        fig.savefig(output_dir / f'slice_1d_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_pca_uncertainty_map(self, X, y, model, groups, output_dir, target):
        """Plot PCA 2D projection with uncertainty coloring."""
        from sklearn.decomposition import PCA
        
        # Fit PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Get predictions
        try:
            mean, std = model.predict_dist(X)
        except:
            mean = model.predict(X)
            std = np.zeros_like(mean)
        
        abs_error = np.abs(y - mean)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Color by uncertainty
        ax1 = axes[0]
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=std, cmap='YlOrRd',
                              s=60 + 100 * abs_error / abs_error.max(),  # Size by error
                              alpha=0.7, edgecolors='k', linewidths=0.5)
        plt.colorbar(scatter1, ax=ax1, label='Predicted σ')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax1.set_title(f'{target.upper()}: PCA Uncertainty Map\n(size ∝ |error|)')
        
        # Right: Color by group
        ax2 = axes[1]
        if groups is not None:
            for g in np.unique(groups):
                mask = groups == g
                ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], label=g, s=60,
                           alpha=0.7, edgecolors='k', linewidths=0.5)
            ax2.legend(loc='upper right', fontsize=9)
        else:
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], c='steelblue', s=60,
                       alpha=0.7, edgecolors='k', linewidths=0.5)
        
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax2.set_title(f'{target.upper()}: PCA by Group')
        
        plt.tight_layout()
        fig.savefig(output_dir / 'pca_uncertainty_map.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_real_tuning_report(session_name: str,
                                session_dir: Path = None,
                                output_dir: Path = None) -> Path:
    """
    Generate complete real tuning visual evaluation report.
    
    Args:
        session_name: Name of tuning session
        session_dir: Path to session directory (auto-detected if None)
        output_dir: Output directory (auto-generated if None)
        
    Returns:
        Path to report directory
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    print("=" * 70)
    print(f"REAL TUNING VISUAL EVALUATION: {session_name}")
    print("=" * 70)
    
    # Auto-detect session directory
    if session_dir is None:
        session_dir = LOGS_DIR / 'tuning' / session_name
    
    session_dir = Path(session_dir)
    
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path('outputs/reports/real_tuning') / session_name
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # BUILD MASTER TABLE
    # =========================================================================
    print("\n[PASO 1] Building master table...")
    
    builder = MasterTableBuilder(session_dir)
    master_df = builder.build()
    
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)
    master_df.to_csv(tables_dir / 'master_table.csv', index=False)
    
    # Quality report
    quality_report = builder.get_quality_report()
    with open(tables_dir / 'data_quality_report.md', 'w', encoding='utf-8') as f:
        f.write(quality_report.to_markdown())
    
    print(f"  - {len(master_df)} rows loaded")
    print(f"  - Feature modes: {master_df['feature_mode'].unique().tolist()}")
    print(f"  - Targets: {master_df['target'].unique().tolist()}")
    print(f"  - Models: {master_df['model_family'].unique().tolist()}")
    
    # =========================================================================
    # FULL VS REDUCED COMPARISON
    # =========================================================================
    print("\n[PASO 2] Generating FULL vs REDUCED comparison...")
    
    fvr_analyzer = FullVsReducedAnalyzer(master_df, output_dir)
    fvr_analyzer.generate_all()
    
    # =========================================================================
    # MODEL COMPARISON
    # =========================================================================
    print("\n[PASO 3] Generating model comparisons...")
    
    model_analyzer = ModelComparisonAnalyzer(master_df, output_dir)
    model_analyzer.generate_all()
    
    # =========================================================================
    # GP DEEP DIVE
    # =========================================================================
    print("\n[PASO 4] Generating GP deep dive...")
    
    gp_analyzer = GPDeepDiveAnalyzer(master_df, output_dir)
    gp_analyzer.generate_all()
    
    # =========================================================================
    # GP UNCERTAINTY VISUALIZATIONS
    # =========================================================================
    print("\n[PASO 5] Generating GP uncertainty visualizations...")
    
    gp_viz = GPRealDataVisualizer(master_df, session_dir, output_dir)
    gp_viz.generate_all()
    
    # =========================================================================
    # GENERATE REPORT
    # =========================================================================
    print("\n[FINAL] Generating report index...")
    
    _generate_report_markdown(output_dir, session_name, master_df)
    
    print("\n" + "=" * 70)
    print(f"REPORT COMPLETE: {output_dir}")
    print("=" * 70)
    
    return output_dir


def _generate_report_markdown(output_dir: Path, session_name: str, master_df: pd.DataFrame):
    """Generate main report markdown file."""
    report_path = output_dir / 'real_tuning_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Real Tuning Visual Evaluation Report\n\n")
        f.write(f"**Session**: {session_name}\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Feature Modes**: {', '.join(master_df['feature_mode'].unique())}\n")
        f.write(f"- **Targets**: {', '.join(master_df['target'].unique())}\n")
        f.write(f"- **Models**: {', '.join(master_df['model_family'].unique())}\n")
        f.write(f"- **Total Folds**: {len(master_df)}\n\n")
        
        # Tables
        f.write("## Tables\n\n")
        f.write("| File | Description |\n")
        f.write("|------|-------------|\n")
        f.write("| [master_table.csv](tables/master_table.csv) | Complete tidy results |\n")
        f.write("| [full_vs_reduced_global.csv](tables/full_vs_reduced_global.csv) | Global FULL vs REDUCED comparison |\n")
        f.write("| [full_vs_reduced_by_target.csv](tables/full_vs_reduced_by_target.csv) | Per-target comparison |\n")
        f.write("| [wins_matrix_rmse.csv](tables/wins_matrix_rmse.csv) | Pairwise win matrix |\n")
        f.write("| [top5_gp_selection.csv](tables/top5_gp_selection.csv) | Top-5 GP configurations |\n\n")
        
        # FULL vs REDUCED figures
        f.write("## FULL vs REDUCED Comparison\n\n")
        f.write("### Delta Plots\n\n")
        f.write("- [RMSE Delta](figures/full_vs_reduced/delta_rmse_by_target.png)\n")
        f.write("- [MAE Delta](figures/full_vs_reduced/delta_mae_by_target.png)\n\n")
        
        f.write("### Heatmaps\n\n")
        f.write("- [FULL RMSE Heatmap](figures/full_vs_reduced/heatmap_rmse_FULL_FEATURES.png)\n")
        f.write("- [REDUCED RMSE Heatmap](figures/full_vs_reduced/heatmap_rmse_REDUCED_FEATURES.png)\n")
        f.write("- [Delta Heatmap](figures/full_vs_reduced/heatmap_delta_rmse.png)\n\n")
        
        # Model comparison figures
        f.write("## Model Comparison\n\n")
        f.write("- [Global Heatmap RMSE](figures/model_comparison/global_heatmap_rmse.png)\n")
        f.write("- [Distribution RMSE](figures/model_comparison/distribution_rmse.png)\n")
        f.write("- [Wins Matrix](figures/model_comparison/wins_matrix_rmse.png)\n")
        f.write("- [Pareto: Time vs RMSE](figures/model_comparison/pareto_time_vs_rmse.png)\n\n")
        
        # GP Deep Dive
        f.write("## GP Deep Dive\n\n")
        f.write("- [Kernel vs RMSE](figures/gp_deep_dive/kernel_vs_rmse.png)\n")
        f.write("- [Kernel × Target Heatmap](figures/gp_deep_dive/kernel_target_heatmap.png)\n")
        f.write("- [Alpha vs RMSE](figures/gp_deep_dive/alpha_vs_rmse.png)\n")
        f.write("- [Calibration Curve](figures/gp_deep_dive/calibration_curve.png)\n\n")
        
        # GP Uncertainty
        f.write("## GP Uncertainty Visualizations\n\n")
        for mode in master_df['feature_mode'].unique():
            f.write(f"### {mode}\n\n")
            for target in master_df['target'].unique():
                f.write(f"**{target}:**\n")
                base = f"figures/gp_uncertainty/{mode}/{target}"
                f.write(f"- [Mythical GP Plot]({base}/mythical_gp_plot.png)\n")
                f.write(f"- [Uncertainty vs Error]({base}/uncertainty_vs_error.png)\n")
                f.write(f"- [PCA Uncertainty Map]({base}/pca_uncertainty_map.png)\n\n")
        
        # Data Quality
        f.write("## Data Quality\n\n")
        f.write("See [data_quality_report.md](tables/data_quality_report.md) for details.\n")
    
    logging.info(f"Report saved: {report_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate real tuning visual evaluation report"
    )
    parser.add_argument(
        "--session", "-s",
        type=str,
        required=True,
        help="Session name (e.g., productivity_hermetia_v2_comprehensive)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available sessions"
    )
    
    args = parser.parse_args()
    
    if args.list:
        tuning_dir = LOGS_DIR / 'tuning'
        print("Available tuning sessions:")
        for d in sorted(tuning_dir.iterdir()):
            if d.is_dir():
                print(f"  - {d.name}")
    else:
        output_path = Path(args.output) if args.output else None
        generate_real_tuning_report(
            session_name=args.session,
            output_dir=output_path,
        )
