# @author: José Arbelaez
"""
benchmark_visual_reporter.py

Comprehensive Benchmark Visual Evaluation Pack for surrogate model results.
Generates publication-ready figures, tables, and reports from benchmark evaluation outputs.

Usage:
    python -m src.evaluation.benchmark_visual_reporter --session benchmark_20260120_131837

    Or programmatically:
        from src.evaluation.benchmark_visual_reporter import generate_benchmark_report
        generate_benchmark_report("benchmark_20260120_131837")
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
from itertools import combinations

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
MODEL_FAMILY_PALETTE = {
    "Dummy": "#95a5a6",   # Gray
    "Ridge": "#3498db",   # Blue
    "PLS": "#9b59b6",     # Purple
    "GP": "#2ecc71",      # Green
}

GP_KERNEL_PALETTE = {
    "GP_Matern32": "#27ae60",   # Green variants
    "GP_Matern52": "#2ecc71",
    "GP_RBF": "#1abc9c",
}

NOISE_PALETTE = {
    "NoNoise": "#2c3e50",
    "GaussianNoise": "#e74c3c",
    "HeteroscedasticNoise": "#f39c12",
    "ProportionalNoise": "#9b59b6",
}

# Extended palette for all model variants
MODEL_VARIANT_PALETTE = {
    "Dummy": "#95a5a6",
    "Ridge": "#3498db",
    "PLS_n2": "#8e44ad",
    "PLS_n3": "#9b59b6",
    "GP_Matern32": "#27ae60",
    "GP_Matern52": "#2ecc71",
    "GP_RBF": "#1abc9c",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DataQualityReport:
    """Report on data quality and disambiguation status."""
    total_records: int
    unique_combinations: int
    duplicates_detected: List[Tuple[str, str, str, int]]  # (benchmark, noise, model, count)
    missing_fields: Dict[str, int]
    disambiguation_status: Dict[str, str]  # field -> status
    recommendations: List[str]


@dataclass
class BenchmarkInventory:
    """Inventory of benchmark evaluation session."""
    session_name: str
    timestamp: str
    total_time_s: float
    benchmarks: List[str]
    noise_types: List[str]
    model_families: List[str]
    model_variants: List[str]
    n_results: int
    completeness_matrix: pd.DataFrame


# =============================================================================
# PASO 0: DATA LOADING AND NORMALIZATION
# =============================================================================

class BenchmarkResultsLoader:
    """
    Loads, normalizes, and disambiguates benchmark evaluation results.

    Handles the critical task of creating model_variant identifiers from
    collapsed model names using model_params.
    """

    def __init__(self, results_json_path: Path):
        """
        Initialize loader with path to results JSON.

        Args:
            results_json_path: Path to *_results.json file
        """
        self.json_path = Path(results_json_path)
        self.session_name = self.json_path.stem.replace("_results", "")

        self._raw_data = None
        self._master_df = None
        self._quality_report = None

    def load_raw(self) -> Dict:
        """Load raw JSON data."""
        if self._raw_data is None:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self._raw_data = json.load(f)
        return self._raw_data

    def _parse_kernel_variant(self, kernel_str: str) -> str:
        """
        Parse GP kernel string to extract variant name.

        Examples:
            "Matern(length_scale=1, nu=1.5) + WhiteKernel(...)" -> "Matern32"
            "Matern(length_scale=1, nu=2.5) + WhiteKernel(...)" -> "Matern52"
            "RBF(length_scale=1) + WhiteKernel(...)" -> "RBF"
        """
        if not kernel_str:
            return "Unknown"

        kernel_lower = kernel_str.lower()

        if "matern" in kernel_lower:
            # Extract nu parameter
            nu_match = re.search(r'nu\s*=\s*([\d.]+)', kernel_str)
            if nu_match:
                nu_val = float(nu_match.group(1))
                if abs(nu_val - 1.5) < 0.1:
                    return "Matern32"
                elif abs(nu_val - 2.5) < 0.1:
                    return "Matern52"
                else:
                    return f"Matern_nu{nu_val}"
            return "Matern"
        elif "rbf" in kernel_lower:
            return "RBF"
        elif "rational" in kernel_lower:
            return "RationalQuadratic"
        else:
            return "CustomKernel"

    def _extract_model_variant(self, model_name: str, model_params: Dict) -> Tuple[str, str]:
        """
        Extract model_family and model_variant from model name and params.

        Returns:
            Tuple of (model_family, model_variant)
        """
        # Map class names to families
        family_map = {
            "DummySurrogateRegressor": "Dummy",
            "RidgeSurrogateRegressor": "Ridge",
            "PLSSurrogateRegressor": "PLS",
            "GPSurrogateRegressor": "GP",
        }

        family = family_map.get(model_name, model_name)

        if family == "PLS":
            n_comp = model_params.get("n_components", "?")
            variant = f"PLS_n{n_comp}"
        elif family == "GP":
            kernel_str = model_params.get("kernel", "")
            kernel_variant = self._parse_kernel_variant(kernel_str)
            variant = f"GP_{kernel_variant}"
        elif family == "Dummy":
            strategy = model_params.get("strategy", "mean")
            variant = f"Dummy" if strategy == "mean" else f"Dummy_{strategy}"
        elif family == "Ridge":
            alpha = model_params.get("alpha", 1.0)
            variant = f"Ridge" if alpha == 1.0 else f"Ridge_a{alpha}"
        else:
            variant = family

        return family, variant

    def build_master_table(self) -> pd.DataFrame:
        """
        Build the normalized master table with all results.

        Creates disambiguated model_variant column and assigns run_ids
        for duplicate combinations (e.g., multiple GaussianNoise runs).

        Supports two JSON formats:
        1. Legacy flat format: {"results": [{"benchmark": ..., "noise": ..., ...}, ...]}
        2. New nested format: {"results": {sampler: {n_train: {benchmark: {noise: {cv_mode: {model: {...}}}}}}}

        Returns:
            Master DataFrame with columns:
                benchmark, noise_type, noise_variant, model_family, model_variant,
                all metrics, fit_time_s, predict_time_s, run_id
        """
        if self._master_df is not None:
            return self._master_df

        raw = self.load_raw()
        records = []

        # Track run_ids for duplicates
        combo_counter = Counter()

        results_data = raw.get("results", [])

        # Detect format: list (legacy) vs dict (new nested)
        if isinstance(results_data, list):
            # Legacy flat format
            for result in results_data:
                benchmark = result.get("benchmark")
                noise = result.get("noise")
                model_name = result.get("model")
                model_params = result.get("model_params", {})
                metrics = result.get("metrics", {})

                family, variant = self._extract_model_variant(model_name, model_params)

                combo_key = (benchmark, noise, variant)
                combo_counter[combo_key] += 1
                run_id = combo_counter[combo_key]

                record = {
                    "benchmark": benchmark,
                    "noise_type": noise,
                    "noise_variant": f"{noise}_run{run_id}" if combo_counter[combo_key] > 1 or noise == "GaussianNoise" else noise,
                    "model_family": family,
                    "model_variant": variant,
                    "model_class": model_name,
                    "run_id": run_id,
                    "fit_time_s": result.get("fit_time_s"),
                    "predict_time_s": result.get("predict_time_s"),
                }

                for metric_name, metric_value in metrics.items():
                    record[metric_name] = metric_value

                record["model_params_json"] = json.dumps(model_params)
                records.append(record)

        elif isinstance(results_data, dict):
            # New nested format: sampler -> n_train -> benchmark -> noise -> cv_mode -> model -> metrics
            for sampler, sampler_data in results_data.items():
                if not isinstance(sampler_data, dict):
                    continue
                for n_train, n_train_data in sampler_data.items():
                    if not isinstance(n_train_data, dict):
                        continue
                    for benchmark, benchmark_data in n_train_data.items():
                        if not isinstance(benchmark_data, dict):
                            continue
                        for noise, noise_data in benchmark_data.items():
                            if not isinstance(noise_data, dict):
                                continue
                            for cv_mode, cv_mode_data in noise_data.items():
                                if not isinstance(cv_mode_data, dict):
                                    continue
                                for model_name, model_metrics in cv_mode_data.items():
                                    if not isinstance(model_metrics, dict):
                                        continue

                                    model_params = model_metrics.get("model_params", {})
                                    family, variant = self._extract_model_variant(model_name, model_params)

                                    combo_key = (benchmark, noise, variant, sampler, n_train, cv_mode)
                                    combo_counter[combo_key] += 1
                                    run_id = combo_counter[combo_key]

                                    record = {
                                        "benchmark": benchmark,
                                        "noise_type": noise,
                                        "noise_variant": f"{noise}_run{run_id}" if combo_counter[combo_key] > 1 or noise == "GaussianNoise" else noise,
                                        "model_family": family,
                                        "model_variant": variant,
                                        "model_class": model_name,
                                        "run_id": run_id,
                                        "sampler": sampler,
                                        "n_train": int(n_train) if n_train.isdigit() else n_train,
                                        "cv_mode": cv_mode,
                                        "fit_time_s": model_metrics.get("fit_time"),
                                        "predict_time_s": model_metrics.get("predict_time"),
                                    }

                                    # Add all metrics (excluding model_params, fit_time, predict_time)
                                    skip_keys = {"model_params", "fit_time", "predict_time"}
                                    for metric_name, metric_value in model_metrics.items():
                                        if metric_name not in skip_keys:
                                            record[metric_name] = metric_value

                                    record["model_params_json"] = json.dumps(model_params)
                                    records.append(record)

        self._master_df = pd.DataFrame(records)

        # Reorder columns for clarity
        primary_cols = [
            "benchmark", "noise_type", "noise_variant", "model_family", "model_variant",
            "run_id", "mae", "rmse", "r2", "nlpd", "coverage_95", "calibration_error_95",
            "sharpness", "fit_time_s", "predict_time_s"
        ]
        other_cols = [c for c in self._master_df.columns if c not in primary_cols]
        ordered_cols = [c for c in primary_cols if c in self._master_df.columns] + other_cols
        self._master_df = self._master_df[ordered_cols]

        return self._master_df

    def generate_quality_report(self) -> DataQualityReport:
        """
        Generate data quality report identifying issues and ambiguities.
        """
        if self._quality_report is not None:
            return self._quality_report

        df = self.build_master_table()
        raw = self.load_raw()

        # Count duplicates by original (benchmark, noise, model_class)
        original_combos = Counter()
        results_data = raw.get("results", [])

        if isinstance(results_data, list):
            # Legacy flat format
            for result in results_data:
                key = (result["benchmark"], result["noise"], result["model"])
                original_combos[key] += 1
        elif isinstance(results_data, dict):
            # New nested format - use the master_df to count original combos
            for _, row in df.iterrows():
                key = (row["benchmark"], row["noise_type"], row["model_class"])
                original_combos[key] += 1

        duplicates = [(b, n, m, c) for (b, n, m), c in original_combos.items() if c > 1]

        # Check missing fields
        missing_fields = {}
        for col in ["nlpd", "coverage_95", "calibration_error_95", "sharpness"]:
            if col in df.columns:
                n_missing = df[col].isna().sum()
                if n_missing > 0:
                    missing_fields[col] = n_missing

        # Disambiguation status
        disambiguation = {
            "model_variant": "RESOLVED via model_params (kernel for GP, n_components for PLS)",
            "noise_variant": "PARTIAL - GaussianNoise sigma NOT stored in outputs (using run_id)",
            "seed": "NOT AVAILABLE - not stored in current outputs",
        }

        # Recommendations
        recommendations = []

        if "GaussianNoise" in df["noise_type"].values:
            n_gauss_variants = df[df["noise_type"] == "GaussianNoise"]["noise_variant"].nunique()
            if n_gauss_variants > 1:
                recommendations.append(
                    "CRITICAL: GaussianNoise has multiple runs but sigma values are not stored. "
                    "Recommend adding 'noise_params' dict to results with sigma value."
                )

        recommendations.append(
            "MINOR: Consider storing explicit 'model_variant' string in outputs "
            "to avoid runtime parsing of kernel strings."
        )

        if df["predict_time_s"].isna().any():
            recommendations.append(
                "MINOR: Some predict_time_s values are missing."
            )

        self._quality_report = DataQualityReport(
            total_records=len(df),
            unique_combinations=len(df.groupby(["benchmark", "noise_type", "model_variant"])),
            duplicates_detected=duplicates,
            missing_fields=missing_fields,
            disambiguation_status=disambiguation,
            recommendations=recommendations,
        )

        return self._quality_report

    def get_inventory(self) -> BenchmarkInventory:
        """Get inventory of the benchmark session."""
        raw = self.load_raw()
        df = self.build_master_table()

        # Build completeness matrix
        completeness = df.groupby(["benchmark", "noise_type", "model_variant"]).size().unstack(fill_value=0)

        # Support both legacy and new format for metadata
        metadata = raw.get("metadata", {})
        timestamp = metadata.get("timestamp", raw.get("timestamp", ""))
        total_time = metadata.get("total_time_s", raw.get("total_time_s", 0))

        return BenchmarkInventory(
            session_name=self.session_name,
            timestamp=timestamp,
            total_time_s=total_time,
            benchmarks=sorted(df["benchmark"].unique()),
            noise_types=sorted(df["noise_type"].unique()),
            model_families=sorted(df["model_family"].unique()),
            model_variants=sorted(df["model_variant"].unique()),
            n_results=len(df),
            completeness_matrix=completeness,
        )

    def export_master_table(self, output_dir: Path) -> Path:
        """Export master table to CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / "master_table.csv"
        self.build_master_table().to_csv(path, index=False)
        logging.info(f"Exported master table: {path}")
        return path

    def export_quality_report(self, output_dir: Path) -> Path:
        """Export quality report to Markdown."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = self.generate_quality_report()
        path = output_dir / "data_quality_report.md"

        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Data Quality Report\n\n")
            f.write(f"**Session**: {self.session_name}\n\n")

            f.write("## Summary\n\n")
            f.write(f"- Total records: {report.total_records}\n")
            f.write(f"- Unique (benchmark, noise, model_variant) combinations: {report.unique_combinations}\n")
            f.write(f"- Duplicate combinations detected: {len(report.duplicates_detected)}\n\n")

            f.write("## Duplicates (model variants collapsed in original data)\n\n")
            if report.duplicates_detected:
                f.write("| Benchmark | Noise | Model Class | Count |\n")
                f.write("|-----------|-------|-------------|-------|\n")
                for b, n, m, c in sorted(report.duplicates_detected):
                    f.write(f"| {b} | {n} | {m} | {c} |\n")
            else:
                f.write("No duplicates detected.\n")

            f.write("\n## Missing Fields\n\n")
            if report.missing_fields:
                f.write("| Field | Missing Count |\n")
                f.write("|-------|---------------|\n")
                for field, count in report.missing_fields.items():
                    f.write(f"| {field} | {count} |\n")
                f.write("\nNote: Missing uncertainty metrics (nlpd, coverage, etc.) are expected for non-GP models.\n")
            else:
                f.write("All fields populated (or appropriately null for non-probabilistic models).\n")

            f.write("\n## Disambiguation Status\n\n")
            for field, status in report.disambiguation_status.items():
                f.write(f"- **{field}**: {status}\n")

            f.write("\n## Recommendations for Pipeline Improvements\n\n")
            for i, rec in enumerate(report.recommendations, 1):
                f.write(f"{i}. {rec}\n")

        logging.info(f"Exported quality report: {path}")
        return path


# =============================================================================
# PASO 1: TABLE GENERATORS
# =============================================================================

class BenchmarkTableGenerator:
    """
    Generates all required summary tables from benchmark results.
    """

    def __init__(self, master_df: pd.DataFrame, output_dir: Path):
        self.df = master_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _aggregate_metric(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Aggregate a metric across groups."""
        agg = df.groupby("model_variant")[metric].agg([
            "mean", "median", "std",
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75),
            "count"
        ]).reset_index()
        agg.columns = ["model_variant", "mean", "median", "std", "p25", "p75", "n_valid"]
        return agg.sort_values("mean", ascending=(metric != "r2"))

    # -------------------------------------------------------------------------
    # A) Global Leaderboards
    # -------------------------------------------------------------------------

    def generate_global_leaderboard(self, metrics: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate global leaderboard by metric.

        Returns dict of metric -> leaderboard DataFrame
        """
        if metrics is None:
            metrics = ["rmse", "mae", "r2"]

        leaderboards = {}

        for metric in metrics:
            if metric not in self.df.columns:
                continue

            # Full aggregation (all benchmarks × noises)
            df_valid = self.df[self.df[metric].notna()]
            leaderboard = self._aggregate_metric(df_valid, metric)
            leaderboard["rank"] = range(1, len(leaderboard) + 1)

            leaderboards[metric] = leaderboard

            # Save
            path = self.output_dir / f"leaderboard_global_{metric}.csv"
            leaderboard.to_csv(path, index=False)

        logging.info(f"Generated {len(leaderboards)} global leaderboards")
        return leaderboards

    def generate_leaderboard_by_benchmark_avg(self, metrics: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate leaderboard averaged over benchmarks (avg over noises first).
        """
        if metrics is None:
            metrics = ["rmse", "mae", "r2"]

        leaderboards = {}

        for metric in metrics:
            if metric not in self.df.columns:
                continue

            # First average over noises within each (benchmark, model)
            bench_avg = self.df.groupby(["benchmark", "model_variant"])[metric].mean().reset_index()

            # Then aggregate over benchmarks
            leaderboard = self._aggregate_metric(bench_avg, metric)
            leaderboard["rank"] = range(1, len(leaderboard) + 1)

            leaderboards[metric] = leaderboard

            path = self.output_dir / f"leaderboard_benchmark_avg_{metric}.csv"
            leaderboard.to_csv(path, index=False)

        return leaderboards

    # -------------------------------------------------------------------------
    # B) Pivot Tables
    # -------------------------------------------------------------------------

    def generate_pivot_tables(self, metrics: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate pivot tables: benchmark × model_variant for each metric.
        """
        if metrics is None:
            metrics = ["rmse", "mae", "r2", "nlpd", "coverage_95", "calibration_error_95", "sharpness"]

        pivots = {}

        for metric in metrics:
            if metric not in self.df.columns:
                continue

            # Average over noise types
            avg_df = self.df.groupby(["benchmark", "model_variant"])[metric].mean().reset_index()
            pivot = avg_df.pivot(index="benchmark", columns="model_variant", values=metric)

            pivots[metric] = pivot

            path = self.output_dir / f"pivot_benchmark_model_{metric}.csv"
            pivot.to_csv(path)

        logging.info(f"Generated {len(pivots)} pivot tables")
        return pivots

    def generate_pivot_by_noise(self, metric: str = "rmse") -> Dict[str, pd.DataFrame]:
        """
        Generate separate pivot tables for each noise type.
        """
        pivots = {}

        for noise in self.df["noise_type"].unique():
            noise_df = self.df[self.df["noise_type"] == noise]

            # Average over runs if multiple
            avg_df = noise_df.groupby(["benchmark", "model_variant"])[metric].mean().reset_index()
            pivot = avg_df.pivot(index="benchmark", columns="model_variant", values=metric)

            pivots[noise] = pivot

            path = self.output_dir / f"pivot_{metric}_{noise}.csv"
            pivot.to_csv(path)

        return pivots

    # -------------------------------------------------------------------------
    # C) Wins / Ties / Losses
    # -------------------------------------------------------------------------

    def generate_wins_matrix(self, metric: str = "rmse",
                              tolerance: float = 0.001) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate wins/ties/losses matrix for model pairs.

        Args:
            metric: Metric to compare (lower is better except r2)
            tolerance: Relative tolerance for ties

        Returns:
            Tuple of (wins_matrix, summary_df)
        """
        lower_better = metric != "r2"

        # Get average metric per (benchmark, noise_type, model_variant)
        avg_df = self.df.groupby(["benchmark", "noise_type", "model_variant"])[metric].mean().reset_index()

        models = sorted(self.df["model_variant"].unique())
        n_models = len(models)

        # Initialize matrices
        wins = np.zeros((n_models, n_models), dtype=int)
        ties = np.zeros((n_models, n_models), dtype=int)
        losses = np.zeros((n_models, n_models), dtype=int)

        # Compare all pairs on each (benchmark, noise)
        for (bench, noise), group in avg_df.groupby(["benchmark", "noise_type"]):
            scores = dict(zip(group["model_variant"], group[metric]))

            for i, m1 in enumerate(models):
                for j, m2 in enumerate(models):
                    if i >= j or m1 not in scores or m2 not in scores:
                        continue

                    s1, s2 = scores[m1], scores[m2]
                    if pd.isna(s1) or pd.isna(s2):
                        continue

                    rel_diff = abs(s1 - s2) / max(abs(s1), abs(s2), 1e-10)

                    if rel_diff < tolerance:
                        ties[i, j] += 1
                        ties[j, i] += 1
                    elif (s1 < s2) == lower_better:
                        wins[i, j] += 1
                        losses[j, i] += 1
                    else:
                        losses[i, j] += 1
                        wins[j, i] += 1

        # Create DataFrames
        wins_df = pd.DataFrame(wins, index=models, columns=models)

        # Summary: total wins for each model
        summary = pd.DataFrame({
            "model_variant": models,
            "total_wins": wins.sum(axis=1),
            "total_ties": ties.sum(axis=1) // 2,  # Avoid double counting
            "total_losses": losses.sum(axis=1),
        }).sort_values("total_wins", ascending=False)

        # Save
        wins_df.to_csv(self.output_dir / f"wins_matrix_{metric}.csv")
        summary.to_csv(self.output_dir / f"wins_summary_{metric}.csv", index=False)

        logging.info(f"Generated wins matrix for {metric}")
        return wins_df, summary

    # -------------------------------------------------------------------------
    # D) Top-1 Model per Benchmark/Noise
    # -------------------------------------------------------------------------

    def generate_top1_table(self, metric: str = "rmse") -> pd.DataFrame:
        """
        Generate table of best model per (benchmark, noise_type).
        """
        lower_better = metric != "r2"

        # Average over runs
        avg_df = self.df.groupby(["benchmark", "noise_type", "model_variant"])[metric].mean().reset_index()

        results = []
        for (bench, noise), group in avg_df.groupby(["benchmark", "noise_type"]):
            sorted_group = group.sort_values(metric, ascending=lower_better)

            if len(sorted_group) >= 1:
                best = sorted_group.iloc[0]
                second = sorted_group.iloc[1] if len(sorted_group) >= 2 else None

                result = {
                    "benchmark": bench,
                    "noise_type": noise,
                    "best_model": best["model_variant"],
                    f"best_{metric}": best[metric],
                }

                if second is not None:
                    result["second_model"] = second["model_variant"]
                    result[f"second_{metric}"] = second[metric]
                    result["gap"] = abs(second[metric] - best[metric])
                    result["gap_pct"] = abs(second[metric] - best[metric]) / max(abs(best[metric]), 1e-10) * 100

                results.append(result)

        top1_df = pd.DataFrame(results)
        top1_df.to_csv(self.output_dir / f"top1_model_{metric}.csv", index=False)

        return top1_df

    # -------------------------------------------------------------------------
    # E) Time vs Performance
    # -------------------------------------------------------------------------

    def generate_time_table(self) -> pd.DataFrame:
        """
        Generate timing summary table.
        """
        time_agg = self.df.groupby("model_variant").agg({
            "fit_time_s": ["mean", "std", lambda x: x.quantile(0.95), "count"],
            "predict_time_s": ["mean", "std"],
            "rmse": "mean",
            "mae": "mean",
        }).reset_index()

        # Flatten column names
        time_agg.columns = [
            "model_variant",
            "fit_time_mean", "fit_time_std", "fit_time_p95", "n_samples",
            "predict_time_mean", "predict_time_std",
            "rmse_mean", "mae_mean"
        ]

        time_agg = time_agg.sort_values("fit_time_mean")
        time_agg.to_csv(self.output_dir / "time_performance_summary.csv", index=False)

        return time_agg

    # -------------------------------------------------------------------------
    # Generate All
    # -------------------------------------------------------------------------

    def generate_all_tables(self) -> Dict[str, Any]:
        """Generate all standard tables."""
        logging.info("Generating all benchmark tables...")

        results = {
            "leaderboard_global": self.generate_global_leaderboard(),
            "leaderboard_benchmark_avg": self.generate_leaderboard_by_benchmark_avg(),
            "pivots": self.generate_pivot_tables(),
            "pivots_by_noise": self.generate_pivot_by_noise(),
            "wins_rmse": self.generate_wins_matrix("rmse"),
            "wins_mae": self.generate_wins_matrix("mae"),
            "top1_rmse": self.generate_top1_table("rmse"),
            "top1_mae": self.generate_top1_table("mae"),
            "time_summary": self.generate_time_table(),
        }

        logging.info(f"All tables saved to: {self.output_dir}")
        return results


# =============================================================================
# PASO 2: GLOBAL VISUALIZATIONS
# =============================================================================

class BenchmarkGlobalPlotter:
    """
    Generates global (cross-benchmark) visualizations.
    """

    def __init__(self, master_df: pd.DataFrame, output_dir: Path):
        self.df = master_df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_color(self, model_variant: str) -> str:
        """Get consistent color for model variant."""
        if model_variant in MODEL_VARIANT_PALETTE:
            return MODEL_VARIANT_PALETTE[model_variant]

        # Fallback based on family
        if model_variant.startswith("GP"):
            return "#2ecc71"
        elif model_variant.startswith("PLS"):
            return "#9b59b6"
        elif model_variant.startswith("Ridge"):
            return "#3498db"
        else:
            return "#95a5a6"

    # -------------------------------------------------------------------------
    # Heatmaps
    # -------------------------------------------------------------------------

    def plot_heatmap_by_noise(self, metric: str = "rmse",
                               use_log: bool = False,
                               save: bool = True) -> Dict[str, plt.Figure]:
        """
        Generate heatmap of metric (benchmark × model_variant) per noise type.
        """
        figures = {}

        for noise in sorted(self.df["noise_type"].unique()):
            noise_df = self.df[self.df["noise_type"] == noise]

            # Average over runs
            pivot = noise_df.groupby(["benchmark", "model_variant"])[metric].mean().unstack()

            if pivot.empty:
                continue

            # Determine scale
            fig, ax = plt.subplots(figsize=(12, 8))

            data = pivot.values
            if use_log and np.nanmin(data) > 0:
                norm = LogNorm(vmin=np.nanmin(data[data > 0]), vmax=np.nanmax(data))
                cmap = "viridis_r" if metric != "r2" else "viridis"
            else:
                norm = None
                cmap = "RdYlGn" if metric == "r2" else "RdYlGn_r"

            sns.heatmap(pivot, annot=True, fmt=".3g", cmap=cmap, norm=norm,
                        ax=ax, linewidths=0.5, cbar_kws={"label": metric.upper()})

            ax.set_title(f"{metric.upper()} Heatmap - {noise}\n(Lower is better)" if metric != "r2"
                         else f"{metric.upper()} Heatmap - {noise}\n(Higher is better)",
                         fontsize=14, fontweight='bold')
            ax.set_xlabel("Model Variant")
            ax.set_ylabel("Benchmark")

            plt.tight_layout()

            if save:
                suffix = "_log" if use_log else ""
                path = self.output_dir / f"heatmap_{metric}_{noise}{suffix}.png"
                fig.savefig(path, bbox_inches='tight')
                fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')

            figures[noise] = fig

        logging.info(f"Generated {len(figures)} heatmaps for {metric}")
        return figures

    # -------------------------------------------------------------------------
    # Ranking Plots
    # -------------------------------------------------------------------------

    def plot_average_rank(self, metric: str = "rmse", save: bool = True) -> plt.Figure:
        """
        Bar plot of average rank across all benchmarks/noises.
        """
        lower_better = metric != "r2"

        # Compute rank within each (benchmark, noise)
        avg_df = self.df.groupby(["benchmark", "noise_type", "model_variant"])[metric].mean().reset_index()

        ranks = []
        for (bench, noise), group in avg_df.groupby(["benchmark", "noise_type"]):
            group = group.copy()
            group["rank"] = group[metric].rank(ascending=lower_better)
            ranks.append(group[["model_variant", "rank"]])

        all_ranks = pd.concat(ranks, ignore_index=True)
        avg_rank = all_ranks.groupby("model_variant")["rank"].agg(["mean", "std"]).reset_index()
        avg_rank = avg_rank.sort_values("mean")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = [self._get_model_color(m) for m in avg_rank["model_variant"]]
        bars = ax.barh(avg_rank["model_variant"], avg_rank["mean"],
                       color=colors, edgecolor='black', alpha=0.8)

        ax.errorbar(avg_rank["mean"], avg_rank["model_variant"], xerr=avg_rank["std"],
                    fmt='none', color='black', capsize=4)

        # Add rank values
        for bar, val in zip(bars, avg_rank["mean"]):
            ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=10)

        ax.set_xlabel("Average Rank (lower is better)")
        ax.set_title(f"Model Ranking by {metric.upper()}\n(Across all benchmarks and noise types)",
                     fontsize=14, fontweight='bold')
        ax.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Best possible')

        plt.tight_layout()

        if save:
            path = self.output_dir / f"ranking_avg_{metric}.png"
            fig.savefig(path, bbox_inches='tight')
            fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')

        return fig

    # -------------------------------------------------------------------------
    # Distribution Plots
    # -------------------------------------------------------------------------

    def plot_metric_distribution(self, metric: str = "rmse",
                                  use_log: bool = False,
                                  save: bool = True) -> plt.Figure:
        """
        Box/violin plot of metric distribution across all evaluations.
        """
        df = self.df[self.df[metric].notna()].copy()

        if df.empty:
            return None

        # Order by median
        order = df.groupby("model_variant")[metric].median().sort_values().index.tolist()

        fig, ax = plt.subplots(figsize=(14, 7))

        # Violin with box
        parts = ax.violinplot(
            [df[df["model_variant"] == m][metric].values for m in order],
            positions=range(len(order)),
            showmeans=False, showmedians=False, showextrema=False
        )

        # Color violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self._get_model_color(order[i]))
            pc.set_alpha(0.6)

        # Overlay box plot
        bp = ax.boxplot(
            [df[df["model_variant"] == m][metric].values for m in order],
            positions=range(len(order)),
            widths=0.3, patch_artist=True, showfliers=True
        )

        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(self._get_model_color(order[i]))
            box.set_alpha(0.8)

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha='right')

        if use_log:
            ax.set_yscale('log')
            ylabel = f"{metric.upper()} (log scale)"
        else:
            ylabel = metric.upper()

        ax.set_ylabel(ylabel)
        ax.set_title(f"Distribution of {metric.upper()} Across All Evaluations",
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            suffix = "_log" if use_log else ""
            path = self.output_dir / f"distribution_{metric}{suffix}.png"
            fig.savefig(path, bbox_inches='tight')

        return fig

    # -------------------------------------------------------------------------
    # Time vs Error Scatter
    # -------------------------------------------------------------------------

    def plot_time_vs_error(self, error_metric: str = "rmse",
                            save: bool = True) -> plt.Figure:
        """
        Scatter plot of fit_time vs error with Pareto frontier.
        """
        df = self.df.copy()
        df = df[df["fit_time_s"].notna() & df[error_metric].notna()]

        if df.empty:
            return None

        # Aggregate by model_variant
        agg = df.groupby("model_variant").agg({
            "fit_time_s": "mean",
            error_metric: "mean",
        }).reset_index()

        fig, ax = plt.subplots(figsize=(12, 8))

        # Scatter by noise type
        for noise in df["noise_type"].unique():
            noise_df = df[df["noise_type"] == noise]
            ax.scatter(noise_df["fit_time_s"], noise_df[error_metric],
                       alpha=0.3, s=30, marker="o",
                       color=NOISE_PALETTE.get(noise, "gray"),
                       label=f"{noise} (points)")

        # Overlay model variant averages
        for _, row in agg.iterrows():
            color = self._get_model_color(row["model_variant"])
            ax.scatter(row["fit_time_s"], row[error_metric],
                       s=200, marker="D", color=color, edgecolor='black',
                       linewidth=2, zorder=10)
            ax.annotate(row["model_variant"],
                        (row["fit_time_s"], row[error_metric]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=9, fontweight='bold')

        # Pareto frontier
        pareto = self._compute_pareto_frontier(
            agg["fit_time_s"].values, agg[error_metric].values
        )
        if len(pareto) > 1:
            pareto_sorted = sorted(pareto, key=lambda x: x[0])
            ax.plot([p[0] for p in pareto_sorted], [p[1] for p in pareto_sorted],
                    'g--', linewidth=2, label='Pareto frontier')

        ax.set_xlabel("Fit Time (seconds)")
        ax.set_ylabel(error_metric.upper())
        ax.set_title(f"Time-Error Trade-off\n(Diamonds = model averages)",
                     fontsize=14, fontweight='bold')
        ax.set_xscale('log')

        # Create legend
        noise_handles = [mpatches.Patch(color=NOISE_PALETTE.get(n, "gray"), label=n)
                         for n in df["noise_type"].unique()]
        ax.legend(handles=noise_handles, loc='upper right')

        plt.tight_layout()

        if save:
            path = self.output_dir / f"time_vs_{error_metric}.png"
            fig.savefig(path, bbox_inches='tight')

        return fig

    def _compute_pareto_frontier(self, x: np.ndarray, y: np.ndarray) -> List[Tuple[float, float]]:
        """Compute Pareto frontier (minimize both x and y)."""
        points = list(zip(x, y))
        pareto = []

        for i, (xi, yi) in enumerate(points):
            is_dominated = False
            for j, (xj, yj) in enumerate(points):
                if i != j and xj <= xi and yj <= yi and (xj < xi or yj < yi):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto.append((xi, yi))

        return pareto

    # -------------------------------------------------------------------------
    # Robustness to Noise
    # -------------------------------------------------------------------------

    def plot_noise_robustness(self, metric: str = "rmse",
                               save: bool = True) -> plt.Figure:
        """
        Slope graph showing how models degrade with different noise types.
        """
        # Order noises by severity (conceptually)
        noise_order = ["NoNoise", "GaussianNoise", "ProportionalNoise", "HeteroscedasticNoise"]
        noise_order = [n for n in noise_order if n in self.df["noise_type"].unique()]

        # Average by (model_variant, noise_type)
        avg = self.df.groupby(["model_variant", "noise_type"])[metric].mean().reset_index()

        models = sorted(self.df["model_variant"].unique())
        n_noises = len(noise_order)

        fig, ax = plt.subplots(figsize=(12, 8))

        for model in models:
            model_data = avg[avg["model_variant"] == model]
            x_vals = []
            y_vals = []

            for i, noise in enumerate(noise_order):
                val = model_data[model_data["noise_type"] == noise][metric].values
                if len(val) > 0:
                    x_vals.append(i)
                    y_vals.append(val[0])

            if len(x_vals) > 1:
                color = self._get_model_color(model)
                ax.plot(x_vals, y_vals, 'o-', color=color, linewidth=2,
                        markersize=8, label=model, alpha=0.8)

        ax.set_xticks(range(n_noises))
        ax.set_xticklabels(noise_order, rotation=30, ha='right')
        ax.set_xlabel("Noise Type")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Model Robustness to Noise ({metric.upper()})\nAveraged across benchmarks",
                     fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()

        if save:
            path = self.output_dir / f"noise_robustness_{metric}.png"
            fig.savefig(path, bbox_inches='tight')

        return fig

    # -------------------------------------------------------------------------
    # Generate All
    # -------------------------------------------------------------------------

    def generate_all_plots(self) -> Dict[str, Any]:
        """Generate all global plots."""
        logging.info("Generating global benchmark plots...")

        results = {}

        # Heatmaps
        results["heatmap_rmse"] = self.plot_heatmap_by_noise("rmse")
        results["heatmap_mae"] = self.plot_heatmap_by_noise("mae")
        results["heatmap_r2"] = self.plot_heatmap_by_noise("r2")

        # Rankings
        results["ranking_rmse"] = self.plot_average_rank("rmse")
        results["ranking_mae"] = self.plot_average_rank("mae")
        results["ranking_r2"] = self.plot_average_rank("r2")

        # Distributions
        results["dist_rmse"] = self.plot_metric_distribution("rmse")
        results["dist_mae"] = self.plot_metric_distribution("mae")
        results["dist_rmse_log"] = self.plot_metric_distribution("rmse", use_log=True)

        # Time vs Error
        results["time_vs_rmse"] = self.plot_time_vs_error("rmse")
        results["time_vs_mae"] = self.plot_time_vs_error("mae")

        # Robustness
        results["robustness_rmse"] = self.plot_noise_robustness("rmse")
        results["robustness_mae"] = self.plot_noise_robustness("mae")

        plt.close('all')
        logging.info(f"Global plots saved to: {self.output_dir}")
        return results


# =============================================================================
# PASO 3: PER-BENCHMARK DASHBOARDS
# =============================================================================

class BenchmarkDashboardGenerator:
    """
    Generates per-benchmark dashboard with figures and tables.
    """

    def __init__(self, master_df: pd.DataFrame, output_dir: Path):
        self.df = master_df.copy()
        self.output_dir = Path(output_dir)

    def _get_model_color(self, model_variant: str) -> str:
        """Get consistent color for model variant."""
        return MODEL_VARIANT_PALETTE.get(model_variant, "#7f8c8d")

    def generate_benchmark_dashboard(self, benchmark: str,
                                      noise_type: str = None) -> Dict[str, Any]:
        """
        Generate complete dashboard for a benchmark (optionally filtered by noise).
        """
        bench_df = self.df[self.df["benchmark"] == benchmark].copy()

        if noise_type:
            bench_df = bench_df[bench_df["noise_type"] == noise_type]
            subfolder = f"{benchmark}/{noise_type}"
        else:
            subfolder = benchmark

        output_path = self.output_dir / subfolder
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # 1. Bar chart RMSE/MAE
        results["bar_rmse"] = self._plot_metric_bar(bench_df, "rmse", output_path, benchmark, noise_type)
        results["bar_mae"] = self._plot_metric_bar(bench_df, "mae", output_path, benchmark, noise_type)

        # 2. R2 bar chart
        results["bar_r2"] = self._plot_metric_bar(bench_df, "r2", output_path, benchmark, noise_type)

        # 3. GP-specific uncertainty metrics
        gp_df = bench_df[bench_df["model_family"] == "GP"]
        if not gp_df.empty:
            results["gp_uncertainty"] = self._plot_gp_uncertainty_bars(gp_df, output_path, benchmark, noise_type)

        # 4. Export CSV
        results["table"] = self._export_benchmark_table(bench_df, output_path, benchmark, noise_type)

        # 5. Generate summary markdown
        results["summary"] = self._generate_benchmark_summary(bench_df, output_path, benchmark, noise_type)

        return results

    def _plot_metric_bar(self, df: pd.DataFrame, metric: str,
                         output_path: Path, benchmark: str,
                         noise_type: Optional[str]) -> plt.Figure:
        """Bar chart of metric by model."""
        if metric not in df.columns or df[metric].isna().all():
            return None

        ascending = metric != "r2"

        # Average if multiple runs
        avg_df = df.groupby("model_variant")[metric].agg(["mean", "std"]).reset_index()
        avg_df = avg_df.sort_values("mean", ascending=ascending)

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = [self._get_model_color(m) for m in avg_df["model_variant"]]
        bars = ax.bar(avg_df["model_variant"], avg_df["mean"],
                      color=colors, edgecolor='black', alpha=0.8)

        # Error bars
        if avg_df["std"].notna().any():
            ax.errorbar(range(len(avg_df)), avg_df["mean"], yerr=avg_df["std"],
                        fmt='none', color='black', capsize=4)

        # Highlight best
        best_idx = 0 if ascending else len(avg_df) - 1
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

        ax.set_xlabel("Model")
        ax.set_ylabel(metric.upper())

        title = f"{benchmark} - {metric.upper()}"
        if noise_type:
            title += f" ({noise_type})"
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        path = output_path / f"bar_{metric}.png"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

        return fig

    def _plot_gp_uncertainty_bars(self, gp_df: pd.DataFrame, output_path: Path,
                                   benchmark: str, noise_type: Optional[str]) -> plt.Figure:
        """Bar charts for GP uncertainty metrics."""
        metrics = ["nlpd", "coverage_95", "calibration_error_95", "sharpness"]
        available = [m for m in metrics if m in gp_df.columns and gp_df[m].notna().any()]

        if not available:
            return None

        n_metrics = len(available)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, available):
            avg_df = gp_df.groupby("model_variant")[metric].agg(["mean", "std"]).reset_index()

            colors = [self._get_model_color(m) for m in avg_df["model_variant"]]
            ax.bar(avg_df["model_variant"], avg_df["mean"], color=colors, edgecolor='black')

            if avg_df["std"].notna().any():
                ax.errorbar(range(len(avg_df)), avg_df["mean"], yerr=avg_df["std"],
                            fmt='none', color='black', capsize=3)

            # Add target line for coverage
            if metric == "coverage_95":
                ax.axhline(0.95, color='green', linestyle='--', label='Target (95%)')
                ax.legend()

            ax.set_xlabel("")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.tick_params(axis='x', rotation=45)

        title = f"{benchmark} - GP Uncertainty Metrics"
        if noise_type:
            title += f" ({noise_type})"
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        path = output_path / "gp_uncertainty_metrics.png"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)

        return fig

    def _export_benchmark_table(self, df: pd.DataFrame, output_path: Path,
                                 benchmark: str, noise_type: Optional[str]) -> pd.DataFrame:
        """Export detailed table for benchmark."""
        cols = ["model_variant", "mae", "rmse", "r2", "nlpd", "coverage_95",
                "calibration_error_95", "sharpness", "fit_time_s"]
        cols = [c for c in cols if c in df.columns]

        # Aggregate
        agg_df = df.groupby("model_variant")[cols[1:]].mean().reset_index()
        agg_df = agg_df.sort_values("rmse")

        path = output_path / "metrics_summary.csv"
        agg_df.to_csv(path, index=False)

        return agg_df

    def _generate_benchmark_summary(self, df: pd.DataFrame, output_path: Path,
                                     benchmark: str, noise_type: Optional[str]) -> str:
        """Generate markdown summary for benchmark."""
        avg_df = df.groupby("model_variant")[["rmse", "mae", "r2"]].mean()

        best_rmse_model = avg_df["rmse"].idxmin()
        best_rmse_val = avg_df.loc[best_rmse_model, "rmse"]

        # GP calibration check
        gp_df = df[df["model_family"] == "GP"]
        calibration_note = ""
        if not gp_df.empty and "coverage_95" in gp_df.columns:
            gp_cov = gp_df.groupby("model_variant")["coverage_95"].mean()
            for kernel, cov in gp_cov.items():
                if pd.notna(cov):
                    if cov > 0.98:
                        calibration_note += f"- {kernel}: **Over-confident** (coverage {cov:.1%})\n"
                    elif cov < 0.90:
                        calibration_note += f"- {kernel}: **Under-confident** (coverage {cov:.1%})\n"
                    else:
                        calibration_note += f"- {kernel}: Well-calibrated (coverage {cov:.1%})\n"

        summary = f"""# {benchmark} Summary

**Noise**: {noise_type or "All types"}

## Best Model
- **Winner**: {best_rmse_model} (RMSE = {best_rmse_val:.4f})

## GP Calibration
{calibration_note if calibration_note else "No GP models evaluated or no coverage data."}
"""

        path = output_path / "summary.md"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(summary)

        return summary

    def generate_all_dashboards(self) -> Dict[str, Dict]:
        """Generate dashboards for all benchmarks and noise combinations."""
        logging.info("Generating per-benchmark dashboards...")

        results = {}

        for benchmark in self.df["benchmark"].unique():
            # Overall (all noises)
            results[f"{benchmark}_all"] = self.generate_benchmark_dashboard(benchmark)

            # Per noise type
            for noise in self.df["noise_type"].unique():
                key = f"{benchmark}_{noise}"
                results[key] = self.generate_benchmark_dashboard(benchmark, noise)

        logging.info(f"Dashboards saved to: {self.output_dir}")
        return results


# =============================================================================
# PASO 4: GP DEEP DIVE
# =============================================================================

class GPDeepDiveAnalyzer:
    """
    Specialized analysis for Gaussian Process model variants.
    """

    def __init__(self, master_df: pd.DataFrame, output_dir: Path):
        self.df = master_df[master_df["model_family"] == "GP"].copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_kernel_color(self, kernel: str) -> str:
        """Get color for GP kernel variant."""
        return GP_KERNEL_PALETTE.get(kernel, "#2ecc71")

    # -------------------------------------------------------------------------
    # A) Kernel Comparison
    # -------------------------------------------------------------------------

    def plot_kernel_comparison_heatmap(self, save: bool = True) -> plt.Figure:
        """
        Heatmap: kernel × metric for each benchmark.
        """
        metrics = ["rmse", "nlpd", "coverage_95", "calibration_error_95", "sharpness"]
        available_metrics = [m for m in metrics if m in self.df.columns and self.df[m].notna().any()]

        if not available_metrics:
            logging.warning("No GP metrics available for heatmap")
            return None

        # Average across benchmarks and noises
        pivot = self.df.groupby("model_variant")[available_metrics].mean()

        fig, ax = plt.subplots(figsize=(12, 6))

        # Normalize each column for visualization
        pivot_norm = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-10)

        sns.heatmap(pivot_norm.T, annot=pivot.T, fmt=".3g", cmap="RdYlGn_r",
                    ax=ax, linewidths=0.5, cbar_kws={"label": "Normalized (0=best)"})

        ax.set_title("GP Kernel Comparison (Averaged across all benchmarks/noises)",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("GP Kernel")
        ax.set_ylabel("Metric")

        plt.tight_layout()

        if save:
            path = self.output_dir / "kernel_comparison_heatmap.png"
            fig.savefig(path, bbox_inches='tight')

        return fig

    def generate_kernel_ranking_table(self) -> pd.DataFrame:
        """
        Generate kernel ranking table per benchmark.
        """
        rankings = []

        for benchmark in self.df["benchmark"].unique():
            bench_df = self.df[self.df["benchmark"] == benchmark]

            avg = bench_df.groupby("model_variant")["rmse"].mean()
            ranked = avg.rank()

            for kernel, rank in ranked.items():
                rankings.append({
                    "benchmark": benchmark,
                    "kernel": kernel,
                    "rmse": avg[kernel],
                    "rank": rank,
                })

        ranking_df = pd.DataFrame(rankings)

        # Pivot
        pivot = ranking_df.pivot(index="benchmark", columns="kernel", values="rank")
        pivot.to_csv(self.output_dir / "kernel_ranking_by_benchmark.csv")

        return ranking_df

    # -------------------------------------------------------------------------
    # B) Calibration vs Sharpness
    # -------------------------------------------------------------------------

    def plot_calibration_sharpness_scatter(self, save: bool = True) -> plt.Figure:
        """
        Scatter: sharpness vs calibration_error (ideal = bottom-left).
        """
        if "calibration_error_95" not in self.df.columns or "sharpness" not in self.df.columns:
            logging.warning("Calibration or sharpness data not available")
            return None

        df = self.df[self.df["calibration_error_95"].notna() & self.df["sharpness"].notna()]

        if df.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))

        kernels = df["model_variant"].unique()

        for kernel in kernels:
            kernel_df = df[df["model_variant"] == kernel]
            color = self._get_kernel_color(kernel)

            ax.scatter(kernel_df["sharpness"], kernel_df["calibration_error_95"],
                       s=80, alpha=0.7, label=kernel, color=color, edgecolor='black')

        # Ideal region indicator
        ax.axhline(0.05, color='green', linestyle='--', alpha=0.5, label='Good calibration (<5%)')
        ax.axvline(df["sharpness"].quantile(0.25), color='blue', linestyle=':', alpha=0.5)

        ax.set_xlabel("Sharpness (lower = tighter intervals)")
        ax.set_ylabel("Calibration Error 95% (lower = better calibrated)")
        ax.set_title("GP Calibration vs Sharpness Trade-off\n(Ideal: bottom-left corner)",
                     fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()

        if save:
            path = self.output_dir / "calibration_sharpness_scatter.png"
            fig.savefig(path, bbox_inches='tight')

        return fig

    def plot_coverage_reliability(self, save: bool = True) -> plt.Figure:
        """
        Reliability diagram: nominal coverage vs observed coverage.
        """
        coverage_cols = ["coverage_50", "coverage_90", "coverage_95"]
        available = [c for c in coverage_cols if c in self.df.columns and self.df[c].notna().any()]

        if not available:
            return None

        nominal = {"coverage_50": 0.50, "coverage_90": 0.90, "coverage_95": 0.95}

        fig, ax = plt.subplots(figsize=(10, 8))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

        for kernel in self.df["model_variant"].unique():
            kernel_df = self.df[self.df["model_variant"] == kernel]

            x_nom = []
            y_obs = []

            for cov_col in available:
                nom = nominal[cov_col]
                obs = kernel_df[cov_col].mean()
                if pd.notna(obs):
                    x_nom.append(nom)
                    y_obs.append(obs)

            if x_nom:
                color = self._get_kernel_color(kernel)
                ax.plot(x_nom, y_obs, 'o-', color=color, markersize=12,
                        linewidth=2, label=kernel, alpha=0.8)

        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.05)
        ax.set_xlabel("Nominal Coverage")
        ax.set_ylabel("Observed Coverage")
        ax.set_title("GP Reliability Diagram\n(Points above line = over-confident)",
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_aspect('equal')

        plt.tight_layout()

        if save:
            path = self.output_dir / "coverage_reliability.png"
            fig.savefig(path, bbox_inches='tight')

        return fig

    # -------------------------------------------------------------------------
    # C) Kernel Performance by Dimension
    # -------------------------------------------------------------------------

    def plot_kernel_by_dimension(self, metric: str = "rmse", save: bool = True) -> plt.Figure:
        """
        Line plot showing kernel performance vs benchmark dimension.
        """
        # Extract dimension from benchmark name
        dim_map = {
            "Forrester1D": 1,
            "Branin2D": 2,
            "SixHumpCamel2D": 2,
            "GoldsteinPrice2D": 2,
            "Hartmann3D": 3,
            "Ishigami3D": 3,
            "Hartmann6D": 6,
            "Borehole8D": 8,
            "WingWeight10D": 10,
        }

        df = self.df.copy()
        df["dimension"] = df["benchmark"].map(dim_map)
        df = df[df["dimension"].notna()]

        if df.empty:
            return None

        # Average by (kernel, dimension)
        avg = df.groupby(["model_variant", "dimension"])[metric].mean().reset_index()

        fig, ax = plt.subplots(figsize=(12, 7))

        for kernel in avg["model_variant"].unique():
            kernel_data = avg[avg["model_variant"] == kernel].sort_values("dimension")
            color = self._get_kernel_color(kernel)
            ax.plot(kernel_data["dimension"], kernel_data[metric], 'o-',
                    color=color, linewidth=2, markersize=10, label=kernel)

        ax.set_xlabel("Benchmark Dimension")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"GP Kernel Performance by Problem Dimension ({metric.upper()})",
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_xticks(sorted(df["dimension"].unique()))

        plt.tight_layout()

        if save:
            path = self.output_dir / f"kernel_by_dimension_{metric}.png"
            fig.savefig(path, bbox_inches='tight')

        return fig

    # -------------------------------------------------------------------------
    # Generate All
    # -------------------------------------------------------------------------

    def generate_all(self) -> Dict[str, Any]:
        """Generate all GP deep dive analyses."""
        logging.info("Generating GP deep dive analysis...")

        if self.df.empty:
            logging.warning("No GP results found for deep dive analysis")
            return {}

        results = {
            "kernel_heatmap": self.plot_kernel_comparison_heatmap(),
            "kernel_ranking": self.generate_kernel_ranking_table(),
            "calibration_sharpness": self.plot_calibration_sharpness_scatter(),
            "coverage_reliability": self.plot_coverage_reliability(),
            "kernel_by_dimension_rmse": self.plot_kernel_by_dimension("rmse"),
            "kernel_by_dimension_nlpd": self.plot_kernel_by_dimension("nlpd"),
        }

        plt.close('all')
        logging.info(f"GP deep dive saved to: {self.output_dir}")
        return results


# =============================================================================
# PASO 5: COVERAGE VERIFICATION
# =============================================================================

class CoverageVerifier:
    """
    Verifies completeness of benchmark evaluation coverage.
    """

    def __init__(self, master_df: pd.DataFrame, output_dir: Path):
        self.df = master_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_completeness_matrix(self) -> pd.DataFrame:
        """
        Generate matrix showing presence/absence of each (benchmark, noise, model) combination.
        """
        # Count occurrences
        presence = self.df.groupby(["benchmark", "noise_type", "model_variant"]).size().reset_index(name="count")

        # Pivot to matrix
        pivot = presence.pivot_table(
            index=["benchmark", "noise_type"],
            columns="model_variant",
            values="count",
            fill_value=0
        )

        # Convert to presence/absence
        pivot_presence = (pivot > 0).astype(int)

        path = self.output_dir / "completeness_matrix.csv"
        pivot_presence.to_csv(path)

        return pivot_presence

    def verify_expected_coverage(self) -> Dict[str, Any]:
        """
        Verify expected benchmarks, noises, and models are present.
        """
        expected_benchmarks = {
            "Forrester1D", "Branin2D", "SixHumpCamel2D", "GoldsteinPrice2D",
            "Hartmann3D", "Ishigami3D", "Hartmann6D", "Borehole8D", "WingWeight10D"
        }

        expected_noises = {"NoNoise", "GaussianNoise", "HeteroscedasticNoise", "ProportionalNoise"}

        expected_models = {"Dummy", "Ridge", "PLS_n2", "PLS_n3",
                          "GP_Matern32", "GP_Matern52", "GP_RBF"}

        found_benchmarks = set(self.df["benchmark"].unique())
        found_noises = set(self.df["noise_type"].unique())
        found_models = set(self.df["model_variant"].unique())

        results = {
            "benchmarks": {
                "expected": expected_benchmarks,
                "found": found_benchmarks,
                "missing": expected_benchmarks - found_benchmarks,
                "extra": found_benchmarks - expected_benchmarks,
            },
            "noises": {
                "expected": expected_noises,
                "found": found_noises,
                "missing": expected_noises - found_noises,
                "extra": found_noises - expected_noises,
            },
            "models": {
                "expected": expected_models,
                "found": found_models,
                "missing": expected_models - found_models,
                "extra": found_models - expected_models,
            },
        }

        # Write report
        path = self.output_dir / "coverage_verification.md"
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Coverage Verification Report\n\n")

            for category, data in results.items():
                f.write(f"## {category.title()}\n\n")
                f.write(f"- **Expected**: {len(data['expected'])}\n")
                f.write(f"- **Found**: {len(data['found'])}\n")

                if data['missing']:
                    f.write(f"- **⚠️ Missing**: {data['missing']}\n")
                else:
                    f.write("- **✓ All expected items present**\n")

                if data['extra']:
                    f.write(f"- **Additional items**: {data['extra']}\n")

                f.write("\n")

        return results


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_benchmark_report(session_name: str,
                               results_json_path: Path = None,
                               output_dir: Path = None) -> Path:
    """
    Generate complete benchmark visual evaluation report.

    Args:
        session_name: Name or partial name of benchmark session
        results_json_path: Explicit path to results JSON (auto-detected if None)
        output_dir: Output directory (auto-generated if None)

    Returns:
        Path to report directory
    """
    warnings.warn(
        "benchmark_visual_reporter.py esta deprecado. Redirigiendo a benchmark_report_active (report_v2).",
        DeprecationWarning,
        stacklevel=2,
    )
    from .benchmark_report_active.pipeline import generate_active_report

    redirected = generate_active_report(
        session=session_name or None,
        json_path=str(results_json_path) if results_json_path else None,
        output_dir=str(output_dir) if output_dir else None,
        phase="all",
        strict=False,
        solo_active=True,
    )
    return redirected

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    print("=" * 70)
    print(f"BENCHMARK VISUAL EVALUATION: {session_name}")
    print("=" * 70)

    # Auto-detect results file

    if results_json_path is None:
        benchmark_dir = LOGS_DIR / "benchmarks"
        # Search recursively in session subdirectories
        candidates = list(benchmark_dir.glob(f"**/*{session_name}*_results.json"))
        if not candidates:
            candidates = list(benchmark_dir.glob("**/*_results.json"))

        if not candidates:
            raise FileNotFoundError(f"No benchmark results found in {benchmark_dir}")

        results_json_path = sorted(candidates)[-1]  # Most recent
        logging.info(f"Auto-detected results: {results_json_path}")

    results_json_path = Path(results_json_path)

    # Setup output directory
    if output_dir is None:
        session_id = results_json_path.stem.replace("_results", "")
        output_dir = LOGS_DIR / "benchmarks" / session_id / "report"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PASO 0: Load and normalize data
    # =========================================================================
    print("\n[PASO 0] Loading and normalizing data...")

    loader = BenchmarkResultsLoader(results_json_path)
    master_df = loader.build_master_table()
    inventory = loader.get_inventory()

    # Export
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    loader.export_master_table(tables_dir)
    loader.export_quality_report(tables_dir)

    print(f"  - {inventory.n_results} results loaded")
    print(f"  - Benchmarks: {inventory.benchmarks}")
    print(f"  - Noises: {inventory.noise_types}")
    print(f"  - Model variants: {inventory.model_variants}")

    # =========================================================================
    # PASO 1: Generate tables
    # =========================================================================
    print("\n[PASO 1] Generating tables...")

    table_gen = BenchmarkTableGenerator(master_df, tables_dir)
    table_gen.generate_all_tables()

    # =========================================================================
    # PASO 2: Global visualizations
    # =========================================================================
    print("\n[PASO 2] Generating global visualizations...")

    global_figs_dir = output_dir / "figures" / "global"
    global_plotter = BenchmarkGlobalPlotter(master_df, global_figs_dir)
    global_plotter.generate_all_plots()

    # =========================================================================
    # PASO 3: Per-benchmark dashboards
    # =========================================================================
    print("\n[PASO 3] Generating per-benchmark dashboards...")

    benchmark_figs_dir = output_dir / "figures" / "by_benchmark"
    dashboard_gen = BenchmarkDashboardGenerator(master_df, benchmark_figs_dir)
    dashboard_gen.generate_all_dashboards()

    # =========================================================================
    # PASO 4: GP Deep Dive
    # =========================================================================
    print("\n[PASO 4] Generating GP deep dive analysis...")

    gp_dir = output_dir / "figures" / "gp_deep_dive"
    gp_analyzer = GPDeepDiveAnalyzer(master_df, gp_dir)
    gp_analyzer.generate_all()

    # =========================================================================
    # PASO 4.5: GP Prediction Visualizations (uncertainty bands)
    # =========================================================================
    print("\n[PASO 4.5] Generating GP prediction visualizations...")

    try:
        from .gp_visualization import GPVisualizationGenerator

        gp_viz_dir = output_dir / "figures" / "gp_predictions"
        gp_viz_generator = GPVisualizationGenerator(
            output_dir=gp_viz_dir,
            n_train=50,
            seed=42,
            n_restarts=3,  # Faster than default 5
        )

        # Use noise types from the actual benchmark
        noise_types_for_viz = [n for n in inventory.noise_types if n in ['NoNoise', 'GaussianNoise']]
        if not noise_types_for_viz:
            noise_types_for_viz = ['NoNoise']

        gp_viz_generator.generate_all_benchmarks(
            noise_types=noise_types_for_viz,
        )
        print(f"  - GP prediction visualizations saved to {gp_viz_dir}")
    except Exception as e:
        logging.warning(f"Could not generate GP visualizations: {e}")
        print(f"  - Skipped GP predictions (error: {e})")

    # =========================================================================
    # PASO 5: Coverage verification
    # =========================================================================
    print("\n[PASO 5] Verifying coverage...")

    verifier = CoverageVerifier(master_df, tables_dir)
    verifier.generate_completeness_matrix()
    coverage_results = verifier.verify_expected_coverage()

    # =========================================================================
    # Generate final report
    # =========================================================================
    print("\n[FINAL] Generating report index...")

    _generate_report_markdown(output_dir, inventory, coverage_results, loader.generate_quality_report())

    print("\n" + "=" * 70)
    print(f"REPORT COMPLETE: {output_dir}")
    print("=" * 70)

    return output_dir


def _generate_report_markdown(output_dir: Path, inventory: BenchmarkInventory,
                               coverage: Dict, quality: DataQualityReport):
    """Generate main report markdown file."""
    report_path = output_dir / "benchmark_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Benchmark Evaluation Report\n\n")
        f.write(f"**Session**: {inventory.session_name}\n\n")
        f.write(f"**Generated**: {inventory.timestamp}\n\n")
        f.write(f"**Total evaluation time**: {inventory.total_time_s:.2f}s\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Benchmarks**: {len(inventory.benchmarks)} ({', '.join(inventory.benchmarks)})\n")
        f.write(f"- **Noise types**: {len(inventory.noise_types)} ({', '.join(inventory.noise_types)})\n")
        f.write(f"- **Model variants**: {len(inventory.model_variants)}\n")
        f.write(f"- **Total evaluations**: {inventory.n_results}\n\n")

        # Model variants detail
        f.write("### Model Variants\n\n")
        for variant in sorted(inventory.model_variants):
            f.write(f"- {variant}\n")
        f.write("\n")

        # Tables index
        f.write("## Tables\n\n")
        f.write("| File | Description |\n")
        f.write("|------|-------------|\n")
        f.write("| [master_table.csv](tables/master_table.csv) | Complete normalized results |\n")
        f.write("| [leaderboard_global_rmse.csv](tables/leaderboard_global_rmse.csv) | Global ranking by RMSE |\n")
        f.write("| [pivot_benchmark_model_rmse.csv](tables/pivot_benchmark_model_rmse.csv) | RMSE by benchmark × model |\n")
        f.write("| [wins_matrix_rmse.csv](tables/wins_matrix_rmse.csv) | Pairwise wins/losses |\n")
        f.write("| [top1_model_rmse.csv](tables/top1_model_rmse.csv) | Best model per scenario |\n")
        f.write("| [completeness_matrix.csv](tables/completeness_matrix.csv) | Coverage check |\n\n")

        # Figures index
        f.write("## Global Figures\n\n")
        f.write("### Heatmaps\n\n")
        for noise in inventory.noise_types:
            f.write(f"- [RMSE Heatmap - {noise}](figures/global/heatmap_rmse_{noise}.png)\n")
        f.write("\n")

        f.write("### Rankings\n\n")
        f.write("- [Average Rank by RMSE](figures/global/ranking_avg_rmse.png)\n")
        f.write("- [Average Rank by R²](figures/global/ranking_avg_r2.png)\n\n")

        f.write("### Distributions\n\n")
        f.write("- [RMSE Distribution](figures/global/distribution_rmse.png)\n")
        f.write("- [MAE Distribution](figures/global/distribution_mae.png)\n\n")

        f.write("### Trade-offs\n\n")
        f.write("- [Time vs RMSE](figures/global/time_vs_rmse.png)\n")
        f.write("- [Noise Robustness](figures/global/noise_robustness_rmse.png)\n\n")

        # GP Deep Dive
        f.write("## GP Deep Dive\n\n")
        f.write("- [Kernel Comparison Heatmap](figures/gp_deep_dive/kernel_comparison_heatmap.png)\n")
        f.write("- [Calibration vs Sharpness](figures/gp_deep_dive/calibration_sharpness_scatter.png)\n")
        f.write("- [Coverage Reliability](figures/gp_deep_dive/coverage_reliability.png)\n")
        f.write("- [Kernel by Dimension](figures/gp_deep_dive/kernel_by_dimension_rmse.png)\n\n")

        # GP Prediction Visualizations
        f.write("## GP Prediction Visualizations\n\n")
        f.write("**Essential GP uncertainty visualizations** showing mean prediction, ±1σ and ±2σ bands, and training points.\n\n")
        f.write("- [Full Index](figures/gp_predictions/GP_VISUALIZATION_INDEX.md)\n\n")

        f.write("### Highlights by Dimensionality\n\n")
        f.write("**1D (Forrester):**\n")
        f.write("- [Kernel Comparison](figures/gp_predictions/Forrester1D/NoNoise/Forrester1D_NoNoise_kernel_comparison.png) - All kernels side-by-side\n")
        f.write("- [With Noise](figures/gp_predictions/Forrester1D/GaussianNoise/Forrester1D_GaussianNoise_kernel_comparison.png)\n\n")

        f.write("**2D (Branin):**\n")
        f.write("- [Matern52 Contour](figures/gp_predictions/Branin2D/NoNoise/Branin2D_NoNoise_Matern52_contour.png)\n")
        f.write("- [Matern52 Slices](figures/gp_predictions/Branin2D/NoNoise/Branin2D_NoNoise_Matern52_slices.png)\n\n")

        f.write("**High-D (Hartmann6D, Borehole8D):**\n")
        f.write("- [Hartmann6D Slices](figures/gp_predictions/Hartmann6D/NoNoise/Hartmann6D_NoNoise_Matern52_slices.png)\n")
        f.write("- [Borehole8D Slices](figures/gp_predictions/Borehole8D/NoNoise/Borehole8D_NoNoise_Matern52_slices.png)\n\n")

        # Per-benchmark
        f.write("## Per-Benchmark Dashboards\n\n")
        for benchmark in inventory.benchmarks:
            f.write(f"### {benchmark}\n\n")
            for noise in inventory.noise_types:
                f.write(f"- [{noise}](figures/by_benchmark/{benchmark}/{noise}/)\n")
            f.write("\n")

        # Data Quality
        f.write("## Data Quality Notes\n\n")
        f.write("See [data_quality_report.md](tables/data_quality_report.md) for details.\n\n")

        f.write("### Recommendations for Pipeline\n\n")
        for rec in quality.recommendations:
            f.write(f"- {rec}\n")

    logging.info(f"Report index saved: {report_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate benchmark visual evaluation report"
    )
    parser.add_argument(
        "--session", "-s",
        type=str,
        default="",
        help="Session name/pattern to match (default: most recent)"
    )
    parser.add_argument(
        "--json", "-j",
        type=str,
        default=None,
        help="Explicit path to results JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for report"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmark sessions"
    )

    args = parser.parse_args()

    if args.list:
        benchmark_dir = LOGS_DIR / "benchmarks"
        print("Available benchmark sessions:")
        for f in sorted(benchmark_dir.glob("*_results.json")):
            print(f"  - {f.stem.replace('_results', '')}")
    else:
        json_path = Path(args.json) if args.json else None
        output_path = Path(args.output) if args.output else None


        generate_benchmark_report(
            session_name=args.session,
            results_json_path=json_path,
            output_dir=output_path,
        )
