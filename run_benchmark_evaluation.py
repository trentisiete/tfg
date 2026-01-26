#!/usr/bin/env python
# @author: José Arbelaez
"""
run_benchmark_evaluation.py

Main script for evaluating surrogate models on synthetic benchmarks.

Usage:
    python run_benchmark_evaluation.py                           # Run comprehensive evaluation (simple CV mode)
    python run_benchmark_evaluation.py --quick                   # Quick evaluation (few benchmarks)
    python run_benchmark_evaluation.py --benchmark forrester branin
    python run_benchmark_evaluation.py --cv-mode simple          # Only simple train/test split (default)
    python run_benchmark_evaluation.py --cv-mode tuning          # Only nested LODO tuning
    python run_benchmark_evaluation.py --cv-mode both            # Both modes
    python run_benchmark_evaluation.py --samplers sobol random   # Use both samplers (default)
    python run_benchmark_evaluation.py --n-train 20 30 40 50 60  # Custom train sizes (default: dynamic by dimension)
    python run_benchmark_evaluation.py --help

Example Output:
    Creates files in outputs/logs/benchmarks/:
        - {session}_comprehensive_results.json: Full structured results
        - {session}_summary.csv: Summary table

    Results structure enables plotting evolution across:
        - Samplers (Sobol vs Random)
        - Training sizes (dynamic: 3*d, 6*d, 9*d, 12*d per benchmark dimension)
        - CV modes (simple vs tuning)

Hyperparameter Grids:
    Per-benchmark grids are configured in src/configs/benchmark_grids.py
    Modify that file to customize hyperparameters for each benchmark.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.benchmarks import (
    list_benchmarks,
    generate_multi_benchmark_suite,
    generate_benchmark_dataset,
    get_benchmark,
    BENCHMARKS_LOW_DIM,
    BENCHMARKS_MEDIUM_DIM,
)

from src.analysis.benchmark_runner import (
    evaluate_models_on_suite,
    evaluate_model_on_dataset,
    run_quick_benchmark,
    save_benchmark_results,
    nested_lodo_tuning_benchmark,
    BenchmarkSuiteResults,
    BenchmarkResult,
)

from src.configs import (
    # Benchmark grids
    get_grid_for_evaluation,
    get_default_grid,
    list_configured_benchmarks,
    merge_with_defaults,
    BENCHMARK_GRIDS,
    DEFAULT_GRIDS,
    # Evaluation defaults
    DEFAULT_SAMPLERS,
    N_TRAIN_MULTIPLIERS,
    EVALUATION_DEFAULTS,
    get_noise_configs,
    get_n_train_for_dimension,
    get_default_models,
    get_base_models,
    get_simple_models,
)

from src.utils.paths import LOGS_DIR


# =============================================================================
# MAIN EVALUATION FUNCTIONS
# =============================================================================

def run_full_evaluation(
    benchmarks: list = None,
    n_train: int = 50,
    n_test: int = 300,
    seed: int = 42,
    output_name: str = None,
    sampler: str = "sobol",
):
    """
    Run comprehensive benchmark evaluation (simple train/test split).

    Args:
        benchmarks: List of benchmark names (None = all)
        n_train: Training samples per benchmark
        n_test: Test samples per benchmark
        seed: Random seed
        output_name: Custom name for output files
        sampler: Sampling strategy ("sobol" or "lhs")

    Returns:
        BenchmarkSuiteResults
    """
    if benchmarks is None:
        benchmarks = list_benchmarks()

    models = get_default_models()
    noise_configs = get_noise_configs(include_heteroscedastic=False)

    print("=" * 70)
    print("SIMPLE BENCHMARK EVALUATION (Train/Test Split)")
    print("=" * 70)
    print(f"\nBenchmarks ({len(benchmarks)}): {benchmarks}")
    print(f"Models ({len(models)}): {list(models.keys())}")
    print(f"Noise configs ({len(noise_configs)}): {[c['type'] for c in noise_configs]}")
    print(f"Train samples: {n_train}, Test samples: {n_test}")
    print(f"Sampler: {sampler}")
    print(f"Seed: {seed}")
    print("=" * 70)

    # Generate datasets
    print("\n[1/3] Generating benchmark datasets...")
    suite = generate_multi_benchmark_suite(
        benchmarks=benchmarks,
        n_train=n_train,
        n_test=n_test,
        sampler=sampler,
        noise_configs=noise_configs,
        seed=seed,
    )

    # Evaluate
    print("\n[2/3] Evaluating models...")
    results = evaluate_models_on_suite(
        models=models,
        suite=suite,
        verbose=True,
    )

    # Save results
    print("\n[3/3] Saving results...")
    if output_name is None:
        output_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = save_benchmark_results(
        results=results,
        output_dir=LOGS_DIR / "benchmarks",
        session_name=output_name,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- Model Ranking by RMSE (lower is better) ---")
    print(results.get_model_ranking("rmse"))

    print("\n--- Model Ranking by R² (higher is better) ---")
    print(results.get_model_ranking("r2", ascending=False))

    if any(r.metrics.coverage_95 is not None for r in results.results):
        print("\n--- Model Ranking by Calibration Error (lower is better) ---")
        print(results.get_model_ranking("calibration_error"))

    print(f"\nTotal evaluation time: {results.total_time:.1f}s")

    return results


def run_tuned_evaluation(
    benchmarks: list = None,
    n_train: int = 50,
    n_test: int = 300,
    n_groups: int = 5,
    seed: int = 42,
    output_name: str = None,
    scoring: str = "mae",
    n_jobs: int = 1,
    use_default_grids: bool = True,
    models_to_tune: List[str] = None,
):
    """
    Run benchmark evaluation with per-benchmark hyperparameter tuning.

    Uses grids defined in src/configs/benchmark_grids.py to tune each model
    specifically for each benchmark function.

    Args:
        benchmarks: List of benchmark names (None = all configured)
        n_train: Training samples per benchmark
        n_test: Test samples per benchmark
        n_groups: Number of groups for LODO cross-validation
        seed: Random seed
        output_name: Custom name for output files
        scoring: Metric for tuning ('mae', 'rmse', 'nlpd')
        n_jobs: Parallel jobs for grid search
        use_default_grids: Fall back to default grids if benchmark-specific not found
        models_to_tune: List of model names to tune (None = all base models)

    Returns:
        Dict with tuning results per benchmark
    """
    import json
    import time

    if benchmarks is None:
        benchmarks = list_configured_benchmarks()

    if models_to_tune is None:
        models_to_tune = list(get_base_models().keys())

    base_models = get_base_models()

    # Filter to requested models
    base_models = {k: v for k, v in base_models.items() if k in models_to_tune}

    print("=" * 70)
    print("TUNED BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"\nBenchmarks ({len(benchmarks)}): {benchmarks}")
    print(f"Models to tune ({len(base_models)}): {list(base_models.keys())}")
    print(f"Train samples: {n_train}, Test samples: {n_test}")
    print(f"Groups for LODO: {n_groups}")
    print(f"Scoring metric: {scoring}")
    print(f"Use default grids: {use_default_grids}")
    print(f"Seed: {seed}")
    print("=" * 70)

    # Output directory
    if output_name is None:
        output_name = f"tuned_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = LOGS_DIR / "benchmarks" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    t_start = time.perf_counter()

    for bench_idx, benchmark_name in enumerate(benchmarks, 1):
        print(f"\n{'='*70}")
        print(f"[{bench_idx}/{len(benchmarks)}] BENCHMARK: {benchmark_name.upper()}")
        print("=" * 70)

        # Get grids for this benchmark
        grids = merge_with_defaults(benchmark_name) if use_default_grids else {}

        # Generate dataset with groups for LODO
        print(f"\nGenerating dataset (n_train={n_train}, n_groups={n_groups})...")
        dataset = generate_benchmark_dataset(
            benchmark=benchmark_name,
            n_train=n_train,
            n_test=n_test,
            sampler="sobol",
            noise="gaussian",
            noise_kwargs={"sigma": 0.1},
            seed=seed,
            n_groups=n_groups,
        )

        benchmark_results = {
            "benchmark": benchmark_name,
            "n_train": n_train,
            "n_test": n_test,
            "n_groups": n_groups,
            "models": {},
        }

        for model_name, base_model in base_models.items():
            # Get grid for this benchmark/model combination
            grid = get_grid_for_evaluation(benchmark_name, model_name, use_default_grids)

            if grid is None or len(grid) == 0:
                print(f"\n  [{model_name}] No grid found, skipping tuning")
                continue

            grid_size = 1
            for v in grid.values():
                if isinstance(v, list):
                    grid_size *= len(v)

            print(f"\n  [{model_name}] Tuning with grid size {grid_size}...")
            print(f"      Grid params: {list(grid.keys())}")

            try:
                tuning_result = nested_lodo_tuning_benchmark(
                    base_model=base_model,
                    param_grid=grid,
                    dataset=dataset,
                    scoring=scoring,
                    n_jobs=n_jobs,
                )

                # Extract summary
                best_params = tuning_result.get("best_params", {})
                summary = tuning_result.get("summary", {})
                macro = summary.get("macro", {})

                benchmark_results["models"][model_name] = {
                    "best_params": _serialize_params(best_params),
                    "macro_mae_mean": macro.get("mae", {}).get("mean"),
                    "macro_rmse_mean": macro.get("rmse", {}).get("mean"),
                    "macro_r2_mean": macro.get("r2", {}).get("mean"),
                    "n_folds": summary.get("n_folds"),
                    "full_results": _serialize_tuning_results(tuning_result),
                }

                print(f"      Best MAE: {macro.get('mae', {}).get('mean', 'N/A'):.4f}")
                print(f"      Best R²: {macro.get('r2', {}).get('mean', 'N/A'):.4f}")

            except Exception as e:
                print(f"      ERROR: {e}")
                benchmark_results["models"][model_name] = {"error": str(e)}

        all_results[benchmark_name] = benchmark_results

        # Save intermediate results
        _save_benchmark_tuning_results(benchmark_results, output_dir / f"{benchmark_name}_tuning.json")

    total_time = time.perf_counter() - t_start

    # Save summary
    summary = {
        "session_name": output_name,
        "timestamp": datetime.now().isoformat(),
        "total_time_s": total_time,
        "benchmarks": benchmarks,
        "models": models_to_tune,
        "settings": {
            "n_train": n_train,
            "n_test": n_test,
            "n_groups": n_groups,
            "scoring": scoring,
            "use_default_grids": use_default_grids,
            "seed": seed,
        },
        "results": all_results,
    }

    summary_path = output_dir / "tuning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")
    print(f"Results saved to: {output_dir}")

    # Print summary table
    _print_tuning_summary(all_results, benchmarks, models_to_tune)

    return all_results


# =============================================================================
# COMPREHENSIVE EVALUATION (Multiple samplers, train sizes, CV modes)
# =============================================================================

def run_comprehensive_evaluation(
    benchmarks: List[str] = None,
    samplers: List[str] = None,
    n_train_list: List[int] = None,
    n_test: int = 300,
    n_groups: int = 5,
    cv_mode: str = "both",
    noise_configs: List[Dict] = None,
    seed: int = 42,
    output_name: str = None,
    scoring: str = "mae",
    n_jobs: int = 1,
    use_default_grids: bool = True,
    models_to_tune: List[str] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark evaluation with multiple configurations.

    This function allows systematic evaluation across:
        - Multiple samplers (Sobol, LHS)
        - Multiple training set sizes
        - Different CV modes (simple train/test, nested LODO tuning, or both)

    Results are saved in a structured format optimized for plotting the evolution
    of model performance across these different configurations.

    Args:
        benchmarks: List of benchmark names (None = all available)
        samplers: List of samplers ["sobol", "lhs"] (default: ["sobol", "lhs"])
        n_train_list: List of training sizes (default: [20, 30, 40, 50, 60])
        n_test: Number of test samples (default: 300)
        n_groups: Number of groups for LODO CV (default: 5)
        cv_mode: "simple" (train/test), "tuning" (nested LODO), or "both" (default)
        noise_configs: List of noise configurations (default: standard set)
        seed: Random seed (default: 42)
        output_name: Custom output directory name
        scoring: Metric for hyperparameter tuning (default: "mae")
        n_jobs: Parallel jobs for grid search (default: 1)
        use_default_grids: Fall back to default grids if no benchmark-specific
        models_to_tune: List of model names to tune (None = all)

    Returns:
        Dict with complete structured results for plotting

    Example:
        >>> results = run_comprehensive_evaluation(
        ...     benchmarks=["forrester", "branin"],
        ...     samplers=["sobol", "lhs"],
        ...     n_train_list=[20, 30, 40, 50],
        ...     cv_mode="both"
        ... )
    """
    import json
    import time
    import pandas as pd

    # Defaults (from src.configs.evaluation_defaults)
    if benchmarks is None:
        benchmarks = list_benchmarks()
    if samplers is None:
        samplers = DEFAULT_SAMPLERS.copy()
    # n_train_list se calcula dinámicamente por benchmark si es None
    # Usando N_TRAIN_MULTIPLIERS de configs
    if noise_configs is None:
        noise_configs = get_noise_configs(include_heteroscedastic=False)
    if models_to_tune is None:
        models_to_tune = list(get_base_models().keys())

    # Validate cv_mode
    valid_cv_modes = ["simple", "tuning", "both"]
    if cv_mode not in valid_cv_modes:
        raise ValueError(f"cv_mode must be one of {valid_cv_modes}, got '{cv_mode}'")

    # Create output directory
    if output_name is None:
        output_name = f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = LOGS_DIR / "benchmarks" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total experiments
    n_benchmarks = len(benchmarks)
    n_samplers = len(samplers)
    n_train_sizes_info = "dynamic (3*d to 12*d)" if n_train_list is None else str(n_train_list)
    n_noise = len(noise_configs)
    n_cv_modes = 2 if cv_mode == "both" else 1
    
    # Calcular total de configs aproximado
    if n_train_list is None:
        # Estimación: promedio de 4 tamaños por benchmark
        total_configs = n_benchmarks * n_samplers * 4 * n_noise
    else:
        total_configs = n_benchmarks * n_samplers * len(n_train_list) * n_noise

    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration Space:")
    print(f"  Benchmarks ({n_benchmarks}): {benchmarks}")
    print(f"  Samplers ({n_samplers}): {samplers}")
    print(f"  Train sizes: {n_train_sizes_info}")
    print(f"  Noise configs ({n_noise}): {[c['type'] for c in noise_configs]}")
    print(f"  CV mode: {cv_mode}")
    print(f"  Test samples: {n_test}")
    print(f"  LODO groups: {n_groups}")
    print(f"\nTotal dataset configurations: {total_configs}")
    print(f"CV modes to run: {['simple', 'tuning'] if cv_mode == 'both' else [cv_mode]}")
    print(f"Seed: {seed}")
    print("=" * 70)

    # Master results container
    all_results = {
        "metadata": {
            "session_name": output_name,
            "timestamp": datetime.now().isoformat(),
            "benchmarks": benchmarks,
            "samplers": samplers,
            "n_train_list": n_train_list if n_train_list is not None else "dynamic_by_dimension",
            "n_test": n_test,
            "n_groups": n_groups,
            "cv_mode": cv_mode,
            "noise_configs": noise_configs,
            "seed": seed,
            "scoring": scoring,
            "models": models_to_tune,
        },
        "results": {},  # Nested: sampler -> n_train -> benchmark -> noise -> cv_mode -> model
    }

    # For summary DataFrame
    summary_rows = []

    t_start = time.perf_counter()
    config_idx = 0

    # Iterate over all configurations
    for sampler in samplers:
        all_results["results"][sampler] = {}

        for bench_idx, benchmark_name in enumerate(benchmarks):
            # Calcular n_train_list dinámicamente si no se especificó
            bench_func = get_benchmark(benchmark_name)
            bench_dim = bench_func.dim
            
            if n_train_list is None:
                current_n_train_list = get_n_train_for_dimension(bench_dim)
            else:
                current_n_train_list = n_train_list
            
            print(f"\n{'='*70}")
            print(f"SAMPLER: {sampler.upper()} | BENCHMARK: {benchmark_name} (dim={bench_dim})")
            print(f"N_TRAIN values: {current_n_train_list}")
            print(f"{'='*70}")
            
            for n_train in current_n_train_list:
                if n_train not in all_results["results"][sampler]:
                    all_results["results"][sampler][n_train] = {}
                if benchmark_name not in all_results["results"][sampler][n_train]:
                    all_results["results"][sampler][n_train][benchmark_name] = {}
                
                for noise_cfg in noise_configs:
                    noise_type = noise_cfg.get("type", "none")
                    noise_label = _get_noise_label(noise_cfg)

                    config_idx += 1
                    print(f"\n[{config_idx}] {benchmark_name} (d={bench_dim}) | {noise_label} | {sampler} | n={n_train}")
                    
                    all_results["results"][sampler][n_train][benchmark_name][noise_label] = {}

                    # Generate dataset
                    try:
                        dataset = generate_benchmark_dataset(
                            benchmark=benchmark_name,
                            n_train=n_train,
                            n_test=n_test,
                            sampler=sampler,
                            noise=noise_type,
                            noise_kwargs={k: v for k, v in noise_cfg.items() if k != "type"},
                            n_groups=n_groups,
                            seed=seed,
                        )
                    except Exception as e:
                        print(f"  ERROR generating dataset: {e}")
                        continue

                    # Run CV modes
                    cv_modes_to_run = ["simple", "tuning"] if cv_mode == "both" else [cv_mode]

                    for current_cv_mode in cv_modes_to_run:
                        print(f"  Running {current_cv_mode} evaluation...")

                        if current_cv_mode == "simple":
                            # Simple train/test evaluation
                            cv_results = _run_simple_evaluation(
                                dataset=dataset,
                                models=get_default_models(),
                            )
                        else:
                            # Nested LODO tuning
                            cv_results = _run_tuning_evaluation(
                                dataset=dataset,
                                base_models=get_base_models(),
                                models_to_tune=models_to_tune,
                                benchmark_name=benchmark_name,
                                scoring=scoring,
                                n_jobs=n_jobs,
                                use_default_grids=use_default_grids,
                            )

                        all_results["results"][sampler][n_train][benchmark_name][noise_label][current_cv_mode] = cv_results

                        # Add to summary rows
                        for model_name, model_results in cv_results.items():
                            summary_rows.append({
                                "sampler": sampler,
                                "n_train": n_train,
                                "benchmark": benchmark_name,
                                "noise": noise_label,
                                "cv_mode": current_cv_mode,
                                "model": model_name,
                                "mae": model_results.get("mae"),
                                "rmse": model_results.get("rmse"),
                                "r2": model_results.get("r2"),
                                "nlpd": model_results.get("nlpd"),
                                "coverage_95": model_results.get("coverage_95"),
                                "fit_time": model_results.get("fit_time"),
                            })

    total_time = time.perf_counter() - t_start
    all_results["metadata"]["total_time_s"] = total_time

    # Save full results
    results_path = output_dir / "comprehensive_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=_json_serializer)

    # Save summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # Save pivot tables for easy plotting
    _save_pivot_tables(summary_df, output_dir)

    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - comprehensive_results.json (full structured results)")
    print(f"  - summary.csv (flat table for plotting)")
    print(f"  - pivot_*.csv (pivot tables by dimension)")

    # Print quick summary
    _print_comprehensive_summary(summary_df)

    return all_results


def _get_noise_label(noise_cfg: Dict) -> str:
    """Generate a label for a noise configuration."""
    noise_type = noise_cfg.get("type", "none")
    if noise_type == "none":
        return "NoNoise"
    elif noise_type == "gaussian":
        sigma = noise_cfg.get("sigma", 0.1)
        return f"Gaussian_s{sigma}"
    elif noise_type == "heteroscedastic":
        return "Heteroscedastic"
    elif noise_type == "proportional":
        return "Proportional"
    return noise_type


def _run_simple_evaluation(
    dataset,
    models: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run simple train/test evaluation on a dataset.

    Returns:
        Dict mapping model_name -> metrics
    """
    results = {}

    for model_name, model in models.items():
        try:
            result = evaluate_model_on_dataset(
                model=model,
                dataset=dataset,
                store_predictions=False,
            )

            metrics = result.metrics
            results[model_name] = {
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "r2": metrics.r2,
                "max_error": metrics.max_error,
                "nlpd": metrics.nlpd,
                "coverage_50": metrics.coverage_50,
                "coverage_90": metrics.coverage_90,
                "coverage_95": metrics.coverage_95,
                "calibration_error_95": metrics.calibration_error_95,
                "sharpness": metrics.sharpness,
                "fit_time": result.fit_time,
                "predict_time": result.predict_time,
                "model_params": _serialize_params(result.model_params),
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}

    return results


def _run_tuning_evaluation(
    dataset,
    base_models: Dict[str, Any],
    models_to_tune: List[str],
    benchmark_name: str,
    scoring: str,
    n_jobs: int,
    use_default_grids: bool,
) -> Dict[str, Any]:
    """
    Run nested LODO tuning evaluation on a dataset.

    Returns:
        Dict mapping model_name -> metrics (macro averaged)
    """
    results = {}

    for model_name in models_to_tune:
        if model_name not in base_models:
            continue

        base_model = base_models[model_name]
        grid = get_grid_for_evaluation(benchmark_name, model_name, use_default_grids)

        if grid is None or len(grid) == 0:
            results[model_name] = {"error": "No grid found"}
            continue

        try:
            tuning_result = nested_lodo_tuning_benchmark(
                base_model=base_model,
                param_grid=grid,
                dataset=dataset,
                scoring=scoring,
                n_jobs=n_jobs,
            )

            summary = tuning_result.get("summary", {})
            macro = summary.get("macro", {})

            results[model_name] = {
                "mae": macro.get("mae", {}).get("mean"),
                "mae_std": macro.get("mae", {}).get("std"),
                "rmse": macro.get("rmse", {}).get("mean"),
                "rmse_std": macro.get("rmse", {}).get("std"),
                "r2": macro.get("r2", {}).get("mean"),
                "r2_std": macro.get("r2", {}).get("std"),
                "nlpd": macro.get("nlpd", {}).get("mean"),
                "coverage_95": macro.get("coverage_95", {}).get("mean"),
                "n_folds": summary.get("n_folds"),
                "best_params": _get_most_common_params(tuning_result.get("chosen_params", [])),
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}

    return results


def _get_most_common_params(chosen_params: List[Dict]) -> Dict:
    """Get most commonly chosen parameters across folds."""
    if not chosen_params:
        return {}

    # For simplicity, return the first fold's params
    # A more sophisticated approach would compute mode
    return _serialize_params(chosen_params[0]) if chosen_params else {}


def _json_serializer(obj):
    """JSON serializer for objects not serializable by default."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)
    return str(obj)


def _save_pivot_tables(df: 'pd.DataFrame', output_dir: Path):
    """Save pivot tables organized by different dimensions for easy plotting."""
    import pandas as pd

    if df.empty:
        return

    # Pivot by n_train (for learning curve plots)
    for metric in ["mae", "rmse", "r2"]:
        if metric in df.columns:
            try:
                pivot = df.pivot_table(
                    index=["benchmark", "model", "sampler", "cv_mode"],
                    columns="n_train",
                    values=metric,
                    aggfunc="mean"
                )
                pivot.to_csv(output_dir / f"pivot_by_ntrain_{metric}.csv")
            except Exception:
                pass

    # Pivot by sampler (for sampler comparison)
    for metric in ["mae", "rmse", "r2"]:
        if metric in df.columns:
            try:
                pivot = df.pivot_table(
                    index=["benchmark", "model", "n_train", "cv_mode"],
                    columns="sampler",
                    values=metric,
                    aggfunc="mean"
                )
                pivot.to_csv(output_dir / f"pivot_by_sampler_{metric}.csv")
            except Exception:
                pass

    # Pivot by cv_mode (for CV comparison)
    for metric in ["mae", "rmse", "r2"]:
        if metric in df.columns:
            try:
                pivot = df.pivot_table(
                    index=["benchmark", "model", "n_train", "sampler"],
                    columns="cv_mode",
                    values=metric,
                    aggfunc="mean"
                )
                pivot.to_csv(output_dir / f"pivot_by_cvmode_{metric}.csv")
            except Exception:
                pass


def _print_comprehensive_summary(df: 'pd.DataFrame'):
    """Print a quick summary of the comprehensive evaluation."""
    if df.empty:
        print("\nNo results to summarize.")
        return

    print("\n--- Quick Summary ---")

    # Best model per benchmark (by MAE)
    if "mae" in df.columns:
        print("\nBest model per benchmark (by MAE, averaged across configs):")
        best = df.groupby(["benchmark", "model"])["mae"].mean().reset_index()
        best_per_bench = best.loc[best.groupby("benchmark")["mae"].idxmin()]
        for _, row in best_per_bench.iterrows():
            print(f"  {row['benchmark']:15s} -> {row['model']:20s} (MAE: {row['mae']:.4f})")

    # Effect of n_train
    if "n_train" in df.columns and "mae" in df.columns:
        print("\nAverage MAE by n_train:")
        by_ntrain = df.groupby("n_train")["mae"].mean()
        for n, mae in by_ntrain.items():
            print(f"  n_train={n:3d} -> MAE: {mae:.4f}")

    # Effect of sampler
    if "sampler" in df.columns and "mae" in df.columns:
        print("\nAverage MAE by sampler:")
        by_sampler = df.groupby("sampler")["mae"].mean()
        for s, mae in by_sampler.items():
            print(f"  {s:10s} -> MAE: {mae:.4f}")


def _serialize_params(params: dict) -> dict:
    """Convert params to JSON-serializable format."""
    result = {}
    for k, v in params.items():
        if hasattr(v, '__class__') and 'kernel' in str(type(v).__name__).lower():
            result[k] = str(v)
        elif hasattr(v, 'tolist'):
            result[k] = v.tolist()
        else:
            result[k] = v
    return result


def _serialize_tuning_results(results: dict) -> dict:
    """Serialize tuning results to JSON-compatible format."""
    import numpy as np

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        return obj

    return _convert(results)


def _save_benchmark_tuning_results(results: dict, path: Path):
    """Save benchmark tuning results to JSON."""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)


def _print_tuning_summary(all_results: dict, benchmarks: list, models: list):
    """Print summary table of tuning results."""
    import pandas as pd

    rows = []
    for bench in benchmarks:
        if bench not in all_results:
            continue
        for model in models:
            if model not in all_results[bench].get("models", {}):
                continue
            res = all_results[bench]["models"][model]
            rows.append({
                "benchmark": bench,
                "model": model,
                "mae": res.get("macro_mae_mean"),
                "rmse": res.get("macro_rmse_mean"),
                "r2": res.get("macro_r2_mean"),
            })

    if rows:
        df = pd.DataFrame(rows)
        print("\n--- Tuning Results Summary ---")
        print(df.to_string(index=False))
    else:
        print("\nNo results to display")


def run_quick_evaluation(seed: int = 42):
    """Run quick benchmark for rapid testing."""
    models = get_simple_models()

    results = run_quick_benchmark(
        models=models,
        benchmarks=["forrester", "branin", "hartmann3"],
        n_train=30,
        n_test=100,
        noise_sigma=0.1,
        seed=seed,
        verbose=True,
    )

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate surrogate models on synthetic benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive evaluation with defaults (both CV modes, sobol+lhs, multiple train sizes)
  python run_benchmark_evaluation.py

  # Quick test with few benchmarks
  python run_benchmark_evaluation.py --quick

  # Specific benchmarks with custom train sizes
  python run_benchmark_evaluation.py -b forrester branin --n-train 20 30 40 50

  # Default: simple CV mode with sobol+random samplers and dynamic n_train
  python run_benchmark_evaluation.py
  
  # Only Sobol sampling
  python run_benchmark_evaluation.py --samplers sobol

  # Nested LODO tuning mode
  python run_benchmark_evaluation.py --cv-mode tuning --n-jobs 4

  # Custom train sizes (overrides dynamic)
  python run_benchmark_evaluation.py --n-train 20 30 40 50 60
  
  # Full custom evaluation
  python run_benchmark_evaluation.py -b forrester branin hartmann3 \\
      --cv-mode both --samplers sobol lhs random --n-train 10 20 30
        """
    )

    # Mode selection
    mode_group = parser.add_argument_group("Evaluation Mode")
    mode_group.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick evaluation with few benchmarks (ignores other options)"
    )
    mode_group.add_argument(
        "--cv-mode",
        type=str,
        default="simple",
        choices=["simple", "tuning", "both"],
        help="CV mode: 'simple' (train/test split), 'tuning' (nested LODO), 'both' (default: simple)"
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--benchmarks", "-b",
        nargs="+",
        default=None,
        help=f"Benchmark names (default: all). Available: {list_benchmarks()}"
    )
    data_group.add_argument(
        "--samplers",
        nargs="+",
        default=["sobol", "random"],
        choices=["sobol", "lhs", "random"],
        help="Sampling strategies to use (default: sobol random)"
    )
    data_group.add_argument(
        "--n-train",
        nargs="+",
        type=int,
        default=None,
        help="Training sample sizes. Default: dynamic [3*d, 6*d, 9*d, 12*d] per benchmark dimension"
    )
    data_group.add_argument(
        "--n-test",
        type=int,
        default=300,
        help="Number of test samples (default: 300)"
    )
    data_group.add_argument(
        "--n-groups",
        type=int,
        default=5,
        help="Number of groups for LODO cross-validation (default: 5)"
    )
    data_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--models", "-m",
        nargs="+",
        default=None,
        help="Model names to evaluate (default: GP, Ridge, PLS, Dummy)"
    )
    model_group.add_argument(
        "--scoring",
        type=str,
        default="mae",
        choices=["mae", "rmse", "nlpd"],
        help="Metric for hyperparameter tuning (default: mae)"
    )
    model_group.add_argument(
        "--no-default-grids",
        action="store_true",
        help="Do not fall back to default grids (only use benchmark-specific)"
    )

    # Computation
    comp_group = parser.add_argument_group("Computation")
    comp_group.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for grid search (default: 1)"
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-name", "-o",
        type=str,
        default=None,
        help="Output directory name (default: auto-generated with timestamp)"
    )

    # Info commands
    info_group = parser.add_argument_group("Information")
    info_group.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List available benchmarks and exit"
    )
    info_group.add_argument(
        "--list-grids",
        action="store_true",
        help="List configured benchmark grids and exit"
    )

    # Legacy support
    parser.add_argument(
        "--tune", "-t",
        action="store_true",
        help="[DEPRECATED] Use --cv-mode tuning instead. Runs only nested LODO tuning."
    )

    args = parser.parse_args()

    # Info commands
    if args.list_benchmarks:
        print("Available benchmarks:")
        print(f"  Low-dimensional: {BENCHMARKS_LOW_DIM}")
        print(f"  Medium-dimensional: {BENCHMARKS_MEDIUM_DIM}")
        print(f"\nAll: {list_benchmarks()}")
        return

    if args.list_grids:
        print("Configured benchmark grids (src/configs/benchmark_grids.py):")
        print(f"\nBenchmarks with custom grids: {list_configured_benchmarks()}")
        print(f"\nDefault grids available for: {list(DEFAULT_GRIDS.keys())}")
        print("\nTo customize grids, edit: src/configs/benchmark_grids.py")
        return

    # Quick mode
    if args.quick:
        run_quick_evaluation(seed=args.seed)
        return

    # Legacy --tune flag
    if args.tune:
        print("WARNING: --tune is deprecated. Use --cv-mode tuning instead.\n")
        args.cv_mode = "tuning"

    # Run comprehensive evaluation
    print("\n" + "=" * 70)
    print("Starting Comprehensive Benchmark Evaluation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  CV Mode:    {args.cv_mode}")
    print(f"  Samplers:   {args.samplers}")
    print(f"  Train sizes: {args.n_train}")
    print(f"  Benchmarks: {args.benchmarks or 'all'}")
    print(f"  Models:     {args.models or 'all default'}")

    run_comprehensive_evaluation(
        benchmarks=args.benchmarks,
        samplers=args.samplers,
        n_train_list=args.n_train,
        n_test=args.n_test,
        n_groups=args.n_groups,
        cv_mode=args.cv_mode,
        seed=args.seed,
        output_name=args.output_name,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        use_default_grids=not args.no_default_grids,
        models_to_tune=args.models,
    )


if __name__ == "__main__":
    main()
