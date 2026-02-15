#!/usr/bin/env python
# @author: José Arbelaez
"""
run_benchmark_evaluation.py

Main script for evaluating surrogate models on synthetic benchmarks.

Usage:
    python run_benchmark_evaluation.py                           # Run comprehensive evaluation (simple + active)
    python run_benchmark_evaluation.py --quick                   # Quick evaluation (few benchmarks)
    python run_benchmark_evaluation.py --benchmark forrester branin
    python run_benchmark_evaluation.py --cv-mode simple          # Only simple train/test split
    python run_benchmark_evaluation.py --cv-mode tuning          # KFold CV tuning on train set
    python run_benchmark_evaluation.py --cv-mode both            # Simple + tuning
    python run_benchmark_evaluation.py --cv-mode active          # Active learning with EI
    python run_benchmark_evaluation.py --cv-mode simple_active   # Simple + active
    python run_benchmark_evaluation.py --cv-mode tuning_active   # Tuning + active
    python run_benchmark_evaluation.py --cv-mode all             # Simple + tuning + active
    python run_benchmark_evaluation.py --samplers sobol random   # Use both samplers (default)
    python run_benchmark_evaluation.py --n-train 20 30 40 50 60  # Custom train sizes (default: dynamic by dimension)
    python run_benchmark_evaluation.py --help

Example Output:
    Creates files in outputs/logs/benchmarks/:
        - {session}_comprehensive_results.json: Full structured results
        - {session}_summary.csv: Summary table

    Results structure enables plotting evolution across:
        - Samplers (Sobol vs Random)
        - Training sizes (dynamic: 1*d, 3*d, 6*d, 9*d per benchmark dimension)
        - CV modes (simple vs tuning vs active)

Hyperparameter Grids:
    Per-benchmark grids are configured in src/configs/benchmark_grids.py
    Modify that file to customize hyperparameters for each benchmark.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import KFold, ParameterGrid

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
    save_benchmark_results,
    nested_lodo_tuning_benchmark,
    BenchmarkSuiteResults,
    BenchmarkResult,
)
from src.analysis.active_learning import run_active_evaluation

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
    ACTIVE_LEARNING_DEFAULTS,
    EVALUATION_DEFAULTS,
    get_noise_configs,
    get_n_train_for_dimension,
    get_active_learning_config,
    get_default_models,
    get_base_models,
)

from src.utils.paths import LOGS_DIR
from src.utils.tools import _to_jsonable
from src.utils.benchmark_evaluation_utils import (
    _get_noise_label,
    _save_pivot_tables,
    _print_comprehensive_summary,
    _print_active_summary,
    _serialize_params,
    _save_benchmark_tuning_results,
    _print_tuning_summary,
    _flatten_active_trajectories_for_csv,
    _collect_active_audit_records,
    _save_active_audit_logs_jsonl,
    _save_json_verified,
    run_quick_evaluation,
)


# =============================================================================
# MAIN EVALUATION FUNCTIONS
# =============================================================================

# TODO: These functions right now don't support the infill criteria neither the n_infill.
# This n_infill is being spent secuentially, 1 by 1.
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
    No Validation methods, It just use the models hardcoded in "Default Models"

    Note: Not necesarilly these are the best models.

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

    # Just take the base models (GP, Dummy) to tune by CV
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
        
        # EXPLAIN_AT: Sobol sampler and gaussian noise is hardcoded here.
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
            # This code is in src/configs/benchmark_grids.py
            # Each benchmark has a specific grid for each model.
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
                    "full_results": _to_jsonable(tuning_result),
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
    _save_json_verified(summary, summary_path, default=_to_jsonable)

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
    n_test: int = 200,
    n_groups: int = 5,
    cv_mode: str = "simple_active",
    n_infill: Optional[int] = None,
    ei_xi: float = 0.01,
    active_cand_mult: int = 500,
    active_cv_check_every: int = 5,
    active_train_all_models: bool = False,
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
        - Different CV modes (simple train/test, KFold tuning, active learning)

    Results are saved in a structured format optimized for plotting the evolution
    of model performance across these different configurations.

    Args:
        benchmarks: List of benchmark names (None = all available)
        samplers: List of samplers ["sobol", "lhs"] (default: ["sobol", "lhs"])
        n_train_list: List of training sizes (default: [20, 30, 40, 50, 60])
        n_test: Number of test samples (default: 200)
        n_groups: Synthetic groups setting for legacy LODO workflows (default: 5)
        cv_mode: "simple", "tuning", "both", "active", "simple_active", "tuning_active", or "all"
        n_infill: Active-learning infill budget (default: 5 * benchmark_dim)
        ei_xi: Exploration parameter for EI (default: 0.01)
        active_cand_mult: Candidate pool multiplier per dim (default: 500)
        active_cv_check_every: Audit interval in active mode (default: 5)
        active_train_all_models: If True, train all active models and compare.
            If False, train only one GP model in active mode.
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
        ...     cv_mode="all"
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
    if ei_xi is None:
        ei_xi = ACTIVE_LEARNING_DEFAULTS.get("ei_xi", 0.01)
    if active_cand_mult is None:
        active_cand_mult = ACTIVE_LEARNING_DEFAULTS.get("active_cand_mult", 500)
    if active_cv_check_every is None:
        active_cv_check_every = ACTIVE_LEARNING_DEFAULTS.get("active_cv_check_every", 5)

    # Validate cv_mode
    valid_cv_modes = ["simple", "tuning", "both", "active", "simple_active", "tuning_active", "all"]
    if cv_mode not in valid_cv_modes:
        raise ValueError(f"cv_mode must be one of {valid_cv_modes}, got '{cv_mode}'")
    if n_infill is not None and n_infill < 1:
        raise ValueError("n_infill must be >= 1 when provided")
    if ei_xi < 0:
        raise ValueError("ei_xi must be >= 0")
    if active_cand_mult < 1:
        raise ValueError("active_cand_mult must be >= 1")
    if active_cv_check_every < 0:
        raise ValueError("active_cv_check_every must be >= 0")

    if cv_mode == "both":
        cv_modes_to_run_global = ["simple", "tuning"]
    elif cv_mode == "simple_active":
        cv_modes_to_run_global = ["simple", "active"]
    elif cv_mode == "tuning_active":
        cv_modes_to_run_global = ["tuning", "active"]
    elif cv_mode == "all":
        cv_modes_to_run_global = ["simple", "tuning", "active"]
    else:
        cv_modes_to_run_global = [cv_mode]

    # Create output directory
    if output_name is None:
        output_name = f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = LOGS_DIR / "benchmarks" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total experiments
    n_benchmarks = len(benchmarks)
    n_samplers = len(samplers)
    n_train_sizes_info = "dynamic (1*d to 9*d)" if n_train_list is None else str(n_train_list)
    n_noise = len(noise_configs)
    n_cv_modes = len(cv_modes_to_run_global)

    # Calculate total configurations approximately
    if n_train_list is None:
        # Estimate: average of 4 sizes per benchmark
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
    print(f"  Groups setting (legacy LODO only): {n_groups}")
    infill_per_dim = ACTIVE_LEARNING_DEFAULTS.get("n_infill_per_dim", 5)
    print(
        f"  Active n_infill: "
        f"{n_infill if n_infill is not None else f'dynamic ({infill_per_dim}*d)'}"
    )
    print(f"  Active EI xi: {ei_xi}")
    print(f"\nTotal dataset configurations: {total_configs}")
    print(f"CV modes to run: {cv_modes_to_run_global}")
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
            "cv_modes_to_run": cv_modes_to_run_global,
            "noise_configs": noise_configs,
            "seed": seed,
            "scoring": scoring,
            "models": models_to_tune,
            "active_config": {
                "n_infill": n_infill,
                "n_infill_per_dim_default": ACTIVE_LEARNING_DEFAULTS.get("n_infill_per_dim", 5),
                "ei_xi": ei_xi,
                "active_cand_mult": active_cand_mult,
                "optimizer_budget_formula": "max(2000, active_cand_mult * dim)",
                "active_cv_check_every": active_cv_check_every,
                "active_switch_enable": ACTIVE_LEARNING_DEFAULTS.get("active_switch_enable", True),
                "active_switch_warmup_steps": ACTIVE_LEARNING_DEFAULTS.get("active_switch_warmup_steps", 5),
                "active_switch_min_improvement": ACTIVE_LEARNING_DEFAULTS.get("active_switch_min_improvement", 0.01),
                "active_switch_cooldown_steps": ACTIVE_LEARNING_DEFAULTS.get("active_switch_cooldown_steps", 5),
                "active_train_all_models": bool(active_train_all_models),
            },
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
            # n_train_list is calculated dynamically based on benchmark dimension
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
                    cv_modes_to_run = cv_modes_to_run_global

                    for current_cv_mode in cv_modes_to_run:
                        print(f"  Running {current_cv_mode} evaluation...")

                        if current_cv_mode == "simple":
                            # Simple train/test evaluation
                            cv_results = _run_simple_evaluation(
                                dataset=dataset,
                                models=get_default_models(),
                            )
                        elif current_cv_mode == "tuning":
                            # KFold CV tuning on train split (no group dependency),
                            # then final evaluation on the fixed benchmark test set.
                            cv_results = _run_tuning_evaluation(
                                dataset=dataset,
                                base_models=get_base_models(),
                                models_to_tune=models_to_tune,
                                benchmark_name=benchmark_name,
                                scoring=scoring,
                                n_jobs=n_jobs,
                                use_default_grids=use_default_grids,
                                seed=seed,
                            )
                        else:
                            if n_train == 0:
                                print("    WARNING: n_train=0 in active mode. Auto-correcting to 1 initial sample.")

                            active_cfg = get_active_learning_config(
                                dim=bench_dim,
                                n_infill=n_infill,
                                ei_xi=ei_xi,
                                active_cand_mult=active_cand_mult,
                                active_cv_check_every=active_cv_check_every,
                            )
                            noise_kwargs = {k: v for k, v in noise_cfg.items() if k != "type"}
                            if active_train_all_models:
                                active_models = get_default_models()
                            else:
                                base_models = get_base_models()
                                if "GP" not in base_models:
                                    raise ValueError(
                                        "Active single-model mode requires 'GP' in get_base_models()."
                                    )
                                active_models = {"GP": base_models["GP"]}
                            cv_results = run_active_evaluation(
                                dataset=dataset,
                                models=active_models,
                                benchmark_name=benchmark_name,
                                sampler=sampler,
                                noise_type=noise_type,
                                noise_kwargs=noise_kwargs,
                                n_infill=active_cfg["n_infill"],
                                xi=active_cfg["ei_xi"],
                                active_cand_mult=active_cfg["active_cand_mult"],
                                active_cv_check_every=active_cfg["active_cv_check_every"],
                                active_switch_enable=active_cfg.get("active_switch_enable", True),
                                active_switch_warmup_steps=active_cfg.get("active_switch_warmup_steps", 5),
                                active_switch_min_improvement=active_cfg.get("active_switch_min_improvement", 0.01),
                                active_switch_cooldown_steps=active_cfg.get("active_switch_cooldown_steps", 5),
                                use_default_grids=use_default_grids,
                                seed=seed,
                                verbose=True,
                            )

                        all_results["results"][sampler][n_train][benchmark_name][noise_label][current_cv_mode] = cv_results

                        # Add to summary rows
                        for model_name, model_results in cv_results.items():
                            if current_cv_mode == "active" and not model_results.get("active_supported", False):
                                continue
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
                                "active_supported": model_results.get("active_supported", True),
                            })

    total_time = time.perf_counter() - t_start
    all_results["metadata"]["total_time_s"] = total_time

    # Save full results
    results_path = output_dir / "comprehensive_results.json"
    _save_json_verified(all_results, results_path, default=_to_jsonable)

    # Save summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # Save active-learning artifacts (if active mode was run)
    active_traj_df = _flatten_active_trajectories_for_csv(all_results)
    if not active_traj_df.empty:
        active_traj_path = output_dir / "active_trajectory.csv"
        active_traj_df.to_csv(active_traj_path, index=False)

    active_audit_records = _collect_active_audit_records(all_results)
    if active_audit_records:
        active_audit_path = output_dir / "active_hparam_audit.jsonl"
        _save_active_audit_logs_jsonl(active_audit_records, active_audit_path)

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
    if not active_traj_df.empty:
        print("  - active_trajectory.csv (active-learning per-step trajectory)")
    if active_audit_records:
        print("  - active_hparam_audit.jsonl (periodic CV audit logs)")

    # Print quick summary
    _print_comprehensive_summary(summary_df)
    if not active_traj_df.empty:
        _print_active_summary(active_traj_df)

    return all_results

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
    seed: int,
) -> Dict[str, Any]:
    """
    Run KFold CV tuning on training data, then evaluate best params on test data.

    Returns:
        Dict mapping model_name -> final test metrics + CV selection diagnostics.
    """
    def _is_invalid_score(value: Any) -> bool:
        return value is None or (isinstance(value, float) and not np.isfinite(value))

    def _mean_std(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"mean": None, "std": None}
        arr = np.asarray(values, dtype=float)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    results = {}
    X_train = dataset.X_train
    y_train = dataset.y_train
    n_samples = int(len(y_train))
    n_splits = min(5, n_samples)

    if n_splits < 2:
        msg = f"Insufficient training samples for KFold tuning (n_train={n_samples}, need >=2)"
        for model_name in models_to_tune:
            results[model_name] = {"error": msg}
        return results

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for model_name in models_to_tune:
        if model_name not in base_models:
            continue

        base_model = base_models[model_name]
        grid = get_grid_for_evaluation(benchmark_name, model_name, use_default_grids)

        if grid is None or len(grid) == 0:
            results[model_name] = {"error": "No grid found"}
            continue

        try:
            param_candidates = list(ParameterGrid(grid))

            def _score_single_param(params: Dict[str, Any]) -> Dict[str, Any]:
                fold_scores: List[float] = []
                fold_errors: List[str] = []

                for train_idx, valid_idx in cv.split(X_train, y_train):
                    m = clone(base_model)
                    try:
                        m.set_params(**params)
                        m.fit(X_train[train_idx], y_train[train_idx])
                        mean_pred, std_pred = m.predict_dist(X_train[valid_idx])
                        fold_metrics = m.compute_metrics(
                            y_train[valid_idx],
                            mean_pred,
                            std_pred,
                            extended=(scoring == "nlpd"),
                        )
                        score = fold_metrics.get(scoring)
                        if _is_invalid_score(score):
                            raise ValueError(f"invalid_{scoring}_score")
                        fold_scores.append(float(score))
                    except Exception as exc:
                        fold_errors.append(str(exc))

                return {
                    "params_raw": params,
                    "mean_score": float(np.mean(fold_scores)) if fold_scores else None,
                    "std_score": float(np.std(fold_scores)) if fold_scores else None,
                    "n_valid_folds": len(fold_scores),
                    "errors": fold_errors,
                }

            if n_jobs is not None and n_jobs > 1:
                from joblib import Parallel, delayed
                ranking = Parallel(n_jobs=n_jobs, prefer="processes")(
                    delayed(_score_single_param)(params) for params in param_candidates
                )
            else:
                ranking = [_score_single_param(params) for params in param_candidates]

            ranking.sort(
                key=lambda rec: (
                    rec["mean_score"] is None,
                    float("inf") if rec["mean_score"] is None else rec["mean_score"],
                )
            )

            if not ranking or ranking[0]["mean_score"] is None:
                results[model_name] = {"error": f"KFold tuning failed for scoring={scoring}"}
                continue

            best = ranking[0]
            best_params = best["params_raw"]

            # Estimate variability for key metrics on CV folds using selected params.
            cv_metric_values: Dict[str, List[float]] = {
                "mae": [],
                "rmse": [],
                "r2": [],
                "nlpd": [],
                "coverage_95": [],
            }
            for train_idx, valid_idx in cv.split(X_train, y_train):
                m = clone(base_model)
                m.set_params(**best_params)
                m.fit(X_train[train_idx], y_train[train_idx])
                mean_pred, std_pred = m.predict_dist(X_train[valid_idx])
                fold_metrics = m.compute_metrics(
                    y_train[valid_idx],
                    mean_pred,
                    std_pred,
                    extended=True,
                )
                for metric_name in cv_metric_values:
                    value = fold_metrics.get(metric_name)
                    if value is not None and np.isfinite(value):
                        cv_metric_values[metric_name].append(float(value))

            # Fit selected model on full train and evaluate on held-out test.
            tuned_model = clone(base_model)
            tuned_model.set_params(**best_params)
            final_result = evaluate_model_on_dataset(
                model=tuned_model,
                dataset=dataset,
                store_predictions=False,
            )
            metrics = final_result.metrics

            results[model_name] = {
                "mae": metrics.mae,
                "mae_std": _mean_std(cv_metric_values["mae"])["std"],
                "rmse": metrics.rmse,
                "rmse_std": _mean_std(cv_metric_values["rmse"])["std"],
                "r2": metrics.r2,
                "r2_std": _mean_std(cv_metric_values["r2"])["std"],
                "nlpd": metrics.nlpd,
                "coverage_95": metrics.coverage_95,
                "n_folds": n_splits,
                "best_params": _serialize_params(best_params),
                "cv_primary_metric": scoring,
                "cv_primary_mean": best["mean_score"],
                "cv_primary_std": best["std_score"],
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}

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
  # Run comprehensive evaluation with defaults (simple + active, sobol+random, dynamic train sizes)
  python run_benchmark_evaluation.py

  # Quick test with few benchmarks
  python run_benchmark_evaluation.py --quick

  # Specific benchmarks with custom train sizes
  python run_benchmark_evaluation.py -b forrester branin --n-train 20 30 40 50

  # Default: simple + active with sobol+random samplers and dynamic n_train
  python run_benchmark_evaluation.py

  # Only Sobol sampling
  python run_benchmark_evaluation.py --samplers sobol

  # KFold tuning mode
  python run_benchmark_evaluation.py --cv-mode tuning --n-jobs 4

  # Active learning mode (EI infill)
  python run_benchmark_evaluation.py --cv-mode active --n-infill 15 --ei-xi 0.01

  # Active learning: train all models and compare
  python run_benchmark_evaluation.py --cv-mode active --active-train-all-models

  # Custom train sizes (overrides dynamic)
  python run_benchmark_evaluation.py --n-train 20 30 40 50 60

  # Full custom evaluation
  python run_benchmark_evaluation.py -b forrester branin hartmann3 \\
      --cv-mode all --samplers sobol lhs random --n-train 10 20 30
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
        default="simple_active",
        choices=["simple", "tuning", "both", "active", "simple_active", "tuning_active", "all"],
        help="CV mode: 'simple', 'tuning', 'both', 'active', 'simple_active', 'tuning_active', or 'all' (default: simple_active)"
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
        help="Training sample sizes. Default: dynamic [1*d, 3*d, 6*d, 9*d] per benchmark dimension"
    )
    data_group.add_argument(
        "--n-test",
        type=int,
        default=200,
        help="Number of test samples (default: 200)"
    )
    data_group.add_argument(
        "--n-groups",
        type=int,
        default=5,
        help="Synthetic groups setting for legacy LODO workflows (default: 5)"
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

    # Active learning configuration
    active_group = parser.add_argument_group("Active Learning")
    active_group.add_argument(
        "--n-infill",
        type=int,
        default=None,
        help="Sequential EI infill steps. Default: dynamic 5 * benchmark_dim"
    )
    active_group.add_argument(
        "--ei-xi",
        type=float,
        default=ACTIVE_LEARNING_DEFAULTS.get("ei_xi", 0.01),
        help=f"EI exploration parameter xi (default: {ACTIVE_LEARNING_DEFAULTS.get('ei_xi', 0.01)})"
    )
    active_group.add_argument(
        "--active-cand-mult",
        type=int,
        default=ACTIVE_LEARNING_DEFAULTS.get("active_cand_mult", 500),
        help=f"Continuous EI optimizer budget multiplier per dimension (default: {ACTIVE_LEARNING_DEFAULTS.get('active_cand_mult', 500)})"
    )
    active_group.add_argument(
        "--active-cv-check-every",
        type=int,
        default=ACTIVE_LEARNING_DEFAULTS.get("active_cv_check_every", 5),
        help=f"Run CV audit every N active steps (default: {ACTIVE_LEARNING_DEFAULTS.get('active_cv_check_every', 5)})"
    )
    active_group.add_argument(
        "--active-train-all-models",
        action="store_true",
        default=ACTIVE_LEARNING_DEFAULTS.get("active_train_all_models", False),
        help=(
            "If set, active mode trains all default active models and compares them. "
            "By default, active mode trains a single GP model."
        ),
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
        help="[DEPRECATED] Use --cv-mode tuning instead."
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
    infill_per_dim = ACTIVE_LEARNING_DEFAULTS.get("n_infill_per_dim", 5)
    print(
        f"  Active n_infill: "
        f"{args.n_infill if args.n_infill is not None else f'dynamic ({infill_per_dim}*d)'}"
    )
    print(f"  Active EI xi: {args.ei_xi}")
    print(f"  Active train all models: {args.active_train_all_models}")

    run_comprehensive_evaluation(
        benchmarks=args.benchmarks,
        samplers=args.samplers,
        n_train_list=args.n_train,
        n_test=args.n_test,
        n_groups=args.n_groups,
        cv_mode=args.cv_mode,
        n_infill=args.n_infill,
        ei_xi=args.ei_xi,
        active_cand_mult=args.active_cand_mult,
        active_cv_check_every=args.active_cv_check_every,
        active_train_all_models=args.active_train_all_models,
        seed=args.seed,
        output_name=args.output_name,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        use_default_grids=not args.no_default_grids,
        models_to_tune=args.models,
    )


if __name__ == "__main__":
    main()
