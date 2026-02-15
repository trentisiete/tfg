import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .tools import _to_jsonable


def _get_noise_label(noise_cfg: Dict[str, Any]) -> str:
    """Generate a label for a noise configuration."""
    noise_type = noise_cfg.get("type", "none")
    if noise_type == "none":
        return "NoNoise"
    if noise_type == "gaussian":
        sigma = noise_cfg.get("sigma", 0.1)
        return f"Gaussian_s{sigma}"
    if noise_type == "heteroscedastic":
        return "Heteroscedastic"
    if noise_type == "proportional":
        return "Proportional"
    return noise_type


def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert params to a JSON-friendlier format."""
    result: Dict[str, Any] = {}
    for k, v in params.items():
        if hasattr(v, "__class__") and "kernel" in str(type(v).__name__).lower():
            result[k] = str(v)
        elif hasattr(v, "tolist"):
            result[k] = v.tolist()
        else:
            result[k] = v
    return result


def _get_most_common_params(chosen_params: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get most commonly chosen parameters across folds."""
    if not chosen_params:
        return {}

    # For simplicity, return the first fold's params.
    # A more sophisticated approach would compute mode.
    return _serialize_params(chosen_params[0]) if chosen_params else {}


def _save_pivot_tables(df: "pd.DataFrame", output_dir: Path):
    """Save pivot tables organized by different dimensions for easy plotting."""
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
                    aggfunc="mean",
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
                    aggfunc="mean",
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
                    aggfunc="mean",
                )
                pivot.to_csv(output_dir / f"pivot_by_cvmode_{metric}.csv")
            except Exception:
                pass


def _print_comprehensive_summary(df: "pd.DataFrame"):
    """Print a quick summary of the comprehensive evaluation."""
    if df.empty:
        print("\nNo results to summarize.")
        return

    print("\n--- Quick Summary ---")

    # Best model per benchmark (by MAE)
    if "mae" in df.columns:
        print("\nBest model per benchmark (by MAE, averaged across configs):")
        best = df.groupby(["benchmark", "model"])["mae"].mean().reset_index()
        best = best.dropna(subset=["mae"])
        if best.empty:
            print("  No valid MAE values available.")
        else:
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


def _save_benchmark_tuning_results(results: Dict[str, Any], path: Path):
    """Save benchmark tuning results to JSON."""
    _save_json_verified(results, path, default=_to_jsonable)


def _save_json_verified(
    payload: Dict[str, Any],
    path: Path,
    default: Optional[Callable[[Any], Any]] = None,
):
    """
    Save JSON atomically and validate by re-loading before final replace.

    This prevents partially-written or invalid JSON files from being left at `path`.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=default)
            f.write("\n")

        # Validation pass: ensure generated file is parseable JSON.
        with open(tmp_path, "r", encoding="utf-8") as f:
            json.load(f)

        tmp_path.replace(path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def _print_tuning_summary(all_results: Dict[str, Any], benchmarks: List[str], models: List[str]):
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
            rows.append(
                {
                    "benchmark": bench,
                    "model": model,
                    "mae": res.get("macro_mae_mean"),
                    "rmse": res.get("macro_rmse_mean"),
                    "r2": res.get("macro_r2_mean"),
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        print("\n--- Tuning Results Summary ---")
        print(df.to_string(index=False))
    else:
        print("\nNo results to display")


def _flatten_active_trajectories_for_csv(all_results: Dict[str, Any]) -> "pd.DataFrame":
    """Flatten active-learning trajectories into a tabular DataFrame."""
    import pandas as pd

    rows: List[Dict[str, Any]] = []

    nested_results = all_results.get("results", {})
    for sampler, by_ntrain in nested_results.items():
        for n_train, by_benchmark in by_ntrain.items():
            for benchmark, by_noise in by_benchmark.items():
                for noise_label, by_mode in by_noise.items():
                    active_payload = by_mode.get("active", {})
                    if not isinstance(active_payload, dict):
                        continue

                    for model_name, model_res in active_payload.items():
                        if not isinstance(model_res, dict):
                            continue
                        if not model_res.get("active_supported", False):
                            continue

                        for step in model_res.get("trajectory", []):
                            row = {
                                "sampler": sampler,
                                "n_train": n_train,
                                "benchmark": benchmark,
                                "noise": noise_label,
                                "cv_mode": "active",
                                "model": model_name,
                            }
                            row.update(step)
                            rows.append(row)

    return pd.DataFrame(rows)


def _collect_active_audit_records(all_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect full active audit records from results tree."""
    records: List[Dict[str, Any]] = []

    nested_results = all_results.get("results", {})
    for sampler, by_ntrain in nested_results.items():
        for n_train, by_benchmark in by_ntrain.items():
            for benchmark, by_noise in by_benchmark.items():
                for noise_label, by_mode in by_noise.items():
                    active_payload = by_mode.get("active", {})
                    if not isinstance(active_payload, dict):
                        continue

                    for model_name, model_res in active_payload.items():
                        if not isinstance(model_res, dict):
                            continue
                        for audit in model_res.get("hyperparam_audit", []):
                            rec = {
                                "sampler": sampler,
                                "n_train": n_train,
                                "benchmark": benchmark,
                                "noise": noise_label,
                                "cv_mode": "active",
                                "model": model_name,
                            }
                            rec.update(audit)
                            records.append(rec)

    return records


def _save_active_audit_logs_jsonl(records: List[Dict[str, Any]], path: Path):
    """Save active hyperparameter audit logs as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True, default=str) + "\n")


def _print_active_summary(df: "pd.DataFrame"):
    """Print concise active-learning summary from flattened trajectory rows."""
    if df.empty:
        print("\nNo active-learning trajectories to summarize.")
        return

    print("\n--- Active Learning Summary ---")
    max_step_by_model = (
        df.sort_values("step")
        .groupby(["benchmark", "model"], as_index=False)
        .tail(1)
    )
    for _, row in max_step_by_model.iterrows():
        rmse = row.get("rmse_test")
        best = row.get("incumbent_best")
        rmse_str = f"{rmse:.4f}" if rmse is not None else "nan"
        best_str = f"{best:.4f}" if best is not None else "nan"
        print(
            f"  {row['benchmark']:15s} -> {row['model']:20s} "
            f"(step={int(row['step'])}, incumbent={best_str}, rmse={rmse_str})"
        )


def run_quick_evaluation(seed: int = 42):
    """Run quick benchmark for rapid testing."""
    from src.analysis.benchmark_runner import run_quick_benchmark
    from src.configs import get_simple_models

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
