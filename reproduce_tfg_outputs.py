from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.utils.paths import LOGS_DIR, OUTPUTS_DIR, PROJECT_ROOT


TUNING_DIR = LOGS_DIR / "tuning" / "TFG_MAIN_real_case_hermetia_no_tpc_tuning"
EXPECTED_FEATURE_MODES = ("REDUCED_FEATURES", "FULL_FEATURES")
EXPECTED_TARGET_SLUGS = ("fcr", "quitina", "proteina")

ASSET_MANIFEST = [
    (
        OUTPUTS_DIR / "plots" / "TFG_MAIN_real_case_results_pov_ei_no_tpc" / "fig_01_lodo_generalization_vs_dummy.png",
        "fig_01_lodo_generalization_vs_dummy.png",
    ),
    (
        OUTPUTS_DIR / "plots" / "TFG_MAIN_real_case_results_pov_ei_no_tpc" / "fig_02_error_by_left_out_diet.png",
        "fig_02_error_by_left_out_diet.png",
    ),
    (
        OUTPUTS_DIR / "plots" / "TFG_MAIN_real_case_results_pov_ei_no_tpc" / "fig_04_kernel_family_mae.png",
        "fig_04_kernel_family_mae.png",
    ),
    (
        OUTPUTS_DIR / "plots" / "TFG_MAIN_real_case_results_pov_ei_no_tpc" / "fig_05_uncertainty_vs_error.png",
        "fig_05_uncertainty_vs_error.png",
    ),
    (
        OUTPUTS_DIR / "plots" / "TFG_MAIN_real_case_results_pov_ei_no_tpc" / "fig_06_prediction_intervals_by_diet.png",
        "fig_06_prediction_intervals_by_diet.png",
    ),
    (
        OUTPUTS_DIR / "plots" / "TFG_MAIN_real_case_results_pov_ei_no_tpc" / "fig_07_ei_pre_infill_candidates.png",
        "fig_07_ei_pre_infill_candidates.png",
    ),
    (
        OUTPUTS_DIR / "plots" / "TFG_MAIN_real_case_results_pov_ei_no_tpc" / "fig_08_ei_candidate_landscape.png",
        "fig_08_ei_candidate_landscape.png",
    ),
    (
        OUTPUTS_DIR / "plots" / "TFG_MAIN_real_case_audit_no_tpc" / "dataset_targets_by_diet.png",
        "app_real_dataset_targets_by_diet.png",
    ),
    (
        OUTPUTS_DIR / "plots" / "TFG_MAIN_real_case_audit_no_tpc" / "observed_equal_weight_ranking.png",
        "app_real_observed_equal_weight_ranking.png",
    ),
    (
        OUTPUTS_DIR / "plots" / "TFG_MAIN_real_case_results_pov_ei_no_tpc" / "fig_03_best_gp_parity_by_target.png",
        "app_real_best_gp_parity_by_target.png",
    ),
    (
        OUTPUTS_DIR
        / "logs"
        / "benchmarks"
        / "forrester_active_sweep_20260219_004822"
        / "ntrain_0_ninfill_10"
        / "report_v2"
        / "figures"
        / "gp_predictions"
        / "forrester"
        / "matrix_por_ruido_forrester_NoNoise.png",
        "app_forrester_gp_evolution_no_noise.png",
    ),
    (
        OUTPUTS_DIR
        / "logs"
        / "benchmarks"
        / "infill_4bench"
        / "report_v2"
        / "figures"
        / "space_understanding"
        / "forrester_ei_decision_synthesis_step5.png",
        "app_forrester_ei_decision_synthesis_step5.png",
    ),
]


def find_tfg_dir() -> Path:
    candidates = [p for p in PROJECT_ROOT.iterdir() if p.is_dir() and p.name.startswith("TFG_Jos")]
    if not candidates:
        raise FileNotFoundError("No TFG_Jos* directory found under project root.")
    return sorted(candidates, key=lambda p: p.name)[0]


def run_python_script(script_name: str) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / script_name)]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def check_tuning_logs() -> dict:
    missing = []
    counts = {"summaries": 0, "folds": 0}

    for feature_mode in EXPECTED_FEATURE_MODES:
        for target_slug in EXPECTED_TARGET_SLUGS:
            target_dir = TUNING_DIR / feature_mode / target_slug
            summary = target_dir / f"{target_slug}_summary.json"
            if not summary.exists():
                missing.append(str(summary.relative_to(PROJECT_ROOT)))
            else:
                counts["summaries"] += 1

            folds = list(target_dir.glob(f"{target_slug}_*_folds.csv"))
            counts["folds"] += len(folds)
            if len(folds) < 7:
                missing.append(f"{target_dir.relative_to(PROJECT_ROOT)} needs at least 7 fold CSV files")

    if missing:
        raise FileNotFoundError(
            "Missing or incomplete tuning logs. Run with --rerun-tuning or restore outputs/logs. "
            + json.dumps(missing, indent=2)
        )

    return counts


def sync_assets() -> list[dict]:
    tfg_dir = find_tfg_dir()
    assets_dir = tfg_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for source, dest_name in ASSET_MANIFEST:
        if not source.exists():
            raise FileNotFoundError(f"Required asset source does not exist: {source}")
        dest = assets_dir / dest_name
        shutil.copy2(source, dest)
        copied.append(
            {
                "source": str(source.relative_to(PROJECT_ROOT)),
                "dest": str(dest.relative_to(PROJECT_ROOT)),
                "bytes": dest.stat().st_size,
            }
        )
    return copied


def write_manifest(payload: dict) -> Path:
    path = OUTPUTS_DIR / "reproducibility_real_case_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce TFG real-case audit tables, final figures, and LaTeX assets."
    )
    parser.add_argument(
        "--rerun-tuning",
        action="store_true",
        help="Run the full LODO tuning before generating audit/figures. This can take a while.",
    )
    parser.add_argument(
        "--no-sync-assets",
        action="store_true",
        help="Generate outputs but do not copy figures into TFG_Jos*/assets.",
    )
    args = parser.parse_args()

    if args.rerun_tuning:
        run_python_script("run_exhaustive_tuning.py")

    tuning_counts = check_tuning_logs()
    run_python_script("audit_entomotive_real_case.py")
    run_python_script("generate_entomotive_pov_figures.py")

    copied_assets = [] if args.no_sync_assets else sync_assets()
    manifest = {
        "project_root": str(PROJECT_ROOT),
        "tuning_dir": str(TUNING_DIR.relative_to(PROJECT_ROOT)),
        "tuning_counts": tuning_counts,
        "assets_synced": copied_assets,
    }
    manifest_path = write_manifest(manifest)

    print(json.dumps({"manifest": str(manifest_path.relative_to(PROJECT_ROOT)), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
