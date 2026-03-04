#!/usr/bin/env python
"""
Run a grid sweep for Forrester in active-learning mode:
    combinations of n_train x n_infill

Each combination is saved in its own session folder under one common root:
    outputs/logs/benchmarks/<root_name>/ntrain_<N>_ninfill_<K>/
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from run_benchmark_evaluation import run_comprehensive_evaluation
from src.evaluation.benchmark_report_active.pipeline import generate_active_report
from src.utils.paths import LOGS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep Forrester active-learning experiments across n_train and n_infill.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_forrester_active_sweep.py
  python run_forrester_active_sweep.py --n-train 0 5 10 --n-infill 5 10 15 20
  python run_forrester_active_sweep.py --n-train 0 5 10 15 --n-infill 5 10 --no-with-report
        """,
    )
    parser.add_argument(
        "--n-train",
        nargs="+",
        type=int,
        default=[0, 5, 10],
        help="Initial training sizes to sweep (default: 0 5 10)",
    )
    parser.add_argument(
        "--n-infill",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        help="Active infill budgets to sweep (default: 5 10 15 20)",
    )
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["sobol", "random"],
        choices=["sobol", "lhs", "random"],
        help="Samplers for each run (default: sobol random)",
    )
    parser.add_argument("--n-test", type=int, default=200, help="Test set size (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    parser.add_argument(
        "--active-train-all-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train all active models (default: true). Use --no-active-train-all-models for single GP.",
    )
    parser.add_argument("--ei-xi", type=float, default=0.01, help="EI xi parameter (default: 0.01)")
    parser.add_argument(
        "--active-cand-mult",
        type=int,
        default=500,
        help="Active EI optimizer budget multiplier per dimension (default: 500)",
    )
    parser.add_argument(
        "--active-cv-check-every",
        type=int,
        default=5,
        help="CV audit interval in active mode (default: 5)",
    )
    parser.add_argument(
        "--with-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate report_v2 for each run (default: true)",
    )
    parser.add_argument(
        "--report-phase",
        choices=["1", "2", "all"],
        default="2",
        help="Reporter phase when --with-report is enabled (default: 2)",
    )
    parser.add_argument("--report-dpi", type=int, default=220, help="Report PNG dpi (default: 220)")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Custom output root folder name under outputs/logs/benchmarks",
    )
    return parser


def _validate_lists(n_train_values: List[int], n_infill_values: List[int]) -> None:
    if not n_train_values:
        raise ValueError("n_train list cannot be empty")
    if not n_infill_values:
        raise ValueError("n_infill list cannot be empty")
    for n_train in n_train_values:
        if n_train < 0:
            raise ValueError(f"n_train must be >= 0, got {n_train}")
    for n_infill in n_infill_values:
        if n_infill < 1:
            raise ValueError(f"n_infill must be >= 1, got {n_infill}")


def _write_manifest(root_dir: Path, records: List[Dict[str, object]]) -> None:
    df = pd.DataFrame(records)
    df.to_csv(root_dir / "sweep_index.csv", index=False)

    lines = [
        "# Forrester Active Sweep",
        "",
        f"- Root: `{root_dir}`",
        f"- Total runs: {len(records)}",
        "",
        "## Runs",
        "",
        "| n_train | n_infill | session_dir | report_dir |",
        "|---:|---:|---|---|",
    ]
    for row in records:
        lines.append(
            f"| {row['n_train']} | {row['n_infill']} | `{row['session_dir']}` | `{row['report_dir']}` |"
        )
    (root_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    _validate_lists(args.n_train, args.n_infill)

    root_name = args.output_root or f"forrester_active_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root_dir = LOGS_DIR / "benchmarks" / root_name
    root_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    total = len(args.n_train) * len(args.n_infill)
    idx = 0

    print("=" * 80)
    print("FORRESTER ACTIVE SWEEP")
    print("=" * 80)
    print(f"Root folder: {root_dir}")
    print(f"n_train values: {args.n_train}")
    print(f"n_infill values: {args.n_infill}")
    print(f"Samplers: {args.samplers}")
    print(f"Train all models: {args.active_train_all_models}")
    print(f"Generate report: {args.with_report} (phase={args.report_phase})")
    print("=" * 80)

    for n_train in args.n_train:
        for n_infill in args.n_infill:
            idx += 1
            run_rel = f"{root_name}/ntrain_{n_train}_ninfill_{n_infill}"
            session_dir = LOGS_DIR / "benchmarks" / run_rel
            print(f"\n[{idx}/{total}] Running n_train={n_train}, n_infill={n_infill}")

            run_comprehensive_evaluation(
                benchmarks=["forrester"],
                samplers=list(args.samplers),
                n_train_list=[n_train],
                n_test=int(args.n_test),
                cv_mode="active",
                n_infill=int(n_infill),
                ei_xi=float(args.ei_xi),
                active_cand_mult=int(args.active_cand_mult),
                active_cv_check_every=int(args.active_cv_check_every),
                active_train_all_models=bool(args.active_train_all_models),
                seed=int(args.seed),
                output_name=run_rel,
            )

            report_dir = ""
            if args.with_report:
                out = generate_active_report(
                    session=str(session_dir),
                    phase=str(args.report_phase),
                    dpi=int(args.report_dpi),
                    solo_active=True,
                    samplers=list(args.samplers),
                    max_gp_benchmarks=1,
                    max_gp_steps=max(20, int(n_infill)),
                )
                report_dir = str(out)

            records.append(
                {
                    "n_train": int(n_train),
                    "n_infill": int(n_infill),
                    "session_dir": str(session_dir),
                    "report_dir": report_dir,
                }
            )

    _write_manifest(root_dir=root_dir, records=records)
    print("\nSweep completed.")
    print(f"Index: {root_dir / 'sweep_index.csv'}")
    print(f"Summary: {root_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

