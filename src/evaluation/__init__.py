# @author: José Arbelaez
"""
Evaluation module for surrogate models.

Provides the active-learning benchmark report pipeline (report_v2), used to
generate the tables and figures consumed by the TFG.

Usage:
    from src.evaluation import generate_active_report
    generate_active_report("infill_4bench")

    Or from CLI:
        python -m src.evaluation.benchmark_report_active --session infill_4bench
"""

from .benchmark_report_active import (
    generate_active_report,
)

__all__ = [
    "generate_active_report",
]
