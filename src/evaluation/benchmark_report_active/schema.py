from __future__ import annotations

from typing import Iterable, List


REQUIRED_TRAJECTORY_COLUMNS = [
    "sampler",
    "n_train",
    "benchmark",
    "noise",
    "cv_mode",
    "model",
    "step",
    "n_train_current",
    "mae_test",
    "rmse_test",
    "r2_test",
    "nlpd_test",
    "coverage_95_test",
    "fit_time_step",
    "predict_time_step",
    "x_next",
    "y_next",
]

ACTIVE_MASTER_COLUMNS = [
    "sampler",
    "n_train",
    "benchmark",
    "noise",
    "cv_mode",
    "model",
    "step",
    "n_train_current",
    "mae",
    "rmse",
    "r2",
    "nlpd",
    "coverage_95",
    "fit_time",
    "predict_time",
    "total_time",
    "incumbent_best",
    "ei_next",
    "x_next",
    "y_next",
]

FINAL_SCENARIO_KEYS = ["sampler", "n_train", "benchmark", "noise", "cv_mode", "model"]


def missing_columns(columns: Iterable[str], required: Iterable[str]) -> List[str]:
    cols = set(columns)
    return [c for c in required if c not in cols]
