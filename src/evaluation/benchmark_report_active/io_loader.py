from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.paths import LOGS_DIR

from .schema import REQUIRED_TRAJECTORY_COLUMNS, missing_columns


@dataclass
class InputBundle:
    session_dir: Path
    results_json_path: Path
    active_trajectory_path: Path
    summary_csv_path: Optional[Path]
    audit_jsonl_path: Optional[Path]
    payload: Dict[str, Any]
    metadata: Dict[str, Any]
    trajectory_df: pd.DataFrame
    summary_df: pd.DataFrame
    audit_df: pd.DataFrame
    warnings: List[str] = field(default_factory=list)


def _resolve_session_dir(session: Optional[str], json_path: Optional[str]) -> Path:
    if json_path:
        p = Path(json_path).expanduser().resolve()
        if p.is_file():
            return p.parent
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Ruta no encontrada: {p}")

    if not session:
        default_dir = LOGS_DIR / "benchmarks" / "comprehensive"
        if (default_dir / "comprehensive_results.json").exists():
            return default_dir
        raise FileNotFoundError(
            "No se proporciono --session/--json y no existe outputs/logs/benchmarks/comprehensive."
        )

    sp = Path(session).expanduser()
    if sp.exists():
        return sp.resolve() if sp.is_dir() else sp.resolve().parent

    root = LOGS_DIR / "benchmarks"
    candidates = [
        d
        for d in root.glob(f"*{session}*")
        if d.is_dir() and (d / "comprehensive_results.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No se encontro sesion con patron '{session}' en {root}")
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_input_bundle(
    session: Optional[str],
    json_path: Optional[str],
    strict: bool = False,
) -> InputBundle:
    session_dir = _resolve_session_dir(session=session, json_path=json_path)
    results_json_path = session_dir / "comprehensive_results.json"
    active_trajectory_path = session_dir / "active_trajectory.csv"
    summary_csv_path = session_dir / "summary.csv"
    audit_jsonl_path = session_dir / "active_hparam_audit.jsonl"

    if not results_json_path.exists():
        raise FileNotFoundError(f"Falta archivo requerido: {results_json_path}")
    if not active_trajectory_path.exists():
        raise FileNotFoundError(f"Falta archivo requerido: {active_trajectory_path}")

    payload = _load_json(results_json_path)
    metadata = payload.get("metadata", {})
    trajectory_df = pd.read_csv(active_trajectory_path)
    summary_df = pd.read_csv(summary_csv_path) if summary_csv_path.exists() else pd.DataFrame()
    audit_df = _load_jsonl(audit_jsonl_path) if audit_jsonl_path.exists() else pd.DataFrame()

    warns: List[str] = []
    miss = missing_columns(trajectory_df.columns, REQUIRED_TRAJECTORY_COLUMNS)
    if miss:
        msg = f"active_trajectory.csv no contiene columnas requeridas: {miss}"
        if strict:
            raise ValueError(msg)
        warns.append(msg)

    if summary_df.empty:
        warns.append("summary.csv no encontrado; se derivaran metricas finales desde active_trajectory.csv")
        summary_csv_path = None
    if audit_df.empty:
        warns.append("active_hparam_audit.jsonl no encontrado o vacio; no se incluira detalle de auditoria.")
        audit_jsonl_path = None

    return InputBundle(
        session_dir=session_dir,
        results_json_path=results_json_path,
        active_trajectory_path=active_trajectory_path,
        summary_csv_path=summary_csv_path if summary_csv_path and summary_csv_path.exists() else None,
        audit_jsonl_path=audit_jsonl_path if audit_jsonl_path and audit_jsonl_path.exists() else None,
        payload=payload,
        metadata=metadata,
        trajectory_df=trajectory_df,
        summary_df=summary_df,
        audit_df=audit_df,
        warnings=warns,
    )
