from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .aggregations import build_final_table, build_master_active_table, summarize_active_coverage
from .io_loader import load_input_bundle
from .plots_by_benchmark import generate_by_benchmark_outputs
from .plots_evolution import generate_evolution_plots
from .plots_global import generate_global_plots
from .plots_gp import generate_gp_predictions
from .plots_sampling import generate_sampling_effects_plots
from .styling import apply_publication_style
from .tables import generate_phase1_tables


def _ensure_dirs(base: Path) -> Dict[str, Path]:
    dirs = {
        "base": base,
        "tables": base / "tables",
        "by_benchmark": base / "figures" / "by_benchmark",
        "global": base / "figures" / "global",
        "sampling_effects": base / "figures" / "sampling_effects",
        "evolution": base / "figures" / "evolution",
        "overview_final": base / "figures" / "overview_final",
        "gp_predictions": base / "figures" / "gp_predictions",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_index(
    out_dir: Path,
    coverage: Dict[str, object],
    warnings: List[str],
    phase: str,
    tables: Dict[str, pd.DataFrame],
) -> None:
    lines: List[str] = [
        "# Benchmark Report v2 (Active-only)",
        "",
        "## Resumen",
        f"- Registros activos: {coverage.get('n_rows', 0)}",
        f"- Benchmarks: {coverage.get('n_benchmarks', 0)}",
        f"- Modelos: {coverage.get('n_models', 0)}",
        f"- Ruidos: {coverage.get('n_noises', 0)}",
        f"- Samplers: {coverage.get('n_samplers', 0)}",
        f"- Modo monomodelo: {coverage.get('single_model_mode', True)}",
        f"- Fase ejecutada: {phase}",
        "",
        "## Tablas",
    ]
    for name in sorted(tables.keys()):
        lines.append(f"- [**{name}**](tables/{name}.csv)")

    lines.extend(
        [
            "",
            "## Figuras",
            "- [Por benchmark](figures/by_benchmark/)",
            "- [Global](figures/global/)",
            "- [Evolucion](figures/evolution/)",
            "- [Sampling effects](figures/sampling_effects/)",
            "- [Overview final](figures/overview_final/)",
            "- [GP predictions](figures/gp_predictions/)",
            "",
            "## Migracion legacy -> nuevo",
            "- `benchmark_visual_reporter.py` ahora delega al pipeline active v2.",
            "- `benchmark_visual_reporter_multimode.py` ahora delega al pipeline active v2.",
            "",
        ]
    )
    if warnings:
        lines.append("## Avisos")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_min_outputs(out_dir: Path) -> None:
    required = [
        out_dir / "tables" / "leaderboard_global_mae.csv",
        out_dir / "tables" / "wins_summary_mae.csv",
        out_dir / "tables" / "top1_model_mae.csv",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Salida requerida no generada: {p}")
        pd.read_csv(p)


def generate_active_report(
    session: Optional[str] = None,
    json_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    phase: str = "all",
    topx: int = 10,
    strict: bool = False,
    dpi: int = 300,
    solo_active: bool = True,
    samplers: Optional[Iterable[str]] = None,
    max_gp_benchmarks: int = 3,
    max_gp_steps: int = 20,
    save_svg: bool = False,
) -> Path:
    if phase not in {"1", "2", "all"}:
        raise ValueError("phase debe ser uno de {'1','2','all'}")
    _ = topx  # reservado para extensiones de ranking TopX

    apply_publication_style(dpi=dpi)
    bundle = load_input_bundle(session=session, json_path=json_path, strict=strict)

    master_df = build_master_active_table(bundle.trajectory_df)
    if solo_active and "cv_mode" in master_df.columns:
        master_df = master_df[master_df["cv_mode"].astype(str).str.lower() == "active"].copy()

    if samplers:
        samplers_norm = {str(s).lower() for s in samplers}
        master_df = master_df[master_df["sampler"].astype(str).str.lower().isin(samplers_norm)].copy()

    final_df = build_final_table(master_df=master_df, summary_df=bundle.summary_df)

    out = Path(output_dir).expanduser().resolve() if output_dir else (bundle.session_dir / "report_v2")
    dirs = _ensure_dirs(out)

    notes: List[str] = list(bundle.warnings)
    coverage = summarize_active_coverage(master_df)
    if coverage.get("single_model_mode", True):
        notes.append(
            "Run active en modo monomodelo: secciones comparativas multi-modelo se muestran como limitadas."
        )

    tables = generate_phase1_tables(master_df=master_df, final_df=final_df, out_dir=dirs["tables"])

    if phase in {"1", "all"}:
        generate_by_benchmark_outputs(
            final_df=final_df,
            audit_df=bundle.audit_df,
            out_dir=dirs["by_benchmark"],
            dpi=dpi,
            save_svg=save_svg,
        )
        generate_global_plots(
            final_df=final_df,
            tables=tables,
            out_dir=dirs["global"],
            overview_dir=dirs["overview_final"],
            dpi=dpi,
            save_svg=save_svg,
        )
        generate_evolution_plots(
            master_df=master_df,
            out_dir=dirs["evolution"],
            dpi=dpi,
            save_svg=save_svg,
        )
        generate_sampling_effects_plots(
            master_df=master_df,
            final_df=final_df,
            out_dir=dirs["sampling_effects"],
            dpi=dpi,
            save_svg=save_svg,
        )

    if phase in {"2", "all"}:
        generate_gp_predictions(
            master_df=master_df,
            final_df=final_df,
            metadata=bundle.metadata,
            out_dir=dirs["gp_predictions"],
            max_gp_benchmarks=max_gp_benchmarks,
            max_gp_steps=max_gp_steps,
            dpi=dpi,
            save_svg=save_svg,
        )

    _write_index(out_dir=out, coverage=coverage, warnings=notes, phase=phase, tables=tables)
    (out / "report_meta.json").write_text(
        json.dumps(
            {
                "session_dir": str(bundle.session_dir),
                "results_json_path": str(bundle.results_json_path),
                "active_trajectory_path": str(bundle.active_trajectory_path),
                "summary_csv_path": str(bundle.summary_csv_path) if bundle.summary_csv_path else None,
                "audit_jsonl_path": str(bundle.audit_jsonl_path) if bundle.audit_jsonl_path else None,
                "coverage": coverage,
                "warnings": notes,
                "phase": phase,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _validate_min_outputs(out)
    return out
