from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import generate_active_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Visual Report v2 (active-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python -m src.evaluation.benchmark_report_active --session comprehensive
  python -m src.evaluation.benchmark_report_active --session comprehensive --fase 1
  python -m src.evaluation.benchmark_report_active --json outputs/logs/benchmarks/comprehensive/comprehensive_results.json
  python -m src.evaluation.benchmark_report_active --session comprehensive --samplers sobol random --max-gp-benchmarks 3 --max-gp-steps 20
        """,
    )
    parser.add_argument("--session", type=str, default=None, help="Nombre de sesion o ruta a carpeta de sesion")
    parser.add_argument("--json", type=str, default=None, help="Ruta explicita a comprehensive_results.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Directorio de salida (default: <session>/report_v2)")
    parser.add_argument("--fase", type=str, default="all", choices=["1", "2", "all"], help="Fase de reporte a ejecutar")
    parser.add_argument("--topx", type=int, default=10, help="Top-X para tablas comparativas (reservado)")
    parser.add_argument("--strict", action="store_true", help="Abortar si faltan columnas/artefactos opcionales")
    parser.add_argument("--dpi", type=int, default=300, help="Resolucion PNG por defecto")
    parser.add_argument("--solo-active", action=argparse.BooleanOptionalAction, default=True, help="Filtrar solo cv_mode=active")
    parser.add_argument("--samplers", nargs="+", default=["sobol", "random"], help="Samplers a incluir en el reporte")
    parser.add_argument("--max-gp-benchmarks", type=int, default=3, help="Max benchmarks para GP atlas (fase 2)")
    parser.add_argument("--max-gp-steps", type=int, default=20, help="Max pasos de evolucion para GP atlas")
    parser.add_argument("--save-svg", action="store_true", help="Guardar tambien SVG ademas de PNG")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output = generate_active_report(
        session=args.session,
        json_path=args.json,
        output_dir=args.output_dir,
        phase=args.fase,
        topx=args.topx,
        strict=args.strict,
        dpi=args.dpi,
        solo_active=bool(args.solo_active),
        samplers=args.samplers,
        max_gp_benchmarks=args.max_gp_benchmarks,
        max_gp_steps=args.max_gp_steps,
        save_svg=args.save_svg,
    )
    print(f"Reporte generado: {Path(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
