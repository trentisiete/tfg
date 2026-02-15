# @author: José Arbelaez
"""
Evaluation module for surrogate models.

Provides comprehensive visualization and analysis tools for:
1. Tuning results from run_exhaustive_tuning.py (visual_reporter)
2. Benchmark evaluation results from run_benchmark_evaluation.py (benchmark_visual_reporter)

Main components:

TUNING EVALUATION:
    - TuningResultsDiscovery: Auto-discovers and inventories all tuning runs
    - UnifiedResultsLoader: Loads and normalizes results to tidy format
    - ComprehensiveReporter: Generates full visual reports
    - GPDiagnostics: Specialized GP uncertainty analysis
    - CoverageVerifier: Verify experimental completeness
    - generate_full_report: One-call complete report generation

BENCHMARK EVALUATION:
    - BenchmarkResultsLoader: Loads and normalizes benchmark results
    - BenchmarkTableGenerator: Generates all summary tables
    - BenchmarkGlobalPlotter: Global cross-benchmark visualizations
    - BenchmarkDashboardGenerator: Per-benchmark dashboards
    - GPDeepDiveAnalyzer: Specialized GP kernel analysis
    - generate_benchmark_report: One-call benchmark report generation
    
Usage:
    # Tuning results
    from src.evaluation import generate_full_report
    generate_full_report("productivity_hermetia_v2_comprehensive")
    
    # Benchmark results
    from src.evaluation import generate_benchmark_report
    generate_benchmark_report("benchmark_20260120_131837")
    
    Or from CLI:
        python -m src.evaluation.visual_reporter --session productivity_hermetia_v2_comprehensive
        python -m src.evaluation.benchmark_visual_reporter --session benchmark_20260120
"""

from .visual_reporter import (
    TuningResultsDiscovery,
    UnifiedResultsLoader,
    ComprehensiveReporter,
    GPDiagnostics,
    CoverageVerifier,
    generate_full_report,
)

from .benchmark_visual_reporter import (
    BenchmarkResultsLoader,
    BenchmarkTableGenerator,
    BenchmarkGlobalPlotter,
    BenchmarkDashboardGenerator,
    GPDeepDiveAnalyzer,
    generate_benchmark_report,
)

from .benchmark_visual_reporter_multimode import (
    MultiModeReportVerifier,
    MultiModeResultsLoader,
    MultiModeTableGenerator,
    SamplingEffectsPlotter,
    CVDiagnosticsPlotter,
    TopXComparator,
    SuperAggregatedPlotter,
    GPPredictionsGenerator,
    generate_benchmark_report_multimode,
)

from .gp_visualization import (
    GPVisualizationGenerator,
    GPModel,
    train_gp,
    train_all_kernels,
    plot_gp_1d,
    plot_gp_1d_comparison,
    plot_gp_2d_contour,
    plot_gp_2d_slices,
    plot_gp_nd_slices,
    generate_kernel_comparison_dashboard,
    add_gp_visualizations_to_report,
)

from .benchmark_report_active import (
    generate_active_report,
)

__all__ = [
    # Tuning evaluation
    "TuningResultsDiscovery",
    "UnifiedResultsLoader", 
    "ComprehensiveReporter",
    "GPDiagnostics",
    "CoverageVerifier",
    "generate_full_report",
    # Benchmark evaluation
    "BenchmarkResultsLoader",
    "BenchmarkTableGenerator",
    "BenchmarkGlobalPlotter",
    "BenchmarkDashboardGenerator",
    "GPDeepDiveAnalyzer",
    "generate_benchmark_report",
    # Benchmark evaluation MULTIMODE
    "MultiModeReportVerifier",
    "MultiModeResultsLoader",
    "MultiModeTableGenerator",
    "SamplingEffectsPlotter",
    "CVDiagnosticsPlotter",
    "TopXComparator",
    "GPHyperparamAtlasGenerator",
    "generate_benchmark_report_multimode",
    # GP Visualization
    "GPVisualizationGenerator",
    "GPModel",
    "train_gp",
    "train_all_kernels",
    "plot_gp_1d",
    "plot_gp_1d_comparison",
    "plot_gp_2d_contour",
    "plot_gp_2d_slices",
    "plot_gp_nd_slices",
    "generate_kernel_comparison_dashboard",
    "add_gp_visualizations_to_report",
    # New active-only benchmark report v2
    "generate_active_report",
]
