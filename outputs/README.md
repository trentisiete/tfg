# Outputs usados en el TFG

Carpetas principales del caso real:

- `logs/tuning/TFG_MAIN_real_case_hermetia_no_tpc_tuning`: resultados base del ajuste LODO para Hermetia, sin TPC, con Dummy y GP.
- `plots/TFG_MAIN_real_case_gp_report_no_tpc`: informe grafico completo de kernels GP y Dummy.
- `reports/TFG_MAIN_real_case_audit_no_tpc`: tablas y checks del caso real.
- `plots/TFG_MAIN_real_case_audit_no_tpc`: graficas de auditoria del caso real.
- `plots/TFG_MAIN_real_case_results_pov_ei_no_tpc`: figuras finales de resultados, incertidumbre y pre-infill con EI.

Carpetas legacy conservadas por referencia:

- `plots/comprehensive_report_gp_ard_kernels_v1`
- `plots/comprehensive_report_gp_kernels_v1`
- `plots/comprehensive_report_v1`
- `logs/benchmarks/forrester_active_sweep_20260219_004822`

Reproduccion del caso real:

1. Para regenerar auditoria, tablas, figuras finales y sincronizar las imagenes usadas por LaTeX:

   `python reproduce_tfg_outputs.py`

2. Para recalcular tambien el tuning LODO desde cero antes de generar figuras:

   `python reproduce_tfg_outputs.py --rerun-tuning`

El primer comando reutiliza `logs/tuning/TFG_MAIN_real_case_hermetia_no_tpc_tuning`; el segundo puede tardar bastante mas.
