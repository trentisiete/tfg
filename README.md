# Surrogate Models: Gaussian Process Surrogates for Experimental Optimization

Código experimental y resultados del TFG:

> Modelos sustitutos para la búsqueda de entradas óptimas

El proyecto estudia modelos sustitutos probabilísticos, en especial Procesos
Gaussianos (GP), para aproximar funciones objetivo caras o desconocidas cuando
el número de evaluaciones disponibles es limitado, y prioriza nuevos
experimentos mediante Expected Improvement (EI).

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Mapa del repositorio

```text
src/
  benchmarks/                  # Funciones benchmark sintéticas, sampling y ruido
  models/                      # Wrappers de modelos sustitutos (GP, Ridge, PLS, Dummy)
  analysis/                    # Tuning LODO anidado, métricas y aprendizaje activo con EI
  configs/                     # Especificaciones de tuning (caso real) y de benchmarks
  evaluation/
    benchmark_report_active/   # Pipeline de informes de benchmarks (report_v2)
  utils/                       # Rutas del proyecto y utilidades
data/
  entomotive_datasets/         # Datasets reales derivados de Entomotive (no en git)
outputs/
  logs/tuning/                 # Logs del tuning LODO del caso real
  logs/benchmarks/             # Logs y reports de benchmarks (no en git, pesados)
  plots/TFG_MAIN_real_case...  # Figuras finales del caso real
  reports/TFG_MAIN_real_case...# Tablas de auditoría del caso real
notebooks/                     # Exploración de datos y tutorial de benchmarks
tests/                         # Test de infill + generadores de figuras teóricas del TFG
TFG_José/                      # Memoria del TFG en LaTeX
presentacion/                  # Defensa: animaciones Manim y PowerPoint
```

## Puntos de entrada

Caso real (Hermetia, sin TPC, Dummy vs familias de GP):

```powershell
python reproduce_tfg_outputs.py                 # auditoría + figuras + assets LaTeX
python reproduce_tfg_outputs.py --rerun-tuning  # recalcula además el tuning LODO
python run_exhaustive_tuning.py                 # solo el tuning LODO anidado
python audit_entomotive_real_case.py            # solo auditoría y tablas
python generate_entomotive_pov_figures.py       # solo figuras finales (POV EI)
python tuning_visual_report.py                  # informe gráfico completo de kernels GP
```

Benchmarks sintéticos (diseño inicial + infill con EI):

```powershell
python run_benchmark_evaluation.py --help       # evaluación configurable de benchmarks
python run_forrester_active_sweep.py            # barrido activo sobre Forrester + report_v2
python -m src.evaluation.benchmark_report_active --help  # regenerar informes report_v2
```

Figuras teóricas del TFG (capítulo de marco teórico):

```powershell
python tests/fig_gp_prior_posterior.py
python tests/fig_ei_demo.py
python tests/fig_kernel_comparison.py
```

## Reproducibilidad del caso real

`reproduce_tfg_outputs.py` reutiliza los logs LODO en
`outputs/logs/tuning/TFG_MAIN_real_case_hermetia_no_tpc_tuning`, regenera la
auditoría y las figuras finales, y copia los assets a `TFG_José/assets`.
El manifiesto queda en `outputs/reproducibility_real_case_manifest.json`.

## Nota metodológica

La sección de Entomotive es prospectiva. Los candidatos de Expected Improvement
no son dietas óptimas confirmadas: son formulaciones candidatas que el modelo
prioriza para futura evaluación experimental bajo incertidumbre. El modelo no
sustituye al experimento real; su papel es decidir qué evaluar a continuación.
