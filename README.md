# Surrogate Lab: Gaussian Process Surrogates for Experimental Optimization

**A reproducible demo for learning from few evaluations, modelling uncertainty and prioritizing future experiments.**

This repository contains the experimental code and outputs for the TFG:

> Modelos sustitutos para la busqueda de entradas optimas

The project studies probabilistic surrogate models, especially Gaussian Processes, for approximating expensive or unknown objective functions when the number of available evaluations is limited.

## Demo

The Streamlit app in `app/streamlit_app.py` turns the work into a tangible portfolio demo:

- Home: short conceptual presentation of surrogate modelling.
- GP + EI 1D: live Forrester demo with GP mean, 95% interval, Expected Improvement and infill evolution.
- Synthetic benchmarks: reads generated benchmark tables and figures from `outputs/`.
- Entomotive real case: shows Hermetia diet data, Leave-One-Diet-Out validation, GP vs Dummy, uncertainty plots and EI candidates.
- Limitations: explains what can and cannot be concluded responsibly.

Screenshot placeholder:

```text
Run the app and capture the GP + EI 1D page or the Entomotive page for portfolio use.
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Repository Map

```text
app/
  streamlit_app.py                  # Interactive Surrogate Lab demo
src/
  benchmarks/                       # Synthetic benchmark functions
  models/                           # Surrogate model wrappers, including GP
  analysis/                         # Metrics, active learning and tuning logic
  evaluation/                       # Report and visualization utilities
data/
  entomotive_datasets/              # Real Entomotive-derived datasets
outputs/
  logs/benchmarks/                  # Synthetic benchmark logs and reports
  plots/TFG_MAIN_real_case...       # Final real-case figures and CSVs
  reports/TFG_MAIN_real_case...     # Real-case audit tables
notebooks/
  *.ipynb                           # Exploration and tutorial notebooks
```

## Reused Components

The app intentionally reuses the existing project instead of duplicating the methodology:

- `src.benchmarks.functions.get_benchmark("forrester")`
- `src.models.gp.GPSurrogateRegressor`
- `src.analysis.active_learning.ei_values_from_model`
- Generated CSV/PNG results in `outputs/`

## Real Case Reproducibility

The main real case uses the Hermetia dataset, excludes TPC as a target, and compares Dummy and GP models.

To regenerate the audit, tables, final figures and copied TFG assets:

```powershell
python reproduce_tfg_outputs.py
```

This command reuses the LODO tuning logs in:

```text
outputs/logs/tuning/TFG_MAIN_real_case_hermetia_no_tpc_tuning
```

To rerun the complete tuning before regenerating figures:

```powershell
python reproduce_tfg_outputs.py --rerun-tuning
```

The reproducibility manifest is written to:

```text
outputs/reproducibility_real_case_manifest.json
```

## Methodological Note

The Entomotive section is prospective. Expected Improvement candidates are not confirmed optimal diets. They are candidate formulations that the model would prioritize for future experimental evaluation under uncertainty.

The model does not replace the real experiment. Its role is to decide what to evaluate next.
