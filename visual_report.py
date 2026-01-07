import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.analysis.tuning_reporter import SurrogatePlotter, ResultsLoader, ModelReconstructor
from src.utils.paths import ENTOMOTIVE_DATA_DIR, LOGS_DIR, TESTS_DIR, PLOTS_DIR
from src.configs.tuning_specs import FEATURE_COLS_FULL, FEATURE_COLS_REDUCED

# ================= Configuration =================
# Targets to analyze
TARGETS = {
    "FCR": "FCR",
    "TPC": "TPC_larva_media",
    "Quitina": "QUITINA (%)",
    "Proteina": "PROTEINA (%)",
}

# Paths
BASE_LOG_DIR = LOGS_DIR / "tuning" / "productivity_hermetia_v2_comprehensive"
DATA_FILE = ENTOMOTIVE_DATA_DIR / "productivity_hermetia_lote.csv"
GLOBAL_OUTPUT_DIR = PLOTS_DIR / "comprehensive_report_v2"

# ================= Helper Functions =================

def build_X_y_groups(df: pd.DataFrame, target_col: str, feature_cols: list):
    """Dynamically builds X matrix based on the requested feature list."""
    data = df.copy()
    data = data.loc[~data[target_col].isna()].reset_index(drop=True)

    groups = data["diet_name"].astype(str).to_numpy()
    y = data[target_col].astype(float).to_numpy()

    # Always include dummy variables for byproduct type as they are structural
    if "byproduct_type" in data.columns:
        byp = pd.get_dummies(data["byproduct_type"], prefix="byproduct", drop_first=False)
    else:
        byp = pd.DataFrame()

    # Filter numeric features requested
    # Note: FEATURE_COLS might contain 'Tratamiento', check if present
    valid_cols = [c for c in feature_cols if c in data.columns]
    Xdf = data[valid_cols].copy()

    if not byp.empty:
        Xdf = pd.concat([Xdf, byp], axis=1)

    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    return Xdf.to_numpy(dtype=float), y, groups, Xdf.columns.tolist()

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.base import clone
def collect_cv_predictions(model, X, y, groups):
    """
    Realiza una validación cruzada LODO para obtener predicciones 'honestas'
    Y TAMBIÉN la incertidumbre 'honesta' (std) para modelos probabilísticos.
    """
    logo = LeaveOneGroupOut()
    y_pred_cv = np.zeros_like(y)
    y_std_cv = np.zeros_like(y) # Nuevo array para guardar la desviación estándar

    for train_idx, test_idx in logo.split(X, y, groups):
        # 1. Clonar
        m_fold = clone(model)
        
        # 2. Entrenar (N-1)
        m_fold.fit(X[train_idx], y[train_idx])
        
        # 3. Predecir (1)
        if hasattr(m_fold, "predict_dist"):
            # GP: Guardamos media y desviación estándar
            mean, std = m_fold.predict_dist(X[test_idx])
            y_pred_cv[test_idx] = mean
            y_std_cv[test_idx] = std 
        else:
            # Ridge/PLS: Solo media, std se queda en 0
            y_pred_cv[test_idx] = m_fold.predict(X[test_idx])
            
    return y_pred_cv, y_std_cv

def run_single_experiment_report(
    target_key: str, 
    target_col: str, 
    experiment_name: str, 
    feature_cols: list
):
    print(f"\n--- Processing {target_key} | {experiment_name} ---")
    
    # 1. Setup Directories
    log_dir = BASE_LOG_DIR / experiment_name
    output_dir = GLOBAL_OUTPUT_DIR / target_key.lower() / experiment_name
    target_slug = target_key.lower()
    
    # 2. Init Tools
    try:
        loader = ResultsLoader(log_dir)
        summary = loader.load_target_summary(target_slug)
    except FileNotFoundError:
        print(f"Skipping {experiment_name} for {target_key} (Log not found)")
        return None

    plotter = SurrogatePlotter(output_dir)
    
    # 3. Load Data & Reconstructor
    df_raw = pd.read_csv(DATA_FILE)
    X, y, groups, feature_names = build_X_y_groups(df_raw, target_col, feature_cols)
    reconstructor = ModelReconstructor(X, y, feature_names)

    # 4. Load Folds & Best Params
    model_folds = {}
    best_params = {}
    
    for m_name, info in summary["models"].items():
        if info["files"]["folds_csv"]:
            model_folds[m_name] = loader.load_fold_results(info["files"]["folds_csv"])
            best_params[m_name] = model_folds[m_name].iloc[0]["params"]

    # 5. Stability Plots
    print(f"[{experiment_name}] Plotting Stability...")
    for m in ["mae", "rmse", "coverage95"]:
        plotter.plot_comparative_metrics_box(model_folds, metric=m)

    # 6. Deep Dive (Reconstruction)
    cv_preds = {} # Solo guardamos medias para el parity global
    
    for m_name, params in best_params.items():
        print(f"[{experiment_name}] Inspecting {m_name}...")
        try:
            # A. Reconstruir modelo base
            model_base = reconstructor.retrain_model(m_name, params)
            
            # B. Feature Relevance (Modelo Full para ver estructura final)
            plotter.plot_feature_relevance(model_base, feature_names, m_name)
            
            # C. PREDICCIONES HONESTAS (CV) con STD
            # Recuperamos media y std calculadas fold a fold
            y_cv, std_cv = collect_cv_predictions(model_base, X, y, groups)
            cv_preds[m_name] = y_cv
            
            # D. Diagnósticos Lineales (Ridge/PLS)
            if m_name in ["Ridge", "PLS"]:
                plotter.plot_residuals_vs_predicted(y, y_cv, groups, m_name)

            # E. GP Specifics
            if m_name == "GP":
                # AQUI ESTABA EL ERROR: Faltaba llamar a la función.
                # Usamos y_cv y std_cv para que el gráfico de incertidumbre sea realista
                plotter.plot_gp_uncertainty_analysis(y, y_cv, std_cv, groups)
                
                # Response Slices (Usamos el modelo Full Fit para ver la curva aprendida final)
                for feat in feature_names[:3]:
                    plotter.plot_1d_response_slice(model_base, X, y, feature_names, feat, groups)

        except Exception as e:
            print(f"Error inspecting {m_name}: {e}")
            import traceback
            traceback.print_exc() # Para ver detalles si falla

    # 7. Global Parity
    plotter.plot_actual_vs_predicted_cv(y, cv_preds, groups)
    
    return summary


def main():
    for target_key, target_col in TARGETS.items():
        print(f"\n{'='*60}")
        print(f"GENERATING REPORT FOR TARGET: {target_key}")
        print(f"{'='*60}")

        # 1. Run REDUCED Report
        summary_reduced = run_single_experiment_report(
            target_key, target_col, "REDUCED_FEATURES", FEATURE_COLS_REDUCED
        )

        # 2. Run FULL Report
        summary_full = run_single_experiment_report(
            target_key, target_col, "FULL_FEATURES", FEATURE_COLS_FULL
        )

        # 3. Compare Both
        if summary_reduced and summary_full:
            print(f"\n>>> Generating Comparison (Full vs Reduced) for {target_key}...")
            comp_output_dir = GLOBAL_OUTPUT_DIR / target_key.lower() / "comparison"
            plotter = SurrogatePlotter(comp_output_dir)
            plotter.plot_experiment_comparison(summary_full, summary_reduced, target_key)

    print(f"\nAll reports generated in: {GLOBAL_OUTPUT_DIR}")

if __name__ == "__main__":
    main()