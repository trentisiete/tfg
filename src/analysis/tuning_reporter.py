import ast
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

# Import model classes
from src.models.gp import GPSurrogateRegressor
from src.models.pls import PLSSurrogateRegressor
from src.models.ridge import RidgeSurrogateRegressor
from src.models.dummy import DummySurrogateRegressor
from src.models.bagging import RandomForestSurrogateRegressor
from src.models.boosting import GradientBoostingSurrogateRegressor

# Publication-ready style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 200,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5
})

class ResultsLoader:
    """Handles loading/parsing of tuning result files."""
    def __init__(self, base_log_dir: Union[str, Path]):
        self.base_dir = Path(base_log_dir)

    def load_target_summary(self, target_slug: str) -> Dict:
        path = self.base_dir / target_slug / f"{target_slug}_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Summary not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_fold_tuning_results(self, target_slug: str, model: str) -> Dict:
        path = self.base_dir / target_slug / f"{target_slug}_{model}_tuning.json"
        if not path.exists():
            raise FileNotFoundError(f"Summary not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_fold_results(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        def _safe_parse(value):
            if not isinstance(value, str): return value
            try: return ast.literal_eval(value)
            except: return value

        if "params" in df.columns: df["params"] = df["params"].apply(_safe_parse)
        if "metrics" in df.columns:
            df["metrics"] = df["metrics"].apply(_safe_parse)
            metrics_df = pd.json_normalize(df["metrics"])
            df = pd.concat([df.drop("metrics", axis=1), metrics_df], axis=1)
        return df

class ModelReconstructor:
    """Reconstructs models from saved parameters for introspection."""
    MODEL_MAP = {
        "GP": GPSurrogateRegressor,
        "PLS": PLSSurrogateRegressor,
        "Ridge": RidgeSurrogateRegressor,
        "Dummy": DummySurrogateRegressor,
        "RandomForest": RandomForestSurrogateRegressor,
        "GradientBoosting": GradientBoostingSurrogateRegressor
    }

    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        self.X = X
        self.y = y
        self.feature_names = feature_names

    def retrain_model(self, model_name: str, params: Union[Dict, str, None]) -> object:
        if model_name not in self.MODEL_MAP:
            raise ValueError(f"Model {model_name} not supported.")

        # Robust param parsing
        clean_params = {}
        if isinstance(params, dict): clean_params = params.copy()
        elif isinstance(params, str):
            try: clean_params = ast.literal_eval(params)
            except: pass

        # Handle GP Kernel string issue
        if model_name == "GP" and isinstance(clean_params.get("kernel"), str):
            warnings.warn(f"Kernel is string in params. Using default kernel for {model_name}.")
            del clean_params['kernel']

        model = self.MODEL_MAP[model_name](**clean_params)
        model.fit(self.X, self.y)
        return model

class SurrogatePlotter:
    """
    Comprehensive visualization module for Surrogate Models.
    """
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Stability & Metrics ---
    def plot_comparative_metrics_box(self, model_folds_map: Dict[str, pd.DataFrame], metric: str = "mae"):
        """
        Plots model stability using boxplots.
        Auto-cleans NaNs to avoid 'ghost' plots for deterministic models (Ridge/PLS).
        """
        data_list = []
        for m_name, df in model_folds_map.items():
            # Verificar si la métrica existe en el DF
            if metric not in df.columns:
                continue

            temp = df.copy()
            temp["Model"] = m_name

            # FILTRO DE SEGURIDAD:
            # Si el modelo devolvió 'None' para esta métrica (caso Ridge/Coverage),
            # nos aseguramos de que sea NaN y lo filtramos.
            temp[metric] = pd.to_numeric(temp[metric], errors='coerce')
            temp = temp.dropna(subset=[metric])

            if not temp.empty:
                data_list.append(temp)

        if not data_list:
            logging.warning(f"No data found for metric {metric}. Skipping plot.")
            return

        # Filter out empty DataFrames before concat to avoid FutureWarning
        data_list = [df for df in data_list if not df.empty and not df[metric].isna().all()]
        if not data_list:
            logging.warning(f"No valid data found for metric {metric} after filtering. Skipping plot.")
            return
            
        combined = pd.concat(data_list, ignore_index=True)

        plt.figure(figsize=(10, 6))
        # Base Boxplot (with hue to avoid seaborn v0.14 deprecation warning)
        sns.boxplot(data=combined, x="Model", y=metric, hue="Model", palette="Pastel1", showfliers=False, legend=False)
        # Stripplot colored by diet
        sns.stripplot(data=combined, x="Model", y=metric, hue="diet",
                      palette="tab20", size=7, alpha=0.9, jitter=True, dodge=False)

        plt.title(f"Model Stability: {metric.upper()} Distribution (Nested LODO)", fontweight="bold")
        plt.ylabel(metric.upper())

        # Mejorar leyenda
        if combined["diet"].nunique() > 0:
            plt.legend(title="Diet (Fold)", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"stability_{metric}.png")
        plt.close()

    # --- 2. Parity & Global Fit ---
    def plot_actual_vs_predicted_cv(self, y_true: np.ndarray, y_pred_dict: Dict[str, np.ndarray], groups: np.ndarray):
        """Multi-panel parity plot with identity line and diet coloring."""
        n = len(y_pred_dict)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows), squeeze=False)
        axes = axes.flatten()

        # Global limits
        all_p = np.concatenate(list(y_pred_dict.values()))
        vmin, vmax = min(y_true.min(), all_p.min()), max(y_true.max(), all_p.max())
        pad = (vmax - vmin) * 0.1
        vmin, vmax = vmin - pad, vmax + pad

        for i, (m_name, preds) in enumerate(y_pred_dict.items()):
            ax = axes[i]
            r2 = r2_score(y_true, preds)
            mae = mean_absolute_error(y_true, preds)

            sns.scatterplot(x=y_true, y=preds, hue=groups, palette="tab20", ax=ax, s=70, alpha=0.8, edgecolor='k')
            ax.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2, label="Perfect Fit")

            # Annotations
            ax.text(vmin + pad, vmax - pad, "Overestimation", color='gray', style='italic')
            ax.text(vmax - pad, vmin + pad, "Underestimation", color='gray', style='italic', ha='right')

            ax.set_title(f"{m_name}\n$R^2$={r2:.2f} | MAE={mae:.2f}")
            ax.set_xlabel("Experimental Value")
            ax.set_ylabel("Predicted Value")
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            if i == 0: ax.legend(loc='upper left', fontsize='small', title="Diet")
            else: ax.get_legend().remove()

        plt.tight_layout()
        plt.savefig(self.output_dir / "parity_plot.png")
        plt.close()

    # --- 3. Linear Model Diagnostics (Ridge/PLS) ---
    def plot_residuals_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, model_name: str):
        """
        Diagnostic plot for linear models. Checks homoscedasticity and linearity.
        Ideal representation for Ridge/PLS fit quality beyond simple lines.
        """
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))

        sns.scatterplot(x=y_pred, y=residuals, hue=groups, palette="tab20", s=70, edgecolor='k', alpha=0.8)
        plt.axhline(0, color='r', linestyle='--', lw=2)

        plt.title(f"{model_name}: Residuals vs Predicted\n(Ideal: Random scatter around 0)")
        plt.xlabel("Predicted Value")
        plt.ylabel("Residual (True - Predicted)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Diet")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"residuals_{model_name}.png")
        plt.close()

    # --- 4. GP Specific Visualizations ---
    def plot_gp_uncertainty_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, groups: np.ndarray):
        """
        Classic GP analysis: Sorted predictions with confidence bands & Error Correlation.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # A. The "Mythical" GP Plot (Sorted by Target)
        sort_idx = np.argsort(y_true)
        x_idx = range(len(y_true))

        # Bandas de confianza (2 sigma)
        ax1.fill_between(x_idx,
                         y_pred[sort_idx] - 1.96 * y_std[sort_idx],
                         y_pred[sort_idx] + 1.96 * y_std[sort_idx],
                         color='cornflowerblue', alpha=0.3, label='95% Confidence (Epistemic)')

        # Predicción media y datos reales
        ax1.plot(x_idx, y_pred[sort_idx], 'b-', lw=1.5, label='GP Mean Prediction')
        sns.scatterplot(x=x_idx, y=y_true[sort_idx], hue=groups[sort_idx],
                        palette="tab20", ax=ax1, s=60, edgecolor='k', zorder=10)

        ax1.set_title("GP Calibration: Confidence Intervals vs Real Data")
        ax1.set_xlabel("Sample Index (Sorted by Value)")
        ax1.set_ylabel("Target Value")
        ax1.legend(loc='upper left', ncol=2, fontsize='small')

        # B. Uncertainty vs Error (Validation of 'Knowing what it doesn't know')
        residuals = np.abs(y_true - y_pred)
        sns.scatterplot(x=y_std, y=residuals, hue=groups, palette="tab20", ax=ax2, s=70, edgecolor='k')
        sns.regplot(x=y_std, y=residuals, scatter=False, ax=ax2, color='red', line_kws={'linestyle':'--'})

        ax2.set_title("Safety Check: Uncertainty vs. Error Correlation")
        ax2.set_xlabel("Predicted Standard Deviation (Uncertainty)")
        ax2.set_ylabel("Absolute Error")
        ax2.get_legend().remove()

        plt.tight_layout()
        plt.savefig(self.output_dir / "gp_uncertainty_deep_dive.png")
        plt.close()

    def plot_1d_response_slice(self, model, X: np.ndarray, y_true: np.ndarray,
                               feature_names: List[str], target_feature: str, groups: np.ndarray):
        """
        Partial Dependence Plot with overlay of real experimental points.
        Shows the 'shape' of the learned function and the uncertainty ballooning in unknown areas.
        """
        if target_feature not in feature_names: return
        idx = feature_names.index(target_feature)

        # Create synthetic grid
        x_min, x_max = X[:, idx].min(), X[:, idx].max()
        padding = (x_max - x_min) * 0.1
        x_grid = np.linspace(x_min - padding, x_max + padding, 200)

        # Baseline: Median of all other features
        baseline = np.median(X, axis=0)
        X_eval = np.tile(baseline, (200, 1))
        X_eval[:, idx] = x_grid

        # Predict
        if hasattr(model, "predict_dist"):
            mean, std = model.predict_dist(X_eval)
        else:
            mean = model.predict(X_eval)
            std = np.zeros_like(mean)

        plt.figure(figsize=(9, 6))

        # 1. Uncertainty Band (The "Balloon")
        if np.any(std > 0):
            plt.fill_between(x_grid, mean - 1.96*std, mean + 1.96*std,
                             color='cornflowerblue', alpha=0.2, label="95% Confidence Region")

        # 2. Mean Prediction
        plt.plot(x_grid, mean, 'b-', lw=2, label="Model Response (Median Baseline)")

        # 3. Real Data Overlay
        # We plot the real points to show if the curve passes near them
        sns.scatterplot(x=X[:, idx], y=y_true, hue=groups, palette="tab20",
                        s=80, edgecolor='k', alpha=0.8, zorder=5)

        # Rug plot on bottom
        sns.rugplot(X[:, idx], color="k", height=0.05, alpha=0.5)

        plt.title(f"Response Profile: {target_feature}\n(Blue band shows GP uncertainty expansion)", fontweight="bold")
        plt.xlabel(target_feature)
        plt.ylabel("Target Response")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"response_profile_{target_feature}.png")
        plt.close()

    def plot_feature_relevance(self, model, feature_names: List[str], model_name: str):
        """Standardized feature importance plot."""
        importance = None
        label = "Importance"

        try:
            if model_name == "GP":
                kernel = model.model_.named_steps["model"].kernel_
                base = kernel.k1 if hasattr(kernel, "k1") else kernel
                # ARD check
                if hasattr(base, "length_scale") and np.ndim(base.length_scale) > 0 and len(base.length_scale) == len(feature_names):
                    importance = 1.0 / (base.length_scale + 1e-9) # Avoid div/0
                    label = "ARD Sensitivity (1/LengthScale)"
            elif model_name in ["Ridge", "PLS"]:
                reg = model.model_.named_steps["model"]
                importance = np.abs(reg.coef_).flatten()
                label = "Absolute Coefficient Magnitude"

            if importance is not None:
                df = pd.DataFrame({"Feature": feature_names, "Value": importance})
                df = df.sort_values("Value", ascending=False)

                plt.figure(figsize=(8, 8))
                sns.barplot(data=df, x="Value", y="Feature", palette="viridis")
                plt.title(f"Global Feature Relevance ({model_name})")
                plt.xlabel(label)
                plt.tight_layout()
                plt.savefig(self.output_dir / f"feature_relevance_{model_name}.png")
                plt.close()
        except Exception as e:
            logging.warning(f"Feature relevance plot skipped for {model_name}: {e}")

    # --- 5. EXPERIMENT COMPARISON (Full vs Reduced) ---
    def plot_experiment_comparison(self, summary_full: Dict, summary_reduced: Dict, target_name: str):
        """
        Side-by-side bar chart comparing Reduced vs Full features for all models.
        
        Compatible with new unified metrics structure:
            macro['mae'] = {'mean': ..., 'std': ..., 'min': ..., 'max': ...}
        """
        # New metrics structure: metric['mean'] instead of metric_mean
        metrics_map = {
            "MAE": "mae",
            "RMSE": "rmse",
            "COV95": "coverage_95"
        }
        data = []

        def _extract_metric_value(macro: dict, metric_key: str):
            """Extract value from new nested structure or fallback to old flat structure."""
            # Try new structure first: macro['mae']['mean']
            if isinstance(macro.get(metric_key), dict):
                return macro[metric_key].get('mean')
            # Fallback to old structure: macro['mae_mean']
            return macro.get(f"{metric_key}_mean")

        # Extract Full
        for model, info in summary_full.get("models", {}).items():
            m = info.get("summary", {}).get("macro", {})
            for label, key in metrics_map.items():
                val = _extract_metric_value(m, key)
                if val is not None:
                    data.append({
                        "Model": model,
                        "Experiment": "FULL FEATURES",
                        "Metric": label,
                        "Value": val
                    })

        # Extract Reduced
        for model, info in summary_reduced.get("models", {}).items():
            m = info.get("summary", {}).get("macro", {})
            for label, key in metrics_map.items():
                val = _extract_metric_value(m, key)
                if val is not None:
                    data.append({
                        "Model": model,
                        "Experiment": "REDUCED FEATURES",
                        "Metric": label,
                        "Value": val
                    })

        if not data:
            logging.warning(f"No data found for experiment comparison on {target_name}")
            return

        df = pd.DataFrame(data)

        # Plot
        g = sns.catplot(
            data=df, kind="bar",
            x="Experiment", y="Value", hue="Model", col="Metric",
            palette="Set2", height=5, aspect=0.8, sharey=False
        )
        g.fig.suptitle(f"Experiment Comparison: Full vs Reduced ({target_name})", y=1.05, fontweight="bold")
        g.set_titles("{col_name}")

        filename = self.output_dir / f"comparison_full_vs_reduced_{target_name}.png"
        g.savefig(filename, bbox_inches="tight")
        plt.close()