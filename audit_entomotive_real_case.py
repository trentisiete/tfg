from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.configs.tuning_specs import (
    FEATURE_COLS_FULL,
    FEATURE_COLS_REDUCED,
    TARGET_MAP,
)
from src.utils.paths import ENTOMOTIVE_DATA_DIR, LOGS_DIR, OUTPUTS_DIR, PROJECT_ROOT


DATA_FILE = ENTOMOTIVE_DATA_DIR / "productivity_hermetia_lote.csv"
LOG_ROOT = LOGS_DIR / "tuning" / "productivity_hermetia_gp_ard_kernels_no_tpc_v1"
REPORT_DIR = OUTPUTS_DIR / "reports" / "entomotive_real_case_no_tpc_v1"
PLOT_DIR = OUTPUTS_DIR / "plots" / "entomotive_real_case_no_tpc_v1"
TFG_REPORT = PROJECT_ROOT / "TFG_José" / "caso_real_entomotive_validacion.md"

ENTOMOTIVE_DATASETS = [
    "productivity_hermetia_lote.csv",
    "productivity_tenebrio_lote.csv",
    "productivity_all_lote.csv",
    "quality_hermetia_dieta.csv",
    "quality_tenebrio_dieta.csv",
    "quality_all_dieta.csv",
]

TARGET_SLUG = {
    "FCR": "fcr",
    "Quitina": "quitina",
    "Proteina": "proteina",
}

FEATURE_MODES = {
    "REDUCED_FEATURES": FEATURE_COLS_REDUCED,
    "FULL_FEATURES": FEATURE_COLS_FULL,
}

METRIC_COLUMNS = ["mae", "rmse", "r2", "nlpd", "coverage_95"]


def _read_metrics_cell(value: str) -> dict:
    if pd.isna(value):
        return {}
    parsed = ast.literal_eval(str(value))
    return parsed if isinstance(parsed, dict) else {}


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def build_entomotive_inventory() -> pd.DataFrame:
    rows = []
    for filename in ENTOMOTIVE_DATASETS:
        path = ENTOMOTIVE_DATA_DIR / filename
        df = pd.read_csv(path)
        rows.append(
            {
                "dataset": filename,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "species": ", ".join(sorted(df["species"].dropna().astype(str).unique()))
                if "species" in df.columns
                else "",
                "n_diets": int(df["diet_name"].nunique()) if "diet_name" in df.columns else None,
                "role": "dataset modelado en el pipeline real"
                if filename == "productivity_hermetia_lote.csv"
                else "dataset generado disponible",
            }
        )
    return pd.DataFrame(rows)


def validate_dataset(df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checks: list[dict] = []

    def add(name: str, status: str, detail: str):
        checks.append({"check": name, "status": status, "detail": detail})

    add("dataset_exists", "PASS", str(DATA_FILE))
    add("dataset_shape", "PASS" if df.shape == (33, 43) else "WARN", f"{df.shape[0]} rows, {df.shape[1]} columns")

    species = sorted(df["species"].dropna().astype(str).unique().tolist())
    add("species", "PASS" if species == ["Hermetia"] else "WARN", ", ".join(species))

    diet_sizes = df.groupby("diet_name").size()
    add(
        "diet_replicates",
        "PASS" if len(diet_sizes) == 11 and diet_sizes.min() == 3 and diet_sizes.max() == 3 else "WARN",
        f"{len(diet_sizes)} diets; replicate range {int(diet_sizes.min())}-{int(diet_sizes.max())}",
    )

    for label, col in TARGET_MAP.items():
        n_valid = int(df[col].notna().sum())
        expected = 31 if label == "FCR" else 33
        add(f"target_{label}", "PASS" if n_valid == expected else "WARN", f"{col}: {n_valid} valid values")

    for mode, cols in FEATURE_MODES.items():
        missing = [c for c in cols if c not in df.columns]
        null_counts = {c: int(df[c].isna().sum()) for c in cols if c in df.columns}
        status = "PASS" if not missing and all(v == 0 for v in null_counts.values()) else "WARN"
        add(f"features_{mode}", status, f"missing={missing}; nulls={null_counts}")

    add(
        "byproduct_type",
        "PASS" if int(df["byproduct_type"].isna().sum()) == 0 else "WARN",
        str(df["byproduct_type"].value_counts(dropna=False).to_dict()),
    )

    by_diet = (
        df.groupby(["diet_name", "byproduct_type", "inclusion_pct"], dropna=False)
        .agg(
            n=("diet_name", "size"),
            fcr_mean=("FCR", "mean"),
            fcr_std=("FCR", "std"),
            quitina_mean=("QUITINA (%)", "mean"),
            quitina_std=("QUITINA (%)", "std"),
            proteina_mean=("PROTEINA (%)", "mean"),
            proteina_std=("PROTEINA (%)", "std"),
        )
        .reset_index()
    )

    objective_rows = [
        ("FCR", "min", by_diet.loc[by_diet["fcr_mean"].idxmin()]),
        ("QUITINA (%)", "max", by_diet.loc[by_diet["quitina_mean"].idxmax()]),
        ("PROTEINA (%)", "max", by_diet.loc[by_diet["proteina_mean"].idxmax()]),
    ]
    candidates = []
    value_col = {
        "FCR": "fcr_mean",
        "QUITINA (%)": "quitina_mean",
        "PROTEINA (%)": "proteina_mean",
    }
    for target, criterion, row in objective_rows:
        candidates.append(
            {
                "target": target,
                "criterion": criterion,
                "diet_name": row["diet_name"],
                "byproduct_type": row["byproduct_type"],
                "inclusion_pct": row["inclusion_pct"],
                "mean_value": row[value_col[target]],
            }
        )
    candidates_df = pd.DataFrame(candidates)

    ranking = by_diet.copy()
    ranking["score_fcr"] = (ranking["fcr_mean"].max() - ranking["fcr_mean"]) / (
        ranking["fcr_mean"].max() - ranking["fcr_mean"].min()
    )
    for col in ["quitina_mean", "proteina_mean"]:
        score_col = "score_" + col.replace("_mean", "")
        ranking[score_col] = (ranking[col] - ranking[col].min()) / (ranking[col].max() - ranking[col].min())
    score_cols = ["score_fcr", "score_quitina", "score_proteina"]
    ranking["exploratory_equal_weight_score"] = ranking[score_cols].mean(axis=1)
    ranking = ranking.sort_values("exploratory_equal_weight_score", ascending=False).reset_index(drop=True)

    return checks, by_diet, candidates_df, ranking


def load_active_model_metrics(checks: list[dict]) -> pd.DataFrame:
    rows = []

    for feature_mode in FEATURE_MODES:
        for target_label, target_col in TARGET_MAP.items():
            slug = TARGET_SLUG[target_label]
            summary_path = LOG_ROOT / feature_mode / slug / f"{slug}_summary.json"
            if not summary_path.exists():
                checks.append(
                    {
                        "check": f"summary_{feature_mode}_{target_label}",
                        "status": "WARN",
                        "detail": f"Missing {summary_path}",
                    }
                )
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for model_name in summary.get("models", {}):
                folds_path = LOG_ROOT / feature_mode / slug / f"{slug}_{model_name.lower()}_folds.csv"
                if not folds_path.exists():
                    checks.append(
                        {
                            "check": f"folds_{feature_mode}_{target_label}_{model_name}",
                            "status": "WARN",
                            "detail": f"Missing {folds_path}",
                        }
                    )
                    continue

                folds = pd.read_csv(folds_path)
                metric_dicts = folds["metrics"].map(_read_metrics_cell)
                record = {
                    "feature_mode": feature_mode,
                    "target": target_label,
                    "target_col": target_col,
                    "model": model_name,
                    "n_folds": int(len(folds)),
                    "n_samples": int(sum(m.get("n_samples", 0) for m in metric_dicts)),
                }
                for metric in METRIC_COLUMNS:
                    values = [_safe_float(m.get(metric)) for m in metric_dicts]
                    values = [v for v in values if v is not None]
                    record[f"{metric}_macro_mean"] = float(np.mean(values)) if values else None
                    record[f"{metric}_macro_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                rows.append(record)

                checks.append(
                    {
                        "check": f"folds_{feature_mode}_{target_label}_{model_name}",
                        "status": "PASS" if len(folds) == 11 else "WARN",
                        "detail": f"{len(folds)} folds, {record['n_samples']} samples",
                    }
                )

    return pd.DataFrame(rows)


def plot_dataset_targets(by_diet: pd.DataFrame):
    plot_data = [
        ("FCR (menor es mejor)", "fcr_mean", True),
        ("Quitina (mayor es mejor)", "quitina_mean", False),
        ("Proteina (mayor es mejor)", "proteina_mean", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (title, col, ascending) in zip(axes.ravel(), plot_data):
        block = by_diet.sort_values(col, ascending=ascending)
        colors = ["#2a9d8f" if i == 0 else "#7a8a99" for i in range(len(block))]
        ax.bar(block["diet_name"], block[col], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "dataset_targets_by_diet.png", dpi=180)
    plt.close(fig)


def plot_equal_weight_ranking(ranking: pd.DataFrame):
    block = ranking.sort_values("exploratory_equal_weight_score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(block["diet_name"], block["exploratory_equal_weight_score"], color="#4c78a8")
    ax.set_xlabel("Exploratory equal-weight score")
    ax.set_title("Ranking exploratorio de dietas observadas")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "observed_equal_weight_ranking.png", dpi=180)
    plt.close(fig)


def plot_active_metrics(metrics: pd.DataFrame):
    for feature_mode in FEATURE_MODES:
        block = metrics[metrics["feature_mode"] == feature_mode].copy()
        if block.empty:
            continue
        targets = list(TARGET_MAP.keys())
        preferred_order = [
            "Dummy",
            "GP_Linear",
            "GP_RBF_NoARD",
            "GP_RBF_ARD",
            "GP_Matern32_NoARD",
            "GP_Matern32_ARD",
            "GP_Matern52_NoARD",
            "GP_Matern52_ARD",
            "GP_Compuesto_NoARD",
        ]
        present = set(block["model"].astype(str))
        models = [m for m in preferred_order if m in present]
        x = np.arange(len(targets))
        width = 0.8 / max(len(models), 1)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        for i, model in enumerate(models):
            vals = []
            for target in targets:
                row = block[(block["target"] == target) & (block["model"] == model)]
                vals.append(float(row["mae_macro_mean"].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width, label=model)

        ax.set_xticks(x)
        ax.set_xticklabels(targets)
        ax.set_ylabel("MAE macro")
        ax.set_title(f"MAE por target - {feature_mode}")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"active_mae_{feature_mode.lower()}.png", dpi=180)
        plt.close(fig)

    gp = metrics[metrics["model"].astype(str).str.startswith("GP")].copy()
    if not gp.empty:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        labels = gp["feature_mode"].str.replace("_FEATURES", "", regex=False) + " / " + gp["target"]
        labels = labels + " / " + gp["model"]
        ax.bar(labels, gp["coverage_95_macro_mean"], color="#f58518")
        ax.axhline(0.95, color="#333333", linestyle="--", linewidth=1.2, label="coverage 0.95")
        ax.set_ylabel("Coverage 95 macro")
        ax.set_title("Cobertura 95 de los GP")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "gp_coverage95_active.png", dpi=180)
        plt.close(fig)


def write_markdown_report(
    checks: list[dict],
    inventory: pd.DataFrame,
    by_diet: pd.DataFrame,
    candidates: pd.DataFrame,
    ranking: pd.DataFrame,
    metrics: pd.DataFrame,
):
    pass_count = sum(1 for c in checks if c["status"] == "PASS")
    warn_count = sum(1 for c in checks if c["status"] != "PASS")
    top_equal = ranking.iloc[0]
    active_models = sorted(metrics["model"].unique().tolist()) if not metrics.empty else []

    lines = [
        "# Validacion del caso real Entomotive",
        "",
        "Este archivo es una auditoria tecnica reproducible del caso real. No redacta el TFG.",
        "",
        "## Estado",
        "",
        f"- Dataset validado: `{DATA_FILE.relative_to(PROJECT_ROOT)}`.",
        f"- Checks PASS: {pass_count}.",
        f"- Checks WARN: {warn_count}.",
        f"- Modelos activos considerados: {', '.join(active_models)}.",
        "- Benchmarks y pruebas LODO de benchmark quedan fuera de alcance.",
        "",
        "## Dataset",
        "",
        "- 33 filas, 43 columnas.",
        "- 11 dietas, 3 replicas por dieta.",
        "- Especie: Hermetia.",
        "- Targets activos: FCR, QUITINA (%) y PROTEINA (%).",
        "- TPC_larva_media queda excluido como target en esta corrida por no ser concluyente.",
        "- FCR tiene 31 valores validos; quitina y proteina tienen 33.",
        "",
        "## Inventario Entomotive completo",
        "",
        "El Excel de Entomotive genera datasets para dos especies: Hermetia y Tenebrio. El pipeline de resultados auditado aqui usa Hermetia, pero Tenebrio tambien queda generado como dataset disponible.",
        "",
        inventory.to_markdown(index=False),
        "",
        "## Criterio de seleccion de especie modelada",
        "",
        "La prediccion del caso real se hace sobre Hermetia. El criterio efectivo no es que Tenebrio sea invalido, sino que Hermetia constituye el bloque mas homogeneo usado por el pipeline principal: 11 dietas, 3 replicas por dieta, todas las filas en `study_block = main` y covariables completas para los dos conjuntos de features.",
        "",
        "Tenebrio tambien esta generado, pero mezcla `main`, `coffee_orujillo` y `water_control`, incorpora tipos adicionales de dieta y presenta distinta disponibilidad de targets. Por eso debe tratarse como segundo caso potencial o subanalisis separado, no mezclarse directamente con Hermetia en los resultados actuales.",
        "",
        "## Como se llega al dataset",
        "",
        "1. `notebooks/02_datasets_creator.ipynb` limpia hojas del Excel Entomotive.",
        "2. Enumera replicas con `add_replica`.",
        "3. Une productividad, composicion larvaria, composicion de dieta y TPC de dieta usado como covariable.",
        "4. Extrae metadatos con `parse_diet_name`.",
        "5. Calcula ratios nutricionales con `add_ratios`.",
        "6. Exporta `data/entomotive_datasets/productivity_hermetia_lote.csv`.",
        "",
        "## Dietas que destacan por objetivo observado",
        "",
        candidates.to_markdown(index=False),
        "",
        "## Ranking exploratorio observado",
        "",
        "Este ranking usa pesos iguales sobre FCR invertido, quitina y proteina. Es util como cribado tecnico, no como verdad biologica cerrada.",
        "",
        ranking[
            [
                "diet_name",
                "byproduct_type",
                "inclusion_pct",
                "fcr_mean",
                "quitina_mean",
                "proteina_mean",
                "exploratory_equal_weight_score",
            ]
        ]
        .head(6)
        .to_markdown(index=False),
        "",
        f"Con ese criterio explicito, la dieta observada mejor posicionada es `{top_equal['diet_name']}`.",
        "",
        "## Evaluacion del surrogate",
        "",
        "El caso real se evalua con LODO por dieta: cada fold deja fuera una dieta completa. Esto evita fuga entre replicas de la misma dieta.",
        "",
        metrics[
            [
                "feature_mode",
                "target",
                "model",
                "n_folds",
                "n_samples",
                "mae_macro_mean",
                "rmse_macro_mean",
                "coverage_95_macro_mean",
            ]
        ]
        .to_markdown(index=False),
        "",
        "## Figuras limpias generadas",
        "",
        f"- `{(PLOT_DIR / 'dataset_targets_by_diet.png').relative_to(PROJECT_ROOT)}`",
        f"- `{(PLOT_DIR / 'observed_equal_weight_ranking.png').relative_to(PROJECT_ROOT)}`",
        f"- `{(PLOT_DIR / 'active_mae_reduced_features.png').relative_to(PROJECT_ROOT)}`",
        f"- `{(PLOT_DIR / 'active_mae_full_features.png').relative_to(PROJECT_ROOT)}`",
        f"- `{(PLOT_DIR / 'gp_coverage95_active.png').relative_to(PROJECT_ROOT)}`",
        "",
        "## Conclusion tecnica",
        "",
        "El caso real esta listo para trazar el dataset Entomotive, justificar como se construye y evaluar el surrogate sobre dietas no vistas.",
        "",
        "Tambien permite proponer dietas observadas candidatas para exploracion posterior, siempre que se declare el objetivo. Los objetivos individuales quedan recogidos en la tabla de candidatos y el cribado multiobjetivo usa solo FCR, quitina y proteina.",
        "",
        "Lo que no queda cerrado por el codigo actual es la generacion automatica de una formulacion nueva no observada. Para eso haria falta definir un espacio de dietas validas y una regla final de adquisicion/ranking.",
        "",
        "## Checks",
        "",
        pd.DataFrame(checks).to_markdown(index=False),
        "",
    ]
    TFG_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    inventory = build_entomotive_inventory()
    checks, by_diet, candidates, ranking = validate_dataset(df)
    metrics = load_active_model_metrics(checks)

    inventory.to_csv(REPORT_DIR / "entomotive_dataset_inventory.csv", index=False)
    by_diet.to_csv(REPORT_DIR / "dataset_by_diet.csv", index=False)
    candidates.to_csv(REPORT_DIR / "candidate_diets_by_objective.csv", index=False)
    ranking.to_csv(REPORT_DIR / "observed_equal_weight_ranking.csv", index=False)
    metrics.to_csv(REPORT_DIR / "active_model_metrics.csv", index=False)
    pd.DataFrame(checks).to_csv(REPORT_DIR / "readiness_checks.csv", index=False)

    plot_dataset_targets(by_diet)
    plot_equal_weight_ranking(ranking)
    plot_active_metrics(metrics)

    summary = {
        "dataset": str(DATA_FILE.relative_to(PROJECT_ROOT)),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "n_diets": int(df["diet_name"].nunique()),
        "active_models": sorted(metrics["model"].unique().tolist()) if not metrics.empty else [],
        "report": str(TFG_REPORT.relative_to(PROJECT_ROOT)),
        "report_dir": str(REPORT_DIR.relative_to(PROJECT_ROOT)),
        "plot_dir": str(PLOT_DIR.relative_to(PROJECT_ROOT)),
        "checks": {
            "pass": int(sum(1 for c in checks if c["status"] == "PASS")),
            "warn": int(sum(1 for c in checks if c["status"] != "PASS")),
        },
    }
    (REPORT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(checks, inventory, by_diet, candidates, ranking, metrics)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
