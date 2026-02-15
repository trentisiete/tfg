from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.benchmarks import get_benchmark

from .styling import build_model_style_map, place_legend, save_figure


METRICS = ["mae", "rmse", "r2", "nlpd", "coverage_95"]


def _winner_model(df: pd.DataFrame, metric: str) -> Optional[str]:
    if df.empty or metric not in df.columns:
        return None
    block = df.dropna(subset=[metric]).copy()
    if block.empty:
        return None
    if metric == "r2":
        idx = block[metric].idxmax()
    elif metric == "coverage_95":
        idx = (block[metric] - 0.95).abs().idxmin()
    else:
        idx = block[metric].idxmin()
    return str(block.loc[idx, "model"])


def _plot_metric_bar(df: pd.DataFrame, metric: str, style_map: Dict[str, Dict], out_path: Path, dpi: int, save_svg: bool):
    agg = (
        df.groupby("model", as_index=False)
        .agg(value=(metric, "mean"), std=(metric, "std"), n=(metric, "count"))
        .sort_values("value", ascending=(metric != "r2"))
    )
    if agg.empty:
        return

    winner = _winner_model(df, metric)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    xs = np.arange(len(agg))
    bars = []
    for i, row in agg.reset_index(drop=True).iterrows():
        model = row["model"]
        st = style_map.get(model, {})
        b = ax.bar(
            xs[i],
            row["value"],
            yerr=0.0 if pd.isna(row["std"]) else float(row["std"]),
            color=st.get("color"),
            hatch=st.get("hatch", ""),
            edgecolor="black",
            linewidth=2.0 if model == winner else 0.8,
            alpha=0.95,
            label=model,
        )
        bars.extend(b)

    ax.set_xticks(xs)
    ax.set_xticklabels(agg["model"], rotation=30, ha="right")
    ax.set_title(f"{metric.upper()} por modelo")
    ax.set_xlabel("Modelo")
    ax.set_ylabel(metric.upper())
    place_legend(ax, outside=True)
    save_figure(fig, out_path, dpi=dpi, save_svg=save_svg)


def _extract_benchmark_noise_summary(
    df_bn: pd.DataFrame,
    benchmark: str,
    noise: str,
    audit_df: pd.DataFrame,
) -> str:
    lines = [f"# Resumen: {benchmark} | {noise}", ""]
    lines.append(f"- Configuraciones evaluadas: {len(df_bn)}")
    lines.append(f"- Modelos presentes: {sorted(df_bn['model'].unique().tolist())}")
    lines.append("")

    winner = _winner_model(df_bn, "mae")
    if winner is not None:
        winner_mae = float(df_bn[df_bn["model"] == winner]["mae"].mean())
        lines.append(f"- Mejor modelo (MAE): **{winner}** ({winner_mae:.6f})")
    else:
        lines.append("- Mejor modelo (MAE): no disponible")

    if "Dummy" in set(df_bn["model"]):
        dummy_mae = float(df_bn[df_bn["model"] == "Dummy"]["mae"].mean())
        if winner is not None and winner != "Dummy":
            gap = dummy_mae - winner_mae
            gap_pct = (gap / abs(dummy_mae) * 100.0) if dummy_mae else np.nan
            lines.append(f"- Gap frente a Dummy: {gap:.6f} ({gap_pct:.2f}%)")
        else:
            lines.append("- Gap frente a Dummy: Dummy es el mejor en este escenario.")
    else:
        lines.append("- Dummy no disponible en este run active; comparativa con baseline no aplicable.")

    lines.append("")
    lines.append("## Hiperparametros (audit active)")
    if audit_df.empty:
        lines.append("- Sin auditoria de hiperparametros en esta sesion.")
    else:
        subset = audit_df[(audit_df.get("benchmark") == benchmark) & (audit_df.get("noise") == noise)]
        if subset.empty:
            lines.append("- No hay registros de auditoria para este benchmark/ruido.")
        else:
            latest = subset.sort_values("step").groupby("model", as_index=False).tail(1)
            for _, row in latest.iterrows():
                best_cv = row.get("best_cv_params")
                current = row.get("current_selected_params")
                switch_reason = row.get("switch_reason")
                lines.append(f"- {row.get('model')}: best_cv={best_cv} | current={current} | switch={switch_reason}")
    return "\n".join(lines) + "\n"


def generate_by_benchmark_outputs(
    final_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    out_dir: Path,
    dpi: int = 300,
    save_svg: bool = False,
) -> None:
    if final_df.empty:
        return

    style_map = build_model_style_map(final_df["model"].unique())
    out_dir = Path(out_dir)

    for (benchmark, noise), block in final_df.groupby(["benchmark", "noise"], dropna=False):
        bdir = out_dir / str(benchmark) / str(noise)
        bdir.mkdir(parents=True, exist_ok=True)

        # Save numeric summary used by the local markdown.
        metrics_summary = (
            block.groupby("model", as_index=False)
            .agg(
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                r2=("r2", "mean"),
                nlpd=("nlpd", "mean"),
                coverage_95=("coverage_95", "mean"),
                fit_time=("fit_time", "mean"),
                predict_time=("predict_time", "mean"),
            )
            .sort_values("mae")
        )
        metrics_summary.to_csv(bdir / "metrics_summary.csv", index=False)

        for metric in METRICS:
            if metric in block.columns:
                _plot_metric_bar(
                    df=block,
                    metric=metric,
                    style_map=style_map,
                    out_path=bdir / f"{metric}_models",
                    dpi=dpi,
                    save_svg=save_svg,
                )

        summary_text = _extract_benchmark_noise_summary(
            df_bn=block,
            benchmark=str(benchmark),
            noise=str(noise),
            audit_df=audit_df,
        )
        (bdir / "summary.md").write_text(summary_text, encoding="utf-8")

    # Benchmark-level robustness plots across all noises
    for benchmark, block in final_df.groupby("benchmark", dropna=False):
        bdir = out_dir / str(benchmark)
        agg = (
            block.groupby("model", as_index=False)
            .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"))
            .sort_values("mae_mean")
        )
        if agg.empty:
            continue
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        xs = np.arange(len(agg))
        for i, row in agg.reset_index(drop=True).iterrows():
            model = row["model"]
            st = style_map.get(model, {})
            ax.bar(
                xs[i],
                row["mae_mean"],
                yerr=0.0 if pd.isna(row["mae_std"]) else float(row["mae_std"]),
                color=st.get("color"),
                hatch=st.get("hatch", ""),
                edgecolor="black",
                linewidth=0.8,
                label=model,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(agg["model"], rotation=30, ha="right")
        ax.set_title(f"Robustez por ruido (MAE medio ± variacion) - {benchmark}")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("MAE medio")
        place_legend(ax, outside=True)
        save_figure(fig, bdir / "robustez_mae_por_ruido", dpi=dpi, save_svg=save_svg)

        # Violin by benchmark for stability view
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        data = [block.loc[block["model"] == m, "mae"].dropna().values for m in agg["model"]]
        vp = ax.violinplot(data, showmeans=True, showextrema=True)
        for i, body in enumerate(vp["bodies"]):
            model = agg.iloc[i]["model"]
            body.set_facecolor(style_map[model]["color"])
            body.set_alpha(0.7)
            body.set_edgecolor("black")
            body.set_hatch(style_map[model]["hatch"])
        ax.set_xticks(np.arange(1, len(agg) + 1))
        ax.set_xticklabels(agg["model"], rotation=30, ha="right")
        ax.set_title(f"Estabilidad MAE por modelo - {benchmark}")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Distribucion MAE")
        save_figure(fig, bdir / "violin_estabilidad_mae", dpi=dpi, save_svg=save_svg)
