from src.analysis.metrics import evaluate_model
from src.plotting.viz import build_X_y_groups, generate_all_figures_for_target
from src.utils.paths import ENTOMOTIVE_DATA_DIR, LOGS_DIR, PLOTS_DIR
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv(ENTOMOTIVE_DATA_DIR / "productivity_hermetia_lote.csv")

    targets = [
        "Ganancia de peso (mg) media diaria por larva",
        "Ganancia de peso (mg) por larva",
        "Peso total de las larvas (g) al finalizar el periodo de alimentación",
        "FCR",
        "Reducción (%) del sustrato (reducción de la masa de dieta suministrada)",
    ]

    from src.models.dummy import DummySurrogateRegressor
    from src.models.ridge import RidgeSurrogateRegressor
    from src.models.pls import PLSSurrogateRegressor
    from src.models.gp import GPSurrogateRegressor

    models = {
        "Dummy": DummySurrogateRegressor(strategy="mean"),
        "Ridge": RidgeSurrogateRegressor(alpha=1.0, fit_intercept=True),
        "PLS": PLSSurrogateRegressor(n_components=2, scale=True),
        "GP": GPSurrogateRegressor(),
    }

    fig_root = PLOTS_DIR / "figures_hermetia"
    save_root = LOGS_DIR / "metrics_hermetia"

    for target in targets:
        task = build_X_y_groups(df, target, use_tratamiento_numeric=True)
        out = evaluate_model(models, task.X, task.y, task.groups, save_path=None)

        # generate output directory safe name
        import re
        safe_target = re.sub(r'[^a-zA-Z0-9]', '_', target)
        safe_target = re.sub(r'_+', '_', safe_target).strip('_').lower()[:50]
        outdir = fig_root / safe_target

        # Pass models explicitly to generate_all_figures_for_target
        generate_all_figures_for_target(df, target, out, task, outdir, models=models)