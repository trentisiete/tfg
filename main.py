import pandas as pd
import numpy as np

from src.analysis.metrics import evaluate_model
from src.models.dummy import DummySurrogateRegressor
from src.models.ridge import RidgeSurrogateRegressor
from src.models.pls import PLSSurrogateRegressor
from src.models.gp import GPSurrogateRegressor
from src.utils.paths import ENTOMOTIVE_DATA_DIR, OUTPUTS_DIR


CSV_PATH = ENTOMOTIVE_DATA_DIR / "productivity_hermetia_lote.csv"
OUTPUT_PATH = OUTPUTS_DIR / "hermetia_lote_productivity_modeling_results_main_1.json"


def build_X_y_groups(df: pd.DataFrame, target: str):
    df = df.copy()

    # 1) filtra y válido
    df = df.loc[~df[target].isna()].reset_index(drop=True)

    # 2) define groups (LODO)
    groups = df["diet_name"].astype(str).to_numpy()

    # 3) define y
    y = df[target].astype(float).to_numpy()

    # 4) define X (solo info de dieta / pre-experimento)
    feature_cols = [
        "inclusion_pct",
        "Tratamiento",
        "Proteína (%)_media",
        "Grasa (%)_media",
        "Fibra (%)_media",
        "Cenizas (%)_media",
        "Carbohidratos (%)_media",
        "ratio_P_C",
        "ratio_P_F",
        "ratio_Fibra_Grasa",
        "TPC_dieta_media",
    ]

    # One-hot de byproduct_type (categórica real)
    byp = pd.get_dummies(df["byproduct_type"], prefix="byproduct", drop_first=False)
    # new columns: control, hoja, orujo, quinoa

    Xdf = pd.concat([df[feature_cols], byp], axis=1)

    # Apply numeric conversion and handle missing values
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True)) # In principle, there should be no missing values

    X = Xdf.to_numpy(dtype=float)

    return X, y, groups, Xdf.columns.tolist()


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    # targets típicos de productividad
    targets = [
        "Ganancia de peso (mg) media diaria por larva",
    #     "Ganancia de peso (mg) por larva",
    #     "Peso total de las larvas (g) al finalizar el periodo de alimentación",
    #     "FCR",
    #     "Reducción (%) del sustrato (reducción de la masa de dieta suministrada)",
    ]

    models = {
        "Dummy": DummySurrogateRegressor(strategy="mean"),
        "Ridge": RidgeSurrogateRegressor(alpha=1.0, fit_intercept=True),
        "PLS": PLSSurrogateRegressor(n_components=2, scale=True),
        "GP": GPSurrogateRegressor(),  # Matern + WhiteKernel
    }

    for target in targets:
        X, y, groups, feat_names = build_X_y_groups(df, target)

        print(f"\n=== Target: {target} | n={len(y)} | p={X.shape[1]} | diets={len(np.unique(groups))} ===")

        out = evaluate_model(models, X, y, groups, save_path=None)

        # only prints out micro + macro
        for m in ["Dummy", "Ridge", "PLS", "GP"]:
            summ = out[m]["results"]["summary"]
            micro = summ["micro"]
            macro = summ["macro"]
            print(
                f"{m:>5} | micro MAE={micro['mae']:.4f} RMSE={micro['rmse']:.4f} cov95={micro.get('coverage95', None)} "
                f"| macro MAE(mean)={macro['mae']['mean']:.4f}"
            )
