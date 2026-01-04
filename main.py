from src.utils.paths import ENTOMOTIVE_DATA_DIR, LOGS_DIR
from src.models.eval import evaluate_model
from src.models.dummy import DummySurrogateRegressor
from src.models.ridge import RidgeSurrogateRegressor
from src.models.pls import PLSSurrogateRegressor
from src.models.gp import GPSurrogateRegressor
import pandas as pd

ENTOMOTIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    data = pd.read_csv(ENTOMOTIVE_DATA_DIR / "sample_data.csv")