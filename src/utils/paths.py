from pathlib import Path

# Obtains the route to the current directory (src/utils)
_CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = _CURRENT_DIR.parent.parent

# Defining relatives paths to the project root

DATA_DIR = PROJECT_ROOT / "data"
ENTOMOTIVE_DATA_DIR = DATA_DIR / "entomotive_datasets"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
