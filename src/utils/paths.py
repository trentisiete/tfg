from pathlib import Path

# Obtains the route to the current directory (src/utils)
_CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = _CURRENT_DIR.parent.parent

# Defining relatives paths to the project root

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ENTOMOTIVE_DATA_DIR = DATA_DIR / "entomotive_datasets"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
CONFIGS_DIR = PROJECT_ROOT / "src" / "configs"

def get_config_path(filename: str) -> Path:
    """Returns the full path to a configuration file given its filename."""
    return CONFIGS_DIR / filename