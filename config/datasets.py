"""Centralized dataset configuration - single source of truth."""

import os
from pathlib import Path

# Project root — works both for PYTHONPATH=. usage and pip-installed packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Root directory for trained model artifacts and Optuna studies.
# Override via environment variable BA_ARTIFACTS_ROOT if needed.
ARTIFACTS_ROOT = os.environ.get(
    "BA_ARTIFACTS_ROOT", str(_PROJECT_ROOT / "artifacts")
)

# Root directory for data files.
_DATA_ROOT = str(_PROJECT_ROOT / "data")

# Dataset configurations: features, labels, CSV paths, and Optuna journal info
DATASET_CONFIGS = {
    "isotherm": {
        "csv_file": os.path.join(_DATA_ROOT, "Clean_Results_Isotherm.csv"),
        "features": [
            "Flow_well", "Temp_diff", "kW_well", "Hydr_gradient",
            "Hydr_conductivity", "Aqu_thickness", "Long_dispersivity",
            "Trans_dispersivity", "Isotherm"
        ],
        "labels": ["Area", "Iso_distance", "Iso_width"],
        "journal_path": os.path.join(ARTIFACTS_ROOT, "optuna_studies", "isotherm", "journal.log"),
        "study_name": "nn_study_isotherm_journal",
        "best_params_file": str(_PROJECT_ROOT / "config" / "best_params_isotherm.json"),
    },
    "cone": {
        "csv_file": os.path.join(_DATA_ROOT, "Depression_cones.csv"),
        "features": ["Flow_well", "Hydr_gradient", "Hydr_conductivity", "Aqu_thickness"],
        "labels": ["Cone"],
        "journal_path": os.path.join(ARTIFACTS_ROOT, "optuna_studies", "cone", "journal.log"),
        "study_name": "depression_cones_mlp_journal_study",
        "best_params_file": str(_PROJECT_ROOT / "config" / "best_params_cone.json"),
    },
}

# All known datasets for validation
KNOWN_DATASETS = set(DATASET_CONFIGS.keys())

# All valid features and labels (for CSV detection)
KNOWN_FEATURES = set()
KNOWN_LABELS = set()
for _cfg in DATASET_CONFIGS.values():
    KNOWN_FEATURES.update(_cfg["features"])
    KNOWN_LABELS.update(_cfg["labels"])

KNOWN_FEATURES = sorted(list(KNOWN_FEATURES))
KNOWN_LABELS = sorted(list(KNOWN_LABELS))

# Default model directories for prediction
DEFAULT_MODEL_DIRS = {
    "isotherm": {
        "mlp": os.path.join(ARTIFACTS_ROOT, "models", "mlp", "isotherm"),
        "random": os.path.join(ARTIFACTS_ROOT, "models", "random", "isotherm"),
    },
    "cone": {
        "mlp": os.path.join(ARTIFACTS_ROOT, "models", "mlp", "cone"),
        "random": os.path.join(ARTIFACTS_ROOT, "models", "random", "cone"),
    },
}


def get_dataset_config(dataset: str) -> dict:
    """Get configuration for a dataset.

    Args:
        dataset: Dataset name ('isotherm' or 'cone')

    Returns:
        Configuration dictionary with features, labels, paths

    Raises:
        ValueError: If dataset not found
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'. Valid options: {sorted(KNOWN_DATASETS)}")
    return DATASET_CONFIGS[dataset].copy()


def detect_features_and_labels(csv_file: str) -> tuple:
    """Detect which features and labels are present in a CSV file.

    Args:
        csv_file: Path to CSV file

    Returns:
        Tuple of (feature_list, label_list) in order they appear in CSV
    """
    import pandas as pd

    df = pd.read_csv(csv_file, nrows=0)
    columns = list(df.columns)

    features = [col for col in columns if col in KNOWN_FEATURES]
    labels = [col for col in columns if col in KNOWN_LABELS]

    return features, labels
