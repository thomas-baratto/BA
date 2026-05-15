"""Centralized dataset configuration - single source of truth."""

import os
from pathlib import Path

# Project root — works both for PYTHONPATH=. usage and pip-installed packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Root directory for trained model artifacts and Optuna studies.
# Override via environment variable BA_ARTIFACTS_ROOT if needed.
_artifacts_env = os.environ.get("BA_ARTIFACTS_ROOT")
ARTIFACTS_ROOT = Path(_artifacts_env) if _artifacts_env else _PROJECT_ROOT / "artifacts"

# Root directory for data files.
_DATA_ROOT = _PROJECT_ROOT / "data"

# Dataset configurations: features, labels, and default model paths
DATASET_CONFIGS = {
    "isotherm": {
        "features": [
            "Flow_well", "Temp_diff", "kW_well", "Hydr_gradient",
            "Hydr_conductivity", "Aqu_thickness", "Long_dispersivity",
            "Trans_dispersivity", "Isotherm"
        ],
        "labels": ["Area", "Iso_distance", "Iso_width"],
        "models": {
            "mlp": ".",
            "randomized": ".",
            "randomized:nRMSE": ".",
            "randomized:KGE": ".",
        }
    },
    "cone": {
        "features": ["Flow_well", "Hydr_gradient", "Hydr_conductivity", "Aqu_thickness"],
        "labels": ["Cone"],
        "models": {
            "mlp": ".",
            "randomized": ".",
        }
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

# All default model directories for prediction
DEFAULT_MODEL_DIRS = {dataset: cfg["models"] for dataset, cfg in DATASET_CONFIGS.items()}


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


def detect_dataset_type(csv_file: str) -> str:
    """Automatically detect dataset type from CSV headers.

    Args:
        csv_file: Path to CSV file

    Returns:
        Dataset name ('isotherm' or 'cone') or None if ambiguous
    """
    import pandas as pd
    try:
        df = pd.read_csv(csv_file, nrows=0)
        cols = set(df.columns)
        
        # Check for isotherm-specific markers
        if any(c in cols for c in ["kW_well", "Temp_diff", "Isotherm", "Long_dispersivity"]):
            return "isotherm"
        
        # Check for cone-specific markers (subset of isotherm, so check absence of isotherm ones)
        cone_features = set(DATASET_CONFIGS["cone"]["features"])
        if all(c in cols for c in cone_features):
            return "cone"
            
        return None
    except Exception:
        return None
