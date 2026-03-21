"""Centralized dataset configuration - single source of truth."""

# Dataset configurations: features, labels, CSV paths, and Optuna journal info
DATASET_CONFIGS = {
    "isotherm": {
        "csv_file": "data/Clean_Results_Isotherm.csv",
        "features": [
            "Flow_well", "Temp_diff", "kW_well", "Hydr_gradient",
            "Hydr_conductivity", "Aqu_thickness", "Long_dispersivity",
            "Trans_dispersivity", "Isotherm"
        ],
        "labels": ["Area", "Iso_distance", "Iso_width"],
        "journal_path": "data/good_runs/global_run_830/optuna_journal_storage/journal.log",
        "study_name": "nn_study_isotherm_journal",
    },
    "cone": {
        "csv_file": "data/Depression_cones.csv",
        "features": ["Flow_well", "Hydr_gradient", "Hydr_conductivity", "Aqu_thickness"],
        "labels": ["Cone"],
        "journal_path": "data/good_runs/global_run_832/optuna_journal_storage/journal.log",
        "study_name": "depression_cones_mlp_journal_study",
    },
}

# All known datasets for validation
KNOWN_DATASETS = set(DATASET_CONFIGS.keys())

# All valid features and labels (for CSV detection)
KNOWN_FEATURES = set()
KNOWN_LABELS = set()
for cfg in DATASET_CONFIGS.values():
    KNOWN_FEATURES.update(cfg["features"])
    KNOWN_LABELS.update(cfg["labels"])

KNOWN_FEATURES = sorted(list(KNOWN_FEATURES))
KNOWN_LABELS = sorted(list(KNOWN_LABELS))


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
