"""Preprocessing utilities - centralized data transformation logic."""

import numpy as np
import pandas as pd

from core.data_loader import load_data
from core.training_utils import to_physical_units

# Registry of engineered features: name → (compute_fn, required_raw_columns)
ENGINEERED_FEATURES: dict[str, tuple] = {
    "Transmissivity": (
        lambda df: df["Hydr_conductivity"] * df["Aqu_thickness"],
        ["Hydr_conductivity", "Aqu_thickness"],
    ),
    "Darcy_velocity": (
        lambda df: df["Hydr_conductivity"] * df["Hydr_gradient"],
        ["Hydr_conductivity", "Hydr_gradient"],
    ),
    "Q_over_T": (
        lambda df: df["Flow_well"]
        / (df["Hydr_conductivity"] * df["Aqu_thickness"]),
        ["Flow_well", "Hydr_conductivity", "Aqu_thickness"],
    ),
}


def compute_engineered_features(
    df: pd.DataFrame, required_features: list[str]
) -> pd.DataFrame:
    """Add any missing engineered columns to *df* (in-place) if they can be
    derived from columns already present.

    Only columns listed in *required_features* that are (a) missing from *df*
    and (b) registered in ENGINEERED_FEATURES will be computed.

    Returns the (possibly augmented) DataFrame.
    """
    for feat in required_features:
        if feat in df.columns:
            continue
        if feat not in ENGINEERED_FEATURES:
            continue
        compute_fn, raw_cols = ENGINEERED_FEATURES[feat]
        missing_raw = [c for c in raw_cols if c not in df.columns]
        if missing_raw:
            raise ValueError(
                f"Cannot compute '{feat}': missing raw columns {missing_raw}"
            )
        df[feat] = compute_fn(df)
    return df


def apply_inverse_transform(scaled_predictions: np.ndarray, scaler, apply_log: bool = True) -> np.ndarray:
    """Apply inverse scaling and log transform to restore predictions to original units.

    Thin wrapper around training_utils.to_physical_units for backwards compatibility.
    """
    return to_physical_units(scaled_predictions, scaler, use_log=apply_log)


def load_scalers_from_dataset(csv_file: str, features: list, labels: list,
                              feature_scaler_type: str = "robust",
                              label_scaler_type: str = "robust",
                              apply_log: bool = True,
                              apply_area_root: bool = False) -> tuple:
    """Load and fit scalers on full dataset.
    
    Args:
        csv_file: Path to CSV file
        features: List of feature column names
        labels: List of label column names
        feature_scaler_type: Scaler type for features ('minmax', 'standard', 'robust', 'quantile')
        label_scaler_type: Scaler type for labels
        apply_log: Whether to apply log1p to labels
        apply_area_root: Whether to apply sqrt to Area label (isotherm only)
        
    Returns:
        Tuple of (X_scaler, y_scaler) fitted on full dataset
    """
    _, _, X_scaler, _, _, y_scaler = load_data(
        csv_file=csv_file,
        feature_cols=features,
        label_cols=labels,
        feature_scaler_type=feature_scaler_type,
        label_scaler_type=label_scaler_type,
        use_log=apply_log,
        use_area_root=apply_area_root,
        test_size=None,  # Load all data for fitting
        random_state=42,
        plots=False
    )
    return X_scaler, y_scaler


def create_train_test_split(csv_file: str, features: list, labels: list,
                            test_size: float = 0.2, random_state: int = 42,
                            feature_scaler_type: str = "robust",
                            label_scaler_type: str = "robust",
                            apply_log: bool = True,
                            apply_area_root: bool = False) -> tuple:
    """Create train/test split with preprocessing.
    
    Args:
        csv_file: Path to CSV file
        features: List of feature column names
        labels: List of label column names
        test_size: Fraction of data for test (0-1)
        random_state: Random seed for reproducibility
        feature_scaler_type: Scaler type for features
        label_scaler_type: Scaler type for labels
        apply_log: Whether to apply log1p to labels
        apply_area_root: Whether to apply sqrt to Area label
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, X_scaler, y_scaler)
    """
    X_train, X_test, X_scaler, y_train, y_test, y_scaler = load_data(
        csv_file=csv_file,
        feature_cols=features,
        label_cols=labels,
        feature_scaler_type=feature_scaler_type,
        label_scaler_type=label_scaler_type,
        use_log=apply_log,
        use_area_root=apply_area_root,
        test_size=test_size,
        random_state=random_state,
        plots=False
    )
    return X_train, X_test, y_train, y_test, X_scaler, y_scaler


def load_raw_data(csv_file: str, features: list, labels: list) -> tuple:
    """Load raw data without scaling or transformation.
    
    Args:
        csv_file: Path to CSV file
        features: List of feature column names
        labels: List of label column names
        
    Returns:
        Tuple of (X, y) as numpy arrays
    """
    df = pd.read_csv(csv_file)
    X = df[features].values
    y = df[labels].values
    return X, y
