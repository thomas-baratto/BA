"""Preprocessing utilities - centralized data transformation logic."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from core.data_loader import load_data


def apply_inverse_transform(scaled_predictions: np.ndarray, scaler, apply_log: bool = True) -> np.ndarray:
    """Apply inverse scaling and log transform to restore predictions to original units.
    
    Args:
        scaled_predictions: Scaled predictions from model (shape: N x D)
        scaler: Fitted scaler object with inverse_transform method
        apply_log: Whether to apply inverse log (expm1)
        
    Returns:
        Predictions in original physical units
    """
    # Inverse scale
    unscaled = scaler.inverse_transform(scaled_predictions)
    
    # Inverse log transform if enabled
    if apply_log:
        # expm1(x) = exp(x) - 1; inverse of log1p
        # Clip to prevent overflow when computing exp()
        unscaled = np.clip(unscaled, -600, 600)  # exp(-600) ≈ 0; exp(600) ≈ inf
        unscaled = np.expm1(unscaled)
    
    return unscaled


def load_scalers_from_dataset(csv_file: str, features: list, labels: list,
                              feature_scaler_type: str = "robust",
                              label_scaler_type: str = "minmax",
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
                            label_scaler_type: str = "minmax",
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
