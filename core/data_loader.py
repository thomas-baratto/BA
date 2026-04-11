import numpy as np
import pandas as pd
import logging
import os
from typing import List, Tuple
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

class CSVDataset(Dataset):
    """
    Custom PyTorch Dataset for loading tabular data from arrays.
    Data is converted to tensors upon initialization.
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

def load_data(csv_file: str = "data/Clean_Results_Isotherm.csv",
              feature_cols: List[str] = ["Flow_well", "Temp_diff", "kW_well", "Hydr_gradient","Hydr_conductivity", "Aqu_thickness", "Long_dispersivity", "Trans_dispersivity", "Isotherm"],
              label_cols: List[str] = ["Area", "Iso_distance", "Iso_width"],
              test_size: float = 0.3,
              random_state: int = 42,
              plots: bool = False,
              rf: str = '.',
              feature_scaler_type: str = 'minmax',
              label_scaler_type: str = 'minmax',
              use_log: bool = True,
              use_area_root: bool = False,
              return_meta: bool = False) -> Tuple[np.ndarray, np.ndarray, object, np.ndarray, np.ndarray, object]:
    """
    Loads, preprocesses, and splits data from a CSV file using column names.
    
    Args:
        csv_file: Path to the CSV file (default: "data/Clean_Results_Isotherm.csv")
        feature_cols: List of feature column names 
        label_cols: List of label column names
        test_size: Proportion of data to use for testing (default: 0.3)
        random_state: Random seed for reproducibility (default: 42)
        plots: Whether to generate distribution plots (default: False)
        rf: Root folder for saving plots (default: '.')
        feature_scaler_type: Type of scaler for features (default: 'minmax')
        label_scaler_type: Type of scaler for labels (default: 'minmax')
        use_log: Whether to apply log1p transformation before scaling (default: True)
        use_area_root: Whether to apply square root transformation to 'Area' before log1p/scaling (default: False)
        return_meta: Whether to return scaler metadata
        
    Returns:
        Tuple of (X_train, X_test, X_scaler, y_train, y_test, y_scaler)
    """
    # Try to load the file
    df = load_file(csv_file)
    
    # Try to select columns
    X, y = select_columns(df, feature_cols, label_cols)
    
    # Apply square root to "Area" target if requested
    if use_area_root and "Area" in label_cols:
        area_idx = label_cols.index("Area")
        y[:, area_idx] = np.sqrt(y[:, area_idx])
        
    # Ensure y is 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    plot_dir_name = "_".join(label_cols).replace(" ", "_").replace("/", "_")
    plot_dir = os.path.join(rf, "plots", plot_dir_name)
    
    if plots:
        os.makedirs(plot_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(plot_dir, "data_before_transform.npz"),
            y=y, label_cols=np.array(label_cols),
        )

    # Log-transform (always applied prior to scaling; helps compress heavy-tail)
    if use_log:
        X = np.log1p(X)
        y = np.log1p(y)

    if plots:
        np.savez_compressed(
            os.path.join(plot_dir, "data_after_log.npz"),
            X=X, y=y, label_cols=np.array(label_cols),
        )

    # Train-test split before scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    # --- Helper to create scaler by type ---
    def make_scaler(kind: str, is_label: bool = False):
        kind = (kind or '').lower()
        if kind == 'minmax':
            return MinMaxScaler(feature_range=(0, 1))
        if kind == 'standard':
            return StandardScaler()
        if kind == 'robust':
            # Robust to outliers (IQR based)
            return RobustScaler()
        if kind == 'quantile':
            # Map distribution to normal; n_quantiles capped by training sample size
            n_train = X_train.shape[0] if not is_label else y_train.shape[0]
            n_q = min(1000, n_train)
            return QuantileTransformer(n_quantiles=n_q, output_distribution='normal', subsample=int(1e9), random_state=random_state)
        if kind == 'none':
            return None
        # Fallback
        logging.warning(f"Unknown scaler type '{kind}' - falling back to minmax")
        return MinMaxScaler(feature_range=(0, 1))

    X_scaler = make_scaler(feature_scaler_type)
    y_scaler = make_scaler(label_scaler_type, is_label=True)

    # Fit on training data to prevent data leakage
    if X_scaler is not None:
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)
    if y_scaler is not None:
        y_train = y_scaler.fit_transform(y_train)
        y_test = y_scaler.transform(y_test)
    
    if plots and y_scaler is not None:
        np.savez_compressed(
            os.path.join(plot_dir, "data_after_scaling.npz"),
            y_train=y_train, label_cols=np.array(label_cols),
        )
    if return_meta:
        scaler_meta = {
            'feature_scaler_type': feature_scaler_type,
            'label_scaler_type': label_scaler_type,
            'use_log': bool(use_log),
            'use_area_root': bool(use_area_root),
            'test_size': float(test_size),
            'random_state': int(random_state),
            'feature_cols': list(feature_cols),
            'label_cols': list(label_cols),
        }
        return X_train, X_test, X_scaler, y_train, y_test, y_scaler, scaler_meta
    return X_train, X_test, X_scaler, y_train, y_test, y_scaler

def load_file(csv_file: str = "data/Clean_Results_Isotherm.csv"):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        logging.error(f"CSV file not found at: {csv_file}")
        raise
    except Exception as e:
        logging.error(f"Failed to read CSV file: {e}")
        raise
    return df

def select_columns(df: pd.DataFrame, feature_cols: List[str], label_cols: List[str]):
    try:
        X = df[feature_cols].values
        y = df[label_cols].values
    except KeyError as e:
        logging.error(f"Column not found: {e}. Check your column names.")
        logging.error(f"Available columns: {df.columns.tolist()}")
        raise
    return X, y
