import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress DataConversionWarning from sklearn
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# --- Dataset Class ---

class CSVDataset(Dataset):
    """
    Custom PyTorch Dataset for loading tabular data from arrays.
    Data is converted to tensors upon initialization.
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        # Convert to tensors once during initialization for efficiency
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

# --- Data Loading & Preprocessing ---

def load_data(csv_file: str,
              feature_cols: List[str],
              label_cols: List[str],
              test_size: float = 0.3,
              random_state: int = 42,
              plots: bool = False,
              rf: str = '.') -> Tuple:
    """
    Loads, preprocesses, and splits data from a CSV file
    using column names.
    """
    # Step 1: Try to load the file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        logging.error(f"CSV file not found at: {csv_file}")
        raise
    except Exception as e:
        logging.error(f"Failed to read CSV file: {e}")
        raise
    
    # Step 2: Try to select columns (df is guaranteed to exist here)
    try:
        X = df[feature_cols].values
        y = df[label_cols].values
    except KeyError as e:
        logging.error(f"Column not found: {e}. Check your column names.")
        # This line is now safe, as df is not None
        logging.error(f"Available columns: {df.columns.tolist()}")
        raise
    
    # Ensure y is 2D
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Create a unique, filesystem-friendly name for the plot directory
    plot_dir_name = "_".join(label_cols).replace(" ", "_").replace("/", "_")
    plot_dir = os.path.join(rf, "plots", plot_dir_name)
    
    if plots:
        os.makedirs(plot_dir, exist_ok=True)
        
        # --- 1. "Before Transformation" Plot ---
        plt.figure()
        for label_idx in range(y.shape[1]):
            plt.hist(y[:, label_idx], bins=200, label=f'Label: {label_cols[label_idx]}')
        plt.title("Label Distribution (Before Transformation)")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "before_transform.png"))
        plt.close()

    # Log-transform
    X = np.log1p(X)
    y = np.log1p(y)

    if plots:
        # --- 2. "After Log-Transformation" Plot ---
        plt.figure()
        for label_idx in range(y.shape[1]):
            plt.hist(y[:, label_idx], bins=200, label=f'Label: {label_cols[label_idx]}')
        plt.title("Label Distribution (After Log-Transformation)")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "after_log_transform.png"))
        plt.close()

    # Train-test split *before* scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    # Initialize scalers
    X_scaler = MinMaxScaler((0, 1))
    y_scaler = MinMaxScaler((0, 1))

    # --- FIX: Fit on training data ONLY to prevent data leakage ---
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)  # Use transform only

    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)  # Use transform only
    
    if plots:
        # --- 3. "After Standardization" Plot ---
        # IMPORTANT: We plot y_train here!
        plt.figure()
        for label_idx in range(y_train.shape[1]):
            plt.hist(y_train[:, label_idx], bins=200, label=f'Label: {label_cols[label_idx]}')
        plt.title("Label Distribution (After Standardization on Training Data)")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "after_standardization.png"))
        plt.close()

    return X_train, X_test, X_scaler, y_train, y_test, y_scaler