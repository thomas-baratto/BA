import argparse
import logging
import random
from typing import List

import numpy as np
import torch

# --- Global Constants for Optuna Tuning ---
MAX_EPOCHS = 200
PATIENCE = 50

# List of input features to select from Clean_Results_Isotherm.csv
FEATURE_COLUMN_NAMES_ISOTHERM = [
    "Flow_well",
    "Temp_diff",
    "kW_well",
    "Hydr_gradient",
    "Hydr_conductivity",
    "Aqu_thickness",
    "Long_dispersivity",
    "Trans_dispersivity",
    "Isotherm"
]

FEATURE_COLUMN_NAMES_DEPRESSION = [
    "Flow_well",
    "Hydr_gradient",
    "Hydr_conductivity",
    "Aqu_thickness"
]

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def validate_target_labels(labels: List[str]):
    """Validates that the target labels are correctly defined."""
    valid_labels = {"Area", "Iso_distance", "Iso_width", "Cone"}
    if not all(label in valid_labels for label in labels):
        logging.error(f"Invalid target labels detected: {labels}. Valid choices are: {valid_labels}")
        raise ValueError("Invalid target label configuration.")

def parse_args():
    """Parses command-line arguments for both Optuna and general runs."""
    parser = argparse.ArgumentParser(description="Run Hyperparameter Optimization or Model Training.")
    
    # --- General Run Arguments ---
    parser.add_argument(
        '--target',
        type=str,
        default='all',
        choices=['Area', 'Iso_distance', 'Iso_width', 'Cone', 'all'],
        help='Target labels to predict (comma-separated or "all").'
    )
    parser.add_argument(
        '--csv-file',
        type=str,
        default='./data/Clean_Results_Isotherm.csv',
        help='Path to the input CSV data file.'
    )
    parser.add_argument(
        '--run-tag',
        type=str,
        default=None,
        help='An optional descriptive tag for the current run\'s artifacts.'
    )

    # --- Optuna & Distributed Storage Arguments (FIXED) ---
    parser.add_argument(
        '--study-name',
        type=str,
        default=None,
        help='Name of the Optuna study.'
    )
    # CRITICAL FIX: Replaced obsolete --storage-url argument
    parser.add_argument(
        '--storage-path', 
        type=str,
        default=None,
        help='Path to the Journal Storage directory (e.g., runs/my_journal).'
    )
    parser.add_argument(
        '--optuna-trials',
        type=int,
        default=100,
        help='Number of trials this specific worker process should run.'
    )
    parser.add_argument(
        '--optuna-workers',
        type=int,
        default=1,
        help='Internal n_jobs parameter for Optuna (should be 1 for multi-process distributed tuning).'
    )
    parser.add_argument(
        '--optuna-max-epochs',
        type=int,
        default=MAX_EPOCHS,
        help='Maximum number of epochs for one Optuna trial.'
    )
    parser.add_argument(
        '--optuna-patience',
        type=int,
        default=PATIENCE,
        help='Patience limit for early stopping during Optuna trials.'
    )

    # --- Objective Configuration (Optional default overrides) ---
    parser.add_argument(
        '--objective-batch-size',
        type=int,
        default=64,
        help='Default batch size for objective (used as a hint for search space).'
    )
    parser.add_argument(
        '--objective-hidden-layers',
        type=int,
        default=3,
        help='Default hidden layers for objective (used as a hint for search range).'
    )
    parser.add_argument(
        '--objective-activation',
        type=str,
        default='GELU',
        choices=['ReLU', 'GELU', 'SiLU', 'LeakyReLU', 'Tanh'],
        help='Default activation function for objective.'
    )
    parser.add_argument(
        '--objective-loss',
        type=str,
        default='SmoothL1',
        choices=['L1', 'SmoothL1', 'MSE'],
        help='Default loss function for objective.'
    )
    
    # --- Power Monitor Arguments (from power_utils) ---
    parser.add_argument(
        '--disable-power-monitor',
        action='store_true',
        help='Disable the per-worker power monitoring session.'
    )
    parser.add_argument(
        '--power-interval',
        type=float,
        default=1.0,
        help='Interval in seconds for power/resource logging.'
    )
    parser.add_argument(
        '--power-filter',
        type=str,
        default='python',
        help='Process name filter for the power monitor.'
    )
    parser.add_argument(
        '--power-log-dir',
        type=str,
        default=None,
        help='Override directory for power logs.'
    )


    args = parser.parse_args()
    return args