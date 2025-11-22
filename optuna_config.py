import argparse
import logging
from typing import Any, Dict, List

import numpy as np
import optuna
import torch

FEATURE_COLUMN_NAMES = [
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

MAX_EPOCHS = 500
PATIENCE = 25


def build_final_model_config(best_params: Dict[str, Any]) -> Dict[str, Any]:
    # Construct final model config from best_params. Missing fields will use conservative defaults.
    return {
        "num_epochs": 1000,
        "batch_size": best_params.get("batch_size", 64),
        "learning_rate": best_params.get("learning_rate", 1e-3),
        "nr_hidden_layers": best_params.get("nr_hidden_layers", 3),
        "nr_neurons": best_params.get("nr_neurons", 128),
        "expanding_layers": False,
        "contracting_layers": False,
        "activation_name": best_params.get("activation_name", "GELU"),
        "dropout_rate": best_params.get("dropout_rate", 0.0),
        "weight_decay": best_params.get("weight_decay", 0.0),
        "loss_criterion": best_params.get("loss_criterion", "SmoothL1"),
        "scheduler_type": best_params.get("scheduler_type", "CosineAnnealingLR"),
        "scheduler_T_max": best_params.get("scheduler_T_max", 1000),
        "warmup_epochs": best_params.get("warmup_epochs", 10),
        "plots": True,
        "patience": best_params.get("patience", 100)
    }


def load_best_params_or_default(study_name: str, storage: str) -> Dict[str, Any]:
    logging.info(f"Skipping Optuna. Loading best parameters from study: {study_name}...")
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )

        best_params = study.best_params.copy()
        best_trial = study.best_trial
        # Preserve user_attrs for fields that were stored (e.g. batch_size overrides)
        hardcoded_fields = ["batch_size", "nr_hidden_layers", "activation_name", "loss_criterion"]

        for field in hardcoded_fields:
            if field in best_trial.user_attrs:
                best_params[field] = best_trial.user_attrs[field]

        logging.info(f"Successfully loaded best parameters from trial {study.best_trial.number}:")
        logging.info(f"  Best value (RMSE): {study.best_value:.6f}")
        logging.info(f"  Number of trials completed: {len(study.trials)}")
        logging.info("  Optimized parameters:")
        for key, value in study.best_params.items():
            logging.info(f"    {key}: {value}")
        return best_params

    except KeyError:
        logging.error(f"Study '{study_name}' not found in database. Available studies:")
        try:
            all_studies = optuna.study.get_all_study_summaries(storage=storage)
            for summary in all_studies:
                logging.error(f"  - {summary.study_name}")
        except Exception:
            pass
    except Exception as exc:
        logging.error(f"Error loading previous study '{study_name}': {exc}")

    logging.info("Falling back to empty parameter set (no defaults).")
    return {}


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization for a NN.")
    parser.add_argument(
        '--target',
        type=str,
        default='all',
        choices=['Area', 'Iso_distance', 'Iso_width', 'all'],
        help="The target label to train on, or 'all' for all three. (Default: 'all')"
    )
    parser.add_argument(
        '--disable-power-monitor',
        action='store_true',
        help='Disable continuous power monitoring (enabled by default)'
    )
    parser.add_argument(
        '--power-interval',
        type=float,
        default=1.0,
        help='Sampling interval in seconds for the power monitor (default: 1.0)'
    )
    parser.add_argument(
        '--power-filter',
        type=str,
        default='python',
        help='Process-name filter for the power monitor (default: python)'
    )
    parser.add_argument(
        '--power-log-dir',
        type=str,
        default=None,
        help='Optional directory for storing power monitor logs (default: runs/.../power_monitor)'
    )
    parser.add_argument(
        '--csv-file',
        type=str,
        default='./data/Clean_Results_Isotherm.csv',
        help='Path to CSV data file'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        default=None,
        help='Optuna study name (default: auto-generated from target labels)'
    )
    parser.add_argument(
        '--storage-url',
        type=str,
        default=None,
        help='Optuna storage URL (supports RDB engines like PostgreSQL/MySQL)'
    )
    parser.add_argument(
        '--optuna-trials',
        type=int,
        default=10000,
        help='Number of trials to run for Optuna tuning'
    )
    parser.add_argument(
        '--optuna-workers',
        type=int,
        default=1,
        help='n_jobs value for Optuna study.optimize (capped by SQLite concurrency)'
    )
    parser.add_argument(
        '--run-tag',
        type=str,
        default=None,
        help='Optional suffix appended to the output run directory name'
    )
    parser.add_argument(
        '--optuna-max-epochs',
        type=int,
        default=MAX_EPOCHS,
        help=f'Maximum epochs per Optuna trial (default: {MAX_EPOCHS})'
    )
    parser.add_argument(
        '--optuna-patience',
        type=int,
        default=PATIENCE,
        help=f'Number of epochs with no RMSE improvement before pruning an Optuna trial (default: {PATIENCE})'
    )
    parser.add_argument(
        '--objective-batch-size',
        type=int,
        default=64,
        help='Batch size used during Optuna trials (default: 64)'
    )
    parser.add_argument(
        '--objective-hidden-layers',
        type=int,
        default=3,
        help='Number of hidden layers for Optuna trials (default: 3)'
    )
    parser.add_argument(
        '--objective-activation',
        type=str,
        default='GELU',
        choices=['ReLU', 'GELU', 'SiLU', 'LeakyReLU', 'Tanh'],
        help='Activation function to use during Optuna trials'
    )
    parser.add_argument(
        '--objective-loss',
        type=str,
        default='SmoothL1',
        choices=['L1', 'SmoothL1', 'MSE'],
        help='Loss function to evaluate Optuna trials'
    )
    args = parser.parse_args()

    return args


def validate_target_labels(targets: List[str]):
    valid_labels = ["Area", "Iso_distance", "Iso_width"]
    for label in targets:
        if label not in valid_labels:
            logging.error(f"Invalid target label: {label}. Valid options: {valid_labels}")
            raise ValueError(f"Invalid target label: {label}")