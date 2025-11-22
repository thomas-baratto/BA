#!/usr/bin/env python3
"""
Train the final model using the best parameters from an Optuna study.
This script should be run AFTER the Optuna optimization is complete.
"""

import torch
import torch.nn as nn
import optuna
import logging
import os
import json
import argparse
import time
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from trainer import main_train, evaluate
from data_loader import load_data, CSVDataset
from utils import compute_regression_metrics, ResourceLogger
from run_optuna import detect_columns_from_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train final model with best Optuna parameters')
    parser.add_argument(
        '--study-name',
        type=str,
        default='nn_study_v3_Area_Iso_distance_Iso_width',
        help='Optuna study name to load best parameters from'
    )
    parser.add_argument(
        '--storage-url',
        type=str,
        default=None,
        help='Optuna storage URL (e.g., postgresql+psycopg2://user:pass@host:5432/db)'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='[Deprecated] Alias for --storage-url'
    )
    parser.add_argument(
        '--csv-file',
        type=str,
        default='data/Clean_Results_Isotherm.csv',
        help='Path to CSV data file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for final model (default: runs/final_model_TIMESTAMP)'
    )
    args = parser.parse_args()

    default_storage = 'sqlite:///runs/optuna_study.db'
    storage_url = args.storage_url or args.db_path or default_storage
    if args.db_path and not args.storage_url:
        logging.warning("--db-path is deprecated; prefer --storage-url going forward.")

    args.storage_url = storage_url

    # Load the Optuna study
    logging.info(f"Loading Optuna study: {args.study_name}")
    logging.info(f"Optuna storage URL: {args.storage_url}")
    try:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=args.storage_url
        )
    except KeyError:
        logging.error(f"Study '{args.study_name}' not found in database {args.storage_url}")
        logging.error("Available studies:")
        storage = optuna.storages.get_storage(args.storage_url)
        for study_summary in storage.get_all_study_summaries():
            logging.error(f"  - {study_summary.study_name}")
        exit(1)

    best_params = study.best_params
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best RMSE: {study.best_value:.6f}")
    logging.info(f"Best parameters: {best_params}")

    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"runs/final_model_{timestamp}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Detect feature and label columns from CSV file (same as run_optuna.py)
    FEATURE_COLUMN_NAMES, AVAILABLE_LABEL_NAMES = detect_columns_from_csv(args.csv_file)
    
    # Use all available labels for training
    LABEL_NAMES = AVAILABLE_LABEL_NAMES
    
    logging.info(f"Features ({len(FEATURE_COLUMN_NAMES)}): {FEATURE_COLUMN_NAMES}")
    logging.info(f"Labels ({len(LABEL_NAMES)}): {LABEL_NAMES}")

    # Build final model configuration
    # Defaults from Trial 4348: RMSE = 0.000404
    final_model_config = {
        "num_epochs": 1000,
        "batch_size": best_params.get("batch_size", 64),
        "learning_rate": best_params.get("learning_rate", 0.0018866994085187415),
        "nr_hidden_layers": best_params.get("nr_hidden_layers", 3),
        "nr_neurons": best_params.get("nr_neurons", 211),
        "expanding_layers": False,
        "contracting_layers": False,
        "activation_name": best_params.get("activation_name", "GELU"),
        "dropout_rate": best_params.get("dropout_rate", 7.536264706091639e-05),
        "weight_decay": best_params.get("weight_decay", 3.151812299704266e-05),
        "loss_criterion": best_params.get("loss_criterion", "L1"),
        "feature_scaler_type": best_params.get("feature_scaler_type", "robust"),
        "label_scaler_type": best_params.get("label_scaler_type", "minmax"),
        "scheduler_type": best_params.get("scheduler_type", "ReduceLROnPlateau"),
        "scheduler_factor": best_params.get("scheduler_factor", 0.6119656651619391),
        "scheduler_patience": best_params.get("scheduler_patience", 12),
        "scheduler_cooldown": best_params.get("scheduler_cooldown", 4),
        "scheduler_min_lr": best_params.get("scheduler_min_lr", 2.7910431835412616e-07),
        "scheduler_threshold": best_params.get("scheduler_threshold", 0.0006218010930699477),
        "scheduler_threshold_mode": best_params.get("scheduler_threshold_mode", "rel"),
        "scheduler_T_max": best_params.get("scheduler_T_max", 500),
        "scheduler_T_0": best_params.get("scheduler_T_0", 50),
        "scheduler_T_mult": best_params.get("scheduler_T_mult", 2),
        "scheduler_eta_min": best_params.get("scheduler_eta_min", 1e-6),
        "lr_scheduler_epoch_step": best_params.get("lr_scheduler_epoch_step", 100),
        "lr_scheduler_gamma": best_params.get("lr_scheduler_gamma", 0.5),
        "warmup_epochs": best_params.get("warmup_epochs", 0),
        "warmup_start_factor": best_params.get("warmup_start_factor", 0.23720906567687325),
        "plots": True,
        "use_batchnorm": False,
        "patience": 100
    }

    # Train the final model
    logging.info("Training final model with best parameters...")
    train_start = time.time()
    
    # First, load data to get metrics on all splits
    X_train_full, X_test, X_scaler, y_train_full, y_test, y_scaler = load_data(
        csv_file=args.csv_file,
        feature_cols=FEATURE_COLUMN_NAMES,
        label_cols=LABEL_NAMES,
        plots=False,
        rf=output_dir,
        feature_scaler_type=final_model_config["feature_scaler_type"],
        label_scaler_type=final_model_config["label_scaler_type"]
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    # Train the model using main_train
    final_model = main_train(
        config=final_model_config,
        rf=output_dir,
        csv_file=args.csv_file,
        feature_cols=FEATURE_COLUMN_NAMES,
        label_cols=LABEL_NAMES,
        device=DEVICE
    )
    
    train_time = time.time() - train_start
    
    # Evaluate on all splits and compute metrics
    logging.info("Computing metrics on all splits...")
    results = {}
    
    # Create data loaders for evaluation
    train_dataset = CSVDataset(X_train, y_train)
    val_dataset = CSVDataset(X_val, y_val)
    test_dataset = CSVDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=final_model_config["batch_size"], 
                             shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=final_model_config["batch_size"], 
                           shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=final_model_config["batch_size"], 
                            shuffle=False, num_workers=0, pin_memory=True)
    
    # Get criterion
    loss_name = final_model_config.get("loss_criterion", "SmoothL1")
    if loss_name == "L1":
        criterion = nn.L1Loss()
    elif loss_name == "SmoothL1":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()
    
    for split_name, loader, X_split, y_split in [
        ('train', train_loader, X_train, y_train),
        ('val', val_loader, X_val, y_val),
        ('test', test_loader, X_test, y_test)
    ]:
        # Get predictions
        loss, predictions, true_values = evaluate(final_model, loader, criterion, DEVICE)
        
        # Inverse transform
        predictions_original = y_scaler.inverse_transform(predictions)
        true_values_original = y_scaler.inverse_transform(true_values)
        
        # Apply inverse log transform (expm1)
        predictions_original = np.expm1(predictions_original)
        true_values_original = np.expm1(true_values_original)
        
        # Compute comprehensive metrics
        metrics = compute_regression_metrics(true_values_original, predictions_original)
        
        results[split_name] = {
            'loss': float(loss),
            **{k: float(v) for k, v in metrics.items()}
        }
        
        logging.info(f"\n{split_name.upper()} metrics:")
        logging.info(f"  Loss ({loss_name}): {loss:.6f}")
        logging.info(f"  RMSE: {metrics['rmse']:.6f}")
        logging.info(f"  MAE:  {metrics['mae']:.6f}")
        logging.info(f"  R²:   {metrics['r2']:.6f}")
    
    # Save comprehensive results
    results_dict = {
        'config': final_model_config,
        'study_info': {
            'study_name': args.study_name,
            'best_trial': study.best_trial.number,
            'best_trial_rmse': study.best_value,
            'n_trials': len(study.trials)
        },
        'metrics': results,
        'data_shapes': {
            'n_features': X_train.shape[1],
            'n_outputs': y_train.shape[1],
            'train_samples': X_train.shape[0],
            'val_samples': X_val.shape[0],
            'test_samples': X_test.shape[0]
        },
        'train_time_seconds': train_time,
        'device': str(DEVICE)
    }
    
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logging.info(f"Results saved to {results_file}")
    
    # Save the model
    final_model_path = os.path.join(output_dir, "best_model.pt")
    torch.save(final_model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    # Save model configuration for easy loading
    model_config_path = os.path.join(output_dir, "model_config.json")
    model_config = {
        'input_size': len(FEATURE_COLUMN_NAMES),
        'output_size': len(LABEL_NAMES),
        'nr_hidden_layers': final_model_config['nr_hidden_layers'],
        'nr_neurons': final_model_config['nr_neurons'],
        'activation_name': final_model_config['activation_name'],
        'dropout_rate': final_model_config['dropout_rate'],
        'feature_scaler_type': final_model_config['feature_scaler_type'],
        'label_scaler_type': final_model_config['label_scaler_type'],
        'best_params': best_params,
        'study_name': args.study_name,
        'best_trial': study.best_trial.number,
        'best_rmse': study.best_value
    }
    with open(model_config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    logging.info(f"Model configuration saved to {model_config_path}")
    
    logging.info("--- Final model training complete ---")
    logging.info(f"Model directory: {output_dir}")
