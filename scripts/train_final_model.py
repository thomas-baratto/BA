# train the best model from isotherm (830) or depression cones (832).
import torch
import torch.nn as nn
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
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

from core.trainer import main_train, evaluate
from core.data_loader import load_data, CSVDataset
from core.utils import compute_regression_metrics, ResourceLogger
from scripts.run_optuna import detect_columns_from_csv
from monitoring.power_utils import power_monitor_session


MAX_EPOCHS = 1000

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

STUDY_NAME_DEPRESSION_CONES = 'depression_cones_mlp_journal_study'
STUDY_NAME_ISOTHERM = 'nn_study_v3_Area_Iso_distance_Iso_width'

PATH_DEPRESSION_CONE_STUDY = '/home/barattts/lavoltabuona/BA/runs/global_run_832/optuna_journal_storage/journal.log'
PATH_ISOTHERM_STUDY = '/home/barattts/lavoltabuona/BA/runs/global_run_830/optuna_journal_storage/journal.log'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train final model with best Optuna parameters')
    parser.add_argument(
        '--study-name',
        type=str,
        default=STUDY_NAME_ISOTHERM,
        help='Optuna study name to load best parameters from'
    )
    parser.add_argument(
        '--csv-file',
        type=str,
        default='data/Clean_Results_Isotherm.csv',
        help='Path to CSV data file'
    )
    parser.add_argument(
        '--journal-path',
        type=str,
        default=PATH_ISOTHERM_STUDY,
        help='Path to the Optuna journal log file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save results and model. If None, a timestamped directory will be created.'
    )
    # --- Power Monitor Arguments ---
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

    # Load the Optuna study
    logging.info(f"Loading Optuna study: {args.study_name}")
    logging.info(f"Journal path: {args.journal_path}")

    if not os.path.exists(args.journal_path):
        raise FileNotFoundError(f"Journal file not found at: {args.journal_path}. Please provide a valid path to an existing Optuna journal.")

    storage = JournalStorage(JournalFileBackend(args.journal_path))

    study = optuna.load_study(
        study_name=args.study_name,
        storage=storage
    )
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
    final_model_config = study.best_params.copy()
    
    # Normalize parameter names (handle variations between studies)
    if "feature_scaler" in final_model_config and "feature_scaler_type" not in final_model_config:
        final_model_config["feature_scaler_type"] = final_model_config["feature_scaler"]
    
    if "label_scaler" in final_model_config and "label_scaler_type" not in final_model_config:
        final_model_config["label_scaler_type"] = final_model_config["label_scaler"]

    # Ensure 'plots' key exists (required by main_train)
    final_model_config["plots"] = True
    
    # Ensure 'num_epochs' exists
    if "num_epochs" not in final_model_config:
        final_model_config["num_epochs"] = MAX_EPOCHS

    with power_monitor_session(args, output_dir):
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
