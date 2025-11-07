import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import optuna
import logging
import os
import time
import datetime
from functools import partial
from typing import Dict, Any, List
import argparse
from sklearn.metrics import root_mean_squared_error  # <-- 1. CLEAN IMPORT

# --- Import from own project files ---
from data_loader import load_data, CSVDataset
from model import NeuralNetwork
from utils import log_resources
from trainer import train_epoch, evaluate, main_train

# --- Setup ---
# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {DEVICE}")

# --- Column Definitions ---
FEATURE_COLUMN_NAMES = [
    "Flow_well", "Temp_diff", "kW_well", "Hydr_gradient", "Hydr_conductivity",
    "Aqu_thickness", "Long_dispersivity", "Trans_dispersivity", "Isotherm"
]

# --- Optuna Settings ---
MAX_EPOCHS = 500
PATIENCE = 25

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial,
              data: Dict[str, Any],
              base_log_dir: str) -> float:
    """Optuna objective function."""
    
    # Hardcoded parameters (found to be less important after tuning)
    # These will be stored even if not optimized
    batch_size = 64  # Not tuned
    nr_hidden_layers = 3  # Not tuned
    activation_name = "GELU"  # Not tuned
    loss_name = "SmoothL1"  # Not tuned
    
    # Parameters being optimized
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    nr_neurons = trial.suggest_int("nr_neurons", 16, 256, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    # Store hardcoded values in trial params for later retrieval
    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("nr_hidden_layers", nr_hidden_layers)
    trial.set_user_attr("activation_name", activation_name)
    trial.set_user_attr("loss_criterion", loss_name)

    train_dataset = CSVDataset(data["X_train"], data["y_train"])
    val_dataset = CSVDataset(data["X_val"], data["y_val"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = NeuralNetwork(
        input_size=data["X_train"].shape[1],
        output_size=data["y_train"].shape[1],
        nr_hidden_layers=nr_hidden_layers,
        nr_neurons=nr_neurons,
        activation_name=activation_name,
        exp_layers=False,
        con_layers=False,
        dropout_rate=dropout_rate,
    ).to(DEVICE)
    
    if loss_name == "L1":
        criterion = nn.L1Loss()
    elif loss_name == "SmoothL1":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    writer = SummaryWriter(log_dir=os.path.join(base_log_dir, f"trial_{trial.number}"))
    start_time = time.time()

    # --- Training & Pruning Loop ---
    best_val_rmse = float('inf')
    patience_counter = 0
    
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        scheduler.step()
        
        val_loss, predictions, true_values = evaluate(model, val_loader, criterion, DEVICE)
        
        current_val_rmse = root_mean_squared_error(true_values, predictions)
        
        writer.add_scalar("Training/Loss", train_loss, epoch)
        writer.add_scalar("Validation/Loss_raw", val_loss, epoch)
        writer.add_scalar("Validation/RMSE", current_val_rmse, epoch)
        log_resources(writer, epoch)
        
        if current_val_rmse < best_val_rmse:
            best_val_rmse = current_val_rmse
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            logging.info(f"Trial {trial.number}: Early stopping at epoch {epoch}.")
            break

        trial.report(current_val_rmse, epoch)
        if trial.should_prune():
            writer.close()
            raise optuna.TrialPruned()

    writer.add_scalar("System/Total_Train_Time_sec", time.time() - start_time, 0)
    writer.add_hparams(trial.params, {"hparam/best_val_rmse": best_val_rmse})
    writer.close()

    return best_val_rmse 

# --- Main Execution ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization for a NN.")
    parser.add_argument(
        '--target', 
        type=str, 
        default='all', 
        choices=['Area', 'Iso_distance', 'Iso_width', 'all'],
        help="The target label to train on, or 'all' for all three. (Default: 'all')"
    )
    parser.add_argument(
        '--skip-optuna', 
        action='store_true', 
        help='Skip hyperparameter tuning and use best params from DB'
    )
    args = parser.parse_args()

    all_labels = ["Area", "Iso_distance", "Iso_width"]
    if args.target == 'all':
        OPTUNA_LABEL_NAMES = all_labels
    else:
        OPTUNA_LABEL_NAMES = [args.target]
    
    logging.info(f"Target labels for this run: {OPTUNA_LABEL_NAMES}")
    logging.info(f"Skip Optuna: {args.skip_optuna}")
    
    # Validate CSV file and columns
    CSV_FILE = './data/Clean_Results_Isotherm.csv'
    if not os.path.exists(CSV_FILE):
        logging.error(f"CSV file not found: {CSV_FILE}")
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")
    
    # Validate target labels
    valid_labels = ["Area", "Iso_distance", "Iso_width"]
    for label in OPTUNA_LABEL_NAMES:
        if label not in valid_labels:
            logging.error(f"Invalid target label: {label}. Valid options: {valid_labels}")
            raise ValueError(f"Invalid target label: {label}")
    
    RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create clean folder and study names
    label_str = "_".join(OPTUNA_LABEL_NAMES)
    ROOT_FOLDER = f"runs/run_{RUN_TIMESTAMP}_{label_str}/"
    os.makedirs(ROOT_FOLDER, exist_ok=True)
    OPTUNA_DB_PATH = "sqlite:///runs/optuna_study.db" 
    OPTUNA_STUDY_NAME = f"nn_study_{label_str}"
    OPTUNA_N_TRIALS = 100  # Adjust as needed 
    
    if args.skip_optuna:
        logging.info(f"Skipping Optuna. Loading best parameters from study: {OPTUNA_STUDY_NAME}...")
        try:
            study = optuna.load_study(
                study_name=OPTUNA_STUDY_NAME,
                storage=OPTUNA_DB_PATH
            )
            
            # Get optimized parameters
            best_params = study.best_params.copy()
            
            # Get hardcoded parameters from user attributes (for new studies)
            # or use defaults (for old studies created before parameter handling update)
            best_trial = study.best_trial
            hardcoded_from_attrs = []
            hardcoded_defaults = []
            
            for attr_name in ["batch_size", "nr_hidden_layers", "activation_name", "loss_criterion"]:
                if attr_name in best_trial.user_attrs:
                    best_params[attr_name] = best_trial.user_attrs[attr_name]
                    hardcoded_from_attrs.append(attr_name)
                elif attr_name not in best_params:
                    # Parameter not in user_attrs or best_params - use default
                    defaults = {"batch_size": 64, "nr_hidden_layers": 3, 
                               "activation_name": "GELU", "loss_criterion": "SmoothL1"}
                    best_params[attr_name] = defaults[attr_name]
                    hardcoded_defaults.append(attr_name)
            
            logging.info(f"Successfully loaded best parameters from trial {study.best_trial.number}:")
            logging.info(f"  Best value (RMSE): {study.best_value:.6f}")
            logging.info(f"  Number of trials completed: {len(study.trials)}")
            logging.info(f"  Optimized parameters:")
            for key, value in study.best_params.items():
                logging.info(f"    {key}: {value}")
            
            if hardcoded_from_attrs:
                logging.info(f"  Hardcoded parameters (from study):")
                for attr_name in hardcoded_from_attrs:
                    logging.info(f"    {attr_name}: {best_params[attr_name]}")
            
            if hardcoded_defaults:
                logging.info(f"  Hardcoded parameters (using defaults, study created before parameter handling update):")
                for attr_name in hardcoded_defaults:
                    logging.info(f"    {attr_name}: {best_params[attr_name]}")

                    
        except KeyError as e:
            logging.error(f"Study '{OPTUNA_STUDY_NAME}' not found in database. Available studies:")
            try:
                all_studies = optuna.study.get_all_study_summaries(storage=OPTUNA_DB_PATH)
                for study_summary in all_studies:
                    logging.error(f"  - {study_summary.study_name}")
            except Exception:
                pass
            logging.info("Falling back to default parameters...")
            best_params = {
                "batch_size": 64, "learning_rate": 0.001, "nr_hidden_layers": 3,
                "nr_neurons": 128, "activation_name": "GELU", "dropout_rate": 0.1,
                "weight_decay": 1e-05, "loss_criterion": "SmoothL1"
            }
        except Exception as e:
            logging.error(f"Error loading previous study: {e}")
            logging.info("Falling back to default parameters...")
            best_params = {
                "batch_size": 64, "learning_rate": 0.001, "nr_hidden_layers": 3,
                "nr_neurons": 128, "activation_name": "GELU", "dropout_rate": 0.1,
                "weight_decay": 1e-05, "loss_criterion": "SmoothL1"
            }
    else:
        logging.info(f"Starting Optuna study: {OPTUNA_STUDY_NAME}...")
        logging.info(f"Loading data for Optuna study (Labels: {OPTUNA_LABEL_NAMES})...")
        X_train, X_test, _, y_train, y_test, _ = load_data(
            csv_file=CSV_FILE, 
            feature_cols=FEATURE_COLUMN_NAMES,
            label_cols=OPTUNA_LABEL_NAMES,
            plots=False, 
            rf=ROOT_FOLDER
        )
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        optuna_data = {
            "X_train": X_train_main, "y_train": y_train_main,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        }
        study = optuna.create_study(
            study_name=OPTUNA_STUDY_NAME,
            direction="minimize",
            storage=OPTUNA_DB_PATH,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        objective_with_data = partial(
            objective,
            data=optuna_data,
            base_log_dir=os.path.join(ROOT_FOLDER, "optuna_tensorboard_logs")
        )
        study.optimize(objective_with_data, n_trials=OPTUNA_N_TRIALS)

        logging.info(f"Optuna study complete. Best trial: {study.best_trial.number}")
        logging.info(f"Best Loss: {study.best_value}")
        best_params = study.best_params
        logging.info(f"Best Params: {best_params}")

    logging.info("Training final model with parameters...")
    final_model_config = {
        "num_epochs": 1000, 
        "batch_size": best_params.get("batch_size", 64),  # Default for old studies
        "learning_rate": best_params["learning_rate"],
        "nr_hidden_layers": best_params.get("nr_hidden_layers", 3),  # Default for old studies
        "nr_neurons": best_params["nr_neurons"],
        "expanding_layers": False,
        "contracting_layers": False,
        "activation_name": best_params.get("activation_name", "GELU"),
        "dropout_rate": best_params.get("dropout_rate", 0.0),
        "weight_decay": best_params.get("weight_decay", 0.0),
        "loss_criterion": best_params.get("loss_criterion", "SmoothL1"),  # Default for old studies
        "scheduler_type": "CosineAnnealingLR",
        "scheduler_T_max": 1000,
        "warmup_epochs": 10,  # Add warmup for stable training
        "plots": True,
        "patience": 100 
    }
    final_model_rf = os.path.join(ROOT_FOLDER, "final_model")
    final_model = main_train(
        config=final_model_config,
        rf=final_model_rf,
        csv_file=CSV_FILE,
        feature_cols=FEATURE_COLUMN_NAMES,
        label_cols=OPTUNA_LABEL_NAMES,
        device=DEVICE
    )
    
    final_model_path = os.path.join(final_model_rf, "best_model.pt")
    torch.save(final_model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    # Save model configuration for easy loading
    model_config_path = os.path.join(final_model_rf, "model_config.json")
    import json
    model_config = {
        'input_size': len(FEATURE_COLUMN_NAMES),
        'output_size': len(OPTUNA_LABEL_NAMES),
        'nr_hidden_layers': final_model_config['nr_hidden_layers'],
        'nr_neurons': final_model_config['nr_neurons'],
        'activation_name': final_model_config['activation_name'],
        'dropout_rate': final_model_config['dropout_rate']
    }
    with open(model_config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    logging.info(f"Model configuration saved to {model_config_path}")
    
    logging.info("--- Run complete ---")
    logging.info(f"To view Optuna results: optuna-dashboard {OPTUNA_DB_PATH}")
    logging.info(f"To view TensorBoard logs: tensorboard --logdir {ROOT_FOLDER}")