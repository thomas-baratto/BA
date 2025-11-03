import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

# --- Import from own project files ---
from data_loader import load_data, CSVDataset
from model import NeuralNetwork
from utils import log_resources
from trainer import train_epoch, evaluate, main_train

# --- Setup ---
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {DEVICE}")

# --- Column Definitions ---
FEATURE_COLUMN_NAMES = [
    "Flow_well",
    "Temp_diff",
    # "Temp_diff_real", # Not used for training
    "kW_well",
    "Hydr_gradient",
    "Hydr_conductivity",
    "Aqu_thickness",
    "Long_dispersivity",
    "Trans_dispersivity",
    "Isotherm"
]

# --- Optuna Settings ---
MAX_EPOCHS = 500 # Max epochs for Optuna trials
PATIENCE = 25   # Early stopping patience

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial,
              data: Dict[str, Any],
              base_log_dir: str) -> float:
    """Optuna objective function."""
    
    # --- Hyperparameters ---
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024, 4096])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    nr_hidden_layers = trial.suggest_int("nr_hidden_layers", 1, 5)
    #expanding_layers = trial.suggest_categorical("expanding_layers", [True, False]) These parameters seemed to be unnecessary during training
    #contracting_layers = trial.suggest_categorical("contracting_layers", [True, False])
    nr_neurons = trial.suggest_int("nr_neurons", 16, 256, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    activation_name = trial.suggest_categorical("activation_name", ["ReLU", "LeakyReLU", "GELU", "ELU"])
    loss_name = trial.suggest_categorical("loss_criterion", ["MSE", "L1"])

    # --- Data Loaders ---
    train_dataset = CSVDataset(data["X_train"], data["y_train"])
    val_dataset = CSVDataset(data["X_val"], data["y_val"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Model, Loss, Optimizer ---
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
    
    criterion = nn.MSELoss() if loss_name == "MSE" else nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=os.path.join(base_log_dir, f"trial_{trial.number}"))
    start_time = time.time()

    # --- Training & Pruning Loop ---
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        scheduler.step()
        val_loss, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        writer.add_scalar("Training/Loss", train_loss, epoch)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        log_resources(writer, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            logging.info(f"Trial {trial.number}: Early stopping at epoch {epoch}.")
            break

        trial.report(val_loss, epoch)
        if trial.should_prune():
            writer.close()
            raise optuna.TrialPruned()

    # --- End of Trial ---
    writer.add_scalar("System/Total_Train_Time_sec", time.time() - start_time, 0)
    writer.add_hparams(trial.params, {"hparam/best_val_loss": best_val_loss})
    writer.close()

    return best_val_loss 

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
    args = parser.parse_args()

    all_labels = ["Area", "Iso_distance", "Iso_width"]
    
    if args.target == 'all':
        OPTUNA_LABEL_NAMES = all_labels
    else:
        OPTUNA_LABEL_NAMES = [args.target]
    
    logging.info(f"Target labels for this run: {OPTUNA_LABEL_NAMES}")
    
    # --- Configuration ---
    CSV_FILE = './data/Clean_Results_Isotherm.csv'
    
    RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ROOT_FOLDER = f"runs/run_{RUN_TIMESTAMP}_{str(OPTUNA_LABEL_NAMES)}/"
    os.makedirs(ROOT_FOLDER, exist_ok=True)
    
    OPTUNA_DB_PATH = "sqlite:///runs/optuna_study.db" 
    OPTUNA_STUDY_NAME = f"nn_study_{str(OPTUNA_LABEL_NAMES)}"
    OPTUNA_N_TRIALS = 100 
    
    # --- 1. Load Data ONCE for Optuna ---
    logging.info(f"Loading data for Optuna study (Labels: {OPTUNA_LABEL_NAMES})...")
    X_train, X_test, _, y_train, y_test, _ = load_data(
        csv_file=CSV_FILE, 
        feature_cols=FEATURE_COLUMN_NAMES,
        label_cols=OPTUNA_LABEL_NAMES,
        plots=False, 
        rf=ROOT_FOLDER
    )
    
    #Create validation set from training data
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    optuna_data = {
        "X_train": X_train_main, "y_train": y_train_main,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }
    
    # --- 2. Run Optuna Study ---
    logging.info(f"Starting Optuna study: {OPTUNA_STUDY_NAME}...")
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
    logging.info(f"Best Params: {study.best_params}")

    # --- 3. Train Final Model with Best Params ---
    logging.info("Training final model with best parameters...")
    
    best_params = study.best_params
    
    final_model_config = {
        "num_epochs": 1000, 
        "batch_size": best_params["batch_size"],
        "learning_rate": best_params["learning_rate"],
        "nr_hidden_layers": best_params["nr_hidden_layers"],
        "nr_neurons": best_params["nr_neurons"],
        "expanding_layers": False,
        "contracting_layers": False,
        "activation_name": best_params.get("activation_name", "ReLU"),
        "dropout_rate": best_params.get("dropout_rate", 0.0),
        "weight_decay": best_params.get("weight_decay", 0.0),
        "loss_criterion": best_params["loss_criterion"],
        "scheduler_type": "CosineAnnealingLR",
        "scheduler_T_max": 1000,
        "plots": True 
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
    
    # --- 4. Save the Final Model ---
    final_model_path = os.path.join(final_model_rf, "best_model.pt")
    torch.save(final_model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    logging.info("--- Run complete ---")
    logging.info(f"To view Optuna results: optuna-dashboard {OPTUNA_DB_PATH}")
    logging.info(f"To view TensorBoard logs: tensorboard --logdir {ROOT_FOLDER}")