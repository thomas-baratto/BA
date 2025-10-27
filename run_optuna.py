import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import optuna
import logging
import os
import time
import datetime
from functools import partial
from typing import Dict, Any, List

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

# Define the label(s) you want to tune for.
# Common options based on your data: ["Iso_width"], ["Iso_distance"], or ["Area"]
OPTUNA_LABEL_NAMES = ["Area"]


# --- Optuna Objective Function ---

def objective(trial: optuna.Trial,
              data: Dict[str, Any],
              base_log_dir: str) -> float:
    """Optuna objective function."""
    
    # --- Hyperparameters ---
    num_epochs = trial.suggest_int("num_epochs", 100, 500)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024, 4096])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    nr_hidden_layers = trial.suggest_int("nr_hidden_layers", 1, 5)
    expanding_layers = trial.suggest_categorical("expanding_layers", [True, False])
    contracting_layers = trial.suggest_categorical("contracting_layers", [True, False])
    nr_neurons = trial.suggest_int("nr_neurons", 16, 256, log=True)
    nr_of_steps = trial.suggest_int("nr_of_steps", 1, 5)
    lr_scheduler_epoch_step = max(1, num_epochs // nr_of_steps) # Ensure step_size >= 1
    lr_scheduler_gamma = trial.suggest_float("lr_scheduler_gamma", 0.1, 0.9, log=True)
    loss_name = trial.suggest_categorical("criterion", ["MSE", "L1"])

    # --- Data Loaders ---
    # Data is pre-loaded and passed via 'data' dict
    train_dataset = CSVDataset(data["X_train"], data["y_train"])
    test_dataset = CSVDataset(data["X_test"], data["y_test"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model, Loss, Optimizer ---
    model = NeuralNetwork(
        input_size=data["X_train"].shape[1],
        output_size=data["y_train"].shape[1],
        nr_hidden_layers=nr_hidden_layers,
        nr_neurons=nr_neurons,
        exp_layers=expanding_layers,
        con_layers=contracting_layers
    ).to(DEVICE)
    
    criterion = nn.MSELoss() if loss_name == "MSE" else nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_scheduler_epoch_step, gamma=lr_scheduler_gamma
    )

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=os.path.join(base_log_dir, f"trial_{trial.number}"))
    start_time = time.time()

    # --- Training & Pruning Loop ---
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        scheduler.step()
        
        writer.add_scalar("Training/Loss", train_loss, epoch)
        log_resources(writer, epoch)

        # Report intermediate result to Optuna
        trial.report(train_loss, epoch)

        # Prune the trial if it's performing badly
        if trial.should_prune():
            writer.close()
            raise optuna.TrialPruned()

    # --- Final Evaluation ---
    test_loss, _, _ = evaluate(model, test_loader, criterion, DEVICE)
    
    writer.add_scalar("Test/Final_Loss", test_loss, 0)
    writer.add_scalar("System/Total_Train_Time_sec", time.time() - start_time, 0)
    writer.add_hparams(trial.params, {"hparam/test_loss": test_loss})
    writer.close()

    return test_loss # Optuna minimizes this value

# --- Main Execution ---

if __name__ == "__main__":
    
    # --- Configuration ---
    CSV_FILE = './data/Clean_Results_Isotherm.csv'
    
    # Create a single root folder for this entire run
    RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ROOT_FOLDER = f"runs/run_{RUN_TIMESTAMP}/"
    os.makedirs(ROOT_FOLDER, exist_ok=True)
    
    # Store DB in the parent 'runs' folder to persist across runs
    OPTUNA_DB_PATH = "sqlite:///runs/optuna_study.db" 
    OPTUNA_STUDY_NAME = "nn_hyperparam_study"
    OPTUNA_N_TRIALS = 50 # Number of trials to run
    
    # --- 1. Load Data ONCE for Optuna ---
    logging.info(f"Loading data for Optuna study (Labels: {OPTUNA_LABEL_NAMES})...")
    X_train, X_test, _, y_train, y_test, _ = load_data(
        csv_file=CSV_FILE, 
        feature_cols=FEATURE_COLUMN_NAMES,
        label_cols=OPTUNA_LABEL_NAMES,
        plots=False, # No plots during tuning
        rf=ROOT_FOLDER
    )
    # Pack data to pass to objective
    optuna_data = {
        "X_train": X_train, "y_train": y_train,
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
    
    # Use functools.partial to pass static data to the objective
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
    
    # Build the config for the final training run
    final_model_config = {
        "num_epochs": 1000, # Train final model for longer
        "batch_size": best_params["batch_size"],
        "learning_rate": best_params["learning_rate"],
        "nr_hidden_layers": best_params["nr_hidden_layers"],
        "nr_neurons": best_params["nr_neurons"],
        "expanding_layers": best_params["expanding_layers"],
        "contracting_layers": best_params["contracting_layers"],
        "lr_scheduler_epoch_step": max(1, 1000 // best_params["nr_of_steps"]),
        "lr_scheduler_gamma": best_params["lr_scheduler_gamma"],
        "plots": True # Enable plots for final model
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