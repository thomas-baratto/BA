import datetime
import logging
import os
import sys

import optuna
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend 

from core.data_loader import load_data
from optimization.optuna_config import (
    parse_args,
    set_seed,
    validate_target_labels,
)
from optimization.optuna_objective import build_objective
from monitoring.power_utils import power_monitor_session

# Known feature and label columns for dynamic detection
KNOWN_FEATURES = [
    "Flow_well", "Temp_diff", "Temp_diff_real", "kW_well", "Hydr_gradient",
    "Hydr_conductivity", "Aqu_thickness", "Long_dispersivity", "Trans_dispersivity", "Isotherm"
]
KNOWN_LABELS = ["Area", "Iso_distance", "Iso_width", "Cone"]

def detect_columns_from_csv(csv_file):
    """Detect feature and label columns from CSV header."""
    df = pd.read_csv(csv_file, nrows=0)
    columns = list(df.columns)
    features = [col for col in columns if col in KNOWN_FEATURES]
    labels = [col for col in columns if col in KNOWN_LABELS]
    return features, labels

# Define the callback function outside of main
def LogCallback(study, trial):
    """Logs the completion status of every trial."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        logging.info(f"Trial {trial.number} finished (Value: {trial.value:.6f}).")
    elif trial.state == optuna.trial.TrialState.PRUNED:
        logging.info(f"Trial {trial.number} pruned.")
    elif trial.state == optuna.trial.TrialState.FAIL:
        logging.error(f"Trial {trial.number} failed!")

def main():
    # Suppress TensorFlow logging to clean up output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    set_seed()
    
    # Log format adjusted to include SLURM_PROCID and stream to stdout/stderr
    worker_id = os.environ.get('SLURM_PROCID', 'MAIN')
    logging.basicConfig(
        level=logging.INFO, 
        format=f'%(asctime)s - Worker-{worker_id} - %(levelname)s - %(message)s',
        # CRITICAL: Ensures logs go to the Slurm output file
        stream=sys.stdout 
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device} (Visible device: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')})")

    args = parse_args()

    csv_file = args.csv_file
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    # Detect feature and label columns from CSV
    FEATURE_COLUMN_NAMES, AVAILABLE_LABEL_NAMES = detect_columns_from_csv(csv_file)
    logging.info(f"Detected features: {FEATURE_COLUMN_NAMES}")
    logging.info(f"Detected labels: {AVAILABLE_LABEL_NAMES}")

    # Set target labels
    target_labels = AVAILABLE_LABEL_NAMES if args.target == 'all' else [args.target]
    if args.target != 'all' and args.target not in AVAILABLE_LABEL_NAMES:
        raise ValueError(f"Target label '{args.target}' not found in CSV. Available: {AVAILABLE_LABEL_NAMES}")
    logging.info(f"Target labels for this run: {target_labels}")

    validate_target_labels(target_labels)

    # Use SLURM_PROCID for unique run directory tagging
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    label_str = "_".join(target_labels)
    run_tag = None
    if args.run_tag:
        safe_tag = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in args.run_tag)
        run_tag = safe_tag.strip()
    tag_suffix = f"_{run_tag}" if run_tag else ""
    root_folder = f"runs/run_{run_timestamp}_{label_str}{tag_suffix}/"
    os.makedirs(root_folder, exist_ok=True)

    # --- JOURNAL STORAGE SETUP (Shared between all 56 workers) ---
    journal_dir = args.storage_path or f"{root_folder}/optuna_journal_storage/"
    journal_file_path = os.path.join(journal_dir, "journal.log")
    
    optuna_study_name = args.study_name or f"nn_study_{label_str}"
    
    # Ensure the directory exists before the backend tries to open the file
    os.makedirs(os.path.dirname(journal_file_path) if os.path.basename(journal_file_path) != "journal.log" else journal_dir, exist_ok=True)
    
    backend = JournalFileBackend(journal_file_path) 
    optuna_storage = JournalStorage(backend)

    optuna_n_trials = max(1, int(args.optuna_trials))
    optuna_n_jobs = max(1, int(args.optuna_workers))
    
    logging.info(
        "Optuna config -> study: %s | storage (file): %s | trials (this worker): %d | n_jobs (internal): %d",
        optuna_study_name,
        journal_file_path, 
        optuna_n_trials,
        optuna_n_jobs
    )

    objective_config = {
        "batch_size": args.objective_batch_size,
        "nr_hidden_layers": args.objective_hidden_layers,
        "activation_name": args.objective_activation,
        "loss_name": args.objective_loss,
        "batch_size_choices": sorted({32, 64, 128, 256, args.objective_batch_size}),
        "nr_hidden_layers_range": (
            max(1, args.objective_hidden_layers - 2),
            max(args.objective_hidden_layers, args.objective_hidden_layers + 2)
        ),
        "activation_choices": ['ReLU', 'GELU', 'SiLU', 'LeakyReLU', 'Tanh'],
        "loss_choices": ['L1', 'SmoothL1', 'MSE'],
        "max_epochs": args.optuna_max_epochs,
        "patience": args.optuna_patience,
        "tune_core_hparams": True,
    }

    with power_monitor_session(args, root_folder):
        logging.info(f"Starting Optuna study: {optuna_study_name}...")
        
        # Data loading code remains identical 
        try:
            X_train, X_test, X_scaler_unused, y_train, y_test, y_scaler_unused = load_data(
                csv_file=csv_file,
                feature_cols=FEATURE_COLUMN_NAMES,
                label_cols=target_labels,
                plots=False,
                rf=root_folder,
                feature_scaler_type='none',
                label_scaler_type='none'
            )
            X_train_main, X_val, y_train_main, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            optuna_data = {
                "X_train": X_train_main,
                "y_train": y_train_main,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
            }
        except Exception as e:
            logging.error(f"FATAL EXCEPTION DURING DATA LOADING: {type(e).__name__}: {e}")
            sys.exit(1)

        
        # CRITICAL DEBUG PRINT: Confirm everything before this point ran.
        logging.info("Data loaded successfully. Attempting to create/load study.")

        try:
            study = optuna.create_study(
                study_name=optuna_study_name,
                direction="minimize",
                storage=optuna_storage,
                load_if_exists=True,
                # CRITICAL: We removed the aggressive pruner here to force longer runs (2 minutes per trial)
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), # Pruner restored per user request for consistency
            )
            objective_fn = build_objective(
                data=optuna_data,
                base_log_dir=os.path.join(root_folder, "optuna_tensorboard_logs"),
                objective_config=objective_config
            )
            
            logging.info(f"Study loaded/created. Starting optimization for {optuna_n_trials} trials.")
            
            study.optimize(
                objective_fn, 
                n_trials=optuna_n_trials, 
                n_jobs=optuna_n_jobs,
                callbacks=[LogCallback] # Add the per-trial logging
            )

            logging.info(f"Optuna study complete. Best trial (Global): {study.best_trial.number}")
            logging.info(f"Best Loss (Global): {study.best_value}")
            logging.info(f"Best Params (Global): {study.best_params}")
            
        except Exception as e:
            # CRITICAL ERROR TRACE: Capture any exception that caused the silent exit.
            logging.error(f"FATAL EXCEPTION DURING STUDY OPTIMIZATION: {type(e).__name__}: {e}")
            sys.exit(1)


    logging.info("--- Optuna tuning run complete ---")
    logging.info(f"To view Optuna results: optuna-dashboard journal://{journal_file_path}")
    logging.info(f"To view TensorBoard logs: tensorboard --logdir {root_folder}")


if __name__ == "__main__":
    main()