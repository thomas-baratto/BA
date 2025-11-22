import datetime
import logging
import os

import optuna
import torch
from sklearn.model_selection import train_test_split

from data_loader import load_data
from optuna_config import (
    FEATURE_COLUMN_NAMES,
    parse_args,
    set_seed,
    validate_target_labels,
)
from optuna_objective import build_objective
from power_utils import power_monitor_session


def main():
    set_seed()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    args = parse_args()

    all_labels = ["Area", "Iso_distance", "Iso_width"]
    target_labels = all_labels if args.target == 'all' else [args.target]
    logging.info(f"Target labels for this run: {target_labels}")

    csv_file = args.csv_file
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    validate_target_labels(target_labels)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    label_str = "_".join(target_labels)
    run_tag = None
    if args.run_tag:
        safe_tag = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in args.run_tag)
        run_tag = safe_tag.strip()
    tag_suffix = f"_{run_tag}" if run_tag else ""
    root_folder = f"runs/run_{run_timestamp}_{label_str}{tag_suffix}/"
    os.makedirs(root_folder, exist_ok=True)

    optuna_db_path = args.storage_url or f"sqlite:///runs/optuna_{label_str}.db"
    optuna_study_name = args.study_name or f"nn_study_{label_str}"
    optuna_n_trials = max(1, int(args.optuna_trials))
    optuna_n_jobs = max(1, int(args.optuna_workers))
    logging.info(
        "Optuna config -> study: %s | storage: %s | trials: %d | n_jobs: %d",
        optuna_study_name,
        optuna_db_path,
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
        # Load data without final scaling so each Optuna trial can choose scalers
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

        study = optuna.create_study(
            study_name=optuna_study_name,
            direction="minimize",
            storage=optuna_db_path,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        objective_fn = build_objective(
            data=optuna_data,
            base_log_dir=os.path.join(root_folder, "optuna_tensorboard_logs"),
            objective_config=objective_config
        )
        study.optimize(objective_fn, n_trials=optuna_n_trials, n_jobs=optuna_n_jobs)

        logging.info(f"Optuna study complete. Best trial: {study.best_trial.number}")
        logging.info(f"Best Loss: {study.best_value}")
        logging.info(f"Best Params: {study.best_params}")
        logging.info("Run artifacts stored in %s", root_folder)
        logging.info("Launch train_final_model.py once tuning converges to materialize the final weights.")

    logging.info("--- Optuna tuning run complete ---")
    logging.info(f"To view Optuna results: optuna-dashboard {optuna_db_path}")
    logging.info(f"To view TensorBoard logs: tensorboard --logdir {root_folder}")


if __name__ == "__main__":
    main()