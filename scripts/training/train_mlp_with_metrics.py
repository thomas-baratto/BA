#!/usr/bin/env python3
"""Train MLP models from Optuna best trials and generate summary CSV with all metrics.

Usage:
  python scripts/training/train_mlp_with_metrics.py --dataset isotherm --output-dir runs/mlp_metrics
  python scripts/training/train_mlp_with_metrics.py --dataset cone --output-dir runs/mlp_metrics
  python scripts/training/train_mlp_with_metrics.py --dataset all --output-dir runs/mlp_metrics
"""

import argparse
import json
import logging
import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from datetime import datetime
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from pathlib import Path

from core.runtime import ensure_dir, get_device, setup_logging
from core.trainer import main_train
from core.training_utils import normalize_best_params, get_loss_criterion
from config.datasets import DATASET_CONFIGS
from core.artifacts import ArtifactManifest

MAX_EPOCHS = 10000
PATIENCE = 250


def load_optuna_best_params(config: dict) -> tuple:
    """Load best parameters from config file or fallback to Optuna journal.

    Returns (best_params, trial_number, best_value) where best_params may
    include the ``use_area_root`` key extracted from the study metadata.
    """
    best_params_file = config.get("best_params_file")
    if best_params_file and os.path.exists(best_params_file):
        with open(best_params_file, "r") as f:
            data = json.load(f)
        logging.info(f"Loaded best parameters from {best_params_file}")
        logging.info(f"Trial number: {data.get('trial_number', 'N/A')}")
        logging.info(f"Best value: {data.get('best_value', 'N/A')}")
        params = data["best_params"]
        # Carry use_area_root from the top-level JSON into the params dict
        if "use_area_root" in data:
            params.setdefault("use_area_root", data["use_area_root"])
        return params, data.get("trial_number", -1), data.get("best_value", -1.0)

    journal_path = config.get("journal_path")
    study_name = config.get("study_name")
    
    if not journal_path or not os.path.exists(journal_path):
        raise FileNotFoundError(f"Journal file or config not found: {journal_path}")

    storage = JournalStorage(JournalFileBackend(journal_path))
    study = optuna.load_study(study_name=study_name, storage=storage)

    logging.info(f"Loaded study: {study_name}")
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best value: {study.best_value:.6f}")

    use_area_root = study.user_attrs.get("use_area_root", False)
    use_log = study.user_attrs.get("use_log", True)
    params = study.best_params
    params["use_area_root"] = use_area_root
    params["use_log"] = use_log
    
    if best_params_file:
        os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
        with open(best_params_file, "w") as f:
            json.dump({
                "best_params": study.best_params,
                "trial_number": study.best_trial.number,
                "best_value": study.best_value,
                "use_area_root": use_area_root,
                "use_log": use_log,
            }, f, indent=2)
        logging.info(f"Saved best parameters to {best_params_file}")

    return study.best_params, study.best_trial.number, study.best_value


def train_and_evaluate(
    dataset_name: str,
    output_dir: str,
    device: torch.device
) -> dict:
    """Train MLP from Optuna best trial and return metrics row for summary CSV."""
    config = DATASET_CONFIGS[dataset_name]

    logging.info(f"\n{'='*60}")
    logging.info(f"Training MLP for dataset: {dataset_name}")
    logging.info(f"{'='*60}")

    # Load best parameters from config or Optuna
    best_params, trial_number, best_value = load_optuna_best_params(config)
    logging.info(f"Best parameters: {best_params}")

    # Normalize parameters
    final_config = normalize_best_params(best_params, max_epochs=MAX_EPOCHS, patience=PATIENCE)

    # Create dataset-specific output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target_str = "_".join(config["labels"])
    run_dir = os.path.join(output_dir, f"{dataset_name}_MLP_{timestamp}_{target_str}")
    ensure_dir(run_dir)
    logging.info(f"Output directory: {run_dir}")

    # Train model
    logging.info("Training model...")
    train_start = time.time()

    model, X_scaler, y_scaler = main_train(
        config=final_config,
        rf=run_dir,
        csv_file=config["csv_file"],
        feature_cols=config["features"],
        label_cols=config["labels"],
        device=device
    )

    train_time = time.time() - train_start
    logging.info(f"Training completed in {train_time:.2f}s")

    # Save model
    model_path = os.path.join(run_dir, "best_model.pt")
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(model_to_save.state_dict(), model_path)
    logging.info(f"Model saved to: {model_path}")

    # Save scalers for inference on new data
    scalers_path = os.path.join(run_dir, "scalers.pkl")
    with open(scalers_path, "wb") as f:
        pickle.dump({
            "feature_scaler": X_scaler,
            "label_scaler": y_scaler,
        }, f)
    logging.info(f"Scalers saved to: {scalers_path}")

    # Save model config with all info needed for inference
    model_config = {
        "input_size": len(config["features"]),
        "output_size": len(config["labels"]),
        "nr_hidden_layers": final_config.get("nr_hidden_layers"),
        "nr_neurons": final_config.get("nr_neurons"),
        "activation_name": final_config.get("activation_name"),
        "dropout_rate": final_config.get("dropout_rate", 0.0),
        "use_batchnorm": final_config.get("use_batchnorm", False),
        "feature_scaler_type": final_config.get("feature_scaler_type"),
        "label_scaler_type": final_config.get("label_scaler_type"),
        "use_log": final_config.get("use_log", True),
        "use_area_root": final_config.get("use_area_root", False),
        "feature_names": config["features"],
        "label_names": config["labels"],
    }
    config_path = os.path.join(run_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    logging.info(f"Model config saved to: {config_path}")

    # Load metrics computed and saved by main_train (physical units, correct inverse transforms)
    metrics_json_path = os.path.join(run_dir, "stats", "metrics_summary.json")
    with open(metrics_json_path) as f:
        saved_metrics = json.load(f)
    metrics = saved_metrics.get("test", {}).get("overall", {})

    # Save artifact manifest
    manifest = ArtifactManifest(
        model_type="mlp",
        dataset=dataset_name,
        targets=config["labels"],
        features=config["features"],
        training={
            "epochs": final_config.get("num_epochs", MAX_EPOCHS),
            "batch_size": final_config.get("batch_size", 32),
            "learning_rate": final_config.get("learning_rate", 0.001),
            "optimizer": final_config.get("optimizer", "Adam"),
            "loss_criterion": final_config.get("loss_criterion", "SmoothL1"),
            "train_time_seconds": train_time,
        },
        performance={
            "test_rmse": metrics.get("rmse"),
            "test_mae": metrics.get("mae"),
            "test_mse": metrics.get("mse"),
            "test_r2": metrics.get("r2"),
            "test_nrmse": metrics.get("nrmse"),
            "test_kge": metrics.get("kge"),
        },
    )
    manifest.save(Path(run_dir))
    logging.info(f"Artifact manifest saved to: {os.path.join(run_dir, 'artifact_manifest.json')}")

    # Build summary row matching the expected CSV format
    return {
        "Model": "MLP",
        "Dataset": dataset_name,
        "RMSE": metrics.get("rmse"),
        "MAE": metrics.get("mae"),
        "MSE": metrics.get("mse"),
        "MAPE": metrics.get("mape"),
        "R2": metrics.get("r2"),
        "nRMSE": metrics.get("nrmse"),
        "KGE": metrics.get("kge"),
        "Time(s)": train_time,
        "Folder": os.path.basename(run_dir),
        "N_Hidden": final_config.get("nr_neurons"),
        "Layers": final_config.get("nr_hidden_layers"),
        "N_Ensemble": None,  # Not applicable for MLP
        "Blocks": None,  # Not applicable for MLP
        "Activation": final_config.get("activation_name"),
    }


def generate_summary_csv(results: list, output_dir: str) -> str:
    """Generate summary_table.csv matching the expected format."""
    df = pd.DataFrame(results)

    # Define column order to match run_sweep_random_1004/summary_table.csv
    column_order = [
        "Model", "Dataset", "RMSE", "MAE", "MSE", "MAPE", "R2",
        "nRMSE", "KGE", "Time(s)", "Folder",
        "N_Hidden", "Layers", "N_Ensemble", "Blocks", "Activation"
    ]

    # Reorder columns (keep only those that exist)
    existing_cols = [c for c in column_order if c in df.columns]
    df = df[existing_cols]

    # Sort by Dataset, then R2 descending
    df = df.sort_values(by=["Dataset", "R2"], ascending=[True, False])

    output_path = os.path.join(output_dir, "summary_table.csv")
    df.to_csv(output_path, index=False)
    logging.info(f"\nSummary table saved to: {output_path}")

    # Print preview
    print("\nSummary Preview:")
    print(df.drop(columns=["Folder"], errors="ignore").to_string(index=False))

    return output_path


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Train MLP models from Optuna best trials with full metrics"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["isotherm", "cone", "all"],
        default="all",
        help="Which dataset to train on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/mlp_from_optuna",
        help="Base output directory for results"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available"
    )
    args = parser.parse_args()

    device = get_device(args.no_cuda)
    logging.info(f"Using device: {device}")

    ensure_dir(args.output_dir)

    # Determine which datasets to train
    datasets = ["isotherm", "cone"] if args.dataset == "all" else [args.dataset]

    results = []
    for dataset in datasets:
        try:
            result = train_and_evaluate(dataset, args.output_dir, device)
            results.append(result)
        except Exception as e:
            logging.error(f"Failed to train {dataset}: {e}")
            raise

    # Generate summary CSV
    if results:
        generate_summary_csv(results, args.output_dir)

    logging.info("\n" + "=" * 60)
    logging.info("All training complete!")
    logging.info(f"Results saved to: {args.output_dir}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
