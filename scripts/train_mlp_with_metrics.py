#!/usr/bin/env python3
"""Train MLP models from Optuna best trials and generate summary CSV with all metrics.

Usage:
  python scripts/train_mlp_with_metrics.py --dataset isotherm --output-dir runs/mlp_metrics
  python scripts/train_mlp_with_metrics.py --dataset cone --output-dir runs/mlp_metrics
  python scripts/train_mlp_with_metrics.py --dataset all --output-dir runs/mlp_metrics
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pathlib import Path

from core.data_loader import CSVDataset, load_data
from core.runtime import ensure_dir, get_device, setup_logging
from core.trainer import main_train, evaluate
from core.utils import compute_regression_metrics
from config.datasets import DATASET_CONFIGS
from core.artifacts import ArtifactManifest

MAX_EPOCHS = 10000
PATIENCE = 250


def normalize_best_params(best_params: dict) -> dict:
    """Normalize parameter naming variants from Optuna trials."""
    config = best_params.copy()

    # Handle scaler naming variants
    if "feature_scaler" in config and "feature_scaler_type" not in config:
        config["feature_scaler_type"] = config["feature_scaler"]
    if "label_scaler" in config and "label_scaler_type" not in config:
        config["label_scaler_type"] = config["label_scaler"]

    # Set defaults
    config.setdefault("plots", True)
    config.setdefault("num_epochs", MAX_EPOCHS)
    config["patience"] = PATIENCE
    config.setdefault("use_log", True)

    return config


def get_loss_criterion(loss_name: str) -> nn.Module:
    """Get loss criterion by name."""
    if loss_name == "L1":
        return nn.L1Loss()
    if loss_name == "SmoothL1":
        return nn.SmoothL1Loss()
    return nn.MSELoss()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute aggregate regression metrics."""
    return compute_regression_metrics(y_true, y_pred)


def load_optuna_best_params(journal_path: str, study_name: str) -> tuple:
    """Load best parameters from an Optuna journal file."""
    if not os.path.exists(journal_path):
        raise FileNotFoundError(f"Journal file not found: {journal_path}")

    storage = JournalStorage(JournalFileBackend(journal_path))
    study = optuna.load_study(study_name=study_name, storage=storage)

    logging.info(f"Loaded study: {study_name}")
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best value: {study.best_value:.6f}")

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

    # Load best parameters from Optuna
    best_params, trial_number, best_value = load_optuna_best_params(
        config["journal_path"],
        config["study_name"]
    )
    logging.info(f"Best parameters: {best_params}")

    # Normalize parameters
    final_config = normalize_best_params(best_params)

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

    # Load raw test data and apply training scalers for evaluation
    df = pd.read_csv(config["csv_file"])
    X_all = df[config["features"]].values
    y_all = df[config["labels"]].values

    # Apply log transform if enabled (same as during training)
    use_log = final_config.get("use_log", True)
    if use_log:
        y_all = np.log1p(y_all)

    # Split with same random state as training
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    # Apply the scalers from training
    X_test = X_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test)

    logging.info(f"Test set: {X_test.shape[0]} samples")

    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_dataset = CSVDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=final_config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    loss_name = final_config.get("loss_criterion", "SmoothL1")
    criterion = get_loss_criterion(loss_name)

    test_loss, predictions, true_values = evaluate(model, test_loader, criterion, device)

    # Inverse transform to physical units
    predictions_original = y_scaler.inverse_transform(predictions)
    true_values_original = y_scaler.inverse_transform(true_values)

    # Apply inverse log transform if used
    use_log = final_config.get("use_log", True)
    if use_log:
        predictions_original = np.expm1(predictions_original)
        true_values_original = np.expm1(true_values_original)

    # Compute aggregate metrics
    metrics = compute_metrics(true_values_original, predictions_original)

    logging.info(f"Test metrics:")
    logging.info(f"  RMSE: {metrics['rmse']:.6f}")
    logging.info(f"  MAE:  {metrics['mae']:.6f}")
    logging.info(f"  R2:   {metrics['r2']:.6f}")
    logging.info(f"  nRMSE: {metrics['nrmse']:.6f}")
    logging.info(f"  KGE:  {metrics['kge']:.6f}")

    # Save results JSON
    results_dict = {
        "config": {
            "model": "MLP",
            "dataset": dataset_name,
            "n_hidden": final_config.get("nr_neurons"),
            "n_layers": final_config.get("nr_hidden_layers"),
            "activation": final_config.get("activation_name"),
            **final_config
        },
        "study_info": {
            "study_name": config["study_name"],
            "best_trial": trial_number,
            "best_value": best_value
        },
        "metrics": {
            "test": {"aggregate": metrics}
        },
        "train_time_seconds": train_time,
        "device": str(device)
    }

    results_file = os.path.join(run_dir, f"results_MLP_{dataset_name}.json")
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    logging.info(f"Results saved to: {results_file}")

    # Save test predictions for future recomputation
    np.savez(
        os.path.join(run_dir, "test_predictions.npz"),
        y_true=true_values_original,
        y_pred=predictions_original,
        label_names=config["labels"]
    )

    # Save model
    model_path = os.path.join(run_dir, "best_model.pt")
    torch.save(model.state_dict(), model_path)
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
        "feature_names": config["features"],
        "label_names": config["labels"],
    }
    config_path = os.path.join(run_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    logging.info(f"Model config saved to: {config_path}")

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
