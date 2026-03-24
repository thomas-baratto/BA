#!/usr/bin/env python3
"""Unified training script for Random Models (ELM, RVFL variants) on Isotherm and Cone datasets.

Usage:
  python train_random_models.py --model dRVFL --dataset cone --targets Cone
  python train_random_models.py --model NF-RVFL --dataset isotherm --targets Area Iso_width
"""

import torch
import numpy as np
import argparse
import os
import json
import time
import gzip
import pickle
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

from core.config_types import RandomTrainingConfig
from core.data_loader import load_data
from core.runtime import ensure_dir, get_device, setup_logging
from core.metrics import compute_regression_metrics
from core.training_utils import to_physical_units
from config.datasets import DATASET_CONFIGS
from core.artifacts import ArtifactManifest
from pathlib import Path

# Import Models
# Import Models
from core.random.ELM import ELM
from core.random.dRVFL import dRVFL
from core.random.edRVFL import edRVFL
from core.random.edRVFL_SC import edRVFL_SC
from core.random.esc_edRVFL import esc_edRVFL
from core.random.SResdRVFL import SResdRVFL


def _aggregate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Keep aggregate metrics aligned with per-target metric implementation.
    return compute_regression_metrics(y_true, y_pred)


def _inverse_predictions(cfg: RandomTrainingConfig, y_scaler, y_values: np.ndarray, label_cols: list[str] | None = None) -> np.ndarray:
    return to_physical_units(
        y_values, y_scaler,
        use_log=cfg.use_log,
        use_area_root=cfg.use_area_root,
        label_cols=label_cols,
    )


def _build_output_dir(cfg: RandomTrainingConfig, label_cols):
    if cfg.output_dir is not None:
        return ensure_dir(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target_str = "_".join(label_cols)
    folder_name = f"{cfg.dataset}_{cfg.model}_{timestamp}_{target_str}"
    return ensure_dir(os.path.join(cfg.base_dir, folder_name))

def parse_args():
    parser = argparse.ArgumentParser(description="Train Random Models on tabular datasets")
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['ELM', 'dRVFL', 'edRVFL', 'edRVFL-SC', 'esc-edRVFL', 'SResdRVFL'],
                        help='Which model to train')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['isotherm', 'cone'],
                        help='Which dataset to use')

    # Data arguments
    parser.add_argument('--targets', type=str, nargs='+', default=None,
                        help='Target label(s). Defaults to all targets for the dataset.')
    parser.add_argument('--feature-scaler', type=str, default='robust',
                        choices=['minmax', 'standard', 'robust', 'quantile'])
    parser.add_argument('--label-scaler', type=str, default='robust',
                        choices=['minmax', 'standard', 'robust', 'quantile'])
    parser.add_argument('--use-log', action='store_true',
                        help='Enable log1p transformation of the data')
    parser.add_argument('--use-area-root', action='store_true',
                        help='Apply square root transformation to Area before log1p/scaling')

    # General Model arguments
    parser.add_argument('--n-hidden', type=int, default=100, help='Number of hidden neurons/rules')
    parser.add_argument('--activation', type=str, default='ReLU', help='Activation function')
    parser.add_argument('--alpha', type=float, default=1e-3, help='Ridge regularization parameter')
    parser.add_argument('--gamma', type=float, default=1.0, help='RBF shape parameter')
    
    # Specific arguments (Deep/Ensemble/Residual)
    parser.add_argument('--n-layers', type=int, default=3, help='For deep models: Number of hidden layers')
    parser.add_argument('--n-ensemble', type=int, default=10, help='For ensemble models: Number of sub-models')
    
    # edRVFL-SC specifics
    parser.add_argument('--sc-mode', type=str, default='dense', choices=['dense', 'random'], help='Skip connection mode')
    parser.add_argument('--rsc-prob', type=float, default=0.5, help='Probability for random skip connections')
    
    # esc-edRVFL specifics
    parser.add_argument('--n-folds', type=int, default=5, help='K-Folds for esc-edRVFL ensemble base')
    
    # SResdRVFL specifics
    parser.add_argument('--n-blocks', type=int, default=5, help='Number of residual blocks')
    parser.add_argument('--direct-link', action='store_true', help='Use asymmetric direct links in SResdRVFL')
    


    # System parameters
    parser.add_argument('--base-dir', type=str, default='runs', help='Base directory for all runs')
    parser.add_argument('--output-dir', type=str, default=None, help='Specific output directory (overrides base-dir + auto-naming)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--n-seeds', type=int, default=1, help='Number of seeds to run (for mean±std reporting). Seeds used: random_state, random_state+1, ...')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no-save-model', action='store_true', help='Do not save the model and test predictions to disk')

    return parser.parse_args()


def init_model(args, device):
    base_kwargs = {
        'alpha': args.alpha,
        'gamma': args.gamma,
        'device': device,
        'random_state': args.random_state
    }
    
    if args.model == 'ELM':
        return ELM(n_hidden=args.n_hidden, activation=args.activation, **base_kwargs)
        
    elif args.model == 'dRVFL':
        return dRVFL(n_layers=args.n_layers, n_hidden=args.n_hidden, activation=args.activation, **base_kwargs)
        
    elif args.model == 'edRVFL':
        return edRVFL(n_ensemble=args.n_ensemble, n_layers=args.n_layers, n_hidden=args.n_hidden, 
                      activation=args.activation, **base_kwargs)
                      
    elif args.model == 'edRVFL-SC':
        return edRVFL_SC(n_ensemble=args.n_ensemble, n_layers=args.n_layers, n_hidden=args.n_hidden,
                         activation=args.activation, mode=args.sc_mode, rsc_prob=args.rsc_prob, **base_kwargs)
                         
    elif args.model == 'esc-edRVFL':
        return esc_edRVFL(n_folds=args.n_folds, n_ensemble=args.n_ensemble, n_layers=args.n_layers,
                          n_hidden=args.n_hidden, activation=args.activation, rsc_prob=args.rsc_prob,
                          **base_kwargs)
                          
    elif args.model == 'SResdRVFL':
        return SResdRVFL(n_blocks=args.n_blocks, n_layers_per_block=args.n_layers, n_hidden=args.n_hidden,
                         activation=args.activation, direct_link=args.direct_link, **base_kwargs)
                         

        
    else:
        raise ValueError(f"Unknown model: {args.model}")



def run_single_seed(args, device, seed, output_dir, label_cols, X_train_full, X_test,
                    y_train_full, y_test, y_scaler):
    """Run a single training/evaluation pass with the given seed. Returns test metrics dict."""
    cfg = RandomTrainingConfig.from_namespace(args).with_seed(seed)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed
    )

    # --- Initialize & Train ---
    model = init_model(cfg, device)
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    logging.info("Seed %d: trained in %.2fs", seed, train_time)

    # --- Evaluate ---
    results = {}

    for split_name, X_split, y_split in [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test)
    ]:
        y_pred_scaled = model.predict(X_split)

        y_true_original = _inverse_predictions(cfg, y_scaler, y_split)

        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)

        y_pred_original = _inverse_predictions(cfg, y_scaler, y_pred_scaled)

        aggregate_metrics = _aggregate_metrics(y_true_original, y_pred_original)

        results[split_name] = {'aggregate': aggregate_metrics}

        if split_name == 'test':
            logging.info(
                "Seed %d test -> RMSE: %.2f, MAE: %.2f, R2: %.4f",
                seed,
                aggregate_metrics['rmse'],
                aggregate_metrics['mae'],
                aggregate_metrics['r2'],
            )

        for i, target_name in enumerate(label_cols):
            y_true_target = y_true_original[:, i]
            y_pred_target = y_pred_original[:, i]
            target_metrics = compute_regression_metrics(y_true_target, y_pred_target)
            display_name = "sqrt(Area)" if (cfg.use_area_root and target_name == "Area") else target_name
            results[split_name][display_name] = {k: float(v) for k, v in target_metrics.items()}

    # --- Save per-seed artifacts ---
    results_dict = {
        'config': cfg.to_dict(),
        'metrics': results,
        'train_time_seconds': train_time
    }

    results_file = os.path.join(output_dir, f"results_{cfg.model}_{cfg.dataset}.json")
    tmp_file = results_file + ".tmp"
    with open(tmp_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    os.replace(tmp_file, results_file)

    if not cfg.no_save_model:
        y_test_pred = model.predict(X_test)
        if y_test_pred.ndim == 1:
            y_test_pred = y_test_pred.reshape(-1, 1)

        y_test_pred_original = _inverse_predictions(cfg, y_scaler, y_test_pred)
        y_test_original = _inverse_predictions(cfg, y_scaler, y_test)

        np.savez_compressed(
            os.path.join(output_dir, 'test_predictions.npz'),
            y_true=y_test_original,
            y_pred=y_test_pred_original,
            label_names=label_cols
        )

        model_file = os.path.join(output_dir, 'model.pkl')
        with gzip.open(model_file, 'wb') as f:
            pickle.dump(model, f)

    return results, train_time


def main():
    setup_logging()
    args = parse_args()
    cfg = RandomTrainingConfig.from_namespace(args)

    # --- Setup Device ---
    device = get_device(cfg.no_cuda)
    logging.info("Using device: %s", device)

    # --- Setup Dataset ---
    dataset_cfg = DATASET_CONFIGS[cfg.dataset]
    csv_file = dataset_cfg['csv_file']
    feature_cols = dataset_cfg['features']
    label_cols = list(cfg.targets) if cfg.targets else dataset_cfg['labels']

    logging.info("Loading dataset: %s", cfg.dataset)
    logging.info("File: %s", csv_file)
    logging.info("Features (%d): %s", len(feature_cols), feature_cols)
    logging.info("Targets: %s", label_cols)

    # --- Setup Base Output Dir ---
    base_output_dir = _build_output_dir(cfg, label_cols)
    logging.info("Output directory: %s", base_output_dir)

    # --- Load Data (applies log1p and scaler intrinsically) ---
    X_train_full, X_test, X_scaler, y_train_full, y_test, y_scaler = load_data(
        csv_file=csv_file,
        feature_cols=feature_cols,
        label_cols=label_cols,
        plots=False,
        rf=base_output_dir,
        feature_scaler_type=cfg.feature_scaler,
        label_scaler_type=cfg.label_scaler,
        use_log=cfg.use_log,
        use_area_root=cfg.use_area_root
    )

    logging.info("Data shapes:")
    logging.info("  X_train_full: %s, y_train_full: %s", X_train_full.shape, y_train_full.shape)
    logging.info("  X_test:       %s, y_test:       %s", X_test.shape, y_test.shape)

    # --- Save scalers and model config for inference ---
    scalers_path = os.path.join(base_output_dir, "scalers.pkl")
    with gzip.open(scalers_path, "wb") as f:
        pickle.dump({
            "feature_scaler": X_scaler,
            "label_scaler": y_scaler,
        }, f)
    logging.info("Scalers saved to: %s", scalers_path)

    model_config = {
        "model_type": "random",
        "model_name": cfg.model,
        "dataset": cfg.dataset,
        "input_size": len(feature_cols),
        "output_size": len(label_cols),
        "feature_scaler_type": cfg.feature_scaler,
        "label_scaler_type": cfg.label_scaler,
        "use_log": cfg.use_log,
        "use_area_root": cfg.use_area_root,
        "feature_names": feature_cols,
        "label_names": label_cols,
    }
    config_path = os.path.join(base_output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    logging.info("Model config saved to: %s", config_path)

    # Save artifact manifest
    manifest = ArtifactManifest(
        model_type="random",
        dataset=cfg.dataset,
        targets=label_cols,
        features=feature_cols,
        training={
            "model": cfg.model,
            "n_hidden": cfg.n_hidden,
            "n_layers": cfg.n_layers,
            "alpha": cfg.alpha,
            "gamma": cfg.gamma,
        },
        performance={},  # Will be populated after training
    )
    manifest.save(Path(base_output_dir))
    logging.info("Artifact manifest saved to: %s", os.path.join(base_output_dir, "artifact_manifest.json"))

    # --- Run over seeds ---
    seeds = [cfg.random_state + i for i in range(cfg.n_seeds)]
    all_test_metrics = []

    for idx, seed in enumerate(seeds, start=1):
        if cfg.n_seeds > 1:
            seed_output_dir = os.path.join(base_output_dir, f"seed_{seed}")
            ensure_dir(seed_output_dir)
            logging.info("--- Seed %d (%d/%d) ---", seed, idx, len(seeds))
        else:
            seed_output_dir = base_output_dir

        results, train_time = run_single_seed(
            args, device, seed, seed_output_dir, label_cols,
            X_train_full, X_test, y_train_full, y_test, y_scaler
        )
        test_metrics = results['test']['aggregate']
        test_metrics['train_time'] = train_time
        test_metrics['seed'] = seed
        all_test_metrics.append(test_metrics)

    # --- Aggregate multi-seed results ---
    if cfg.n_seeds > 1:
        logging.info("%s", '=' * 70)
        logging.info("MULTI-SEED SUMMARY (%d seeds)", cfg.n_seeds)
        logging.info("%s", '=' * 70)

        summary = {}
        for key in ['rmse', 'mae', 'r2', 'mape', 'mse', 'nrmse', 'kge']:
            values = [m[key] for m in all_test_metrics if not np.isnan(m.get(key, float('nan')))]
            if values:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': values
                }
                logging.info(
                    "  %5s: %.4f +/- %.4f  (min=%.4f, max=%.4f)",
                    key.upper(),
                    summary[key]['mean'],
                    summary[key]['std'],
                    summary[key]['min'],
                    summary[key]['max'],
                )

        train_times = [m['train_time'] for m in all_test_metrics]
        summary['train_time'] = {
            'mean': float(np.mean(train_times)),
            'std': float(np.std(train_times))
        }
        logging.info("  TIME:  %.2f +/- %.2fs", summary['train_time']['mean'], summary['train_time']['std'])

        # Save aggregated summary
        multi_seed_result = {
            'config': cfg.to_dict(),
            'seeds': seeds,
            'n_seeds': cfg.n_seeds,
            'per_seed_test_metrics': all_test_metrics,
            'aggregated': summary
        }
        summary_file = os.path.join(base_output_dir, 'multi_seed_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(multi_seed_result, f, indent=2)
        logging.info("Multi-seed summary saved to: %s", summary_file)
    else:
        logging.info("Run artifacts saved in: %s", base_output_dir)


if __name__ == "__main__":
    main()
