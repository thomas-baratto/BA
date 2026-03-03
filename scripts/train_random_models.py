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
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from core.data_loader import load_data
from core.utils import compute_regression_metrics

# Import Models
from scripts.RandomNetwork.RandomModels.ELM import ELM
from scripts.RandomNetwork.RandomModels.dRVFL import dRVFL
from scripts.RandomNetwork.RandomModels.edRVFL import edRVFL
from scripts.RandomNetwork.RandomModels.edRVFL_SC import edRVFL_SC
from scripts.RandomNetwork.RandomModels.esc_edRVFL import esc_edRVFL
from scripts.RandomNetwork.RandomModels.SResdRVFL import SResdRVFL
from scripts.RandomNetwork.RandomModels.NF_RVFL import NF_RVFL
from scripts.RandomNetwork.RandomModels.FELM import FELM

DATASET_CONFIGS = {
    "isotherm": {
        "file": "./data/Clean_Results_Isotherm.csv",
        "features": [
            "Flow_well", "Temp_diff", "kW_well", "Hydr_gradient", "Hydr_conductivity",
            "Aqu_thickness", "Long_dispersivity", "Trans_dispersivity", "Isotherm"
        ],
        "default_targets": ["Area", "Iso_distance", "Iso_width"]
    },
    "cone": {
        "file": "./data/Depression_cones.csv",
        "features": [
            "Flow_well", "Hydr_gradient", "Hydr_conductivity", "Aqu_thickness"
        ],
        "default_targets": ["Cone"]
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train Random Models on tabular datasets")
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        choices=['ELM', 'dRVFL', 'edRVFL', 'edRVFL-SC', 'esc-edRVFL', 'SResdRVFL', 'NF-RVFL', 'FELM'],
                        help='Which model to train')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['isotherm', 'cone'],
                        help='Which dataset to use')

    # Data arguments
    parser.add_argument('--targets', type=str, nargs='+', default=None,
                        help='Target label(s). Defaults to all targets for the dataset.')
    parser.add_argument('--feature-scaler', type=str, default='robust',
                        choices=['minmax', 'standard', 'robust', 'quantile'])
    parser.add_argument('--label-scaler', type=str, default='minmax',
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
    
    # NF-RVFL specifics
    parser.add_argument('--nf-variation', type=str, default='K', choices=['R', 'K', 'C'], help='Fuzzy rule clustering type')
    
    # FELM specifics
    parser.add_argument('--basis-type', type=str, default='polynomial', choices=['polynomial', 'fourier', 'rbf'])

    # System parameters
    parser.add_argument('--base-dir', type=str, default='runs', help='Base directory for all runs')
    parser.add_argument('--output-dir', type=str, default=None, help='Specific output directory (overrides base-dir + auto-naming)')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--n-seeds', type=int, default=1, help='Number of seeds to run (for mean±std reporting). Seeds used: random_state, random_state+1, ...')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')

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
                         
    elif args.model == 'NF-RVFL':
        return NF_RVFL(n_hidden=args.n_hidden, n_rules=args.n_hidden, variation=args.nf_variation,
                       activation=args.activation, **base_kwargs)
                       
    elif args.model == 'FELM':
        return FELM(n_basis=args.n_hidden, basis_type=args.basis_type, **base_kwargs)
        
    else:
        raise ValueError(f"Unknown model: {args.model}")



def run_single_seed(args, device, seed, output_dir, label_cols, X_train_full, X_test,
                    y_train_full, y_test, y_scaler):
    """Run a single training/evaluation pass with the given seed. Returns test metrics dict."""
    # Override seed for this run
    args_copy = argparse.Namespace(**vars(args))
    args_copy.random_state = seed

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed
    )

    # --- Initialize & Train ---
    model = init_model(args_copy, device)
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"  Seed {seed}: trained in {train_time:.2f}s")

    # --- Evaluate ---
    results = {}

    for split_name, X_split, y_split in [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test)
    ]:
        y_pred_scaled = model.predict(X_split)

        if args.use_log:
            y_true_original = np.expm1(y_scaler.inverse_transform(y_split))
        else:
            y_true_original = y_scaler.inverse_transform(y_split)

        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)

        if args.use_log:
            y_pred_original = np.expm1(y_scaler.inverse_transform(y_pred_scaled))
        else:
            y_pred_original = y_scaler.inverse_transform(y_pred_scaled)

        if args.use_area_root and "Area" in label_cols:
            area_idx = label_cols.index("Area")
            y_true_original[:, area_idx] = y_true_original[:, area_idx] ** 2
            y_pred_original[:, area_idx] = y_pred_original[:, area_idx] ** 2

        y_true_flat = y_true_original.flatten()
        y_pred_flat = y_pred_original.flatten()

        aggregate_metrics = {
            'mse': float(np.mean((y_pred_flat - y_true_flat) ** 2)),
            'rmse': float(np.sqrt(np.mean((y_pred_flat - y_true_flat) ** 2))),
            'mae': float(np.mean(np.abs(y_pred_flat - y_true_flat))),
            'r2': float(r2_score(y_true_flat, y_pred_flat))
        }

        mask = y_true_flat != 0
        if np.any(mask):
            aggregate_metrics['mape'] = float(np.mean(np.abs((y_pred_flat[mask] - y_true_flat[mask]) / y_true_flat[mask])))
        else:
            aggregate_metrics['mape'] = float('nan')

        results[split_name] = {'aggregate': aggregate_metrics}

        if split_name == 'test':
            print(f"    Test RMSE: {aggregate_metrics['rmse']:.2f}, MAE: {aggregate_metrics['mae']:.2f}, R²: {aggregate_metrics['r2']:.4f}")

        for i, target_name in enumerate(label_cols):
            y_true_target = y_true_original[:, i]
            y_pred_target = y_pred_original[:, i]
            target_metrics = compute_regression_metrics(y_true_target, y_pred_target)
            display_name = "Area (Squared from Root)" if (args.use_area_root and target_name == "Area") else target_name
            results[split_name][display_name] = {k: float(v) for k, v in target_metrics.items()}

    # --- Save per-seed artifacts ---
    results_dict = {
        'config': vars(args_copy),
        'metrics': results,
        'train_time_seconds': train_time
    }

    results_file = os.path.join(output_dir, f"results_{args.model}_{args.dataset}.json")
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    y_test_pred = model.predict(X_test)
    if y_test_pred.ndim == 1:
        y_test_pred = y_test_pred.reshape(-1, 1)

    if args.use_log:
        y_test_pred_original = np.expm1(y_scaler.inverse_transform(y_test_pred))
        y_test_original = np.expm1(y_scaler.inverse_transform(y_test))
    else:
        y_test_pred_original = y_scaler.inverse_transform(y_test_pred)
        y_test_original = y_scaler.inverse_transform(y_test)

    if args.use_area_root and "Area" in label_cols:
        area_idx = label_cols.index("Area")
        y_test_original[:, area_idx] = y_test_original[:, area_idx] ** 2
        y_test_pred_original[:, area_idx] = y_test_pred_original[:, area_idx] ** 2

    np.savez(
        os.path.join(output_dir, 'test_predictions.npz'),
        y_true=y_test_original,
        y_pred=y_test_pred_original,
        label_names=label_cols
    )

    model_file = os.path.join(output_dir, 'model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    return results, train_time


def main():
    args = parse_args()

    # --- Setup Device ---
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Setup Dataset ---
    dataset_cfg = DATASET_CONFIGS[args.dataset]
    csv_file = dataset_cfg['file']
    feature_cols = dataset_cfg['features']
    label_cols = args.targets if args.targets else dataset_cfg['default_targets']

    print(f"\nLoading dataset: {args.dataset}")
    print(f"File: {csv_file}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Targets: {label_cols}")

    # --- Setup Base Output Dir ---
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target_str = "_".join(label_cols)
        folder_name = f"{args.dataset}_{args.model}_{timestamp}_{target_str}"
        base_output_dir = os.path.join(args.base_dir, folder_name)
    else:
        base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Output directory: {base_output_dir}")

    # --- Load Data (applies log1p and scaler intrinsically) ---
    X_train_full, X_test, X_scaler, y_train_full, y_test, y_scaler = load_data(
        csv_file=csv_file,
        feature_cols=feature_cols,
        label_cols=label_cols,
        plots=False,
        rf=base_output_dir,
        feature_scaler_type=args.feature_scaler,
        label_scaler_type=args.label_scaler,
        use_log=args.use_log,
        use_area_root=args.use_area_root
    )

    print(f"\nData shapes:")
    print(f"  X_train_full: {X_train_full.shape}, y_train_full: {y_train_full.shape}")
    print(f"  X_test:       {X_test.shape}, y_test:       {y_test.shape}")

    # --- Run over seeds ---
    seeds = [args.random_state + i for i in range(args.n_seeds)]
    all_test_metrics = []

    for seed in seeds:
        if args.n_seeds > 1:
            seed_output_dir = os.path.join(base_output_dir, f"seed_{seed}")
            os.makedirs(seed_output_dir, exist_ok=True)
            print(f"\n--- Seed {seed} ({seeds.index(seed)+1}/{len(seeds)}) ---")
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
    if args.n_seeds > 1:
        print(f"\n{'='*70}")
        print(f"MULTI-SEED SUMMARY ({args.n_seeds} seeds)")
        print(f"{'='*70}")

        summary = {}
        for key in ['rmse', 'mae', 'r2', 'mape', 'mse']:
            values = [m[key] for m in all_test_metrics if not np.isnan(m.get(key, float('nan')))]
            if values:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': values
                }
                print(f"  {key.upper():>5s}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}  (min={summary[key]['min']:.4f}, max={summary[key]['max']:.4f})")

        train_times = [m['train_time'] for m in all_test_metrics]
        summary['train_time'] = {
            'mean': float(np.mean(train_times)),
            'std': float(np.std(train_times))
        }
        print(f"  TIME:  {summary['train_time']['mean']:.2f} ± {summary['train_time']['std']:.2f}s")

        # Save aggregated summary
        multi_seed_result = {
            'config': vars(args),
            'seeds': seeds,
            'n_seeds': args.n_seeds,
            'per_seed_test_metrics': all_test_metrics,
            'aggregated': summary
        }
        summary_file = os.path.join(base_output_dir, 'multi_seed_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(multi_seed_result, f, indent=2)
        print(f"\nMulti-seed summary saved to: {summary_file}")
    else:
        print(f"\nRun artifacts saved in: {base_output_dir}")


if __name__ == "__main__":
    main()
