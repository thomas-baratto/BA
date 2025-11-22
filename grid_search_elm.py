#!/usr/bin/env python3
"""
Grid search for ELM hyperparameters.
Fast hyperparameter optimization for Extreme Learning Machine.
"""

import torch
import numpy as np
import argparse
import os
import json
import time
import itertools
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from elm import ExtremeLearningMachine
from data_loader import load_data

# Feature columns
FEATURE_COLUMN_NAMES = [
    "Flow_well", "Temp_diff", "kW_well", "Hydr_gradient", "Hydr_conductivity",
    "Aqu_thickness", "Long_dispersivity", "Trans_dispersivity", "Isotherm"
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Grid search for ELM hyperparameters")
    parser.add_argument('--target', type=str, default='all', 
                        choices=['Area', 'Iso_distance', 'Iso_width', 'all'],
                        help='Target label(s) to train on (default: all)')
    parser.add_argument('--feature-scaler', type=str, default='standard',
                        choices=['minmax', 'standard', 'robust', 'quantile'],
                        help='Feature scaling method (default: standard)')
    parser.add_argument('--label-scaler', type=str, default='standard',
                        choices=['minmax', 'standard', 'robust', 'quantile'],
                        help='Label scaling method (default: standard)')
    parser.add_argument('--csv-file', type=str, default='./data/Clean_Results_Isotherm.csv',
                        help='Path to CSV data file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: runs/elm_grid_TIMESTAMP)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    return parser.parse_args()


def train_and_evaluate(n_hidden, activation, alpha, X_train, X_val, y_train, y_val, device, random_state):
    """Train ELM with given hyperparameters and return validation metrics."""
    
    elm = ExtremeLearningMachine(
        n_hidden=n_hidden,
        activation=activation,
        alpha=alpha,
        include_bias=True,
        device=device,
        random_state=random_state
    )
    
    # Train
    start_time = time.time()
    elm.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Evaluate on validation set
    y_val_pred = elm.predict(X_val)
    
    # Ensure 2D for metrics
    if y_val_pred.ndim == 1:
        y_val_pred = y_val_pred.reshape(-1, 1)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    return {
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'train_time': train_time
    }


def main():
    """Main grid search."""
    args = parse_args()
    
    # Determine target labels
    if args.target == 'all':
        label_names = ['Area', 'Iso_distance', 'Iso_width']
    else:
        label_names = [args.target]
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label_str = "_".join(label_names)
        output_dir = f"runs/elm_grid_{timestamp}_{label_str}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Setup device
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}\n")

    # Load data
    logging.info(f"Loading data for targets: {label_names}")
    logging.info(f"  Feature scaler: {args.feature_scaler}")
    logging.info(f"  Label scaler: {args.label_scaler}\n")
    
    X_train_full, X_test, X_scaler, y_train_full, y_test, y_scaler = load_data(
        csv_file=args.csv_file,
        feature_cols=FEATURE_COLUMN_NAMES,
        label_cols=label_names,
        plots=False,
        rf=output_dir,
        feature_scaler_type=args.feature_scaler,
        label_scaler_type=args.label_scaler
    )
    
    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    logging.info(f"Data shapes:")
    logging.info(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    logging.info(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    logging.info(f"  X_test: {X_test.shape}, y_test: {y_test.shape}\n")
    
    # Define hyperparameter grid
    param_grid = {
        'n_hidden': [500, 1000, 2000, 3000, 5000, 7000],
        'activation': ['ReLU', 'LeakyReLU', 'GELU', 'ELU'],
        'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    }
    
    # Generate all combinations
    param_combinations = list(itertools.product(
        param_grid['n_hidden'],
        param_grid['activation'],
        param_grid['alpha']
    ))
    
    total_combinations = len(param_combinations)
    logging.info(f"Grid search configuration:")
    logging.info(f"  n_hidden: {param_grid['n_hidden']}")
    logging.info(f"  activation: {param_grid['activation']}")
    logging.info(f"  alpha: {param_grid['alpha']}")
    logging.info(f"  Total combinations: {total_combinations}\n")
    
    # Run grid search
    logging.info("="*80)
    logging.info("Starting grid search...")
    logging.info("="*80 + "\n")
    
    results = []
    best_val_r2 = -float('inf')
    best_params = None
    
    total_start = time.time()
    
    for idx, (n_hidden, activation, alpha) in enumerate(param_combinations, 1):
        logging.info(f"[{idx}/{total_combinations}] Testing: n_hidden={n_hidden}, activation={activation}, alpha={alpha:.0e}")
        
        metrics = train_and_evaluate(
            n_hidden, activation, alpha,
            X_train, X_val, y_train, y_val,
            device, args.random_state
        )
        
        result = {
            'n_hidden': n_hidden,
            'activation': activation,
            'alpha': alpha,
            **metrics
        }
        results.append(result)
        
        logging.info(f"  -> Val R²: {metrics['val_r2']:.6f}, RMSE: {metrics['val_rmse']:.6f}, "
              f"MAE: {metrics['val_mae']:.6f}, Time: {metrics['train_time']:.3f}s")
        
        # Track best
        if metrics['val_r2'] > best_val_r2:
            best_val_r2 = metrics['val_r2']
            best_params = result.copy()
            logging.info(f"  *** NEW BEST! ***")
        
        logging.info("")
    
    total_time = time.time() - total_start
    
    logging.info("="*80)
    logging.info(f"Grid search complete! Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logging.info("="*80 + "\n")
    
    # Print best results
    logging.info("BEST HYPERPARAMETERS:")
    logging.info(f"  n_hidden: {best_params['n_hidden']}")
    logging.info(f"  activation: {best_params['activation']}")
    logging.info(f"  alpha: {best_params['alpha']:.0e}")
    logging.info(f"\nBEST VALIDATION METRICS:")
    logging.info(f"  R²:   {best_params['val_r2']:.6f}")
    logging.info(f"  RMSE: {best_params['val_rmse']:.6f}")
    logging.info(f"  MAE:  {best_params['val_mae']:.6f}")
    logging.info(f"  Training time: {best_params['train_time']:.3f}s\n")
    
    # Train final model with best params on full training set and evaluate on test
    logging.info("Training final model with best hyperparameters on full training set...")
    best_elm = ExtremeLearningMachine(
        n_hidden=best_params['n_hidden'],
        activation=best_params['activation'],
        alpha=best_params['alpha'],
        include_bias=True,
        device=device,
        random_state=args.random_state
    )
    
    final_start = time.time()
    best_elm.fit(X_train_full, y_train_full)
    final_train_time = time.time() - final_start
    
    # Evaluate on test set
    y_test_pred = best_elm.predict(X_test)
    if y_test_pred.ndim == 1:
        y_test_pred = y_test_pred.reshape(-1, 1)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    logging.info(f"\nFINAL TEST METRICS (trained on full train set):")
    logging.info(f"  R²:   {test_r2:.6f}")
    logging.info(f"  RMSE: {test_rmse:.6f}")
    logging.info(f"  MAE:  {test_mae:.6f}")
    logging.info(f"  Training time: {final_train_time:.3f}s\n")
    
    # Save all results
    summary = {
        'config': {
            'targets': label_names,
            'feature_scaler': args.feature_scaler,
            'label_scaler': args.label_scaler,
            'random_state': args.random_state,
            'device': str(device)
        },
        'grid': param_grid,
        'total_combinations': total_combinations,
        'total_time_seconds': total_time,
        'best_params': {
            'n_hidden': best_params['n_hidden'],
            'activation': best_params['activation'],
            'alpha': best_params['alpha']
        },
        'best_validation_metrics': {
            'r2': best_params['val_r2'],
            'rmse': best_params['val_rmse'],
            'mae': best_params['val_mae'],
            'train_time': best_params['train_time']
        },
        'final_test_metrics': {
            'r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae,
            'train_time': final_train_time
        },
        'all_results': results
    }
    
    # Save summary
    summary_file = os.path.join(output_dir, 'grid_search_results.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Results saved to: {summary_file}")
    
    # Save results as CSV for easy viewing
    import pandas as pd
    df = pd.DataFrame(results)
    df = df.sort_values('val_r2', ascending=False)
    csv_file = os.path.join(output_dir, 'grid_search_results.csv')
    df.to_csv(csv_file, index=False)
    logging.info(f"Results table saved to: {csv_file}")
    
    # Print top 10 configurations
    logging.info(f"\nTop 10 configurations by validation R²:")
    logging.info(df[['n_hidden', 'activation', 'alpha', 'val_r2', 'val_rmse', 'val_mae', 'train_time']].head(10).to_string(index=False))
    
    logging.info("\n" + "="*80)
    logging.info("Grid search complete!")
    logging.info(f"Results saved in: {output_dir}")
    logging.info("="*80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
