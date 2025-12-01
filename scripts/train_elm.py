#!/usr/bin/env python3
"""Train Extreme Learning Machine on the isotherm dataset.

This script trains an ELM for regression on Area, Iso_distance, and Iso_width targets.
It uses the same data loading pipeline as the neural network training scripts.
"""

import torch
import numpy as np
import argparse
import os
import json
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from core.elm import ExtremeLearningMachine
from core.data_loader import load_data
from core.utils import create_regression_plots, create_qq_plots, compute_regression_metrics

FEATURE_COLUMN_NAMES = [
    "Flow_well", "Temp_diff", "kW_well", "Hydr_gradient", "Hydr_conductivity",
    "Aqu_thickness", "Long_dispersivity", "Trans_dispersivity", "Isotherm"
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Extreme Learning Machine on isotherm data")
    parser.add_argument('--target', type=str, default='all', 
                        choices=['Area', 'Iso_distance', 'Iso_width', 'all'],
                        help='Target label(s) to train on (default: all)')
    parser.add_argument('--n-hidden', type=int, default=100,
                        help='Number of hidden neurons (default: 100)')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'GELU', 'ELU'],
                        help='Activation function (default: ReLU)')
    parser.add_argument('--alpha', type=float, default=1e-2,
                        help='Ridge regularization parameter (default: 1e-2)')
    parser.add_argument('--feature-scaler', type=str, default='standard',
                        choices=['minmax', 'standard', 'robust', 'quantile'],
                        help='Feature scaling method (default: standard)')
    parser.add_argument('--label-scaler', type=str, default='standard',
                        choices=['minmax', 'standard', 'robust', 'quantile'],
                        help='Label scaling method (default: standard)')
    parser.add_argument('--csv-file', type=str, default='./data/Clean_Results_Isotherm.csv',
                        help='Path to CSV data file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: runs/elm_TIMESTAMP)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    return parser.parse_args()


def train_elm_for_targets(args, label_names, output_dir):
    """Train ELM for given target labels."""
    
    # Setup device
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data for targets: {label_names}")
    print(f"  Feature scaler: {args.feature_scaler}")
    print(f"  Label scaler: {args.label_scaler}")
    
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
        X_train_full, y_train_full, test_size=0.2, random_state=args.random_state
    )
    
    print(f"Data shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Create ELM
    print(f"\nCreating ELM:")
    print(f"  Hidden neurons: {args.n_hidden}")
    print(f"  Activation: {args.activation}")
    print(f"  Alpha (regularization): {args.alpha}")
    
    elm = ExtremeLearningMachine(
        n_hidden=args.n_hidden,
        activation=args.activation,
        alpha=args.alpha,
        include_bias=True,
        device=device,
        random_state=args.random_state
    )
    
    # Train
    print("\nTraining ELM...")
    start_time = time.time()
    elm.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate on all splits
    results = {}
    
    for split_name, X_split, y_split in [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test)
    ]:
        y_pred_scaled = elm.predict(X_split)
        
        # Inverse transform to original scale for metrics (undo scaling, then undo log transform)
        y_true_original = np.expm1(y_scaler.inverse_transform(y_split))
        y_pred_original = np.expm1(y_scaler.inverse_transform(y_pred_scaled))
        
        # Compute aggregate metrics on flattened data (true overall metrics)
        y_true_flat = y_true_original.flatten()
        y_pred_flat = y_pred_original.flatten()
        
        aggregate_metrics = {
            'mse': float(np.mean((y_pred_flat - y_true_flat) ** 2)),
            'rmse': float(np.sqrt(np.mean((y_pred_flat - y_true_flat) ** 2))),
            'mae': float(np.mean(np.abs(y_pred_flat - y_true_flat))),
            'r2': float(r2_score(y_true_flat, y_pred_flat)),
            'max_error': float(np.max(np.abs(y_pred_flat - y_true_flat)))
        }
        # Compute MAPE safely (avoid division by zero)
        mask = y_true_flat != 0
        if np.any(mask):
            aggregate_metrics['mape'] = float(np.mean(np.abs((y_pred_flat[mask] - y_true_flat[mask]) / y_true_flat[mask])))
        else:
            aggregate_metrics['mape'] = float('nan')
        
        results[split_name] = {
            'aggregate': aggregate_metrics
        }
        
        print(f"\n{'='*70}")
        print(f"{split_name.upper()} METRICS (on original scale)")
        print(f"{'='*70}")
        print(f"\nAGGREGATE (all targets flattened):")
        print(f"  RMSE: {aggregate_metrics['rmse']:.2f}")
        print(f"  MAE:  {aggregate_metrics['mae']:.2f}")
        print(f"  R²:   {aggregate_metrics['r2']:.4f}")
        print(f"  MAPE: {aggregate_metrics['mape']:.4f} ({aggregate_metrics['mape']*100:.2f}%)")
        print(f"  Max Error: {aggregate_metrics['max_error']:.2f}")
        
        # Compute per-target metrics
        print(f"\nPER-TARGET METRICS:")
        for i, target_name in enumerate(label_names):
            y_true_target = y_true_original[:, i]
            y_pred_target = y_pred_original[:, i]
            
            target_metrics = compute_regression_metrics(y_true_target, y_pred_target)
            results[split_name][target_name] = {k: float(v) for k, v in target_metrics.items()}
            
            print(f"\n  {target_name}:")
            print(f"    RMSE: {target_metrics['rmse']:.2f}")
            print(f"    MAE:  {target_metrics['mae']:.2f}")
            print(f"    R²:   {target_metrics['r2']:.4f}")
            print(f"    MAPE: {target_metrics['mape']:.4f} ({target_metrics['mape']*100:.2f}%)")
            print(f"    Max Error: {target_metrics['max_error']:.2f}")
    
    # Generate plots for test set
    print("\nGenerating plots...")
    y_test_pred = elm.predict(X_test)
    
    # Ensure predictions are 2D for inverse_transform
    if y_test_pred.ndim == 1:
        y_test_pred = y_test_pred.reshape(-1, 1)
    
    # Inverse transform for plotting (undo scaling, then undo log transform)
    y_test_original = y_scaler.inverse_transform(y_test)
    y_test_pred_original = y_scaler.inverse_transform(y_test_pred)
    
    # Undo log1p transform (data_loader applies log1p before scaling)
    y_test_original = np.expm1(y_test_original)
    y_test_pred_original = np.expm1(y_test_pred_original)
    
    # Create regression plots (predicted vs actual)
    create_regression_plots(
        y_test_original, 
        y_test_pred_original, 
        label_names, 
        output_dir
    )
    
    # Create Q-Q plots
    create_qq_plots(
        y_test_original,
        y_test_pred_original,
        label_names,
        output_dir
    )
    
    # Save results
    results_dict = {
        'config': {
            'targets': label_names,
            'n_hidden': args.n_hidden,
            'activation': args.activation,
            'alpha': args.alpha,
            'feature_scaler': args.feature_scaler,
            'label_scaler': args.label_scaler,
            'random_state': args.random_state,
            'device': str(device),
            'train_time_seconds': train_time
        },
        'metrics': results,
        'data_shapes': {
            'n_features': X_train.shape[1],
            'n_outputs': y_train.shape[1],
            'train_samples': X_train.shape[0],
            'val_samples': X_val.shape[0],
            'test_samples': X_test.shape[0]
        }
    }
    
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Save predictions
    predictions_file = os.path.join(output_dir, 'test_predictions.npz')
    np.savez(
        predictions_file,
        y_true=y_test_original,
        y_pred=y_test_pred_original,
        label_names=label_names
    )
    print(f"Test predictions saved to: {predictions_file}")
    
    return results


def main():
    """Main entry point."""
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
        output_dir = f"runs/elm_{timestamp}_{label_str}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save command-line args
    args_file = os.path.join(output_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Train
    results = train_elm_for_targets(args, label_names, output_dir)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Results saved in: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
