#!/usr/bin/env python3
"""
Summarize results from a random training job folder into a CSV table.
"""
import os
import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from core.metrics import compute_regression_metrics


def _load_or_compute_test_metrics(subdir: Path, data: dict) -> dict:
    metrics = data.get('metrics', {}).get('test', {}).get('aggregate', {}) or {}

    # If advanced metrics are absent in JSON, try computing from saved predictions.
    required = {'nrmse', 'kge'}
    if required.issubset(metrics.keys()):
        return metrics

    pred_file = subdir / 'test_predictions.npz'
    if not pred_file.exists():
        return metrics

    try:
        arr = np.load(pred_file)
        y_true = arr['y_true']
        y_pred = arr['y_pred']
        computed = compute_regression_metrics(y_true, y_pred)

        # Keep original JSON values when present; fill missing keys from recomputation.
        merged = dict(computed)
        merged.update(metrics)
        return merged
    except Exception:
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Summarize training results into a table")
    parser.add_argument('--run-dir', type=str, required=True, help='Overarching run folder (e.g. runs/run_training_random_941)')
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    if not run_path.exists():
        print(f"Error: Directory {args.run_dir} not found.")
        return

    summary_data = []

    # Iterate through each subfolder in the run directory
    for subdir in run_path.iterdir():
        if not subdir.is_dir():
            continue
            
        # Look for the results JSON file (pattern: results_{model}_{dataset}.json)
        results_files = list(subdir.glob("results_*.json"))
        if not results_files:
            continue
            
        res_file = results_files[0]
        try:
            with open(res_file, 'r') as f:
                data = json.load(f)
                
            config = data.get('config', {})
            metrics = _load_or_compute_test_metrics(subdir, data)
            
            row = {
                'Model': config.get('model'),
                'Dataset': config.get('dataset'),
                'RMSE': metrics.get('rmse'),
                'MAE': metrics.get('mae'),
                'MSE': metrics.get('mse'),
                'MAPE': metrics.get('mape'),
                'R2': metrics.get('r2'),
                'nRMSE': metrics.get('nrmse'),
                'KGE': metrics.get('kge'),
                'Time(s)': data.get('train_time_seconds'),
                'Folder': subdir.name
            }
            
            # Add specific hyperparams (capitalized)
            if config.get('n_hidden'): row['N_Hidden'] = config.get('n_hidden')
            if config.get('n_layers'): row['Layers'] = config.get('n_layers')
            if config.get('n_ensemble'): row['N_Ensemble'] = config.get('n_ensemble')
            if config.get('n_blocks'): row['Blocks'] = config.get('n_blocks')
            if config.get('basis_type'): row['Basis'] = config.get('basis_type')
            if config.get('activation'): row['Activation'] = config.get('activation')
            
            # Add all other hyperparameters dynamically
            mapped_keys = {'model', 'dataset', 'n_hidden', 'n_layers', 'n_ensemble', 'n_blocks', 'basis_type', 'activation'}
            for k, v in config.items():
                if k not in mapped_keys:
                    row[k] = v
            
            summary_data.append(row)
        except Exception as e:
            print(f"Warning: Could not process {res_file}: {e}")

    if not summary_data:
        print("No results files found.")
        return

    df = pd.DataFrame(summary_data)
    
    # Sort by Dataset, then R2 (descending)
    df = df.sort_values(by=['Dataset', 'R2'], ascending=[True, False])
    
    output_file = run_path / "summary_table.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\nSummary table created at: {output_file}")
    print("\nSummary Preview:")
    print(df.drop(columns=['Folder']).to_string(index=False))

if __name__ == "__main__":
    main()
