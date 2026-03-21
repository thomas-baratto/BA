#!/usr/bin/env python3
"""
Select best and most efficient random models from sweep summary_table.csv.
Outputs CSV with hyperparameters for retraining.
"""

import argparse
import pandas as pd
from pathlib import Path


def select_best_and_efficient(summary_csv, top_k=3, output_csv=None):
    """
    Select top-k models by RMSE (best accuracy) and efficiency (low error per time).
    
    Args:
        summary_csv: Path to summary_table.csv from sweep run
        top_k: Number of models to select per dataset per category (default: 3)
        output_csv: Output file path (default: same dir as input, named 'selected_for_retrain.csv')
    """
    # Read summary table
    df = pd.read_csv(summary_csv)
    print(f"Loaded {len(df)} models from {summary_csv}")
    
    # Validate required columns
    required_cols = {'Dataset', 'Model', 'RMSE', 'Time(s)', 'Folder', 'N_Hidden', 'Layers', 'N_Ensemble', 'Blocks', 'Activation'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Group by dataset
    datasets = df['Dataset'].unique()
    selected_rows = []
    
    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset].copy()
        print(f"\n{dataset}: {len(dataset_df)} models")
        
        # Sort by RMSE (best accuracy)
        best_by_rmse = dataset_df.nsmallest(top_k, 'RMSE')
        print(f"  Top {top_k} by RMSE: {best_by_rmse['Model'].tolist()}")
        best_by_rmse['selection_reason'] = 'best_rmse'
        selected_rows.append(best_by_rmse)
        
        # Sort by efficiency: minimize (RMSE / Time) ratio
        # Lower ratio = lower error per unit time = more efficient
        dataset_df['efficiency'] = dataset_df['RMSE'] / dataset_df['Time(s)']
        best_by_efficiency = dataset_df.nsmallest(top_k, 'efficiency')
        print(f"  Top {top_k} by efficiency (RMSE/Time): {best_by_efficiency['Model'].tolist()}")
        best_by_efficiency['selection_reason'] = 'best_efficiency'
        
        # Remove efficiency column before appending (was temporary)
        best_by_efficiency = best_by_efficiency.drop('efficiency', axis=1)
        
        # Only add efficiency-selected if not already in RMSE selection
        for idx, row in best_by_efficiency.iterrows():
            if not any((selected_rows[-1]['Model'] == row['Model']).any() 
                      if len(selected_rows[-1]) > 0 else False):
                selected_rows.append(best_by_efficiency.loc[[idx]])
    
    # Concatenate all selections
    selected = pd.concat(selected_rows, ignore_index=True)
    # Remove duplicates by Model+Dataset (prefer best_rmse over best_efficiency)
    selected = selected.drop_duplicates(subset=['Model', 'Dataset'], keep='first')
    
    print(f"\nTotal selected: {len(selected)} models")
    print(f"  {(selected['selection_reason'] == 'best_rmse').sum()} by RMSE")
    print(f"  {(selected['selection_reason'] == 'best_efficiency').sum()} by efficiency")
    
    # Prepare output: keep hyperparameter columns needed for retraining
    output_cols = ['Dataset', 'Model', 'RMSE', 'Time(s)', 'Folder', 'N_Hidden', 'Layers', 'N_Ensemble', 'Blocks', 'Activation', 'selection_reason']
    output_df = selected[output_cols].sort_values(['Dataset', 'RMSE']).reset_index(drop=True)
    
    # Determine output path
    if output_csv is None:
        output_csv = Path(summary_csv).parent / 'selected_for_retrain.csv'
    
    output_df.to_csv(output_csv, index=False)
    print(f"\nWrote {output_csv}")
    print("\nSelected models:")
    print(output_df.to_string())
    
    return output_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Select best and most efficient random models for retraining'
    )
    parser.add_argument(
        '--summary-csv',
        required=True,
        help='Path to summary_table.csv from sweep run'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of models to select per dataset per category (default: 3)'
    )
    parser.add_argument(
        '--output-csv',
        help='Output CSV path (default: same dir as input)'
    )
    
    args = parser.parse_args()
    select_best_and_efficient(args.summary_csv, top_k=args.top_k, output_csv=args.output_csv)
