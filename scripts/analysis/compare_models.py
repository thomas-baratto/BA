#!/usr/bin/env python3
"""
Compare optimized MLP models against the best Random Network Models.
Generates a consolidated table of the best results per dataset.
"""

import argparse
import pandas as pd
import json
import os
import glob

def print_separator(title):
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")

def process_mlp_results(mlp_results_file):
    if not mlp_results_file or not os.path.exists(mlp_results_file):
        print(f"Warning: MLP results file not found at {mlp_results_file}")
        return None
        
    with open(mlp_results_file, 'r') as f:
        data = json.load(f)
        
    # We want test metrics
    all_metrics = data.get('metrics', {})
    metrics = all_metrics.get('test', all_metrics)
    
    # Check if this is the Isotherm or Cone model based on the labels
    dataset = "unknown"
    if "Isotherm" in str(data) or "Area" in str(data):
        dataset = "isotherm"
    elif "Cone" in str(data):
        dataset = "cone"
        
    # Standardize metric extraction
    return {
        'Dataset': dataset,
        'Model': 'Optimized MLP',
        'R2': metrics.get('r2', 0),
        'RMSE': metrics.get('rmse', 0),
        'MAE': metrics.get('mae', 0),
        'nRMSE': metrics.get('nrmse'),
        'KGE': metrics.get('kge'),
        'Time(s)': data.get('train_time_seconds', 0),
        'Architecture': f"MLP {data.get('config', {}).get('nr_hidden_layers', 'N/A')}x{data.get('config', {}).get('nr_neurons', 'N/A')}"
    }

def collect_best_random_models(random_csv):
    if not random_csv or not os.path.exists(random_csv):
        print(f"Warning: Random summary CSV not found at {random_csv}")
        return pd.DataFrame()
        
    df = pd.read_csv(random_csv)
    
    # We want the best model for each dataset
    best_models = []
    
    for dataset in df['Dataset'].unique():
        # Filter for dataset
        df_ds = df[df['Dataset'] == dataset]
        
        # Sort by R2 descending, then RMSE ascending
        df_ds_sorted = df_ds.sort_values(by=['R2', 'RMSE'], ascending=[False, True])
        
        if len(df_ds_sorted) > 0:
            best_row = df_ds_sorted.iloc[0]
            
            # Construct architecture string
            arch = best_row['Model']
            if pd.notna(best_row.get('N_Hidden')): arch += f" H={int(best_row['N_Hidden'])}"
            if pd.notna(best_row.get('Layers')): arch += f" L={int(best_row['Layers'])}"
            if pd.notna(best_row.get('Blocks')): arch += f" B={int(best_row['Blocks'])}"
            if pd.notna(best_row.get('N_Ensemble')): arch += f" E={int(best_row['N_Ensemble'])}"
            if pd.notna(best_row.get('Activation')): arch += f" Act={best_row['Activation']}"
            
            best_models.append({
                'Dataset': dataset,
                'Model': f"Best Random ({best_row['Model']})",
                'R2': best_row.get('R2', 0),
                'RMSE': best_row.get('RMSE', 0),
                'MAE': best_row.get('MAE', 0),
                'nRMSE': best_row.get('nRMSE'),
                'KGE': best_row.get('KGE'),
                'Time(s)': best_row.get('Time(s)', 0),
                'Architecture': arch
            })
            
    return pd.DataFrame(best_models)

def main():
    parser = argparse.ArgumentParser(description="Compare final models")
    parser.add_argument('--random-summary', type=str, 
                        help='Path to the summary_table.csv from the final random sweep')
    parser.add_argument('--mlp-isotherm', type=str,
                        help='Path to results.json for the Isotherm MLP')
    parser.add_argument('--mlp-cone', type=str,
                        help='Path to results.json for the Cone MLP')
    parser.add_argument('--output-csv', type=str, default='final_comparison.csv',
                        help='Output CSV file name')
    
    args = parser.parse_args()
    
    # If not provided, try to find the latest sweep
    if not args.random_summary:
        sweep_dirs = sorted(glob.glob("runs/run_sweep_random_*"), reverse=True)
        if sweep_dirs:
            potential_csv = os.path.join(sweep_dirs[0], "summary_table.csv")
            if os.path.exists(potential_csv):
                args.random_summary = potential_csv
                print(f"Auto-detected random summary: {args.random_summary}")
    
    results = []
    
    # Process MLPs
    if args.mlp_isotherm:
        res = process_mlp_results(args.mlp_isotherm)
        if res:
            res['Dataset'] = 'isotherm' # Force standard name
            results.append(res)
            
    if args.mlp_cone:
        res = process_mlp_results(args.mlp_cone)
        if res:
            res['Dataset'] = 'cone' # Force standard name
            results.append(res)
            
    # Process Random Models
    if args.random_summary:
        random_df = collect_best_random_models(args.random_summary)
        if not random_df.empty:
            for _, row in random_df.iterrows():
                results.append(row.to_dict())
                
    if not results:
        print("No valid results found to compare.")
        return
        
    # Create final dataframe
    final_df = pd.DataFrame(results)
    
    # Sort first by Dataset, then by R2
    final_df = final_df.sort_values(by=['Dataset', 'R2'], ascending=[True, False])
    
    # Display results
    for dataset in final_df['Dataset'].unique():
        print_separator(f"Results for dataset: {dataset.upper()}")
        ds_df = final_df[final_df['Dataset'] == dataset].drop('Dataset', axis=1)
        
        # Format for display
        display_df = ds_df.copy()
        for col in ['R2']: display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        for col in ['RMSE', 'MAE']: display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if x < 100 else f"{x:.1f}")
        if 'Time(s)' in display_df: display_df['Time(s)'] = display_df['Time(s)'].apply(lambda x: f"{x:.1f}")
        
        print(display_df.to_string(index=False))
        
    print_separator("Writing to CSV")
    final_df.to_csv(args.output_csv, index=False)
    print(f"Comparison saved to {args.output_csv}")
    
    # Generate LaTeX
    print("\nFor LaTeX version, run:")
    print(f"python scripts/analysis/csv_to_latex.py {args.output_csv} --caption \"Final Model Comparison\"")

if __name__ == "__main__":
    main()
