#!/usr/bin/env python3
"""Optuna tuning script for discovering the best RBF gamma value for random models."""

import os
import argparse
import optuna
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from core.data_loader import load_data

from core.random.ELM import ELM
from core.random.dRVFL import dRVFL
from core.random.edRVFL import edRVFL
from core.random.edRVFL_SC import edRVFL_SC
from core.random.esc_edRVFL import esc_edRVFL
from core.random.SResdRVFL import SResdRVFL

MODEL_CLASSES = {
    'ELM': ELM,
    'dRVFL': dRVFL,
    'edRVFL': edRVFL,
    'edRVFL-SC': edRVFL_SC,
    'esc-edRVFL': esc_edRVFL,
    'SResdRVFL': SResdRVFL
}

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

def objective(trial, model_name, X_train, y_train, X_val, y_val, device):
    gamma = trial.suggest_float('gamma', 1e-4, 1e2, log=True)
    
    # Base kwargs for the model to train quickly
    kwargs = {
        'n_hidden': 200, 
        'device': device,
        'random_state': 42
    }
    
    kwargs['activation'] = 'rbf'
        
    if model_name in ['dRVFL', 'edRVFL', 'edRVFL-SC', 'esc-edRVFL']:
        kwargs['n_layers'] = 3
    if model_name in ['edRVFL', 'edRVFL-SC', 'esc-edRVFL']:
        kwargs['n_ensemble'] = 5
    if model_name == 'SResdRVFL':
        kwargs['n_blocks'] = 3
        kwargs['n_layers_per_block'] = 3
    
    kwargs['gamma'] = gamma
    
    model_class = MODEL_CLASSES[model_name]
    model = model_class(**kwargs)
    
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
            
        # Using RMSE as the primary optimization metric
        mse = ((y_val - y_pred) ** 2).mean()
        rmse = float(np.sqrt(mse))
        
        # We can also log R2 for tracking
        r2 = float(r2_score(y_val, y_pred))
        trial.set_user_attr('r2', r2)
        
        return rmse
    except Exception as e:
        print(f"Trial failed for gamma={gamma}: {e}")
        raise optuna.exceptions.TrialPruned()

def main():
    parser = argparse.ArgumentParser(description="Tune RBF Gamma for Random Models")
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_CLASSES.keys()))
    parser.add_argument('--dataset', type=str, default='cone', choices=['cone', 'isotherm'])
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")

    # --- Load Data (applies scaler intrinsically) ---
    dataset_cfg = DATASET_CONFIGS[args.dataset]
    csv_file = dataset_cfg['file']
    feature_cols = dataset_cfg['features']
    label_cols = dataset_cfg['default_targets']
    
    print(f"Loading data for {args.dataset}...")
    use_area_root = (args.dataset == 'isotherm')
    
    X_train_full, X_test, X_scaler, y_train_full, y_test, y_scaler = load_data(
        csv_file=csv_file,
        feature_cols=feature_cols,
        label_cols=label_cols,
        plots=False,
        rf='runs',
        feature_scaler_type="minmax",
        label_scaler_type="robust",
        use_log=False,
        use_area_root=use_area_root
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    # Needs to be numpy for some models, or torch tensors. 
    # train_random_models.py passes output of load_data directly, which are numpy arrays.
    
    # Make sure output dirs exist
    os.makedirs("runs", exist_ok=True)
    
    study_name = f"tune_gamma_{args.model}_{args.dataset}"
    storage_name = f"sqlite:///runs/optuna_gamma.db"
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name,
        direction="minimize",
        load_if_exists=True
    )
    
    print(f"\nOptimization target: {storage_name}")
    print(f"Starting tuning for {args.model} on {args.dataset} (Trials: {args.n_trials})...")
    
    try:
        study.optimize(
            lambda trial: objective(trial, args.model, X_train, y_train, X_val, y_val, device),
            n_trials=args.n_trials,
            catch=(Exception,)
        )
    except KeyboardInterrupt:
        print("\nTuning interrupted by user.")
        
    print("\n================================")
    print(f"Best trial for {args.model} ({args.dataset}):")
    if len(study.trials) > 0 and study.best_trial:
        print(f"  Value (RMSE in scaled space): {study.best_trial.value:.4f}")
        print(f"  Tracking R2: {study.best_trial.user_attrs.get('r2', 'N/A')}")
        print(f"  Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("  No trials finished successfully.")

if __name__ == "__main__":
    main()
