#!/usr/bin/env python3
"""
Test script to verify parameter handling in Optuna studies.
This tests that hardcoded parameters are properly stored and retrieved.
"""

import optuna
import sqlite3
import json

DB_PATH = "runs/optuna_study.db"

def test_parameter_retrieval():
    """Test that we can retrieve both optimized and hardcoded parameters."""
    
    # List all studies
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT study_name FROM studies")
    studies = cursor.fetchall()
    conn.close()
    
    print(f"Found {len(studies)} studies in database:")
    for study_name, in studies:
        print(f"  - {study_name}")
    
    if not studies:
        print("\n⚠️  No studies found in database. Run an optimization first.")
        return
    
    # Load the first study
    study_name = studies[0][0]
    print(f"\n📊 Loading study: {study_name}")
    
    storage = f"sqlite:///{DB_PATH}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    print(f"   Study has {len(study.trials)} trials")
    print(f"   Best trial: #{study.best_trial.number}")
    print(f"   Best value: {study.best_trial.value:.6f}")
    
    # Get best trial parameters
    print("\n✅ Optimized parameters (from study.best_trial.params):")
    for key, value in study.best_trial.params.items():
        print(f"   {key}: {value}")
    
    # Get user attributes (hardcoded parameters)
    print("\n🔧 Hardcoded parameters (from study.best_trial.user_attrs):")
    hardcoded_params = ['batch_size', 'nr_hidden_layers', 'activation_name', 'loss_criterion']
    found_hardcoded = False
    for key, value in study.best_trial.user_attrs.items():
        if key in hardcoded_params:
            print(f"   {key}: {value}")
            found_hardcoded = True
    
    if not found_hardcoded:
        print("   ⚠️  No hardcoded parameters found in user_attrs!")
        print("   This study may have been created before the parameter handling update.")
        print("   Run a new optimization to test the updated code.")
    
    # Demonstrate loading all parameters together
    print("\n📦 Complete configuration (merged):")
    best_params = study.best_trial.params.copy()
    for key, value in study.best_trial.user_attrs.items():
        if key in hardcoded_params:
            best_params[key] = value
    
    for key, value in sorted(best_params.items()):
        print(f"   {key}: {value}")
    
    print("\n✨ Parameter retrieval test complete!")

if __name__ == "__main__":
    test_parameter_retrieval()
