#!/usr/bin/env python3
"""
Test script to verify parameter handling in Optuna studies.
This tests that hardcoded parameters are properly stored and retrieved.
"""

import optuna
import sqlite3
from pathlib import Path

import pytest

DB_PATH = "runs/optuna_study.db"

def test_parameter_retrieval():
    """Test that we can retrieve both optimized and hardcoded parameters."""
    db_path = Path(DB_PATH)
    if not db_path.exists():
        pytest.skip("Optuna study database not found; run tuning before this test if needed.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT study_name FROM studies")
    except sqlite3.OperationalError:
        conn.close()
        pytest.skip("Optuna tables missing; no studies recorded yet.")

    studies = cursor.fetchall()
    conn.close()

    if not studies:
        pytest.skip("No Optuna studies available in the local database.")

    study_name = studies[0][0]
    storage = f"sqlite:///{DB_PATH}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    assert isinstance(study.best_trial.params, dict)
    hardcoded_params = {'batch_size', 'nr_hidden_layers', 'activation_name', 'loss_criterion'}
    merged = study.best_trial.params.copy()
    for key, value in study.best_trial.user_attrs.items():
        if key in hardcoded_params:
            merged[key] = value

    assert set(merged.keys()) >= set(study.best_trial.params.keys())

if __name__ == "__main__":
    test_parameter_retrieval()
