"""Smoke tests for real pretrained models in the models/ directory.

These tests are dynamically generated based on what is found in the models/ folder.
If no models are present, these tests are skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from core.model_wrapper import TrainedModel
from core.inference import make_predictions


def discover_pretrained_models():
    """Find all directories in 'models/' that contain model artifacts."""
    models_root = Path("models")
    if not models_root.is_dir():
        return []
    
    found = []
    # Search for model_config.json, ignoring hidden dirs
    for path in models_root.rglob("model_config.json"):
        if any(part.startswith('.') for part in path.parts):
            continue
        found.append(path.parent)
    return sorted(found)


# Dynamically discover models
PRETRAINED_PATHS = discover_pretrained_models()


@pytest.mark.skipif(not PRETRAINED_PATHS, reason="No pretrained models found in models/ directory")
@pytest.mark.parametrize("model_path", PRETRAINED_PATHS, ids=lambda p: p.name)
def test_pretrained_model_inference(model_path):
    """Verify that a real pretrained model can load and make a prediction."""
    # 1. Load the model
    tm = TrainedModel(str(model_path))
    config = tm.get_config()
    
    # 2. Load sample data matching the dataset type
    dataset_type = config.get("dataset")
    if dataset_type is None:
        dataset_type = "isotherm" if config.get("input_size") == 9 else "cone"
    
    sample_file = Path(f"data/sample_{dataset_type}.csv")
    if not sample_file.exists():
        # Fallback to random if sample missing, but with safer ranges
        X = np.random.uniform(0.01, 0.1, (3, config.get("input_size")))
    else:
        import pandas as pd
        df = pd.read_csv(sample_file)
        # Filter only the features the model expects
        features = config.get("feature_names")
        X = df[features].values
    
    # 3. Run inference using the same logic as predict.py
    try:
        # Respect config flags for log transform and area root
        use_log = config.get("use_log", True)
        use_area_root = config.get("use_area_root", False)
        
        preds = make_predictions(
            tm, 
            X, 
            tm.feature_scaler, 
            tm.label_scaler,
            apply_feature_log=use_log,
            apply_label_expm1=use_log,
            use_area_root=use_area_root
        )
        
        # 4. Assertions
        assert preds.shape == (len(X), config.get("output_size")), "Output shape mismatch"
        assert np.isfinite(preds).all(), "Predictions contain non-finite values (NaN/Inf)"
        
    except Exception as e:
        pytest.fail(f"Model at {model_path} failed inference: {e}")


@pytest.mark.skipif(not PRETRAINED_PATHS, reason="No pretrained models found in models/ directory")
@pytest.mark.parametrize("model_path", PRETRAINED_PATHS, ids=lambda p: p.name)
def test_pretrained_model_accuracy(model_path):
    """Verify accuracy against original training data subsets if available."""
    tm = TrainedModel(str(model_path))
    config = tm.get_config()
    dataset_type = config.get("dataset")
    if dataset_type is None:
        dataset_type = "isotherm" if config.get("input_size") == 9 else "cone"

    # Map dataset type to official CSV file
    data_map = {
        "isotherm": "data/Clean_Results_Isotherm.csv",
        "cone": "data/Depression_cones.csv"
    }
    data_file = Path(data_map.get(dataset_type, ""))
    
    if not data_file.exists():
        pytest.skip(f"Official data file {data_file} not found for accuracy check")

    import pandas as pd
    # Load 20 random rows for a robust check
    df = pd.read_csv(data_file).sample(n=20, random_state=42)
    
    features = config.get("feature_names")
    labels = config.get("label_names")
    
    X = df[features].values
    y_true = df[labels].values
    
    # Run inference
    use_log = config.get("use_log", True)
    use_area_root = config.get("use_area_root", False)
    
    y_pred = make_predictions(
        tm, X, tm.feature_scaler, tm.label_scaler,
        apply_feature_log=use_log, apply_label_expm1=use_log,
        use_area_root=use_area_root, label_names=labels
    )
    
    # Calculate R2 or relative error
    # We expect high correlation for pretrained models on their own data
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    # For some difficult randomized models, R2 might be lower on 20 samples, 
    # but it should definitely be positive!
    assert r2 > 0.0, f"Model at {model_path} has zero or negative R2 on official data. Scaling might be wrong."
    
    # Check that individual label ranges match
    for i, label in enumerate(labels):
        true_mean = np.mean(y_true[:, i])
        pred_mean = np.mean(y_pred[:, i])
        # Allow 50% mean deviation (very loose, just to catch unit mismatches)
        if true_mean != 0:
            rel_diff = abs(true_mean - pred_mean) / (abs(true_mean) + 1e-6)
            assert rel_diff < 0.5, f"Unit mismatch suspected for {label} in {model_path}. Mean diff: {rel_diff:.2%}"


def test_no_corrupted_configs():
    """Ensure every found model has a valid config and scaler."""
    for model_path in PRETRAINED_PATHS:
        assert (model_path / "model_config.json").is_file()
        assert (model_path / "scalers.pkl").is_file()
        # Weights can be .pt (MLP) or .pkl (Randomized)
        weights_exist = (model_path / "best_model.pt").is_file() or (model_path / "model.pkl").is_file()
        assert weights_exist, f"No weight file (.pt or .pkl) found in {model_path}"
