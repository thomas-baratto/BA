#!/usr/bin/env python3
"""Run inference with trained MLP models.

Usage:
  # Predict from CSV file
  python scripts/predict_mlp.py --model-dir data/good_runs/mlp_final/isotherm_MLP_*/  --input data/new_data.csv

  # Predict from command line values (isotherm example)
  python scripts/predict_mlp.py --model-dir data/good_runs/mlp_final/isotherm_MLP_*/ \
      --values 0.001 5.0 4.8 0.1 0.001 0.0001 50 10 1 1

  # Predict from command line values (cone example)
  python scripts/predict_mlp.py --model-dir data/good_runs/mlp_final/cone_MLP_*/ \
      --values 0.001 0.001 0.0001 50
"""

import argparse
import json
import pickle
import sys
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, ".")

from core.model import NeuralNetwork as MLP


def load_model(model_dir: str) -> tuple:
    """Load trained model, scalers, and config from a model directory."""
    import os

    # Load config
    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load scalers
    scalers_path = os.path.join(model_dir, "scalers.pkl")
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    # Build model
    model = MLP(
        input_size=config["input_size"],
        output_size=config["output_size"],
        nr_hidden_layers=config["nr_hidden_layers"],
        nr_neurons=config["nr_neurons"],
        activation_name=config["activation_name"],
        dropout_rate=config.get("dropout_rate", 0.0),
        use_batchnorm=config.get("use_batchnorm", False),
    )

    # Load weights
    model_path = os.path.join(model_dir, "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    return model, scalers, config


def predict(model, scalers, config, features: np.ndarray) -> np.ndarray:
    """Run prediction on feature array.

    Args:
        model: Loaded MLP model
        scalers: Dict with 'feature_scaler' and 'label_scaler'
        config: Model config dict
        features: Input features array of shape (n_samples, n_features)

    Returns:
        Predictions in original scale of shape (n_samples, n_outputs)
    """
    # Scale features
    X_scaled = scalers["feature_scaler"].transform(features)

    # Run inference
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_scaled = model(X_tensor).numpy()

    # Inverse transform predictions
    y_pred = scalers["label_scaler"].inverse_transform(y_scaled)

    # Undo log transform if used
    if config.get("use_log", True):
        y_pred = np.expm1(y_pred)

    return y_pred


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained MLP")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory containing best_model.pt, scalers.pkl, model_config.json"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to CSV file with input features"
    )
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=None,
        help="Feature values for single prediction (in order listed in model_config.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: print to stdout)"
    )
    args = parser.parse_args()

    if args.input is None and args.values is None:
        parser.error("Either --input or --values must be provided")

    # Load model
    print(f"Loading model from: {args.model_dir}")
    model, scalers, config = load_model(args.model_dir)

    print(f"Features ({len(config['feature_names'])}): {config['feature_names']}")
    print(f"Outputs ({len(config['label_names'])}): {config['label_names']}")

    # Prepare input
    if args.input:
        df = pd.read_csv(args.input)
        # Select only the required feature columns in correct order
        missing = set(config["feature_names"]) - set(df.columns)
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")
        features = df[config["feature_names"]].values
        print(f"Loaded {len(features)} samples from {args.input}")
    else:
        if len(args.values) != len(config["feature_names"]):
            raise ValueError(
                f"Expected {len(config['feature_names'])} values, got {len(args.values)}. "
                f"Features: {config['feature_names']}"
            )
        features = np.array([args.values])
        print(f"Input values: {dict(zip(config['feature_names'], args.values))}")

    # Run prediction
    predictions = predict(model, scalers, config, features)

    # Format output
    results = pd.DataFrame(predictions, columns=config["label_names"])

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")
    else:
        print("\nPredictions:")
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
