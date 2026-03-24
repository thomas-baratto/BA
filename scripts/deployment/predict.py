#!/usr/bin/env python3
"""Unified prediction CLI for thermal plume models.

Predicts isotherm or cone values from input CSV using trained MLP or random models.

Usage:
  # Predict isotherm using MLP
  python scripts/deployment/predict.py --input data/new_samples.csv --dataset isotherm --model mlp

  # Predict cone using random model (ELM)
  python scripts/deployment/predict.py --input data/new_samples.csv --dataset cone --model random

  # Specify output location
  python scripts/deployment/predict.py --input data.csv --dataset isotherm --model mlp --output results/

Examples:
  # Isotherm prediction - expects columns:
  #   Flow_well, Temp_diff, kW_well, Hydr_gradient,
  #   Hydr_conductivity, Aqu_thickness, Long_dispersivity, Trans_dispersivity, Isotherm
  # Outputs: Area, Iso_distance, Iso_width

  # Cone prediction - expects columns:
  #   Flow_well, Hydr_gradient, Hydr_conductivity, Aqu_thickness
  # Outputs: Cone
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow running as both a script (PYTHONPATH=.) and installed package (pip install .)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.model import NeuralNetwork
from config.datasets import DATASET_CONFIGS, DEFAULT_MODEL_DIRS

# Feature columns for each dataset (used for validation) - imported from centralized config
DATASET_FEATURES = {dataset: cfg["features"] for dataset, cfg in DATASET_CONFIGS.items()}
from core.model_wrapper import TrainedModel
from core.inference import make_predictions

def find_model_dir(base_dir: str) -> str:
    """Find model directory containing model_config.json.

    Checks base_dir itself first, then looks for subdirectories
    (e.g. timestamped run directories).
    """
    import glob

    # Check if the base directory itself contains model artifacts
    if os.path.isfile(os.path.join(base_dir, "model_config.json")):
        return base_dir

    # Otherwise look for subdirectories with model artifacts
    pattern = os.path.join(base_dir, "*", "model_config.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No model found in: {base_dir}")
    # Return parent dir of most recent match
    return os.path.dirname(matches[-1])


def predict_with_model(trained_model: TrainedModel, config: dict, features: np.ndarray) -> np.ndarray:
    """Run unified prediction for both MLP and random models."""
    predictions = make_predictions(
        model=trained_model,
        X=features,
        feature_scaler=trained_model.feature_scaler,
        label_scaler=trained_model.label_scaler,
        apply_feature_log=config.get("use_log", True),
        apply_inverse_transform=True,
        apply_label_expm1=config.get("use_log", True),
    )
    
    if config.get("use_area_root", False):
        label_names = config.get("label_names", [])
        if "Area" in label_names:
            area_idx = label_names.index("Area")
            predictions[:, area_idx] = predictions[:, area_idx] ** 2

    return predictions


def generate_report(
    input_file: str,
    output_dir: str,
    dataset: str,
    model_type: str,
    model_dir: str,
    config: dict,
    df_input: pd.DataFrame,
    df_predictions: pd.DataFrame,
) -> str:
    """Generate markdown report with prediction details."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Prediction Report

**Generated:** {timestamp}

## Configuration

| Parameter | Value |
|-----------|-------|
| Input File | `{input_file}` |
| Dataset Type | {dataset} |
| Model Type | {model_type.upper()} |
| Model Directory | `{model_dir}` |
| Samples | {len(df_predictions)} |

## Model Details

- **Model Name:** {config.get('model_name', config.get('activation_name', 'MLP'))}
- **Features ({len(config.get('feature_names', []))}):** {', '.join(config.get('feature_names', []))}
- **Outputs ({len(config.get('label_names', []))}):** {', '.join(config.get('label_names', []))}
- **Log Transform:** {config.get('use_log', False)}

## Input Summary

{df_input.describe().round(4).to_markdown()}

## Prediction Summary

{df_predictions.describe().round(4).to_markdown()}

## Output Files

- `predictions.csv` - Raw prediction values
- `report.md` - This report

## Usage Notes

- Predictions are in physical units (not scaled/transformed)
- For isotherm: Area (m²), Iso_distance (m), Iso_width (m)
- For cone: Cone (m)
"""
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Predict thermal plume parameters from input data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input CSV file with feature columns"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        choices=["isotherm", "cone"],
        help="Dataset type (determines features and outputs)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="mlp",
        choices=["mlp", "random"],
        help="Model type to use (default: mlp)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Custom model directory (overrides default)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: predictions_<timestamp>)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating markdown report"
    )
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load input data
    print(f"Loading input data: {args.input}")
    df_input = pd.read_csv(args.input)
    print(f"  Loaded {len(df_input)} samples")

    # Find model directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        base_dir = DEFAULT_MODEL_DIRS[args.dataset][args.model]
        try:
            model_dir = find_model_dir(base_dir)
        except FileNotFoundError:
            print(f"Error: No model found in {base_dir}")
            print(f"\nMake sure you have trained the {args.model} model for {args.dataset}.")
            print(f"Run: sbatch scripts/slurm/train_mlp_metrics.sbatch")
            sys.exit(1)

    print(f"Using model: {model_dir}")

    # Load model using unified wrapper (works for both MLP and random models)
    try:
        trained_model = TrainedModel(model_dir)
        config = trained_model.get_config()
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Get feature and label names
    feature_names = config.get("feature_names", DATASET_FEATURES[args.dataset])
    label_names = config.get("label_names", [])

    print(f"  Features: {feature_names}")
    print(f"  Outputs: {label_names}")

    # Validate input columns
    missing_cols = set(feature_names) - set(df_input.columns)
    if missing_cols:
        print(f"\nError: Input CSV missing required columns: {missing_cols}")
        print(f"Required columns: {feature_names}")
        sys.exit(1)

    # Extract features in correct order
    features = df_input[feature_names].values
    print(f"  Input shape: {features.shape}")

    # Run prediction using unified interface
    print("Running prediction...")
    predictions = predict_with_model(trained_model, config, features)

    # Create output DataFrame
    df_predictions = pd.DataFrame(predictions, columns=label_names)

    # Setup output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"predictions_{args.dataset}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Save predictions
    predictions_file = os.path.join(output_dir, "predictions.csv")
    df_predictions.to_csv(predictions_file, index=False)
    print(f"\nPredictions saved: {predictions_file}")

    # Generate report
    if not args.no_report:
        report = generate_report(
            input_file=args.input,
            output_dir=output_dir,
            dataset=args.dataset,
            model_type=args.model,
            model_dir=model_dir,
            config=config,
            df_input=df_input[feature_names],
            df_predictions=df_predictions,
        )
        report_file = os.path.join(output_dir, "report.md")
        with open(report_file, "w") as f:
            f.write(report)
        print(f"Report saved: {report_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(df_predictions.describe().round(4).to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
