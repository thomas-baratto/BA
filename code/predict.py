#!/usr/bin/env python3
"""Unified prediction CLI for thermal plume models.

Predicts isotherm or cone values from input CSV using trained MLP or randomized models (RVFL).

The dataset type (isotherm/cone) is automatically detected from CSV headers if omitted.

Usage:
  # Predict with auto-detection (recommended)
  ba-predict --input data/sample_cone.csv

  # Explicitly specify model and dataset
  ba-predict --input data/sample_isotherm.csv --dataset isotherm --model randomized:nRMSE

  # Specify output location
  ba-predict --input data.csv --output results/

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
import sys
from typing import Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as both a script (PYTHONPATH=.) and installed package (pip install .)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.datasets import DATASET_CONFIGS, DEFAULT_MODEL_DIRS, detect_dataset_type


# Feature columns for each dataset (used for validation) - imported from centralized config
DATASET_FEATURES = {dataset: cfg["features"] for dataset, cfg in DATASET_CONFIGS.items()}
from core.model_wrapper import TrainedModel, auto_extract_zip_files
from core.inference import make_predictions

def find_model_dir(base_dir: str, expected_type: Optional[str] = None, expected_dataset: Optional[str] = None, variant_hint: Optional[str] = None) -> str:
    """Find model directory containing model_config.json.

    Searches recursively but ignores hidden directories for performance.
    If expected_type/dataset/variant are provided, it verifies them in the config or path.
    """
    import json
    base_path = Path(base_dir)
    
    # 1. On-the-fly zip extraction if model files are compressed
    auto_extract_zip_files(base_path)
    
    # Search recursively for model_config.json, skipping hidden directories
    for path in base_path.rglob("model_config.json"):
        if any(part.startswith('.') for part in path.parts):
            continue
        
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
            
            # 1. Check Model Type
            actual_type = cfg.get("model_type")
            if actual_type is None:
                if "nr_hidden_layers" in cfg:
                    actual_type = "mlp"
            
            if actual_type == "random":
                actual_type = "randomized"
            
            if expected_type and actual_type != expected_type:
                continue

            # 2. Check Dataset
            actual_dataset = cfg.get("dataset")
            if actual_dataset is None:
                input_size = cfg.get("input_size")
                if input_size == 9:
                    actual_dataset = "isotherm"
                elif input_size == 4:
                    actual_dataset = "cone"
            
            if expected_dataset and actual_dataset != expected_dataset:
                continue

            # 3. Check Variant Hint (e.g. nRMSE, KGE, or a specific model name)
            if variant_hint:
                vh = variant_hint.lower()
                # Check config keys
                config_str = json.dumps(cfg).lower()
                # Check path parts (folder names)
                path_str = str(path).lower()
                
                if vh not in config_str and vh not in path_str:
                    continue

            return str(path.parent)

        except Exception:
            continue

    raise FileNotFoundError(f"No model artifacts found in: {base_dir}")


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
        use_area_root=config.get("use_area_root", False),
        label_names=config.get("label_names"),
    )

    return predictions


def generate_report(
    input_file: str,
    dataset: str,
    model_type: str,
    model_dir: str,
    config: dict,
    df_input: pd.DataFrame,
    df_predictions: pd.DataFrame,
) -> str:
    """Generate professional markdown report with prediction details."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine architecture name
    arch = config.get('model_name', 'MLP')
    if model_type == 'mlp':
        arch = f"MLP ({config.get('nr_neurons', 128)} neurons, {config.get('nr_hidden_layers', 5)} layers)"
    
    report = f"""# Inference Report - {dataset.capitalize()} Dataset

**Run Date:** {timestamp}
**Source File:** `{input_file}`

---

## 🛠 Model Configuration

| Parameter | Value |
|:---|:---|
| **Architecture** | {arch} |
| **Model Family** | {model_type.upper()} |
| **Discovery Path** | `{model_dir}` |
| **Sample Count** | {len(df_predictions)} |

### Preprocessing Settings
- **Log Transform applied:** `{config.get('use_log', False)}`
- **Area square-root used:** `{config.get('use_area_root', False)}`
- **Feature Scaler:** `{config.get('feature_scaler_type', 'robust')}`

---

## 📊 Statistics Summary

### Input Features
{df_input.describe().round(4).to_markdown()}

### Predicted Values
{df_predictions.describe().round(4).to_markdown()}

---

## 📂 Output Files

All results are saved in the output directory:
- `predictions.csv`: Full prediction matrix in physical units.
- `report.md`: This summary report.

## 💡 Technical Notes

- **Units:** All outputs are restored to their physical units (m, m², etc.) and are NOT scaled.
- **Isotherm targets:** Area (m²), Iso_distance (m), Iso_width (m)
- **Cone targets:** Cone (m)
- **Inference Hardware:** CPU (portability mode enabled)
"""
    return report


def run_tests():
    """Run pytest on the local tests directory."""
    try:
        import pytest
        print("\n" + "="*60)
        print("RUNNING VERIFICATION SUITE")
        print("="*60 + "\n")
        tests_dir = str(Path(__file__).resolve().parent / "tests")
        exit_code = pytest.main([tests_dir, "-v"])
        if exit_code == 0:
            print("\n" + "="*60)
            print("VERIFICATION SUCCESSFUL: All tests passed!")
            print("="*60 + "\n")
        else:
            print("\n" + "!"*60)
            print(f"VERIFICATION FAILED: Exit code {exit_code}")
            print("Please check your environment and dependencies.")
            print("!"*60 + "\n")
        return exit_code
    except ImportError:
        print("\n" + "!"*60)
        print("Error: 'pytest' not found.")
        print("Please install development dependencies: pip install '.[dev]'")
        print("!"*60 + "\n")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Predict thermal plume parameters from input data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--test", action="store_true", help="Run the test suite to verify installation"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=False,
        help="Input CSV file with feature columns"
    )
    parser.add_argument("--dataset", "-d", choices=["isotherm", "cone"], help="Dataset type (auto-detected if omitted)")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="mlp",
        choices=["mlp", "randomized", "randomized:nRMSE", "randomized:KGE"],
        help=(
            "Model type to use (default: mlp). "
            "'randomized' uses an optimized randomized neural network (RVFL). "
            "For isotherm, 'randomized:nRMSE' and 'randomized:KGE' select models optimized for different metrics. "
            "For cone, all randomized variants use the same model."
        ),
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

    # Run tests and exit if requested
    if args.test:
        sys.exit(run_tests())

    # Validate input file (required for prediction)
    if not args.input:
        parser.error("the following arguments are required: --input/-i (unless --test is used)")

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load input data
    print(f"Loading input data: {args.input}")
    df_input = pd.read_csv(args.input)
    print(f"  Loaded {len(df_input)} samples")

    # Auto-detect dataset if not provided
    dataset = args.dataset
    if not dataset:
        print("Auto-detecting dataset type...")
        dataset = detect_dataset_type(args.input)
        if dataset:
            print(f"  Detected dataset: {dataset}")
        else:
            print(f"Error: Could not automatically detect dataset type for {args.input}")
            print("Please specify manually using --dataset {isotherm,cone}")
            sys.exit(1)

    # Find model directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_key = args.model
        dataset_dirs = DEFAULT_MODEL_DIRS[dataset]
        # Fall back to "randomized" if the specific variant isn't available
        # (e.g. cone has only one optimized randomized model)
        if model_key not in dataset_dirs and model_key.startswith("randomized"):
            model_key = "randomized"
        base_dir = dataset_dirs[model_key]
        # Extract variant hint if present (e.g. "randomized:nRMSE" -> "nRMSE")
        parts = model_key.split(":")
        expected_type = parts[0]
        variant_hint = parts[1] if len(parts) > 1 else None

        try:
            model_dir = find_model_dir(
                base_dir, 
                expected_type=expected_type,
                expected_dataset=dataset,
                variant_hint=variant_hint
            )
        except FileNotFoundError:
            print(f"Error: No model found in {base_dir}")
            print(f"\nMake sure the DaRUS archive is fully extracted and the")
            print(f"models/ folder is present in the repository root.")
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
    feature_names = config.get("feature_names", DATASET_FEATURES[dataset])
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
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        output_dir = f"outputs/{dataset}_{trained_model.model_type}_{timestamp}"
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save predictions
    predictions_file = out_path / "predictions.csv"
    df_predictions.to_csv(predictions_file, index=False)
    print(f"\nPredictions saved: {predictions_file}")

    # Generate report
    if not args.no_report:
        report = generate_report(
            input_file=args.input,
            dataset=dataset,
            model_type=args.model,
            model_dir=model_dir,
            config=config,
            df_input=df_input[feature_names],
            df_predictions=df_predictions,
        )
        report_file = out_path / "report.md"
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
