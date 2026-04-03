#!/usr/bin/env python3
"""Evaluate Böttcher & Zosseder analytical regression baseline on the isotherm dataset.

Applies the empirical power-law formulas (eqs. 5–7 from the GeoKW poster) to predict
thermal anomaly area, down-gradient length (Iso_distance), and cross-gradient width
(Iso_width) for the 1 K isotherm contour.

The formulas use derived quantities:
    V_tech  = Flow_well / 86400          [m³/s]  (Flow_well is in m³/day)
    ΔT      = Temp_diff                  [K]
    v_D     = Hydr_conductivity * Hydr_gradient  [m/s]
    b       = Aqu_thickness              [m]

Regression equations (fitted for αL = 10 m, αT = 1 m, 1 K isotherm):
    A_therm = 0.47 · V_tech^2.31 · ΔT^2.92 · (v_D · b)^(−2.31)
    L_therm = 0.54 · V_tech^1.50 · ΔT^1.96 · (v_D · b)^(−1.50)
    W_therm = 1.16 · V_tech       · ΔT^0.76 · (v_D · b)^(−1.00)

Usage:
    PYTHONPATH=. .venv/bin/python scripts/analysis/evaluate_boettcher_baseline.py
    PYTHONPATH=. .venv/bin/python scripts/analysis/evaluate_boettcher_baseline.py --dispersivity-filter
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.datasets import DATASET_CONFIGS, DEFAULT_MODEL_DIRS
from core.metrics import compute_regression_metrics
from core.inference import load_model_and_scalers, make_predictions


def _per_label_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_cols: list[str]
) -> dict[str, dict]:
    """Compute regression metrics independently for each label column."""
    result = {}
    for i, name in enumerate(label_cols):
        result[name] = compute_regression_metrics(
            y_true[:, i : i + 1], y_pred[:, i : i + 1]
        )
    return result


def boettcher_predict(df: pd.DataFrame) -> np.ndarray:
    """Apply Böttcher regression formulas to a DataFrame of raw features.

    Args:
        df: DataFrame containing Flow_well, Temp_diff, Hydr_conductivity,
            Hydr_gradient, Aqu_thickness columns.

    Returns:
        Array of shape (n, 3) with columns [Area, Iso_distance, Iso_width].
    """
    v_tech = df["Flow_well"].values / 86400.0  # m³/day → m³/s
    delta_t = df["Temp_diff"].values  # K
    v_d = df["Hydr_conductivity"].values * df["Hydr_gradient"].values  # m/s
    b = df["Aqu_thickness"].values  # m
    vd_b = v_d * b

    area = 0.47 * v_tech**2.31 * delta_t**2.92 * vd_b ** (-2.31)
    length = 0.54 * v_tech**1.50 * delta_t**1.96 * vd_b ** (-1.50)
    width = 1.16 * v_tech * delta_t**0.76 * vd_b ** (-1.00)

    return np.column_stack([area, length, width])


def main():
    parser = argparse.ArgumentParser(description="Evaluate Böttcher analytical baseline")
    parser.add_argument(
        "--dispersivity-filter",
        action="store_true",
        help="Additionally evaluate on the subset where αL=10, αT=1 (matching the formula's fitting data).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/baseline/boettcher",
        help="Directory to save results JSON.",
    )
    args = parser.parse_args()

    cfg = DATASET_CONFIGS["isotherm"]
    label_cols = cfg["labels"]  # ["Area", "Iso_distance", "Iso_width"]

    # ── Load full dataset ───────────────────────────────────────────────
    df = pd.read_csv(cfg["csv_file"])
    print(f"Loaded {len(df)} rows from {cfg['csv_file']}")

    # ── Reproduce the same train/test split as the NN pipeline ──────────
    # The data_loader splits X, y arrays *after* selecting columns but
    # *before* any log-transform or scaling.  We replicate that split on
    # the full DataFrame by splitting indices.
    indices = np.arange(len(df))
    _, test_idx = train_test_split(
        indices, test_size=0.3, random_state=42, shuffle=True
    )
    df_test = df.iloc[test_idx].copy()
    print(f"Test set: {len(df_test)} rows")

    # ── Filter to 1 K isotherm ──────────────────────────────────────────
    df_iso1 = df_test[df_test["Isotherm"] == 1].copy()
    print(f"Test set, Isotherm==1: {len(df_iso1)} rows")

    results = {}

    # ── Evaluate on all Isotherm==1 test rows ───────────────────────────
    y_true = df_iso1[label_cols].values
    y_pred = boettcher_predict(df_iso1)

    overall = compute_regression_metrics(y_true, y_pred)
    per_label = _per_label_metrics(y_true, y_pred, label_cols)

    results["all_dispersivities"] = {
        "n_samples": len(df_iso1),
        "overall": overall,
        "per_label": per_label,
    }

    print("\n=== Böttcher baseline — Isotherm==1, all dispersivities ===")
    print(f"  n = {len(df_iso1)}")
    print(f"  R²    = {overall['r2']:.6f}")
    print(f"  nRMSE = {overall['nrmse']:.6f}")
    print(f"  KGE   = {overall['kge']:.6f}")
    print(f"  MAE   = {overall['mae']:.2f}")
    print(f"  RMSE  = {overall['rmse']:.2f}")
    for label, m in per_label.items():
        print(f"  {label:15s}  R²={m['r2']:.6f}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}")

    # ── Optional: filter to matching dispersivity (αL=10, αT=1) ─────────
    if args.dispersivity_filter:
        df_match = df_iso1[
            (df_iso1["Long_dispersivity"] == 10)
            & (df_iso1["Trans_dispersivity"] == 1.0)
        ].copy()
        print(f"\nTest set, Isotherm==1, αL=10, αT=1: {len(df_match)} rows")

        if len(df_match) > 0:
            y_true_m = df_match[label_cols].values
            y_pred_m = boettcher_predict(df_match)

            overall_m = compute_regression_metrics(y_true_m, y_pred_m)
            per_label_m = _per_label_metrics(y_true_m, y_pred_m, label_cols)

            results["matching_dispersivity"] = {
                "n_samples": len(df_match),
                "dispersivity": {"Long": 10, "Trans": 1.0},
                "overall": overall_m,
                "per_label": per_label_m,
            }

            print("\n=== Böttcher baseline — Isotherm==1, αL=10, αT=1 ===")
            print(f"  n = {len(df_match)}")
            print(f"  R²    = {overall_m['r2']:.6f}")
            print(f"  nRMSE = {overall_m['nrmse']:.6f}")
            print(f"  KGE   = {overall_m['kge']:.6f}")
            print(f"  MAE   = {overall_m['mae']:.2f}")
            print(f"  RMSE  = {overall_m['rmse']:.2f}")
            for label, m in per_label_m.items():
                print(f"  {label:15s}  R²={m['r2']:.6f}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}")
        else:
            print("  No matching rows found.")

    # ── Evaluate NN models on the same Isotherm==1 subset ─────────────
    feature_cols = cfg["features"]
    nn_models = {
        "MLP": DEFAULT_MODEL_DIRS["isotherm"]["mlp"],
        "dRVFL": DEFAULT_MODEL_DIRS["isotherm"]["random:KGE"],
        "SResdRVFL": DEFAULT_MODEL_DIRS["isotherm"]["random:nRMSE"],
    }

    X_iso1 = df_iso1[feature_cols].values
    y_true_iso1 = df_iso1[label_cols].values

    for model_name, model_dir in nn_models.items():
        try:
            model, feat_scaler, lbl_scaler = load_model_and_scalers(model_dir)
            cfg_m = model.config
            use_log = cfg_m.get("use_log", True)
            use_area_root = cfg_m.get("use_area_root", False)

            y_nn = make_predictions(
                model, X_iso1, feat_scaler, lbl_scaler,
                apply_feature_log=use_log,
                apply_inverse_transform=True,
                apply_label_expm1=use_log,
                use_area_root=use_area_root,
                label_names=label_cols,
            )
            nn_overall = compute_regression_metrics(y_true_iso1, y_nn)
            nn_per_label = _per_label_metrics(y_true_iso1, y_nn, label_cols)

            results[f"nn_{model_name}"] = {
                "n_samples": len(df_iso1),
                "overall": nn_overall,
                "per_label": nn_per_label,
            }

            print(f"\n=== {model_name} — Isotherm==1, all dispersivities ===")
            print(f"  n = {len(df_iso1)}")
            print(f"  R²    = {nn_overall['r2']:.6f}")
            print(f"  nRMSE = {nn_overall['nrmse']:.6f}")
            print(f"  KGE   = {nn_overall['kge']:.6f}")
            print(f"  MAE   = {nn_overall['mae']:.2f}")
            print(f"  RMSE  = {nn_overall['rmse']:.2f}")
            for label, m in nn_per_label.items():
                print(f"  {label:15s}  R²={m['r2']:.6f}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}")
        except Exception as e:
            print(f"\n  [WARN] Could not evaluate {model_name}: {e}")

    # ── Save results ────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "results_boettcher_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
