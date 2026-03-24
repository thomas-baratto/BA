"""Regression metrics computation."""

import logging
import warnings

import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.metrics import r2_score, explained_variance_score

# --- Physical Units Mapping ---
LABEL_UNITS = {
    "Area": "m²",
    "sqrt(Area)": "m",
    "Iso_distance": "m",
    "Iso_width": "m",
    "Cone": "m",
}


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Calculates a comprehensive dictionary of regression metrics on unscaled data.

    For multi-output data, computes metrics on flattened arrays (true overall metrics).

    Args:
        y_true: Ground truth target values (can be 1D or 2D)
        y_pred: Predicted target values (can be 1D or 2D)

    Returns:
        Dictionary containing various regression metrics
    """
    metrics = {}

    # Flatten arrays for true overall metrics (not averaged per-output)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Compute metrics on flattened data
    metrics["mae"] = float(np.mean(np.abs(y_pred_flat - y_true_flat)))
    metrics["mse"] = float(np.mean((y_pred_flat - y_true_flat) ** 2))
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    if len(y_true_flat) < 2:
        metrics["r2"] = float("nan")
        metrics["explained_variance"] = float("nan")
    else:
        metrics["r2"] = float(r2_score(y_true_flat, y_pred_flat))
        metrics["explained_variance"] = float(
            explained_variance_score(y_true_flat, y_pred_flat)
        )
    metrics["medae"] = float(np.median(np.abs(y_pred_flat - y_true_flat)))

    # max_error
    metrics["max_error"] = float(np.max(np.abs(y_pred_flat - y_true_flat)))

    # MAPE (mean absolute percentage error)
    mask = y_true_flat != 0
    if np.any(mask):
        metrics["mape"] = float(
            np.mean(
                np.abs(
                    (y_pred_flat[mask] - y_true_flat[mask]) / y_true_flat[mask]
                )
            )
        )
    else:
        logging.warning("Could not calculate MAPE: all true values are zero")
        metrics["mape"] = float("nan")

    # nRMSE normalized by target range.
    y_range = float(np.max(y_true_flat) - np.min(y_true_flat))
    if y_range > 0:
        metrics["nrmse"] = float(metrics["rmse"] / y_range)
    else:
        metrics["nrmse"] = float("nan")

    # Kling-Gupta Efficiency (KGE): 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    y_true_std = float(np.std(y_true_flat))
    y_pred_std = float(np.std(y_pred_flat))
    y_true_mean = float(np.mean(y_true_flat))
    y_pred_mean = float(np.mean(y_pred_flat))

    if y_true_std > 0 and y_pred_std > 0:
        r = float(np.corrcoef(y_true_flat, y_pred_flat)[0, 1])
    else:
        r = float("nan")

    alpha = float(y_pred_std / y_true_std) if y_true_std > 0 else float("nan")
    beta = (
        float(y_pred_mean / y_true_mean) if y_true_mean != 0 else float("nan")
    )

    if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
        metrics["kge"] = float(
            1.0
            - np.sqrt(
                (r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2
            )
        )
    else:
        metrics["kge"] = float("nan")

    # Relative error statistics
    if np.any(mask):
        err_rel = (y_pred_flat[mask] - y_true_flat[mask]) / y_true_flat[mask]
        metrics["rel_err_std"] = float(np.std(err_rel))
        metrics["rel_err_mean_abs"] = float(np.mean(np.abs(err_rel)))
    else:
        metrics["rel_err_std"] = float("nan")
        metrics["rel_err_mean_abs"] = float("nan")

    # Residual statistics
    residuals = y_pred_flat - y_true_flat
    metrics["residual_mean"] = float(np.mean(residuals))
    metrics["residual_std"] = float(np.std(residuals))
    with warnings.catch_warnings():
        # Suppress RuntimeWarning from scipy when residuals are near-constant
        # (catastrophic cancellation in moment calculation)
        warnings.simplefilter("ignore", RuntimeWarning)
        metrics["residual_skew"] = float(skew(residuals))
        metrics["residual_kurtosis"] = float(kurtosis(residuals))
    metrics["residual_p95"] = float(np.percentile(residuals, 95))
    metrics["residual_p99"] = float(np.percentile(residuals, 99))

    return metrics
