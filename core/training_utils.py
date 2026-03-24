"""Shared training utilities used across MLP and random model scripts."""

import numpy as np
import torch.nn as nn


def normalize_best_params(
    best_params: dict,
    max_epochs: int = 10000,
    patience: int = 250,
) -> dict:
    """Normalize parameter naming variants from Optuna trials.

    Handles legacy scaler naming (feature_scaler -> feature_scaler_type)
    and sets training defaults.
    """
    config = best_params.copy()

    # Handle scaler naming variants
    if "feature_scaler" in config and "feature_scaler_type" not in config:
        config["feature_scaler_type"] = config["feature_scaler"]
    if "label_scaler" in config and "label_scaler_type" not in config:
        config["label_scaler_type"] = config["label_scaler"]

    # Set defaults
    config.setdefault("plots", True)
    config.setdefault("num_epochs", max_epochs)
    config["patience"] = patience
    config.setdefault("use_log", True)

    return config


def get_loss_criterion(loss_name: str) -> nn.Module:
    """Get PyTorch loss criterion by name."""
    if loss_name == "L1":
        return nn.L1Loss()
    if loss_name == "SmoothL1":
        return nn.SmoothL1Loss()
    return nn.MSELoss()


def to_physical_units(
    y_scaled: np.ndarray,
    y_scaler,
    use_log: bool = True,
    use_area_root: bool = False,
    label_cols: list[str] | None = None,
) -> np.ndarray:
    """Convert scaled model output back to physical units.

    Applies inverse scaling, inverse log1p, and inverse sqrt(Area) in order.

    Args:
        y_scaled: Scaled predictions from model (shape: N x D)
        y_scaler: Fitted scaler with inverse_transform method (or None)
        use_log: Whether log1p was applied during preprocessing
        use_area_root: Whether sqrt was applied to Area during preprocessing
        label_cols: Label column names (needed for Area index if use_area_root)

    Returns:
        Predictions in original physical units
    """
    if y_scaler is not None:
        y_unscaled = y_scaler.inverse_transform(y_scaled)
    else:
        y_unscaled = y_scaled.copy()

    if use_log:
        y_unscaled = np.clip(y_unscaled, -600, 600)
        y_unscaled = np.expm1(y_unscaled)

    if use_area_root and label_cols and "Area" in label_cols:
        area_idx = label_cols.index("Area")
        y_unscaled = y_unscaled.copy()
        y_unscaled[:, area_idx] = y_unscaled[:, area_idx] ** 2

    return y_unscaled
