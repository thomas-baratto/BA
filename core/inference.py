"""Centralized inference utilities for trained models.

Provides unified functions for:
- Loading models and associated scalers
- Preprocessing features (scaling)
- Postprocessing predictions (inverse transform)
- Making end-to-end predictions

This module replaces duplicated logic in predict.py and isotherm_reverse/main.py
and provides a consistent interface for model inference.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from core.model_wrapper import TrainedModel, postprocess_predictions, preprocess_features


def load_model_and_scalers(
    model_dir: Union[str, Path]
) -> Tuple[TrainedModel, dict, dict]:
    """
    Load a trained model and its associated scalers.
    
    Args:
        model_dir: Path to directory containing model artifacts
    
    Returns:
        Tuple of (TrainedModel, feature_scaler, label_scaler)
    """
    model = TrainedModel(str(model_dir))
    
    return (
        model,
        model.feature_scaler,
        model.label_scaler,
    )





def make_predictions(
    model: TrainedModel,
    X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
    feature_scaler,
    label_scaler,
    apply_feature_log: bool = True,
    apply_inverse_transform: bool = True,
    apply_label_expm1: bool = True,
    use_area_root: bool = False,
    label_names: Optional[list] = None,
) -> np.ndarray:
    """
    End-to-end prediction pipeline.
    
    Handles:
    1. Feature preprocessing (log + scaling)
    2. Model prediction
    3. Output postprocessing (inverse scaling + inverse log)
    4. Inverse sqrt(Area) if use_area_root was applied during training
    
    Args:
        model: TrainedModel instance
        X: Input features
        feature_scaler: Feature scaler
        label_scaler: Label scaler
        apply_feature_log: Apply log1p to features
        apply_inverse_transform: Apply inverse scaling to predictions
        apply_label_expm1: Apply expm1 (inverse log) to predictions
        use_area_root: Whether sqrt(Area) was used during training
            (if True, Area column is squared to restore original units)
        label_names: Label column names (needed to locate "Area" column
            when use_area_root is True). Falls back to model.label_names.
    
    Returns:
        Predictions in original units
    """
    # Preprocess features
    X_preprocessed = preprocess_features(
        X,
        feature_scaler=feature_scaler,
        apply_log=apply_feature_log,
    )
    
    # Make predictions
    y_pred = model.predict(X_preprocessed, inverse_transform=False)
    
    # Postprocess predictions
    y_pred = postprocess_predictions(
        y_pred,
        label_scaler=label_scaler,
        inverse_transform=apply_inverse_transform,
        apply_expm1=apply_label_expm1,
    )
    
    # Reverse sqrt(Area) transform if applied during training
    if use_area_root:
        if label_names is None:
            label_names = getattr(model, "label_names", [])
        if "Area" in label_names:
            area_idx = label_names.index("Area")
            y_pred = y_pred.copy()
            y_pred[:, area_idx] = y_pred[:, area_idx] ** 2
    
    return y_pred
