"""Centralized inference utilities for trained models.

Provides unified functions for:
- Loading models and associated scalers
- Preprocessing features (scaling)
- Postprocessing predictions (inverse transform)
- Making end-to-end predictions

This module replaces duplicated logic in predict.py and isotherm_reverse/main.py
and provides a consistent interface for model inference.
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from core.model_wrapper import TrainedModel


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


def preprocess_features(
    X: Union[np.ndarray, pd.DataFrame],
    feature_scaler,
    apply_log: bool = True,
) -> np.ndarray:
    """
    Preprocess input features for model prediction.
    
    Args:
        X: Input features, shape (n_samples, n_features) or DataFrame
        feature_scaler: Fitted feature scaler (e.g., MinMaxScaler)
        apply_log: Whether to apply log1p transform first
    
    Returns:
        Preprocessed features ready for model input
    """
    # Convert DataFrame to numpy
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Ensure 2D
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    # Apply log transform
    if apply_log:
        X = np.log1p(X)
    
    # Apply feature scaling
    if feature_scaler is not None:
        X = feature_scaler.transform(X)
    
    return X


def postprocess_predictions(
    y_pred: np.ndarray,
    label_scaler,
    inverse_transform: bool = True,
    apply_expm1: bool = True,
) -> np.ndarray:
    """
    Postprocess model predictions to original scale and units.
    
    Args:
        y_pred: Raw predictions from model
        label_scaler: Fitted label scaler
        inverse_transform: Whether to apply inverse scaling
        apply_expm1: Whether to apply inverse log transform (expm1)
    
    Returns:
        Predictions in original units
    """
    # Ensure 2D
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Apply inverse scaling
    if inverse_transform and label_scaler is not None:
        y_pred = label_scaler.inverse_transform(y_pred)
    
    # Apply inverse log transform
    if apply_expm1:
        # Clip to prevent extreme negative values that would cause issues with expm1
        y_pred = np.maximum(y_pred, -10)
        y_pred = np.expm1(y_pred)
    
    return y_pred


def make_predictions(
    model: TrainedModel,
    X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
    feature_scaler,
    label_scaler,
    apply_feature_log: bool = True,
    apply_inverse_transform: bool = True,
    apply_label_expm1: bool = True,
) -> np.ndarray:
    """
    End-to-end prediction pipeline.
    
    Handles:
    1. Feature preprocessing (log + scaling)
    2. Model prediction
    3. Output postprocessing (inverse scaling + inverse log)
    
    Args:
        model: TrainedModel instance
        X: Input features
        feature_scaler: Feature scaler
        label_scaler: Label scaler
        apply_feature_log: Apply log1p to features
        apply_inverse_transform: Apply inverse scaling to predictions
        apply_label_expm1: Apply expm1 (inverse log) to predictions
    
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
    
    return y_pred
