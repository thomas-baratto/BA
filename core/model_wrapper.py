"""Unified model wrapper providing consistent interface for MLP and random models.

This module provides TrainedModel class that abstracts loading, configuration, and
prediction for both PyTorch MLP and scikit-learn/custom random models (ELM, RVFL, etc.).

Key Features:
- Auto-detects model type from config or saved artifacts
- Provides unified predict() interface
- Handles scaler loading and application
- Consistent configuration access
"""

import json
import gzip
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import InconsistentVersionWarning

from core.model import NeuralNetwork

# Silencing scikit-learn version mismatch warnings globally for this module
# These are harmless for inference and unavoidable on Python 3.9
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def preprocess_features(
    X: Union[np.ndarray, pd.DataFrame],
    feature_scaler: Any,
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
    label_scaler: Any,
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


def _load_pickle(path: Path):
    """Load a pickle file, transparently handling gzip compression.

    Tries gzip first; falls back to plain pickle for backward compatibility
    with uncompressed artifacts.  If the pickle embeds PyTorch CUDA tensors,
    they are automatically mapped to CPU.
    """
    import io

    def _unpickle_bytes(data: bytes):
        """Unpickle bytes, remapping CUDA tensors to CPU.

        Pickle files produced by ``pickle.dump`` that contain ``torch.Tensor``
        objects serialise each tensor via ``torch.storage._load_from_bytes``,
        which internally calls ``torch.load`` **without** ``map_location``.
        We monkeypatch that inner call to force ``map_location='cpu'``.
        """
        import torch.storage as _ts

        if hasattr(_ts, "_load_from_bytes"):
            _original = _ts._load_from_bytes

            def _cpu_load_from_bytes(b):
                # Use a temporary flag or restore original to avoid infinite recursion
                # since torch.load itself calls _load_from_bytes
                _ts._load_from_bytes = _original
                try:
                    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
                finally:
                    _ts._load_from_bytes = _cpu_load_from_bytes

            _ts._load_from_bytes = _cpu_load_from_bytes
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                    return pickle.loads(data)
            finally:
                _ts._load_from_bytes = _original
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                return pickle.loads(data)

    try:
        with gzip.open(path, "rb") as f:
            data = f.read()
        return _unpickle_bytes(data)
    except gzip.BadGzipFile:
        with open(path, "rb") as f:
            data = f.read()
        return _unpickle_bytes(data)


class TrainedModel:
    """
    Unified wrapper for trained models (MLP or random).
    
    Supports:
    - MLP models (PyTorch neural networks)
    - Random models (ELM, dRVFL, edRVFL, etc. via sklearn-compatible interface)
    
    Provides consistent interface for:
    - Loading from model directory
    - Accessing configuration
    - Making predictions
    - Handling feature/label scaling
    """

    def __init__(
        self,
        model_dir: str,
        model_type: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize wrapper.
        
        Args:
            model_dir: Path to directory containing model artifacts
            model_type: "mlp" or "random" (auto-detected if None)
            device: "cpu" or "cuda" (only affects MLP models)
        """
        self.model_dir = Path(model_dir)
        self.device = device
        
        # Load config first to determine model type
        self.config = self._load_config()
        
        # Auto-detect model type if not specified
        if model_type is None:
            model_type = self._detect_model_type()
        
        self.model_type = model_type.lower()
        
        if self.model_type == "mlp":
            self._load_mlp()
        elif self.model_type == "randomized":
            self._load_randomized()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load scalers
        self.scalers = self._load_scalers()

    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from JSON."""
        config_path = self.model_dir / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, "r") as f:
            return json.load(f)

    def _detect_model_type(self) -> str:
        """Auto-detect model type from artifacts or config."""
        # Check if MLP model exists
        if (self.model_dir / "best_model.pt").exists():
            return "mlp"
        
        # Check if randomized model exists
        if (self.model_dir / "model.pkl").exists():
            return "randomized"
        
        # Check config field
        if "model_type" in self.config:
            mtype = self.config["model_type"].lower()
            return "randomized" if mtype == "random" else mtype
        
        if "model_name" in self.config:
            # Randomized models (ELM, dRVFL, etc.)
            return "randomized"
        
        raise ValueError(f"Cannot detect model type in {self.model_dir}")

    def _load_mlp(self):
        """Load MLP model."""
        model_path = self.model_dir / "best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"MLP model not found: {model_path}")
        
        # Create model from config
        self.model = NeuralNetwork(
            input_size=self.config["input_size"],
            output_size=self.config["output_size"],
            nr_hidden_layers=self.config.get("nr_hidden_layers", 5),
            nr_neurons=self.config.get("nr_neurons", 128),
            activation_name=self.config.get("activation_name", "ReLU"),
            dropout_rate=self.config.get("dropout_rate", 0.0),
            use_batchnorm=self.config.get("use_batchnorm", False),
        )
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _load_randomized(self):
        """Load randomized model (ELM, RVFL, etc.)."""
        model_path = self.model_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Randomized model not found: {model_path}")
        
        self.model = _load_pickle(model_path)
        self._relocate_randomized_to_device()

    def _relocate_randomized_to_device(self):
        """Ensure a deserialized random model's tensors live on ``self.device``.

        When a model is trained on CUDA and loaded on a CPU-only machine,
        ``_load_pickle`` remaps the tensors to CPU but the model's
        ``device`` attribute still references the original CUDA device.
        This method patches the device and moves any lingering tensors.
        """
        target = torch.device(self.device)

        def _move(obj):
            if isinstance(obj, torch.Tensor) and obj.device != target:
                return obj.to(target)
            return obj

        # Fix top-level model(s) — randomized wrappers may contain sub-models
        models_to_fix = [self.model]
        if hasattr(self.model, "models"):
            models_to_fix.extend(self.model.models)

        for m in models_to_fix:
            if hasattr(m, "device"):
                m.device = target
            for attr_name in list(vars(m)):
                val = getattr(m, attr_name)
                if isinstance(val, torch.Tensor):
                    setattr(m, attr_name, _move(val))
                elif isinstance(val, list):
                    setattr(
                        m,
                        attr_name,
                        [_move(v) if isinstance(v, torch.Tensor) else v for v in val],
                    )
                elif isinstance(val, torch.nn.Module):
                    val.to(target)

    def _load_scalers(self) -> Dict[str, Any]:
        """Load feature and label scalers."""
        scalers_path = self.model_dir / "scalers.pkl"
        
        # If not in model dir, check parent (for multi-seed runs)
        if not scalers_path.exists():
            scalers_path = self.model_dir.parent / "scalers.pkl"
        
        if not scalers_path.exists():
            raise FileNotFoundError(
                f"Scalers not found in {self.model_dir} or parent. "
                "Model may need to be retrained."
            )
        
        return _load_pickle(scalers_path)

    @property
    def feature_scaler(self):
        """Get feature scaler."""
        return self.scalers.get("feature_scaler")

    @property
    def label_scaler(self):
        """Get label scaler."""
        return self.scalers.get("label_scaler")

    @property
    def feature_names(self) -> list:
        """Get feature column names."""
        return self.config.get("feature_names", [])

    @property
    def label_names(self) -> list:
        """Get label/target column names."""
        return self.config.get("label_names", [])

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        inverse_transform: bool = True,
        apply_log: bool = False
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features, shape (n_samples, n_features)
            inverse_transform: Whether to apply inverse scaling to predictions
            apply_log: Whether to apply inverse log transform (expm1)
        
        Returns:
            Predictions, shape (n_samples, n_targets)
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        # Make predictions
        if self.model_type == "mlp":
            # Convert to torch tensor for MLP
            X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                y_pred = self.model(X_torch)
            y_pred = y_pred.cpu().numpy()
        else:
            # Randomized model prediction
            y_pred = self.model.predict(X)
            if len(y_pred.shape) == 1:
                y_pred = y_pred.reshape(-1, 1)
        
        # Apply inverse transformations
        return postprocess_predictions(
            y_pred,
            label_scaler=self.label_scaler,
            inverse_transform=inverse_transform,
            apply_expm1=apply_log
        )

    def get_config(self, key: Optional[str] = None) -> Union[Dict, Any]:
        """
        Get model configuration.
        
        Args:
            key: Specific config key to retrieve (if None, returns full config)
        
        Returns:
            Configuration value or dict
        """
        if key is None:
            return self.config
        return self.config.get(key)
