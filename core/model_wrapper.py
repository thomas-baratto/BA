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
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from core.model import NeuralNetwork


def _load_pickle(path: Path):
    """Load a pickle file, transparently handling gzip compression.

    Tries gzip first; falls back to plain pickle for backward compatibility
    with uncompressed artifacts.
    """
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except gzip.BadGzipFile:
        with open(path, "rb") as f:
            return pickle.load(f)


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
        elif self.model_type == "random":
            self._load_random()
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
        
        # Check if random model exists
        if (self.model_dir / "model.pkl").exists():
            return "random"
        
        # Check config field
        if "model_type" in self.config:
            return self.config["model_type"]
        
        if "model_name" in self.config:
            # Random models (ELM, dRVFL, etc.)
            return "random"
        
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

    def _load_random(self):
        """Load random model (ELM, RVFL, etc.)."""
        model_path = self.model_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Random model not found: {model_path}")
        
        self.model = _load_pickle(model_path)

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
            # Random model prediction
            y_pred = self.model.predict(X)
            if len(y_pred.shape) == 1:
                y_pred = y_pred.reshape(-1, 1)
        
        # Apply inverse transformations
        if inverse_transform and self.label_scaler is not None:
            y_pred = self.label_scaler.inverse_transform(y_pred)
        
        if apply_log:
            y_pred = np.expm1(y_pred)
        
        return y_pred

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
