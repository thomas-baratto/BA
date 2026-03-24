"""Standardized artifact manifest for trained models.

Defines the structure and validation for model artifacts to ensure consistency
across MLP and random model training workflows.

All trained models should create artifacts following this schema:
- model_config.json: Model configuration and hyperparameters
- artifact_manifest.json: Metadata about training and artifacts (this schema)
- scalers.pkl: Feature and label scalers
- model.pkl (random) or best_model.pt (MLP): Trained model weights
- results.json: Training/test metrics and results
- test_predictions.npz: Predictions on test set
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ArtifactManifest:
    """
    Standardized artifact manifest for trained models.
    
    Example manifest.json:
    ```json
    {
        "version": "1.0",
        "created_at": "2026-03-21T15:23:45Z",
        "model_type": "mlp",
        "dataset": "isotherm",
        "targets": ["Area", "Iso_distance", "Iso_width"],
        "features": ["Flow_well", "Temp_diff", ..., "Isotherm"],
        "artifacts": {
            "config": "model_config.json",
            "model": "best_model.pt",
            "scalers": "scalers.pkl",
            "results": "results.json",
            "predictions": "test_predictions.npz"
        },
        "training": {
            "epochs": 1500,
            "batch_size": 32,
            "optimizer": "Adam",
            "learning_rate": 0.001
        },
        "performance": {
            "test_r2": 0.95,
            "test_rmse": 0.12,
            "test_mae": 0.08
        }
    }
    ```
    """

    SCHEMA_VERSION = "1.0"

    # Standard artifact filenames
    ARTIFACTS = {
        "config": "model_config.json",
        "model_mlp": "best_model.pt",
        "model_random": "model.pkl",
        "scalers": "scalers.pkl",
        "results": "results.json",
        "predictions": "test_predictions.npz",
        "manifest": "artifact_manifest.json",
    }

    def __init__(
        self,
        model_type: str,
        dataset: str,
        targets: List[str],
        features: List[str],
        training: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize artifact manifest.
        
        Args:
            model_type: "mlp" or "random"
            dataset: Dataset name (e.g. "isotherm", "cone")
            targets: List of target column names
            features: List of feature column names
            training: Optional dict with training parameters
            performance: Optional dict with performance metrics
        """
        self.manifest = {
            "version": self.SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_type": model_type,
            "dataset": dataset,
            "targets": targets,
            "features": features,
            "artifacts": self._build_artifacts_dict(model_type),
            "training": training or {},
            "performance": performance or {},
        }

    def _build_artifacts_dict(self, model_type: str) -> Dict[str, str]:
        """Build artifacts dictionary with correct model filename."""
        artifacts = {
            "config": self.ARTIFACTS["config"],
            "scalers": self.ARTIFACTS["scalers"],
            "results": self.ARTIFACTS["results"],
            "predictions": self.ARTIFACTS["predictions"],
        }
        
        if model_type.lower() == "mlp":
            artifacts["model"] = self.ARTIFACTS["model_mlp"]
        else:
            artifacts["model"] = self.ARTIFACTS["model_random"]
        
        return artifacts

    def save(self, output_dir: Path):
        """Save manifest to JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = output_dir / self.ARTIFACTS["manifest"]
        with open(manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def to_dict(self) -> Dict:
        """Return manifest as dictionary."""
        return self.manifest

    @classmethod
    def load(cls, manifest_path: Path) -> "ArtifactManifest":
        """Load manifest from JSON file."""
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, "r") as f:
            data = json.load(f)
        
        # Create instance
        instance = cls(
            model_type=data["model_type"],
            dataset=data["dataset"],
            targets=data["targets"],
            features=data["features"],
            training=data.get("training"),
            performance=data.get("performance"),
        )
        
        # Override created_at with loaded value
        instance.manifest["created_at"] = data.get("created_at")
        
        return instance


def validate_artifact_directory(artifact_dir: Path) -> bool:
    """
    Validate that artifact directory has required files.
    
    Args:
        artifact_dir: Path to directory containing model artifacts
    
    Returns:
        True if valid, raises exception otherwise
    """
    artifact_dir = Path(artifact_dir)
    
    # Check required files
    required_files = [
        ArtifactManifest.ARTIFACTS["config"],
        ArtifactManifest.ARTIFACTS["scalers"],
        ArtifactManifest.ARTIFACTS["results"],
    ]
    
    # Check for either MLP or random model
    mlp_exists = (artifact_dir / ArtifactManifest.ARTIFACTS["model_mlp"]).exists()
    random_exists = (artifact_dir / ArtifactManifest.ARTIFACTS["model_random"]).exists()
    
    if not (mlp_exists or random_exists):
        raise FileNotFoundError(
            f"No model file found in {artifact_dir}. "
            f"Expected either {ArtifactManifest.ARTIFACTS['model_mlp']} or "
            f"{ArtifactManifest.ARTIFACTS['model_random']}"
        )
    
    # Check required files exist
    for filename in required_files:
        filepath = artifact_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Missing required artifact: {filepath}")
    
    return True
