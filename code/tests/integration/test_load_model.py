"""Tests for core.inference.load_model_and_scalers.

load_model_and_scalers is re-exported from core/__init__.py and is the
convenience loader that returns (TrainedModel, feature_scaler, label_scaler).
"""

from __future__ import annotations

import pytest

from core.inference import load_model_and_scalers


class TestLoadModelAndScalers:

    def test_returns_model_and_two_scalers(self, mlp_artifact_dir):
        model, feat_scaler, lbl_scaler = load_model_and_scalers(mlp_artifact_dir)
        assert feat_scaler is not None
        assert lbl_scaler is not None

    def test_model_type_is_mlp(self, mlp_artifact_dir):
        model, _, _ = load_model_and_scalers(mlp_artifact_dir)
        assert model.model_type == "mlp"

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_and_scalers(tmp_path / "nonexistent")

    def test_scaler_is_same_as_property(self, mlp_artifact_dir):
        """Returned scalers must match the model's own properties."""
        model, feat_scaler, lbl_scaler = load_model_and_scalers(mlp_artifact_dir)
        assert feat_scaler is model.feature_scaler
        assert lbl_scaler is model.label_scaler
