"""Tests for core.inference — the full prediction pipeline.

Covers:
1. preprocess_features: scaling and logging
2. postprocess_predictions: inverse-scaling, expm1, and numerical clipping
3. make_predictions: the end-to-end integration (preprocess -> predict -> postprocess)
4. Device Relocation: ensuring models move from GPU to CPU for portability
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock
from sklearn.preprocessing import MinMaxScaler

from core.inference import preprocess_features, postprocess_predictions, make_predictions
from core.model_wrapper import TrainedModel
from core.model import NeuralNetwork


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

class DummyPickleModel:
    """Helper class for pickling tests (must be at module level)."""
    def __init__(self):
        self.device = torch.device("cpu")
    def to(self, device): 
        self.device = device
        return self


@pytest.fixture()
def mlp_inference_dir(tmp_path: Path) -> Path:
    """Minimal 4-in / 2-out MLP artifact directory for integration tests."""
    d = tmp_path / "mlp_integration"
    d.mkdir()

    model = NeuralNetwork(
        input_size=4, 
        output_size=2, 
        nr_hidden_layers=1, 
        nr_neurons=16,
        activation_name="ReLU",
        dropout_rate=0.0,
        use_batchnorm=False
    )
    torch.save(model.state_dict(), d / "best_model.pt")

    config = {
        "input_size": 4, 
        "output_size": 2,
        "nr_hidden_layers": 1, 
        "nr_neurons": 16,
        "activation_name": "ReLU", 
        "dropout_rate": 0.0, 
        "use_batchnorm": False,
        "feature_names": ["f1", "f2", "f3", "f4"],
        "label_names": ["Area", "Iso_width"],
        "use_log": True,
        "use_area_root": True,
    }
    with open(d / "model_config.json", "w") as f:
        json.dump(config, f)

    rng = np.random.default_rng(42)
    feat_scaler = MinMaxScaler().fit(rng.uniform(1, 10, (20, 4)))
    lbl_scaler = MinMaxScaler().fit(rng.uniform(1, 10, (20, 2)))
    with open(d / "scalers.pkl", "wb") as f:
        pickle.dump({"feature_scaler": feat_scaler, "label_scaler": lbl_scaler}, f)

    return d


# ---------------------------------------------------------------------------
# Unit Tests: Preprocessing & Postprocessing
# ---------------------------------------------------------------------------

class TestPreprocessFeatures:

    @pytest.fixture()
    def fitted_scaler(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 100, (30, 4))
        return MinMaxScaler().fit(np.log1p(X))

    def test_log_and_scale(self, fitted_scaler):
        X = np.array([[10.0, 20.0, 30.0, 40.0]])
        out = preprocess_features(X, fitted_scaler, apply_log=True)
        assert out.shape == (1, 4)
        assert np.isfinite(out).all()

    def test_none_scaler_skips_scaling(self):
        X = np.array([[1.0, 2.0]])
        out = preprocess_features(X, feature_scaler=None, apply_log=False)
        np.testing.assert_array_equal(out, X)


class TestPostprocessPredictions:

    def test_numerical_clipping(self):
        """Ensure postprocess_predictions clips values before expm1 to prevent overflow."""
        y_raw = np.array([[-100.0]])
        out = postprocess_predictions(y_raw, label_scaler=None, inverse_transform=False, apply_expm1=True)
        expected = np.expm1(-10.0)
        np.testing.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# Integration Tests: make_predictions
# ---------------------------------------------------------------------------

class TestMakePredictions:

    def test_end_to_end_pipeline(self, mlp_inference_dir):
        """Verify the full flow: preprocess -> predict -> postprocess."""
        tm = TrainedModel(str(mlp_inference_dir))
        X = np.random.uniform(1, 5, (10, 4))
        
        preds = make_predictions(tm, X, tm.feature_scaler, tm.label_scaler, use_area_root=True)
        
        assert preds.shape == (10, 2)
        assert np.isfinite(preds).all()
        # Area (column 0) should be squared because use_area_root=True in fixture
        assert (preds[:, 0] >= 0).all()

    def test_accepts_dataframe_input(self, mlp_inference_dir):
        import pandas as pd
        tm = TrainedModel(str(mlp_inference_dir))
        df = pd.DataFrame(np.random.uniform(1, 5, (5, 4)), columns=["f1", "f2", "f3", "f4"])
        
        preds = make_predictions(tm, df, tm.feature_scaler, tm.label_scaler)
        assert preds.shape == (5, 2)


# ---------------------------------------------------------------------------
# Portability Tests: Device Relocation
# ---------------------------------------------------------------------------

class TestDeviceRelocation:

    def test_relocate_randomized_to_cpu_mocked(self, monkeypatch):
        """Simulate a model trained on CUDA being moved to CPU for portability."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        
        class MockModel:
            def __init__(self):
                self.device = torch.device("cuda:0")
                self.weight = torch.randn(2, 2)
        
        class MockWrapper:
            def __init__(self, model):
                self.model = model
                self.device = "cpu"
        
        model = MockModel()
        wrapper = MockWrapper(model)
        TrainedModel._relocate_randomized_to_device(wrapper)
        assert model.device == torch.device("cpu")

    def test_detect_randomized_type_mapping(self, tmp_path):
        """Ensure 'random' type string in old configs maps to 'randomized' internal type."""
        config_path = tmp_path / "model_config.json"
        with open(config_path, "w") as f:
            json.dump({"model_type": "random", "input_size": 1, "output_size": 1}, f)
        
        with open(tmp_path / "model.pkl", "wb") as f:
            pickle.dump(DummyPickleModel(), f)
        with open(tmp_path / "scalers.pkl", "wb") as f:
            pickle.dump({"feature_scaler": None, "label_scaler": None}, f)

        model = TrainedModel(str(tmp_path))
        assert model.model_type == "randomized"
