"""Tests for core.model_wrapper — TrainedModel loading and prediction."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from core.model_wrapper import TrainedModel


class TestTrainedModelMLP:
    """Load an MLP from a fake artifact dir and run predictions."""

    def test_load_and_predict(self, mlp_artifact_dir: Path):
        tm = TrainedModel(str(mlp_artifact_dir))
        assert tm.model_type == "mlp"

        X = np.random.randn(5, 4).astype(np.float32)
        y = tm.predict(X, inverse_transform=False)
        assert y.shape == (5, 1)
        assert np.isfinite(y).all()

    def test_auto_detects_mlp_type(self, mlp_artifact_dir: Path):
        tm = TrainedModel(str(mlp_artifact_dir))
        assert tm.model_type == "mlp"

    def test_feature_and_label_names(self, mlp_artifact_dir: Path):
        tm = TrainedModel(str(mlp_artifact_dir))
        assert tm.feature_names == ["f1", "f2", "f3", "f4"]
        assert tm.label_names == ["target"]

    def test_scaler_access(self, mlp_artifact_dir: Path):
        tm = TrainedModel(str(mlp_artifact_dir))
        assert tm.feature_scaler is not None
        assert tm.label_scaler is not None

    def test_predict_with_inverse_transform(self, mlp_artifact_dir: Path):
        tm = TrainedModel(str(mlp_artifact_dir))
        X = np.random.randn(3, 4).astype(np.float32)

        y_raw = tm.predict(X, inverse_transform=False)
        y_inv = tm.predict(X, inverse_transform=True)

        # Inverse-transformed values should differ from raw (scaler was non-trivial)
        assert not np.allclose(y_raw, y_inv)

    def test_get_config(self, mlp_artifact_dir: Path):
        tm = TrainedModel(str(mlp_artifact_dir))
        assert tm.get_config("input_size") == 4
        assert isinstance(tm.get_config(), dict)

    def test_missing_config_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            TrainedModel(str(tmp_path))


class TestTrainedModelRandom:
    """Load a random-weight model from a fake artifact dir."""

    @pytest.fixture()
    def random_artifact_dir(self, tmp_path: Path) -> Path:
        from core.randomized.ELM import ELM
        from sklearn.preprocessing import MinMaxScaler

        d = tmp_path / "random_model"
        d.mkdir()

        elm = ELM(n_hidden=16, random_state=42)
        X = np.random.randn(20, 3).astype(np.float32)
        y = np.random.randn(20, 1).astype(np.float32)
        elm.fit(X, y)

        with open(d / "model.pkl", "wb") as f:
            pickle.dump(elm, f)

        config = {
            "model_type": "randomized",
            "model_name": "ELM",
            "input_size": 3,
            "output_size": 1,
            "feature_names": ["a", "b", "c"],
            "label_names": ["t"],
        }
        with open(d / "model_config.json", "w") as f:
            json.dump(config, f)

        fs = MinMaxScaler().fit(X)
        ls = MinMaxScaler().fit(y)
        with open(d / "scalers.pkl", "wb") as f:
            pickle.dump({"feature_scaler": fs, "label_scaler": ls}, f)

        return d

    def test_load_and_predict(self, random_artifact_dir: Path):
        tm = TrainedModel(str(random_artifact_dir))
        assert tm.model_type == "randomized"

        X = np.random.randn(5, 3).astype(np.float32)
        y = tm.predict(X, inverse_transform=False)
        assert y.shape == (5, 1)
        assert np.isfinite(y).all()


def test_auto_extract_zip_files(tmp_path: Path):
    """Test that auto_extract_zip_files automatically detects and extracts zipped model files."""
    import zipfile
    from core.model_wrapper import auto_extract_zip_files

    # 1. Create a dummy model directory with model_config.json
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    config_file = model_dir / "model_config.json"
    with open(config_file, "w") as f:
        json.dump({"model_type": "mlp"}, f)

    # 2. Package it into a zip file inside the tmp_path directory
    zip_file_path = tmp_path / "test-model-zipped.zip"
    with zipfile.ZipFile(zip_file_path, "w") as zip_ref:
        # Save config file inside a nested folder matching zip name
        zip_ref.write(config_file, arcname="test-model-zipped/model_config.json")

    # 3. Delete the original dummy directory so only the zip exists
    import shutil
    shutil.rmtree(model_dir)

    # Verify original folder doesn't exist
    extracted_dir = tmp_path / "test-model-zipped"
    assert not extracted_dir.exists()

    # 4. Trigger auto-extraction
    auto_extract_zip_files(tmp_path)

    # 5. Assert that the directory was automatically unzipped and contains the config file
    assert extracted_dir.exists()
    assert (extracted_dir / "model_config.json").exists()
    with open(extracted_dir / "model_config.json", "r") as f:
        cfg = json.load(f)
    assert cfg["model_type"] == "mlp"

