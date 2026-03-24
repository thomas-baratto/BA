"""Tests for core.artifacts — ArtifactManifest and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.artifacts import ArtifactManifest, validate_artifact_directory


class TestArtifactManifest:

    def test_create_mlp_manifest(self):
        m = ArtifactManifest(
            model_type="mlp",
            dataset="cone",
            targets=["Cone"],
            features=["Flow_well", "Hydr_gradient", "Hydr_conductivity", "Aqu_thickness"],
        )
        d = m.to_dict()
        assert d["model_type"] == "mlp"
        assert d["artifacts"]["model"] == "best_model.pt"
        assert d["version"] == ArtifactManifest.SCHEMA_VERSION

    def test_create_random_manifest(self):
        m = ArtifactManifest(
            model_type="random",
            dataset="isotherm",
            targets=["Area", "Iso_distance", "Iso_width"],
            features=["Flow_well"],
        )
        d = m.to_dict()
        assert d["artifacts"]["model"] == "model.pkl"

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        m = ArtifactManifest(
            model_type="mlp",
            dataset="cone",
            targets=["Cone"],
            features=["f1"],
            training={"epochs": 100},
            performance={"test_r2": 0.95},
        )
        m.save(tmp_path)

        loaded = ArtifactManifest.load(tmp_path / "artifact_manifest.json")
        assert loaded.to_dict()["dataset"] == "cone"
        assert loaded.to_dict()["training"]["epochs"] == 100
        assert loaded.to_dict()["performance"]["test_r2"] == 0.95

    def test_load_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ArtifactManifest.load(tmp_path / "no_such_file.json")


class TestValidateArtifactDirectory:

    def test_valid_mlp_dir(self, mlp_artifact_dir: Path):
        # mlp_artifact_dir has best_model.pt, model_config.json, scalers.pkl
        # We also need results.json
        (mlp_artifact_dir / "results.json").write_text("{}", encoding="utf-8")
        assert validate_artifact_directory(mlp_artifact_dir) is True

    def test_missing_model_raises(self, tmp_path: Path):
        (tmp_path / "model_config.json").write_text("{}")
        (tmp_path / "scalers.pkl").write_bytes(b"")
        (tmp_path / "results.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="No model file"):
            validate_artifact_directory(tmp_path)

    def test_missing_config_raises(self, tmp_path: Path):
        (tmp_path / "best_model.pt").write_bytes(b"")
        (tmp_path / "scalers.pkl").write_bytes(b"")
        (tmp_path / "results.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="Missing required artifact"):
            validate_artifact_directory(tmp_path)
