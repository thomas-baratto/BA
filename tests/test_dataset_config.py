"""Tests for config/datasets.py — dataset configuration single source of truth."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.datasets import (
    DATASET_CONFIGS,
    KNOWN_DATASETS,
    KNOWN_FEATURES,
    KNOWN_LABELS,
    get_dataset_config,
    detect_features_and_labels,
)


class TestDatasetConfigs:
    """DATASET_CONFIGS must be complete and self-consistent."""

    def test_isotherm_and_cone_present(self):
        assert "isotherm" in DATASET_CONFIGS
        assert "cone" in DATASET_CONFIGS

    @pytest.mark.parametrize("name", ["isotherm", "cone"])
    def test_required_keys(self, name: str):
        cfg = DATASET_CONFIGS[name]
        for key in ("csv_file", "features", "labels"):
            assert key in cfg, f"Missing key '{key}' in {name} config"

    def test_isotherm_features_and_labels(self):
        cfg = DATASET_CONFIGS["isotherm"]
        assert len(cfg["features"]) == 9
        assert cfg["labels"] == ["Area", "Iso_distance", "Iso_width"]

    def test_cone_features_and_labels(self):
        cfg = DATASET_CONFIGS["cone"]
        assert len(cfg["features"]) == 4
        assert cfg["labels"] == ["Cone"]

    def test_no_feature_label_overlap(self):
        """Features and labels within each dataset should not overlap."""
        for name, cfg in DATASET_CONFIGS.items():
            overlap = set(cfg["features"]) & set(cfg["labels"])
            assert not overlap, f"{name}: overlap between features and labels: {overlap}"


class TestGetDatasetConfig:

    def test_returns_copy(self):
        cfg1 = get_dataset_config("cone")
        cfg2 = get_dataset_config("cone")
        cfg1["labels"] = ["MODIFIED"]
        assert cfg2["labels"] != ["MODIFIED"], "get_dataset_config should return a copy"

    def test_raises_for_unknown(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_config("nonexistent_dataset")


class TestDetectFeaturesAndLabels:
    """detect_features_and_labels should infer columns from a CSV header."""

    def test_isotherm_csv(self, isotherm_csv: Path):
        features, labels = detect_features_and_labels(str(isotherm_csv))
        assert "Flow_well" in features
        assert "Area" in labels and "Iso_width" in labels

    def test_cone_csv(self, cone_csv: Path):
        features, labels = detect_features_and_labels(str(cone_csv))
        assert "Hydr_conductivity" in features
        assert "Cone" in labels


class TestKnownSets:
    """KNOWN_FEATURES and KNOWN_LABELS should be union of all dataset configs."""

    def test_known_features_superset(self):
        for cfg in DATASET_CONFIGS.values():
            for f in cfg["features"]:
                assert f in KNOWN_FEATURES

    def test_known_labels_superset(self):
        for cfg in DATASET_CONFIGS.values():
            for l in cfg["labels"]:
                assert l in KNOWN_LABELS

    def test_known_datasets(self):
        assert KNOWN_DATASETS == set(DATASET_CONFIGS.keys())
