"""Tests for core.training_utils — to_physical_units and normalize_best_params."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from core.training_utils import to_physical_units, normalize_best_params


class TestToPhysicalUnits:

    @pytest.fixture()
    def label_scaler(self):
        rng = np.random.default_rng(0)
        y = np.log1p(rng.uniform(10, 500, (50, 3)))
        return MinMaxScaler().fit(y)

    def test_inverse_log_roundtrip(self, label_scaler):
        """log1p → scale → to_physical_units should recover original values."""
        rng = np.random.default_rng(1)
        y_raw = rng.uniform(10, 500, (10, 3))
        y_log = np.log1p(y_raw)
        y_scaled = label_scaler.transform(y_log)

        y_recovered = to_physical_units(y_scaled, label_scaler, use_log=True)
        np.testing.assert_allclose(y_recovered, y_raw, rtol=1e-5)

    def test_no_log(self, label_scaler):
        """When use_log=False, only inverse scaling is applied (no expm1)."""
        rng = np.random.default_rng(2)
        y_log = np.log1p(rng.uniform(10, 500, (5, 3)))
        y_scaled = label_scaler.transform(y_log)

        y_out = to_physical_units(y_scaled, label_scaler, use_log=False)
        np.testing.assert_allclose(y_out, y_log, rtol=1e-5)

    def test_area_root_inverse(self):
        """When use_area_root=True, Area column is squared."""
        scaler = MinMaxScaler().fit(np.log1p(np.sqrt(np.array([[100.0, 10.0], [400.0, 20.0]]))))
        y_raw = np.array([[400.0, 10.0], [900.0, 20.0]])
        y_sqrt = np.copy(y_raw)
        y_sqrt[:, 0] = np.sqrt(y_sqrt[:, 0])
        y_log = np.log1p(y_sqrt)
        y_scaled = scaler.transform(y_log)

        y_out = to_physical_units(
            y_scaled, scaler, use_log=True,
            use_area_root=True, label_cols=["Area", "Other"],
        )
        np.testing.assert_allclose(y_out, y_raw, rtol=1e-4)

    def test_none_scaler_copies(self):
        y = np.array([[1.0, 2.0]])
        out = to_physical_units(y, y_scaler=None, use_log=False)
        np.testing.assert_array_equal(out, y)
        # Must be a copy, not the same object
        out[0, 0] = 999.0
        assert y[0, 0] != 999.0


class TestNormalizeBestParams:

    def test_scaler_key_migration(self):
        raw = {"feature_scaler": "robust", "label_scaler": "minmax", "learning_rate": 0.001}
        cfg = normalize_best_params(raw)
        assert cfg["feature_scaler_type"] == "robust"
        assert cfg["label_scaler_type"] == "minmax"

    def test_defaults_filled(self):
        cfg = normalize_best_params({})
        assert cfg["plots"] is True
        assert cfg["use_log"] is True
        assert "num_epochs" in cfg
        assert "patience" in cfg

    def test_existing_keys_not_overwritten(self):
        cfg = normalize_best_params({"plots": False, "use_log": False})
        assert cfg["plots"] is False
        assert cfg["use_log"] is False
