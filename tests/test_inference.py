"""Tests for core.inference — preprocess / postprocess / make_predictions pipeline."""

from __future__ import annotations


import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from core.inference import preprocess_features, postprocess_predictions


class TestPreprocessFeatures:

    @pytest.fixture()
    def fitted_scaler(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(1, 100, (30, 4))
        return MinMaxScaler().fit(np.log1p(X))

    def test_log_and_scale(self, fitted_scaler):
        X = np.array([[10.0, 20.0, 30.0, 40.0]])
        out = preprocess_features(X, fitted_scaler, apply_log=True)
        # After log1p + MinMax, values should be roughly in [0,1]
        assert out.shape == (1, 4)
        assert np.isfinite(out).all()

    def test_no_log(self, fitted_scaler):
        X = np.array([[10.0, 20.0, 30.0, 40.0]])
        out = preprocess_features(X, fitted_scaler, apply_log=False)
        assert out.shape == (1, 4)

    def test_accepts_1d_input(self, fitted_scaler):
        X = np.array([10.0, 20.0, 30.0, 40.0])
        out = preprocess_features(X, fitted_scaler, apply_log=True)
        assert out.shape == (1, 4)

    def test_none_scaler_skips_scaling(self):
        X = np.array([[1.0, 2.0]])
        out = preprocess_features(X, feature_scaler=None, apply_log=False)
        np.testing.assert_array_equal(out, X)


class TestPostprocessPredictions:

    @pytest.fixture()
    def fitted_label_scaler(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(0, 100, (30, 2))
        return MinMaxScaler().fit(np.log1p(y))

    def test_inverse_and_expm1(self, fitted_label_scaler):
        """Full inverse pipeline: inverse_transform → expm1."""
        y_scaled = np.array([[0.5, 0.5]])
        out = postprocess_predictions(y_scaled, fitted_label_scaler, inverse_transform=True, apply_expm1=True)
        assert out.shape == (1, 2)
        assert np.isfinite(out).all()
        # Values should be in physical space (positive for log1p data)
        assert (out >= 0).all()

    def test_roundtrip_consistency(self, fitted_label_scaler):
        """Scaling then inverse-scaling should recover original log-space values."""
        rng = np.random.default_rng(7)
        y_orig = rng.uniform(1, 100, (10, 2))
        y_log = np.log1p(y_orig)
        y_scaled = fitted_label_scaler.transform(y_log)

        y_recovered = postprocess_predictions(y_scaled, fitted_label_scaler, inverse_transform=True, apply_expm1=True)
        np.testing.assert_allclose(y_recovered, y_orig, rtol=1e-5)

    def test_1d_input_reshaped(self, fitted_label_scaler):
        # Use length matching scaler's n_features (2 columns)
        y = np.array([0.3, 0.7])  # 1D with 2 elements → (1, 2)
        out = postprocess_predictions(y, fitted_label_scaler, inverse_transform=False, apply_expm1=False)
        assert out.ndim == 2

    def test_none_scaler(self):
        y = np.array([[1.0, 2.0]])
        out = postprocess_predictions(y, label_scaler=None, inverse_transform=True, apply_expm1=False)
        np.testing.assert_array_equal(out, y)
