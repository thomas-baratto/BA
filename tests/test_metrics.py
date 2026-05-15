"""Tests for core.metrics — regression metric computation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import skew, kurtosis
from sklearn.metrics import r2_score

from core.metrics import compute_regression_metrics, LABEL_UNITS


# ── Metric correctness ──────────────────────────────────────────────────────

class TestMetricCorrectness:
    """Verify each metric against manual / sklearn reference values."""

    def test_perfect_predictions(self):
        y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        m = compute_regression_metrics(y, y)
        assert m["mae"] == pytest.approx(0.0, abs=1e-10)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert m["r2"] == pytest.approx(1.0, abs=1e-10)
        assert m["mape"] == pytest.approx(0.0, abs=1e-10)

    def test_basic_mae_mse_rmse(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])

        m = compute_regression_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx(0.5, abs=1e-10)
        assert m["mse"] == pytest.approx(0.25, abs=1e-10)
        assert m["rmse"] == pytest.approx(0.5, abs=1e-10)

    def test_r2_matches_sklearn(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        m = compute_regression_metrics(y_true, y_pred)
        assert m["r2"] == pytest.approx(r2_score(y_true, y_pred), abs=1e-10)

    def test_mape_excludes_zeros(self):
        """MAPE should skip samples where y_true == 0."""
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([10.0, 110.0, 180.0])

        m = compute_regression_metrics(y_true, y_pred)
        mask = y_true != 0
        expected = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]))
        assert m["mape"] == pytest.approx(expected, abs=1e-10)

    def test_max_error(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.0, 3.0, 10.0])
        m = compute_regression_metrics(y_true, y_pred)
        assert m["max_error"] == pytest.approx(6.0, abs=1e-10)

    def test_median_absolute_error(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 150.0])
        m = compute_regression_metrics(y_true, y_pred)
        assert m["medae"] == pytest.approx(0.0, abs=1e-10)
        # MAE should be larger than median because of outlier
        assert m["mae"] > m["medae"]

    def test_residual_statistics(self):
        y_true = np.arange(1.0, 11.0)
        y_pred = y_true + np.array([0.5, -0.2, 0.2, 0.1, -0.1, 0.3, -0.2, 0.5, -0.3, 0.2])

        m = compute_regression_metrics(y_true, y_pred)
        residuals = y_pred - y_true

        assert m["residual_mean"] == pytest.approx(np.mean(residuals), abs=1e-10)
        assert m["residual_std"] == pytest.approx(np.std(residuals), abs=1e-10)
        assert m["residual_skew"] == pytest.approx(float(skew(residuals)), abs=1e-6)
        assert m["residual_kurtosis"] == pytest.approx(float(kurtosis(residuals)), abs=1e-6)
        assert m["residual_p95"] == pytest.approx(np.percentile(residuals, 95), abs=1e-10)
        assert m["residual_p99"] == pytest.approx(np.percentile(residuals, 99), abs=1e-10)

    def test_nrmse_normalized_by_range(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 21.0, 31.0])
        m = compute_regression_metrics(y_true, y_pred)
        expected_nrmse = m["rmse"] / (30.0 - 10.0)
        assert m["nrmse"] == pytest.approx(expected_nrmse, abs=1e-10)

    def test_kge_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = compute_regression_metrics(y, y)
        assert m["kge"] == pytest.approx(1.0, abs=1e-10)


# ── Multi-output behaviour ──────────────────────────────────────────────────

class TestMultiOutput:
    """Metrics are computed on flattened arrays, not averaged per-output."""

    def test_multi_output_flattened(self):
        y_true = np.array([[10.0, 1000.0], [20.0, 2000.0], [30.0, 3000.0]])
        y_pred = np.array([[11.0, 1100.0], [21.0, 2100.0], [31.0, 3100.0]])

        m = compute_regression_metrics(y_true, y_pred)

        flat_true = y_true.flatten()
        flat_pred = y_pred.flatten()
        expected_mae = np.mean(np.abs(flat_pred - flat_true))
        expected_r2 = r2_score(flat_true, flat_pred)

        assert m["mae"] == pytest.approx(expected_mae, abs=1e-10)
        assert m["r2"] == pytest.approx(expected_r2, abs=1e-10)


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestMetricEdgeCases:

    def test_single_sample(self):
        m = compute_regression_metrics(np.array([[1.0]]), np.array([[1.5]]))
        assert "mae" in m
        assert "rmse" in m
        # R² and explained_variance are undefined for < 2 samples
        assert np.isnan(m["r2"])
        assert np.isnan(m["explained_variance"])

    def test_all_true_values_zero_gives_nan_mape(self):
        m = compute_regression_metrics(np.array([0.0, 0.0]), np.array([1.0, 2.0]))
        assert np.isnan(m["mape"])
        assert np.isnan(m["rel_err_mean_abs"])

    def test_negative_values(self):
        m = compute_regression_metrics(np.array([-1.0, -2.0]), np.array([-1.1, -2.1]))
        assert m["mae"] > 0

    def test_large_values(self):
        m = compute_regression_metrics(np.array([1e6, 2e6]), np.array([1.1e6, 2.1e6]))
        assert np.isfinite(m["rmse"])

    def test_all_expected_keys_present(self):
        m = compute_regression_metrics(np.arange(5.0), np.arange(5.0) + 0.1)
        expected = {
            "mae", "mse", "rmse", "r2", "medae", "explained_variance",
            "max_error", "mape", "nrmse", "kge",
            "rel_err_mean_abs", "rel_err_std",
            "residual_mean", "residual_std", "residual_skew",
            "residual_kurtosis", "residual_p95", "residual_p99",
        }
        assert expected.issubset(m.keys())


# ── LABEL_UNITS mapping ─────────────────────────────────────────────────────

class TestLabelUnits:

    def test_known_labels_have_units(self):
        for label in ("Area", "Iso_distance", "Iso_width", "Cone"):
            assert label in LABEL_UNITS
