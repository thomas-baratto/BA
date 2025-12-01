#!/usr/bin/env python3
"""Tests for metrics computation to ensure correctness."""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import compute_regression_metrics
from sklearn.metrics import r2_score


def test_single_output_metrics():
    """Test metrics on single-output (1D) regression."""
    print("\n" + "="*70)
    print("TEST 1: Single-output (1D) regression")
    print("="*70)
    
    # Create simple test data
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    
    metrics = compute_regression_metrics(y_true, y_pred)
    
    # Expected values (computed manually)
    expected_mae = np.mean(np.abs(y_pred - y_true))  # 0.14
    expected_mse = np.mean((y_pred - y_true) ** 2)   # 0.026
    expected_rmse = np.sqrt(expected_mse)            # 0.1612
    expected_r2 = r2_score(y_true, y_pred)
    
    print(f"MAE:  {metrics['mae']:.4f} (expected: {expected_mae:.4f})")
    print(f"MSE:  {metrics['mse']:.4f} (expected: {expected_mse:.4f})")
    print(f"RMSE: {metrics['rmse']:.4f} (expected: {expected_rmse:.4f})")
    print(f"R²:   {metrics['r2']:.4f} (expected: {expected_r2:.4f})")
    
    # Assertions
    assert np.isclose(metrics['mae'], expected_mae), f"MAE mismatch: {metrics['mae']} != {expected_mae}"
    assert np.isclose(metrics['mse'], expected_mse), f"MSE mismatch: {metrics['mse']} != {expected_mse}"
    assert np.isclose(metrics['rmse'], expected_rmse), f"RMSE mismatch: {metrics['rmse']} != {expected_rmse}"
    assert np.isclose(metrics['r2'], expected_r2), f"R² mismatch: {metrics['r2']} != {expected_r2}"
    
    print("✓ All single-output metrics correct!")


def test_multi_output_metrics():
    """Test metrics on multi-output (2D) regression - should flatten first."""
    print("\n" + "="*70)
    print("TEST 2: Multi-output (2D) regression - flattened metrics")
    print("="*70)
    
    # Create multi-output test data with very different scales
    # This will show the difference between flattening and averaging
    y_true = np.array([
        [10.0, 1000.0],
        [20.0, 2000.0],
        [30.0, 3000.0]
    ])
    y_pred = np.array([
        [11.0, 1100.0],  # Error: 1 vs 100
        [21.0, 2100.0],  # Error: 1 vs 100
        [31.0, 3100.0]   # Error: 1 vs 100
    ])
    
    metrics = compute_regression_metrics(y_true, y_pred)
    
    # Expected: compute on flattened arrays
    y_true_flat = y_true.flatten()  # [1, 10, 2, 20, 3, 30]
    y_pred_flat = y_pred.flatten()  # [1.1, 10.5, 2.2, 19.5, 2.8, 30.2]
    
    expected_mae = np.mean(np.abs(y_pred_flat - y_true_flat))
    expected_mse = np.mean((y_pred_flat - y_true_flat) ** 2)
    expected_rmse = np.sqrt(expected_mse)
    expected_r2 = r2_score(y_true_flat, y_pred_flat)
    
    print(f"MAE:  {metrics['mae']:.4f} (expected: {expected_mae:.4f})")
    print(f"MSE:  {metrics['mse']:.4f} (expected: {expected_mse:.4f})")
    print(f"RMSE: {metrics['rmse']:.4f} (expected: {expected_rmse:.4f})")
    print(f"R²:   {metrics['r2']:.4f} (expected: {expected_r2:.4f})")
    
    # What we DON'T want: averaging per-output RMSEs (this is what sklearn does by default)
    rmse_per_output = [np.sqrt(np.mean((y_pred[:, i] - y_true[:, i]) ** 2)) for i in range(2)]
    wrong_rmse = np.mean(rmse_per_output)  # Average of RMSEs
    correct_rmse = metrics['rmse']  # RMSE of flattened
    
    print(f"\nRMSE comparison:")
    print(f"  Per-output RMSEs: {rmse_per_output}")
    print(f"  WRONG approach (averaging RMSEs): {wrong_rmse:.4f}")
    print(f"  Our approach (flatten first, then RMSE): {correct_rmse:.4f}")
    
    # Assertions
    assert np.isclose(metrics['mae'], expected_mae), f"MAE mismatch: {metrics['mae']} != {expected_mae}"
    assert np.isclose(metrics['mse'], expected_mse), f"MSE mismatch: {metrics['mse']} != {expected_mse}"
    assert np.isclose(metrics['rmse'], expected_rmse), f"RMSE mismatch: {metrics['rmse']} != {expected_rmse}"
    assert np.isclose(metrics['r2'], expected_r2), f"R² mismatch: {metrics['r2']} != {expected_r2}"
    
    # Verify RMSE is different from averaging approach (this is the key difference!)
    assert not np.isclose(correct_rmse, wrong_rmse), f"RMSE should differ from averaging approach! {correct_rmse} vs {wrong_rmse}"
    
    print("✓ All multi-output metrics correct (flattened)!")


def test_multi_output_different_scales():
    """Test with outputs at very different scales (like Area vs Iso_width)."""
    print("\n" + "="*70)
    print("TEST 3: Multi-output with different scales (like Area, Iso_distance, Iso_width)")
    print("="*70)
    
    # Simulate Area (large values) and Iso_width (small values)
    y_true = np.array([
        [10000.0, 10.0],  # Area, Iso_width
        [20000.0, 20.0],
        [30000.0, 30.0]
    ])
    y_pred = np.array([
        [11000.0, 11.0],  # 10% error on both
        [22000.0, 22.0],
        [33000.0, 33.0]
    ])
    
    metrics = compute_regression_metrics(y_true, y_pred)
    
    # Compute per-output metrics for comparison
    mae_area = np.mean(np.abs(y_pred[:, 0] - y_true[:, 0]))
    mae_width = np.mean(np.abs(y_pred[:, 1] - y_true[:, 1]))
    
    print(f"Per-output MAE:")
    print(f"  Area (col 0):  {mae_area:.2f}")
    print(f"  Width (col 1): {mae_width:.2f}")
    print(f"  Average of two: {(mae_area + mae_width) / 2:.2f}")
    
    print(f"\nAggregate MAE (flattened): {metrics['mae']:.2f}")
    
    # The flattened MAE should be dominated by Area (large values)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    expected_mae = np.mean(np.abs(y_pred_flat - y_true_flat))
    
    print(f"Expected (manual): {expected_mae:.2f}")
    
    assert np.isclose(metrics['mae'], expected_mae), f"MAE mismatch"
    
    print("✓ Multi-scale metrics correct!")


def test_mape_computation():
    """Test MAPE (Mean Absolute Percentage Error) computation."""
    print("\n" + "="*70)
    print("TEST 4: MAPE computation")
    print("="*70)
    
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 180.0, 330.0])
    
    metrics = compute_regression_metrics(y_true, y_pred)
    
    # MAPE = mean(|y_pred - y_true| / |y_true|)
    expected_mape = np.mean(np.abs((y_pred - y_true) / y_true))
    # (10/100 + 20/200 + 30/300) / 3 = (0.1 + 0.1 + 0.1) / 3 = 0.1
    
    print(f"MAPE: {metrics['mape']:.4f} (expected: {expected_mape:.4f})")
    print(f"MAPE as percentage: {metrics['mape']*100:.2f}%")
    
    assert np.isclose(metrics['mape'], expected_mape), f"MAPE mismatch"
    
    print("✓ MAPE computation correct!")


def test_zero_handling():
    """Test handling of zero values in MAPE computation."""
    print("\n" + "="*70)
    print("TEST 5: Zero value handling")
    print("="*70)
    
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([10.0, 110.0, 180.0])
    
    metrics = compute_regression_metrics(y_true, y_pred)
    
    # MAPE should skip zero values
    mask = y_true != 0
    expected_mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]))
    
    print(f"MAPE (skipping zeros): {metrics['mape']:.4f} (expected: {expected_mape:.4f})")
    
    assert np.isclose(metrics['mape'], expected_mape), f"MAPE mismatch with zeros"
    
    print("✓ Zero handling correct!")


def test_all_metrics_present():
    """Test that all expected metrics are computed and returned."""
    print("\n" + "="*70)
    print("TEST 6: All metrics present")
    print("="*70)
    
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    
    metrics = compute_regression_metrics(y_true, y_pred)
    
    expected_keys = [
        'mae', 'mse', 'rmse', 'r2', 'medae', 'explained_variance',
        'max_error', 'mape', 'rel_err_std', 'rel_err_mean_abs',
        'residual_mean', 'residual_std', 'residual_skew', 'residual_kurtosis',
        'residual_p95', 'residual_p99'
    ]
    
    print(f"Expected {len(expected_keys)} metrics:")
    for key in expected_keys:
        present = key in metrics
        status = "✓" if present else "✗"
        value = metrics.get(key, "MISSING")
        print(f"  {status} {key}: {value}")
        assert key in metrics, f"Missing metric: {key}"
    
    print(f"\n✓ All {len(expected_keys)} metrics present!")


def test_median_and_max_error():
    """Test median absolute error and max error."""
    print("\n" + "="*70)
    print("TEST 7: Median AE and Max Error")
    print("="*70)
    
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # Include outlier
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 150.0])  # Large error on outlier
    
    metrics = compute_regression_metrics(y_true, y_pred)
    
    errors = np.abs(y_pred - y_true)  # [0, 0, 0, 0, 50]
    expected_medae = np.median(errors)  # 0.0
    expected_max_error = np.max(errors)  # 50.0
    
    print(f"Errors: {errors}")
    print(f"Median AE: {metrics['medae']:.2f} (expected: {expected_medae:.2f})")
    print(f"Max Error: {metrics['max_error']:.2f} (expected: {expected_max_error:.2f})")
    
    assert np.isclose(metrics['medae'], expected_medae), "Median AE mismatch"
    assert np.isclose(metrics['max_error'], expected_max_error), "Max error mismatch"
    
    # Median should be much smaller than mean when there's an outlier
    print(f"MAE: {metrics['mae']:.2f} (should be larger than median due to outlier)")
    assert metrics['mae'] > metrics['medae'], "MAE should be larger than median with outlier"
    
    print("✓ Median AE and Max Error correct!")


def test_explained_variance():
    """Test explained variance score."""
    print("\n" + "="*70)
    print("TEST 8: Explained Variance")
    print("="*70)
    
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Perfect predictions
    
    metrics = compute_regression_metrics(y_true, y_pred)
    
    print(f"Perfect predictions:")
    print(f"  Explained Variance: {metrics['explained_variance']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    
    assert np.isclose(metrics['explained_variance'], 1.0), "Should be 1.0 for perfect predictions"
    assert np.isclose(metrics['r2'], 1.0), "R² should also be 1.0"
    
    print("✓ Explained variance correct!")


def test_relative_error_stats():
    """Test relative error statistics."""
    print("\n" + "="*70)
    print("TEST 9: Relative Error Statistics")
    print("="*70)
    
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])  # +10%, -5%, +10%
    
    metrics = compute_regression_metrics(y_true, y_pred)
    
    # Relative errors: 0.1, -0.05, 0.1
    rel_errors = (y_pred - y_true) / y_true
    expected_rel_err_mean_abs = np.mean(np.abs(rel_errors))
    expected_rel_err_std = np.std(rel_errors)
    
    print(f"Relative errors: {rel_errors}")
    print(f"Mean absolute rel error: {metrics['rel_err_mean_abs']:.4f} (expected: {expected_rel_err_mean_abs:.4f})")
    print(f"Rel error std: {metrics['rel_err_std']:.4f} (expected: {expected_rel_err_std:.4f})")
    
    assert np.isclose(metrics['rel_err_mean_abs'], expected_rel_err_mean_abs), "Rel err mean abs mismatch"
    assert np.isclose(metrics['rel_err_std'], expected_rel_err_std), "Rel err std mismatch"
    
    # Note: rel_err_mean_abs should equal MAPE
    assert np.isclose(metrics['rel_err_mean_abs'], metrics['mape']), "rel_err_mean_abs should equal MAPE"
    
    print("✓ Relative error statistics correct!")


def test_residual_statistics():
    """Test residual statistics (mean, std, skew, kurtosis, percentiles)."""
    print("\n" + "="*70)
    print("TEST 10: Residual Statistics")
    print("="*70)
    
    from scipy.stats import skew, kurtosis
    
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_pred = np.array([1.5, 1.8, 3.2, 4.1, 4.9, 6.3, 6.8, 8.5, 8.7, 10.2])
    
    metrics = compute_regression_metrics(y_true, y_pred)
    
    residuals = y_pred - y_true
    expected_residual_mean = np.mean(residuals)
    expected_residual_std = np.std(residuals)
    expected_residual_skew = skew(residuals)
    expected_residual_kurtosis = kurtosis(residuals)
    expected_p95 = np.percentile(residuals, 95)
    expected_p99 = np.percentile(residuals, 99)
    
    print(f"Residuals: {residuals}")
    print(f"Residual mean: {metrics['residual_mean']:.4f} (expected: {expected_residual_mean:.4f})")
    print(f"Residual std: {metrics['residual_std']:.4f} (expected: {expected_residual_std:.4f})")
    print(f"Residual skew: {metrics['residual_skew']:.4f} (expected: {expected_residual_skew:.4f})")
    print(f"Residual kurtosis: {metrics['residual_kurtosis']:.4f} (expected: {expected_residual_kurtosis:.4f})")
    print(f"Residual 95th percentile: {metrics['residual_p95']:.4f} (expected: {expected_p95:.4f})")
    print(f"Residual 99th percentile: {metrics['residual_p99']:.4f} (expected: {expected_p99:.4f})")
    
    assert np.isclose(metrics['residual_mean'], expected_residual_mean), "Residual mean mismatch"
    assert np.isclose(metrics['residual_std'], expected_residual_std), "Residual std mismatch"
    assert np.isclose(metrics['residual_skew'], expected_residual_skew), "Residual skew mismatch"
    assert np.isclose(metrics['residual_kurtosis'], expected_residual_kurtosis), "Residual kurtosis mismatch"
    assert np.isclose(metrics['residual_p95'], expected_p95), "Residual p95 mismatch"
    assert np.isclose(metrics['residual_p99'], expected_p99), "Residual p99 mismatch"
    
    print("✓ All residual statistics correct!")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "="*70)
    print("RUNNING METRICS TESTS")
    print("="*70)
    
    try:
        test_single_output_metrics()
        test_multi_output_metrics()
        test_multi_output_different_scales()
        test_mape_computation()
        test_zero_handling()
        test_all_metrics_present()
        test_median_and_max_error()
        test_explained_variance()
        test_relative_error_stats()
        test_residual_statistics()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
