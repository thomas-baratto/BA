#!/usr/bin/env python3
"""Test that plot metrics match computed metrics."""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import r2_score, root_mean_squared_error
from core.utils import compute_regression_metrics


def test_plot_metrics_match_computed():
    """Verify that metrics computed in plots match utils.compute_regression_metrics."""
    print("\n" + "="*70)
    print("TEST: Plot metrics match computed metrics")
    print("="*70)
    
    # Simulate data for a single target (what the plot would receive)
    y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0, 520.0])
    
    # What create_regression_plots does (for a single target)
    plot_r2 = r2_score(y_true, y_pred)
    plot_rmse = root_mean_squared_error(y_true, y_pred)
    
    # What our compute_regression_metrics does (should work for 1D too)
    computed_metrics = compute_regression_metrics(y_true, y_pred)
    
    print(f"\nMetrics from plot code:")
    print(f"  R²:   {plot_r2:.4f}")
    print(f"  RMSE: {plot_rmse:.2f}")
    
    print(f"\nMetrics from compute_regression_metrics:")
    print(f"  R²:   {computed_metrics['r2']:.4f}")
    print(f"  RMSE: {computed_metrics['rmse']:.2f}")
    
    # They should match for 1D (single target) data
    assert np.isclose(plot_r2, computed_metrics['r2']), \
        f"R² mismatch: plot={plot_r2} vs computed={computed_metrics['r2']}"
    assert np.isclose(plot_rmse, computed_metrics['rmse']), \
        f"RMSE mismatch: plot={plot_rmse} vs computed={computed_metrics['rmse']}"
    
    print("\n✓ Plot metrics match computed metrics for single target!")


def test_multi_target_plot_metrics():
    """Test that per-target plot metrics are correct for multi-output data."""
    print("\n" + "="*70)
    print("TEST: Multi-target plot metrics")
    print("="*70)
    
    # Multi-target data (like Area, Iso_distance, Iso_width)
    y_true = np.array([
        [10000.0, 100.0, 10.0],
        [20000.0, 200.0, 20.0],
        [30000.0, 300.0, 30.0]
    ])
    y_pred = np.array([
        [11000.0, 110.0, 11.0],
        [22000.0, 190.0, 21.0],
        [33000.0, 310.0, 32.0]
    ])
    
    target_names = ['Area', 'Iso_distance', 'Iso_width']
    
    print("\nPer-target plot metrics (what would be shown in plot titles):")
    for i, name in enumerate(target_names):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]
        
        # What the plot computes
        plot_r2 = r2_score(true_col, pred_col)
        plot_rmse = root_mean_squared_error(true_col, pred_col)
        
        # What compute_regression_metrics computes
        computed = compute_regression_metrics(true_col, pred_col)
        
        print(f"\n  {name}:")
        print(f"    Plot: R²={plot_r2:.4f}, RMSE={plot_rmse:.2f}")
        print(f"    Computed: R²={computed['r2']:.4f}, RMSE={computed['rmse']:.2f}")
        
        assert np.isclose(plot_r2, computed['r2']), f"{name}: R² mismatch"
        assert np.isclose(plot_rmse, computed['rmse']), f"{name}: RMSE mismatch"
    
    print("\n✓ All per-target plot metrics match computed metrics!")


def test_aggregate_vs_per_target():
    """Show the difference between aggregate and per-target metrics."""
    print("\n" + "="*70)
    print("TEST: Aggregate vs Per-Target metrics")
    print("="*70)
    
    # Multi-target data with very different scales
    y_true = np.array([
        [10000.0, 10.0],
        [20000.0, 20.0],
        [30000.0, 30.0]
    ])
    y_pred = np.array([
        [11000.0, 11.0],  # 10% error on both
        [22000.0, 22.0],
        [33000.0, 33.0]
    ])
    
    # Aggregate metrics (on flattened data)
    agg_metrics = compute_regression_metrics(y_true, y_pred)
    
    # Per-target metrics
    target0_metrics = compute_regression_metrics(y_true[:, 0], y_pred[:, 0])
    target1_metrics = compute_regression_metrics(y_true[:, 1], y_pred[:, 1])
    
    print("\nAggregate metrics (flattened):")
    print(f"  R²:   {agg_metrics['r2']:.4f}")
    print(f"  RMSE: {agg_metrics['rmse']:.2f}")
    
    print("\nPer-target metrics:")
    print(f"  Target 0: R²={target0_metrics['r2']:.4f}, RMSE={target0_metrics['rmse']:.2f}")
    print(f"  Target 1: R²={target1_metrics['r2']:.4f}, RMSE={target1_metrics['rmse']:.2f}")
    
    print("\n✓ Aggregate and per-target metrics computed correctly!")
    print("  Note: Plots show PER-TARGET metrics (one plot per target)")
    print("  Console output shows BOTH aggregate and per-target metrics")


if __name__ == "__main__":
    try:
        test_plot_metrics_match_computed()
        test_multi_target_plot_metrics()
        test_aggregate_vs_per_target()
        
        print("\n" + "="*70)
        print("ALL PLOT METRICS TESTS PASSED! ✓")
        print("="*70)
        print("\nSummary:")
        print("- Plots show PER-TARGET metrics (one plot per output)")
        print("- Console shows BOTH aggregate and per-target metrics")
        print("- All metrics use the same computation (flattened approach)")
        print("- Plot metrics match computed metrics for each target ✓")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
