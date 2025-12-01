"""
Unit tests for utils.py
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
import tempfile
import os

from core.utils import compute_regression_metrics, create_scatter_plot, create_residual_plots, create_qq_plots


class TestComputeRegressionMetrics:
    """Tests for compute_regression_metrics function."""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        assert metrics['mae'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['rmse'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['mse'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['r2'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['medae'] == pytest.approx(0.0, abs=1e-6)
    
    def test_basic_metrics(self):
        """Test basic metric calculations."""
        y_true = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_pred = np.array([[1.5], [2.5], [3.5], [4.5]])
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        # MAE should be 0.5
        assert metrics['mae'] == pytest.approx(0.5, abs=1e-6)
        # MSE should be 0.25
        assert metrics['mse'] == pytest.approx(0.25, abs=1e-6)
        # RMSE should be 0.5
        assert metrics['rmse'] == pytest.approx(0.5, abs=1e-6)
    
    def test_multidimensional_output(self):
        """Test metrics with multiple output dimensions."""
        y_true = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        y_pred = np.array([[1.1, 11.0], [2.2, 22.0], [3.3, 33.0]])
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mse' in metrics
        assert metrics['mae'] > 0
    
    def test_residual_statistics(self):
        """Test that residual statistics are calculated."""
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = np.array([[1.2], [1.8], [3.1], [4.2], [4.9]])
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        assert 'residual_mean' in metrics
        assert 'residual_std' in metrics
        assert 'residual_skew' in metrics
        assert 'residual_kurtosis' in metrics
        assert 'residual_p95' in metrics
        assert 'residual_p99' in metrics
    
    def test_explained_variance(self):
        """Test explained variance calculation."""
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = np.array([[1.1], [2.1], [3.1], [4.1], [5.1]])
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        assert 'explained_variance' in metrics
        assert metrics['explained_variance'] > 0.9  # Should be high for good predictions
    
    def test_max_error(self):
        """Test max error calculation."""
        y_true = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_pred = np.array([[1.1], [2.0], [3.0], [10.0]])  # Last one has large error
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        assert 'max_error' in metrics
        assert metrics['max_error'] == pytest.approx(6.0, abs=1e-6)
    
    def test_relative_error_with_zeros(self):
        """Test relative error calculation when true values contain zeros."""
        y_true = np.array([[0.0], [2.0], [3.0]])
        y_pred = np.array([[1.0], [2.5], [3.5]])
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        # Should handle zeros gracefully
        assert 'rel_err_mean_abs' in metrics
        assert 'rel_err_std' in metrics
    
    def test_all_zeros(self):
        """Test when all true values are zero."""
        y_true = np.array([[0.0], [0.0], [0.0]])
        y_pred = np.array([[1.0], [2.0], [3.0]])
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        # Relative error metrics should be NaN
        assert np.isnan(metrics['rel_err_mean_abs'])
        assert np.isnan(metrics['rel_err_std'])


class TestPlottingFunctions:
    """Tests for plotting functions."""
    
    def test_create_scatter_plot(self):
        """Test scatter plot creation."""
        y_true = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        y_pred = np.array([[1.1, 11.0], [2.2, 22.0], [3.3, 33.0]])
        labels = ['Feature1', 'Feature2']
        
        figures = create_scatter_plot(y_true, y_pred, labels)
        
        assert len(figures) == 2
        assert figures[0][0] == 'True_vs_Predicted/Label_Feature1'
        assert figures[1][0] == 'True_vs_Predicted/Label_Feature2'
        
        # Close figures
        for _, fig in figures:
            plt.close(fig)
    
    def test_create_scatter_plot_single_feature(self):
        """Test scatter plot with single feature."""
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.1], [2.2], [3.3]])
        labels = ['Feature1']
        
        figures = create_scatter_plot(y_true, y_pred, labels)
        
        assert len(figures) == 1
        plt.close(figures[0][1])
    
    def test_create_residual_plots(self):
        """Test residual plot creation."""
        y_true = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        y_pred = np.array([[1.1, 11.0], [2.2, 22.0], [3.3, 33.0]])
        labels = ['Feature1', 'Feature2']
        
        figures = create_residual_plots(y_true, y_pred, labels)
        
        assert len(figures) == 2
        assert 'Residuals_vs_True' in figures[0][0]
        assert 'Residuals_vs_True' in figures[1][0]
        
        # Close figures
        for _, fig in figures:
            plt.close(fig)
    
    def test_create_qq_plots(self):
        """Test Q-Q plot creation."""
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = np.array([[1.1], [2.1], [3.1], [4.1], [5.1]])
        labels = ['Feature1']
        
        figures = create_qq_plots(y_true, y_pred, labels)
        
        assert len(figures) == 1
        assert 'QQ_Plot' in figures[0][0]
        
        # Close figure
        plt.close(figures[0][1])
    
    def test_1d_arrays_reshaped(self):
        """Test that 1D arrays are properly reshaped."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 3.3])
        labels = ['Feature1']
        
        figures = create_scatter_plot(y_true, y_pred, labels)
        
        assert len(figures) == 1
        plt.close(figures[0][1])


class TestMetricEdgeCases:
    """Tests for edge cases in metrics calculation."""
    
    def test_single_sample(self):
        """Test metrics with single sample."""
        y_true = np.array([[1.0]])
        y_pred = np.array([[1.5]])
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        # Should not crash, but some metrics may be NaN
        assert 'mae' in metrics
        assert 'rmse' in metrics
    
    def test_negative_values(self):
        """Test metrics with negative values."""
        y_true = np.array([[-1.0], [-2.0], [-3.0]])
        y_pred = np.array([[-1.1], [-2.1], [-3.1]])
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
    
    def test_large_values(self):
        """Test metrics with large values."""
        y_true = np.array([[1e6], [2e6], [3e6]])
        y_pred = np.array([[1.1e6], [2.1e6], [3.1e6]])
        
        metrics = compute_regression_metrics(y_true, y_pred)
        
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
        assert not np.isinf(metrics['mae'])
        assert not np.isinf(metrics['rmse'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
