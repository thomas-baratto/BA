"""Tests for core.plotting — scatter, residual, and Q-Q plot generation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from core.plotting import create_scatter_plot, create_residual_plots, create_qq_plots


class TestScatterPlot:

    def test_produces_one_figure_per_label(self):
        y_true = np.random.randn(20, 3)
        y_pred = y_true + 0.1
        labels = ["A", "B", "C"]
        figs = create_scatter_plot(y_true, y_pred, labels)
        assert len(figs) == 3
        for name, fig in figs:
            assert "True_vs_Predicted" in name
            plt.close(fig)

    def test_single_label(self):
        figs = create_scatter_plot(
            np.random.randn(10, 1), np.random.randn(10, 1), ["X"]
        )
        assert len(figs) == 1
        plt.close(figs[0][1])

    def test_1d_input_reshaped(self):
        figs = create_scatter_plot(
            np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 3.1]), ["F"]
        )
        assert len(figs) == 1
        plt.close(figs[0][1])


class TestResidualPlots:

    def test_produces_figures(self):
        figs = create_residual_plots(
            np.random.randn(15, 2), np.random.randn(15, 2), ["L1", "L2"]
        )
        assert len(figs) == 2
        for name, fig in figs:
            assert "Residuals_vs_True" in name
            plt.close(fig)


class TestQQPlots:

    def test_produces_figures(self):
        figs = create_qq_plots(
            np.random.randn(20, 1), np.random.randn(20, 1), ["Z"]
        )
        assert len(figs) == 1
        assert "QQ_Plot" in figs[0][0]
        plt.close(figs[0][1])
