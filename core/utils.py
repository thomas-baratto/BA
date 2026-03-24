"""Backwards-compatible re-exports from core.metrics and core.plotting."""

# Re-export metrics
from core.metrics import LABEL_UNITS, compute_regression_metrics

# Re-export plotting
from core.plotting import (
    ResourceLogger,
    create_qq_plots,
    create_regression_plots,
    create_residual_plots,
    create_scatter_plot,
    plot_results,
    plot_split_metric_bars,
)

__all__ = [
    "LABEL_UNITS",
    "compute_regression_metrics",
    "ResourceLogger",
    "create_qq_plots",
    "create_regression_plots",
    "create_residual_plots",
    "create_scatter_plot",
    "plot_results",
    "plot_split_metric_bars",
]
