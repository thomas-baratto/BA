#!/usr/bin/env python3
"""Generate training result plots from saved .npz data.

Reads training_curves.npz and prediction_data.npz from model artifact
directories and produces:
  - Loss curve (train + validation, log scale)
  - Q-Q plots of residuals (one per label)
  - Per-label metric bar charts (from stats/metrics_summary.json)

Scatter and residual plots are already handled by the model_comparison
pipeline (generate_model_comparison.py), so they are not duplicated here.

Usage:
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/plot_training_curves.py \
        --model-dir artifacts/models/mlp/cone \
        --output-dir docs/plots/training/cone
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.metrics import LABEL_UNITS
from core.thesis_style import (
    apply_thesis_style,
    COLORS,
    FIG_SINGLE,
    FIG_WIDE,
    label_with_unit,
    save_fig,
)

apply_thesis_style()


def _find_plot_subdir(model_dir: Path) -> Path:
    """Find the plots/<label_str>/ subdirectory that contains .npz files."""
    plots_dir = model_dir / "plots"
    if not plots_dir.exists():
        raise FileNotFoundError(f"No plots/ directory in {model_dir}")
    for sub in sorted(plots_dir.iterdir()):
        if sub.is_dir() and list(sub.glob("*.npz")):
            return sub
    raise FileNotFoundError(f"No subdirectory with .npz files in {plots_dir}")


# ── Loss curve ──────────────────────────────────────────────────────────────


def plot_loss_curve(data_dir: Path, output_dir: Path) -> None:
    """Plot training and validation loss curves."""
    npz_path = data_dir / "training_curves.npz"
    if not npz_path.exists():
        print("  Skipping loss curve (training_curves.npz not found)")
        return

    data = np.load(npz_path, allow_pickle=True)
    epochs = data["epochs"]
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]
    epoch_times = data["epoch_times"]
    loss_fn_name = str(data["loss_fn_name"])

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.semilogy(epochs, train_losses, label="Training", color=COLORS["primary"])
    ax.semilogy(epochs, val_losses, label="Validation", color=COLORS["accent1"], ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_fn_name)
    ax.legend()

    # Secondary x-axis for wall-clock time
    if len(epoch_times) == len(epochs):
        epoch_mins = np.array(epoch_times) / 60.0

        def _epoch_to_min(ep):
            return np.interp(ep, epochs, epoch_mins)

        def _min_to_epoch(mn):
            return np.interp(mn, epoch_mins, epochs)

        ax2 = ax.secondary_xaxis("top", functions=(_epoch_to_min, _min_to_epoch))
        ax2.set_xlabel("Time [min]")

    fig.tight_layout()
    save_fig(fig, output_dir / "loss")
    print("  Saved: loss.pdf")


# ── Q-Q plots ──────────────────────────────────────────────────────────────


def plot_qq(data_dir: Path, output_dir: Path) -> None:
    """Plot Q-Q residual plots for each label."""
    npz_path = data_dir / "prediction_data.npz"
    if not npz_path.exists():
        print("  Skipping Q-Q plots (prediction_data.npz not found)")
        return

    data = np.load(npz_path, allow_pickle=True)
    test_true = data["test_true"]
    test_pred = data["test_pred"]
    label_cols = list(data["label_cols"])

    if test_true.ndim == 1:
        test_true = test_true.reshape(-1, 1)
    if test_pred.ndim == 1:
        test_pred = test_pred.reshape(-1, 1)

    for i, lbl in enumerate(label_cols):
        residuals = test_pred[:, i] - test_true[:, i]
        fig = sm.qqplot(residuals, line="45", fit=True, markersize=3)
        fig.tight_layout()
        save_fig(fig, output_dir / f"qq_plot_{lbl}")
        print(f"  Saved: qq_plot_{lbl}.pdf")


# ── Metric bars ─────────────────────────────────────────────────────────────


def plot_metric_bars(model_dir: Path, output_dir: Path) -> None:
    """Plot per-label metric comparison (validation vs test)."""
    metrics_path = model_dir / "stats" / "metrics_summary.json"
    if not metrics_path.exists():
        print("  Skipping metric bars (stats/metrics_summary.json not found)")
        return

    with open(metrics_path) as f:
        split_metrics = json.load(f)

    splits = list(split_metrics.keys())
    if not splits:
        return

    # Determine labels from the first split's per_label keys
    first_split = split_metrics[splits[0]]
    labels = list(first_split.get("per_label", {}).keys())
    if not labels:
        return

    x = np.arange(len(labels))
    bar_width = 0.75 / max(1, len(splits))

    for metric in ("mae", "rmse"):
        fig, ax = plt.subplots(figsize=FIG_WIDE)
        for idx, split in enumerate(splits):
            per_label_values = [
                split_metrics[split]["per_label"]
                .get(label, {})
                .get(metric, float("nan"))
                for label in labels
            ]
            offset = (idx - (len(splits) - 1) / 2) * bar_width
            ax.bar(x + offset, per_label_values, width=bar_width, label=split.title())

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        unit = LABEL_UNITS.get(labels[0], "") if labels else ""
        unit_str = f" [{unit}]" if unit else ""
        ax.set_ylabel(f"{metric.upper()}{unit_str}")
        ax.legend()
        fig.tight_layout()
        save_fig(fig, output_dir / f"{metric}_per_label")
        print(f"  Saved: {metric}_per_label.pdf")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training results from saved data")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to model artifact directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for plots")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = _find_plot_subdir(model_dir)

    plot_loss_curve(data_dir, output_dir)
    plot_qq(data_dir, output_dir)
    plot_metric_bars(model_dir, output_dir)

    print(f"\nAll training plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
