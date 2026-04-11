#!/usr/bin/env python3
"""Generate label distribution histograms from saved .npz data.

Reads data_before_transform.npz, data_after_log.npz, and data_after_scaling.npz
from model artifact directories and produces thesis-styled histogram plots.

Usage:
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/plot_distributions.py \
        --model-dir artifacts/models/mlp/cone \
        --output-dir docs/plots/distributions/cone
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.thesis_style import (
    apply_thesis_style,
    COLORS,
    FIG_SINGLE,
    label_with_unit,
    save_fig,
)

apply_thesis_style()

# Colour cycle for multiple labels on the same histogram
_HIST_COLORS = [COLORS["primary"], COLORS["accent1"], COLORS["accent3"]]


def _find_plot_subdir(model_dir: Path) -> Path:
    """Find the plots/<label_str>/ subdirectory that contains .npz files."""
    plots_dir = model_dir / "plots"
    if not plots_dir.exists():
        raise FileNotFoundError(f"No plots/ directory in {model_dir}")
    # Find the first subdirectory containing .npz files
    for sub in sorted(plots_dir.iterdir()):
        if sub.is_dir() and list(sub.glob("data_*.npz")):
            return sub
    raise FileNotFoundError(f"No subdirectory with data_*.npz in {plots_dir}")


def plot_distribution(
    npz_path: Path,
    x_label: str,
    out_path: Path,
) -> None:
    """Plot a histogram from a saved .npz file."""
    data = np.load(npz_path, allow_pickle=True)

    # Determine the array key: 'y' for before/after_log, 'y_train' for after_scaling
    if "y_train" in data:
        y = data["y_train"]
    else:
        y = data["y"]

    label_cols = list(data["label_cols"])

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for i, lbl in enumerate(label_cols):
        col = y[:, i] if y.ndim > 1 else y
        color = _HIST_COLORS[i % len(_HIST_COLORS)]
        ax.hist(col, bins=200, label=lbl, alpha=0.7, color=color)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot label distributions from saved data")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to model artifact directory (e.g. artifacts/models/mlp/cone)")
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

    plots = [
        ("data_before_transform.npz", "Value", "before_transform"),
        ("data_after_log.npz", "Value (log-transformed)", "after_log_transform"),
        ("data_after_scaling.npz", "Value (scaled)", "after_scaling"),
    ]

    for npz_name, x_label, out_name in plots:
        npz_path = data_dir / npz_name
        if not npz_path.exists():
            print(f"  Skipping {npz_name} (not found)")
            continue
        plot_distribution(npz_path, x_label, output_dir / out_name)
        print(f"  Saved: {out_name}.pdf")

    print(f"\nAll distribution plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
