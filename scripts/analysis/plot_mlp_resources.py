#!/usr/bin/env python3
"""Plot MLP training resource usage (CPU, RAM, GPU memory) in thesis style.

Reads resource_usage.json and results JSON from each MLP artifact directory
and produces publication-quality plots with a time (minutes) x-axis.
Plots are written to per-dataset subdirectories under the output root:
  docs/plots/power/mlp_cone/, docs/plots/power/mlp_isotherm/, etc.

Usage
-----
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/plot_mlp_resources.py
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/plot_mlp_resources.py \\
        --output-dir docs/plots/power
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.thesis_style import (
    apply_thesis_style,
    COLORS,
    FIG_SINGLE,
    FIG_WIDE,
    save_fig,
)

ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent.parent / "artifacts" / "models" / "mlp"

DATASETS = {
    "cone": "Cone",
    "isotherm": "Isotherm",
}


def _load_resource_json(dataset: str) -> dict | None:
    """Load resource_usage.json for a given dataset, or None if missing."""
    path = ARTIFACTS_ROOT / dataset / "resources" / "resource_usage.json"
    if not path.exists():
        print(f"  Warning: {path} not found, skipping")
        return None
    with open(path) as f:
        return json.load(f)


def _load_train_time(dataset: str) -> float | None:
    """Load training time in seconds from the results JSON."""
    pattern = str(ARTIFACTS_ROOT / dataset / "results_*.json")
    for p in glob.glob(pattern):
        with open(p) as f:
            data = json.load(f)
        t = data.get("train_time_seconds")
        if t is not None:
            return float(t)
    return None


def _make_time_axis(n_epochs: int, total_seconds: float | None) -> tuple[np.ndarray, str]:
    """Return (x_values, xlabel).  Uses minutes if time is known, else epochs."""
    if total_seconds is not None and total_seconds > 0:
        time_per_epoch = total_seconds / n_epochs
        x = np.arange(n_epochs) * time_per_epoch / 60.0
        return x, "Time (min)"
    return np.arange(n_epochs, dtype=float), "Epoch"


def _has_gpu_data(data: dict) -> bool:
    """Return True if the resource data contains non-None GPU measurements."""
    gpu_alloc = data.get("gpu_memory_allocated_mib", [])
    return any(v is not None for v in gpu_alloc)


def plot_gpu_memory(data: dict, dataset: str, out: Path,
                    x: np.ndarray, xlabel: str) -> None:
    """GPU memory (allocated vs reserved) over training time."""
    if not _has_gpu_data(data):
        print(f"  Skipping GPU memory plot for {dataset} (CPU-only training)")
        return
    allocated = np.array(data["gpu_memory_allocated_mib"], dtype=float)
    reserved = np.array(data["gpu_memory_reserved_mib"], dtype=float)

    fig, ax = plt.subplots(figsize=FIG_WIDE)

    ax.plot(x, allocated, linewidth=1.2, color=COLORS["primary"],
            label="Allocated", alpha=0.85)
    ax.plot(x, reserved, linewidth=1.2, color=COLORS["accent1"],
            label="Reserved", alpha=0.85)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("GPU Memory (MiB)")
    ax.set_title(f"GPU Memory Usage — MLP {DATASETS[dataset]}")
    ax.legend()
    fig.tight_layout()

    save_fig(fig, out / f"mlp_{dataset}_gpu_memory")


def plot_cpu_ram(data: dict, dataset: str, out: Path,
                x: np.ndarray, xlabel: str) -> None:
    """CPU utilization and RAM usage over training time."""
    cpu = np.array(data["cpu_percent"])
    ram = np.array(data["ram_percent"])

    fig, ax1 = plt.subplots(figsize=FIG_WIDE)

    color_cpu = COLORS["accent1"]
    color_ram = COLORS["primary"]

    ax1.plot(x, cpu, linewidth=0.8, alpha=0.3, color=color_cpu)
    # Smoothed line for readability
    window = min(50, len(x) // 10)
    if window > 1:
        from pandas import Series
        cpu_smooth = Series(cpu).rolling(window, center=True).mean()
        ax1.plot(x, cpu_smooth, linewidth=1.5, color=color_cpu,
                 label=f"CPU util. ({window}-epoch avg)")
    else:
        ax1.plot(x, cpu, linewidth=1.5, color=color_cpu, label="CPU util.")

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("CPU Utilization (%)")
    ax1.set_ylim(0, max(cpu.max() * 1.15, 10))
    ax1.tick_params(axis="y", labelcolor=color_cpu)

    ax2 = ax1.twinx()
    ax2.plot(x, ram, linewidth=1.2, color=color_ram, alpha=0.85, label="RAM usage")
    ax2.set_ylabel("RAM Usage (%)")
    ax2.set_ylim(0, max(ram.max() * 1.5, 5))
    ax2.tick_params(axis="y", labelcolor=color_ram)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    ax1.set_title(f"CPU and RAM Usage — MLP {DATASETS[dataset]}")
    fig.tight_layout()

    save_fig(fig, out / f"mlp_{dataset}_cpu_ram")


def plot_gpu_utilization(data: dict, dataset: str, out: Path,
                        x: np.ndarray, xlabel: str) -> None:
    """GPU memory utilization percentage over training time."""
    if not _has_gpu_data(data):
        print(f"  Skipping GPU utilization plot for {dataset} (CPU-only training)")
        return
    gpu_pct = np.array(data["gpu_memory_percent"], dtype=float)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    ax.plot(x, gpu_pct, linewidth=1.0, color=COLORS["primary"], alpha=0.7)
    ax.axhline(gpu_pct.mean(), ls="--", lw=1.2, color=COLORS["secondary"],
               label=f"Mean: {gpu_pct.mean():.2f}%")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("GPU Memory (%)")
    ax.set_title(f"GPU Memory Utilization — MLP {DATASETS[dataset]}")
    ax.legend()
    fig.tight_layout()

    save_fig(fig, out / f"mlp_{dataset}_gpu_util")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot MLP training resource usage")
    p.add_argument("--output-dir", default="docs/plots/power",
                   help="Output directory for plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_thesis_style()

    root_out = Path(args.output_dir)

    for dataset in DATASETS:
        print(f"Processing {dataset}...")
        data = _load_resource_json(dataset)
        if data is None:
            continue

        n_epochs = len(data["step"])
        train_time = _load_train_time(dataset)
        x, xlabel = _make_time_axis(n_epochs, train_time)

        if train_time is not None:
            print(f"  {n_epochs} epochs, {train_time:.1f}s total → {xlabel}")
        else:
            print(f"  {n_epochs} epochs (no timing data, using epoch index)")

        out = root_out / f"mlp_{dataset}"
        out.mkdir(parents=True, exist_ok=True)

        plot_gpu_memory(data, dataset, out, x, xlabel)
        plot_cpu_ram(data, dataset, out, x, xlabel)
        plot_gpu_utilization(data, dataset, out, x, xlabel)

    print(f"\nAll plots saved under {root_out}/")


if __name__ == "__main__":
    main()
