#!/usr/bin/env python3
"""Aggregate power analysis across training runs.

Combines per-second CSV time series and summary JSONs from SLURM training
runs into thesis-quality plots.  Supports multiple run types (Optuna HPO
sweep, random model sweep, etc.) via configurable glob patterns.

Since all workers on a node monitor the *same* machine, the script selects
the longest-running worker's CSV for the full power timeline and aggregates
summary JSONs for energy / cost stats.

Usage
-----
    # Optuna HPO sweep (default)
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/plot_power_analysis.py \\
        --runs-dir runs --output-dir docs/plots/power/optuna

    # Random model sweep
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/plot_power_analysis.py \\
        --runs-dir runs --run-type random --output-dir docs/plots/power/random

    # Custom glob pattern
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/plot_power_analysis.py \\
        --runs-dir runs --run-pattern "run_20260328*_worker*" \\
        --output-dir docs/plots/power/custom
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from core.thesis_style import (
    apply_thesis_style,
    COLORS,
    DPI,
    FIG_SINGLE,
    FIG_WIDE,
    FIG_TALL,
    save_fig,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

# Predefined glob patterns for known run types
# Each pattern is joined with "power_monitor" to find the data dirs
RUN_PATTERNS: dict[str, str] = {
    "optuna": "run_20260325-19291*_worker*",
    "random": "run_sweep_random_*",
    "mlp": "*_MLP_*",
}


def _find_worker_dirs(runs_dir: str, run_type: str | None = None,
                      run_pattern: str | None = None) -> list[str]:
    """Return sorted power_monitor dirs matching the given run pattern.

    Priority: explicit *run_pattern* > predefined *run_type* > auto-detect.
    """
    if run_pattern:
        pattern = os.path.join(runs_dir, run_pattern, "power_monitor")
    elif run_type and run_type in RUN_PATTERNS:
        pattern = os.path.join(runs_dir, RUN_PATTERNS[run_type], "power_monitor")
    else:
        # Auto-detect: try each known pattern
        for name, pat in RUN_PATTERNS.items():
            candidate = os.path.join(runs_dir, pat, "power_monitor")
            dirs = sorted(glob.glob(candidate))
            if dirs:
                print(f"Auto-detected run type: {name}")
                return dirs
        raise FileNotFoundError(
            f"No power_monitor dirs found under {runs_dir}. "
            f"Use --run-type or --run-pattern to specify."
        )

    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f"No worker power_monitor dirs found for pattern: {pattern}")
    return dirs


def _load_longest_csv(worker_dirs: list[str]) -> pd.DataFrame:
    """Load the CSV from the longest-duration worker (most complete timeline)."""
    best_path, best_rows = "", 0
    for d in worker_dirs:
        csvs = glob.glob(os.path.join(d, "power_log_*.csv"))
        if not csvs:
            continue
        n = sum(1 for _ in open(csvs[0])) - 1
        if n > best_rows:
            best_rows = n
            best_path = csvs[0]

    df = pd.read_csv(best_path)
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601")
    return df


def _load_summaries(worker_dirs: list[str]) -> list[dict]:
    """Load all power_summary_*.json files."""
    summaries = []
    for d in worker_dirs:
        jsons = glob.glob(os.path.join(d, "power_summary_*.json"))
        if jsons:
            with open(jsons[0]) as f:
                summaries.append(json.load(f))
    return summaries


# ── Plot functions ───────────────────────────────────────────────────────────

def _elapsed_hours(df: pd.DataFrame) -> np.ndarray:
    """Convert datetime column to elapsed hours from t=0."""
    dt = df["datetime"]
    return (dt - dt.iloc[0]).dt.total_seconds().to_numpy() / 3600


def plot_power_timeline(df: pd.DataFrame, out: Path, run_label: str = "HPO Sweep",
                        prefix: str = "") -> None:
    """Stacked area: GPU and CPU power over time."""
    fig, ax = plt.subplots(figsize=FIG_WIDE)

    t = _elapsed_hours(df)
    gpu = df["total_gpu_power_w"]
    cpu = df["total_cpu_power_w"]

    ax.fill_between(t, 0, gpu, alpha=0.65, label="GPU power", color=COLORS["primary"])
    ax.fill_between(t, gpu, gpu + cpu, alpha=0.65, label="CPU power", color=COLORS["accent1"])
    ax.plot(t, gpu + cpu, linewidth=0.6, color=COLORS["text"], alpha=0.5)

    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_title(f"Node Power Consumption During {run_label}")
    ax.legend(loc="lower left")

    save_fig(fig, out / f"{prefix}power_timeline")


def plot_power_distribution(df: pd.DataFrame, out: Path, prefix: str = "") -> None:
    """Histogram of total system power draw."""
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    total = df["total_power_w"]
    ax.hist(total, bins=60, color=COLORS["primary"], edgecolor="white", linewidth=0.3)
    ax.axvline(total.mean(), ls="--", lw=1.5, color=COLORS["secondary"],
               label=f"Mean: {total.mean():.0f} W")
    ax.axvline(total.median(), ls=":", lw=1.5, color=COLORS["accent3"],
               label=f"Median: {total.median():.0f} W")

    ax.set_xlabel("Total power (W)")
    ax.set_ylabel("Count (1-second samples)")
    ax.set_title("Power Draw Distribution")
    ax.legend()

    save_fig(fig, out / f"{prefix}power_distribution")


def plot_gpu_breakdown(summaries: list[dict], out: Path, prefix: str = "") -> None:
    """Per-GPU bar chart: avg power, utilization, temperature."""
    # Use the longest-running summary (most representative)
    longest = max(summaries, key=lambda s: s["monitoring_info"]["duration_seconds"])
    gpu_stats = longest["gpu_statistics"]

    gpu_names = sorted(gpu_stats.keys())
    n = len(gpu_names)
    x = np.arange(n)

    avg_power = [gpu_stats[g]["avg_power"] for g in gpu_names]
    avg_util = [gpu_stats[g]["avg_utilization"] for g in gpu_names]
    avg_temp = [gpu_stats[g]["avg_temperature"] for g in gpu_names]

    fig, axes = plt.subplots(1, 3, figsize=FIG_WIDE, sharey=False)
    labels = [f"GPU {i}" for i in range(n)]

    # Power
    axes[0].barh(x, avg_power, color=COLORS["primary"], height=0.6)
    axes[0].set_xlabel("Avg power (W)")
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(labels)
    axes[0].set_title("Power")

    # Utilization
    axes[1].barh(x, avg_util, color=COLORS["accent3"], height=0.6)
    axes[1].set_xlabel("Avg utilization (%)")
    axes[1].set_xlim(0, 100)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels([])
    axes[1].set_title("Utilization")

    # Temperature
    axes[2].barh(x, avg_temp, color=COLORS["accent1"], height=0.6)
    axes[2].set_xlabel("Avg temperature (°C)")
    axes[2].set_yticks(x)
    axes[2].set_yticklabels([])
    axes[2].set_title("Temperature")

    fig.suptitle("Per-GPU Statistics", y=1.02)
    fig.tight_layout()
    save_fig(fig, out / f"{prefix}gpu_breakdown")


def plot_energy_breakdown(summaries: list[dict], out: Path, prefix: str = "") -> None:
    """Pie + summary bar: GPU vs CPU energy and cost."""
    # All workers monitor the same node; use the longest for the definitive numbers
    longest = max(summaries, key=lambda s: s["monitoring_info"]["duration_seconds"])
    energy = longest["energy_consumption"]

    gpu_kwh = energy["gpu_energy_wh"] / 1000
    cpu_kwh = energy["cpu_energy_wh"] / 1000
    total_kwh = energy["total_energy_kwh"]
    duration_h = longest["monitoring_info"]["duration_hours"]

    fig, axes = plt.subplots(1, 2, figsize=FIG_WIDE)

    # Pie chart
    sizes = [gpu_kwh, cpu_kwh]
    labels = [f"GPU\n{gpu_kwh:.2f} kWh", f"CPU\n{cpu_kwh:.2f} kWh"]
    colors = [COLORS["primary"], COLORS["accent1"]]
    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(9)
    axes[0].set_title("Energy Split")

    # Summary bar
    cost_rates = {"$0.10/kWh": 0.10, "$0.15/kWh": 0.15, "$0.20/kWh": 0.20}
    rate_labels = list(cost_rates.keys())
    costs = [total_kwh * r for r in cost_rates.values()]

    bars = axes[1].bar(rate_labels, costs, color=COLORS["primary"], width=0.5)
    for bar, cost in zip(bars, costs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"${cost:.2f}", ha="center", va="bottom", fontsize=9)

    axes[1].set_ylabel("Cost (USD)")
    axes[1].set_title("Estimated Electricity Cost")

    # Add annotation with total stats
    stats_text = (
        f"Total: {total_kwh:.2f} kWh\n"
        f"Duration: {duration_h:.1f} h\n"
        f"CO$_2$: ~{total_kwh * 0.4:.2f} kg"
    )
    axes[1].annotate(
        stats_text, xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["grid"], alpha=0.9),
    )

    fig.suptitle("Energy Consumption and Cost", y=1.02)
    fig.tight_layout()
    save_fig(fig, out / f"{prefix}energy_breakdown")


def plot_worker_durations(summaries: list[dict], out: Path, run_label: str = "HPO",
                          prefix: str = "") -> None:
    """Horizontal bar: per-worker duration sorted longest-first."""
    durations = sorted(
        [s["monitoring_info"]["duration_hours"] for s in summaries],
        reverse=True,
    )
    n = len(durations)
    fig, ax = plt.subplots(figsize=(FIG_SINGLE[0], max(4.0, n * 0.14)))

    y = np.arange(n)
    ax.barh(y, durations, color=COLORS["primary"], height=0.7, alpha=0.8)

    ax.axvline(np.mean(durations), ls="--", lw=1.2, color=COLORS["secondary"],
               label=f"Mean: {np.mean(durations):.1f} h")
    ax.axvline(np.median(durations), ls=":", lw=1.2, color=COLORS["accent3"],
               label=f"Median: {np.median(durations):.1f} h")

    ax.set_xlabel("Duration (hours)")
    ax.set_ylabel("Worker")
    ax.set_yticks(y[::5])
    ax.set_yticklabels([f"W{i}" for i in y[::5]])
    ax.set_title(f"{run_label} Worker Durations ({n} workers)")
    ax.invert_yaxis()
    ax.legend(loc="lower right")

    save_fig(fig, out / f"{prefix}worker_durations")


def plot_utilization_timeline(df: pd.DataFrame, out: Path, run_label: str = "HPO Sweep",
                              prefix: str = "") -> None:
    """CPU utilization and active GPU count over time."""
    fig, ax1 = plt.subplots(figsize=FIG_WIDE)

    t = _elapsed_hours(df)

    color_cpu = COLORS["accent1"]
    ax1.plot(t, df["cpu_utilization_pct"], linewidth=0.4, alpha=0.3, color=color_cpu)
    # Rolling mean for readability
    window = min(300, len(df) // 10)
    if window > 1:
        ax1.plot(t, df["cpu_utilization_pct"].rolling(window, center=True).mean(),
                 linewidth=1.5, color=color_cpu, label="CPU util. (5-min avg)")
    ax1.set_ylabel("CPU utilization (%)")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor=color_cpu)

    ax2 = ax1.twinx()
    color_gpu = COLORS["primary"]
    ax2.plot(t, df["num_active_gpus"], linewidth=0.4, alpha=0.3, color=color_gpu)
    if window > 1:
        ax2.plot(t, df["num_active_gpus"].rolling(window, center=True).mean(),
                 linewidth=1.5, color=color_gpu, label="Active GPUs (5-min avg)")
    ax2.set_ylabel("Active GPUs")
    ax2.set_ylim(0, 8)
    ax2.tick_params(axis="y", labelcolor=color_gpu)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    ax1.set_xlabel("Elapsed time (hours)")
    ax1.set_title(f"CPU and GPU Utilization During {run_label}")

    save_fig(fig, out / f"{prefix}utilization_timeline")


def print_summary(summaries: list[dict]) -> None:
    """Print aggregate statistics to console."""
    longest = max(summaries, key=lambda s: s["monitoring_info"]["duration_seconds"])
    energy = longest["energy_consumption"]
    power = longest["power_statistics"]
    info = longest["monitoring_info"]
    n_workers = len(summaries)

    durations = [s["monitoring_info"]["duration_hours"] for s in summaries]
    total_gpu_hours = sum(durations)

    print("\n" + "=" * 60)
    print("AGGREGATE HPO POWER ANALYSIS")
    print("=" * 60)
    print(f"  Workers:           {n_workers}")
    print(f"  Node duration:     {info['duration_hours']:.1f} h")
    print(f"  Total GPU-hours:   {total_gpu_hours:.0f} h")
    print(f"  Samples:           {info['num_samples']:,}")
    print(f"\n  Avg power:         {power['avg_total_power_w']:.0f} W")
    print(f"  Peak power:        {power['max_total_power_w']:.0f} W")
    print(f"  GPU / CPU split:   {power['avg_gpu_power_w']:.0f} / {power['avg_cpu_power_w']:.0f} W")
    print(f"\n  Total energy:      {energy['total_energy_kwh']:.2f} kWh")
    print(f"  GPU energy:        {energy['gpu_energy_wh'] / 1000:.2f} kWh ({energy['gpu_percentage']:.1f}%)")
    print(f"  CPU energy:        {energy['cpu_energy_wh'] / 1000:.2f} kWh ({energy['cpu_percentage']:.1f}%)")
    print(f"  Est. CO₂:          ~{energy['total_energy_kwh'] * 0.4:.2f} kg")
    print(f"  Est. cost:         ${energy['total_energy_kwh'] * 0.15:.2f} (@$0.15/kWh)")

    gpu_stats = longest["gpu_statistics"]
    print(f"\n  Per-GPU (7 GPUs):")
    for name, g in sorted(gpu_stats.items()):
        print(f"    {name}: {g['avg_power']:.0f}W avg, "
              f"{g['avg_utilization']:.0f}% util, "
              f"{g['avg_temperature']:.0f}°C avg / {g['max_temperature']:.0f}°C max")
    print("=" * 60 + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate power analysis for training runs")
    p.add_argument("--runs-dir", default="runs", help="Root runs directory")
    p.add_argument("--output-dir", default="docs/plots/power", help="Output plot directory")
    p.add_argument("--run-type", choices=list(RUN_PATTERNS.keys()), default=None,
                   help="Predefined run type (selects glob pattern automatically)")
    p.add_argument("--run-pattern", default=None,
                   help="Custom glob pattern for run dirs (relative to runs-dir)")
    p.add_argument("--run-label", default=None,
                   help="Label for plot titles (default: auto from run-type)")
    p.add_argument("--prefix", default=None,
                   help="Filename prefix for plots, e.g. 'random_' (default: auto from run-type)")
    return p.parse_args()


_DEFAULT_LABELS: dict[str, str] = {
    "optuna": "HPO Sweep",
    "random": "Random Model Sweep",
}

_DEFAULT_PREFIXES: dict[str, str] = {
    "optuna": "optuna_",
    "random": "random_",
}


def main() -> None:
    args = parse_args()
    apply_thesis_style()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_label = args.run_label or _DEFAULT_LABELS.get(args.run_type or "", "Training")
    prefix = args.prefix if args.prefix is not None else _DEFAULT_PREFIXES.get(args.run_type or "", "")

    print("Discovering worker directories...")
    worker_dirs = _find_worker_dirs(args.runs_dir, args.run_type, args.run_pattern)
    print(f"Found {len(worker_dirs)} workers")

    print("Loading longest CSV timeline...")
    df = _load_longest_csv(worker_dirs)
    print(f"Loaded {len(df):,} samples ({df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]})")

    print("Loading summaries...")
    summaries = _load_summaries(worker_dirs)
    print(f"Loaded {len(summaries)} summaries")

    print_summary(summaries)

    print("Generating plots...")
    plot_power_timeline(df, out, run_label, prefix)
    plot_power_distribution(df, out, prefix)
    plot_gpu_breakdown(summaries, out, prefix)
    plot_energy_breakdown(summaries, out, prefix)
    plot_worker_durations(summaries, out, run_label, prefix)
    plot_utilization_timeline(df, out, run_label, prefix)

    print(f"\nAll plots saved to {out}/")


if __name__ == "__main__":
    main()
