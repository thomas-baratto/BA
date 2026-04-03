#!/usr/bin/env python3
"""
Plot Pareto frontier visualizations from a sweep summary_table.csv.

Generates 4 plots (2 datasets × 2 metrics):
  - Time(s) vs nRMSE  with Pareto frontier
  - Time(s) vs 1-KGE  with Pareto frontier

Each plot uses log-log axes, consistent model colors/markers,
automatic outlier filtering, iso-cost contour lines, and highlights
the knee-point winner with a star marker.

Usage:
    PYTHONPATH=. python scripts/analysis/plot_pareto_frontiers.py \\
        --summary-csv runs/run_sweep_random_1048/summary_table.csv \\
        --output-dir  docs/plots
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.thesis_style import apply_thesis_style, FIG_WIDE, save_fig
from scripts.analysis.pareto_manager import is_pareto_efficient

apply_thesis_style()

# ── Consistent model styling ─────────────────────────────────────────────────

MODEL_STYLE = {
    "ELM": {"color": "#d62728", "marker": "s", "label": "ELM"},
    "SResdRVFL": {"color": "#bcbd22", "marker": "D", "label": "SResdRVFL"},
    "dRVFL": {"color": "#2ca02c", "marker": "^", "label": "dRVFL"},
    "edRVFL": {"color": "#00bfff", "marker": "+", "label": "edRVFL"},
    "edRVFL-SC": {"color": "#1f77b4", "marker": "X", "label": "edRVFL-SC"},
    "esc-edRVFL": {"color": "#e377c2", "marker": "p", "label": "esc-edRVFL"},
}

# Fallback for unknown model types
_FALLBACK_COLORS = [
    "#ff7f0e", "#9467bd", "#8c564b", "#7f7f7f", "#17becf"
]
_FALLBACK_MARKERS = ["o", "v", "<", ">", "h"]


def _get_style(model_name: str, idx: int = 0) -> dict:
    """Return color/marker/label for a model type."""
    if model_name in MODEL_STYLE:
        return MODEL_STYLE[model_name]
    return {
        "color": _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)],
        "marker": _FALLBACK_MARKERS[idx % len(_FALLBACK_MARKERS)],
        "label": model_name,
    }


# ── Outlier filtering ────────────────────────────────────────────────────────

def _filter_outliers(df: pd.DataFrame, col: str, factor: float = 3.0) -> pd.DataFrame:
    """Remove rows where `col` is more than `factor` × IQR above Q3."""
    vals = df[col].dropna()
    if vals.empty:
        return df
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + factor * iqr
    return df[df[col] <= upper]


# ── Iso-cost curves ──────────────────────────────────────────────────────────

def _draw_iso_cost_curves(ax, x_range, y_range, n_curves: int = 5):
    """Draw dashed diagonal iso-cost lines (Time × Error = const) on log-log axes."""
    log_x = np.log10(x_range)
    log_y = np.log10(y_range)

    # Choose iso-cost constants spanning the data
    log_costs = np.linspace(log_x[0] + log_y[0], log_x[1] + log_y[1], n_curves + 2)[1:-1]

    xs = np.logspace(log_x[0] - 0.5, log_x[1] + 0.5, 200)
    for log_c in log_costs:
        c = 10 ** log_c
        ys = c / xs
        ax.plot(xs, ys, color="lightgray", ls="--", lw=0.8, zorder=0)

        # Place label along the isoline where it intersects the visible area.
        # Try a point ~30% from the left edge of the visible x-range (in log space).
        label_log_x = log_x[0] + 0.3 * (log_x[1] - log_x[0])
        label_x = 10 ** label_log_x
        label_y = c / label_x

        # If that y falls outside the visible range, slide along the curve.
        if label_y > y_range[1]:
            label_y = y_range[1] * 0.85
            label_x = c / label_y
        elif label_y < y_range[0]:
            label_y = y_range[0] * 1.2
            label_x = c / label_y

        if x_range[0] <= label_x <= x_range[1] and y_range[0] <= label_y <= y_range[1]:
            ax.text(
                label_x, label_y, f"{c:.2g}",
                fontsize=6.5, color="gray", rotation=-45,
                ha="center", va="bottom", zorder=0,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
            )


# ── Main plot function ───────────────────────────────────────────────────────

def plot_pareto_frontier(
    df: pd.DataFrame,
    dataset: str,
    error_col: str,
    error_label: str,
    output_path: Path,
    knee_folder: str | None = None,
):
    """
    Create a single Pareto frontier plot.

    Parameters
    ----------
    df : DataFrame filtered to one dataset, with Time(s) and error_col.
    dataset : dataset name for title.
    error_col : column name for the y-axis error metric.
    error_label : human-readable y-axis label.
    output_path : where to save the PNG.
    knee_folder : optional Folder name of the knee-point winner (starred).
    """
    # Filter outliers
    clean = _filter_outliers(df.dropna(subset=["Time(s)", error_col]), error_col)
    clean = _filter_outliers(clean, "Time(s)")

    if clean.empty:
        print(f"  SKIP {dataset}/{error_label}: no data after filtering")
        return

    time_vals = clean["Time(s)"].astype(float).values
    error_vals = clean[error_col].astype(float).values

    # Compute Pareto frontier
    costs = np.column_stack([time_vals, error_vals])
    mask = is_pareto_efficient(costs)
    frontier_df = clean.loc[mask].copy()
    frontier_costs = costs[mask]

    # Sort frontier by time for the connecting line
    sort_idx = np.argsort(frontier_costs[:, 0])
    frontier_sorted = frontier_costs[sort_idx]

    # Axis ranges (with padding) — computed early so staircase can use x_range
    x_range = (time_vals.min() * 0.7, time_vals.max() * 1.5)
    y_range = (error_vals.min() * 0.85, error_vals.max() * 1.2)

    # Build staircase (orthogonal segments) for the Pareto frontier
    stair_x, stair_y = [], []
    for i, (tx, ey) in enumerate(frontier_sorted):
        if i > 0:
            # Horizontal segment to the new x at the previous y
            stair_x.append(tx)
            stair_y.append(frontier_sorted[i - 1, 1])
        stair_x.append(tx)
        stair_y.append(ey)
    # Extend rightward from last point so the frontier doesn't just stop
    stair_x.append(x_range[1])
    stair_y.append(frontier_sorted[-1, 1])

    # ── Create figure ──
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Iso-cost curves
    _draw_iso_cost_curves(ax, x_range, y_range)

    # Scatter by model type
    model_types = sorted(clean["Model"].unique())
    unknown_idx = 0
    for model_name in model_types:
        style = _get_style(model_name, unknown_idx)
        if model_name not in MODEL_STYLE:
            unknown_idx += 1

        sub = clean[clean["Model"] == model_name]
        ax.scatter(
            sub["Time(s)"], sub[error_col],
            c=style["color"], marker=style["marker"],
            s=50, alpha=0.7, edgecolors="none",
            label=style["label"], zorder=3,
        )

    # Pareto frontier staircase
    ax.plot(
        stair_x, stair_y,
        color="red", ls="--", lw=1.2, zorder=4,
        label="Pareto Frontier",
    )

    # Knee-point star
    if knee_folder and "Folder" in clean.columns:
        knee_rows = clean[clean["Folder"] == knee_folder]
        if not knee_rows.empty:
            kr = knee_rows.iloc[0]
            ax.scatter(
                [kr["Time(s)"]], [kr[error_col]],
                marker="*", s=350, c="gold", edgecolors="black",
                linewidths=1.2, zorder=5, label="Knee Point",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(error_label)
    ax.set_title(
        f"{error_label} vs. Training Time — {dataset.capitalize()}",
    )
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    ax.legend(
        title="Model Type",
        loc="upper right", framealpha=0.9,
    )
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    save_fig(fig, output_path)
    print(f"  Saved: {output_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def generate_all_pareto_plots(
    summary_csv: str,
    output_dir: str,
    knee_csv: str | None = None,
) -> list[str]:
    """Generate all 4 Pareto frontier plots. Returns list of filenames."""
    df = pd.read_csv(summary_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load knee-point winners if available
    knee_map = {}  # (dataset, frontier_name) -> folder
    if knee_csv and Path(knee_csv).exists():
        kdf = pd.read_csv(knee_csv)
        for _, row in kdf.iterrows():
            knee_map[(row["dataset"], row["frontier"])] = row["folder"]

    filenames = []

    frontiers = [
        ("nRMSE", "nRMSE", "nRMSE"),
        ("1-KGE", "KGE", "1 − KGE"),
    ]

    for dataset in sorted(df["Dataset"].dropna().unique()):
        sub = df[df["Dataset"] == dataset].copy()

        for frontier_name, col, ylabel in frontiers:
            if col == "KGE":
                # Create a temporary 1-KGE column
                sub = sub.copy()
                sub["_1mKGE"] = 1.0 - sub["KGE"].astype(float)
                plot_col = "_1mKGE"
            else:
                plot_col = col

            fname = f"pareto_{dataset}_{frontier_name.replace('-', '')}.png"
            knee_folder = knee_map.get((dataset, frontier_name))

            plot_pareto_frontier(
                df=sub,
                dataset=dataset,
                error_col=plot_col,
                error_label=ylabel,
                output_path=out / fname,
                knee_folder=knee_folder,
            )
            filenames.append(fname)

    return filenames


def main():
    parser = argparse.ArgumentParser(description="Plot Pareto frontiers from sweep results")
    parser.add_argument("--summary-csv", required=True, help="Path to summary_table.csv")
    parser.add_argument("--output-dir", default="docs/plots/pareto", help="Output directory for plots")
    parser.add_argument("--knee-csv", default=None, help="Path to knee_point_winners.csv (optional)")
    args = parser.parse_args()

    # Auto-detect knee CSV if not provided
    if args.knee_csv is None:
        candidate = Path(args.summary_csv).parent / "knee_point_winners.csv"
        if candidate.exists():
            args.knee_csv = str(candidate)

    generate_all_pareto_plots(args.summary_csv, args.output_dir, args.knee_csv)


if __name__ == "__main__":
    main()
