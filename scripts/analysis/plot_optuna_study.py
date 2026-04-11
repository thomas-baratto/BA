#!/usr/bin/env python3
"""
Generate thesis-quality plots from an Optuna journal log.

Produces the following plots:
  1. Optimization history   — best objective vs. trial number
  2. Parameter importance    — fANOVA-based bar chart
  3. Slice plots             — objective vs. each continuous HP
  4. Parallel coordinate     — HP combinations colour-coded by value
  5. Contour plots           — pairwise HP interaction heatmaps

Usage:
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/plot_optuna_study.py \
        --journal runs/global_run_1049/optuna_journal_storage/journal.log \
        --output-dir docs/plots/optuna
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.thesis_style import (
    apply_thesis_style,
    COLORS,
    FIG_WIDE,
    FIG_SINGLE,
    save_fig,
)

apply_thesis_style()

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_study(journal_path: str, study_name: str | None = None):
    """Load an Optuna study from a journal file."""
    import optuna
    from optuna.storages import JournalStorage, JournalFileStorage

    storage = JournalStorage(JournalFileStorage(journal_path))
    names = optuna.study.get_all_study_names(storage)

    if study_name is None:
        if len(names) == 1:
            study_name = names[0]
        else:
            raise ValueError(f"Multiple studies found: {names}. Use --study-name.")

    study = optuna.load_study(study_name=study_name, storage=storage)
    print(f"Loaded study '{study_name}': {len(study.trials)} trials "
          f"({sum(1 for t in study.trials if t.state.name == 'COMPLETE')} completed)")
    return study


def completed_trials(study):
    return [t for t in study.trials if t.state.name == "COMPLETE"]


# ── Plot 1: Optimization History ─────────────────────────────────────────────

def plot_optimization_history(study, output_dir: Path):
    """Best objective value vs. trial number (convergence curve)."""
    trials = study.trials
    best_so_far = float("inf")
    xs, ys_best, ys_all = [], [], []

    for t in trials:
        if t.state.name == "COMPLETE":
            xs.append(t.number)
            ys_all.append(t.value)
            if t.value < best_so_far:
                best_so_far = t.value
            ys_best.append(best_so_far)

    # Clip extreme outliers for a cleaner y-axis range
    q99 = np.percentile(ys_all, 99)
    y_upper = min(q99 * 1.5, max(ys_all))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.scatter(xs, ys_all, s=5, alpha=0.20, color=COLORS["primary"],
               label="Completed trials", zorder=2, rasterized=True)

    # Running-best line
    ax.plot(xs, ys_best, color=COLORS["secondary"], lw=2.0,
            label="Best so far", zorder=3)

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Objective (val RMSE)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=min(ys_best) * 0.8, top=y_upper)
    ax.legend(loc="upper right")
    fig.tight_layout()

    save_fig(fig, output_dir / "optuna_history.png")
    print(f"  Saved: optuna_history.png/pdf")


# ── Plot 2: Parameter Importance ─────────────────────────────────────────────

def plot_param_importance(study, output_dir: Path):
    """fANOVA-based hyperparameter importance bar chart."""
    from optuna.importance import FanovaImportanceEvaluator, get_param_importances

    evaluator = FanovaImportanceEvaluator(seed=42)
    importances = get_param_importances(study, evaluator=evaluator)

    # Sort descending
    names = list(importances.keys())
    values = list(importances.values())

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=COLORS["primary"], edgecolor="white", height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (fANOVA)")

    # Value labels
    for bar, val in zip(bars, values):
        if val > 0.01:
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    save_fig(fig, output_dir / "optuna_param_importance.png")
    print(f"  Saved: optuna_param_importance.png/pdf")


# ── Plot 3: Slice Plots ─────────────────────────────────────────────────────

CONTINUOUS_PARAMS = [
    "learning_rate", "weight_decay", "dropout_rate",
    "nr_neurons", "nr_hidden_layers", "batch_size",
    "rlr_factor", "rlr_patience",
]


def plot_slices(study, output_dir: Path):
    """Objective vs. each continuous hyperparameter (marginal views)."""
    trials = completed_trials(study)
    # Only plot params that exist and have variance
    available = []
    for p in CONTINUOUS_PARAMS:
        vals = [t.params[p] for t in trials if p in t.params]
        if vals and len(set(vals)) > 1:
            available.append(p)

    # Compute a shared y-axis range (clip top 1% outliers)
    all_obj = [t.value for t in trials]
    y_upper = np.percentile(all_obj, 99) * 1.2
    y_lower = min(all_obj) * 0.8

    n = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 3.0 * nrows))
    axes = np.array(axes).flatten()

    for i, param in enumerate(available):
        ax = axes[i]
        xs = [t.params[param] for t in trials]
        ys = [t.value for t in trials]

        ax.scatter(xs, ys, s=5, alpha=0.25, color=COLORS["primary"], rasterized=True)
        ax.set_xlabel(param.replace("_", " "), fontsize=9)
        ax.set_ylabel("Objective", fontsize=9)
        ax.set_yscale("log")
        ax.set_ylim(y_lower, y_upper)

        # Highlight best trial value
        best_val = study.best_trial.params.get(param)
        if best_val is not None:
            ax.axvline(best_val, color=COLORS["secondary"], ls="--", lw=1.2,
                       alpha=0.7, label="Best")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    save_fig(fig, output_dir / "optuna_slices.png")
    print(f"  Saved: optuna_slices.png/pdf")


# ── Plot 4: Parallel Coordinate ──────────────────────────────────────────────

def plot_parallel_coordinate(study, output_dir: Path):
    """Parallel coordinate plot of top trials, colour-coded by objective."""
    from matplotlib.collections import LineCollection

    trials = completed_trials(study)
    trials.sort(key=lambda t: t.value)
    # Use top 200 for readability
    top_n = min(200, len(trials))
    top = trials[:top_n]

    params_to_show = [
        "nr_hidden_layers", "nr_neurons", "learning_rate",
        "weight_decay", "dropout_rate", "batch_size",
    ]
    # Filter to params that exist
    params_to_show = [p for p in params_to_show
                      if any(p in t.params for t in top)]

    n_axes = len(params_to_show)

    # Normalise each param to [0, 1] for plotting
    param_data = {}
    for p in params_to_show:
        vals = [t.params.get(p, np.nan) for t in top]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax != vmin else 1
        param_data[p] = {"raw": vals, "norm": [(v - vmin) / rng for v in vals],
                         "min": vmin, "max": vmax}

    # Colour by objective
    obj_vals = np.array([t.value for t in top])
    obj_min, obj_max = obj_vals.min(), obj_vals.max()
    cmap = plt.cm.viridis_r
    norm = plt.Normalize(vmin=obj_min, vmax=obj_max)

    # Single-axes approach: x = axis index, y = normalised param value
    fig, ax = plt.subplots(figsize=FIG_WIDE)

    # Build line segments for LineCollection (much faster than individual plot calls)
    # Draw worst trials first so best trials render on top
    order = np.argsort(-obj_vals)  # worst first
    for trial_idx in order:
        ys = [param_data[p]["norm"][trial_idx] for p in params_to_show]
        xs = list(range(n_axes))
        ax.plot(xs, ys, color=cmap(norm(obj_vals[trial_idx])),
                alpha=0.35, lw=0.8, zorder=2)

    # Vertical axis lines and tick labels
    for i, p in enumerate(params_to_show):
        ax.axvline(i, color="black", lw=0.8, zorder=3)
        # Min/max annotations beside each axis
        ax.text(i, -0.08, f"{param_data[p]['min']:.4g}",
                ha="center", va="top", fontsize=7, color="0.3")
        ax.text(i, 1.08, f"{param_data[p]['max']:.4g}",
                ha="center", va="bottom", fontsize=7, color="0.3")

    ax.set_xlim(-0.3, n_axes - 0.7)
    ax.set_ylim(-0.15, 1.15)
    ax.set_xticks(range(n_axes))
    ax.set_xticklabels([p.replace("_", "\n") for p in params_to_show], fontsize=8)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04, aspect=30)
    cbar.set_label("Objective (val RMSE)", fontsize=9)

    fig.tight_layout()
    save_fig(fig, output_dir / "optuna_parallel_coordinate.png")
    print(f"  Saved: optuna_parallel_coordinate.png/pdf")


# ── Plot 5: Contour Plots ───────────────────────────────────────────────────

def plot_contours(study, output_dir: Path):
    """Pairwise contour plots for key HP interactions."""
    pairs = [
        ("learning_rate", "weight_decay"),
        ("nr_neurons", "nr_hidden_layers"),
        ("learning_rate", "nr_neurons"),
    ]

    trials = completed_trials(study)
    # Filter to top 50% for clearer contours
    trials.sort(key=lambda t: t.value)
    cutoff = len(trials) // 2
    top_half = trials[:cutoff]

    fig, axes = plt.subplots(1, len(pairs), figsize=(7.2, 3.5))
    if len(pairs) == 1:
        axes = [axes]

    cmap = plt.cm.viridis_r
    # Shared colour norm across all panels
    all_cs = [t.value for t in top_half]
    vmin_c, vmax_c = min(all_cs), max(all_cs)
    norm = plt.Normalize(vmin=vmin_c, vmax=vmax_c)

    for idx, (px, py) in enumerate(pairs):
        ax = axes[idx]
        xs = [t.params[px] for t in top_half if px in t.params and py in t.params]
        ys = [t.params[py] for t in top_half if px in t.params and py in t.params]
        cs = [t.value for t in top_half if px in t.params and py in t.params]

        sc = ax.scatter(xs, ys, c=cs, cmap=cmap, norm=norm,
                        s=10, alpha=0.6, rasterized=True)
        ax.set_xlabel(px.replace("_", " "), fontsize=9)
        ax.set_ylabel(py.replace("_", " "), fontsize=9)

        # Mark best trial
        bx = study.best_trial.params.get(px)
        by = study.best_trial.params.get(py)
        if bx is not None and by is not None:
            ax.scatter([bx], [by], marker="*", s=150, color=COLORS["accent1"],
                       edgecolors="black", zorder=5, linewidths=0.8,
                       label="Best trial")

    axes[0].legend(fontsize=8, loc="upper left")

    fig.tight_layout(rect=[0, 0, 0.88, 1.0])
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax)
    cbar.set_label("Objective", fontsize=9)
    save_fig(fig, output_dir / "optuna_contours.png")
    print(f"  Saved: optuna_contours.png/pdf")


# ── Plot 6: Trial Duration Distribution ─────────────────────────────────────

def plot_trial_durations(study, output_dir: Path):
    """Histogram of completed trial wall-clock durations."""
    trials = completed_trials(study)
    durations = []
    for t in trials:
        wall = t.user_attrs.get("trial_wall_time_sec")
        if wall:
            durations.append(wall)
        elif t.duration:
            durations.append(t.duration.total_seconds())

    if not durations:
        print("  Skipping duration plot: no duration data available.")
        return

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.hist(durations, bins=40, color=COLORS["primary"], edgecolor="white", alpha=0.85)
    ax.axvline(np.median(durations), color=COLORS["secondary"], ls="--", lw=1.5,
               label=f"Median: {np.median(durations):.0f} s")
    ax.set_xlabel("Trial duration [s]")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()

    save_fig(fig, output_dir / "optuna_trial_durations.png")
    print(f"  Saved: optuna_trial_durations.png/pdf")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Optuna study results.")
    parser.add_argument("--journal", required=True,
                        help="Path to the Optuna journal.log file.")
    parser.add_argument("--study-name", default=None,
                        help="Name of the study (auto-detected if only one).")
    parser.add_argument("--output-dir", default="docs/plots/optuna",
                        help="Directory for output plots.")
    parser.add_argument("--skip-importance", action="store_true",
                        help="Skip fANOVA importance (slow on large studies).")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study = load_study(args.journal, args.study_name)

    print("\nGenerating plots...")
    plot_optimization_history(study, output_dir)
    plot_trial_durations(study, output_dir)
    plot_slices(study, output_dir)
    plot_contours(study, output_dir)
    plot_parallel_coordinate(study, output_dir)

    if not args.skip_importance:
        print("  Computing fANOVA importance (may take a moment)...")
        plot_param_importance(study, output_dir)
    else:
        print("  Skipped: param importance (--skip-importance)")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
