#!/usr/bin/env python3
"""
Select the knee-point model from each Pareto frontier.

Four frontiers are considered (2 datasets × 2 metric pairs):
  - isotherm: (Time, nRMSE)  and  (Time, 1-KGE)
  - cone:     (Time, nRMSE)  and  (Time, 1-KGE)

The knee point is the frontier member farthest from the line connecting the
two extreme points of the frontier.  This gives the best speed-vs-accuracy
trade-off without requiring manual threshold tuning.

Output: a CSV with one row per frontier winner, plus the full training command
needed to retrain that model with saving enabled.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.pareto_manager import is_pareto_efficient


# ── Knee-point detection ──────────────────────────────────────────────────────

def _knee_point_index(costs: np.ndarray) -> int:
    """Return the index of the knee point on a 2-D Pareto frontier.

    Uses the maximum perpendicular distance from the line connecting the two
    extreme points of the frontier (sorted by the first objective).

    Parameters
    ----------
    costs : (N, 2) array
        Each row is (obj1, obj2), both to *minimize*.

    Returns
    -------
    int – index into the *original* ``costs`` array.
    """
    if len(costs) <= 2:
        # With ≤ 2 points, pick the one with lowest second objective.
        return int(np.argmin(costs[:, 1]))

    # Sort by first objective (time)
    order = np.argsort(costs[:, 0])
    sorted_costs = costs[order]

    # Line from first to last point
    p1 = sorted_costs[0]
    p2 = sorted_costs[-1]
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return int(order[np.argmin(sorted_costs[:, 1])])

    # Perpendicular distance of each point to the line p1→p2
    # Using 2D cross-product formula: |line_vec × (p1 - p)| / |line_vec|
    diff = p1 - sorted_costs
    dists = np.abs(line_vec[0] * diff[:, 1] - line_vec[1] * diff[:, 0]) / line_len
    knee_sorted_idx = int(np.argmax(dists))
    return int(order[knee_sorted_idx])


# ── Build retrain command from results JSON ───────────────────────────────────

_FLAG_MAP = {
    "model": "--model",
    "dataset": "--dataset",
    "n_hidden": "--n-hidden",
    "n_layers": "--n-layers",
    "n_ensemble": "--n-ensemble",
    "n_blocks": "--n-blocks",
    "activation": "--activation",
    "feature_scaler": "--feature-scaler",
    "label_scaler": "--label-scaler",
    "sc_mode": "--sc-mode",
    "rsc_prob": "--rsc-prob",
    "alpha": "--alpha",
    "gamma": "--gamma",
    "random_state": "--random-state",
}

_BOOL_FLAGS = {
    "use_log": "--use-log",
    "use_area_root": "--use-area-root",
    "direct_link": "--direct-link",
}


def _build_retrain_cmd(config: dict, output_dir: str) -> str:
    """Build a CLI command string to retrain a model from its saved config."""
    parts = ["python scripts/training/train_random_models.py"]
    for key, flag in _FLAG_MAP.items():
        val = config.get(key)
        if val is not None:
            parts.append(f"{flag} {val}")
    for key, flag in _BOOL_FLAGS.items():
        if config.get(key):
            parts.append(flag)
    parts.append(f"--output-dir {output_dir}")
    return " ".join(parts)


def _load_config_from_folder(run_dir: Path, folder: str) -> dict:
    """Load the full config dict from a results JSON inside a run folder."""
    subdir = run_dir / folder
    for jf in subdir.glob("results_*.json"):
        with open(jf) as f:
            return json.load(f).get("config", {})
    raise FileNotFoundError(f"No results_*.json in {subdir}")


# ── Main logic ────────────────────────────────────────────────────────────────

FRONTIERS = [
    ("nRMSE", "nRMSE", lambda df: df["nRMSE"].astype(float).values),
    ("1-KGE", "KGE",   lambda df: 1.0 - df["KGE"].astype(float).values),
]


def select_knee_points(
    summary_csv: str,
    run_dir: str,
    retrain_base: str = "runs",
) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    run_dir = Path(run_dir)

    rows = []

    for dataset in sorted(df["Dataset"].dropna().unique()):
        sub = df[df["Dataset"] == dataset].copy()

        for frontier_name, col_needed, cost_fn in FRONTIERS:
            fdf = sub.dropna(subset=["Time(s)", col_needed]).copy()
            if fdf.empty:
                print(f"  SKIP {dataset}/{frontier_name}: no valid rows", file=sys.stderr)
                continue

            time_vals = fdf["Time(s)"].astype(float).values
            error_vals = cost_fn(fdf)
            costs = np.column_stack([time_vals, error_vals])

            # Compute frontier
            mask = is_pareto_efficient(costs)
            frontier_df = fdf.loc[mask].copy()
            frontier_costs = costs[mask]

            if frontier_df.empty:
                continue

            # Find knee point
            knee_idx = _knee_point_index(frontier_costs)
            winner = frontier_df.iloc[knee_idx]
            folder = str(winner["Folder"])

            # Load full config from results JSON
            try:
                config = _load_config_from_folder(run_dir, folder)
            except FileNotFoundError:
                print(f"  WARN: cannot load config for {folder}", file=sys.stderr)
                continue

            out_dir = f"{retrain_base}/{dataset}_{frontier_name}_winner"
            cmd = _build_retrain_cmd(config, out_dir)

            rows.append({
                "dataset": dataset,
                "frontier": frontier_name,
                "model": str(winner.get("Model", "")),
                "folder": folder,
                "Time(s)": float(winner["Time(s)"]),
                "nRMSE": float(winner.get("nRMSE", float("nan"))),
                "KGE": float(winner.get("KGE", float("nan"))),
                "frontier_size": int(mask.sum()),
                "retrain_cmd": cmd,
            })

            print(f"  {dataset:10s} / {frontier_name:5s}  →  {winner['Model']:15s}  "
                  f"Time={winner['Time(s)']:.3f}s  "
                  f"nRMSE={winner.get('nRMSE', '?')}  "
                  f"KGE={winner.get('KGE', '?')}  "
                  f"(frontier={mask.sum()} pts, folder={folder})")

    if not rows:
        print("ERROR: no knee points found.", file=sys.stderr)
        sys.exit(1)

    result = pd.DataFrame(rows)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Select knee-point winners from Pareto frontiers "
                    "(2 datasets × 2 frontiers = 4 models)."
    )
    parser.add_argument(
        "--summary-csv", required=True,
        help="Path to summary_table.csv from sweep.",
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Sweep run directory (contains per-model subfolders with results JSON).",
    )
    parser.add_argument(
        "--retrain-base", default="runs",
        help="Base directory for retrain output folders (default: runs).",
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Output CSV path.  Default: <run-dir>/knee_point_winners.csv",
    )

    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = str(Path(args.run_dir) / "knee_point_winners.csv")

    print("Selecting knee points from Pareto frontiers...")
    result = select_knee_points(args.summary_csv, args.run_dir, args.retrain_base)

    result.to_csv(args.output_csv, index=False)
    print(f"\nWrote {len(result)} winners to: {args.output_csv}")

    print("\nRetrain commands:")
    for _, row in result.iterrows():
        print(f"  # {row['dataset']} / {row['frontier']}")
        print(f"  {row['retrain_cmd']}")
        print()


if __name__ == "__main__":
    main()
