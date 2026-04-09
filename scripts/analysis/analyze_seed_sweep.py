#!/usr/bin/env python3
"""Analyse multi-seed sweep results for random model winners.

Reads ``multi_seed_summary.json`` from each winner directory produced by
``train_random_models.py --n-seeds N``, and generates:

1. A consolidated LaTeX table (``docs/tables/seed_sweep_summary.tex``)
   with per-winner, per-target mean ± std for KGE, nRMSE, MAE, R².
2. Box-plot PDFs (``docs/plots/seed_sweep/``) showing metric distributions
   across seeds for each winner.

Usage:
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/analyze_seed_sweep.py \\
        --sweep-dir runs/seed_sweep_12345

    # Or point at individual summary files:
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/analyze_seed_sweep.py \\
        --summary-files runs/seed_sweep_12345/cone_edRVFL-SC/multi_seed_summary.json \\
                        runs/seed_sweep_12345/isotherm_SResdRVFL/multi_seed_summary.json \\
                        runs/seed_sweep_12345/isotherm_dRVFL/multi_seed_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from core.thesis_style import (
    apply_thesis_style,
    COLORS,
    FIG_SINGLE,
    MODEL_COLORS,
    save_fig,
)

apply_thesis_style()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Metrics to report (order matters for table columns)
REPORT_METRICS = ["kge", "nrmse", "mae", "r2"]

# Friendly names for LaTeX
METRIC_NAMES = {"kge": "KGE", "nrmse": "nRMSE", "mae": "MAE", "r2": r"$R^2$"}


def _find_summaries(sweep_dir: Path) -> list[Path]:
    """Discover multi_seed_summary.json files under *sweep_dir*."""
    found = sorted(sweep_dir.rglob("multi_seed_summary.json"))
    if not found:
        raise FileNotFoundError(f"No multi_seed_summary.json found under {sweep_dir}")
    return found


def _load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _label_from_config(cfg: dict) -> str:
    """Build a short human label like 'cone / edRVFL-SC'."""
    return f"{cfg['dataset']} / {cfg['model']}"


# ── LaTeX table ──────────────────────────────────────────────────────────────

def _latex_val(mean: float, std: float, fmt: str = ".4f") -> str:
    return f"${mean:{fmt}} \\pm {std:{fmt}}$"


def generate_latex_table(summaries: list[dict], out_path: Path) -> None:
    """Write a consolidated LaTeX table covering all winners and targets."""
    n_metrics = len(REPORT_METRICS)
    col_spec = "ll" + "r" * n_metrics
    header_cells = " & ".join(METRIC_NAMES[m] for m in REPORT_METRICS)

    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Multi-seed test-set performance (mean $\pm$ std) for random model winners.}",
        r"  \label{tab:seed-sweep}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    Winner & Target & {header_cells} \\\\",
        r"    \midrule",
    ]

    for s in summaries:
        cfg = s["config"]
        label = _label_from_config(cfg)
        per_target = s.get("per_target_aggregated", {})
        agg = s.get("aggregated", {})

        # Per-target rows
        targets = list(per_target.keys())
        for i, target in enumerate(targets):
            winner_cell = label if i == 0 else ""
            cells = [f"    {winner_cell}", target]
            for m in REPORT_METRICS:
                if m in per_target[target]:
                    cells.append(_latex_val(per_target[target][m]["mean"],
                                            per_target[target][m]["std"]))
                else:
                    cells.append("---")
            lines.append(" & ".join(cells) + r" \\")

        # Aggregate row
        cells = ["", r"\textit{Aggregate}"]
        for m in REPORT_METRICS:
            if m in agg:
                cells.append(_latex_val(agg[m]["mean"], agg[m]["std"]))
            else:
                cells.append("---")
        lines.append("    " + " & ".join(cells) + r" \\")
        lines.append(r"    \midrule")

    # Replace last \midrule with \bottomrule
    lines[-1] = r"    \bottomrule"
    lines += [
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("LaTeX table written to %s", out_path)


# ── Box plots ────────────────────────────────────────────────────────────────

def generate_box_plots(summaries: list[dict], out_dir: Path) -> None:
    """Generate one box-plot PDF per (winner, target) pair."""
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = [COLORS["primary"], COLORS["accent3"], COLORS["accent1"], COLORS["secondary"]]

    for s in summaries:
        cfg = s["config"]
        label = _label_from_config(cfg)
        model_name = cfg["model"]
        dataset = cfg["dataset"]
        per_target_seeds = s.get("per_seed_per_target_metrics", {})
        targets = list(per_target_seeds.keys())

        if not targets:
            logger.warning("No per-target seed data for %s, skipping plot.", label)
            continue

        for target in targets:
            seed_records = per_target_seeds[target]
            box_data = []
            box_labels = []
            for mi, m in enumerate(REPORT_METRICS):
                vals = [r[m] for r in seed_records if m in r and not np.isnan(r.get(m, float("nan")))]
                if vals:
                    box_data.append(vals)
                    box_labels.append(METRIC_NAMES[m].replace("$", ""))

            if not box_data:
                continue

            fig, ax = plt.subplots(figsize=FIG_SINGLE)
            bp = ax.boxplot(
                box_data,
                patch_artist=True,
                tick_labels=box_labels,
                medianprops={"color": COLORS["text"], "linewidth": 1.5},
                flierprops={"markersize": 3, "alpha": 0.5},
            )
            for patch, colour in zip(bp["boxes"], palette[: len(box_data)]):
                patch.set_facecolor(colour)
                patch.set_alpha(0.7)

            ax.set_ylabel("Metric value")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()

            fname = f"seed_sweep_{dataset}_{model_name}_{target}"
            save_fig(fig, out_dir / fname)
            logger.info("Box plot saved: %s", out_dir / (fname + ".pdf"))


# ── Seed metric distribution plots ───────────────────────────────────────────

DIST_METRICS = ["kge", "nrmse", "r2"]
DIST_METRIC_NAMES = {"kge": "KGE", "nrmse": "nRMSE", "r2": r"$R^2$"}


def generate_seed_distribution_plots(summaries: list[dict], out_dir: Path) -> None:
    """One PDF per (winner, metric, target): histogram + KDE."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_colors = {
        "kge": COLORS["primary"],
        "nrmse": COLORS["accent3"],
        "r2": COLORS["accent1"],
    }

    for s in summaries:
        cfg = s["config"]
        model_name = cfg["model"]
        dataset = cfg["dataset"]
        per_target_seeds = s.get("per_seed_per_target_metrics", {})
        targets = list(per_target_seeds.keys())

        if not targets:
            continue

        for target in targets:
            records = per_target_seeds[target]

            for m in DIST_METRICS:
                vals = np.array([
                    r.get(m, float("nan")) for r in records
                    if not np.isnan(r.get(m, float("nan")))
                ])
                if len(vals) == 0:
                    continue

                color = metric_colors[m]
                fig, ax = plt.subplots(figsize=FIG_SINGLE)

                # Histogram
                ax.hist(
                    vals, bins=min(80, max(20, len(vals) // 50)),
                    density=True, alpha=0.5, color=color, edgecolor="none",
                )

                # KDE overlay
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vals, bw_method="scott")
                x_grid = np.linspace(vals.min(), vals.max(), 300)
                ax.plot(x_grid, kde(x_grid), color=color, linewidth=1.5)

                # Mean + std lines
                mean, std = vals.mean(), vals.std()
                ax.axvline(mean, color=COLORS["secondary"], linewidth=1.2,
                           linestyle="--", label=f"mean={mean:.4f}")
                ax.axvline(mean - std, color=COLORS["text"], linewidth=0.8,
                           linestyle=":", alpha=0.6)
                ax.axvline(mean + std, color=COLORS["text"], linewidth=0.8,
                           linestyle=":", alpha=0.6)
                ax.legend(fontsize=7, loc="upper left")

                ax.set_ylabel("Density")
                ax.set_xlabel(DIST_METRIC_NAMES[m])
                fig.tight_layout()

                fname = f"seed_vs_metric_{dataset}_{model_name}_{m}_{target}"
                save_fig(fig, out_dir / fname)
                logger.info("Distribution plot saved: %s", out_dir / (fname + ".pdf"))


# ── Aggregate stats printout ─────────────────────────────────────────────────

def print_summary(summaries: list[dict]) -> None:
    """Log a compact summary table to stdout."""
    for s in summaries:
        cfg = s["config"]
        label = _label_from_config(cfg)
        n = s["n_seeds"]
        logger.info("=" * 70)
        logger.info("%s  (%d seeds)", label, n)
        logger.info("-" * 70)

        agg = s.get("aggregated", {})
        for m in REPORT_METRICS:
            if m in agg:
                logger.info(
                    "  Agg %5s: %.4f ± %.4f  [%.4f, %.4f]",
                    m.upper(),
                    agg[m]["mean"], agg[m]["std"], agg[m]["min"], agg[m]["max"],
                )

        for target, tm in s.get("per_target_aggregated", {}).items():
            logger.info("  Target: %s", target)
            for m in REPORT_METRICS:
                if m in tm:
                    logger.info(
                        "    %5s: %.4f ± %.4f  [%.4f, %.4f]",
                        m.upper(),
                        tm[m]["mean"], tm[m]["std"], tm[m]["min"], tm[m]["max"],
                    )
    logger.info("=" * 70)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse random model seed sweep results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sweep-dir",
        type=Path,
        help="Root directory of the seed sweep run (searches for multi_seed_summary.json).",
    )
    group.add_argument(
        "--summary-files",
        type=Path,
        nargs="+",
        help="Explicit paths to multi_seed_summary.json files.",
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "tables" / "seed_sweep_summary.tex",
        help="Output path for LaTeX table.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "plots" / "seed_sweep",
        help="Output directory for box-plot PDFs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip box-plot generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sweep_dir:
        summary_paths = _find_summaries(args.sweep_dir)
    else:
        summary_paths = args.summary_files

    logger.info("Found %d summary file(s):", len(summary_paths))
    for p in summary_paths:
        logger.info("  %s", p)

    summaries = [_load_summary(p) for p in summary_paths]

    print_summary(summaries)
    generate_latex_table(summaries, args.table_out)

    if not args.no_plots:
        generate_box_plots(summaries, args.plot_dir)
        generate_seed_distribution_plots(summaries, args.plot_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
