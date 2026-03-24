#!/usr/bin/env python3
"""Retrain all random model winners from stored configs.

This script retrains the 3 random model winners (cone, isotherm nRMSE,
isotherm KGE) using the exact hyperparameters stored in
``config/random_model_winners.json``.  Since all models use deterministic seeds
(random_state=42) the resulting ``model.pkl`` files are bit-identical to the
originals that were trained on GPU during the sweep.

Run after a fresh ``pip install`` or ``git clone`` to recreate the model binaries
that are excluded from version control (they can be > 200 MB each).

Usage:
    PYTHONPATH=. .venv/env/bin/python scripts/deployment/retrain_random_models.py
    PYTHONPATH=. .venv/env/bin/python scripts/deployment/retrain_random_models.py --only cone
    PYTHONPATH=. .venv/env/bin/python scripts/deployment/retrain_random_models.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (two levels up from this file → BA/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Winner output directories (relative to project root)
WINNER_DIRS: dict[str, Path] = {
    "cone": PROJECT_ROOT / "artifacts/models/random/cone/winner",
    "isotherm_nRMSE": PROJECT_ROOT / "artifacts/models/random/isotherm/nRMSE_winner",
    "isotherm_KGE": PROJECT_ROOT / "artifacts/models/random/isotherm/KGE_winner",
}

CONFIGS_FILE = PROJECT_ROOT / "config/random_model_winners.json"


def _load_winner_configs() -> dict:
    with open(CONFIGS_FILE) as f:
        return json.load(f)


def _config_to_cli_args(key: str, cfg: dict, output_dir: Path) -> list[str]:
    """Convert a winner config dict into CLI args for ``train_random_models.py``."""
    args: list[str] = [
        "--model", cfg["model"],
        "--dataset", cfg["dataset"],
        "--feature-scaler", cfg["feature_scaler"],
        "--label-scaler", cfg["label_scaler"],
        "--n-hidden", str(cfg["n_hidden"]),
        "--activation", cfg["activation"],
        "--alpha", str(cfg["alpha"]),
        "--gamma", str(cfg["gamma"]),
        "--n-layers", str(cfg["n_layers"]),
        "--n-ensemble", str(cfg["n_ensemble"]),
        "--sc-mode", cfg["sc_mode"],
        "--rsc-prob", str(cfg["rsc_prob"]),
        "--n-folds", str(cfg["n_folds"]),
        "--n-blocks", str(cfg["n_blocks"]),
        "--random-state", str(cfg["random_state"]),
        "--n-seeds", str(cfg["n_seeds"]),
        "--output-dir", str(output_dir),
    ]

    if cfg.get("use_log"):
        args.append("--use-log")
    if cfg.get("use_area_root"):
        args.append("--use-area-root")
    if cfg.get("direct_link"):
        args.append("--direct-link")

    return args


def _needs_retrain(output_dir: Path, force: bool) -> bool:
    """Return True if model.pkl is missing or *force* is set."""
    if force:
        return True
    model_file = output_dir / "model.pkl"
    return not model_file.exists()


def retrain_one(key: str, cfg: dict, output_dir: Path) -> float:
    """Retrain a single winner model.  Returns wall-clock seconds."""
    # We import train_random_models here so PYTHONPATH only needs to be set
    # once, at the top level.
    import scripts.training.train_random_models as trm

    cli_args = _config_to_cli_args(key, cfg, output_dir)
    # Monkey-patch sys.argv so argparse picks up our args
    old_argv = sys.argv
    sys.argv = ["train_random_models.py"] + cli_args

    t0 = time.time()
    try:
        trm.main()
    finally:
        sys.argv = old_argv
    return time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain random model winners from stored configs"
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        choices=list(WINNER_DIRS),
        default=None,
        help="Retrain only the specified winner(s). Default: all four.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if model.pkl already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    configs = _load_winner_configs()
    keys_to_train = args.only or list(WINNER_DIRS)

    total_time = 0.0
    skipped = 0
    trained = 0

    for key in keys_to_train:
        output_dir = WINNER_DIRS[key]
        cfg = configs[key]

        if not _needs_retrain(output_dir, args.force):
            logging.info("SKIP  %s — model.pkl already exists (use --force to overwrite)", key)
            skipped += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info("TRAIN %s (%s on %s)", key, cfg["model"], cfg["dataset"])
        elapsed = retrain_one(key, cfg, output_dir)
        total_time += elapsed
        trained += 1
        logging.info("DONE  %s in %.1fs", key, elapsed)

    logging.info("")
    logging.info("=" * 60)
    logging.info(
        "Finished: %d trained, %d skipped (%.1fs total)",
        trained, skipped, total_time,
    )
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
