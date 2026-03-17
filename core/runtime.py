"""Runtime helpers shared by CLI scripts."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import torch


def detect_worker_id(default: str = "MAIN") -> str:
    """Return SLURM worker identifier when available."""
    return os.environ.get("SLURM_PROCID", default)


def setup_logging(level: int = logging.INFO, worker_id: Optional[str] = None) -> None:
    """Configure process logging with consistent formatting."""
    effective_worker_id = worker_id if worker_id is not None else detect_worker_id()
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s - Worker-{effective_worker_id} - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )


def get_device(no_cuda: bool = False) -> torch.device:
    """Resolve execution device from CUDA availability and user preference."""
    if no_cuda:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> str:
    """Create a directory if missing and return the same path."""
    os.makedirs(path, exist_ok=True)
    return path
