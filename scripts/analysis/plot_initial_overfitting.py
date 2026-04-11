#!/usr/bin/env python3
"""Generate initial overfitting proof-of-concept plots.

Trains MLP and ELM models on tiny subsets (1 and 10 samples) and evaluates
on the *same* data to demonstrate that the architectures can memorize perfectly.

Produces for each (model_type × dataset × n_samples):
  - Predicted vs True scatter plot
  - Training loss curve (MLP only — ELM has no iterative training)

Usage:
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/plot_initial_overfitting.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.datasets import DATASET_CONFIGS
from core.data_loader import load_file, select_columns
from core.model import NeuralNetwork
from core.random.ELM import ELM
from core.thesis_style import (
    apply_thesis_style,
    COLORS,
    FIG_SQUARE,
    label_with_unit,
    save_fig,
)

apply_thesis_style()

OUTPUT_DIR = PROJECT_ROOT / "docs" / "plots" / "initial_overfitting"

DATASETS = {
    "cone": "Cone",
    "isotherm": "Isotherm",
}

SAMPLE_SIZES = [1, 10]

# ── MLP training ────────────────────────────────────────────────────────────


def train_mlp_overfit(
    X: np.ndarray,
    y: np.ndarray,
    num_epochs: int = 5000,
    lr: float = 1e-3,
) -> tuple[NeuralNetwork, list[float], dict]:
    """Train a small MLP to memorize the given data.

    Internally normalises X and y (mean/std) so the loss landscape is
    well-conditioned regardless of physical scales.

    Returns (model, loss_history, norm_stats).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalise inputs and outputs
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-8
    X_n = (X - X_mean) / X_std
    y_n = (y - y_mean) / y_std

    model = NeuralNetwork(
        input_size=X.shape[1],
        output_size=y.shape[1],
        nr_hidden_layers=3,
        nr_neurons=64,
        activation_name="ReLU",
        dropout_rate=0.0,
        use_batchnorm=False,
    ).to(device)

    X_t = torch.tensor(X_n, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_n, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses: list[float] = []
    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    norm_stats = {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}
    return model, losses, norm_stats


def predict_mlp(
    model: NeuralNetwork, X: np.ndarray, norm_stats: dict
) -> np.ndarray:
    """Predict in original (physical) units."""
    device = next(model.parameters()).device
    X_n = (X - norm_stats["X_mean"]) / norm_stats["X_std"]
    X_t = torch.tensor(X_n, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_n = model(X_t).cpu().numpy()
    return y_n * norm_stats["y_std"] + norm_stats["y_mean"]


# ── ELM training ────────────────────────────────────────────────────────────


def train_elm_overfit(
    X: np.ndarray, y: np.ndarray
) -> tuple[ELM, dict]:
    """Train an ELM to memorize the data (closed-form, no loss history).

    Normalises X and y internally so the random projection operates on
    features with comparable scales.

    Returns (model, norm_stats).
    """
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-8
    X_n = (X - X_mean) / X_std
    y_n = (y - y_mean) / y_std

    model = ELM(n_hidden=256, activation="ReLU", alpha=1e-6, random_state=42)
    model.fit(X_n, y_n)
    norm_stats = {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}
    return model, norm_stats


def predict_elm(
    model: ELM, X: np.ndarray, norm_stats: dict
) -> np.ndarray:
    """Predict in original (physical) units."""
    X_n = (X - norm_stats["X_mean"]) / norm_stats["X_std"]
    y_n = model.predict(X_n)
    return y_n * norm_stats["y_std"] + norm_stats["y_mean"]


# ── Plotting helpers ────────────────────────────────────────────────────────


def plot_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Predicted vs True scatter — one subplot per label."""
    n_labels = len(label_names)
    fig, axes = plt.subplots(1, n_labels, figsize=(5.5 * n_labels, 5.5), squeeze=False)

    for i, lbl in enumerate(label_names):
        ax = axes[0, i]
        yt = y_true[:, i] if y_true.ndim > 1 else y_true
        yp = y_pred[:, i] if y_pred.ndim > 1 else y_pred

        ax.scatter(yt, yp, s=80, color=COLORS["primary"], edgecolors="white",
                   linewidths=0.5, zorder=3)

        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        margin = max((hi - lo) * 0.1, abs(hi) * 0.01)
        lo -= margin
        hi += margin
        ax.plot([lo, hi], [lo, hi], color=COLORS["secondary"], ls="--", lw=1.5,
                label="Ideal ($y = x$)", zorder=2)

        ax.set_xlabel(label_with_unit(f"True {lbl}"))
        ax.set_ylabel(label_with_unit(f"Predicted {lbl}"))
        ax.legend(fontsize=8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    save_fig(fig, out_path)


def plot_loss_curve(
    losses: list[float],
    title: str,
    out_path: Path,
) -> None:
    """Training loss over epochs (log scale)."""
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    epochs = np.arange(1, len(losses) + 1)

    ax.semilogy(epochs, losses, linewidth=1.2, color=COLORS["primary"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    fig.tight_layout()
    save_fig(fig, out_path)


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    for ds_key, ds_label in DATASETS.items():
        cfg = DATASET_CONFIGS[ds_key]
        df = load_file(cfg["csv_file"])
        X_all, y_all = select_columns(df, cfg["features"], cfg["labels"])
        label_names = cfg["labels"]

        for n in SAMPLE_SIZES:
            X, y = X_all[:n], y_all[:n]
            tag = f"{ds_key}_{n}pt"
            out = OUTPUT_DIR / ds_key
            print(f"\n{'='*60}")
            print(f"  {ds_label} dataset — {n} sample(s)")
            print(f"{'='*60}")

            # ── MLP ─────────────────────────────────────────────────────
            print(f"  Training MLP on {n} sample(s)...")
            mlp, losses, norm_stats = train_mlp_overfit(X, y)
            y_pred_mlp = predict_mlp(mlp, X, norm_stats)

            plot_pred_vs_true(
                y, y_pred_mlp, label_names,
                title=f"MLP Overfit — {ds_label} ({n} sample{'s' if n > 1 else ''})",
                out_path=out / f"mlp_{tag}_pred_vs_true",
            )
            plot_loss_curve(
                losses,
                title=f"MLP Training Loss — {ds_label} ({n} sample{'s' if n > 1 else ''})",
                out_path=out / f"mlp_{tag}_loss",
            )
            final_loss = losses[-1]
            print(f"  MLP final loss: {final_loss:.2e}")

            # ── ELM ─────────────────────────────────────────────────────
            print(f"  Training ELM on {n} sample(s)...")
            elm, elm_norm = train_elm_overfit(X, y)
            y_pred_elm = predict_elm(elm, X, elm_norm)
            if y_pred_elm.ndim == 1:
                y_pred_elm = y_pred_elm.reshape(-1, 1)

            plot_pred_vs_true(
                y, y_pred_elm, label_names,
                title=f"ELM Overfit — {ds_label} ({n} sample{'s' if n > 1 else ''})",
                out_path=out / f"elm_{tag}_pred_vs_true",
            )
            elm_mse = float(np.mean((y - y_pred_elm) ** 2))
            print(f"  ELM MSE: {elm_mse:.2e}")

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
