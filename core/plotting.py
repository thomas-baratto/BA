"""Plotting utilities and resource logging."""

import logging
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from matplotlib.figure import Figure
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter
import statsmodels.api as sm

from core.metrics import LABEL_UNITS
from core.thesis_style import (
    apply_thesis_style,
    COLORS,
    DPI,
    FIG_SQUARE,
    FIG_WIDE,
    FIG_SINGLE,
    FIG_TALL,
    label_with_unit,
    save_fig,
)

# Apply thesis-quality defaults for every plot created via this module
apply_thesis_style()

# --- GPU Logging Setup ---
GPU_AVAILABLE = torch.cuda.is_available()


# --- Plotting ---
def create_residual_plots(
    true_values: np.ndarray,
    predictions: np.ndarray,
    labels: list[str],
) -> list[tuple[str, Figure]]:
    """Creates residual plots (residuals vs. true values) for each label."""
    figures = []
    if true_values.ndim == 1:
        true_values = true_values.reshape(-1, 1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    for label_idx in range(true_values.shape[1]):
        fig, ax = plt.subplots(figsize=FIG_WIDE)

        true_col = true_values[:, label_idx]
        pred_col = predictions[:, label_idx]
        residuals = pred_col - true_col

        label_name = labels[label_idx]

        ax.scatter(true_col, residuals, alpha=0.4, s=12, color=COLORS["accent1"],
                   edgecolors="none")
        ax.axhline(y=0, color=COLORS["secondary"], linestyle="--", lw=1.5)

        ax.set_xlabel(label_with_unit(f"True {label_name}"))
        ax.set_ylabel(label_with_unit(f"Residuals", LABEL_UNITS.get(label_name)))
        ax.set_title(f"Residuals — {label_name}")

        plot_name = f"Residuals_vs_True/Label_{label_name}"
        figures.append((plot_name, fig))

    return figures


def create_qq_plots(
    true_values: np.ndarray,
    predictions: np.ndarray,
    labels: list[str],
    output_dir: str = None,
) -> list[tuple[str, Figure]]:
    """Creates Q-Q plots of residuals to check for normality.

    Args:
        true_values: Ground truth values
        predictions: Predicted values
        labels: List of label names
        output_dir: Optional directory to save plots to disk
    """
    figures = []
    if true_values.ndim == 1:
        true_values = true_values.reshape(-1, 1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    for label_idx in range(true_values.shape[1]):
        true_col = true_values[:, label_idx]
        pred_col = predictions[:, label_idx]
        residuals = pred_col - true_col

        fig = sm.qqplot(residuals, line="45", fit=True, markersize=3)
        ax = plt.gca()
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Q-Q Plot — {labels[label_idx]}")

        plot_name = f"QQ_Plot/Label_{labels[label_idx]}"
        figures.append((plot_name, fig))

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"qq_plot_{labels[label_idx]}.png"
            save_fig(fig, os.path.join(output_dir, filename))

    return figures


def create_regression_plots(
    true_values: np.ndarray,
    predictions: np.ndarray,
    labels: list[str],
    output_dir: str = None,
):
    """Creates and saves regression plots (predicted vs actual scatter plots).

    Args:
        true_values: Ground truth values (unscaled)
        predictions: Predicted values (unscaled)
        labels: List of label names
        output_dir: Directory to save plots
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if true_values.ndim == 1:
        true_values = true_values.reshape(-1, 1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    for label_idx in range(true_values.shape[1]):
        fig, ax = plt.subplots(figsize=FIG_SQUARE)

        true_col = true_values[:, label_idx]
        pred_col = predictions[:, label_idx]

        label_name = labels[label_idx]

        ax.scatter(true_col, pred_col, alpha=0.4, s=12, color=COLORS["primary"],
                   edgecolors="none", label="Predictions")

        # Plot ideal line
        min_val = min(true_col.min(), pred_col.min()) * 0.95
        max_val = max(true_col.max(), pred_col.max()) * 1.05
        lims = [min_val, max_val]
        ax.plot(lims, lims, color=COLORS["secondary"], ls="--", lw=1.5,
                label="Ideal ($y = x$)")

        # Compute R²
        r2 = r2_score(true_col, pred_col)
        rmse = np.sqrt(np.mean((true_col - pred_col) ** 2))

        unit = LABEL_UNITS.get(label_name, "")
        unit_suffix = f" {unit}" if unit else ""
        ax.set_xlabel(label_with_unit(f"True {label_name}"))
        ax.set_ylabel(label_with_unit(f"Predicted {label_name}"))
        ax.set_title(
            f"Regression — {label_name}\n"
            f"$R^2 = {r2:.4f}$,  RMSE $= {rmse:.2f}${unit_suffix}",
        )
        ax.legend()
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")

        if output_dir:
            filename = f"regression_plot_{label_name}.png"
            save_fig(fig, os.path.join(output_dir, filename))

    logging.info(f"Saved {len(labels)} regression plots to {output_dir}")


def create_scatter_plot(
    true_values: np.ndarray,
    predictions: np.ndarray,
    labels: list[str],
) -> list[tuple[str, Figure]]:
    figures = []
    if true_values.ndim == 1:
        true_values = true_values.reshape(-1, 1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    for label_idx in range(true_values.shape[1]):
        fig, ax = plt.subplots(figsize=FIG_SQUARE)
        true_col = true_values[:, label_idx]
        pred_col = predictions[:, label_idx]

        label_name = labels[label_idx]

        ax.scatter(true_col, pred_col, alpha=0.4, s=12, color=COLORS["primary"],
                   edgecolors="none")
        min_val = min(true_col.min(), pred_col.min()) * 0.95
        max_val = max(true_col.max(), pred_col.max()) * 1.05
        lims = [min_val, max_val]
        ax.plot(lims, lims, color=COLORS["secondary"], ls="--", lw=1.5,
                label="Ideal ($y = x$)")
        ax.set_xlabel(label_with_unit(f"True {label_name}"))
        ax.set_ylabel(label_with_unit(f"Predicted {label_name}"))
        ax.set_title(f"Predicted vs. True — {label_name}")
        ax.legend()
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        plot_name = f"True_vs_Predicted/Label_{label_name}"
        figures.append((plot_name, fig))
    return figures


def plot_results(
    rf: str,
    x_loss: np.ndarray,
    train_losses: list[float],
    val_losses: list[float],
    true_values: np.ndarray,
    predictions: np.ndarray,
    labels: list[str],
    writer: SummaryWriter = None,
):
    plot_dir_name = "_".join(labels).replace(" ", "_").replace("/", "_")
    plot_dir = os.path.join(rf, "plots", plot_dir_name)
    os.makedirs(plot_dir, exist_ok=True)
    fig_loss, ax = plt.subplots(figsize=FIG_WIDE)
    ax.semilogy(x_loss, train_losses, label="Training Loss",
                color=COLORS["primary"])
    ax.semilogy(x_loss, val_losses, label="Validation Loss",
                color=COLORS["accent1"], ls="--")
    ax.set_title("Training vs. Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.legend()
    if writer:
        writer.add_figure("Loss_Curve", fig_loss, global_step=len(x_loss))
    save_fig(fig_loss, os.path.join(plot_dir, "loss.png"))

    # --- Generate and Save All Plots ---
    all_plots = [
        *create_scatter_plot(true_values, predictions, labels),
        *create_residual_plots(true_values, predictions, labels),
        *create_qq_plots(true_values, predictions, labels),
    ]

    for plot_name, fig in all_plots:
        filename = plot_name.replace("/", "_") + ".png"
        if writer:
            writer.add_figure(plot_name, fig)
        save_fig(fig, os.path.join(plot_dir, filename))


def plot_split_metric_bars(
    rf: str,
    labels: list[str],
    split_metrics: dict,
    writer: SummaryWriter = None,
    metric_keys: tuple[str, ...] = ("mae", "rmse"),
):
    """Visualize per-label metrics across validation/test splits to assess overfitting."""
    if not split_metrics:
        return

    plot_dir = os.path.join(rf, "plots", "split_metrics")
    os.makedirs(plot_dir, exist_ok=True)
    splits = list(split_metrics.keys())
    if not splits or not labels:
        return

    x = np.arange(len(labels))
    bar_width = 0.75 / max(1, len(splits))

    for metric in metric_keys:
        fig, ax = plt.subplots(figsize=FIG_WIDE)
        for idx, split in enumerate(splits):
            per_label_values = [
                split_metrics[split]["per_label"]
                .get(label, {})
                .get(metric, np.nan)
                for label in labels
            ]
            offset = (idx - (len(splits) - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                per_label_values,
                width=bar_width,
                label=split.title(),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")

        # Get unit from first label
        unit = LABEL_UNITS.get(labels[0], "") if labels else ""
        unit_str = f" ({unit})" if unit else ""

        ax.set_ylabel(f"{metric.upper()}{unit_str}")
        ax.set_title(f"{metric.upper()} per Label across Splits")
        ax.legend()
        fig.tight_layout()

        if writer:
            writer.add_figure(f"Metrics/{metric.upper()}_per_label", fig)
        save_fig(fig, os.path.join(plot_dir, f"{metric}_per_label.png"))


# --- Resource Logging ---


class ResourceLogger:
    """Logger for tracking CPU, RAM, and GPU usage over time."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        self.logs = {
            "step": [],
            "cpu_percent": [],
            "ram_percent": [],
            "gpu_memory_allocated_mib": [],
            "gpu_memory_reserved_mib": [],
            "gpu_memory_percent": [],
        }
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def log(self, step: int):
        self.logs["step"].append(step)
        self.logs["cpu_percent"].append(psutil.cpu_percent())
        self.logs["ram_percent"].append(psutil.virtual_memory().percent)

        if GPU_AVAILABLE:
            try:
                current_device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(current_device) / (
                    1024**2
                )
                reserved = torch.cuda.memory_reserved(current_device) / (
                    1024**2
                )
                total_memory = torch.cuda.get_device_properties(
                    current_device
                ).total_memory / (1024**2)
                percent = (allocated / total_memory) * 100

                self.logs["gpu_memory_allocated_mib"].append(allocated)
                self.logs["gpu_memory_reserved_mib"].append(reserved)
                self.logs["gpu_memory_percent"].append(percent)
            except Exception:
                self.logs["gpu_memory_allocated_mib"].append(None)
                self.logs["gpu_memory_reserved_mib"].append(None)
                self.logs["gpu_memory_percent"].append(None)
        else:
            self.logs["gpu_memory_allocated_mib"].append(None)
            self.logs["gpu_memory_reserved_mib"].append(None)
            self.logs["gpu_memory_percent"].append(None)

    def save(self):
        if not self.output_dir or len(self.logs["step"]) == 0:
            return

        # Save to JSON
        json_path = os.path.join(self.output_dir, "resource_usage.json")
        with open(json_path, "w") as f:
            json.dump(self.logs, f, indent=2)
        logging.info(f"Saved resource usage logs to {json_path}")

        # Save to text file with summary statistics
        txt_path = os.path.join(self.output_dir, "resource_usage.txt")
        with open(txt_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("RESOURCE USAGE SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total steps logged: {len(self.logs['step'])}\n\n")

            f.write("CPU Usage (%):\n")
            f.write(f"  Mean: {np.mean(self.logs['cpu_percent']):.2f}\n")
            f.write(f"  Max:  {np.max(self.logs['cpu_percent']):.2f}\n")
            f.write(f"  Min:  {np.min(self.logs['cpu_percent']):.2f}\n\n")

            f.write("RAM Usage (%):\n")
            f.write(f"  Mean: {np.mean(self.logs['ram_percent']):.2f}\n")
            f.write(f"  Max:  {np.max(self.logs['ram_percent']):.2f}\n")
            f.write(f"  Min:  {np.min(self.logs['ram_percent']):.2f}\n\n")

            if GPU_AVAILABLE and any(
                x is not None for x in self.logs["gpu_memory_percent"]
            ):
                gpu_data = [
                    x
                    for x in self.logs["gpu_memory_percent"]
                    if x is not None
                ]
                f.write("GPU Memory Usage (%):\n")
                f.write(f"  Mean: {np.mean(gpu_data):.2f}\n")
                f.write(f"  Max:  {np.max(gpu_data):.2f}\n")
                f.write(f"  Min:  {np.min(gpu_data):.2f}\n\n")

                gpu_allocated = [
                    x
                    for x in self.logs["gpu_memory_allocated_mib"]
                    if x is not None
                ]
                f.write("GPU Memory Allocated (MiB):\n")
                f.write(f"  Mean: {np.mean(gpu_allocated):.2f}\n")
                f.write(f"  Max:  {np.max(gpu_allocated):.2f}\n")
                f.write(f"  Min:  {np.min(gpu_allocated):.2f}\n")
            else:
                f.write("GPU: Not available or not logged\n")

        logging.info(f"Saved resource usage summary to {txt_path}")

        # Create plots
        self._create_plots()

    def _create_plots(self):
        if not self.output_dir or len(self.logs["step"]) == 0:
            return

        steps = self.logs["step"]

        # Plot 1: CPU and RAM usage
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(
            steps,
            self.logs["cpu_percent"],
            label="CPU Usage (%)",
            color="blue",
        )
        ax1.set_xlabel("Step/Epoch")
        ax1.set_ylabel("CPU Usage (%)")
        ax1.set_title("CPU Usage Over Time")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(
            steps,
            self.logs["ram_percent"],
            label="RAM Usage (%)",
            color="green",
        )
        ax2.set_xlabel("Step/Epoch")
        ax2.set_ylabel("RAM Usage (%)")
        ax2.set_title("RAM Usage Over Time")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        cpu_ram_path = os.path.join(self.output_dir, "cpu_ram_usage.png")
        save_fig(fig, cpu_ram_path)
        logging.info(f"Saved CPU/RAM usage plot to {cpu_ram_path}")

        # Plot 2: GPU usage (if available)
        if GPU_AVAILABLE and any(
            x is not None for x in self.logs["gpu_memory_percent"]
        ):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            valid_indices = [
                i
                for i, x in enumerate(self.logs["gpu_memory_percent"])
                if x is not None
            ]
            valid_steps = [steps[i] for i in valid_indices]
            valid_percent = [
                self.logs["gpu_memory_percent"][i] for i in valid_indices
            ]
            valid_allocated = [
                self.logs["gpu_memory_allocated_mib"][i]
                for i in valid_indices
            ]

            ax1.plot(
                valid_steps,
                valid_percent,
                label="GPU Memory (%)",
                color="red",
            )
            ax1.set_xlabel("Step/Epoch")
            ax1.set_ylabel("GPU Memory Usage (%)")
            ax1.set_title("GPU Memory Usage Over Time")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(
                valid_steps,
                valid_allocated,
                label="GPU Memory Allocated (MiB)",
                color="orange",
            )
            ax2.set_xlabel("Step/Epoch")
            ax2.set_ylabel("Memory (MiB)")
            ax2.set_title("GPU Memory Allocated Over Time")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            gpu_path = os.path.join(self.output_dir, "gpu_usage.png")
            save_fig(fig, gpu_path)
            logging.info(f"Saved GPU usage plot to {gpu_path}")
