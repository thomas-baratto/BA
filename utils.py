import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
from torch.utils.tensorboard import SummaryWriter
import logging
from typing import List, Tuple, Dict, Any
from matplotlib.figure import Figure 
from scipy.stats import skew, kurtosis
import statsmodels.api as sm

# --- Import scikit-learn metrics ---
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_percentage_error, 
    median_absolute_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    max_error
)

# --- GPU Logging Setup ---
try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
    )
    NVIDIA_SMI_LOADED = True
except ImportError:
    NVIDIA_SMI_LOADED = False
    logging.warning("nvidia-ml-py not found. GPU monitoring will be disabled.")
except Exception as e:
    NVIDIA_SMI_LOADED = False
    logging.warning(f"Error initializing pynvml: {e}. GPU monitoring will be disabled.")

# --- Metrics ---

def compute_regression_metrics(y_true: np.ndarray, 
                               y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates a comprehensive dictionary of regression metrics on unscaled data.
    
    Args:
        y_true: Ground truth target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary containing various regression metrics
    """
    metrics = {}
    
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["rmse"] = root_mean_squared_error(y_true, y_pred)
    metrics["r2"] = r2_score(y_true, y_pred)
    metrics["medae"] = median_absolute_error(y_true, y_pred)
    metrics["mse"] = mean_squared_error(y_true, y_pred)
    metrics["explained_variance"] = explained_variance_score(y_true, y_pred)
    
    # max_error doesn't support multi-output, so compute manually
    try:
        metrics["max_error"] = max_error(y_true, y_pred)
    except ValueError:
        # For multi-output, compute max absolute error manually
        metrics["max_error"] = np.max(np.abs(y_true - y_pred))
    
    try:
        metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    except ValueError as e:
        logging.warning(f"Could not calculate MAPE: {e}")
        metrics["mape"] = float('nan')
        
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mask = y_true_flat != 0
    if np.any(mask): 
        err_rel = (y_pred_flat[mask] - y_true_flat[mask]) / y_true_flat[mask]
        metrics["rel_err_std"] = np.std(err_rel)
        metrics["rel_err_mean_abs"] = np.mean(np.abs(err_rel))
    else:
        metrics["rel_err_std"] = float('nan')
        metrics["rel_err_mean_abs"] = float('nan')

    residuals = (y_pred - y_true).flatten()
    metrics["residual_mean"] = np.mean(residuals)
    metrics["residual_std"] = np.std(residuals)
    metrics["residual_skew"] = skew(residuals)
    metrics["residual_kurtosis"] = kurtosis(residuals)
    metrics["residual_p95"] = np.percentile(residuals, 95)
    metrics["residual_p99"] = np.percentile(residuals, 99)
    
    return metrics

# --- Plotting ---
def create_residual_plots(true_values: np.ndarray, 
                          predictions: np.ndarray, 
                          labels: List[str]) -> List[Tuple[str, Figure]]:
    """Creates residual plots (residuals vs. true values) for each label."""
    figures = []
    if true_values.ndim == 1: true_values = true_values.reshape(-1, 1)
    if predictions.ndim == 1: predictions = predictions.reshape(-1, 1)

    for label_idx in range(true_values.shape[1]):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        true_col = true_values[:, label_idx]
        pred_col = predictions[:, label_idx]
        residuals = pred_col - true_col
        
        ax.scatter(true_col, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--')
        
        ax.set_xlabel("True Values")
        ax.set_ylabel("Residuals (Predicted - True)")
        ax.set_title(f"Residuals vs. True Values for {labels[label_idx]}")
        ax.grid(True)
        
        plot_name = f"Residuals_vs_True/Label_{labels[label_idx]}"
        figures.append((plot_name, fig))
        
    return figures

def create_qq_plots(true_values: np.ndarray, 
                    predictions: np.ndarray, 
                    labels: List[str]) -> List[Tuple[str, Figure]]:
    """Creates Q-Q plots of residuals to check for normality."""
    figures = []
    if true_values.ndim == 1: true_values = true_values.reshape(-1, 1)
    if predictions.ndim == 1: predictions = predictions.reshape(-1, 1)

    for label_idx in range(true_values.shape[1]):
        true_col = true_values[:, label_idx]
        pred_col = predictions[:, label_idx]
        residuals = pred_col - true_col
        
        fig = sm.qqplot(residuals, line='45', fit=True)
        plt.title(f"Q-Q Plot of Residuals for {labels[label_idx]}")
        
        plot_name = f"QQ_Plot/Label_{labels[label_idx]}"
        figures.append((plot_name, fig))
        
    return figures

def create_scatter_plot(true_values: np.ndarray, 
                        predictions: np.ndarray, 
                        labels: List[str]) -> List[Tuple[str, Figure]]:
    figures = []
    if true_values.ndim == 1: true_values = true_values.reshape(-1, 1)
    if predictions.ndim == 1: predictions = predictions.reshape(-1, 1)

    for label_idx in range(true_values.shape[1]):
        fig = plt.figure() 
        ax = fig.add_subplot(1, 1, 1)
        true_col = true_values[:, label_idx]
        pred_col = predictions[:, label_idx]
        ax.scatter(true_col, pred_col, alpha=0.5, s=10)
        min_val = min(true_col.min(), pred_col.min()) * 0.95
        max_val = max(true_col.max(), pred_col.max()) * 1.05
        lims = [min_val, max_val]
        ax.plot(lims, lims, 'r--', label="Ideal (y=x)")
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"True vs. Predicted for Label: {labels[label_idx]}")
        ax.legend()
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')
        plot_name = f"True_vs_Predicted/Label_{labels[label_idx]}"
        figures.append((plot_name, fig))
    return figures

def plot_results(rf: str, x_loss: np.ndarray, 
                 train_losses: List[float], val_losses: List[float],
                 true_values: np.ndarray, predictions: np.ndarray, labels: List[str],
                 writer: SummaryWriter = None):
    plot_dir_name = "_".join(labels).replace(" ", "_").replace("/", "_")
    plot_dir = os.path.join(rf, "plots", plot_dir_name)
    os.makedirs(plot_dir, exist_ok=True)
    fig_loss = plt.figure()
    ax = fig_loss.add_subplot(1, 1, 1)
    ax.semilogy(x_loss, train_losses, label='Training Loss')
    ax.semilogy(x_loss, val_losses, label='Validation Loss', linestyle='--')
    ax.set_title("Training vs. Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig_loss.savefig(os.path.join(plot_dir, "loss.png"))
    if writer:
        writer.add_figure("Loss_Curve", fig_loss, global_step=len(x_loss))
    plt.close(fig_loss)

    # --- Generate and Save All Plots ---
    all_plots = [
        *create_scatter_plot(true_values, predictions, labels),
        *create_residual_plots(true_values, predictions, labels),
        *create_qq_plots(true_values, predictions, labels)
    ]

    for plot_name, fig in all_plots:
        filename = plot_name.replace("/", "_") + ".png"
        fig.savefig(os.path.join(plot_dir, filename))
        if writer:
            writer.add_figure(plot_name, fig)
        plt.close(fig)

def plot_error_histograms(rf: str, 
                          predictions: np.ndarray, 
                          true_values: np.ndarray, 
                          labels: List[str],
                          writer: SummaryWriter = None):
    stats_dir_name = "_".join(labels).replace(" ", "_").replace("/", "_")
    stats_dir = os.path.join(rf, "stats", stats_dir_name)
    os.makedirs(stats_dir, exist_ok=True)
    if true_values.ndim == 1: true_values = true_values.reshape(-1, 1)
    if predictions.ndim == 1: predictions = predictions.reshape(-1, 1)
    for label_idx in range(true_values.shape[1]):
        true_col = true_values[:, label_idx]
        pred_col = predictions[:, label_idx]
        mask = true_col != 0
        if np.any(mask):
            err_rel = (pred_col[mask] - true_col[mask]) / true_col[mask]
            fig = plt.figure()
            plt.hist(err_rel, bins=100, range=(-2, 2))
            plt.title(f"Relative Error (Pred - True) / True for Label: {labels[label_idx]}")
            plt.xlabel("Relative Error")
            plt.ylabel("Frequency")
            
            filename = f"rel_err_hist_label_{labels[label_idx]}.png"
            fig.savefig(os.path.join(stats_dir, filename))
            if writer:
                writer.add_figure(f"Relative_Error_Histogram/Label_{labels[label_idx]}", fig)
            plt.close(fig)
        else:
            logging.warning(f"Skipping relative error plot for label {labels[label_idx]}; all true values are zero.")

# --- Resource Logging ---
def log_resources(writer: SummaryWriter, step: int):
    writer.add_scalar("System/CPU_Usage_Percent", psutil.cpu_percent(), step)
    writer.add_scalar("System/RAM_Usage_Percent", psutil.virtual_memory().percent, step)
    if NVIDIA_SMI_LOADED:
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            util = nvmlDeviceGetUtilizationRates(handle)
            writer.add_scalar("GPU/Memory_Used_MiB", float(mem_info.used) / (1024**2), step)
            writer.add_scalar("GPU/Memory_Used_Percent", (float(mem_info.used) / float(mem_info.total)) * 100, step)
            writer.add_scalar("GPU/Utilization_Percent", int(util.gpu), step)
            nvmlShutdown()
        except Exception as e:
            if step == 0: 
                logging.warning(f"Could not log GPU stats: {e}")