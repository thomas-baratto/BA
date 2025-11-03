import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
from torch.utils.tensorboard import SummaryWriter
import logging
from typing import List, Tuple, Dict, Any
from matplotlib.figure import Figure # Import Figure for type hinting

# --- Import scikit-learn metrics ---
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_percentage_error, 
    median_absolute_error,
    mean_absolute_error
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
    Calculates a dictionary of regression metrics on unscaled data.
    """
    metrics = {}
    
    # Calculate metrics that are safe (don't divide by zero)
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
    metrics["r2"] = r2_score(y_true, y_pred)
    metrics["medae"] = median_absolute_error(y_true, y_pred)
    
    # Calculate percentage-based metrics safely
    try:
        metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    except ValueError as e:
        logging.warning(f"Could not calculate MAPE: {e}")
        metrics["mape"] = float('nan')
        
    # Calculate relative error stats
    # Ensure y_true and y_pred are flattened for safe masking
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mask = y_true_flat != 0
    if np.any(mask): # Only calculate if there are non-zero elements
        err_rel = (y_pred_flat[mask] - y_true_flat[mask]) / y_true_flat[mask]
        metrics["rel_err_std"] = np.std(err_rel)
        metrics["rel_err_mean_abs"] = np.mean(np.abs(err_rel))
    else:
        metrics["rel_err_std"] = float('nan')
        metrics["rel_err_mean_abs"] = float('nan')
    
    return metrics

# --- Plotting ---

def create_scatter_plot(true_values: np.ndarray, 
                        predictions: np.ndarray, 
                        labels: List[str]) -> List[Tuple[str, Figure]]:
    """
    Creates a list of (name, figure) tuples for the true vs. predicted scatter plots.
    """
    figures = []
    
    # Ensure arrays are 2D
    if true_values.ndim == 1: true_values = true_values.reshape(-1, 1)
    if predictions.ndim == 1: predictions = predictions.reshape(-1, 1)

    for label_idx in range(true_values.shape[1]):
        fig = plt.figure() # Create a new figure
        ax = fig.add_subplot(1, 1, 1) # Add axes
        
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
        
        # We don't call plt.close() here because trainer.py needs the fig object
        
    return figures

def plot_results(rf: str, x_loss: np.ndarray, 
                 train_losses: List[float], val_losses: List[float],
                 true_values: np.ndarray, predictions: np.ndarray, labels: List[str]):
    """Saves loss and prediction plots to disk."""
    
    plot_dir_name = "_".join(labels).replace(" ", "_").replace("/", "_")
    plot_dir = os.path.join(rf, "plots", plot_dir_name)
    os.makedirs(plot_dir, exist_ok=True)

    # --- Loss Plot ---
    fig_loss = plt.figure() # Create figure
    ax = fig_loss.add_subplot(1, 1, 1)
    ax.semilogy(x_loss, train_losses, label='Training Loss')
    ax.semilogy(x_loss, val_losses, label='Validation Loss', linestyle='--')
    ax.set_title("Training vs. Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig_loss.savefig(os.path.join(plot_dir, "loss.png"))
    plt.close(fig_loss) # Close the specific figure

    # --- True vs. Predicted Plots ---
    # Use the helper to create the plots, then save and close them
    scatter_plots = create_scatter_plot(true_values, predictions, labels)
    for plot_name, fig in scatter_plots:
        # Sanitize filename
        filename = plot_name.replace("/", "_") + ".png"
        fig.savefig(os.path.join(plot_dir, filename))
        plt.close(fig) # Close the figure

def plot_error_histograms(rf: str, 
                          predictions: np.ndarray, 
                          true_values: np.ndarray, 
                          labels: List[str]):
    """
    Calculates and plots relative error histograms.
    """
    
    stats_dir_name = "_".join(labels).replace(" ", "_").replace("/", "_")
    stats_dir = os.path.join(rf, "stats", stats_dir_name)
    os.makedirs(stats_dir, exist_ok=True)
    
    if true_values.ndim == 1: true_values = true_values.reshape(-1, 1)
    if predictions.ndim == 1: predictions = predictions.reshape(-1, 1)

    # We just write the plots here. Metrics are calculated elsewhere.
    for label_idx in range(true_values.shape[1]):
        true_col = true_values[:, label_idx]
        pred_col = predictions[:, label_idx]
        
        mask = true_col != 0
        if np.any(mask):
            err_rel = (pred_col[mask] - true_col[mask]) / true_col[mask]

            plt.figure()
            plt.hist(err_rel, bins=100, range=(-2, 2))
            plt.title(f"Relative Error (Pred - True) / True for Label: {labels[label_idx]}")
            plt.xlabel("Relative Error")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(stats_dir, f"rel_err_hist_label_{labels[label_idx]}.png"))
            plt.close()
        else:
            logging.warning(f"Skipping relative error plot for label {labels[label_idx]}; all true values are zero.")


# --- Resource Logging ---

def log_resources(writer: SummaryWriter, step: int):
    """Logs CPU, RAM, and (if available) GPU usage to TensorBoard."""
    writer.add_scalar("System/CPU_Usage_Percent", psutil.cpu_percent(), step)
    writer.add_scalar("System/RAM_Usage_Percent", psutil.virtual_memory().percent, step)

    if NVIDIA_SMI_LOADED:
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0) # Assumes GPU 0
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            util = nvmlDeviceGetUtilizationRates(handle)
            writer.add_scalar("GPU/Memory_Used_MiB", float(mem_info.used) / (1024**2), step)
            writer.add_scalar("GPU/Memory_Used_Percent", (float(mem_info.used) / float(mem_info.total)) * 100, step)
            writer.add_scalar("GPU/Utilization_Percent", int(util.gpu), step)
            nvmlShutdown()
        except Exception as e:
            # Log only once to avoid spamming
            if step == 0: 
                logging.warning(f"Could not log GPU stats: {e}")