import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
from torch.utils.tensorboard import SummaryWriter
import logging
from typing import List
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates 
# --- GPU Logging Setup ---
try:
    import pynvml
    NVIDIA_SMI_LOADED = True
except ImportError:
    NVIDIA_SMI_LOADED = False
    logging.warning("pynvml not found. GPU monitoring will be disabled.")
except Exception as e:
    NVIDIA_SMI_LOADED = False
    logging.warning(f"Error initializing pynvml: {e}. GPU monitoring will be disabled.")

# --- Plotting & Error Metrics ---

def plot_results(rf: str, x_loss: np.ndarray, y_loss: List[float],
                 true_values: np.ndarray, predictions: np.ndarray, labels: List[str]):
    """Saves loss and prediction plots."""
    
    # Use label names to create a unique, filesystem-friendly directory name
    plot_dir_name = "_".join(labels).replace(" ", "_").replace("/", "_")
    plot_dir = os.path.join(rf, "plots", plot_dir_name)
    os.makedirs(plot_dir, exist_ok=True)

    # --- Loss Plot ---
    plt.figure()
    plt.semilogy(x_loss, y_loss)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(plot_dir, "loss.png"))
    plt.close()

    # Ensure arrays are 2D for consistent indexing
    if true_values.ndim == 1:
        true_values = true_values.reshape(-1, 1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    # --- True vs. Predicted Plots (one per label) ---
    for label_idx in range(true_values.shape[1]):
        plt.figure()
        true_col = true_values[:, label_idx]
        pred_col = predictions[:, label_idx]
        
        plt.scatter(true_col, pred_col, alpha=0.5, s=10) # Smaller points for clarity
        
        # Determine plot limits to make it square
        min_val = min(true_col.min(), pred_col.min()) * 0.95
        max_val = max(true_col.max(), pred_col.max()) * 1.05
        lims = [min_val, max_val]
        
        plt.plot(lims, lims, 'r--', label="Ideal (y=x)")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"True vs. Predicted for Label: {labels[label_idx]}")
        plt.legend()
        plt.xlim(lims)
        plt.ylim(lims)
        plt.gca().set_aspect('equal', adjustable='box') # Make plot square
        plt.savefig(os.path.join(plot_dir, f"true_pred_label_{labels[label_idx]}.png"))
        plt.close()

def relative_error(rf: str, predictions: np.ndarray, true_values: np.ndarray, labels: List[str]):
    """Calculates and plots relative error histograms."""
    
    # Use label names for directory
    stats_dir_name = "_".join(labels).replace(" ", "_").replace("/", "_")
    stats_dir = os.path.join(rf, "stats", stats_dir_name)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Ensure arrays are 2D
    if true_values.ndim == 1:
        true_values = true_values.reshape(-1, 1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    with open(os.path.join(stats_dir, "errors.txt"), "w") as f:
        for label_idx in range(true_values.shape[1]):
            true_col = true_values[:, label_idx]
            pred_col = predictions[:, label_idx]
            
            # Avoid division by zero, though log-transform + expm1 makes 0 unlikely
            mask = true_col != 0
            err_rel = (pred_col[mask] - true_col[mask]) / true_col[mask]
            
            std_dev = np.std(err_rel)
            mean_abs_err = np.mean(np.abs(err_rel))
            
            f.write(f"--- Label: {labels[label_idx]} ---\n")
            f.write(f"Std Dev of Relative Error: {std_dev:.4f}\n")
            f.write(f"Mean Absolute Relative Error: {mean_abs_err:.4f}\n\n")

            plt.figure()
            plt.hist(err_rel, bins=100, range=(-2, 2)) # Cap range for readability
            plt.title(f"Relative Error (Pred - True) / True for Label: {labels[label_idx]}")
            plt.xlabel("Relative Error")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(stats_dir, f"rel_err_hist_label_{labels[label_idx]}.png"))
            plt.close()

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
            writer.add_scalar("GPU/Memory_Used_MB", float(mem_info.used) / (1024**2), step)
            writer.add_scalar("GPU/Memory_Used_Percent", (float(mem_info.used) / float(mem_info.total)) * 100, step)
            writer.add_scalar("GPU/Utilization_Percent", util.gpu, step)
            nvmlShutdown()
        except Exception as e:
            # Log only once to avoid spamming
            if step == 0: 
                logging.warning(f"Could not log GPU stats: {e}")