import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR, ReduceLROnPlateau
from torchinfo import summary
import numpy as np 
import logging
import os
from typing import Tuple, Dict, Any, List
import time
import copy
import json
import math
from sklearn.model_selection import train_test_split

from core.model import NeuralNetwork
from core.data_loader import CSVDataset, load_data
from core.utils import (
    plot_results, 
    compute_regression_metrics,
    plot_split_metric_bars,
    ResourceLogger
)

def train_epoch(model: nn.Module,
                loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                max_grad_norm: float = 1.0) -> float:
    """
    Train for one epoch with optional gradient clipping.
    """
    model.train()
    epoch_loss = 0.0
    for batch_data, batch_labels in loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_true_values = []
    with torch.no_grad():
        for batch_data, batch_labels in loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_true_values.append(batch_labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    predictions = np.concatenate(all_predictions, axis=0)
    true_values = np.concatenate(all_true_values, axis=0)
    return avg_loss, predictions, true_values

def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> Any:
    """
    Create learning rate scheduler based on configuration.
    """
    warmup_epochs = config.get("warmup_epochs", 0)
    scheduler_type = config.get("scheduler_type", "CosineAnnealingLR")
    
    if scheduler_type == "CosineAnnealingLR":
        main_scheduler = CosineAnnealingLR(optimizer, T_max=config.get("scheduler_T_max", 1000))
        logging.info(f"Using CosineAnnealingLR with T_max={config.get('scheduler_T_max', 1000)}")
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
            logging.info(f"Added warmup for {warmup_epochs} epochs")
        else:
            scheduler = main_scheduler
    
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        # Gentler annealing with periodic restarts - good for noisy validation
        T_0 = config.get("scheduler_T_0", 50)  # Initial restart period
        T_mult = config.get("scheduler_T_mult", 2)  # Period multiplier after each restart
        eta_min = config.get("scheduler_eta_min", 1e-6)  # Minimum learning rate
        main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
        logging.info(f"Using CosineAnnealingWarmRestarts with T_0={T_0}, T_mult={T_mult}, eta_min={eta_min}")
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
            logging.info(f"Added warmup for {warmup_epochs} epochs")
        else:
            scheduler = main_scheduler
    
    elif scheduler_type == "ReduceLROnPlateau":
        # Adaptive scheduler based on validation loss - tuned for noisy metrics
        mode = 'min'
        factor = config.get("scheduler_factor", 0.7)  # Gentler reduction
        patience_plateau = config.get("scheduler_patience", 30)
        cooldown = config.get("scheduler_cooldown", 5)
        min_lr = config.get("scheduler_min_lr", 1e-6)
        threshold = config.get("scheduler_threshold", 1e-3)
        threshold_mode = config.get("scheduler_threshold_mode", 'rel')  # relative improvement
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience_plateau,
            cooldown=cooldown,
            min_lr=min_lr,
            threshold=threshold,
            threshold_mode=threshold_mode,
            verbose=True
        )
        logging.info(
            f"Using ReduceLROnPlateau(factor={factor}, patience={patience_plateau}, "
            f"cooldown={cooldown}, min_lr={min_lr}, threshold={threshold} {threshold_mode})"
        )
        # Log warmup info for plateau scheduler
        if warmup_epochs > 0:
            logging.info(f"Manual LR warmup for {warmup_epochs} epochs before plateau scheduling")
    
    else:
        # Default StepLR
        step_size = config.get("lr_scheduler_epoch_step", 100)
        gamma = config.get("lr_scheduler_gamma", 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logging.info(f"Using default StepLR with step_size={step_size}, gamma={gamma}")
    
    return scheduler


# --- Main Training Function ---
def main_train(config: Dict[str, Any],
               rf: str,
               csv_file: str,
               feature_cols: List[str],
               label_cols: List[str],
               device: torch.device) -> nn.Module:
    
    logging.info(f"Starting main training for labels {label_cols}...")
    X_train_full, X_test, X_scaler, y_train_full, y_test, y_scaler = load_data(
        csv_file,
        feature_cols=feature_cols,
        label_cols=label_cols,
        plots=config["plots"],
        rf=rf,
        feature_scaler_type=config.get("feature_scaler_type", "minmax"),
        label_scaler_type=config.get("label_scaler_type", "minmax")
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    logging.info(f"Final training: {len(X_train)} train samples, {len(X_val)} val samples, {len(X_test)} test samples.")
    train_dataset = CSVDataset(X_train, y_train)
    val_dataset = CSVDataset(X_val, y_val)
    test_dataset = CSVDataset(X_test, y_test)
    
    # CRITICAL FIX: num_workers=0 to prevent contention
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False,
                             num_workers=0, pin_memory=True)
    
    model = NeuralNetwork(
        input_size=X_train.shape[1],
        output_size=y_train.shape[1],
        nr_hidden_layers=config["nr_hidden_layers"],
        nr_neurons=config["nr_neurons"],
        activation_name=config.get("activation_name", "ReLU"),
        dropout_rate=config.get("dropout_rate", 0.0),
        use_batchnorm=config.get("use_batchnorm", False)
    ).to(device)
    
    loss_name = config.get("loss_criterion", "SmoothL1")
    if loss_name == "L1":
        criterion = nn.L1Loss()
        logging.info("Using L1Loss (MAE)")
    elif loss_name == "SmoothL1":
        criterion = nn.SmoothL1Loss()
        logging.info("Using SmoothL1Loss (Huber Loss)")
    else:
        criterion = nn.MSELoss()
        logging.info("Using MSELoss")

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0)
    )

    # Create learning rate scheduler
    scheduler = create_scheduler(optimizer, config)

    os.makedirs(rf, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(rf,"tensorboard_log"))
    try:
        sample_input = torch.from_numpy(X_train[:1]).float().to(device)
        writer.add_graph(model, sample_input)
        logging.info("Logged model graph to TensorBoard.")
    except Exception as exc:
        logging.warning(f"Could not log model graph to TensorBoard: {exc}")
    
    # Initialize ResourceLogger for saving GPU/CPU stats to files
    resource_logger = ResourceLogger(output_dir=os.path.join(rf, "resources"))
    
    with open(os.path.join(rf, "model_summary.txt"), "w") as f:
        print(model, file=f)
        try:
            summary_str = str(summary(model, input_size=(config["batch_size"], X_train.shape[1]), device=device))
            f.write("\n\n--- Torchinfo Summary ---\n")
            f.write(summary_str)
        except Exception as e:
            f.write(f"\n\n(Could not run torchinfo summary: {e})")

    train_losses = []
    val_losses = []
    x_loss_epochs = []
    patience = config.get("patience", 100)
    patience_counter = 0
    best_val_loss = float("inf")
    best_model_state = None
    num_epochs = config["num_epochs"]
    logging.info(f"Starting final training loop for max {num_epochs} epochs (Patience={patience})...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        x_loss_epochs.append(epoch+1)
        
        # Get current learning rate (different method for ReduceLROnPlateau)
        if isinstance(scheduler, ReduceLROnPlateau):
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = scheduler.get_last_lr()[0]
        
        # Log system resources to local files each epoch (TensorBoard logging disabled)
        resource_logger.log(epoch)
        
        # Step scheduler (ReduceLROnPlateau needs validation loss, with optional warmup phase)
        if isinstance(scheduler, ReduceLROnPlateau):
            plateau_warmup = config.get("warmup_epochs", 0)
            base_lr = config["learning_rate"]
            if plateau_warmup > 0 and epoch < plateau_warmup:
                # Linear warmup from base_lr * start_factor to base_lr
                start_factor = config.get("warmup_start_factor", 0.1)
                progress = (epoch + 1) / plateau_warmup
                target_lr = base_lr * (start_factor + (1 - start_factor) * progress)
                optimizer.param_groups[0]['lr'] = target_lr
                current_lr = target_lr
            else:
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
        else:
            scheduler.step()
        
        if (epoch + 1) % (max(1, num_epochs // 20)) == 0 or epoch == 0:
             logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            logging.info(f"New best model found at epoch {epoch + 1} with val_loss: {val_loss:.6f}")
        else:
            patience_counter +=1
        if patience_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1}.")
            break
        
    logging.info("Training complete. Evaluating best model on validation and test sets...")
    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        logging.warning("Early stopping never triggered; using model from final epoch.")

    val_loss_final, val_pred_scaled, val_true_scaled = evaluate(model, val_loader, criterion, device)
    test_loss, test_pred_scaled, test_true_scaled = evaluate(model, test_loader, criterion, device)
    logging.info(f"Final Validation Loss ({loss_name}): {val_loss_final:.6f}")
    logging.info(f"Final Test Loss ({loss_name}): {test_loss:.6f}")

    def to_physical_units(y_scaled: np.ndarray) -> np.ndarray:
        """Inverse any scaling/log transforms so metrics are reported in physical units."""
        if y_scaler is not None:
            y_log_space = y_scaler.inverse_transform(y_scaled)
        else:
            y_log_space = y_scaled
        return np.expm1(y_log_space)

    def _format_metric_for_text(value: Any) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"

    split_metrics: Dict[str, Dict[str, Any]] = {}
    split_plot_data: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name, split_loss, preds_scaled, targets_scaled in [
        ("validation", val_loss_final, val_pred_scaled, val_true_scaled),
        ("test", test_loss, test_pred_scaled, test_true_scaled),
    ]:
        y_true_physical = to_physical_units(targets_scaled)
        y_pred_physical = to_physical_units(preds_scaled)
        split_plot_data[split_name] = {
            "true": y_true_physical,
            "pred": y_pred_physical
        }
        overall_metrics = compute_regression_metrics(y_true_physical, y_pred_physical)
        per_label_metrics = {}
        for idx, label in enumerate(label_cols):
            per_label_metrics[label] = compute_regression_metrics(
                y_true_physical[:, idx:idx+1],
                y_pred_physical[:, idx:idx+1]
            )
        split_metrics[split_name] = {
            "loss": float(split_loss),
            "sample_count": int(len(y_true_physical)),
            "overall": overall_metrics,
            "per_label": per_label_metrics
        }

    stats_dir = os.path.join(rf, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    metrics_txt_path = os.path.join(stats_dir, "metrics_summary.txt")
    with open(metrics_txt_path, "w") as f:
        for split_name in ("validation", "test"):
            if split_name not in split_metrics:
                continue
            metrics = split_metrics[split_name]
            header = f"{split_name.upper()} METRICS (Physical Units)"
            f.write("=" * len(header) + "\n")
            f.write(header + "\n")
            f.write("=" * len(header) + "\n")
            f.write(f"Samples: {metrics['sample_count']}\n")
            f.write(f"Loss ({loss_name}): {metrics['loss']:.6f}\n")
            f.write("Overall Metrics:\n")
            for key, value in metrics["overall"].items():
                logging.info(f"{split_name.title()} {key.upper()}: {value:.6f}")
                f.write(f"  {key}: {value:.6f}\n")
            f.write("Per-Label Metrics:\n")
            for label in label_cols:
                f.write(f"  --- {label} ---\n")
                feature_metrics = metrics["per_label"].get(label, {})
                for key, value in feature_metrics.items():
                    f.write(f"    {key}: {value:.6f}\n")
            f.write("\n")

    metrics_json_path = os.path.join(stats_dir, "metrics_summary.json")
    with open(metrics_json_path, "w") as f:
        json.dump(split_metrics, f, indent=2, default=float)

    if writer:
        summary_lines: List[str] = []
        for split_name in ("validation", "test"):
            metrics = split_metrics.get(split_name)
            if not metrics:
                continue
            summary_lines.append(f"### {split_name.title()} Split")
            summary_lines.append(f"Loss ({loss_name}): {_format_metric_for_text(metrics['loss'])}  ")
            summary_lines.append(f"Samples: {metrics['sample_count']}")
            summary_lines.append("| Metric | Value |")
            summary_lines.append("| --- | --- |")
            for key in ["mae", "rmse", "r2", "mape", "max_error"]:
                value = metrics["overall"].get(key)
                if value is not None:
                    summary_lines.append(f"| {key.upper()} | {_format_metric_for_text(value)} |")
            summary_lines.append("")
        if summary_lines:
            writer.add_text("Metrics/Validation_vs_Test", "\n".join(summary_lines))

    if config["plots"]:
        logging.info("Generating plots and logging to TensorBoard...")
        test_true_physical = split_plot_data["test"]["true"]
        test_pred_physical = split_plot_data["test"]["pred"]
        plot_results(
            rf,
            np.array(x_loss_epochs),
            train_losses,
            val_losses,
            test_true_physical,
            test_pred_physical,
            label_cols,
            writer=writer
        )
        plot_split_metric_bars(rf, label_cols, split_metrics, writer=writer)
            
    writer.close()
    
    # Save resource logging data
    resource_logger.save()
    
    logging.info("Main training function finished.")
    return model