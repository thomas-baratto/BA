import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchinfo import summary
import numpy as np 
import logging
import os
from typing import Tuple, Dict, Any, List
import time
import copy
from sklearn.model_selection import train_test_split

# --- REMOVED old sklearn imports ---

# Import from your other project files
from model import NeuralNetwork
from data_loader import CSVDataset, load_data
from utils import (
    plot_results, 
    compute_regression_metrics,
    plot_error_histograms,
    log_resources, 
    create_scatter_plot
)

# --- Training & Evaluation Functions ---
def train_epoch(model: nn.Module,
                loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> float:
    # (This function is unchanged)
    model.train()
    epoch_loss = 0.0
    for batch_data, batch_labels in loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    # (This function is unchanged)
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

# --- Main Training Function ---
def main_train(config: Dict[str, Any],
               rf: str,
               csv_file: str,
               feature_cols: List[str],
               label_cols: List[str],
               device: torch.device) -> nn.Module:
    
    # (Setup is unchanged)
    logging.info(f"Starting main training for labels {label_cols}...")
    X_train_full, X_test, X_scaler, y_train_full, y_test, y_scaler = load_data(
        csv_file,
        feature_cols=feature_cols,
        label_cols=label_cols,
        plots=config["plots"],
        rf=rf
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    logging.info(f"Final training: {len(X_train)} train samples, {len(X_val)} val samples, {len(X_test)} test samples.")
    train_dataset = CSVDataset(X_train, y_train)
    val_dataset = CSVDataset(X_val, y_val)
    test_dataset = CSVDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    model = NeuralNetwork(
        input_size=X_train.shape[1],
        output_size=y_train.shape[1],
        nr_hidden_layers=config["nr_hidden_layers"],
        nr_neurons=config["nr_neurons"],
        exp_layers=config.get("expanding_layers", False),
        con_layers=config.get("contracting_layers", False),
        activation_name=config.get("activation_name", "ReLU"),
        dropout_rate=config.get("dropout_rate", 0.0)
    ).to(device)
    
    loss_name = config.get("loss_criterion", "MSE")
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

    if config.get("scheduler_type") == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=config.get("scheduler_T_max", 1000))
        logging.info(f"Using CosineAnnealingLR with T_max={config.get('scheduler_T_max', 1000)}")
    else:
        step_size = config.get("lr_scheduler_epoch_step", 100)
        gamma = config.get("lr_scheduler_gamma", 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logging.info(f"Using default StepLR with step_size={step_size}, gamma={gamma}")

    os.makedirs(rf, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(rf,"tensorboard_log"))
    with open(os.path.join(rf, "model_summary.txt"), "w") as f:
        print(model, file=f)
        try:
            summary_str = str(summary(model, input_size=(config["batch_size"], X_train.shape[1]), device=device))
            f.write("\n\n--- Torchinfo Summary ---\n")
            f.write(summary_str)
        except Exception as e:
            f.write(f"\n\n(Could not run torchinfo summary: {e})")

    # (Training loop is unchanged)
    train_losses = []
    val_losses = []
    x_loss_epochs = []
    patience = config.get("patience", 25)
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
        writer.add_scalar("Final_Run/Training_Loss",train_loss, epoch)
        writer.add_scalar("Final_Run/Validation_Loss",val_loss, epoch)
        writer.add_scalar("Final_Run/Learning_Rate", scheduler.get_last_lr()[0],epoch)
        log_resources(writer, epoch)
        scheduler.step()
        if (epoch + 1) % (max(1, num_epochs // 20)) == 0 or epoch == 0:
             logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
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
        
    logging.info("Training complete. Evaluating on test set...")
    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        logging.warning("Early stopping never triggered; using model from final epoch.")
        
    test_loss, predictions, true_values = evaluate(model, test_loader, criterion, device)
    logging.info(f"Final Test Loss ({loss_name}): {test_loss:.6f}")
    writer.add_scalar(f"Final_Run/Test_Loss_{loss_name}", test_loss, 0)
    
    # --- METRICS AND PLOTTING (THIS IS THE FIXED SECTION) ---

    if config["plots"]:
        logging.info("Inverting transforms and generating plots...")
        inverted_predictions = y_scaler.inverse_transform(predictions)
        inverted_true_values = y_scaler.inverse_transform(true_values)
        inverted_predictions = np.expm1(inverted_predictions)
        inverted_true_values = np.expm1(inverted_true_values)
        
        # --- 1. Call compute_regression_metrics ---
        logging.info("Calculating final metrics...")
        final_metrics = compute_regression_metrics(inverted_true_values, inverted_predictions)
        
        stats_dir = os.path.join(rf, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        with open(os.path.join(stats_dir, "final_metrics.txt"), "w") as f:
            for key, value in final_metrics.items():
                logging.info(f"Final Test {key.upper()}: {value:.6f}")
                f.write(f"{key}: {value}\n")
                if not np.isnan(value):
                    writer.add_scalar(f"Final_Run/Test_{key.upper()}", value, 0)

        # --- 2. Call plot_results (unchanged) ---
        plot_results(rf, np.array(x_loss_epochs), train_losses, val_losses,
                     inverted_true_values, inverted_predictions, label_cols)
        
        # --- 3. Call plot_error_histograms (RENAMED) ---
        plot_error_histograms(rf, inverted_true_values, inverted_predictions, label_cols)
        
        # --- 4. Call create_scatter_plot (unchanged) ---
        logging.info("Logging plots to tensorboard...")
        scatter_plots_for_tensorboard = create_scatter_plot(
            inverted_true_values, inverted_predictions, label_cols
        )
        
        for label, fig in scatter_plots_for_tensorboard:
            writer.add_figure(label,fig)
            
    writer.add_scalar("Final_Run/Total_Train_Time_sec", time.time() - start_time, 0)        
    writer.close()
    logging.info("Main training function finished.")
    return model