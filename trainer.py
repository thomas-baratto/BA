import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import os
from typing import Tuple, Dict, Any, List
from torchinfo import summary

# Import from your other project files
from model import NeuralNetwork
from data_loader import CSVDataset, load_data
from utils import plot_results, relative_error

# --- Training & Evaluation Functions ---

def train_epoch(model: nn.Module,
                loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> float:
    """Runs a single training epoch."""
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
    """Evaluates the model on a dataset."""
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
    """
    Main function to load data, train, and evaluate a model based on a config.
    """
    logging.info(f"Starting main training for labels {label_cols}...")
    
    # Load data
    X_train, X_test, X_scaler, y_train, y_test, y_scaler = load_data(
        csv_file,
        feature_cols=feature_cols,
        label_cols=label_cols,
        plots=config["plots"],
        rf=rf
    )

    # Data loaders
    train_dataset = CSVDataset(X_train, y_train)
    test_dataset = CSVDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Initialize model
    model = NeuralNetwork(
        input_size=X_train.shape[1],
        output_size=y_train.shape[1],
        nr_hidden_layers=config["nr_hidden_layers"],
        nr_neurons=config["nr_neurons"],
        exp_layers=config["expanding_layers"],
        con_layers=config["contracting_layers"]
    ).to(device)
    
    criterion = nn.MSELoss() # Or select based on config
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr_scheduler_epoch_step"],
        gamma=config["lr_scheduler_gamma"]
    )

    # Save model summary
    os.makedirs(rf, exist_ok=True)
    with open(os.path.join(rf, "model_summary.txt"), "w") as f:
        print(model, file=f)
        # Consider using torchinfo for a more detailed summary
        try:
            summary_str = str(summary(model, input_size=(config["batch_size"], X_train.shape[1]), device=device))
            f.write("\n\n--- Torchinfo Summary ---\n")
            f.write(summary_str)
        except ImportError:
            f.write("\n\n(Install torchinfo for a detailed summary)")


    # Training loop
    train_losses = []
    x_loss_epochs = np.arange(1, config["num_epochs"] + 1)
    
    logging.info("Starting training loop...")
    for epoch in range(config["num_epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        scheduler.step()
        
        # Log progress periodically
        if (epoch + 1) % (max(1, config["num_epochs"] // 20)) == 0 or epoch == 0:
             logging.info(f"Epoch [{epoch + 1}/{config['num_epochs']}], "
                          f"Loss: {train_loss:.6f}, "
                          f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Final evaluation
    logging.info("Training complete. Evaluating on test set...")
    test_loss, predictions, true_values = evaluate(model, test_loader, criterion, device)
    logging.info(f"Final Test Loss: {test_loss:.6f}")
    
    # Invert scaling and log-transform for plotting and error metrics
    if config["plots"]:
        logging.info("Inverting transforms and generating plots...")
        inverted_predictions = y_scaler.inverse_transform(predictions)
        inverted_true_values = y_scaler.inverse_transform(true_values)
        
        # Inverse of log1p is expm1
        inverted_predictions = np.expm1(inverted_predictions)
        inverted_true_values = np.expm1(inverted_true_values)
        
        plot_results(rf, x_loss_epochs, train_losses,
                     inverted_true_values, inverted_predictions, label_cols)
        
        relative_error(rf, inverted_predictions, inverted_true_values, label_cols)

    logging.info("Main training function finished.")
    return model