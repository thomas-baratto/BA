import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import torch
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import math
import datetime
import time

import optuna
import logging
import sys
from optuna_dashboard import run_server

import psutil
from pynvml import *


class CSVDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=8, output_size=3, nr_hidden_layers=10, nr_neurons=128, activation=nn.ReLU(),exp_layers=False,con_layers=False):
        super(NeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.nr_hidden_layers = nr_hidden_layers
        self.exp_layers = exp_layers
        self.con_layers = con_layers
        self.nr_neurons = nr_neurons
        self.activation = activation
        
        neurons = 2 ** math.ceil(math.log2(self.input_size))

        # Expanding layers
        self.input_layer = nn.Linear(input_size, neurons)
        if not exp_layers:
            self.input_layer = nn.Linear(input_size, nr_neurons)
        
        if exp_layers:
            self.expanding_layers = nn.ModuleList()
            while neurons < self.nr_neurons/2:
                self.expanding_layers.append(nn.Linear(neurons, neurons * 2))
                neurons *= 2
            self.expanding_layers.append(nn.Linear(neurons,nr_neurons))    

        # Hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(nr_neurons, nr_neurons) for _ in range(nr_hidden_layers)])

        # Contracting layers
        if con_layers:
            self.contracting_layers = nn.ModuleList()
            neurons = nr_neurons
            while neurons > self.output_size*4:
                self.contracting_layers.append(nn.Linear(neurons, neurons // 2))
                neurons //= 2
            self.output_layer = nn.Linear(neurons, output_size)
        else:
            self.output_layer = nn.Linear(nr_neurons, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)

        # Expanding layers
        if self.exp_layers:
            for layer in self.expanding_layers:
                x = layer(x)
                x = self.activation(x)

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        # Contracting layers
        if self.con_layers:
            for layer in self.contracting_layers:
                x = layer(x)
                x = self.activation(x)

        # Output layer
        x = self.output_layer(x)
        
        return x

def load_data(rf,csv_file, labels=[9, 10, 11],plots = False):
    # Load data
    df = pd.read_csv(csv_file)

    # Split features and labels
    X = df.iloc[:,[0, 1, 3, 4, 5, 6, 7, 8, 12]].values  # Ignore column with real Temperature and use isotherm temperature
    y = df.iloc[:, labels].values  # Three outputs to predict
    if plots:
        os.makedirs(rf + f"plots/%s" % str(labels), exist_ok=True)

        for label in range(len(labels)):
            plt.hist(y[:, label], bins=200)
        plt.title("Before Transformation")
        plt.savefig(rf + f"plots/%s/before.png" % str(labels))
        plt.close()

    X = np.log1p(X)
    y = np.log1p(y)
    if plots:
        for label in range(len(labels)):
            plt.hist(y[:, label], bins=200)
        plt.title("After Log-Transformation")
        plt.savefig(rf + f"plots/%s/afterlog.png" % str(labels))
        plt.close()

    # Standardize features
    X_scaler = MinMaxScaler((0,1))
    y_scaler = MinMaxScaler((0,1))

    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y)
    
    if plots:
        for label in range(len(labels)):
            plt.hist(y[:, label], bins=200)
        plt.title("After Standardization")
        plt.savefig(rf + f"plots/%s/afterstandardization.png" % str(labels))
        plt.close()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
    return X_train, X_test, X_scaler, y_train, y_test, y_scaler

def plot_results(rf,x_loss, y_loss, true_values, predictions, labels):
    plt.semilogy(x_loss, y_loss)
    plt.title("Loss")
    filename = rf + f"plots/%s/loss.png" % str(labels)
    plt.savefig(filename)
    plt.close()

    for label in range(len(labels)):
        true_values_label = true_values[:, label]
        predictions_label = predictions[:, label]
        plt.scatter(true_values_label, predictions_label, label=label)
        plt.plot([min(true_values_label), max(true_values_label)], [min(true_values_label), max(true_values_label)],
                 'r--')
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        text = str(labels)
        plt.title("plot of labels:" + text)
        plt.legend()
        filename = rf + f"plots/%s/truepred.png" % str(labels)
        plt.savefig(filename)
        plt.close()

def relative_error(rf,pred, true, label, name):
    dir = f"stats/%s" % name
    dir = rf + dir
    os.makedirs(dir, exist_ok=True)
    err_rel = (pred - true) / true
    std = np.std(err_rel)
    with open(dir + "/errors.txt", "a") as file:
        print("The standard deviation of the relative error of label:", label, " is: ", std, file=file)
    plt.hist(err_rel, bins=200)
    plt.title(f"Relative Error for label:%s" % label)
    plt.savefig(dir + "/plot")
    plt.close()

def main(rf, csv_file='./Daten/Clean_Results_Isotherm.csv', num_epochs=10, batch_size=4096, learning_rate=0.01,
         labels=[9], nr_hidden_layers=3, nr_neurons=16
         , lr_scheduler_epoch_step = 1000, lr_scheduler_gamma = 0.1, plots = False, expanding_layers = False, contracting_layers = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Loss plot
    x_loss = np.linspace(start=1, stop=num_epochs, num=num_epochs)
    y_loss = []

    # Load data
    X_train, X_test, X_scaler, y_train, y_test, y_scaler = load_data(rf, csv_file, labels=labels,plots=plots)

    # Convert to PyTorch datasets
    train_dataset = CSVDataset(X_train, y_train)
    test_dataset = CSVDataset(X_test, y_test)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    model = NeuralNetwork(input_size, output_size, nr_hidden_layers=nr_hidden_layers, nr_neurons=nr_neurons, exp_layers = expanding_layers, con_layers = contracting_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_epoch_step, gamma=lr_scheduler_gamma)
    with open(rf + "model.txt", "a") as file:
        print(model,file = file)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            # Move batch to GPU
            batch_data, batch_labels  = batch_data.to(device), batch_labels.to(device)
            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Learning rate: {scheduler.get_last_lr()}")
        y_loss.append(loss.item())

    # Evaluate on the test set
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        total_loss = 0
        predictions = []
        true_values = []
        for batch_data, batch_labels in test_loader:  # direkt test daten vergleichen mit labels
            # Move batch to GPU
            outputs = model(batch_data)
            predictions.extend(outputs.squeeze().tolist())
            true_values.extend(batch_labels.tolist())
            loss = criterion(outputs.squeeze(), batch_labels.squeeze())
            total_loss += loss.item()

    print(f"Test Loss: {total_loss / len(test_loader):.4f}")
    
    if plots:
        predictions = np.array(predictions).reshape(-1, len(labels))
        true_values = np.array(true_values).reshape(-1, len(labels))

        inverted_predictions = y_scaler.inverse_transform(predictions)
        inverted_true_values = y_scaler.inverse_transform(true_values)

        inverted_predictions = np.expm1(inverted_predictions)
        inverted_true_values = np.expm1(inverted_true_values)
        plot_results(rf,x_loss, y_loss, inverted_true_values, inverted_predictions, labels)
        for label in labels:
            relative_error(rf,inverted_predictions, inverted_true_values, label, labels)

    return model

def log_resources(writer, step):
    # CPU usage
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    writer.add_scalar("System/CPU_Usage", cpu, step)
    writer.add_scalar("System/RAM_Usage", ram, step)

    # GPU usage (if available)
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        util = nvmlDeviceGetUtilizationRates(handle)
        writer.add_scalar("GPU/Memory_Used_MB", mem_info.used / 1024**2, step)
        writer.add_scalar("GPU/Utilization_Percent", util.gpu, step)
        nvmlShutdown()
    except Exception:
        pass
    
def objective(trial):
    # Define hyperparameters to optimize
    #num_epochs = trial.suggest_int("num_epochs", 200, 500, step=100)
    num_epochs = 10
    #batch_size = trial.suggest_categorical("batch_size", [ 64, 128, 256, 512])
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True)
    nr_hidden_layers = trial.suggest_int("nr_hidden_layers", 1,5)
    expanding_layers = trial.suggest_categorical("expanding_layers",[True,False])
    contracting_layers = trial.suggest_categorical("contracting_layers",[True,False])
    nr_neurons = trial.suggest_int("nr_neurons", 8, 256,log=True)
    nr_of_steps = trial.suggest_int("nr_of_steps",1,4)
    #lr_scheduler_epoch_step = trial.suggest_int("lr_scheduler_epoch_step", num_epochs/5, num_epochs/2, step=num_epochs/10)
    lr_scheduler_epoch_step = num_epochs//nr_of_steps
    lr_scheduler_gamma = trial.suggest_float("lr_scheduler_gamma", 0.4, 0.8,log=True)

    rf = "optuna_results/"
    csv_file = './Daten/Clean_Results_Isotherm.csv'
    # Load Data
    X_train, X_test, X_scaler, y_train, y_test, y_scaler = load_data(rf, csv_file, labels=[11])

    # Convert to PyTorch datasets and DataLoader
    train_dataset = CSVDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

    test_dataset = CSVDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork(input_size=X_train.shape[1], output_size = y_train.shape[1], nr_hidden_layers=nr_hidden_layers, nr_neurons=nr_neurons, exp_layers= expanding_layers, con_layers=contracting_layers).to(device)
    loss_name = trial.suggest_categorical("criterion", ["MSE", "L1"])
    if loss_name == "MSE":
        criterion = nn.MSELoss()
    elif loss_name == "L1":
        criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_epoch_step, gamma=lr_scheduler_gamma)

    # === TensorBoard writer ===
    log_dir = f"runs/trial_{trial.number}"
    writer = SummaryWriter(log_dir=log_dir)

    # Track time
    start_time = time.time()

    # Training with pruning
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Training/Loss", avg_loss, epoch)
        
        log_resources(writer, epoch)

        # Report intermediate result to Optuna
        trial.report(avg_loss, epoch)

        # Prune the trial if it's performing badly
        if trial.should_prune():
            writer.close()
            raise optuna.TrialPruned()

    # Final evaluation on test set
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            
    test_loss /= len(test_loader)
    writer.add_scalar("Test/Loss", test_loss, 0)
    writer.add_scalar("System/Total_Train_Time_sec", time.time() - start_time, 0)
    writer.close()

    return test_loss  # Minimized loss

if __name__ == "__main__":
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    filename = './Daten/Clean_Results_Isotherm.csv'
    rootfolder = str(datetime.datetime.now()) + "/"
    os.makedirs(f"%sModels/"%rootfolder)
    
    
    storage_url = f"sqlite:///%soptuna_study.db"%rootfolder
    study = optuna.create_study(study_name="idk",direction="minimize", storage=storage_url, load_if_exists=True)
    study.optimize(objective, n_trials=10)
    
    best_params = study.best_params
    final_model = main(
    rf= rootfolder + "final_model/",
    csv_file=filename,
    num_epochs=1000,
    learning_rate=best_params["learning_rate"],
    labels=[9],
    nr_hidden_layers=best_params["nr_hidden_layers"],
    nr_neurons=best_params["nr_neurons"],
    plots=True,
    expanding_layers=best_params["expanding_layers"],
    contracting_layers=best_params["contracting_layers"],
    )
    
    torch.save(final_model.state_dict(), rootfolder + "final_model/best_model.pt")
    
    '''
    model9 = main(rf=rootfolder,csv_file=filename, labels=[9])
    torch.save(model9.state_dict(), f"%sModels/9.pt"%rootfolder)
    model10 = main(rf=rootfolder,csv_file=filename, labels=[10])
    torch.save(model10.state_dict(), f"%sModels/10.pt"%rootfolder)
    model11 = main(rf=rootfolder,csv_file=filename, labels=[11])
    torch.save(model11.state_dict(), f"%sModels/11.pt"%rootfolder)
    modeltotal = main(rf=rootfolder,csv_file=filename, labels=[9, 10, 11])
    torch.save(modeltotal.state_dict(), f"%sModels/myall.pt"%rootfolder)
    '''