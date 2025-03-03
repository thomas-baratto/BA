import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import torch
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os
import math
import visualiesierung

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
    def __init__(self, input_size=8, output_size=3, nr_hidden_layers=3, nr_neurons=64, activation=nn.Tanh()):
        super(NeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.nr_hidden_layers = nr_hidden_layers
        self.nr_neurons = nr_neurons
        self.activation = activation
        
        # Input layer
        neurons = 2 ** math.ceil(math.log2(self.input_size))
        self.input_layer = nn.Linear(input_size, neurons)

        # Expanding layers
        self.expanding_layers = nn.ModuleList()
        while neurons < self.nr_neurons:
            self.expanding_layers.append(nn.Linear(neurons, neurons * 2))
            neurons *= 2

        # Hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons, neurons) for _ in range(nr_hidden_layers)])

        # Contracting layers
        self.contracting_layers = nn.ModuleList()
        while neurons > self.output_size*2:
            self.contracting_layers.append(nn.Linear(neurons, neurons // 2))
            neurons //= 2

        # Output layer
        self.output_layer = nn.Linear(neurons, output_size)
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)

        # Expanding layers
        for layer in self.expanding_layers:
            x = layer(x)
            x = self.activation(x)

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        # Contracting layers
        for layer in self.contracting_layers:
            x = layer(x)
            x = self.activation(x)

        # Output layer
        x = self.output_layer(x)
        x = self.activation(x)
        
        return x

def load_data(csv_file,labels=[9,10,11]):
    # Load data
    df = pd.read_csv(csv_file)
    
    # Split features and labels
    X = df.iloc[:, [0,1,3,4,5,6,7,8,12] ].values  # Ignore column with real Temperature and use isotherm temperature
    y = df.iloc[:, labels].values # Three outputs to predict

    os.makedirs(f"plots/%s"%str(labels), exist_ok=True)
    
    for label in range(len(labels)):
        plt.hist(y[:,label],bins=200)
    plt.title("Before Transformation")
    plt.savefig(f"plots/%s/before.png"%str(labels))
    plt.close()
    
    X = np.log1p(X)
    y = np.log1p(y)
    
    for label in range(len(labels)):
        plt.hist(y[:,label],bins=200)
    plt.title("After Log-Transformation")
    plt.savefig(f"plots/%s/afterlog.png"%str(labels))
    plt.close()   
    
    # Standardize features
    X_scaler = MinMaxScaler((-1,1))
    y_scaler = MinMaxScaler((-1,1))
    
    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y)
    
    for label in range(len(labels)):
        plt.hist(y[:,label],bins=200)
    plt.title("After Standardization")
    plt.savefig(f"plots/%s/afterstandardization.png"%str(labels))
    plt.close()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    return X_train, X_test, X_scaler, y_train, y_test, y_scaler

def plot_results(x_loss,y_loss,true_values,predictions,labels):
    
    plt.semilogy(x_loss, y_loss)
    plt.title("Loss")
    plt.savefig(f"plots/%s/loss.png"%str(labels))
    plt.close()
    
    for label in range(len(labels)):
        true_values_label = true_values[:,label]
        predictions_label = predictions[:,label]
        plt.scatter(true_values_label, predictions_label, label=label)
        plt.plot([min(true_values_label), max(true_values_label)], [min(true_values_label), max(true_values_label)], 'r--')    
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        text = str(labels)
        plt.title("plot of labels:" + text)
        plt.legend()
        plt.savefig(f"plots/%s/truepred.png"%str(labels))
        plt.close()

def relative_error(pred,true,label,name):
    dir = f"stats/%s"%name
    os.makedirs(dir,exist_ok=True)
    err_rel = (pred - true) / true
    avg = np.mean(err_rel)
    with open(dir + "/errors.txt","a") as file:
        print("The average relative error of label:",label," is: ", avg, file=file)
    print("after whith")
    print(type(pred))
    plt.hist(err_rel,bins=200)
    plt.title(f"Relative Error for label:%s"%label)
    plt.savefig(dir + "/plot")
    plt.close()
    print("after error plt")

def main(csv_file='../Daten/Clean_Results_Isotherm.csv', num_epochs=1000, batch_size=4096, learning_rate=0.01,labels=[9,10,11],nr_hidden_layers=3,nr_neurons=16):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Loss plot
    x_loss = np.linspace(start=1, stop=num_epochs,num=num_epochs)
    y_loss = [] 
    
    # Load data
    X_train, X_test, X_scaler, y_train, y_test, y_scaler = load_data(csv_file,labels=labels)
    
    # Convert to PyTorch datasets
    train_dataset = CSVDataset(X_train, y_train)
    test_dataset = CSVDataset(X_test, y_test)
        
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    model = NeuralNetwork(input_size, output_size, nr_hidden_layers=nr_hidden_layers, nr_neurons=nr_neurons).to(device)
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(model)
    #summary(model, input_size = (1, input_size))
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            # Move batch to GPU
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        y_loss.append(loss.item())
    
    # Evaluate on the test set
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        total_loss = 0
        predictions = []
        true_values = []
        for batch_data, batch_labels in test_loader: #direkt test daten vergleichen mit labels
            # Move batch to GPU
            outputs = model(batch_data)
            predictions.extend(outputs.squeeze().tolist())
            true_values.extend(batch_labels.tolist())
            loss = criterion(outputs.squeeze(), batch_labels.squeeze())
            total_loss += loss.item()
    
    print(f"Test Loss: {total_loss / len(test_loader):.4f}")
    
    predictions = np.array(predictions).reshape(-1,len(labels))
    true_values = np.array(true_values).reshape(-1,len(labels))
                                      
    inverted_predictions = y_scaler.inverse_transform(predictions)
    inverted_true_values = y_scaler.inverse_transform(true_values)
    
    inverted_predictions = np.expm1(inverted_predictions)
    inverted_true_values = np.expm1(inverted_true_values)
    print("Before plot")
    plot_results(x_loss,y_loss,inverted_true_values,inverted_predictions,labels)
    print("before error plot")
    for label in labels:
        relative_error(inverted_predictions,inverted_true_values,label,labels)
    print("before return")
    return model

if __name__ == "__main__":

    filename = '../Daten/Clean_Results_Isotherm.csv'
    os.makedirs("Models/",exist_ok=True)
    model9 = main(csv_file=filename, labels=[9])
    torch.save(model9.state_dict(),"Models/9.pt")
    model10 = main(csv_file=filename, labels=[10])
    torch.save(model10.state_dict(), "Models/10.pt")
    model11 = main(csv_file=filename, labels=[11])
    torch.save(model11.state_dict(), "Models/11.pt")
    modeltotal = main(csv_file=filename, labels=[9,10,11])
    torch.save(modeltotal.state_dict(), "Models/total.pt")


