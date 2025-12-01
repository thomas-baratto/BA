import torch
import torch.nn as nn
import math

def get_activation(name: str) -> nn.Module:
    """Returns the activation function corresponding to the given name."""
    activations = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
        'ELU': nn.ELU(),
        'GELU': nn.GELU(),
        'Tanh': nn.Tanh()
    }
    return activations.get(name, nn.ReLU())  # Default to ReLU if not found

class NeuralNetwork(nn.Module):
    """
    A flexible fully-connected neural network with optional
    expanding and contracting layer sections.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 nr_hidden_layers: int = 5,
                 nr_neurons: int = 128,
                 activation_name: str = 'ReLU',
                 dropout_rate: float = 0.0,
                 use_batchnorm: bool = True):
        super(NeuralNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        current_neurons = input_size
        activation = get_activation(activation_name)
        self.use_batchnorm = use_batchnorm

        # 1. Input Layer
        self.layers.append(nn.Linear(current_neurons, nr_neurons))
        if self.use_batchnorm:
            self.layers.append(nn.BatchNorm1d(nr_neurons))
        self.layers.append(activation)
        if self.dropout:
            self.layers.append(self.dropout)
        current_neurons = nr_neurons

        # 2. Hidden Layers
        for _ in range(nr_hidden_layers):
            self.layers.append(nn.Linear(current_neurons, nr_neurons))
            if self.use_batchnorm:
                self.layers.append(nn.BatchNorm1d(nr_neurons))
            self.layers.append(activation)
            if self.dropout:
                self.layers.append(self.dropout)
        
        # 3. Output Layer
        self.layers.append(nn.Linear(current_neurons, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:       
        for layer in self.layers:
            x = layer(x)
        return x