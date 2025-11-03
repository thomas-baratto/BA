import torch
import torch.nn as nn
import math
from typing import Optional

def get_activation(name: str) -> nn.Module:
    """Returns the activation function corresponding to the given name."""
    activations = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
        'ELU': nn.ELU(),
        'GELU': nn.GELU()
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
                 exp_layers: bool = False,
                 con_layers: bool = False,
                 dropout_rate: float = 0.0):
        super(NeuralNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        current_neurons = input_size
        activation = get_activation(activation_name)

        # 1. Input and Expanding Layers
        if exp_layers:
            # Start with a power of 2 >= input_size
            neurons = 2 ** math.ceil(math.log2(input_size))
            self.layers.append(nn.Linear(current_neurons, neurons))
            self.layers.append(activation)
            if self.dropout:
                self.layers.append(self.dropout)
            current_neurons = neurons
            
            # Expand until we are near the target neuron count
            while current_neurons < nr_neurons / 2:
                next_neurons = current_neurons * 2
                self.layers.append(nn.Linear(current_neurons, next_neurons))
                self.layers.append(activation)
                if self.dropout:
                    self.layers.append(self.dropout)
                current_neurons = next_neurons
            
            # Final expanding layer to target neuron count
            if current_neurons != nr_neurons:
                self.layers.append(nn.Linear(current_neurons, nr_neurons))
                self.layers.append(activation)
                if self.dropout:
                    self.layers.append(self.dropout)
                current_neurons = nr_neurons
        else:
            # Simple input layer
            self.layers.append(nn.Linear(current_neurons, nr_neurons))
            self.layers.append(activation)
            if self.dropout:
                self.layers.append(self.dropout)
            current_neurons = nr_neurons

        # 2. Hidden Layers
        for _ in range(nr_hidden_layers):
            self.layers.append(nn.Linear(current_neurons, nr_neurons))
            self.layers.append(activation)
            if self.dropout:
                self.layers.append(self.dropout)

        # 3. Contracting Layers
        if con_layers:
            # Contract until we are near the output size
            while current_neurons > output_size * 4 and current_neurons > nr_neurons // 2: 
                next_neurons = current_neurons // 2
                if next_neurons < output_size:
                    break
                self.layers.append(nn.Linear(current_neurons, next_neurons))
                self.layers.append(activation)
                if self.dropout:
                    self.layers.append(self.dropout)
                current_neurons = next_neurons
        
        # 4. Output Layer
        self.layers.append(nn.Linear(current_neurons, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x