import torch
import torch.nn as nn
import math
from typing import Optional

class NeuralNetwork(nn.Module):
    """
    A flexible fully-connected neural network with optional
    expanding and contracting layer sections.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 nr_hidden_layers: int = 10,
                 nr_neurons: int = 128,
                 activation: nn.Module = nn.ReLU(),
                 exp_layers: bool = False,
                 con_layers: bool = False):
        super(NeuralNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        current_neurons = input_size

        # 1. Input and Expanding Layers
        if exp_layers:
            # Start with a power of 2 >= input_size
            neurons = 2 ** math.ceil(math.log2(input_size))
            self.layers.append(nn.Linear(current_neurons, neurons))
            self.layers.append(activation)
            current_neurons = neurons
            
            # Expand until we are near the target neuron count
            while current_neurons < nr_neurons / 2:
                next_neurons = current_neurons * 2
                self.layers.append(nn.Linear(current_neurons, next_neurons))
                self.layers.append(activation)
                current_neurons = next_neurons
            
            # Final expanding layer to target neuron count
            if current_neurons != nr_neurons:
                self.layers.append(nn.Linear(current_neurons, nr_neurons))
                self.layers.append(activation)
                current_neurons = nr_neurons
        else:
            # Simple input layer
            self.layers.append(nn.Linear(current_neurons, nr_neurons))
            self.layers.append(activation)
            current_neurons = nr_neurons

        # 2. Hidden Layers
        for _ in range(nr_hidden_layers):
            self.layers.append(nn.Linear(current_neurons, nr_neurons))
            self.layers.append(activation)
            current_neurons = nr_neurons # This is constant here

        # 3. Contracting Layers
        if con_layers:
            # Contract until we are near the output size
            while current_neurons > output_size * 4 and current_neurons > nr_neurons // 2: # Avoid over-shrinking
                next_neurons = current_neurons // 2
                if next_neurons < output_size: # Don't shrink smaller than output
                    break
                self.layers.append(nn.Linear(current_neurons, next_neurons))
                self.layers.append(activation)
                current_neurons = next_neurons
        
        # 4. Output Layer
        self.layers.append(nn.Linear(current_neurons, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x