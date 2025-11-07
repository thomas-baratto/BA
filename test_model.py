"""
Unit tests for model.py
"""
import pytest
import torch
import torch.nn as nn

from model import get_activation, NeuralNetwork


class TestGetActivation:
    """Tests for get_activation function."""
    
    def test_relu_activation(self):
        """Test ReLU activation."""
        activation = get_activation('ReLU')
        assert isinstance(activation, nn.ReLU)
    
    def test_leakyrelu_activation(self):
        """Test LeakyReLU activation."""
        activation = get_activation('LeakyReLU')
        assert isinstance(activation, nn.LeakyReLU)
    
    def test_elu_activation(self):
        """Test ELU activation."""
        activation = get_activation('ELU')
        assert isinstance(activation, nn.ELU)
    
    def test_gelu_activation(self):
        """Test GELU activation."""
        activation = get_activation('GELU')
        assert isinstance(activation, nn.GELU)
    
    def test_invalid_activation_defaults_to_relu(self):
        """Test that invalid activation name defaults to ReLU."""
        activation = get_activation('InvalidActivation')
        assert isinstance(activation, nn.ReLU)


class TestNeuralNetwork:
    """Tests for NeuralNetwork class."""
    
    def test_basic_network_creation(self):
        """Test basic network creation."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64,
            activation_name='ReLU'
        )
        
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64
        )
        
        batch_size = 5
        x = torch.randn(batch_size, 10)
        output = model(x)
        
        assert output.shape == (batch_size, 3)
    
    def test_different_activations(self):
        """Test network with different activation functions."""
        for activation in ['ReLU', 'LeakyReLU', 'ELU', 'GELU']:
            model = NeuralNetwork(
                input_size=5,
                output_size=2,
                nr_hidden_layers=1,
                nr_neurons=32,
                activation_name=activation
            )
            
            x = torch.randn(10, 5)
            output = model(x)
            assert output.shape == (10, 2)
    
    def test_dropout(self):
        """Test network with dropout."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64,
            dropout_rate=0.5
        )
        
        x = torch.randn(5, 10)
        
        # In training mode, dropout should be active
        model.train()
        output1 = model(x)
        output2 = model(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2)
        
        # In eval mode, dropout should be inactive
        model.eval()
        output3 = model(x)
        output4 = model(x)
        
        # Outputs should be identical
        assert torch.allclose(output3, output4)
    
    def test_expanding_layers(self):
        """Test network with expanding layers."""
        model = NeuralNetwork(
            input_size=8,
            output_size=2,
            nr_hidden_layers=2,
            nr_neurons=128,
            exp_layers=True
        )
        
        x = torch.randn(5, 8)
        output = model(x)
        assert output.shape == (5, 2)
    
    def test_contracting_layers(self):
        """Test network with contracting layers."""
        model = NeuralNetwork(
            input_size=10,
            output_size=2,
            nr_hidden_layers=2,
            nr_neurons=64,
            con_layers=True
        )
        
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 2)
    
    def test_both_expanding_and_contracting(self):
        """Test network with both expanding and contracting layers."""
        model = NeuralNetwork(
            input_size=8,
            output_size=2,
            nr_hidden_layers=3,
            nr_neurons=128,
            exp_layers=True,
            con_layers=True
        )
        
        x = torch.randn(5, 8)
        output = model(x)
        assert output.shape == (5, 2)
    
    def test_single_hidden_layer(self):
        """Test network with single hidden layer."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=1,
            nr_neurons=32
        )
        
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 3)
    
    def test_many_hidden_layers(self):
        """Test network with many hidden layers."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=10,
            nr_neurons=64
        )
        
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 3)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64
        )
        
        x = torch.randn(5, 10, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_parameter_count(self):
        """Test that network has trainable parameters."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64
        )
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 0
    
    def test_zero_dropout(self):
        """Test network with zero dropout (should be disabled)."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64,
            dropout_rate=0.0
        )
        
        x = torch.randn(5, 10)
        model.train()
        output1 = model(x)
        output2 = model(x)
        
        # With dropout_rate=0.0, outputs should be identical even in training mode
        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
