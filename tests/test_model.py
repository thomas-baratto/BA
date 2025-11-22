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
    
    def test_batchnorm_enabled(self):
        """Test network with BatchNorm enabled."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64,
            use_batchnorm=True
        )
        
        # Check that BatchNorm layers are present
        has_batchnorm = any(isinstance(layer, nn.BatchNorm1d) for layer in model.layers)
        assert has_batchnorm, "Model should contain BatchNorm layers when use_batchnorm=True"
        
        # Test forward pass works
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 3)
    
    def test_batchnorm_disabled(self):
        """Test network with BatchNorm disabled."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64,
            use_batchnorm=False
        )
        
        # Check that BatchNorm layers are NOT present
        has_batchnorm = any(isinstance(layer, nn.BatchNorm1d) for layer in model.layers)
        assert not has_batchnorm, "Model should not contain BatchNorm layers when use_batchnorm=False"
        
        # Test forward pass works
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 3)
    
    def test_batchnorm_with_expanding_layers(self):
        """Test BatchNorm with expanding layer architecture."""
        model = NeuralNetwork(
            input_size=8,
            output_size=2,
            nr_hidden_layers=2,
            nr_neurons=64,
            exp_layers=True,
            use_batchnorm=True
        )
        
        # Should have BatchNorm layers
        batchnorm_count = sum(1 for layer in model.layers if isinstance(layer, nn.BatchNorm1d))
        assert batchnorm_count > 0, "Expanding layers with BatchNorm should contain BatchNorm1d layers"
        
        # Test forward pass
        x = torch.randn(4, 8)
        output = model(x)
        assert output.shape == (4, 2)
    
    def test_batchnorm_with_contracting_layers(self):
        """Test BatchNorm with contracting layer architecture."""
        model = NeuralNetwork(
            input_size=10,
            output_size=2,
            nr_hidden_layers=2,
            nr_neurons=128,
            con_layers=True,
            use_batchnorm=True
        )
        
        # Should have BatchNorm layers
        batchnorm_count = sum(1 for layer in model.layers if isinstance(layer, nn.BatchNorm1d))
        assert batchnorm_count > 0, "Contracting layers with BatchNorm should contain BatchNorm1d layers"
        
        # Test forward pass
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 2)
    
    def test_batchnorm_parameter_count_difference(self):
        """Test that BatchNorm adds parameters to the model."""
        model_without_bn = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64,
            use_batchnorm=False
        )
        
        model_with_bn = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64,
            use_batchnorm=True
        )
        
        params_without_bn = sum(p.numel() for p in model_without_bn.parameters())
        params_with_bn = sum(p.numel() for p in model_with_bn.parameters())
        
        # Model with BatchNorm should have more parameters
        assert params_with_bn > params_without_bn, "BatchNorm should add parameters to the model"
    
    def test_model_save_load(self, tmp_path):
        """Test that model weights can be saved and loaded correctly."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64
        )
        
        # Create input and get output before saving
        x = torch.randn(5, 10)
        output_before = model(x)
        
        # Save model
        save_path = tmp_path / "model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Create new model and load weights
        model_loaded = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64
        )
        model_loaded.load_state_dict(torch.load(save_path))
        
        # Get output from loaded model
        output_after = model_loaded(x)
        
        # Outputs should be identical
        assert torch.allclose(output_before, output_after), "Loaded model should produce same outputs"
    
    def test_model_eval_train_mode(self):
        """Test that model behaves differently in train vs eval mode with BatchNorm/Dropout."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64,
            dropout_rate=0.5,
            use_batchnorm=True
        )
        
        x = torch.randn(32, 10)
        
        # In training mode, dropout is active
        model.train()
        output_train_1 = model(x)
        output_train_2 = model(x)
        
        # In eval mode, dropout is disabled and BatchNorm uses running stats
        model.eval()
        with torch.no_grad():
            output_eval_1 = model(x)
            output_eval_2 = model(x)
        
        # Outputs in train mode might differ due to dropout
        # Outputs in eval mode should be identical
        assert torch.allclose(output_eval_1, output_eval_2), "Eval mode should be deterministic"
    
    def test_numerical_stability(self):
        """Test that model doesn't produce NaN or Inf values."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=3,
            nr_neurons=128
        )
        
        # Test with normal inputs
        x_normal = torch.randn(16, 10)
        output_normal = model(x_normal)
        assert not torch.isnan(output_normal).any(), "Model should not produce NaN with normal inputs"
        assert not torch.isinf(output_normal).any(), "Model should not produce Inf with normal inputs"
        
        # Test with large inputs
        x_large = torch.randn(16, 10) * 100
        output_large = model(x_large)
        assert not torch.isnan(output_large).any(), "Model should not produce NaN with large inputs"
        assert not torch.isinf(output_large).any(), "Model should not produce Inf with large inputs"
        
        # Test with small inputs
        x_small = torch.randn(16, 10) * 0.01
        output_small = model(x_small)
        assert not torch.isnan(output_small).any(), "Model should not produce NaN with small inputs"
        assert not torch.isinf(output_small).any(), "Model should not produce Inf with small inputs"
    
    def test_model_determinism(self):
        """Test that model produces same outputs with same inputs and seed."""
        # Set seed and create model
        torch.manual_seed(42)
        model1 = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64
        )
        
        # Set same seed and create another model
        torch.manual_seed(42)
        model2 = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64
        )
        
        # Both models should have identical weights
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Models with same seed should have identical weights"
        
        # Same input should give same output
        torch.manual_seed(100)
        x = torch.randn(8, 10)
        
        model1.eval()
        model2.eval()
        with torch.no_grad():
            output1 = model1(x)
            output2 = model2(x)
        
        assert torch.allclose(output1, output2), "Identical models should produce identical outputs"
    
    def test_batch_size_invariance(self):
        """Test that predictions are independent of batch size."""
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=64,
            use_batchnorm=False  # Disable BatchNorm for this test
        )
        model.eval()
        
        # Create test data
        x = torch.randn(10, 10)
        
        with torch.no_grad():
            # Process all at once
            output_full = model(x)
            
            # Process one by one
            outputs_individual = [model(x[i:i+1]) for i in range(10)]
            output_individual = torch.cat(outputs_individual, dim=0)
        
        # Results should be identical
        assert torch.allclose(output_full, output_individual, atol=1e-6), \
            "Predictions should be independent of batch size"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

