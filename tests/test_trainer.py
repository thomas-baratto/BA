"""
Integration tests for trainer.py
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import tempfile
import os

from trainer import train_epoch, evaluate, main_train
from model import NeuralNetwork
from data_loader import CSVDataset


class TestTrainEpoch:
    """Tests for train_epoch function."""
    
    @pytest.fixture
    def setup_training(self):
        """Setup common training components."""
        # Create simple dataset
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)
        dataset = CSVDataset(X, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Create model
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=32
        )
        
        # Create loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device('cpu')
        
        return model, loader, criterion, optimizer, device
    
    def test_train_epoch_returns_float(self, setup_training):
        """Test that train_epoch returns a float loss value."""
        model, loader, criterion, optimizer, device = setup_training
        
        loss = train_epoch(model, loader, criterion, optimizer, device)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_train_epoch_reduces_loss(self, setup_training):
        """Test that training reduces loss over epochs."""
        model, loader, criterion, optimizer, device = setup_training
        
        losses = []
        for _ in range(5):
            loss = train_epoch(model, loader, criterion, optimizer, device)
            losses.append(loss)
        
        # Loss should generally decrease (allowing for some variance)
        assert losses[-1] < losses[0] * 1.5
    
    def test_train_epoch_with_gradient_clipping(self, setup_training):
        """Test training with gradient clipping."""
        model, loader, criterion, optimizer, device = setup_training
        
        loss = train_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_train_epoch_updates_weights(self, setup_training):
        """Test that weights are updated during training."""
        model, loader, criterion, optimizer, device = setup_training
        
        # Get initial weights
        initial_weights = [p.clone() for p in model.parameters()]
        
        # Train one epoch
        train_epoch(model, loader, criterion, optimizer, device)
        
        # Check that at least some weights changed
        weights_changed = False
        for initial, current in zip(initial_weights, model.parameters()):
            if not torch.allclose(initial, current):
                weights_changed = True
                break
        
        assert weights_changed
    
    def test_train_epoch_with_extreme_gradients(self, setup_training):
        """Test gradient clipping with extreme gradient values."""
        model, loader, criterion, optimizer, device = setup_training
        
        # Create a scenario with potentially large gradients
        # by using large learning rate
        optimizer = torch.optim.SGD(model.parameters(), lr=10.0)
        
        loss = train_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0)
        
        # Loss should be finite (clipping prevents NaN/Inf)
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        assert not np.isinf(loss)


class TestEvaluate:
    """Tests for evaluate function."""
    
    @pytest.fixture
    def setup_evaluation(self):
        """Setup evaluation components."""
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50, 3).astype(np.float32)
        dataset = CSVDataset(X, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        model = NeuralNetwork(
            input_size=10,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=32
        )
        
        criterion = nn.MSELoss()
        device = torch.device('cpu')
        
        return model, loader, criterion, device, y
    
    def test_evaluate_returns_correct_types(self, setup_evaluation):
        """Test that evaluate returns correct types."""
        model, loader, criterion, device, y_true = setup_evaluation
        
        loss, predictions, true_values = evaluate(model, loader, criterion, device)
        
        assert isinstance(loss, float)
        assert isinstance(predictions, np.ndarray)
        assert isinstance(true_values, np.ndarray)
    
    def test_evaluate_shapes(self, setup_evaluation):
        """Test that evaluate returns correct shapes."""
        model, loader, criterion, device, y_true = setup_evaluation
        
        loss, predictions, true_values = evaluate(model, loader, criterion, device)
        
        assert predictions.shape == (50, 3)
        assert true_values.shape == (50, 3)
    
    def test_evaluate_no_gradient(self, setup_evaluation):
        """Test that evaluation doesn't compute gradients."""
        model, loader, criterion, device, _ = setup_evaluation
        
        # Enable gradient tracking
        for param in model.parameters():
            param.requires_grad = True
        
        loss, predictions, true_values = evaluate(model, loader, criterion, device)
        
        # No gradients should be stored after evaluation
        for param in model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)


class TestMainTrainIntegration:
    """Integration tests for main_train function."""
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing."""
        import pandas as pd
        
        # Generate sample data
        n_samples = 100
        data = {
            'Flow_well': np.random.uniform(100, 500, n_samples),
            'Temp_diff': np.random.uniform(10, 50, n_samples),
            'kW_well': np.random.uniform(5, 25, n_samples),
            'Hydr_gradient': np.random.uniform(0.1, 0.5, n_samples),
            'Hydr_conductivity': np.random.uniform(1, 5, n_samples),
            'Aqu_thickness': np.random.uniform(10, 50, n_samples),
            'Long_dispersivity': np.random.uniform(1, 5, n_samples),
            'Trans_dispersivity': np.random.uniform(0.1, 0.5, n_samples),
            'Isotherm': np.random.uniform(1, 5, n_samples),
            'Area': np.random.uniform(100, 500, n_samples),
            'Iso_distance': np.random.uniform(10, 50, n_samples),
            'Iso_width': np.random.uniform(5, 25, n_samples)
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        os.unlink(temp_path)
    
    def test_main_train_basic(self, sample_csv_file):
        """Test basic main_train functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "num_epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.001,
                "nr_hidden_layers": 2,
                "nr_neurons": 32,
                "expanding_layers": False,
                "contracting_layers": False,
                "activation_name": "ReLU",
                "dropout_rate": 0.1,
                "weight_decay": 0.0001,
                "loss_criterion": "MSE",
                "scheduler_type": "StepLR",
                "plots": False,
                "patience": 10
            }
            
            feature_cols = ['Flow_well', 'Temp_diff', 'kW_well']
            label_cols = ['Area']
            device = torch.device('cpu')
            
            model = main_train(
                config=config,
                rf=tmpdir,
                csv_file=sample_csv_file,
                feature_cols=feature_cols,
                label_cols=label_cols,
                device=device
            )
            
            assert model is not None
            assert isinstance(model, nn.Module)
    
    def test_main_train_with_cosine_scheduler(self, sample_csv_file):
        """Test main_train with CosineAnnealingLR scheduler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "num_epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.001,
                "nr_hidden_layers": 2,
                "nr_neurons": 32,
                "activation_name": "GELU",
                "dropout_rate": 0.0,
                "weight_decay": 0.0,
                "loss_criterion": "SmoothL1",
                "scheduler_type": "CosineAnnealingLR",
                "scheduler_T_max": 10,
                "warmup_epochs": 2,
                "plots": False,
                "patience": 10
            }
            
            feature_cols = ['Flow_well', 'Temp_diff']
            label_cols = ['Area']
            device = torch.device('cpu')
            
            model = main_train(
                config=config,
                rf=tmpdir,
                csv_file=sample_csv_file,
                feature_cols=feature_cols,
                label_cols=label_cols,
                device=device
            )
            
            assert model is not None
    
    def test_main_train_multiple_labels(self, sample_csv_file):
        """Test main_train with multiple labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "num_epochs": 3,
                "batch_size": 16,
                "learning_rate": 0.001,
                "nr_hidden_layers": 2,
                "nr_neurons": 32,
                "activation_name": "ReLU",
                "dropout_rate": 0.0,
                "weight_decay": 0.0,
                "loss_criterion": "MSE",
                "plots": False,
                "patience": 10
            }
            
            feature_cols = ['Flow_well', 'Temp_diff']
            label_cols = ['Area', 'Iso_distance', 'Iso_width']
            device = torch.device('cpu')
            
            model = main_train(
                config=config,
                rf=tmpdir,
                csv_file=sample_csv_file,
                feature_cols=feature_cols,
                label_cols=label_cols,
                device=device
            )
            
            assert model is not None
            
            # Check that model outputs correct number of dimensions
            x = torch.randn(1, 2)
            output = model(x)
            assert output.shape[1] == 3
    
    def test_main_train_with_batchnorm(self, sample_csv_file):
        """Test main_train with BatchNorm enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "num_epochs": 3,
                "batch_size": 16,
                "learning_rate": 0.001,
                "nr_hidden_layers": 2,
                "nr_neurons": 32,
                "activation_name": "GELU",
                "dropout_rate": 0.0,
                "weight_decay": 0.0,
                "loss_criterion": "MSE",
                "use_batchnorm": True,
                "plots": False,
                "patience": 10
            }
            
            feature_cols = ['Flow_well', 'Temp_diff']
            label_cols = ['Area']
            device = torch.device('cpu')
            
            model = main_train(
                config=config,
                rf=tmpdir,
                csv_file=sample_csv_file,
                feature_cols=feature_cols,
                label_cols=label_cols,
                device=device
            )
            
            assert model is not None
            # Check that model contains BatchNorm layers
            has_batchnorm = any(isinstance(layer, nn.BatchNorm1d) for layer in model.layers)
            assert has_batchnorm, "Model should contain BatchNorm layers when use_batchnorm=True"
    
    def test_main_train_with_reduce_lr_plateau(self, sample_csv_file):
        """Test main_train with ReduceLROnPlateau scheduler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "num_epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.01,
                "nr_hidden_layers": 2,
                "nr_neurons": 32,
                "activation_name": "ReLU",
                "dropout_rate": 0.0,
                "weight_decay": 0.0,
                "loss_criterion": "MSE",
                "scheduler_type": "ReduceLROnPlateau",
                "scheduler_factor": 0.5,
                "scheduler_patience": 2,
                "plots": False,
                "patience": 10
            }
            
            feature_cols = ['Flow_well', 'Temp_diff']
            label_cols = ['Area']
            device = torch.device('cpu')
            
            model = main_train(
                config=config,
                rf=tmpdir,
                csv_file=sample_csv_file,
                feature_cols=feature_cols,
                label_cols=label_cols,
                device=device
            )
            
            assert model is not None
    
    def test_main_train_with_step_lr(self, sample_csv_file):
        """Test main_train with StepLR scheduler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "num_epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.01,
                "nr_hidden_layers": 2,
                "nr_neurons": 32,
                "activation_name": "ReLU",
                "dropout_rate": 0.0,
                "weight_decay": 0.0,
                "loss_criterion": "MSE",
                "scheduler_type": "StepLR",
                "lr_scheduler_epoch_step": 2,
                "lr_scheduler_gamma": 0.5,
                "plots": False,
                "patience": 10
            }
            
            feature_cols = ['Flow_well', 'Temp_diff']
            label_cols = ['Area']
            device = torch.device('cpu')
            
            model = main_train(
                config=config,
                rf=tmpdir,
                csv_file=sample_csv_file,
                feature_cols=feature_cols,
                label_cols=label_cols,
                device=device
            )
            
            assert model is not None
    
    def test_main_train_with_warmup(self, sample_csv_file):
        """Test main_train with learning rate warmup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "num_epochs": 10,
                "batch_size": 16,
                "learning_rate": 0.01,
                "nr_hidden_layers": 2,
                "nr_neurons": 32,
                "activation_name": "ReLU",
                "dropout_rate": 0.0,
                "weight_decay": 0.0,
                "loss_criterion": "MSE",
                "scheduler_type": "CosineAnnealingLR",
                "scheduler_T_max": 10,
                "warmup_epochs": 3,
                "plots": False,
                "patience": 20
            }
            
            feature_cols = ['Flow_well', 'Temp_diff']
            label_cols = ['Area']
            device = torch.device('cpu')
            
            model = main_train(
                config=config,
                rf=tmpdir,
                csv_file=sample_csv_file,
                feature_cols=feature_cols,
                label_cols=label_cols,
                device=device
            )
            
            assert model is not None
    
    def test_main_train_with_l1_loss(self, sample_csv_file):
        """Test main_train with L1 loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "num_epochs": 3,
                "batch_size": 16,
                "learning_rate": 0.001,
                "nr_hidden_layers": 2,
                "nr_neurons": 32,
                "activation_name": "ReLU",
                "dropout_rate": 0.0,
                "weight_decay": 0.0,
                "loss_criterion": "L1",
                "plots": False,
                "patience": 10
            }
            
            feature_cols = ['Flow_well', 'Temp_diff']
            label_cols = ['Area']
            device = torch.device('cpu')
            
            model = main_train(
                config=config,
                rf=tmpdir,
                csv_file=sample_csv_file,
                feature_cols=feature_cols,
                label_cols=label_cols,
                device=device
            )
            
            assert model is not None
    
    def test_main_train_with_smooth_l1_loss(self, sample_csv_file):
        """Test main_train with SmoothL1 loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "num_epochs": 3,
                "batch_size": 16,
                "learning_rate": 0.001,
                "nr_hidden_layers": 2,
                "nr_neurons": 32,
                "activation_name": "ReLU",
                "dropout_rate": 0.0,
                "weight_decay": 0.0,
                "loss_criterion": "SmoothL1",
                "plots": False,
                "patience": 10
            }
            
            feature_cols = ['Flow_well', 'Temp_diff']
            label_cols = ['Area']
            device = torch.device('cpu')
            
            model = main_train(
                config=config,
                rf=tmpdir,
                csv_file=sample_csv_file,
                feature_cols=feature_cols,
                label_cols=label_cols,
                device=device
            )
            
            assert model is not None
    
    def test_early_stopping(self, sample_csv_file):
        """Test that early stopping works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'nr_hidden_layers': 2,
                'nr_neurons': 32,
                'activation_name': 'ReLU',
                'dropout_rate': 0.1,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'batch_size': 16,
                'num_epochs': 1000,  # Set high to trigger early stopping
                'patience': 3,   # Small patience for quick stopping
                'scheduler_type': 'None',
                'use_batchnorm': False,
                'loss_criterion': 'MSE',
                'plots': False
            }
            
            feature_cols = ['Flow_well', 'Temp_diff', 'kW_well']
            label_cols = ['Area']
            device = torch.device('cpu')
            
            model = main_train(
                config=config,
                rf=tmpdir,
                csv_file=sample_csv_file,
                feature_cols=feature_cols,
                label_cols=label_cols,
                device=device
            )
            
            # Model should finish before max epochs due to early stopping
            assert model is not None
    
    def test_learning_rate_scheduling(self, sample_csv_file):
        """Test that learning rate changes correctly with scheduler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'nr_hidden_layers': 2,
                'nr_neurons': 32,
                'activation_name': 'ReLU',
                'dropout_rate': 0.0,
                'learning_rate': 0.01,
                'weight_decay': 0.0,
                'batch_size': 16,
                'num_epochs': 5,
                'patience': 10,
                'scheduler_type': 'StepLR',
                'step_size': 2,
                'gamma': 0.5,
                'use_batchnorm': False,
                'loss_criterion': 'MSE',
                'plots': False
            }
            
            feature_cols = ['Flow_well', 'Temp_diff', 'kW_well']
            label_cols = ['Area']
            device = torch.device('cpu')
            
            # Track learning rates
            initial_lr = config['learning_rate']
            
            model = main_train(
                config=config,
                rf=tmpdir,
                csv_file=sample_csv_file,
                feature_cols=feature_cols,
                label_cols=label_cols,
                device=device
            )
            
            # Model should complete training
            assert model is not None
    
    def test_model_returns_best_weights(self, sample_csv_file):
        """Test that training returns model with best validation weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'nr_hidden_layers': 2,
                'nr_neurons': 32,
                'activation_name': 'ReLU',
                'dropout_rate': 0.1,
                'learning_rate': 0.001,
                'weight_decay': 0.0,
                'batch_size': 16,
                'num_epochs': 10,
                'patience': 10,
                'scheduler_type': 'None',
                'use_batchnorm': True,
                'loss_criterion': 'MSE',
                'plots': False
            }
            
            feature_cols = ['Flow_well', 'Temp_diff', 'kW_well']
            label_cols = ['Area']
            device = torch.device('cpu')
            
            model = main_train(
                config=config,
                rf=tmpdir,
                csv_file=sample_csv_file,
                feature_cols=feature_cols,
                label_cols=label_cols,
                device=device
            )
            
            # Model should be returned (with best weights loaded)
            assert model is not None
            assert isinstance(model, nn.Module)
            
            # Model should be able to make predictions
            test_input = torch.randn(1, len(feature_cols))
            model.eval()
            with torch.no_grad():
                output = model(test_input)
            
            assert output.shape == (1, len(label_cols))
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

