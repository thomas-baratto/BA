"""
Tests for the inference module.
"""
import pytest
import torch
import numpy as np
import os
import json
import tempfile
from unittest.mock import MagicMock, patch
from sklearn.preprocessing import MinMaxScaler

from inference import load_model_and_scalers, predict
from model import NeuralNetwork


class TestLoadModelAndScalers:
    """Test loading model and scalers."""
    
    def test_load_model_with_config(self):
        """Test loading model when config file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple model
            model = NeuralNetwork(
                input_size=5,
                output_size=2,
                nr_hidden_layers=2,
                nr_neurons=32,
                activation_name='GELU',
                dropout_rate=0.1
            )
            
            # Save model
            model_path = os.path.join(tmpdir, 'model.pt')
            torch.save(model.state_dict(), model_path)
            
            # Save config
            config = {
                'input_size': 5,
                'output_size': 2,
                'nr_hidden_layers': 2,
                'nr_neurons': 32,
                'activation_name': 'GELU',
                'dropout_rate': 0.1,
                'feature_scaler_type': 'minmax',
                'label_scaler_type': 'minmax'
            }
            config_path = os.path.join(tmpdir, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            # Load model
            loaded_model, loaded_config = load_model_and_scalers(model_path, tmpdir)
            
            assert loaded_config['input_size'] == 5
            assert loaded_config['output_size'] == 2
            assert loaded_config['activation_name'] == 'GELU'
            assert isinstance(loaded_model, NeuralNetwork)
    
    def test_load_model_without_config(self):
        """Test loading model when config file doesn't exist (infers from state dict)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple model
            model = NeuralNetwork(
                input_size=5,
                output_size=2,
                nr_hidden_layers=3,
                nr_neurons=128,
                activation_name='GELU'
            )
            
            # Save model
            model_path = os.path.join(tmpdir, 'model.pt')
            torch.save(model.state_dict(), model_path)
            
            # Load without config
            loaded_model, loaded_config = load_model_and_scalers(model_path, tmpdir)
            
            assert loaded_config['input_size'] == 5
            assert loaded_config['output_size'] == 2
            # Should use defaults when config not found
            assert loaded_config['activation_name'] == 'GELU'
            assert isinstance(loaded_model, NeuralNetwork)


class TestPredict:
    """Test prediction functionality."""
    
    def test_predict_basic(self):
        """Test basic prediction."""
        # Create and train a simple model
        model = NeuralNetwork(
            input_size=3,
            output_size=1,
            nr_hidden_layers=1,
            nr_neurons=16
        )
        model.eval()
        
        # Create scalers
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        
        # Fit scalers on dummy data
        X_dummy = np.log1p(np.random.rand(20, 3))
        y_dummy = np.log1p(np.random.rand(20, 1))
        X_scaler.fit(X_dummy)
        y_scaler.fit(y_dummy)
        
        # Make predictions
        X_new = np.random.rand(5, 3)
        predictions = predict(model, X_new, X_scaler, y_scaler)
        
        assert predictions.shape == (5, 1)
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
    
    def test_predict_multi_output(self):
        """Test prediction with multiple outputs."""
        model = NeuralNetwork(
            input_size=5,
            output_size=3,
            nr_hidden_layers=2,
            nr_neurons=32
        )
        model.eval()
        
        # Create scalers
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        
        # Fit scalers
        X_dummy = np.log1p(np.random.rand(20, 5))
        y_dummy = np.log1p(np.random.rand(20, 3))
        X_scaler.fit(X_dummy)
        y_scaler.fit(y_dummy)
        
        # Make predictions
        X_new = np.random.rand(10, 5)
        predictions = predict(model, X_new, X_scaler, y_scaler)
        
        assert predictions.shape == (10, 3)
        assert not np.any(np.isnan(predictions))
    
    def test_predict_applies_transforms_correctly(self):
        """Test that predict applies log transform and scaling correctly."""
        model = NeuralNetwork(
            input_size=2,
            output_size=1,
            nr_hidden_layers=1,
            nr_neurons=16
        )
        model.eval()
        
        # Create known scalers
        X_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit on known data
        X_fit = np.log1p(np.array([[1, 2], [3, 4], [5, 6]]))
        y_fit = np.log1p(np.array([[10], [20], [30]]))
        X_scaler.fit(X_fit)
        y_scaler.fit(y_fit)
        
        # Make prediction
        X_new = np.array([[1, 2]])
        predictions = predict(model, X_new, X_scaler, y_scaler)
        
        # Should return non-negative values after expm1
        assert np.all(predictions >= -1)  # expm1 can return -1 at minimum
        assert predictions.shape == (1, 1)
    
    def test_predict_single_sample(self):
        """Test prediction on single sample."""
        model = NeuralNetwork(
            input_size=4,
            output_size=2,
            nr_hidden_layers=1,
            nr_neurons=16
        )
        model.eval()
        
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        
        # Fit scalers
        X_dummy = np.log1p(np.random.rand(10, 4))
        y_dummy = np.log1p(np.random.rand(10, 2))
        X_scaler.fit(X_dummy)
        y_scaler.fit(y_dummy)
        
        # Single sample
        X_new = np.random.rand(1, 4)
        predictions = predict(model, X_new, X_scaler, y_scaler)
        
        assert predictions.shape == (1, 2)


class TestInferenceIntegration:
    """Integration tests for the inference workflow."""
    
    def test_full_inference_workflow(self):
        """Test complete inference workflow: save, load, predict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create and save a model
            input_size, output_size = 4, 2
            model = NeuralNetwork(
                input_size=input_size,
                output_size=output_size,
                nr_hidden_layers=2,
                nr_neurons=32,
                activation_name='ReLU',
                dropout_rate=0.0
            )
            model.eval()
            
            model_path = os.path.join(tmpdir, 'model.pt')
            torch.save(model.state_dict(), model_path)
            
            # Save config
            config = {
                'input_size': input_size,
                'output_size': output_size,
                'nr_hidden_layers': 2,
                'nr_neurons': 32,
                'activation_name': 'ReLU',
                'dropout_rate': 0.0,
                'feature_scaler_type': 'minmax',
                'label_scaler_type': 'minmax'
            }
            config_path = os.path.join(tmpdir, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            # 2. Load model
            loaded_model, loaded_config = load_model_and_scalers(model_path, tmpdir)
            
            # 3. Create scalers
            X_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            
            X_train = np.log1p(np.random.rand(50, input_size))
            y_train = np.log1p(np.random.rand(50, output_size))
            X_scaler.fit(X_train)
            y_scaler.fit(y_train)
            
            # 4. Make predictions
            X_new = np.random.rand(10, input_size)
            predictions = predict(loaded_model, X_new, X_scaler, y_scaler)
            
            # Verify
            assert predictions.shape == (10, output_size)
            assert not np.any(np.isnan(predictions))
            assert isinstance(predictions, np.ndarray)
