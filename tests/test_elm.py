"""
Tests for the Extreme Learning Machine (ELM) implementation.
"""
import pytest
import numpy as np
import torch
from elm import ExtremeLearningMachine


class TestELMBasics:
    """Test basic ELM functionality."""
    
    def test_elm_initialization(self):
        """Test ELM can be initialized with various parameters."""
        elm = ExtremeLearningMachine(n_hidden=100, activation='ReLU', alpha=1e-3)
        assert elm.n_hidden == 100
        assert elm.activation == 'ReLU'
        assert elm.alpha == 1e-3
        assert elm.include_bias is True
        
    def test_elm_initialization_with_device(self):
        """Test ELM initialization with specific device."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elm = ExtremeLearningMachine(n_hidden=50, device=device)
        assert elm.device == device
        
    def test_elm_initialization_with_random_state(self):
        """Test ELM initialization with random state for reproducibility."""
        elm1 = ExtremeLearningMachine(n_hidden=100, random_state=42)
        elm2 = ExtremeLearningMachine(n_hidden=100, random_state=42)
        
        # Fit on same data
        X = np.random.randn(50, 5)
        y = np.random.randn(50, 1)
        
        elm1.fit(X, y)
        elm2.fit(X, y)
        
        # Predictions should be identical
        pred1 = elm1.predict(X)
        pred2 = elm2.predict(X)
        np.testing.assert_array_almost_equal(pred1, pred2)


class TestELMTraining:
    """Test ELM training functionality."""
    
    def test_elm_fit_numpy_arrays(self):
        """Test fitting ELM with numpy arrays."""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100, 1)
        
        elm = ExtremeLearningMachine(n_hidden=50, alpha=1e-3)
        elm.fit(X_train, y_train)
        
        assert elm.W_hidden is not None
        assert elm.W_out is not None
        assert elm.W_hidden.shape == (5, 50)
        assert elm.W_out.shape == (50, 1)
    
    def test_elm_fit_torch_tensors(self):
        """Test fitting ELM with torch tensors."""
        X_train = torch.randn(100, 5)
        y_train = torch.randn(100, 1)
        
        elm = ExtremeLearningMachine(n_hidden=50, alpha=1e-3)
        elm.fit(X_train, y_train)
        
        assert elm.W_hidden is not None
        assert elm.W_out is not None
    
    def test_elm_fit_1d_target(self):
        """Test fitting ELM with 1D target array."""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)  # 1D array
        
        elm = ExtremeLearningMachine(n_hidden=50, alpha=1e-3)
        elm.fit(X_train, y_train)
        
        assert elm.W_out.shape == (50, 1)
    
    def test_elm_fit_multi_output(self):
        """Test fitting ELM with multiple outputs."""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100, 3)  # 3 outputs
        
        elm = ExtremeLearningMachine(n_hidden=50, alpha=1e-3)
        elm.fit(X_train, y_train)
        
        assert elm.W_out.shape == (50, 3)
    
    def test_elm_fit_with_batch_size(self):
        """Test fitting ELM with batch processing."""
        X_train = np.random.randn(1000, 10)
        y_train = np.random.randn(1000, 1)
        
        elm = ExtremeLearningMachine(n_hidden=200, alpha=1e-3)
        elm.fit(X_train, y_train, batch_size=100)
        
        assert elm.W_hidden is not None
        assert elm.W_out is not None


class TestELMPrediction:
    """Test ELM prediction functionality."""
    
    def test_elm_predict_shape(self):
        """Test that predictions have correct shape."""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100, 1)
        X_test = np.random.randn(20, 5)
        
        elm = ExtremeLearningMachine(n_hidden=50, alpha=1e-3)
        elm.fit(X_train, y_train)
        predictions = elm.predict(X_test)
        
        assert predictions.shape == (20,)  # 1D for single output
    
    def test_elm_predict_multi_output_shape(self):
        """Test predictions shape for multi-output."""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100, 3)
        X_test = np.random.randn(20, 5)
        
        elm = ExtremeLearningMachine(n_hidden=50, alpha=1e-3)
        elm.fit(X_train, y_train)
        predictions = elm.predict(X_test)
        
        assert predictions.shape == (20, 3)
    
    def test_elm_decision_function(self):
        """Test decision_function method."""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100, 1)
        X_test = np.random.randn(20, 5)
        
        elm = ExtremeLearningMachine(n_hidden=50, alpha=1e-3)
        elm.fit(X_train, y_train)
        scores = elm.decision_function(X_test)
        
        assert scores.shape == (20, 1)
        assert isinstance(scores, np.ndarray)


class TestELMActivations:
    """Test different activation functions."""
    
    @pytest.mark.parametrize("activation", ['ReLU', 'LeakyReLU', 'ELU', 'GELU'])
    def test_elm_with_different_activations(self, activation):
        """Test ELM with different activation functions."""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100, 1)
        
        elm = ExtremeLearningMachine(n_hidden=50, activation=activation, alpha=1e-3)
        elm.fit(X_train, y_train)
        predictions = elm.predict(X_train)
        
        assert predictions.shape == (100,)
        assert not np.any(np.isnan(predictions))


class TestELMScore:
    """Test ELM scoring functionality."""
    
    def test_elm_score_r2(self):
        """Test R² score calculation."""
        # Create simple linear data
        X_train = np.random.randn(100, 5)
        y_train = X_train.sum(axis=1, keepdims=True) + 0.1 * np.random.randn(100, 1)
        
        elm = ExtremeLearningMachine(n_hidden=100, alpha=1e-4)
        elm.fit(X_train, y_train)
        r2 = elm.score(X_train, y_train)
        
        assert -1 <= r2 <= 1
        assert r2 > 0.5  # Should fit reasonably well
    
    def test_elm_score_perfect_fit(self):
        """Test score on training data should be high."""
        X_train = np.random.randn(50, 3)
        y_train = np.random.randn(50, 1)
        
        # Use many hidden neurons for perfect fit
        elm = ExtremeLearningMachine(n_hidden=200, alpha=1e-6)
        elm.fit(X_train, y_train)
        r2 = elm.score(X_train, y_train)
        
        assert r2 > 0.9  # Should fit very well


class TestELMRegularization:
    """Test regularization effects."""
    
    def test_elm_different_alpha_values(self):
        """Test that different alpha values produce different results."""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100, 1)
        X_test = np.random.randn(20, 5)
        
        elm_small_alpha = ExtremeLearningMachine(n_hidden=50, alpha=1e-6, random_state=42)
        elm_large_alpha = ExtremeLearningMachine(n_hidden=50, alpha=1e-1, random_state=42)
        
        elm_small_alpha.fit(X_train, y_train)
        elm_large_alpha.fit(X_train, y_train)
        
        pred_small = elm_small_alpha.predict(X_test)
        pred_large = elm_large_alpha.predict(X_test)
        
        # Predictions should differ
        assert not np.allclose(pred_small, pred_large)


class TestELMBias:
    """Test bias handling."""
    
    def test_elm_with_bias(self):
        """Test ELM with bias enabled."""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100, 1)
        
        elm = ExtremeLearningMachine(n_hidden=50, include_bias=True)
        elm.fit(X_train, y_train)
        
        assert elm.b_hidden is not None
        assert elm.b_hidden.shape == (50,)
    
    def test_elm_without_bias(self):
        """Test ELM with bias disabled."""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100, 1)
        
        elm = ExtremeLearningMachine(n_hidden=50, include_bias=False)
        elm.fit(X_train, y_train)
        
        # Bias should be zeros
        assert torch.all(elm.b_hidden == 0)


class TestELMEdgeCases:
    """Test edge cases and error handling."""
    
    def test_elm_small_dataset(self):
        """Test ELM with very small dataset."""
        X_train = np.random.randn(10, 3)
        y_train = np.random.randn(10, 1)
        
        elm = ExtremeLearningMachine(n_hidden=20, alpha=1e-3)
        elm.fit(X_train, y_train)
        predictions = elm.predict(X_train)
        
        assert predictions.shape == (10,)
    
    def test_elm_large_hidden_layer(self):
        """Test ELM with more hidden neurons than samples (uses dual form)."""
        X_train = np.random.randn(50, 5)
        y_train = np.random.randn(50, 1)
        
        # More hidden neurons than samples triggers dual form
        elm = ExtremeLearningMachine(n_hidden=100, alpha=1e-3)
        elm.fit(X_train, y_train)
        predictions = elm.predict(X_train)
        
        assert predictions.shape == (50,)
        assert elm.W_out.shape == (100, 1)
    
    def test_elm_with_gpu_if_available(self):
        """Test ELM can use GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100, 1)
        
        elm = ExtremeLearningMachine(n_hidden=50, device=device, alpha=1e-3)
        elm.fit(X_train, y_train)
        
        assert elm.W_hidden.device.type == 'cuda'
        assert elm.W_out.device.type == 'cuda'
        
        predictions = elm.predict(X_train)
        assert predictions.shape == (100,)
