"""Tests for the ELM implementation in core.random."""

import numpy as np
import pytest
import torch

from core.random.ELM import ELM


class TestELMBasics:
    def test_initialization_defaults(self):
        model = ELM()
        assert model.n_hidden == 1024
        assert model.activation == "ReLU"
        assert model.alpha == 1e-3
        assert model.include_bias is True

    def test_initialization_custom_values(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ELM(n_hidden=64, activation="GELU", alpha=1e-2, include_bias=False, device=device)
        assert model.n_hidden == 64
        assert model.activation == "GELU"
        assert model.alpha == 1e-2
        assert model.include_bias is False
        assert model.device == device


class TestELMFitPredict:
    def test_fit_predict_single_output_numpy(self):
        X = np.random.randn(120, 5)
        y = np.random.randn(120, 1)

        model = ELM(n_hidden=50, random_state=42)
        model.fit(X, y)

        pred = model.predict(X)
        assert pred.shape == (120,)
        assert model.W_out is not None

    def test_fit_predict_multi_output_numpy(self):
        X = np.random.randn(80, 4)
        y = np.random.randn(80, 3)

        model = ELM(n_hidden=40, random_state=42)
        model.fit(X, y)

        pred = model.predict(X)
        assert pred.shape == (80, 3)
        assert model.W_out.shape == (40, 3)

    def test_fit_accepts_torch_tensors(self):
        X = torch.randn(64, 6)
        y = torch.randn(64, 1)

        model = ELM(n_hidden=32)
        model.fit(X, y)

        pred = model.predict(X)
        assert pred.shape == (64,)


class TestELMBehavior:
    def test_reproducibility_with_random_state(self):
        X = np.random.randn(50, 3)
        y = np.random.randn(50, 1)

        model_a = ELM(n_hidden=30, random_state=123)
        model_b = ELM(n_hidden=30, random_state=123)

        model_a.fit(X, y)
        model_b.fit(X, y)

        pred_a = model_a.predict(X)
        pred_b = model_b.predict(X)
        np.testing.assert_allclose(pred_a, pred_b, atol=1e-10)

    @pytest.mark.parametrize("activation", ["ReLU", "LeakyReLU", "ELU", "GELU", "rbf"])
    def test_supported_activations(self, activation):
        X = np.random.randn(70, 4)
        y = np.random.randn(70, 1)

        model = ELM(n_hidden=25, activation=activation, gamma=0.5)
        model.fit(X, y)
        pred = model.predict(X)

        assert pred.shape == (70,)
        assert np.isfinite(pred).all()

    def test_without_bias_uses_zero_bias(self):
        X = np.random.randn(60, 3)
        y = np.random.randn(60, 1)

        model = ELM(n_hidden=20, include_bias=False)
        model.fit(X, y)

        assert torch.all(model.b_hidden == 0)
