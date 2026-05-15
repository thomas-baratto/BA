"""Tests for core.model — MLP architecture (NeuralNetwork) and get_activation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from core.model import NeuralNetwork, get_activation


# ── get_activation ───────────────────────────────────────────────────────────

class TestGetActivation:

    @pytest.mark.parametrize("name,cls", [
        ("ReLU", nn.ReLU), ("LeakyReLU", nn.LeakyReLU),
        ("ELU", nn.ELU), ("GELU", nn.GELU), ("Tanh", nn.Tanh),
    ])
    def test_known_activations(self, name: str, cls: type):
        assert isinstance(get_activation(name), cls)

    def test_unknown_defaults_to_relu(self):
        assert isinstance(get_activation("FooBar"), nn.ReLU)


# ── NeuralNetwork construction ───────────────────────────────────────────────

class TestNetworkConstruction:

    def test_forward_shape(self):
        m = NeuralNetwork(input_size=8, output_size=3, nr_hidden_layers=2, nr_neurons=32)
        assert m(torch.randn(5, 8)).shape == (5, 3)

    @pytest.mark.parametrize("n_layers", [1, 5, 10])
    def test_varying_depth(self, n_layers: int):
        m = NeuralNetwork(input_size=4, output_size=2, nr_hidden_layers=n_layers, nr_neurons=16)
        assert m(torch.randn(3, 4)).shape == (3, 2)

    def test_batchnorm_layers_present(self):
        m = NeuralNetwork(input_size=4, output_size=1, nr_hidden_layers=2, nr_neurons=16, use_batchnorm=True)
        assert any(isinstance(l, nn.BatchNorm1d) for l in m.layers)

    def test_batchnorm_layers_absent(self):
        m = NeuralNetwork(input_size=4, output_size=1, nr_hidden_layers=2, nr_neurons=16, use_batchnorm=False)
        assert not any(isinstance(l, nn.BatchNorm1d) for l in m.layers)

    def test_batchnorm_adds_parameters(self):
        kw = dict(input_size=8, output_size=2, nr_hidden_layers=2, nr_neurons=32)
        p_no = sum(p.numel() for p in NeuralNetwork(**kw, use_batchnorm=False).parameters())
        p_bn = sum(p.numel() for p in NeuralNetwork(**kw, use_batchnorm=True).parameters())
        assert p_bn > p_no


# ── Dropout behaviour ────────────────────────────────────────────────────────

class TestDropout:

    def test_nonzero_dropout_stochastic_in_train(self):
        m = NeuralNetwork(input_size=4, output_size=2, nr_hidden_layers=2,
                          nr_neurons=64, dropout_rate=0.5, use_batchnorm=False)
        m.train()
        x = torch.randn(16, 4)
        assert not torch.allclose(m(x), m(x))

    def test_zero_dropout_deterministic_in_train(self):
        m = NeuralNetwork(input_size=4, output_size=2, nr_hidden_layers=2,
                          nr_neurons=64, dropout_rate=0.0, use_batchnorm=False)
        m.train()
        x = torch.randn(16, 4)
        assert torch.allclose(m(x), m(x))

    def test_eval_mode_deterministic(self):
        m = NeuralNetwork(input_size=4, output_size=2, nr_hidden_layers=2,
                          nr_neurons=64, dropout_rate=0.5, use_batchnorm=False)
        m.eval()
        x = torch.randn(5, 4)
        with torch.no_grad():
            assert torch.allclose(m(x), m(x))


# ── Gradient flow ────────────────────────────────────────────────────────────

class TestGradientFlow:

    def test_gradients_flow(self):
        m = NeuralNetwork(input_size=4, output_size=2, nr_hidden_layers=3, nr_neurons=32)
        x = torch.randn(5, 4, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)


# ── Save / load ──────────────────────────────────────────────────────────────

class TestSaveLoad:

    def test_weights_roundtrip(self, tmp_path):
        m = NeuralNetwork(input_size=4, output_size=2, nr_hidden_layers=1, nr_neurons=16)
        x = torch.randn(3, 4)
        before = m(x)
        torch.save(m.state_dict(), tmp_path / "w.pt")

        m2 = NeuralNetwork(input_size=4, output_size=2, nr_hidden_layers=1, nr_neurons=16)
        m2.load_state_dict(torch.load(tmp_path / "w.pt", weights_only=True))
        assert torch.allclose(before, m2(x))


# ── Numerical stability ─────────────────────────────────────────────────────

class TestNumericalStability:

    @pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
    def test_finite_outputs(self, scale: float):
        m = NeuralNetwork(input_size=4, output_size=2, nr_hidden_layers=2, nr_neurons=32)
        out = m(torch.randn(8, 4) * scale)
        assert torch.isfinite(out).all()


# ── Determinism ──────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_same_seed_same_weights(self):
        torch.manual_seed(0)
        m1 = NeuralNetwork(input_size=4, output_size=2, nr_hidden_layers=1, nr_neurons=16)
        torch.manual_seed(0)
        m2 = NeuralNetwork(input_size=4, output_size=2, nr_hidden_layers=1, nr_neurons=16)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    def test_batch_size_invariance_no_batchnorm(self):
        m = NeuralNetwork(input_size=4, output_size=2, nr_hidden_layers=1,
                          nr_neurons=16, use_batchnorm=False)
        m.eval()
        x = torch.randn(6, 4)
        with torch.no_grad():
            full = m(x)
            parts = torch.cat([m(x[i:i+1]) for i in range(6)])
        assert torch.allclose(full, parts, atol=1e-6)
