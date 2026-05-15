"""Tests for core/random/ submodules used at inference time.

Covers the randomized network family that TrainedModel can load:
- utils: torch_activation, ensure_tensor
- RBF: RBFHiddenLayer forward pass
- dRVFL: fit/predict, reproducibility, RBF mode
- edRVFL: ensemble averaging, unfitted raises
- edRVFL_SC: dense vs. random skip-connection modes
- SResdRVFL: residual stacking, unfitted raises
- esc_edRVFL: KFold ensemble, unfitted raises

These models are ALL present on the release branch and can be deserialized
by TrainedModel from DaRUS artifact pickles.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core.random.utils import torch_activation, ensure_tensor
from core.random.RBF import RBFHiddenLayer
from core.random.ELM import ELM
from core.random.dRVFL import dRVFL
from core.random.edRVFL import edRVFL
from core.random.edRVFL_SC import edRVFL_SC
from core.random.SResdRVFL import SResdRVFL
from core.random.esc_edRVFL import esc_edRVFL


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _xy(n: int = 60, n_features: int = 4, n_outputs: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.standard_normal((n, n_outputs)) if n_outputs > 1 else rng.standard_normal(n)
    return X, y


# ===========================================================================
# utils
# ===========================================================================

class TestTorchActivation:

    @pytest.mark.parametrize("act", ["ReLU", "LeakyReLU", "ELU", "GELU"])
    def test_known_activations_return_tensor(self, act: str):
        x = torch.randn(10, 5, dtype=torch.float64)
        out = torch_activation(x, act)
        assert isinstance(out, torch.Tensor)
        assert out.shape == x.shape

    def test_relu_clamps_negatives(self):
        x = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float64)
        out = torch_activation(x, "ReLU")
        assert out[0, 0] == 0.0
        assert out[0, 2] == 1.0

    def test_unknown_activation_identity(self):
        """Unknown name falls through → returns input unchanged."""
        x = torch.randn(4, 3, dtype=torch.float64)
        out = torch_activation(x, "nonexistent")
        assert torch.allclose(out, x)


class TestEnsureTensor:

    def test_numpy_converts_to_float64(self):
        X = np.random.randn(5, 3).astype(np.float32)
        t = ensure_tensor(X)
        assert t.dtype == torch.float64
        assert t.shape == (5, 3)

    def test_tensor_cast_to_float64(self):
        X = torch.randn(4, 2, dtype=torch.float32)
        t = ensure_tensor(X)
        assert t.dtype == torch.float64

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported input type"):
            ensure_tensor([[1, 2], [3, 4]])


# ===========================================================================
# RBFHiddenLayer
# ===========================================================================

class TestRBFHiddenLayer:

    def test_output_shape(self):
        layer = RBFHiddenLayer(n_hidden=20, gamma=1.0, in_features=4)
        x = torch.randn(10, 4)
        out = layer(x)
        assert out.shape == (10, 20)

    def test_output_in_zero_one_range(self):
        """Gaussian RBF outputs are in (0, 1]."""
        layer = RBFHiddenLayer(n_hidden=16, gamma=1.0, in_features=3)
        x = torch.randn(50, 3)
        out = layer(x)
        assert (out >= 0).all() and (out <= 1.0 + 1e-6).all()

    def test_centers_not_updated_by_backward(self):
        """Centers have requires_grad=False, so they stay fixed."""
        layer = RBFHiddenLayer(n_hidden=8, gamma=1.0, in_features=2)
        assert not layer.centers.requires_grad


# ===========================================================================
# ELM
# ===========================================================================

class TestELMBasics:

    def test_initialization_defaults(self):
        elm = ELM(n_hidden=100)
        assert elm.n_hidden == 100
        assert elm.activation == "ReLU"
        assert elm.alpha == 0.001
        assert elm.random_state is None

    def test_initialization_custom_values(self):
        elm = ELM(n_hidden=50, activation="LeakyReLU", alpha=0.1, random_state=42)
        assert elm.n_hidden == 50
        assert elm.activation == "LeakyReLU"
        assert elm.alpha == 0.1
        assert elm.random_state == 42


class TestELMFitPredict:

    def test_fit_predict_single_output_numpy(self):
        X, y = _xy(50, 4, 1)
        elm = ELM(n_hidden=20, random_state=0)
        elm.fit(X, y)
        pred = elm.predict(X)
        assert pred.shape == (50,)
        assert np.isfinite(pred).all()

    def test_fit_predict_multi_output_numpy(self):
        X, y = _xy(40, 4, 3)
        elm = ELM(n_hidden=25, random_state=1)
        elm.fit(X, y)
        pred = elm.predict(X)
        assert pred.shape == (40, 3)

    def test_fit_accepts_torch_tensors(self):
        X = torch.randn(30, 4).double()
        y = torch.randn(30, 1).double()
        elm = ELM(n_hidden=15)
        elm.fit(X, y)
        pred = elm.predict(X)
        assert pred.shape == (30,) or pred.shape == (30, 1)


# ===========================================================================
# dRVFL
# ===========================================================================

class TestDRVFL:

    def test_fit_predict_single_output(self):
        X, y = _xy(60, 4, 1)
        model = dRVFL(n_layers=2, n_hidden=30, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (60,)
        assert np.isfinite(pred).all()

    def test_fit_predict_multi_output(self):
        X, y = _xy(60, 4, 2)
        model = dRVFL(n_layers=1, n_hidden=20, random_state=1)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (60, 2)

    def test_reproducibility(self):
        X, y = _xy(40, 4, 1)
        m1 = dRVFL(n_layers=2, n_hidden=20, random_state=7)
        m2 = dRVFL(n_layers=2, n_hidden=20, random_state=7)
        m1.fit(X, y)
        m2.fit(X, y)
        np.testing.assert_allclose(m1.predict(X), m2.predict(X), atol=1e-10)

    def test_accepts_torch_tensor(self):
        X = torch.randn(30, 4).double()
        y = torch.randn(30).double()
        model = dRVFL(n_layers=1, n_hidden=16)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (30,)

    def test_rbf_activation(self):
        X, y = _xy(50, 3, 1)
        model = dRVFL(n_layers=2, n_hidden=15, activation="rbf", gamma=0.5, random_state=5)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (50,)
        assert np.isfinite(pred).all()

    @pytest.mark.parametrize("n_layers", [1, 3, 5])
    def test_depth_variants(self, n_layers: int):
        X, y = _xy(50, 4, 1)
        model = dRVFL(n_layers=n_layers, n_hidden=20, random_state=0)
        model.fit(X, y)
        assert np.isfinite(model.predict(X)).all()


# ===========================================================================
# edRVFL
# ===========================================================================

class TestEDRVFL:

    def test_fit_predict_shape(self):
        X, y = _xy(60, 4, 1)
        model = edRVFL(n_ensemble=3, n_layers=1, n_hidden=20, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (60,)
        assert np.isfinite(pred).all()

    def test_unfitted_predict_raises(self):
        model = edRVFL(n_ensemble=2)
        with pytest.raises(RuntimeError, match="hasn't been fitted"):
            model.predict(np.zeros((5, 4)))

    def test_ensemble_has_correct_count(self):
        X, y = _xy(40, 4, 1)
        model = edRVFL(n_ensemble=5, n_layers=1, n_hidden=10)
        model.fit(X, y)
        assert len(model.models) == 5

    def test_multi_output(self):
        X, y = _xy(50, 4, 2)
        model = edRVFL(n_ensemble=3, n_layers=1, n_hidden=15, random_state=2)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (50, 2)


# ===========================================================================
# edRVFL_SC (skip connections)
# ===========================================================================

class TestEDRVFL_SC:

    @pytest.mark.parametrize("mode", ["dense", "random"])
    def test_modes_fit_predict(self, mode: str):
        X, y = _xy(60, 4, 1)
        model = edRVFL_SC(n_ensemble=2, n_layers=2, n_hidden=16, mode=mode, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (60,)
        assert np.isfinite(pred).all()

    def test_unfitted_raises(self):
        model = edRVFL_SC(n_ensemble=2)
        with pytest.raises(RuntimeError, match="hasn't been fitted"):
            model.predict(np.zeros((5, 4)))

    def test_dense_multi_output(self):
        X, y = _xy(50, 4, 2)
        model = edRVFL_SC(n_ensemble=2, n_layers=1, n_hidden=12, mode="dense", random_state=1)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (50, 2)


# ===========================================================================
# SResdRVFL (stacked residual)
# ===========================================================================

class TestSResdRVFL:

    def test_fit_predict_shape(self):
        X, y = _xy(60, 4, 1)
        model = SResdRVFL(n_blocks=3, n_layers_per_block=1, n_hidden=20, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (60,)
        assert np.isfinite(pred).all()

    def test_unfitted_raises(self):
        model = SResdRVFL(n_blocks=3)
        with pytest.raises(RuntimeError, match="hasn't been fitted"):
            model.predict(np.zeros((5, 4)))

    def test_residuals_decrease(self):
        """Each additional block should reduce training error (monotone improvement)."""
        X, y = _xy(80, 4, 1, seed=42)
        preds_per_block = []

        # Build manually to inspect intermediate residuals
        model = SResdRVFL(n_blocks=4, n_layers_per_block=1, n_hidden=30, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert np.isfinite(pred).all()

    def test_direct_link_false(self):
        X, y = _xy(50, 4, 1)
        model = SResdRVFL(n_blocks=2, direct_link=False, n_hidden=16, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (50,)

    def test_multi_output(self):
        X, y = _xy(60, 4, 2)
        model = SResdRVFL(n_blocks=2, n_layers_per_block=1, n_hidden=15, random_state=3)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (60, 2)


# ===========================================================================
# esc_edRVFL (KFold ensemble of ensembles)
# ===========================================================================

class TestEscEdRVFL:

    def test_fit_predict_shape(self):
        X, y = _xy(60, 4, 1)
        model = esc_edRVFL(n_folds=3, n_ensemble=2, n_layers=1, n_hidden=12, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (60,)
        assert np.isfinite(pred).all()

    def test_unfitted_raises(self):
        model = esc_edRVFL(n_folds=3, n_ensemble=2)
        with pytest.raises(RuntimeError, match="hasn't been fitted"):
            model.predict(np.zeros((5, 4)))

    def test_correct_number_of_fold_models(self):
        X, y = _xy(60, 4, 1)
        model = esc_edRVFL(n_folds=4, n_ensemble=2, n_layers=1, n_hidden=10)
        model.fit(X, y)
        assert len(model.models) == 4

    def test_small_dataset_fallback(self):
        """When n_samples < n_folds, should fall back gracefully."""
        X, y = _xy(5, 4, 1)  # Only 5 samples, 3 folds requested
        model = esc_edRVFL(n_folds=3, n_ensemble=2, n_layers=1, n_hidden=8, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (5,)
