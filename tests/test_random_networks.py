"""Tests for random network models in core.random."""

import numpy as np
import pytest

from core.random.ELM import ELM
from core.random.dRVFL import dRVFL
from core.random.edRVFL import edRVFL
from core.random.edRVFL_SC import edRVFL_SC
from core.random.esc_edRVFL import esc_edRVFL
from core.random.SResdRVFL import SResdRVFL


def _toy_regression(n_samples=32, n_features=4, n_outputs=1):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.normal(size=(n_samples, n_outputs))
    return X, y


@pytest.mark.parametrize(
    "model",
    [
        ELM(n_hidden=16, activation="ReLU", random_state=1),
        dRVFL(n_layers=2, n_hidden=12, activation="ReLU", random_state=2),
        edRVFL(n_ensemble=3, n_layers=2, n_hidden=10, activation="ReLU", random_state=3),
        edRVFL_SC(n_ensemble=3, n_layers=2, n_hidden=10, mode="dense", random_state=4),
        edRVFL_SC(n_ensemble=3, n_layers=2, n_hidden=10, mode="random", rsc_prob=0.4, random_state=5),
        esc_edRVFL(n_folds=3, n_ensemble=2, n_layers=2, n_hidden=10, random_state=6),
        SResdRVFL(n_blocks=3, n_layers_per_block=1, n_hidden=12, random_state=7),
    ],
)
def test_random_models_fit_predict_single_output(model):
    X, y = _toy_regression(n_samples=36, n_features=5, n_outputs=1)

    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == (36,)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize(
    "model",
    [
        ELM(n_hidden=20, activation="rbf", gamma=0.7, random_state=11),
        dRVFL(n_layers=2, n_hidden=14, activation="rbf", gamma=0.7, random_state=12),
        edRVFL_SC(n_ensemble=2, n_layers=2, n_hidden=12, activation="rbf", mode="dense", gamma=0.7, random_state=13),
        SResdRVFL(n_blocks=2, n_layers_per_block=1, n_hidden=12, activation="rbf", gamma=0.7, random_state=14),
    ],
)
def test_random_models_fit_predict_rbf_activation(model):
    X, y = _toy_regression(n_samples=28, n_features=3, n_outputs=1)

    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == (28,)
    assert np.isfinite(pred).all()


def test_random_models_multi_output_shape():
    X, y = _toy_regression(n_samples=30, n_features=4, n_outputs=3)

    models = [
        ELM(n_hidden=12, random_state=21),
        dRVFL(n_layers=2, n_hidden=10, random_state=22),
        edRVFL(n_ensemble=2, n_layers=2, n_hidden=8, random_state=23),
        edRVFL_SC(n_ensemble=2, n_layers=2, n_hidden=8, mode="dense", random_state=24),
        esc_edRVFL(n_folds=2, n_ensemble=2, n_layers=2, n_hidden=8, random_state=25),
        SResdRVFL(n_blocks=2, n_layers_per_block=1, n_hidden=10, random_state=26),
    ]

    for model in models:
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (30, 3)
        assert np.isfinite(pred).all()


def test_ensemble_predict_before_fit_raises():
    model = edRVFL(n_ensemble=2, n_layers=1, n_hidden=8)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 3)))

    model_sc = edRVFL_SC(n_ensemble=2, n_layers=1, n_hidden=8)
    with pytest.raises(RuntimeError):
        model_sc.predict(np.zeros((5, 3)))

    model_esc = esc_edRVFL(n_folds=2, n_ensemble=2, n_layers=1, n_hidden=8)
    with pytest.raises(RuntimeError):
        model_esc.predict(np.zeros((5, 3)))

    model_sres = SResdRVFL(n_blocks=2, n_layers_per_block=1, n_hidden=8)
    with pytest.raises(RuntimeError):
        model_sres.predict(np.zeros((5, 3)))
