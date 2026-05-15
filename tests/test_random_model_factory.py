"""Contract tests for random model factory in train_random_models."""

from __future__ import annotations

import argparse

import torch

from core.random.ELM import ELM
from core.random.SResdRVFL import SResdRVFL
from core.random.dRVFL import dRVFL
from core.random.edRVFL import edRVFL
from core.random.edRVFL_SC import edRVFL_SC
from core.random.esc_edRVFL import esc_edRVFL
from scripts.training.train_random_models import init_model


def _base_args(model: str) -> argparse.Namespace:
    return argparse.Namespace(
        model=model,
        n_hidden=16,
        activation='ReLU',
        alpha=1e-3,
        gamma=1.0,
        random_state=42,
        n_layers=2,
        n_ensemble=3,
        sc_mode='random',
        rsc_prob=0.4,
        n_folds=2,
        n_blocks=2,
        direct_link=True,
    )


def test_init_model_returns_expected_types():
    device = torch.device('cpu')

    mapping = {
        'ELM': ELM,
        'dRVFL': dRVFL,
        'edRVFL': edRVFL,
        'edRVFL-SC': edRVFL_SC,
        'esc-edRVFL': esc_edRVFL,
        'SResdRVFL': SResdRVFL,
    }

    for model_name, expected_cls in mapping.items():
        args = _base_args(model_name)
        model = init_model(args, device)
        assert isinstance(model, expected_cls)


def test_init_model_passes_key_configuration():
    device = torch.device('cpu')

    args_sc = _base_args('edRVFL-SC')
    model_sc = init_model(args_sc, device)
    assert model_sc.mode == 'random'
    assert model_sc.rsc_prob == 0.4

    args_res = _base_args('SResdRVFL')
    model_res = init_model(args_res, device)
    assert model_res.direct_link is True
    assert model_res.n_blocks == 2
