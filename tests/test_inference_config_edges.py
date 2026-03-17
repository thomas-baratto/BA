"""Edge-case tests for inference model config handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from core.model import NeuralNetwork
from scripts.inference import load_model_and_scalers


def _save_model(model_path: Path):
    model = NeuralNetwork(input_size=3, output_size=1, nr_hidden_layers=1, nr_neurons=8)
    torch.save(model.state_dict(), model_path)


def test_load_model_and_scalers_invalid_json_raises(tmp_path):
    model_path = tmp_path / 'model.pt'
    _save_model(model_path)

    config_path = tmp_path / 'model_config.json'
    config_path.write_text('{not valid json}', encoding='utf-8')

    with pytest.raises(json.JSONDecodeError):
        load_model_and_scalers(str(model_path), str(tmp_path))


def test_load_model_and_scalers_missing_required_key_raises(tmp_path):
    model_path = tmp_path / 'model.pt'
    _save_model(model_path)

    config_path = tmp_path / 'model_config.json'
    # Missing input_size/output_size and others required by loader
    config_path.write_text(json.dumps({'nr_hidden_layers': 1}), encoding='utf-8')

    with pytest.raises(KeyError):
        load_model_and_scalers(str(model_path), str(tmp_path))
