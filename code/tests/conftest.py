"""Pytest shared fixtures and path resolution."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Reusable CSV fixtures
# ---------------------------------------------------------------------------

def _isotherm_df(n: int = 100) -> pd.DataFrame:
    """Build a synthetic isotherm-like DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Flow_well": rng.uniform(100, 500, n),
        "Temp_diff": rng.uniform(5, 50, n),
        "kW_well": rng.uniform(5, 25, n),
        "Hydr_gradient": rng.uniform(0.01, 0.5, n),
        "Hydr_conductivity": rng.uniform(1, 10, n),
        "Aqu_thickness": rng.uniform(10, 50, n),
        "Long_dispersivity": rng.uniform(1, 10, n),
        "Trans_dispersivity": rng.uniform(0.1, 1.0, n),
        "Isotherm": rng.uniform(1, 5, n),
        "Area": rng.uniform(100, 50_000, n),
        "Iso_distance": rng.uniform(10, 200, n),
        "Iso_width": rng.uniform(5, 50, n),
    })


def _cone_df(n: int = 40) -> pd.DataFrame:
    """Build a synthetic cone-like DataFrame."""
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "Flow_well": rng.uniform(100, 500, n),
        "Hydr_gradient": rng.uniform(0.01, 0.5, n),
        "Hydr_conductivity": rng.uniform(1, 10, n),
        "Aqu_thickness": rng.uniform(10, 50, n),
        "Cone": rng.uniform(0.5, 20.0, n),
    })


@pytest.fixture()
def isotherm_csv(tmp_path: Path) -> Path:
    """Write a temporary isotherm CSV and return its path."""
    p = tmp_path / "isotherm.csv"
    _isotherm_df().to_csv(p, index=False)
    return p


@pytest.fixture()
def cone_csv(tmp_path: Path) -> Path:
    """Write a temporary cone CSV and return its path."""
    p = tmp_path / "cone.csv"
    _cone_df().to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Model artifact fixtures (for inference / model_wrapper tests)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mlp_artifact_dir(tmp_path: Path) -> Path:
    """Create a minimal MLP artifact directory with model, config, and scalers."""
    from core.model import NeuralNetwork

    d = tmp_path / "mlp_model"
    d.mkdir()
    (d / "stats").mkdir()

    input_size, output_size = 4, 1
    model = NeuralNetwork(
        input_size=input_size,
        output_size=output_size,
        nr_hidden_layers=1,
        nr_neurons=16,
        activation_name="ReLU",
        dropout_rate=0.0,
        use_batchnorm=False,
    )
    torch.save(model.state_dict(), d / "best_model.pt")

    config = {
        "input_size": input_size,
        "output_size": output_size,
        "nr_hidden_layers": 1,
        "nr_neurons": 16,
        "activation_name": "ReLU",
        "dropout_rate": 0.0,
        "use_batchnorm": False,
        "feature_names": ["f1", "f2", "f3", "f4"],
        "label_names": ["target"],
    }
    with open(d / "model_config.json", "w") as f:
        json.dump(config, f)

    feat_scaler = MinMaxScaler().fit(np.random.randn(20, input_size))
    lbl_scaler = MinMaxScaler().fit(np.random.randn(20, output_size))
    with open(d / "scalers.pkl", "wb") as f:
        pickle.dump({"feature_scaler": feat_scaler, "label_scaler": lbl_scaler}, f)

    return d
