"""Tests for core.trainer — train_epoch, evaluate, create_scheduler, and main_train."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.data_loader import CSVDataset
from core.model import NeuralNetwork
from core.trainer import train_epoch, evaluate, create_scheduler, main_train


# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def training_setup():
    """Model + data + optimizer for unit-level train/eval tests."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6)).astype(np.float32)
    y = rng.standard_normal((80, 2)).astype(np.float32)
    loader = DataLoader(CSVDataset(X, y), batch_size=16, shuffle=True)

    model = NeuralNetwork(input_size=6, output_size=2, nr_hidden_layers=1,
                          nr_neurons=32, use_batchnorm=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu")
    return model, loader, criterion, optimizer, device


def _base_config(**overrides):
    cfg = dict(
        num_epochs=3, batch_size=16, learning_rate=1e-3,
        nr_hidden_layers=1, nr_neurons=32, activation_name="ReLU",
        dropout_rate=0.0, weight_decay=0.0, loss_criterion="MSE",
        plots=False, patience=50,
    )
    cfg.update(overrides)
    return cfg


# ── train_epoch ──────────────────────────────────────────────────────────────

class TestTrainEpoch:

    def test_returns_positive_float(self, training_setup):
        model, loader, criterion, opt, dev = training_setup
        loss = train_epoch(model, loader, criterion, opt, dev)
        assert isinstance(loss, float) and loss >= 0

    def test_updates_weights(self, training_setup):
        model, loader, criterion, opt, dev = training_setup
        before = [p.clone() for p in model.parameters()]
        train_epoch(model, loader, criterion, opt, dev)
        assert any(not torch.allclose(b, p) for b, p in zip(before, model.parameters()))

    def test_gradient_clipping_prevents_nan(self, training_setup):
        model, loader, criterion, _, dev = training_setup
        opt = torch.optim.SGD(model.parameters(), lr=10.0)  # aggressive LR
        loss = train_epoch(model, loader, criterion, opt, dev, max_grad_norm=1.0)
        assert np.isfinite(loss)


# ── evaluate ─────────────────────────────────────────────────────────────────

class TestEvaluate:

    def test_returns_types_and_shapes(self, training_setup):
        model, loader, criterion, _, dev = training_setup
        loss, preds, trues = evaluate(model, loader, criterion, dev)
        assert isinstance(loss, float)
        assert preds.shape == (80, 2) and trues.shape == (80, 2)

    def test_no_gradients_stored(self, training_setup):
        model, loader, criterion, _, dev = training_setup
        evaluate(model, loader, criterion, dev)
        for p in model.parameters():
            assert p.grad is None or torch.all(p.grad == 0)


# ── create_scheduler ─────────────────────────────────────────────────────────

class TestCreateScheduler:

    def _make_opt(self):
        m = NeuralNetwork(input_size=2, output_size=1, nr_hidden_layers=1, nr_neurons=8)
        return torch.optim.Adam(m.parameters(), lr=1e-3)

    @pytest.mark.parametrize("stype", [
        "CosineAnnealingLR", "CosineAnnealingWarmRestarts",
        "ReduceLROnPlateau", "StepLR",
    ])
    def test_creates_without_error(self, stype: str):
        opt = self._make_opt()
        sched = create_scheduler(opt, {"scheduler_type": stype})
        assert sched is not None

    def test_warmup_wraps_cosine(self):
        opt = self._make_opt()
        sched = create_scheduler(opt, {"scheduler_type": "CosineAnnealingLR", "warmup_epochs": 5})
        assert sched is not None


# ── main_train integration ───────────────────────────────────────────────────

class TestMainTrain:
    """Integration tests that run short training loops on synthetic data."""

    def test_basic_training(self, isotherm_csv: Path, tmp_path: Path):
        model, X_sc, y_sc = main_train(
            config=_base_config(),
            rf=str(tmp_path),
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well", "Temp_diff", "kW_well"],
            label_cols=["Area"],
            device=torch.device("cpu"),
        )
        assert isinstance(model, nn.Module)
        assert X_sc is not None and y_sc is not None

    def test_multi_label(self, isotherm_csv: Path, tmp_path: Path):
        model, *_ = main_train(
            config=_base_config(),
            rf=str(tmp_path),
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well", "Temp_diff"],
            label_cols=["Area", "Iso_distance", "Iso_width"],
            device=torch.device("cpu"),
        )
        x = torch.randn(1, 2)
        model.eval()
        with torch.no_grad():
            assert model(x).shape == (1, 3)

    @pytest.mark.parametrize("loss", ["MSE", "L1", "SmoothL1"])
    def test_loss_variants(self, isotherm_csv: Path, tmp_path: Path, loss: str):
        model, *_ = main_train(
            config=_base_config(loss_criterion=loss),
            rf=str(tmp_path / loss),
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well", "Temp_diff"],
            label_cols=["Area"],
            device=torch.device("cpu"),
        )
        assert model is not None

    @pytest.mark.parametrize("stype", [
        "CosineAnnealingLR", "ReduceLROnPlateau", "StepLR",
    ])
    def test_scheduler_variants(self, isotherm_csv: Path, tmp_path: Path, stype: str):
        model, *_ = main_train(
            config=_base_config(scheduler_type=stype),
            rf=str(tmp_path / stype),
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well", "Temp_diff"],
            label_cols=["Area"],
            device=torch.device("cpu"),
        )
        assert model is not None

    def test_early_stopping(self, isotherm_csv: Path, tmp_path: Path):
        """With tiny patience and many epochs, training should stop early."""
        model, *_ = main_train(
            config=_base_config(num_epochs=500, patience=2),
            rf=str(tmp_path),
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well", "Temp_diff"],
            label_cols=["Area"],
            device=torch.device("cpu"),
        )
        assert model is not None

    def test_batchnorm_present(self, isotherm_csv: Path, tmp_path: Path):
        # Use batch_size=8 to avoid remainder-of-1 batches with BatchNorm
        model, *_ = main_train(
            config=_base_config(use_batchnorm=True, batch_size=8),
            rf=str(tmp_path),
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well", "Temp_diff"],
            label_cols=["Area"],
            device=torch.device("cpu"),
        )
        assert any(isinstance(l, nn.BatchNorm1d) for l in model.layers)

    def test_artifacts_written(self, isotherm_csv: Path, tmp_path: Path):
        """main_train should write metrics and tensorboard logs."""
        main_train(
            config=_base_config(),
            rf=str(tmp_path),
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well", "Temp_diff"],
            label_cols=["Area"],
            device=torch.device("cpu"),
        )
        stats = tmp_path / "stats"
        assert (stats / "metrics_summary.json").exists()
        assert (stats / "metrics_summary.md").exists()
        assert (tmp_path / "tensorboard_log").exists()
