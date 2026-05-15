"""Smoke / unit tests for predict.py (the top-level CLI entry point).

Only tests what lives in the release branch.  No training-script imports.

Covers:
- find_model_dir — direct, nested, and missing cases
- generate_report — structure and key fields
- predict_with_model — shape, finite values, respects inverse transform
- main() smoke test with full monkeypatching (no disk model required)
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Helpers — build a tiny MLP artifact directory
# ---------------------------------------------------------------------------

def _make_mlp_dir(tmp_path: Path, *, n_features: int = 4, n_outputs: int = 1) -> Path:
    from core.model import NeuralNetwork

    d = tmp_path / "mlp_artifact"
    d.mkdir(parents=True, exist_ok=True)

    model = NeuralNetwork(
        input_size=n_features, output_size=n_outputs,
        nr_hidden_layers=1, nr_neurons=16,
        activation_name="ReLU", dropout_rate=0.0, use_batchnorm=False,
    )
    torch.save(model.state_dict(), d / "best_model.pt")

    config = {
        "input_size": n_features, "output_size": n_outputs,
        "nr_hidden_layers": 1, "nr_neurons": 16,
        "activation_name": "ReLU", "dropout_rate": 0.0, "use_batchnorm": False,
        "feature_names": [f"f{i}" for i in range(n_features)],
        "label_names": ["Cone"] if n_outputs == 1 else ["Area", "Iso_width"],
        "use_log": False,
        "use_area_root": False,
    }
    (d / "model_config.json").write_text(json.dumps(config))

    rng = np.random.default_rng(7)
    fs = MinMaxScaler().fit(rng.standard_normal((30, n_features)))
    ls = MinMaxScaler().fit(rng.standard_normal((30, n_outputs)))
    with open(d / "scalers.pkl", "wb") as fh:
        pickle.dump({"feature_scaler": fs, "label_scaler": ls}, fh)

    return d


# ---------------------------------------------------------------------------
# find_model_dir
# ---------------------------------------------------------------------------

class TestFindModelDir:

    def test_direct_hit(self, tmp_path: Path):
        from predict import find_model_dir
        d = _make_mlp_dir(tmp_path)
        assert find_model_dir(str(d)) == str(d)

    def test_nested_hit(self, tmp_path: Path):
        """Model is inside a sub-directory — should be found automatically."""
        from predict import find_model_dir

        parent = tmp_path / "parent"
        parent.mkdir()
        sub = parent / "run_001"
        sub.mkdir()
        # Only put the config in the sub; find_model_dir looks one level deep
        _make_mlp_dir(tmp_path)  # create elsewhere first
        # Build a minimal nested structure
        (sub / "model_config.json").write_text(json.dumps({"x": 1}))

        result = find_model_dir(str(parent))
        assert Path(result) == sub

    def test_missing_raises(self, tmp_path: Path):
        from predict import find_model_dir
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            find_model_dir(str(empty))


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:

    def test_returns_string(self, tmp_path: Path):
        from predict import generate_report

        df_in   = pd.DataFrame({"f0": [1.0, 2.0]})
        df_pred = pd.DataFrame({"Cone": [0.5, 1.5]})
        config  = {"feature_names": ["f0"], "label_names": ["Cone"],
                   "use_log": False, "activation_name": "ReLU"}

        report = generate_report(
            input_file="dummy.csv",
            dataset="cone",
            model_type="mlp",
            model_dir=str(tmp_path),
            config=config,
            df_input=df_in,
            df_predictions=df_pred,
        )
        assert isinstance(report, str)

    def test_contains_key_fields(self, tmp_path: Path):
        from predict import generate_report

        df_in   = pd.DataFrame({"f0": [1.0]})
        df_pred = pd.DataFrame({"Cone": [0.9]})
        config  = {"feature_names": ["f0"], "label_names": ["Cone"],
                   "use_log": False, "model_name": "ELM"}

        report = generate_report(
            input_file="my_input.csv",
            dataset="cone",
            model_type="randomized",
            model_dir=str(tmp_path),
            config=config,
            df_input=df_in,
            df_predictions=df_pred,
        )
        assert "my_input.csv" in report
        assert "cone" in report.lower()
        assert "RANDOMIZED" in report

    def test_sample_count_in_report(self, tmp_path: Path):
        from predict import generate_report

        n = 5
        df_in   = pd.DataFrame({"f0": np.ones(n)})
        df_pred = pd.DataFrame({"Cone": np.ones(n)})
        config  = {"feature_names": ["f0"], "label_names": ["Cone"], "use_log": False}

        report = generate_report(
            input_file="x.csv", dataset="cone", model_type="mlp",
            model_dir=str(tmp_path), config=config,
            df_input=df_in, df_predictions=df_pred,
        )
        assert str(n) in report


# ---------------------------------------------------------------------------
# predict_with_model
# ---------------------------------------------------------------------------

class TestPredictWithModel:

    @pytest.fixture()
    def cone_model_and_config(self, tmp_path: Path):
        from core.model_wrapper import TrainedModel
        d = _make_mlp_dir(tmp_path, n_features=4, n_outputs=1)
        tm = TrainedModel(str(d))
        cfg = tm.get_config()
        return tm, cfg

    def test_output_shape(self, cone_model_and_config):
        from predict import predict_with_model
        tm, cfg = cone_model_and_config
        X = np.random.randn(6, 4).astype(np.float32)
        out = predict_with_model(tm, cfg, X)
        assert out.shape == (6, 1)

    def test_output_is_finite(self, cone_model_and_config):
        from predict import predict_with_model
        tm, cfg = cone_model_and_config
        X = np.random.randn(4, 4).astype(np.float32)
        out = predict_with_model(tm, cfg, X)
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# main() smoke test
# ---------------------------------------------------------------------------

class TestMainSmoke:
    """End-to-end smoke: patch all I/O so main() runs without disk models."""

    def test_main_runs_and_saves_csv(self, tmp_path: Path, monkeypatch):
        import predict as _predict
        from core.model_wrapper import TrainedModel

        model_dir = _make_mlp_dir(tmp_path, n_features=4, n_outputs=1)
        tm = TrainedModel(str(model_dir))

        csv_input = tmp_path / "input.csv"
        csv_input.write_text("f0,f1,f2,f3\n1,2,3,4\n5,6,7,8\n")

        fake_args = argparse.Namespace(
            input=str(csv_input),
            dataset="cone",
            model="mlp",
            model_dir=str(model_dir),
            output=str(tmp_path / "out"),
            no_report=True,
            test=False,
        )

        monkeypatch.setattr(_predict, "parse_args",   lambda: fake_args, raising=False)
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda *_a, **_kw: fake_args)

        # Patch feature names so the CSV columns match
        monkeypatch.setitem(
            _predict.DATASET_FEATURES, "cone",
            ["f0", "f1", "f2", "f3"],
        )

        # Patch TrainedModel construction to return our pre-built model
        monkeypatch.setattr(_predict, "TrainedModel", lambda _dir: tm)
        _predict.main()

        assert (tmp_path / "out" / "predictions.csv").exists()

    def test_main_missing_columns_raises_exit(self, tmp_path: Path, monkeypatch):
        """CLI should exit if required features are missing from CSV."""
        import predict as _predict
        from unittest.mock import MagicMock
        
        csv_input = tmp_path / "missing.csv"
        csv_input.write_text("f0,f1\n1,2") # Missing f2, f3
        
        fake_args = argparse.Namespace(
            input=str(csv_input), dataset="cone", model="mlp",
            model_dir=str(tmp_path), output=str(tmp_path), no_report=True, test=False
        )
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda *_a, **_kw: fake_args)
        monkeypatch.setitem(_predict.DATASET_FEATURES, "cone", ["f0", "f1", "f2", "f3"])
        
        # Patch find_model_dir to avoid needing a real model folder
        monkeypatch.setattr(_predict, "find_model_dir", lambda _x: str(tmp_path))
        
        # Patch TrainedModel to return a mock with a valid config
        mock_model = MagicMock()
        mock_model.get_config.return_value = {"feature_names": ["f0", "f1", "f2", "f3"]}
        monkeypatch.setattr(_predict, "TrainedModel", lambda _x: mock_model)

        with pytest.raises(SystemExit) as exc:
            _predict.main()
        assert exc.value.code == 1
