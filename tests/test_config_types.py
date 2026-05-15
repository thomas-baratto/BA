"""Tests for core.config_types — typed configuration dataclasses."""

from __future__ import annotations

import argparse

from core.config_types import RandomTrainingConfig


class TestRandomTrainingConfig:

    def _sample_namespace(self, **overrides):
        defaults = dict(
            model="ELM", dataset="cone", targets=None,
            feature_scaler="robust", label_scaler="minmax",
            use_log=True, use_area_root=False,
            n_hidden=64, activation="ReLU", alpha=1e-3, gamma=1.0,
            n_layers=2, n_ensemble=3, sc_mode="dense", rsc_prob=0.5,
            n_folds=2, n_blocks=2, direct_link=False,
            base_dir="/tmp", output_dir=None,
            random_state=42, n_seeds=1,
            no_cuda=True, no_save_model=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_from_namespace(self):
        ns = self._sample_namespace()
        cfg = RandomTrainingConfig.from_namespace(ns)
        assert cfg.model == "ELM"
        assert cfg.dataset == "cone"
        assert cfg.n_hidden == 64

    def test_with_seed(self):
        cfg = RandomTrainingConfig.from_namespace(self._sample_namespace())
        cfg2 = cfg.with_seed(99)
        assert cfg2.random_state == 99
        assert cfg.random_state == 42  # original untouched (frozen dataclass)

    def test_to_dict(self):
        cfg = RandomTrainingConfig.from_namespace(self._sample_namespace())
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["model"] == "ELM"
        assert "random_state" in d

    def test_frozen(self):
        cfg = RandomTrainingConfig.from_namespace(self._sample_namespace())
        import dataclasses
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            cfg.model = "dRVFL"  # type: ignore[misc]
