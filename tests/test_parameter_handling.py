"""Parameter handling tests aligned to journal-based Optuna workflow."""

from __future__ import annotations

from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


HARDCODED_PARAMS = {"batch_size", "nr_hidden_layers", "activation_name", "loss_criterion"}


def _build_temp_study(tmp_path: Path):
    journal_dir = tmp_path / "optuna_journal_storage"
    journal_dir.mkdir(parents=True, exist_ok=True)
    journal_file = journal_dir / "journal.log"

    storage = JournalStorage(JournalFileBackend(str(journal_file)))
    study = optuna.create_study(study_name="test_journal_study", direction="minimize", storage=storage)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        trial.set_user_attr("batch_size", 64)
        trial.set_user_attr("nr_hidden_layers", 3)
        trial.set_user_attr("activation_name", "GELU")
        trial.set_user_attr("loss_criterion", "SmoothL1")
        return lr

    study.optimize(objective, n_trials=1)
    return study, storage


def test_parameter_retrieval_from_journal_storage(tmp_path):
    """Can retrieve optimized params and user attrs from journal-backed study."""
    _, storage = _build_temp_study(tmp_path)
    loaded = optuna.load_study(study_name="test_journal_study", storage=storage)

    assert isinstance(loaded.best_trial.params, dict)
    assert "learning_rate" in loaded.best_trial.params
    assert HARDCODED_PARAMS.issubset(set(loaded.best_trial.user_attrs.keys()))


def test_parameter_merge_includes_hardcoded_attrs(tmp_path):
    """Merging trial params with hardcoded attrs yields complete config dict."""
    _, storage = _build_temp_study(tmp_path)
    loaded = optuna.load_study(study_name="test_journal_study", storage=storage)

    merged = loaded.best_trial.params.copy()
    for key, value in loaded.best_trial.user_attrs.items():
        if key in HARDCODED_PARAMS:
            merged[key] = value

    assert "learning_rate" in merged
    assert HARDCODED_PARAMS.issubset(set(merged.keys()))
