"""Lightweight CLI smoke tests for script wiring."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler


@contextmanager
def _noop_context(*args, **kwargs):
    yield


def test_train_random_models_main_smoke(monkeypatch, tmp_path):
    import scripts.training.train_random_models as trm

    # Use real scalers that can be pickled (the script writes scalers.pkl)
    _feat_scaler = MinMaxScaler().fit(np.random.randn(12, 4))
    _lbl_scaler = MinMaxScaler().fit(np.random.randn(12, 1))

    args = argparse.Namespace(
        model='ELM',
        dataset='cone',
        targets=None,
        feature_scaler='robust',
        label_scaler='minmax',
        use_log=False,
        use_area_root=False,
        n_hidden=8,
        activation='ReLU',
        alpha=1e-3,
        gamma=1.0,
        n_layers=1,
        n_ensemble=2,
        sc_mode='dense',
        rsc_prob=0.5,
        n_folds=2,
        n_blocks=2,
        direct_link=False,
        base_dir=str(tmp_path),
        output_dir=str(tmp_path / 'smoke_out'),
        random_state=42,
        n_seeds=1,
        no_cuda=True,
        no_save_model=True,
    )

    X_train = np.random.randn(12, 4)
    X_test = np.random.randn(4, 4)
    y_train = np.random.randn(12, 1)
    y_test = np.random.randn(4, 1)

    monkeypatch.setattr(trm, 'parse_args', lambda: args)
    monkeypatch.setattr(
        trm,
        'load_data',
        lambda **kwargs: (X_train, X_test, _feat_scaler, y_train, y_test, _lbl_scaler),
    )

    def _fake_run_single_seed(*_a, **_kw):
        return ({'test': {'aggregate': {'rmse': 1.0, 'mae': 1.0, 'r2': 0.0, 'mape': 0.0, 'mse': 1.0}}}, 0.01)

    monkeypatch.setattr(trm, 'run_single_seed', _fake_run_single_seed)

    trm.main()

    assert (tmp_path / 'smoke_out').exists()


def test_run_optuna_main_smoke(monkeypatch, tmp_path):
    import scripts.training.run_optuna as ro

    csv_file = tmp_path / 'tiny.csv'
    csv_file.write_text(
        'Flow_well,Hydr_gradient,Hydr_conductivity,Aqu_thickness,Cone\n'
        '1,2,3,4,5\n'
        '2,3,4,5,6\n',
        encoding='utf-8',
    )

    args = argparse.Namespace(
        target='Cone',
        csv_file=str(csv_file),
        run_tag='smoke',
        study_name='smoke_study',
        storage_path=str(tmp_path / 'journal_store'),
        optuna_trials=1,
        optuna_workers=1,
        optuna_max_epochs=1,
        optuna_patience=1,
        objective_batch_size=8,
        objective_hidden_layers=1,
        objective_activation='ReLU',
        objective_loss='MSE',
        disable_power_monitor=True,
        power_interval=1.0,
        power_filter='python',
        power_log_dir=None,
    )

    X = np.random.randn(8, 4)
    y = np.random.randn(8, 1)

    class _FakeTrial:
        number = 0
        value = 0.1
        state = None

    class _FakeBestTrial:
        number = 0

    class _FakeStudy:
        best_trial = _FakeBestTrial()
        best_value = 0.1
        best_params = {'x': 1}

        def optimize(self, objective_fn, n_trials, n_jobs, callbacks):
            assert n_trials == 1
            assert n_jobs == 1
            # Ensure objective callable is usable
            _ = objective_fn
            for cb in callbacks:
                cb(self, _FakeTrial())

        def set_user_attr(self, key, value):
            pass

    monkeypatch.setattr(ro, 'parse_args', lambda: args)
    monkeypatch.setattr(ro, 'set_seed', lambda: None)
    monkeypatch.setattr(ro, 'validate_target_labels', lambda labels: None)
    monkeypatch.setattr(ro, 'detect_features_and_labels', lambda _p: (['Flow_well', 'Hydr_gradient', 'Hydr_conductivity', 'Aqu_thickness'], ['Cone']))
    monkeypatch.setattr(ro, 'load_data', lambda **kwargs: (X, X, None, y, y, None))
    monkeypatch.setattr(ro, 'build_objective', lambda **kwargs: (lambda _trial: 0.1))
    monkeypatch.setattr(ro, 'power_monitor_session', _noop_context)
    monkeypatch.setattr(ro.optuna, 'create_study', lambda **kwargs: _FakeStudy())

    ro.main()
