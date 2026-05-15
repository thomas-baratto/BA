"""Contract checks for canonical SLURM scripts."""

from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding='utf-8')


def test_run_optuna_mlp_script_contains_expected_flags():
    content = _read('scripts/slurm/run_optuna_mlp.sbatch')
    assert 'python scripts/training/run_optuna.py' in content
    assert '--storage-path' in content
    assert '--optuna-trials' in content
    assert 'JOURNAL_STORAGE_PATH' in content


def test_sweep_random_script_uses_no_save_and_knee_selection():
    content = _read('scripts/slurm/sweep_random_params.sbatch')
    # Sweep runs --no-save-model for speed; only retrain winners
    assert '--no-save-model' in content
    # Knee-point selection replaces full Pareto frontier retention
    assert 'select_knee_points.py' in content
    assert 'launch_sweep_workers.py' in content


def test_run_random_model_script_wires_entrypoint_and_direct_link():
    content = _read('scripts/slurm/run_random_model.sbatch')
    assert 'python scripts/training/train_random_models.py' in content
    assert 'MODEL=${MODEL:-ELM}' in content
    assert '--model ${MODEL}' in content
    assert 'if [ "$MODEL" = "SResdRVFL" ]; then' in content


def test_train_mlp_metrics_script_wires_entrypoint():
    content = _read('scripts/slurm/train_mlp_metrics.sbatch')
    assert 'python scripts/training/train_mlp_with_metrics.py' in content
    assert 'DATASET' in content


def test_canonical_slurm_scripts_do_not_use_optional_legacy_entrypoints():
    paths = [
        'scripts/slurm/run_optuna_mlp.sbatch',
        'scripts/slurm/sweep_random_params.sbatch',
        'scripts/slurm/run_random_model.sbatch',
        'scripts/slurm/train_mlp_metrics.sbatch',
    ]
    disallowed_refs = [
        'scripts/inference.py',
        'scripts/tune_gamma.py',
        'scripts/RandomNetwork/',
    ]

    for path in paths:
        content = _read(path)
        for ref in disallowed_refs:
            assert ref not in content, f"{path} must not reference {ref}"
