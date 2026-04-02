# Copilot Instructions — ba-thermal-plume

## Web Access

Agents are free to fetch any website or URL they need — no domain restrictions apply.

## Project Overview

Bachelor thesis project for predicting thermal plume parameters (isotherm geometry and depression
cone size) from hydrogeological inputs using neural networks. Built with PyTorch, scikit-learn,
and Optuna.

## Python Environment

🚨 **CRITICAL** — Always use the project virtual environment:
- **Python:** `.venv/bin/python`
- **Pip:** `.venv/bin/pip`
- **PYTHONPATH:** Set `PYTHONPATH=.` when running scripts from the project root.

Never use bare `python` or `pip`. Never `source activate`.

## Key Commands

```bash
# Run tests (fast subset)
.venv/bin/python -m pytest -m "not slow"

# Run all tests
.venv/bin/python -m pytest

# Run a single test file
.venv/bin/python -m pytest tests/test_model.py -v

# Run a script
PYTHONPATH=. .venv/bin/python scripts/deployment/predict.py --help

# Lint
.venv/bin/python -m ruff check .
```

## Architecture

- `core/model.py` — `NeuralNetwork` (MLP), PyTorch `nn.Module`
- `core/trainer.py` — Training loop, early stopping, LR scheduling
- `core/data_loader.py` — CSV loading, train/val/test splits, scaling
- `core/preprocessing.py` — Feature engineering, data transforms
- `core/inference.py` — Load saved model + scalers → predict
- `core/model_wrapper.py` — Unified interface for MLP and random models
- `core/metrics.py` — MAE, MSE, RMSE, R², MAPE, nRMSE, KGE
- `core/utils.py` — Seed setting, logging, metric computation
- `core/plotting.py` — Visualization helpers
- `core/args.py` — Shared argparse definitions
- `core/artifacts.py` — Model artifact path resolution
- `core/config_types.py` — Typed config dataclasses
- `core/random/` — Random-weight networks: ELM, dRVFL, edRVFL, RBF, etc.
- `config/datasets.py` — `DATASET_CONFIGS` — single source of truth for features/labels/paths
- `scripts/` — CLI entry points (run with `PYTHONPATH=.`), organized into subdirectories:
  `training/`, `analysis/`, `deployment/`, `sweep/`, `slurm/`
- `optimization/` — Optuna integration
- `tests/` — pytest test suite

## Datasets

Two prediction tasks, configured in `config/datasets.py`:

- **isotherm**: Features: Flow_well, Temp_diff, kW_well, Hydr_gradient, Hydr_conductivity,
  Aqu_thickness, Long_dispersivity, Trans_dispersivity, Isotherm → Targets: Area, Iso_distance,
  Iso_width (CSV: `data/Clean_Results_Isotherm.csv`)
- **cone**: Features: Flow_well, Hydr_gradient, Hydr_conductivity, Aqu_thickness → Target: Cone
  (CSV: `data/Depression_cones.csv`)

## Code Conventions

- **Python ≥ 3.10**, type hints encouraged.
- **Line length:** 100 (configured in `pyproject.toml` for ruff and black).
- **Formatting:** Do NOT run formatters via terminal — VS Code "Format on Save" handles it.
- **Imports:** Use `from core.xxx import ...` (project uses `PYTHONPATH=.` convention).
- **Config:** Dataset features/labels live in `config/datasets.py` — never hardcode them elsewhere.
- **Metrics:** Use functions from `core/metrics.py` or `core/utils.py` — don't reimplement.
- **Models:** MLP in `core/model.py`, random networks in `core/random/`. Both share the
  `model_wrapper.py` interface.
- **Tests:** pytest with markers `@pytest.mark.fast`, `@pytest.mark.slow`, `@pytest.mark.unit`,
  `@pytest.mark.integration`. Test files in `tests/test_*.py`.

## Important Patterns

- **Inference pipeline:** `core/inference.py` loads `best_model.pt` + `model_config.json` +
  `scalers.pkl` from a model directory, reconstructs the network, and runs predictions.
- **Training artifacts:** Each run produces `best_model.pt`, `model_config.json`, `scalers.pkl`,
  and `results_*.json` in a timestamped directory.
- **Optuna:** Hyperparameters optimized via `scripts/training/run_optuna.py` → stored in journal logs →
  best params extracted to `config/best_params_*.json` → used by `train_mlp_with_metrics.py`.
- **Scaling:** Data is scaled during training (stored in `scalers.pkl`); inference must use the
  same scalers — never fit new scalers at inference time.
