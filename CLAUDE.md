# CLAUDE.md — Project Instructions for AI Assistants

## Project Overview

**ba-thermal-plume** — Bachelor thesis project for predicting thermal plume parameters
(isotherm geometry and depression cone size) from hydrogeological inputs using neural networks.
Built with PyTorch, scikit-learn, and Optuna.

## Python Environment

🚨 **CRITICAL** — Always use the project virtual environment:
- **Python:** `.venv/env/bin/python`
- **Pip:** `.venv/env/bin/pip`
- **PYTHONPATH:** Set `PYTHONPATH=.` when running scripts from the project root.

Never use bare `python` or `pip`. Never `source activate`.

## Key Commands

```bash
# Run tests (fast subset)
.venv/env/bin/python -m pytest -m "not slow"

# Run all tests
.venv/env/bin/python -m pytest

# Run a single test file
.venv/env/bin/python -m pytest tests/test_model.py -v

# Run a script
PYTHONPATH=. .venv/env/bin/python scripts/deployment/predict.py --help

# Lint
.venv/env/bin/python -m ruff check .
```

## Architecture

```
BA/
├── core/                    # Core ML library (importable modules)
│   ├── model.py             # NeuralNetwork (MLP) — PyTorch nn.Module
│   ├── trainer.py           # Training loop, early stopping, LR scheduling
│   ├── data_loader.py       # CSV loading, train/val/test splits, scaling
│   ├── preprocessing.py     # Feature engineering, data transforms
│   ├── inference.py         # Load saved model + scalers → predict
│   ├── model_wrapper.py     # Unified interface for MLP and random models
│   ├── metrics.py           # MAE, MSE, RMSE, R², MAPE, nRMSE, KGE
│   ├── utils.py             # Seed setting, logging, metric computation
│   ├── plotting.py          # Visualization helpers
│   ├── args.py              # Shared argparse definitions
│   ├── artifacts.py         # Model artifact path resolution
│   ├── config_types.py      # Typed config dataclasses
│   ├── runtime.py           # Runtime/resource tracking
│   ├── training_utils.py    # Training helper functions
│   └── random/              # Random-weight network implementations
│       ├── ELM.py           # Extreme Learning Machine
│       ├── dRVFL.py         # Direct Random Vector Functional Link
│       ├── edRVFL.py        # Ensemble deep RVFL
│       ├── edRVFL_SC.py     # edRVFL with stochastic config
│       ├── esc_edRVFL.py    # ESC variant
│       ├── SResdRVFL.py     # Stacked residual dRVFL
│       ├── RBF.py           # Radial Basis Function network
│       └── utils.py         # Shared random-network utilities
├── config/
│   ├── datasets.py          # DATASET_CONFIGS — single source of truth for features/labels/paths
│   ├── best_params_isotherm.json
│   └── best_params_cone.json
├── scripts/                 # CLI entry points (run with PYTHONPATH=.)
│   ├── training/            # Model training scripts
│   │   ├── train_mlp_with_metrics.py
│   │   ├── train_random_models.py
│   │   └── run_optuna.py        # Hyperparameter optimization
│   ├── analysis/            # Evaluation & comparison
│   │   ├── compare_models.py    # MLP vs random model comparison
│   │   ├── summarize_results.py
│   │   ├── select_knee_points.py
│   │   ├── pareto_manager.py
│   │   └── csv_to_latex.py
│   ├── deployment/          # Prediction & packaging
│   │   ├── predict.py           # Main prediction CLI
│   │   ├── retrain_random_models.py  # Rebuild random model.pkl from stored configs
│   │   ├── package_models.py
│   │   └── extract_best_params.py
│   ├── sweep/               # Sweep orchestration
│   │   └── launch_sweep_workers.py
│   └── slurm/               # SLURM batch scripts for HPC cluster
├── optimization/            # Optuna integration
│   ├── optuna_config.py
│   └── optuna_objective.py
├── monitoring/              # Power/resource monitoring
├── data/                    # CSV datasets
│   ├── Clean_Results_Isotherm.csv
│   └── Depression_cones.csv
├── artifacts/               # Trained model artifacts
│   ├── models/              # Full training outputs (model + plots + stats)
│   └── packages/            # Lean deployment packages
├── tests/                   # pytest test suite
└── runs/                    # Sweep and training run outputs
```

## Datasets

Two prediction tasks, configured in `config/datasets.py`:

| Dataset | Features | Targets | CSV |
|---------|----------|---------|-----|
| **isotherm** | Flow_well, Temp_diff, kW_well, Hydr_gradient, Hydr_conductivity, Aqu_thickness, Long_dispersivity, Trans_dispersivity, Isotherm | Area, Iso_distance, Iso_width | `data/Clean_Results_Isotherm.csv` |
| **cone** | Flow_well, Hydr_gradient, Hydr_conductivity, Aqu_thickness | Cone | `data/Depression_cones.csv` |

## Code Conventions

- **Python ≥ 3.10**, type hints encouraged.
- **Line length:** 100 (configured in `pyproject.toml` for both ruff and black).
- **Formatting:** Do NOT run formatters via terminal — VS Code "Format on Save" handles it.
- **Imports:** Use `from core.xxx import ...` (project uses `PYTHONPATH=.` convention).
- **Config:** Dataset features/labels live in `config/datasets.py` — never hardcode them elsewhere.
- **Metrics:** Use functions from `core/metrics.py` or `core/utils.py` — don't reimplement.
- **Models:** MLP in `core/model.py`, random networks in `core/random/`. Both share the
  `model_wrapper.py` interface.
- **Tests:** pytest with markers `@pytest.mark.fast`, `@pytest.mark.slow`, `@pytest.mark.unit`,
  `@pytest.mark.integration`. Test files in `tests/test_*.py`.

## Important Patterns

- **Inference pipeline:** `core/inference.py` loads `best_model.pt` + `model_config.json` + `scalers.pkl`
  from a model directory, reconstructs the network, and runs predictions.
- **Training artifacts:** Each training run produces `best_model.pt`, `model_config.json`,
  `scalers.pkl`, and `results_*.json` in a timestamped directory.
- **Optuna:** Hyperparameters are optimized via `scripts/training/run_optuna.py` → stored in journal logs →
  best params extracted to `config/best_params_*.json` → used by `train_mlp_with_metrics.py`.
- **Scaling:** Data is scaled during training (stored in `scalers.pkl`); inference must use the
  same scalers — never fit new scalers at inference time.

## VS Code Integration

- **File Review:** After creating/modifying files, open them with `code <filename>` so the user can review.
- **Formatting:** Handled by VS Code "Format on Save" — never run `ruff format` or `black` in terminal.
- **Navigation:** Use `grep` or `rg` (ripgrep) for searching across the project.
