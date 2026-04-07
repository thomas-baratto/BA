# Copilot Instructions — ba-thermal-plume

## Web Access

Agents are free to fetch any website or URL they need — no domain restrictions apply.

## Project Overview

Bachelor thesis project for predicting thermal plume parameters (isotherm geometry and depression
cone size) from hydrogeological inputs using neural networks. Built with PyTorch, scikit-learn,
and Optuna.

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

- `core/model.py` — `NeuralNetwork` (MLP), PyTorch `nn.Module`
- `core/trainer.py` — Training loop, early stopping, LR scheduling
- `core/training_utils.py` — Training helper functions
- `core/data_loader.py` — CSV loading, train/val/test splits, scaling
- `core/preprocessing.py` — Feature engineering, data transforms
- `core/inference.py` — Load saved model + scalers → predict
- `core/model_wrapper.py` — Unified interface for MLP and random models
- `core/metrics.py` — MAE, MSE, RMSE, R², MAPE, nRMSE, KGE
- `core/utils.py` — Seed setting, logging, metric computation
- `core/runtime.py` — Runtime/timing utilities
- `core/plotting.py` — Visualization helpers
- `core/thesis_style.py` — Thesis-quality plot styling (SciencePlots, colours, `save_fig()`)
- `core/args.py` — Shared argparse definitions
- `core/artifacts.py` — Model artifact path resolution
- `core/config_types.py` — Typed config dataclasses
- `core/random/` — Random-weight networks: ELM, dRVFL, edRVFL, RBF, etc.
- `config/datasets.py` — `DATASET_CONFIGS` — single source of truth for features/labels/paths
- `scripts/` — CLI entry points (run with `PYTHONPATH=.`), organized into subdirectories:
  `training/`, `analysis/`, `deployment/`, `sweep/`, `slurm/`
  Key scripts: `training/train_random_models.py` (`--n-seeds N` for multi-seed sweeps),
  `analysis/analyze_seed_sweep.py` (LaTeX table + box plots from sweep results),
  `slurm/seed_sweep.sbatch` (100 seeds × 3 RaNN winners on argon-gtx)
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

## Thesis Data Pipeline

When BA produces new results (training runs, sweeps, benchmarks), the thesis workspace must be
updated so the writer and reviewer agents have accurate data:

1. **Run the experiment** — output lands in `artifacts/models/` or `runs/`.
2. **Re-extract server data** — run the `/server-extract-ba-data` prompt (in the thesis workspace)
   on the compute server. This reads gitignored dirs and writes everything to
   `thesis/.github/skills/ba-data/references/server-data.md`.
3. **Commit `server-data.md`** — this file is tracked in git and is the single source of truth
   for all numeric claims in the thesis.

The thesis agents (`@thesis-writer`, `@thesis-reviewer`) load the `ba-data` skill, which reads
`server-data.md` to fact-check metrics. If you change artifacts without updating `server-data.md`,
the thesis will contain stale numbers.

## Skills

Load these skills when their domain applies:
- **`/thesis-results`** — when implementing scripts, models, or analysis in BA to produce results
  needed by the thesis (missing plots, baseline comparisons, inference benchmarking, SLURM jobs).
  Contains execution constraints, gap tracker, and analysis script reference.
- **`/thesis-plots`** — when creating, editing, or reviewing matplotlib plots. Enforces the thesis
  style guide (SciencePlots base, colour-blind palette, PDF output, consistent figure dimensions).
  Covers colours, sizes, `save_fig()`, axis labels, and output paths.
