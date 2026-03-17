# Bachelor Thesis Codebase: Heat Plume Prediction

This repository compares optimized MLPs against randomized network families for two geothermal targets:

1. Isotherm dataset: `Area`, `Iso_distance`, `Iso_width`
2. Depression cones dataset: `Cone`

Core logic is in `core/`, `optimization/`, and `monitoring/`. CLI entry points are in `scripts/`.

## Current Script Map

Primary training and evaluation entry points:

1. `scripts/run_optuna.py`: distributed Optuna tuning for MLP (journal storage)
2. `scripts/train_final_model.py`: train final MLP from best Optuna trial
3. `scripts/train_random_models.py`: train random-model families (ELM, dRVFL, edRVFL, edRVFL-SC, esc-edRVFL, SResdRVFL)
4. `scripts/summarize_results.py`: aggregate random run folders into `summary_table.csv`

Reporting and packaging utilities:

1. `scripts/compare_models.py`: compare best MLP vs best random models per dataset
2. `scripts/csv_to_latex.py`: export CSV table to LaTeX
3. `scripts/package_models.py`: package finalized artifacts into a consistent deployable layout

Canonical SLURM scripts:

1. `scripts/slurm/run_optuna_mlp.sbatch`
2. `scripts/slurm/train_isotherm_journal.sbatch`
3. `scripts/slurm/sweep_random_params.sbatch`
4. `scripts/slurm/run_random_model.sbatch`

## Local Quickstart

```bash
source .venv/env/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
pytest -m "not slow"
# or
pytest
```

Quality tooling:

```bash
pip install pre-commit ruff black
pre-commit install
pre-commit run --all-files
```

## Metrics

Regression metrics are computed in `core/utils.py` and include:

1. MAE, MSE, RMSE, R2, MAPE
2. RMSLE
3. nRMSE
4. KGE (Kling-Gupta Efficiency)
5. residual distribution statistics

These metrics are used in random-model summaries and final MLP result files produced by current training code.

## End-to-End Workflow

### 1) Tune MLP with Optuna (journal storage)

```bash
# Isotherm
sbatch --export=CSV_FILE=data/Clean_Results_Isotherm.csv,TARGET=all,STUDY_NAME=nn_study_isotherm_journal,TOTAL_TRIALS=10000 scripts/slurm/run_optuna_mlp.sbatch

# Cone
sbatch --export=CSV_FILE=data/Depression_cones.csv,TARGET=Cone,STUDY_NAME=depression_cones_mlp_journal_study,TOTAL_TRIALS=10000 scripts/slurm/run_optuna_mlp.sbatch
```

### 2) Train final MLP from best trial

```bash
# Isotherm
sbatch --export=STUDY_NAME=nn_study_isotherm_journal,JOURNAL_PATH=data/good_runs/global_run_830/optuna_journal_storage/journal.log,CSV_FILE=data/Clean_Results_Isotherm.csv scripts/slurm/train_isotherm_journal.sbatch

# Cone
sbatch --export=STUDY_NAME=depression_cones_mlp_journal_study,JOURNAL_PATH=data/good_runs/global_run_832/optuna_journal_storage/journal.log,CSV_FILE=data/Depression_cones.csv scripts/slurm/train_isotherm_journal.sbatch
```

Final MLP artifacts are saved under `runs/final_model_<timestamp>/` and include:

1. `best_model.pt`
2. `model_config.json`
3. `results.json`

### 3) Random-model sweep

```bash
sbatch scripts/slurm/sweep_random_params.sbatch
```

Current sweep behavior:

1. Runs many random-model configs in parallel
2. Generates `summary_table.csv`
3. By default deletes per-run subdirectories to save disk

To keep all per-run artifacts:

```bash
sbatch --export=KEEP_RUN_ARTIFACTS=1 scripts/slurm/sweep_random_params.sbatch
```

### 4) Promote one random model configuration (with model saved)

Sweeps typically use `--no-save-model`. To package random models, rerun selected winners with save enabled:

```bash
sbatch --export=MODEL=edRVFL,DATASET=isotherm,ACTIVATION=GELU,N_HIDDEN=1000,N_LAYERS=2,N_ENSEMBLE=20,NO_SAVE_MODEL=0,USE_AREA_ROOT=1 scripts/slurm/run_random_model.sbatch
```

## Comparison Table

Build a final comparison CSV with best MLP and random entries per dataset:

```bash
python scripts/compare_models.py \
    --random-summary runs/run_sweep_random_XXXX/summary_table.csv \
    --mlp-isotherm runs/final_model_ISOTHERM_TIMESTAMP/results.json \
    --mlp-cone runs/final_model_CONE_TIMESTAMP/results.json \
    --output-csv final_comparison.csv
```

Export to LaTeX:

```bash
python scripts/csv_to_latex.py final_comparison.csv --caption "Final Model Comparison"
```

## Packaging Artifacts

Package finalized model artifacts into a consistent structure under `data/good_runs/packages`:

```bash
python scripts/package_models.py \
    --mlp-isotherm-dir runs/final_model_ISOTHERM_TIMESTAMP \
    --mlp-cone-dir runs/final_model_CONE_TIMESTAMP \
    --random-summary runs/run_sweep_random_XXXX/summary_table.csv \
    --random-run-dir runs/run_sweep_random_XXXX \
    --output-root data/good_runs/packages
```

Notes:

1. Random packaging requires `model.pkl` in selected run folders.
2. If missing (e.g. sweep ran with `--no-save-model` or post-sweep cleanup), rerun selected winners once with `NO_SAVE_MODEL=0`.
