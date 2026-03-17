# Bachelor Thesis Codebase: Heat Plume Prediction

## Models & Capabilities

This project compares optimized Multi-Layer Perceptrons (MLPs) against purely randomized network architectures (like Extreme Learning Machines and randomized variations of RVFL networks) on predicting scalar heat plume characteristics from geometrical and operational inputs.

The two main datasets are:
1. **Isotherm (Study 830)**: Predicts Iso-surface Area, distance, and width.
2. **Depression Cones (Study 832)**: Predicts cone geometric parameters.

---

## Running the Training Pipelines

## Local Quickstart

Use the existing project virtual environment:

```bash
source .venv/env/bin/activate
pip install -r requirements.txt
```

Run the fast local validation path:

```bash
pytest -m "not slow"
```

For full validation:

```bash
pytest
```

## Code Quality Workflow

This repository is configured for Ruff + Black + pre-commit.

```bash
# One-time setup
pip install pre-commit ruff black
pre-commit install

# Run once across repository
pre-commit run --all-files
```

Direct commands are also available:

```bash
ruff check .
ruff format .
black .
```

The SLURM folder is intentionally reduced to four canonical scripts:

1. `scripts/slurm/run_optuna_mlp.sbatch`
2. `scripts/slurm/train_isotherm_journal.sbatch`
3. `scripts/slurm/sweep_random_params.sbatch`
4. `scripts/slurm/run_random_model.sbatch`

### 1. Optuna Hyperparameter Tuning (MLP)
Run distributed Optuna workers with journal storage:

```bash
# Isotherm
sbatch --export=CSV_FILE=./data/Clean_Results_Isotherm.csv,TARGET=all,STUDY_NAME=nn_study_isotherm_journal,TOTAL_TRIALS=10000 scripts/slurm/run_optuna_mlp.sbatch

# Depression cones
sbatch --export=CSV_FILE=./data/Depression_cones.csv,TARGET=Cone,STUDY_NAME=depression_cones_mlp_journal_study,TOTAL_TRIALS=10000 scripts/slurm/run_optuna_mlp.sbatch
```

### 2. Final MLP Training (Best Trial)
Train the final model using the best trial from Optuna journal storage:

```bash
# For Isotherm Dataset
sbatch --export=STUDY_NAME=nn_study_isotherm_journal,JOURNAL_PATH=runs/global_run_830/optuna_journal_storage/journal.log scripts/slurm/train_isotherm_journal.sbatch

# For Depression Cones Dataset
sbatch --export=STUDY_NAME=depression_cones_mlp_journal_study,JOURNAL_PATH=runs/global_run_832/optuna_journal_storage/journal.log,CSV_FILE=data/Depression_cones.csv scripts/slurm/train_isotherm_journal.sbatch
```

### 3. Random Network Parameter Sweep
To execute a massive parallel parameter sweep over 384 configurations of random networks (ELM, dRVFL, edRVFL, edRVFL-SC, esc-edRVFL, SResdRVFL), sweeping over network depth, ensemble size, and activation functions (ReLU, GELU, ELU):

```bash
sbatch scripts/slurm/sweep_random_params.sbatch
```
This script uses multiple nodes/GPUs and aggregates results automatically into a `summary_table.csv`.

### 4. Run One Random Model
Run a single random model configuration with SLURM environment variables:

```bash
# Example: one edRVFL run on isotherm
sbatch --export=MODEL=edRVFL,DATASET=isotherm,ACTIVATION=GELU,N_HIDDEN=1000,N_LAYERS=2,N_ENSEMBLE=20,USE_AREA_ROOT=1 scripts/slurm/run_random_model.sbatch

# Example: one SResdRVFL run on cone
sbatch --export=MODEL=SResdRVFL,DATASET=cone,ACTIVATION=ReLU,N_HIDDEN=1000,N_BLOCKS=5 scripts/slurm/run_random_model.sbatch
```

---

## Packaging Results

Once the MLP and Random Model sweeps are finished, you can generate a consolidated `final_comparison.csv` that picks the best models for each dataset and formats the metrics for the thesis.

```bash
python scripts/compare_models.py \
    --random-summary runs/run_sweep_random_XXXX/summary_table.csv \
    --mlp-isotherm good_runs/final_model_20251209-101803/results.json
```

Generate a LaTeX table from the resulting CSV:
```bash
python scripts/csv_to_latex.py final_comparison.csv --caption "Final Model Comparison"
```
