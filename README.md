# Thermal Plume Prediction

Predict thermal plume parameters (isotherm geometry or depression cone size) from hydrogeological inputs using trained neural networks.

## Changelog

### 2026-04-11 — Plot uniformity & resource logging refactor

**`core/plotting.py`**
- All regression/residual scatter plots now use `FIG_SQUARE` (5.5 × 5.5) instead of `FIG_SINGLE` for consistent aspect ratio
- Loss curve label shortened from `"Training Loss"` / `"Validation Loss"` to `"Training"` / `"Validation"`
- Loss plots now use `FIG_SQUARE`
- `ResourceLogger.generate_plots()` merges the old separate CPU and RAM plots into a single combined `cpu_ram_usage.pdf` subplot (CPU on top, RAM below); GPU memory % and allocated MiB similarly merged into one `gpu_usage.pdf` subplot
- Removed the `training_data.npz` data-persistence step that was previously saved alongside every plot run

**`core/thesis_style.py`**
- `label_with_unit()` now renders units in LaTeX math mode (`$\mathrm{…}$`) and converts Unicode superscripts (`²` → `^2`, `³` → `^3`) for proper rendering in saved PDFs

**`scripts/analysis/analyze_seed_sweep.py`**
- Box plots: all targets for a given winner are now combined into a single multi-subplot PDF (`seed_sweep_{dataset}_{model}.pdf`) instead of one file per target
- Seed distribution plots: all metrics × targets for a winner are combined in a grid subplot PDF (`seed_vs_metric_{dataset}_{model}.pdf`) instead of one file per metric × target

**`scripts/analysis/generate_model_comparison.py`**
- Residual scatter plots switched from `FIG_SINGLE` to `FIG_SQUARE`

**`scripts/analysis/plot_initial_overfitting.py`**
- ELM regularisation parameter corrected from `alpha=0` to `alpha=1e-6` (prevents degenerate solutions)
- Predicted-vs-true plots for multi-output models now rendered as a single multi-subplot PDF instead of one file per label

**`scripts/analysis/plot_mlp_resources.py`**, **`plot_optuna_study.py`**, **`plot_pareto_frontiers.py`**, **`plot_power_analysis.py`**
- Switched figure size constants from `FIG_SINGLE` to `FIG_SQUARE` / `FIG_WIDE` as appropriate for uniform thesis output

## Quick Start

### Installation

```bash
# Clone the repository (default branch is 'release' — inference only)
git clone https://github.com/thomas-baratto/BA.git
cd BA

# For the full training pipeline, switch to master
git checkout master

# Create and activate virtual environment
python3 -m venv .venv/env
source .venv/env/bin/activate

# Install as package (recommended)
pip install -e .
```

This makes the `ba-predict` command available. See [docs/INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md) for the full step-by-step guide including Docker.

### Using the Prediction CLI

The prediction tool takes a CSV file with hydrogeological parameters and outputs predicted thermal plume values.

**Isotherm prediction:**

```bash
ba-predict -i your_data.csv -d isotherm -m mlp
```

Required input columns for isotherm:
- `Flow_well` - Well flow rate (m³/s)
- `Temp_diff` - Temperature difference (K)
- `kW_well` - Thermal power (kW)
- `Hydr_gradient` - Hydraulic gradient (-)
- `Hydr_conductivity` - Hydraulic conductivity (m/s)
- `Aqu_thickness` - Aquifer thickness (m)
- `Long_dispersivity` - Longitudinal dispersivity (m)
- `Trans_dispersivity` - Transverse dispersivity (m)
- `Isotherm` - Target isotherm value (K)

Outputs: `Area` (m²), `Iso_distance` (m), `Iso_width` (m)

**Depression cone prediction:**

```bash
ba-predict -i your_data.csv -d cone -m mlp
```

Required input columns for cone:
- `Flow_well` - Well flow rate (m³/s)
- `Hydr_gradient` - Hydraulic gradient (-)
- `Hydr_conductivity` - Hydraulic conductivity (m/s)
- `Aqu_thickness` - Aquifer thickness (m)

Outputs: `Cone` (m)

### CLI Options

```bash
ba-predict --help

Options:
  -i, --input      Input CSV file (required)
  -d, --dataset    Dataset type: isotherm or cone (required)
  -m, --model      Model type: mlp, random, random:nRMSE, or random:KGE
                   (default: mlp). 'random' uses the nRMSE Pareto winner.
  -o, --output     Output directory (default: predictions_<timestamp>)
  --model-dir      Custom model directory path
  --no-report      Skip generating markdown report
```

### Output Files

The prediction tool creates an output directory containing:
- `predictions.csv` - Predicted values
- `report.md` - Markdown report with input/output statistics

### Example

```bash
# Create a sample input file
echo "Flow_well,Hydr_gradient,Hydr_conductivity,Aqu_thickness
0.001,0.001,0.0001,50
0.002,0.002,0.0002,100" > sample_cone.csv

# Run prediction
ba-predict -i sample_cone.csv -d cone -m mlp -o my_predictions/

# View results
cat my_predictions/predictions.csv
cat my_predictions/report.md
```

---

## Training Models (Advanced)

This section is for training new models or retraining with different data.

### Project Structure

```
BA/
├── core/                    # Core ML components
│   ├── model.py            # MLP architecture
│   ├── model_wrapper.py    # Unified model interface (MLP + random)
│   ├── inference.py        # Inference pipeline
│   ├── trainer.py          # Training loop
│   ├── data_loader.py      # Data loading and scaling
│   └── random/             # Random network implementations (ELM, RVFL, etc.)
├── scripts/                 # CLI entry points
│   ├── deployment/         # Prediction & packaging
│   │   └── predict.py      # Prediction CLI
│   ├── training/           # Model training scripts
│   │   ├── train_mlp_with_metrics.py    # Train MLP from Optuna results
│   │   ├── train_random_models.py       # Train random networks
│   │   └── run_optuna.py               # Hyperparameter optimization
│   ├── analysis/           # Comparison & evaluation
│   │   ├── generate_model_comparison.py # Full comparison report
│   │   ├── plot_pareto_frontiers.py     # Pareto frontier plots
│   │   ├── select_knee_points.py        # Knee-point model selection
│   │   ├── pareto_manager.py
│   │   └── csv_to_latex.py
│   ├── sweep/              # Sweep orchestration
│   │   └── launch_sweep_workers.py
│   └── slurm/              # SLURM job scripts
├── data/
│   ├── Clean_Results_Isotherm.csv   # Isotherm training data
│   └── Depression_cones.csv         # Cone training data
├── artifacts/
│   └── models/              # Pre-trained model artifacts
│       ├── mlp/             # MLP models (cone, isotherm)
│       └── random/          # Random models (4 Pareto winners)
├── docs/                    # Documentation and plots
│   ├── INFERENCE_GUIDE.md   # Step-by-step inference guide
│   └── MODEL_COMPARISON.md  # Model comparison report with plots
└── tests/                   # Unit tests (172 tests)
```

### Training MLP Models

Train MLP models using best hyperparameters from Optuna optimization:

```bash
# Train both isotherm and cone MLPs
sbatch scripts/slurm/train_mlp_metrics.sbatch

# Train only one dataset
sbatch --export=DATASET=isotherm scripts/slurm/train_mlp_metrics.sbatch
sbatch --export=DATASET=cone scripts/slurm/train_mlp_metrics.sbatch
```

Or run locally:

```bash
source .venv/env/bin/activate
PYTHONPATH=. python scripts/training/train_mlp_with_metrics.py --dataset all --output-dir artifacts/models/mlp
```

Output artifacts per model:
- `best_model.pt` - Model weights
- `model_config.json` - Architecture and feature configuration
- `scalers.pkl` - Fitted data scalers for inference
- `results_MLP_*.json` - Training metrics

### Training Random Network Models

Train random network families (ELM, dRVFL, edRVFL, etc.):

```bash
# Single model
sbatch scripts/slurm/run_random_model.sbatch

# Parameter sweep
sbatch scripts/slurm/sweep_random_params.sbatch
```

Or run locally:

```bash
PYTHONPATH=. python scripts/training/train_random_models.py \
    --model ELM \
    --dataset isotherm \
    --n-hidden 100 \
    --activation ReLU
```

### Hyperparameter Optimization (Optuna)

Run distributed hyperparameter search:

```bash
# Isotherm
sbatch --export=CSV_FILE=data/Clean_Results_Isotherm.csv,TARGET=all,STUDY_NAME=nn_study_isotherm_journal,TOTAL_TRIALS=10000 scripts/slurm/run_optuna_mlp.sbatch

# Cone
sbatch --export=CSV_FILE=data/Depression_cones.csv,TARGET=Cone,STUDY_NAME=depression_cones_mlp_journal_study,TOTAL_TRIALS=10000 scripts/slurm/run_optuna_mlp.sbatch
```

### Model Comparison

Compare MLP vs random models:

```bash
python scripts/analysis/compare_models.py \
    --random-summary runs/run_sweep_random_<jobid>/summary_table.csv \
    --mlp-isotherm artifacts/models/mlp/isotherm/results_MLP_isotherm.json \
    --mlp-cone artifacts/models/mlp/cone/results_MLP_cone.json \
    --output-csv comparison.csv
```

Export to LaTeX:

```bash
python scripts/analysis/csv_to_latex.py comparison.csv --caption "Model Comparison"
```

### Metrics

All models are evaluated using these regression metrics (computed in `core/utils.py`):
- MAE, MSE, RMSE, R²
- MAPE
- nRMSE (normalized RMSE)
- KGE (Kling-Gupta Efficiency)

### Testing

```bash
# Run fast tests
pytest -m "not slow"

# Run all tests
pytest

# With coverage
pytest --cov=core --cov-report=html
```

### Code Quality

```bash
pip install pre-commit ruff black
pre-commit install
pre-commit run --all-files
```

## Pre-trained Models

Pre-trained models are stored in `artifacts/models/`:

| Directory | Model | Description |
|-----------|-------|-------------|
| `artifacts/models/mlp/cone/` | Optimized MLP | Optuna-tuned MLP for cone prediction (R²=0.99) |
| `artifacts/models/mlp/isotherm/` | Optimized MLP | Optuna-tuned MLP for isotherm prediction (R²≈1.0) |
| `artifacts/models/random/cone/winner/` | edRVFL-SC | Pareto-frontier winner for cone (R²=0.977) |
| `artifacts/models/random/isotherm/nRMSE_winner/` | SResdRVFL | Best random model for isotherm by nRMSE (R²=0.900) |
| `artifacts/models/random/isotherm/KGE_winner/` | dRVFL | Best random model for isotherm by KGE (R²=0.860) |

Random models were selected as knee-point winners from Pareto frontiers
(accuracy vs. training time) across a sweep of 558 configurations (job 1048).
See [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md) for detailed plots and metrics.

> **Note:** Random model binaries (`model.pkl`) are excluded from git due to
> size (up to 223 MB each). After cloning, retrain them in ~15 seconds:
>
> ```bash
> PYTHONPATH=. .venv/env/bin/python scripts/deployment/retrain_random_models.py
> # or, if installed via pip:
> ba-retrain-random
> ```

## Data & Models

All trained model weights, scaler objects, and evaluation results are archived on DaRUS.
Each model can be downloaded individually, or use "Download All" for the complete set.

> Baratto, Thomas (2026). *Trained Neural Networks on Simulated Data of Groundwater Heat Plume Characteristics*. DaRUS. <https://doi.org/10.18419/DARUS-5815>

| DaRUS file | Contents |
|------------|----------|
| `mlp-cone.zip` | MLP cone model (best_model.pt, scalers.pkl, plots, power logs) |
| `mlp-isotherm.zip` | MLP isotherm model (best_model.pt, scalers.pkl, plots, power logs) |
| `random-cone-edRVFL-SC.zip` | edRVFL-SC cone winner (model.pkl, scalers.pkl) |
| `random-isotherm-dRVFL-KGE.zip` | dRVFL isotherm KGE winner (model.pkl, scalers.pkl) |
| `random-isotherm-SResdRVFL-nRMSE.zip` | SResdRVFL isotherm nRMSE winner (model.pkl, scalers.pkl) |
| `ba-thermal-plume-v1.0.0-code.zip` | Release branch code (no model weights) |

For the inference-only package (no training code), see the
[`release` branch](https://github.com/thomas-baratto/BA) (default).
