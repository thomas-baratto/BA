# Thermal Plume Prediction

Predict thermal plume parameters (isotherm geometry or depression cone size) from hydrogeological inputs using trained neural networks.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/thomas-baratto/BA.git
cd BA

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
  -m, --model      Model type: mlp or random (default: mlp)
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
│   │   ├── compare_models.py
│   │   ├── summarize_results.py
│   │   ├── select_knee_points.py
│   │   ├── pareto_manager.py
│   │   └── csv_to_latex.py
│   ├── sweep/              # Sweep orchestration
│   │   └── launch_sweep_workers.py
│   └── slurm/              # SLURM job scripts
├── data/
│   ├── Clean_Results_Isotherm.csv   # Isotherm training data
│   └── Depression_cones.csv         # Cone training data
├── artifacts/
│   ├── models/              # Full training outputs (model + plots + stats)
│   └── packages/            # Lean deployment packages (model + config only)
└── tests/                   # Unit tests
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

Pre-trained models are stored in `artifacts/`:

| Directory | Contents |
|-----------|----------|
| `artifacts/models/mlp/isotherm/` | Full MLP training output (model, plots, stats, tensorboard) |
| `artifacts/models/mlp/cone/` | Full MLP training output |
| `artifacts/packages/mlp/` | Lean deployment packages (model + config + results only) |
| `runs/run_sweep_random_*/` | Random model sweep results (populated after sweep job) |

## License

[Add license information]
