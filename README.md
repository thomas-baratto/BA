# Thermal Plume Prediction — Inference Package

Predict thermal plume parameters (isotherm geometry or depression cone size)
from hydrogeological inputs using pre-trained machine learning models.

**No GPU required** — all predictions run on CPU in under a second.

> **This is the inference-only release package.**
> It contains pre-trained model weights and the prediction CLI.
> For the full source code (training scripts, data, tests), see the
> [GitHub repository](https://github.com/thomas-baratto/BA).

---

## 1. Setup

**Requirements:** Python ≥ 3.10

```bash
# Extract the archive (if downloaded from DaRUS)
tar xzf ba-thermal-plume-v1.0.0.tar.gz
cd ba-thermal-plume-v1.0.0

# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the `ba-predict` command.
For CPU-only PyTorch (smaller download), use:

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

### Docker alternative

```bash
docker build -t ba-predict .
docker run --rm -v $(pwd):/data ba-predict \
    -i /data/my_input.csv -d cone -m mlp -o /data/results/
```

---

## 2. Running Predictions

### Isotherm prediction (thermal plume geometry)

Predicts the shape of a thermal plume isotherm from 9 hydrogeological inputs.

```bash
ba-predict -i your_data.csv -d isotherm -m mlp
```

Required input columns:

| Column                 | Description                  | Unit  |
|------------------------|------------------------------|-------|
| `Flow_well`            | Well flow rate               | m³/s  |
| `Temp_diff`            | Temperature difference       | K     |
| `kW_well`              | Thermal power                | kW    |
| `Hydr_gradient`        | Hydraulic gradient           | –     |
| `Hydr_conductivity`    | Hydraulic conductivity       | m/s   |
| `Aqu_thickness`        | Aquifer thickness            | m     |
| `Long_dispersivity`    | Longitudinal dispersivity    | m     |
| `Trans_dispersivity`   | Transverse dispersivity      | m     |
| `Isotherm`             | Target isotherm value        | K     |

Outputs: `Area` (m²), `Iso_distance` (m), `Iso_width` (m)

### Depression cone prediction

Predicts the size of a hydraulic depression cone from 4 inputs.

```bash
ba-predict -i your_data.csv -d cone -m mlp
```

Required input columns:

| Column              | Description              | Unit  |
|---------------------|--------------------------|-------|
| `Flow_well`         | Well flow rate           | m³/s  |
| `Hydr_gradient`     | Hydraulic gradient       | –     |
| `Hydr_conductivity` | Hydraulic conductivity   | m/s   |
| `Aqu_thickness`     | Aquifer thickness        | m     |

Output: `Cone` (m)

---

## 3. Quick Example

Sample CSV files are included in the package.

```bash
# Cone prediction with the MLP model
ba-predict -i sample_cone.csv -d cone -m mlp -o results_cone/

# Isotherm prediction with the MLP model
ba-predict -i sample_isotherm.csv -d isotherm -m mlp -o results_isotherm/

# Use a random network model instead
ba-predict -i sample_cone.csv -d cone -m random -o results_cone_random/
```

---

## 4. CLI Reference

```
ba-predict --help

Options:
  -i, --input      Input CSV file (required)
  -d, --dataset    Prediction task: isotherm or cone (required)
  -m, --model      Model type (default: mlp)
                     mlp          — Optuna-tuned MLP neural network
                     random       — Pareto-frontier winner (random network)
                     random:nRMSE — Best random model by nRMSE (isotherm only)
                     random:KGE   — Best random model by KGE (isotherm only)
  -o, --output     Output directory (default: predictions_<timestamp>)
  --model-dir      Custom model directory path
  --no-report      Skip generating markdown report
```

### Output files

Each prediction run creates a directory containing:
- `predictions.csv` — predicted values in physical units
- `report.md` — summary report with input/output statistics

---

## 5. Included Models

All models are pre-trained and ready to use:

| Model | Task | Architecture | R² |
|-------|------|-------------|-----|
| `artifacts/models/mlp/cone/` | Cone | Optuna-tuned MLP | 0.99 |
| `artifacts/models/mlp/isotherm/` | Isotherm | Optuna-tuned MLP | ≈1.0 |
| `artifacts/models/random/cone/winner/` | Cone | edRVFL-SC | 0.977 |
| `artifacts/models/random/isotherm/nRMSE_winner/` | Isotherm | SResdRVFL | 0.900 |
| `artifacts/models/random/isotherm/KGE_winner/` | Isotherm | dRVFL | 0.860 |

Random models were selected as knee-point winners from Pareto frontiers
(accuracy vs. training time) across 558 configurations.

---

## 6. Package Contents

```
ba-thermal-plume-v1.0.0/
├── core/                        # Inference code
│   ├── model.py                 #   MLP architecture (PyTorch)
│   ├── model_wrapper.py         #   Unified model interface
│   ├── inference.py             #   Prediction pipeline
│   └── random/                  #   Random network implementations
├── config/
│   └── datasets.py              # Dataset and model path configuration
├── scripts/deployment/
│   └── predict.py               # Prediction CLI entry point
├── artifacts/models/            # Pre-trained model weights and scalers
│   ├── mlp/                     #   MLP models (cone + isotherm)
│   └── random/                  #   Random network models
├── docs/
│   └── INFERENCE_GUIDE.md       # Detailed inference guide
├── sample_cone.csv              # Example input (cone)
├── sample_isotherm.csv          # Example input (isotherm)
├── requirements.txt             # Python dependencies (CPU-only)
├── pyproject.toml               # Package metadata
├── Dockerfile                   # Container build file
└── README.md                    # This file
```

---

## 7. Troubleshooting

**scikit-learn version warning:** If you see `InconsistentVersionWarning` when
loading scalers, this is harmless — predictions remain correct.

**Missing `torch`:** Install CPU-only PyTorch to avoid a large download:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**PYTHONPATH alternative** (no `pip install`):
```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/deployment/predict.py -i data.csv -d cone -m mlp
```

---


## License

[Add license information]
