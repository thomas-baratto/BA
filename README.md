# Thermal Plume Prediction — Inference Package

Predict thermal plume parameters (isotherm geometry or depression cone size)
from hydrogeological inputs using pre-trained neural network models.

**No GPU required** — all predictions run on CPU in under a second.

> **Companion code for the bachelor thesis:**
> *Comparison of Neural Network Architectures on the Task of Modeling Heat Plume
> Dimensions in Groundwater* — Thomas Baratto, University of Stuttgart (IPVS), 2026.
>
> Pre-trained model weights are distributed via DaRUS:
> **[DOI: 10.18419/DARUS-5815](https://doi.org/10.18419/DARUS-5815)**

---

## Quick Start

### 1. Get the code

```bash
git clone https://github.com/thomas-baratto/BA.git
cd BA
```

### 2. Download model weights

Download the model archive from [DaRUS (10.18419/DARUS-5815)](https://doi.org/10.18419/DARUS-5815)
and extract the `artifacts/models/` directory into the repository root so the
structure looks like:

```
BA/
├── artifacts/models/
│   ├── mlp/
│   │   ├── cone/          (best_model.pt, scalers.pkl, model_config.json)
│   │   └── isotherm/      (best_model.pt, scalers.pkl, model_config.json)
│   └── random/
│       ├── cone/winner/           (model.pkl, scalers.pkl, model_config.json)
│       └── isotherm/
│           ├── KGE_winner/        (model.pkl, scalers.pkl, model_config.json)
│           └── nRMSE_winner/      (model.pkl, scalers.pkl, model_config.json)
├── core/
├── config/
...
```

> **Tip:** Each model can be downloaded individually from DaRUS.
> Extract it into the repository root so the `artifacts/models/` path matches.

### 3. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For CPU-only PyTorch (smaller download):

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

### 4. Run predictions

```bash
# Cone prediction with the MLP model
ba-predict -i sample_cone.csv -d cone -m mlp

# Isotherm prediction with the MLP model
ba-predict -i sample_isotherm.csv -d isotherm -m mlp

# Use a random-weight network model instead
ba-predict -i sample_cone.csv -d cone -m random
```

---

## Prediction Tasks

### Isotherm (thermal plume geometry)

Predicts the shape of a thermal plume isotherm from 9 hydrogeological inputs.

```bash
ba-predict -i your_data.csv -d isotherm -m mlp
```

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

**Outputs:** `Area` (m²), `Iso_distance` (m), `Iso_width` (m)

### Depression cone

Predicts the size of a hydraulic depression cone from 4 inputs.

```bash
ba-predict -i your_data.csv -d cone -m mlp
```

| Column              | Description              | Unit  |
|---------------------|--------------------------|-------|
| `Flow_well`         | Well flow rate           | m³/s  |
| `Hydr_gradient`     | Hydraulic gradient       | –     |
| `Hydr_conductivity` | Hydraulic conductivity   | m/s   |
| `Aqu_thickness`     | Aquifer thickness        | m     |

**Output:** `Cone` (m)

---

## CLI Reference

```
ba-predict --help

Options:
  -i, --input      Input CSV file (required)
  -d, --dataset    Prediction task: isotherm or cone (required)
  -m, --model      Model type (default: mlp)
                     mlp          — Optuna-tuned MLP neural network
                     random       — Pareto-frontier random-weight network winner
                     random:nRMSE — Best random model by nRMSE (isotherm only)
                     random:KGE   — Best random model by KGE (isotherm only)
  -o, --output     Output directory (default: predictions_<timestamp>)
  --model-dir      Custom model directory path
  --no-report      Skip generating markdown report
```

### Output files

Each run creates a directory containing:

| File              | Description                                       |
|-------------------|---------------------------------------------------|
| `predictions.csv` | Predicted values (one column per output variable)  |
| `report.md`       | Summary with input/output statistics               |

---

## Included Models

| Model | Task | Architecture | KGE | nRMSE | R² |
|-------|------|-------------|-----|-------|-----|
| MLP | Cone | 2-hidden-layer MLP (244 neurons, LeakyReLU) | 0.993 | 0.015 | 0.991 |
| MLP | Isotherm | 5-hidden-layer MLP (256 neurons, GELU) | 0.999 | 0.0002 | ≈1.0 |
| Random | Cone | edRVFL-SC | 0.976 | 0.022 | 0.977 |
| Random | Isotherm (nRMSE) | SResdRVFL | 0.900 | — | 0.900 |
| Random | Isotherm (KGE) | dRVFL | 0.860 | — | 0.860 |

MLP models were optimized with Optuna (200 trials each).
Random-weight network winners were selected from Pareto frontiers
(accuracy vs. training time) across 558 configurations.

---

## Docker

```bash
docker build -t ba-predict .
docker run --rm -v $(pwd):/data ba-predict \
    -i /data/my_input.csv -d cone -m mlp -o /data/results/
```

---

## License & Citation

Part of the bachelor thesis by Thomas Baratto, supervised by M.Sc. Julia Pelzer,
examined by Prof. Dr. Miriam Schulte — University of Stuttgart, IPVS, 2026.

For the full training pipeline, experiments, and analysis code, see the
[`master` branch](https://github.com/thomas-baratto/BA/tree/master).

**Model weights:** [DaRUS 10.18419/DARUS-5815](https://doi.org/10.18419/DARUS-5815)

---

## Package Contents

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

## Troubleshooting

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

