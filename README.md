# Thermal Plume Prediction — Neural Network Models and Inference Package

[![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--5815-d45815.svg)](https://doi.org/10.18419/darus-5815)

Predict thermal plume parameters (isotherm geometry or depression cone size)
from hydrogeological inputs using pre-trained neural network models.

**No GPU required** — all predictions run on CPU in under a second.

> **Companion dataset for the bachelor thesis:**
> *Comparison of Neural Network Architectures on the Task of Modeling Heat Plume
> Dimensions in Groundwater* — Thomas Baratto, University of Stuttgart (IPVS), 2026.
>
> **DOI: [10.18419/DARUS-5815](https://doi.org/10.18419/DARUS-5815)**

---

## Dataset Contents

This dataset contains pre-trained neural network models, the inference source
code, and the training data used in the thesis. All files are available for
individual download.

### File Inventory

| File | Category | Size | Description |
|------|----------|------|-------------|
| `README.md` | — | 2.8 KB | This file |
| `ba-thermal-plume-v1.0.0-code.zip` | `code/` | 44.9 KB | Inference source code (Python package with CLI) |
| `Clean_Results_Isotherm.tab` | `data/` | 7.7 MB | Isotherm training dataset (85 531 observations, 13 variables) |
| `Depression_cones.tab` | `data/` | 436.7 KB | Depression cone training dataset (12 835 observations, 5 variables) |
| `mlp-cone.zip` | `models/` | 562.0 KB | Pre-trained MLP for depression cone prediction |
| `mlp-isotherm.zip` | `models/` | 4.0 MB | Pre-trained MLP for isotherm geometry prediction |
| `random-cone-edRVFL-SC.zip` | `models/` | 221.3 MB | Pre-trained edRVFL-SC for depression cone prediction |
| `random-isotherm-dRVFL-KGE.zip` | `models/` | 1.3 MB | Pre-trained dRVFL for isotherm prediction (best KGE) |
| `random-isotherm-SResdRVFL-nRMSE.zip` | `models/` | 2.3 MB | Pre-trained SResdRVFL for isotherm prediction (best nRMSE) |

---

## Quick Start

### 1. Download and extract the code

Download `ba-thermal-plume-v1.0.0-code.zip` and extract it:

```bash
unzip ba-thermal-plume-v1.0.0-code.zip
cd ba-thermal-plume-v1.0.0
```

### 2. Download and place model weights

Download one or more model zip files and extract them into the code directory.
Each model zip extracts into the correct `artifacts/models/` subdirectory:

```bash
# Example: download mlp-cone.zip and extract it
unzip mlp-cone.zip -d .
```

After extracting, the directory should look like:

```
ba-thermal-plume-v1.0.0/
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

## Models

Five pre-trained models are provided, covering two prediction tasks and two
architecture families (MLP and random-weight networks).

| Archive | Task | Architecture | KGE | nRMSE | R² |
|---------|------|-------------|-----|-------|-----|
| `mlp-cone.zip` | Cone | 2-hidden-layer MLP (244 neurons, LeakyReLU) | 0.993 | 0.015 | 0.991 |
| `mlp-isotherm.zip` | Isotherm | 5-hidden-layer MLP (256 neurons, GELU) | 0.999 | 0.0002 | ≈1.0 |
| `random-cone-edRVFL-SC.zip` | Cone | edRVFL-SC (1000 neurons, 3 layers, GELU, 10-member ensemble) | 0.976 | 0.022 | 0.977 |
| `random-isotherm-dRVFL-KGE.zip` | Isotherm | dRVFL (1500 neurons, 1 layer, ELU) | 0.860 | — | 0.860 |
| `random-isotherm-SResdRVFL-nRMSE.zip` | Isotherm | SResdRVFL (1500 neurons, 1 layer, GELU, 8 residual blocks) | 0.900 | — | 0.900 |

MLP models were optimized with Optuna (200 trials each).
Random-weight network winners were selected from Pareto frontiers
(accuracy vs. training time) across 558 configurations.

Each model archive contains the serialized model weights, input/output scalers,
training diagnostics plots, power consumption logs, and evaluation metrics.

---

## Training Data

The two `.tab` files contain the datasets used to train all models.

- **`Clean_Results_Isotherm.tab`** — 85 531 numerical simulation results for
  isotherm geometry (9 input features → 3 output labels: Area, Iso_distance,
  Iso_width).
- **`Depression_cones.tab`** — 12 835 numerical simulation results for
  depression cone size (4 input features → 1 output label: Cone).

Both datasets were split 56 % train / 14 % validation / 30 % test for model
development.

---

## Docker

```bash
docker build -t ba-predict .
docker run --rm -v $(pwd):/data ba-predict \
    -i /data/my_input.csv -d cone -m mlp -o /data/results/
```

---

## Code Package Contents

The `ba-thermal-plume-v1.0.0-code.zip` archive contains:

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
├── docs/
│   └── INFERENCE_GUIDE.md       # Detailed inference guide
├── sample_cone.csv              # Example input (cone)
├── sample_isotherm.csv          # Example input (isotherm)
├── requirements.txt             # Python dependencies (CPU-only)
├── pyproject.toml               # Package metadata
├── Dockerfile                   # Container build file
└── README.md                    # This file
```

No model weights are included in the code archive — download the model zip
files separately and extract them into the code directory.

---

## License & Citation

This project is released under the [MIT License](LICENSE).

Part of the bachelor thesis by Thomas Baratto, supervised by M.Sc. Julia Pelzer,
examined by Prof. Dr. Miriam Schulte — University of Stuttgart, IPVS, 2026.

If you use this software, please cite:

```bibtex
@misc{baratto2026thermal,
  author    = {Baratto, Thomas},
  title     = {Thermal Plume Prediction --- Neural Network Models and Inference Package},
  year      = {2026},
  version   = {1.0.0},
  doi       = {10.18419/DARUS-5815},
  url       = {https://doi.org/10.18419/DARUS-5815}
}
```

For the full training pipeline, experiments, and analysis code, see the
companion GitHub repository.

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

