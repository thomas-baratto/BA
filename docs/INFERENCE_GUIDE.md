# Inference Guide — Thermal Plume Prediction

This guide walks you through running predictions with the pre-trained neural
network models. No GPU required — inference runs on CPU in under a second.

> Pre-trained model weights are distributed via DaRUS:
> **[DOI: 10.18419/DARUS-5815](https://doi.org/10.18419/DARUS-5815)**

Three installation methods are available — pick whichever suits you:

| Method | Best for | Setup time |
|--------|----------|------------|
| [pip install](#option-a--pip-install-recommended) | Most users | ~2 min |
| [PYTHONPATH](#option-b--pythonpath-no-install) | Quick testing | ~2 min |
| [Docker](#option-c--docker-zero-setup) | Reproducibility, no Python needed | ~5 min |

---

## 1. Prerequisites

- **Python ≥ 3.10** (not needed for Docker)
- **Git**
- **Docker** (only for the Docker method)

---

## 2. Get Models & Install

### Step 1 — Clone the repository

```bash
git clone -b release https://github.com/thomas-baratto/BA.git
cd BA
```

### Step 2 — Download model weights

If you downloaded the **complete zip from DaRUS**, the models are already
included — skip to Step 3.

Otherwise, download the model archive from
[DaRUS (10.18419/DARUS-5815)](https://doi.org/10.18419/DARUS-5815) and place
the contents into `artifacts/models/` so the structure looks like:

```
BA/artifacts/models/
├── mlp/
│   ├── cone/       (best_model.pt, scalers.pkl, model_config.json)
│   └── isotherm/   (best_model.pt, scalers.pkl, model_config.json)
└── random/
    ├── cone/winner/            (model.pkl, scalers.pkl, model_config.json)
    └── isotherm/
        ├── KGE_winner/         (model.pkl, scalers.pkl, model_config.json)
        └── nRMSE_winner/       (model.pkl, scalers.pkl, model_config.json)
```

### Step 3 — Install

#### Option A — pip install (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the `ba-predict` command that you can run from **any directory**:

```bash
ba-predict --help
```

#### Option B — PYTHONPATH (no install)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run from the project root with:

```bash
PYTHONPATH=. python scripts/deployment/predict.py --help
```

#### Option C — Docker (zero setup)

```bash
# Build the image (once)
docker build -t ba-predict .

# Show help
docker run --rm ba-predict

# Run a prediction (mount your CSV into the container)
docker run --rm -v $(pwd):/data ba-predict \
    -i /data/my_input.csv -d cone -m mlp -o /data/results/
```

---

## 3. Choose Your Prediction Task

Two prediction tasks are available:

### Isotherm (thermal plume geometry)

Predicts the shape of a thermal plume isotherm from 9 hydrogeological inputs.

| Input Column          | Description                    | Unit  |
|-----------------------|--------------------------------|-------|
| `Flow_well`           | Well flow rate                 | m³/s  |
| `Temp_diff`           | Temperature difference         | K     |
| `kW_well`             | Thermal power                  | kW    |
| `Hydr_gradient`       | Hydraulic gradient             | –     |
| `Hydr_conductivity`   | Hydraulic conductivity         | m/s   |
| `Aqu_thickness`       | Aquifer thickness              | m     |
| `Long_dispersivity`   | Longitudinal dispersivity      | m     |
| `Trans_dispersivity`  | Transverse dispersivity        | m     |
| `Isotherm`            | Target isotherm value          | K     |

**Outputs:** `Area` (m²), `Iso_distance` (m), `Iso_width` (m)

### Cone (depression cone size)

Predicts the size of the depression cone from 4 hydrogeological inputs.

| Input Column          | Description                    | Unit  |
|-----------------------|--------------------------------|-------|
| `Flow_well`           | Well flow rate                 | m³/s  |
| `Hydr_gradient`       | Hydraulic gradient             | –     |
| `Hydr_conductivity`   | Hydraulic conductivity         | m/s   |
| `Aqu_thickness`       | Aquifer thickness              | m     |

**Output:** `Cone` (m)

---

## 4. Prepare Your Input CSV

Create a CSV file with the required columns as header. One row per sample.

**Example — `my_cone_data.csv`:**

```csv
Flow_well,Hydr_gradient,Hydr_conductivity,Aqu_thickness
5.78705,0.008,0.007,10
115.741,0.008,0.007,25
0.4050935,0.0015,0.0035,16
```

**Example — `my_isotherm_data.csv`:**

```csv
Flow_well,Temp_diff,kW_well,Hydr_gradient,Hydr_conductivity,Aqu_thickness,Long_dispersivity,Trans_dispersivity,Isotherm
5543.4,3,804.563,0.0015,0.058,25,20,1,1
30,5,7.257,0.0015,0.00085,16,5,0.25,3
```

> **Note:** Column order does not matter, but column names must match exactly
> (case-sensitive). Extra columns are ignored.

---

## 5. Run Prediction

Make sure your virtualenv is active (or use Docker).

> All examples below use the `ba-predict` command (from `pip install -e .`).
> If you used Option B instead, replace `ba-predict` with
> `PYTHONPATH=. python scripts/deployment/predict.py`.

### Cone prediction

```bash
ba-predict --input my_cone_data.csv --dataset cone --model mlp
```

### Isotherm prediction

```bash
ba-predict --input my_isotherm_data.csv --dataset isotherm --model mlp
```

### Use a random network model

Random network models are available (Pareto frontier knee-point winners from a
sweep of 558 configurations). For **cone**, there is a single winner (the nRMSE
and KGE Pareto winners were the same model). For **isotherm**, two different
winners are available:

```bash
# Cone — single random winner
ba-predict -i my_cone_data.csv -d cone -m random

# Isotherm — nRMSE winner (default)
ba-predict -i my_isotherm_data.csv -d isotherm -m random

# Isotherm — KGE winner
ba-predict -i my_isotherm_data.csv -d isotherm -m random:KGE
```

| Model key | Cone model | Isotherm model |
|-----------|------------|----------------|
| `random` / `random:nRMSE` | edRVFL-SC (R²=0.977) | SResdRVFL (R²=0.900) |
| `random:KGE` | edRVFL-SC (R²=0.977) | dRVFL (R²=0.860) |

### Specify an output directory

By default, results are saved to `predictions_<dataset>_<timestamp>/`. You can
override this:

```bash
ba-predict -i my_cone_data.csv -d cone -m mlp -o my_results/
```

### Docker

```bash
docker run --rm -v $(pwd):/data ba-predict \
    -i /data/my_cone_data.csv -d cone -m mlp -o /data/results/
```

---

## 6. Inspect the Output

The output directory contains:

| File              | Description                                       |
|-------------------|---------------------------------------------------|
| `predictions.csv` | Predicted values (one column per output variable)  |
| `report.md`       | Summary report with input/output statistics        |

**Example — `predictions.csv` for cone:**

```csv
Cone
0.137943
1.139360
0.017241
```

To skip the report and only produce `predictions.csv`:

```bash
ba-predict -i my_cone_data.csv -d cone -m mlp --no-report
```

---

## 7. Full Example (copy-paste)

```bash
# 1. Clone and install
git clone -b release https://github.com/thomas-baratto/BA.git
cd BA
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Download models from DaRUS (https://doi.org/10.18419/DARUS-5815)
#    and place them in artifacts/models/ (or use the complete DaRUS zip)

# 3. Run prediction on included sample data
ba-predict -i sample_cone.csv -d cone -m mlp -o results/

# 4. View results
cat results/predictions.csv
cat results/report.md
```

### Full Example with Docker

```bash
# 1. Clone and build
git clone -b release https://github.com/thomas-baratto/BA.git
cd BA
# (place model weights in artifacts/models/ first)
docker build -t ba-predict .

# 2. Create a sample input
echo "Flow_well,Hydr_gradient,Hydr_conductivity,Aqu_thickness
5.78705,0.008,0.007,10
115.741,0.008,0.007,25
0.4050935,0.0015,0.0035,16" > sample.csv

# 3. Run prediction
docker run --rm -v $(pwd):/data ba-predict \
    -i /data/sample.csv -d cone -m mlp -o /data/results/

# 4. View results (on your host machine)
cat results/predictions.csv
```

---

## 8. CLI Reference

```
usage: ba-predict [-h] --input INPUT --dataset {isotherm,cone}
                  [--model {mlp,random,random:nRMSE,random:KGE}]
                  [--model-dir MODEL_DIR] [--output OUTPUT] [--no-report]

Options:
  -i, --input INPUT        Input CSV file with feature columns (required)
  -d, --dataset DATASET    Prediction task: isotherm or cone (required)
  -m, --model MODEL        Model type (default: mlp)
                           mlp          — Optuna-optimized MLP
                           random       — Pareto-frontier winner (nRMSE for isotherm)
                           random:nRMSE — same as 'random'
                           random:KGE   — KGE Pareto-frontier winner (isotherm only,
                                          cone has a single winner)
  --model-dir MODEL_DIR    Custom model directory (overrides default)
  -o, --output OUTPUT      Output directory (default: predictions_<timestamp>)
  --no-report              Skip generating the markdown report
```

---

## 9. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'core'` | Use `pip install -e .` or run with `PYTHONPATH=.` |
| `Error: Input CSV missing required columns: {...}` | Check that your CSV header matches the column names exactly (case-sensitive) |
| `No model found in artifacts/models/mlp/...` | Make sure the repository was cloned completely — the pre-trained models ship in `artifacts/models/` |
| `No model found in artifacts/models/random/...` | Random model binaries aren't in git — run `ba-retrain-random` to build them (~15 s) |
| `InconsistentVersionWarning` for scikit-learn | Safe to ignore — the scalers are compatible across minor versions |
| Predictions are all identical or negative | Your input values may be outside the training data range — check the units and magnitudes |

---

## 10. Using a Custom Model

If you trained your own model (see `README.md` → Training section), point
`predict.py` to your model directory:

```bash
PYTHONPATH=. python scripts/deployment/predict.py \
    --input data.csv \
    --dataset isotherm \
    --model mlp \
    --model-dir runs/my_custom_training_run/
```

The directory must contain `model_config.json` and `scalers.pkl`, plus either
`best_model.pt` (MLP) or `model.pkl` (random network). The model type is
auto-detected from the artifacts present.
