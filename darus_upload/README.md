# Thermal Plume Prediction — DaRUS Data Package

**Companion data for the bachelor thesis:**
*Comparison of Neural Network Architectures on the Task of Modeling Heat Plume
Dimensions in Groundwater* — Thomas Baratto, University of Stuttgart (IPVS), 2026.

**Source code:** <https://github.com/thomas-baratto/BA> (release branch)

---

## Files

### Complete archive

| File | Description |
|------|-------------|
| `ba-thermal-plume-v1.0.0-complete.zip` | Full inference package: code + all 5 pre-trained models. Extract and follow the README inside to run predictions. |
| `ba-thermal-plume-v1.0.0-code.zip` | Code only (no model weights). Use if you only need specific models — download them individually below. |

### Individual models

Each model zip extracts into the `artifacts/models/` subdirectory expected by the
inference code. Download only the model(s) you need and extract into the code
directory.

| File | Architecture | Task | Targets | Selection criterion |
|------|-------------|------|---------|---------------------|
| `models/mlp-cone.zip` | MLP (2 hidden layers, 244 neurons, LeakyReLU) | Depression cone | Cone | Optuna-tuned (200 trials) |
| `models/mlp-isotherm.zip` | MLP (5 hidden layers, 256 neurons, GELU) | Isotherm geometry | Area, Iso_distance, Iso_width | Optuna-tuned (200 trials) |
| `models/random-cone-edRVFL-SC.zip` | edRVFL-SC (1000 hidden, 3 layers, GELU) | Depression cone | Cone | Pareto frontier winner |
| `models/random-isotherm-dRVFL-KGE.zip` | dRVFL (1500 hidden, 1 layer, ELU) | Isotherm geometry | Area, Iso_distance, Iso_width | Best KGE on Pareto frontier |
| `models/random-isotherm-SResdRVFL-nRMSE.zip` | SResdRVFL (1500 hidden, 1 layer, GELU) | Isotherm geometry | Area, Iso_distance, Iso_width | Best nRMSE on Pareto frontier |

### Quick start (individual model download)

```bash
# 1. Get the code
unzip ba-thermal-plume-v1.0.0-code.zip
cd ba-thermal-plume-v1.0.0

# 2. Download and extract just the model you need (e.g. MLP cone)
unzip ../models/mlp-cone.zip        # extracts into artifacts/models/mlp/cone/

# 3. Install and predict
pip install -e .
ba-predict -i sample_cone.csv -d cone -m mlp
```

### Each model zip contains

| File | Description |
|------|-------------|
| `model_config.json` | Architecture hyperparameters |
| `artifact_manifest.json` | Training metadata, features, targets, performance metrics |
| `best_model.pt` or `model.pkl` | Trained model weights (PyTorch or scikit-learn) |
| `scalers.pkl` | Input/output scalers (must match training data) |
| `results_*.json` | Detailed test-set metrics (where available) |
| `plots/` | Training diagnostics and prediction quality plots (PDF) |
| `power_monitor/` | Energy consumption logs from training |
| `resources/` | CPU/GPU usage plots (MLP models only) |

---

## License

See the LICENSE file in the code archive for details.
