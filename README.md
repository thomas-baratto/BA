# Thermal Plume Prediction — Machine Learning Models

Predict thermal plume parameters (isotherm geometry or depression cone size) from hydrogeological inputs using pre-trained neural network models.

---

## 1. Quick Start

### Installation
Set up your virtual environment and install the package:
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install ./code
```
*(Note: Pretrained model files under `models/` are packaged as compressed .zip archives for archive compliance and do not need to be unzipped manually; they are automatically extracted on-the-fly when you run predictions!)*

### Usage
The `ba-predict` command automatically detects the dataset type from your CSV headers.

> **TIP (DaRUS Download Hint)**: When downloading sample datasets directly from the DaRUS portal, the platform may automatically ingest and convert CSV files into tab-delimited `.tab` files. To ensure seamless compatibility with the CLI, please select **"Original Format"** from the file download options to preserve them as `.csv`!

Run prediction using auto-detection (recommended):
```bash
ba-predict -i data/sample_cone.csv
```

Run prediction with a specific model variant (using the model types defined in the table below):
```bash
ba-predict -i data/sample_isotherm.csv -m randomized:SResdRVFL
```

### Available Pre-trained Models

The following **5 pre-trained models** are distributed inside the `models/` directory. You can specify a model explicitly using the `-m` / `--model` option flag:

| Task / Dataset | Model Architecture | CLI Selection Flag | Validation Accuracy (R²) |
|:---|:---|:---|:---|
| **Isotherm Plume** | Multi-Layer Perceptron (MLP) | `-m mlp` | **≈1.000** |
| **Isotherm Plume** | Optimized SResdRVFL (Randomized) | `-m randomized:SResdRVFL` *(default)* | **0.900** |
| **Isotherm Plume** | Optimized dRVFL (Randomized) | `-m randomized:dRVFL` | **0.860** |
| **Depression Cone** | Multi-Layer Perceptron (MLP) | `-m mlp` | **0.991** |
| **Depression Cone** | Optimized edRVFL-SC (Randomized Ensemble) | `-m randomized` / `-m randomized:edRVFL_SC` | **0.977** |

### Verify Installation
Run the automated validation suite to verify the package installation and model weights integrity:
```bash
ba-predict --test
```

### Docker Alternative
If you prefer not to install Python dependencies locally, you can run the entire inference suite in Docker. 

> **IMPORTANT**: You **must** run these commands from the **repository root directory** (which contains the `models/` and `data/` directories) so the build context has access to all weights and inputs:

```bash
# Build the image using the nested Dockerfile
docker build -t ba-predict -f code/Dockerfile .

# Run predictions by mounting the current directory
docker run --rm -v $(pwd):/app/host ba-predict -i /app/host/data/sample_cone.csv -o /app/host/outputs/
```

---

## 2. Project Overview

This repository contains the inference engine and pre-trained artifacts for the bachelor thesis:
> *Comparison of Neural Network Architectures on the Task of Modeling Heat Plume Dimensions in Groundwater* — Thomas Baratto, University of Stuttgart, 2026.

**Key Capabilities:**
- **Pre-trained Models**: High-accuracy MLP and Randomized (RVFL) models for immediate use.
- **Auto-Detection**: Zero-config prediction; the tool identifies the task (Cone vs. Isotherm) from input headers.
- **Verified Accuracy**: Includes validation suites to ensure numerical consistency with the original research.

### Repository Structure
```text
.
├── models/             # Pre-trained MLP and Randomized weights
├── data/               # Sample inputs and full simulation datasets
└── code/               # Software and packaging workspace
    ├── pyproject.toml  # Package configuration
    ├── Dockerfile      # Containerized environment
    ├── predict.py      # Main CLI entry point
    ├── core/           # Unified inference logic
    ├── config/         # Dataset metadata and column mappings
    └── tests/          # Unit and validation test suites
```

---

## 3. Supported Tasks

### Isotherm (Plume Geometry)
Predicts the geometry of a thermal plume isotherm from 9 hydrogeological inputs.

| Column | Description | Unit |
| :--- | :--- | :--- |
| `Flow_well` | Well flow rate | m³/s |
| `Temp_diff` | Temperature difference | K |
| `Temp_diff_real` | Physical temperature difference | K |
| `kW_well` | Thermal power | kW |
| `Hydr_gradient` | Hydraulic gradient | – |
| `Hydr_conductivity`| Hydraulic conductivity | m/s |
| `Aqu_thickness` | Aquifer thickness | m |
| `Long_dispersivity` | Longitudinal dispersivity | m |
| `Trans_dispersivity` | Transverse dispersivity | m |
| `Isotherm` | Target isotherm value | K |

**Outputs**: `Area` (m²), `Iso_distance` (m), `Iso_width` (m)

### Depression Cone
Predicts the radius of a hydraulic depression cone from 4 inputs.

| Column | Description | Unit |
| :--- | :--- | :--- |
| `Flow_well` | Well flow rate | m³/s |
| `Hydr_gradient` | Hydraulic gradient | – |
| `Hydr_conductivity` | Hydraulic conductivity | m/s |
| `Aqu_thickness` | Aquifer thickness | m |

**Output**: `Cone` (m)

---

## 4. Data Origin

The training datasets used to develop these models were generated by **Dr. Fabian Böttcher** (Technical University of Munich, Chair of Hydrogeology) using **FEFLOW** numerical simulations. 

This work was part of the [Geo.KW project](https://www.cee.ed.tum.de/hydro/projects/geothermal-energy-group/geokw/), focusing on modeling heat plume dimensions in groundwater.

---

## 5. License & Citation

- **Software**: The Python code is released under the [MIT License](LICENSE).
- **Data & Models**: The datasets and pre-trained models are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

If you use this software or the models, please cite:
```bibtex
@misc{baratto2026thermal,
  author    = {Baratto, Thomas},
  title     = {Trained Neural Networks on Simulated Data of Groundwater Heat Plume Characteristics},
  year      = {2026},
  doi       = {10.18419/DARUS-5815},
  url       = {https://doi.org/10.18419/DARUS-5815}
}
```

---

## 6. Data & Model Archive

The permanent archive containing the training datasets and model artifacts is available on **DaRUS**.

[![DaRUS DOI](https://img.shields.io/badge/doi-10.18419%2Fdarus--5815-d45815.svg)](https://doi.org/10.18419/darus-5815)

*Created as part of the Bachelor Thesis at the University of Stuttgart (IPVS), 2026.*
