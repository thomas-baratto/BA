# Pre-trained Models

This directory is the recommended location for storing pre-trained model artifacts downloaded from [DaRUS](https://doi.org/10.18419/DARUS-5815).

### 📥 Usage Instructions

1.  **Download**: Download the model ZIP files from the DaRUS archive.
2.  **Unzip**: Extract the ZIP files into this directory.
    *   *Example*: `unzip mlp-isotherm.zip -d models/mlp-isotherm`
3.  **Run**: The `predict.py` script will automatically detect any models placed here.

### 📁 Expected Structure
When unzipped, a model directory should contain at least:
- `model_config.json`: Architecture configuration
- `scalers.pkl`: Data scaling artifacts
- `best_model.pt` (MLP) or `model.pkl` (Randomized): Model weights
