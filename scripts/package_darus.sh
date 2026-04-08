#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Package the release-branch content for DaRUS upload.
#
# Produces a staging directory  darus_upload/  with:
#   1. Individual model zips   (each model downloadable on its own)
#   2. Code-only zip           (release branch without model binaries)
#   3. Complete zip            (code + all models in one archive)
#   4. A top-level README describing the structure
#
# Run from the repo root while on master (needs model binaries):
#   bash scripts/package_darus.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

VERSION="v1.0.0"
STAGING="darus_upload"
CODE_DIR="ba-thermal-plume-${VERSION}"

# ── clean slate ─────────────────────────────────────────────────────
rm -rf "${STAGING}"
mkdir -p "${STAGING}/models"

echo "▸ Exporting release-branch code …"
# Export the release branch tree (no .git) into the staging code dir
git archive --format=tar --prefix="${CODE_DIR}/" release | tar -xf - -C "${STAGING}"

# ── model definitions ──────────────────────────────────────────────
# Each entry:  ZIP_NAME  SOURCE_DIR  DEST_SUBDIR (inside code tree)
declare -a MODELS=(
    "mlp-cone|artifacts/models/mlp/cone|artifacts/models/mlp/cone"
    "mlp-isotherm|artifacts/models/mlp/isotherm|artifacts/models/mlp/isotherm"
    "random-cone-edRVFL-SC|artifacts/models/random/cone/winner|artifacts/models/random/cone/winner"
    "random-isotherm-dRVFL-KGE|artifacts/models/random/isotherm/KGE_winner|artifacts/models/random/isotherm/KGE_winner"
    "random-isotherm-SResdRVFL-nRMSE|artifacts/models/random/isotherm/nRMSE_winner|artifacts/models/random/isotherm/nRMSE_winner"
)

echo "▸ Creating individual model zips …"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r zip_name src_dir dest_subdir <<< "${entry}"

    # Create a temp dir mirroring the expected tree so the zip extracts cleanly
    tmpdir=$(mktemp -d)
    mkdir -p "${tmpdir}/${dest_subdir}"
    cp -r "${src_dir}/." "${tmpdir}/${dest_subdir}/"

    # Remove tensorboard event files (large, not needed for inference)
    find "${tmpdir}" -name "events.out.tfevents.*" -delete 2>/dev/null || true

    # Zip with paths relative to tmpdir root
    (cd "${tmpdir}" && zip -r -q "${OLDPWD}/${STAGING}/models/${zip_name}.zip" .)
    rm -rf "${tmpdir}"

    size=$(du -sh "${STAGING}/models/${zip_name}.zip" | cut -f1)
    echo "   ✓ ${zip_name}.zip  (${size})"
done

echo "▸ Creating code-only zip …"
(cd "${STAGING}" && zip -r -q "${CODE_DIR}-code.zip" "${CODE_DIR}/")
code_size=$(du -sh "${STAGING}/${CODE_DIR}-code.zip" | cut -f1)
echo "   ✓ ${CODE_DIR}-code.zip  (${code_size})"

echo "▸ Building complete archive (code + all models) …"
# Copy model binaries into the code tree (they were excluded on the release branch)
for entry in "${MODELS[@]}"; do
    IFS='|' read -r zip_name src_dir dest_subdir <<< "${entry}"
    mkdir -p "${STAGING}/${CODE_DIR}/${dest_subdir}"
    cp -r "${src_dir}/." "${STAGING}/${CODE_DIR}/${dest_subdir}/"
    # Remove tensorboard event files
    find "${STAGING}/${CODE_DIR}/${dest_subdir}" -name "events.out.tfevents.*" -delete 2>/dev/null || true
done

(cd "${STAGING}" && zip -r -q "${CODE_DIR}-complete.zip" "${CODE_DIR}/")
complete_size=$(du -sh "${STAGING}/${CODE_DIR}-complete.zip" | cut -f1)
echo "   ✓ ${CODE_DIR}-complete.zip  (${complete_size})"

# ── DaRUS README ───────────────────────────────────────────────────
cat > "${STAGING}/README.md" << 'EOF'
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
EOF

# ── summary ────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  DaRUS upload staging complete → ${STAGING}/"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Upload these files to DaRUS:"
echo ""
ls -lhS "${STAGING}"/*.zip "${STAGING}/models/"*.zip "${STAGING}/README.md" 2>/dev/null | \
    awk '{printf "    %-50s %s\n", $NF, $5}'
echo ""
echo "  Suggested DaRUS directory structure:"
echo "    /                          ← README.md"
echo "    /                          ← ba-thermal-plume-v1.0.0-complete.zip"
echo "    /                          ← ba-thermal-plume-v1.0.0-code.zip"
echo "    /models/                   ← individual model zips"
echo ""
