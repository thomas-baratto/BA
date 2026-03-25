#!/usr/bin/env bash
# ============================================================================
# package_release.sh — Build a self-contained inference archive for DaRUS
#
# Creates: ba-thermal-plume-v1.0.0.tar.gz
#
# The archive contains everything needed to run predictions on CPU:
#   - Python package files (core + config + scripts/deployment)
#   - Pre-trained model artifacts (MLP + random model .pkl)
#   - Sample input CSVs
#   - Documentation (README, INFERENCE_GUIDE)
#   - Dockerfile for containerised usage
#   - pyproject.toml + requirements.txt for pip install
#
# Usage:
#   cd BA/
#   bash scripts/deployment/package_release.sh
# ============================================================================
set -euo pipefail

VERSION="1.0.0"
ARCHIVE_NAME="ba-thermal-plume-v${VERSION}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/dist/${ARCHIVE_NAME}"

echo "============================================================"
echo "  Packaging ${ARCHIVE_NAME}"
echo "============================================================"

# Clean previous build
rm -rf "${BUILD_DIR}" "${BUILD_DIR}.tar.gz"
mkdir -p "${BUILD_DIR}"

# --- Core inference code ---------------------------------------------------
echo "[1/7] Copying core inference code..."
mkdir -p "${BUILD_DIR}/core/random"
for f in __init__.py model.py model_wrapper.py inference.py; do
    cp "${PROJECT_ROOT}/core/${f}" "${BUILD_DIR}/core/${f}"
done
# Random model implementations (needed for unpickling)
cp "${PROJECT_ROOT}/core/random/"*.py "${BUILD_DIR}/core/random/"

# --- Config ----------------------------------------------------------------
echo "[2/7] Copying config..."
mkdir -p "${BUILD_DIR}/config"
cp "${PROJECT_ROOT}/config/__init__.py"   "${BUILD_DIR}/config/"
cp "${PROJECT_ROOT}/config/datasets.py"   "${BUILD_DIR}/config/"
# JSON configs used by datasets.py
cp "${PROJECT_ROOT}/config/"*.json        "${BUILD_DIR}/config/" 2>/dev/null || true

# --- Deployment script -----------------------------------------------------
echo "[3/7] Copying deployment script..."
mkdir -p "${BUILD_DIR}/scripts/deployment"
cp "${PROJECT_ROOT}/scripts/__init__.py"              "${BUILD_DIR}/scripts/"
touch "${BUILD_DIR}/scripts/deployment/__init__.py"
cp "${PROJECT_ROOT}/scripts/deployment/predict.py"    "${BUILD_DIR}/scripts/deployment/"

# --- Model artifacts -------------------------------------------------------
echo "[4/7] Copying model artifacts..."

# MLP models (small, always included)
mkdir -p "${BUILD_DIR}/artifacts/models/mlp"
cp -r "${PROJECT_ROOT}/artifacts/models/mlp/cone"     "${BUILD_DIR}/artifacts/models/mlp/"
cp -r "${PROJECT_ROOT}/artifacts/models/mlp/isotherm" "${BUILD_DIR}/artifacts/models/mlp/"

# Random models — include ONLY inference-essential files (skip plots, test_predictions, etc.)
for variant in cone/winner isotherm/nRMSE_winner isotherm/KGE_winner; do
    src="${PROJECT_ROOT}/artifacts/models/random/${variant}"
    dst="${BUILD_DIR}/artifacts/models/random/${variant}"
    mkdir -p "${dst}"
    for f in model_config.json scalers.pkl artifact_manifest.json; do
        [ -f "${src}/${f}" ] && cp "${src}/${f}" "${dst}/"
    done
    # model.pkl — include if it exists (built by retrain_random_models.py)
    if [ -f "${src}/model.pkl" ]; then
        cp "${src}/model.pkl" "${dst}/"
    else
        echo "  WARNING: ${variant}/model.pkl not found — run retrain_random_models.py first"
    fi
done

# --- Samples, docs, and build files ----------------------------------------
echo "[5/7] Copying samples and documentation..."
cp "${PROJECT_ROOT}/sample_cone.csv"      "${BUILD_DIR}/"
cp "${PROJECT_ROOT}/sample_isotherm.csv"  "${BUILD_DIR}/"
cp "${PROJECT_ROOT}/requirements.txt"     "${BUILD_DIR}/"
cp "${PROJECT_ROOT}/pyproject.toml"       "${BUILD_DIR}/"
cp "${PROJECT_ROOT}/README.md"            "${BUILD_DIR}/"
cp "${PROJECT_ROOT}/Dockerfile"           "${BUILD_DIR}/"

mkdir -p "${BUILD_DIR}/docs"
cp "${PROJECT_ROOT}/docs/INFERENCE_GUIDE.md" "${BUILD_DIR}/docs/"

# --- Strip unnecessary files from MLP dirs (plots, resources, etc.) --------
echo "[6/7] Stripping non-essential files from artifact dirs..."
find "${BUILD_DIR}/artifacts/models/mlp" -type d -name "plots"     -exec rm -rf {} + 2>/dev/null || true
find "${BUILD_DIR}/artifacts/models/mlp" -type d -name "resources" -exec rm -rf {} + 2>/dev/null || true
find "${BUILD_DIR}/artifacts/models/mlp" -type d -name "stats"     -exec rm -rf {} + 2>/dev/null || true
find "${BUILD_DIR}/artifacts/models"     -name "test_predictions.npz" -delete 2>/dev/null || true
find "${BUILD_DIR}/artifacts/models"     -name "results_*.json"      -delete 2>/dev/null || true
find "${BUILD_DIR}"                      -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# --- Create tarball ---------------------------------------------------------
echo "[7/7] Creating archive..."
cd "${PROJECT_ROOT}/dist"
tar czf "${ARCHIVE_NAME}.tar.gz" "${ARCHIVE_NAME}"

# Summary
ARCHIVE_PATH="${PROJECT_ROOT}/dist/${ARCHIVE_NAME}.tar.gz"
SIZE=$(du -sh "${ARCHIVE_PATH}" | cut -f1)
FILE_COUNT=$(tar tzf "${ARCHIVE_PATH}" | wc -l)

echo ""
echo "============================================================"
echo "  Archive created: dist/${ARCHIVE_NAME}.tar.gz"
echo "  Size:  ${SIZE}"
echo "  Files: ${FILE_COUNT}"
echo "============================================================"
echo ""
echo "Contents:"
tar tzf "${ARCHIVE_PATH}" | head -40
echo "  ..."
echo ""
echo "Upload to DaRUS:"
echo "  1. Go to your DaRUS dataset → Files → Upload"
echo "  2. Upload dist/${ARCHIVE_NAME}.tar.gz"
echo "  3. Add description: 'Inference package — CPU-only, pip install -e .'"
echo "============================================================"
