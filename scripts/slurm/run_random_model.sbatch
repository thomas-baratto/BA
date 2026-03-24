#!/bin/bash
#SBATCH --job-name=random_model
#SBATCH --output=./slurm_jobs/random_model_%j.out
#SBATCH --error=./slurm_jobs/random_model_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH -w argon-gtx

set -euo pipefail

module purge
module load cuda/12.4.1
source /home/barattts/lavoltabuona/BA/.venv/env/bin/activate

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_jobs
export PYTHONPATH=.

MODEL=${MODEL:-ELM}
DATASET=${DATASET:-isotherm}
ACTIVATION=${ACTIVATION:-ReLU}
N_HIDDEN=${N_HIDDEN:-1000}
N_LAYERS=${N_LAYERS:-2}
N_ENSEMBLE=${N_ENSEMBLE:-20}
N_BLOCKS=${N_BLOCKS:-5}
SC_MODE=${SC_MODE:-dense}
RSC_PROB=${RSC_PROB:-0.3}
FEATURE_SCALER=${FEATURE_SCALER:-robust}
LABEL_SCALER=${LABEL_SCALER:-robust}
BASE_DIR=${BASE_DIR:-runs}
NO_SAVE_MODEL=${NO_SAVE_MODEL:-1}
USE_AREA_ROOT=${USE_AREA_ROOT:-0}

CMD="python scripts/training/train_random_models.py \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --activation ${ACTIVATION} \
    --n-hidden ${N_HIDDEN} \
    --n-layers ${N_LAYERS} \
    --n-ensemble ${N_ENSEMBLE} \
    --n-blocks ${N_BLOCKS} \
    --sc-mode ${SC_MODE} \
    --rsc-prob ${RSC_PROB} \
    --feature-scaler ${FEATURE_SCALER} \
    --label-scaler ${LABEL_SCALER} \
    --base-dir ${BASE_DIR}"

if [ "$NO_SAVE_MODEL" = "1" ]; then
    CMD="${CMD} --no-save-model"
fi

if [ "$USE_AREA_ROOT" = "1" ]; then
    CMD="${CMD} --use-area-root"
fi

if [ "$MODEL" = "SResdRVFL" ]; then
    CMD="${CMD} --direct-link"
fi

echo "Running: ${CMD}"
eval "$CMD"
