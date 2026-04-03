#!/bin/bash
#SBATCH --job-name=random_model
#SBATCH --output=./slurm_jobs/random_model_%j.out
#SBATCH --error=./slurm_jobs/random_model_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:7
#SBATCH --mem=754G
#SBATCH --time=1-00:00:00
#SBATCH -w argon-gtx

# ============================================================================
# Train a single random-weight network model
# ============================================================================
# Usage:
#   sbatch scripts/slurm/run_random_model.sbatch                          # Train ELM on cone
#   sbatch --export=MODEL=dRVFL,DATASET=isotherm scripts/slurm/run_random_model.sbatch
#   sbatch --export=MODEL=SResdRVFL,DATASET=isotherm scripts/slurm/run_random_model.sbatch
#
# Parameters (via --export or defaults):
#   MODEL=<ELM|dRVFL|edRVFL|edRVFL-SC|esc-edRVFL|SResdRVFL>  (default: ELM)
#   DATASET=<cone|isotherm>                                    (default: cone)
#   OUTPUT_DIR=<path>                                          (default: auto)
# ============================================================================

# --- Setup ---
mkdir -p ./slurm_jobs

# Load modules
module purge
module load cuda/12.4.1

# Activate virtualenv
source /home/barattts/lavoltabuona/BA/.venv/env/bin/activate
export PYTHONPATH=.

# --- Defaults ---
MODEL=${MODEL:-ELM}
DATASET=${DATASET:-cone}

# --- Job Info ---
echo "============================================================================"
echo "SLURM Job Information"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "============================================================================"

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

# Show GPU info
nvidia-smi

# --- Build command ---
CMD="python scripts/training/train_random_models.py --model ${MODEL} --dataset ${DATASET}"

# SResdRVFL benefits from direct links (asymmetric skip connections)
if [ "$MODEL" = "SResdRVFL" ]; then
    CMD="$CMD --direct-link"
    echo "SResdRVFL: enabling --direct-link"
fi

# Optional output directory override
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

echo "Running: $CMD"
echo "============================================================================"

# --- Run ---
$CMD

echo "============================================================================"
echo "Job completed at $(date)"
echo "============================================================================"
