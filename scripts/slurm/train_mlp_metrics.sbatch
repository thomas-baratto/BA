#!/bin/bash
#SBATCH --job-name=mlp_metrics
#SBATCH --output=./slurm_jobs/mlp_metrics_%j.out
#SBATCH --error=./slurm_jobs/mlp_metrics_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH -w argon-gtx

# ============================================================================
# Train MLP from Optuna best trials with full metrics
# ============================================================================
# Usage:
#   sbatch scripts/slurm/train_mlp_metrics.sbatch                    # Train both
#   sbatch --export=DATASET=cone scripts/slurm/train_mlp_metrics.sbatch
#   sbatch --export=DATASET=isotherm scripts/slurm/train_mlp_metrics.sbatch
#
# Optional parameters:
#   DATASET=<cone|isotherm|all>   # Which dataset(s) to train (default: all)
#   OUTPUT_DIR=<path>             # Output directory (default: runs/mlp_from_optuna)
# ============================================================================

# --- Setup ---
mkdir -p ./slurm_jobs

# Load modules
module purge
module load cuda/12.4.1

# Activate virtualenv
source /home/barattts/lavoltabuona/BA/.venv/env/bin/activate
export PYTHONPATH=.

# --- Job Info ---
echo "============================================================================"
echo "SLURM Job Information"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Dataset: ${DATASET:-all}"
echo "============================================================================"

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

# Show GPU info
nvidia-smi

# --- Run MLP Training ---
echo "============================================================================"
echo "Starting MLP training from Optuna best trials..."
echo "============================================================================"

# Build command with defaults
TRAIN_CMD="python scripts/train_mlp_with_metrics.py"

if [ -n "$DATASET" ]; then
    TRAIN_CMD="$TRAIN_CMD --dataset $DATASET"
else
    TRAIN_CMD="$TRAIN_CMD --dataset all"
fi

if [ -n "$OUTPUT_DIR" ]; then
    TRAIN_CMD="$TRAIN_CMD --output-dir $OUTPUT_DIR"
else
    TRAIN_CMD="$TRAIN_CMD --output-dir data/good_runs/mlp_final"
fi

echo "Command: $TRAIN_CMD"
eval $TRAIN_CMD

EXIT_CODE=$?

echo "============================================================================"
echo "Job Completed"
echo "============================================================================"
echo "Exit Code: $EXIT_CODE"

exit $EXIT_CODE
