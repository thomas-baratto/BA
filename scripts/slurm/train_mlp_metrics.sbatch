#!/bin/bash
#SBATCH --job-name=mlp_metrics
#SBATCH --output=./slurm_jobs/mlp_metrics_%j.out
#SBATCH --error=./slurm_jobs/mlp_metrics_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:7
#SBATCH --mem=754G
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

OUT_DIR="${OUTPUT_DIR:-artifacts/models/mlp}"

# ── Job-level power monitor ─────────────────────────────────
POWER_LOG_DIR="${OUT_DIR}/power_monitor"
mkdir -p "$POWER_LOG_DIR"
echo "Starting job-level power monitor → $POWER_LOG_DIR"
python monitoring/power_monitor.py \
    --output-dir "$POWER_LOG_DIR" \
    --interval 1.0 \
    --filter python &
POWER_PID=$!

if [ "$DATASET" == "isotherm" ] || [ "$DATASET" == "cone" ]; then
    TRAIN_CMD="python scripts/training/train_mlp_with_metrics.py --dataset $DATASET --output-dir \"$OUT_DIR\""
    echo "Command: $TRAIN_CMD"
    eval $TRAIN_CMD
    EXIT_CODE=$?
else
    echo "============================================================================"
    echo "Running BOTH datasets simultaneously on distinct isolated GPUs..."
    echo "============================================================================"
    
    CUDA_VISIBLE_DEVICES=0,1,2 python scripts/training/train_mlp_with_metrics.py --dataset isotherm --output-dir "$OUT_DIR" &
    PID1=$!
    
    CUDA_VISIBLE_DEVICES=3,4,5,6 python scripts/training/train_mlp_with_metrics.py --dataset cone --output-dir "$OUT_DIR" &
    PID2=$!
    
    wait $PID1
    EXIT1=$?
    
    wait $PID2
    EXIT2=$?
    
    if [ $EXIT1 -ne 0 ] || [ $EXIT2 -ne 0 ]; then
        EXIT_CODE=1
    else
        EXIT_CODE=0
    fi
fi

echo "Stopping power monitor (PID=$POWER_PID)..."
kill $POWER_PID 2>/dev/null && wait $POWER_PID 2>/dev/null || true

echo "============================================================================"
echo "Job Completed"
echo "============================================================================"
echo "Exit Code: $EXIT_CODE"

exit $EXIT_CODE
