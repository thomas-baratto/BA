#!/bin/bash
#SBATCH --job-name=random_winners
#SBATCH --output=./slurm_jobs/random_winners_%j.out
#SBATCH --error=./slurm_jobs/random_winners_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:7
#SBATCH --mem=350G
#SBATCH --time=0-01:00:00
#SBATCH -w argon-gtx

# ============================================================================
# Train random model winners from config/random_model_winners.json
# ============================================================================
# Usage:
#   sbatch scripts/slurm/train_random_winners.sbatch                 # Train all 3 winners
#   sbatch --export=ONLY=cone scripts/slurm/train_random_winners.sbatch
#   sbatch --export=ONLY="cone isotherm_nRMSE" scripts/slurm/train_random_winners.sbatch
#   sbatch --export=FORCE=1 scripts/slurm/train_random_winners.sbatch  # Force retrain
#
# Winners are defined in config/random_model_winners.json and trained to
# artifacts/models/random/{cone,isotherm}/. Uses the same hyperparameters
# that won the Pareto frontier selection in the sweep.
# ============================================================================

# --- Setup ---
mkdir -p ./slurm_jobs

module purge
module load cuda/12.4.1

source /home/barattts/lavoltabuona/BA/.venv/env/bin/activate
export PYTHONPATH=.

# --- Job Info ---
echo "============================================================================"
echo "SLURM Job Information"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Winners: ${ONLY:-all}"
echo "Force retrain: ${FORCE:-0}"
echo "============================================================================"

cd $SLURM_SUBMIT_DIR

nvidia-smi

# --- Build CLI args ---
RETRAIN_ARGS=""
if [ -n "$ONLY" ]; then
    RETRAIN_ARGS="--only $ONLY"
fi
if [ "${FORCE:-0}" = "1" ]; then
    RETRAIN_ARGS="$RETRAIN_ARGS --force"
fi

# --- Power monitor ---
POWER_LOG_DIR="artifacts/models/random/power_monitor"
mkdir -p "$POWER_LOG_DIR"
echo "Starting job-level power monitor → $POWER_LOG_DIR"
python monitoring/power_monitor.py \
    --output-dir "$POWER_LOG_DIR" \
    --interval 1.0 \
    --filter python &
POWER_PID=$!

# --- Train ---
echo "============================================================================"
echo "Training random model winners..."
echo "  Command: python scripts/deployment/retrain_random_models.py $RETRAIN_ARGS"
echo "============================================================================"

python scripts/deployment/retrain_random_models.py $RETRAIN_ARGS
EXIT_CODE=$?

# --- Stop power monitor ---
echo "Stopping power monitor (PID=$POWER_PID)..."
kill $POWER_PID 2>/dev/null && wait $POWER_PID 2>/dev/null || true

echo "============================================================================"
echo "Job Completed"
echo "============================================================================"
echo "Exit Code: $EXIT_CODE"

exit $EXIT_CODE
