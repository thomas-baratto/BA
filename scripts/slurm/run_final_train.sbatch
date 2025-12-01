#!/bin/bash
#SBATCH --job-name=final_train_isotherm
#SBATCH --output=./slurm_jobs/final_train_%j.out
#SBATCH --error=./slurm_jobs/final_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH -w argon-gtx

# ============================================================================
# Final Model Training with Integrated Power Monitoring
# ============================================================================
# Usage:
#   sbatch --export=STUDY_NAME=my_study,JOURNAL_PATH=runs/my_run/journal.log scripts/slurm/run_final_train.sbatch
#
# Required parameters:
#   STUDY_NAME=<study_name>    # Name of the Optuna study
#   JOURNAL_PATH=<path>        # Path to the Optuna journal log file
#
# Optional parameters:
#   CSV_FILE=<path>            # CSV file path (default: data/Clean_Results_Isotherm.csv)
#   OUTPUT_DIR=<path>          # Output directory (default: timestamped in runs/)
#   POWER_INTERVAL=1.0         # Power sampling interval (default: 1.0)
# ============================================================================

# --- Check required parameters ---
if [ -z "$STUDY_NAME" ]; then
    echo "ERROR: STUDY_NAME is required"
    exit 1
fi

if [ -z "$JOURNAL_PATH" ]; then
    echo "ERROR: JOURNAL_PATH is required"
    exit 1
fi

# --- Setup ---
# Create directories
mkdir -p ./slurm_jobs/logs

# Load modules
module purge
module load cuda/12.4.1

# Activate virtualenv
source /home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona/bin/activate
export PYTHONPATH=.

# --- Job Info ---
echo "============================================================================"
echo "SLURM Job Information"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Study Name: $STUDY_NAME"
echo "Journal Path: $JOURNAL_PATH"
echo "============================================================================"

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

# Show GPU info
nvidia-smi

# --- Run Final Training ---
echo "============================================================================"
echo "Starting final model training..."
echo "============================================================================"

# Build command
TRAIN_CMD="python scripts/train_final_model.py --study-name \"$STUDY_NAME\" --journal-path \"$JOURNAL_PATH\""

if [ -n "$CSV_FILE" ]; then
    TRAIN_CMD="$TRAIN_CMD --csv-file \"$CSV_FILE\""
fi

if [ -n "$OUTPUT_DIR" ]; then
    TRAIN_CMD="$TRAIN_CMD --output-dir \"$OUTPUT_DIR\""
fi

if [ -n "$POWER_INTERVAL" ]; then
    TRAIN_CMD="$TRAIN_CMD --power-interval \"$POWER_INTERVAL\""
fi

echo "Command: $TRAIN_CMD"
eval $TRAIN_CMD

EXIT_CODE=$?

echo "============================================================================"
echo "Job Completed"
echo "============================================================================"
echo "Exit Code: $EXIT_CODE"

exit $EXIT_CODE
