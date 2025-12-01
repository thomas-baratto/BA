#!/bin/bash
#SBATCH --job-name=final_train_generic
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
# Generic Final Model Training with Power Monitoring
# ============================================================================
# Usage:
#   sbatch --export=STUDY_NAME=nn_study_Clean_Results_Isotherm_all scripts/final_train_generic.sbatch
#   sbatch --export=STUDY_NAME=nn_study_Depression_cones_Cone,CSV_FILE=./data/Depression_cones.csv scripts/final_train_generic.sbatch
#
# Required parameters:
#   STUDY_NAME=<study_name>  # Name of the Optuna study to load best params from
#
# Optional parameters:
#   CSV_FILE=<path>          # CSV file path (optional, depends on train_final_model.py)
#   POWER_INTERVAL=2.0       # Power sampling interval (default: 2.0)
#   DB_PATH=<path>           # Legacy database path (auto-selected if not provided)
#   OPTUNA_STORAGE_URL=postgresql+psycopg2://user:pass@host/db  # RDB storage override
# ============================================================================

# --- Check required parameters ---
if [ -z "$STUDY_NAME" ]; then
    echo "ERROR: STUDY_NAME is required"
    echo "Usage: sbatch --export=STUDY_NAME=<study_name> scripts/final_train_generic.sbatch"
    exit 1
fi

# --- Default Parameters ---
POWER_INTERVAL=${POWER_INTERVAL:-2.0}

# RDB storage overrides take precedence
if [ -n "$OPTUNA_STORAGE_URL" ]; then
    DB_PATH="$OPTUNA_STORAGE_URL"
    echo "Using OPTUNA_STORAGE_URL override"
elif [ -n "$STORAGE_URL" ]; then
    DB_PATH="$STORAGE_URL"
    echo "Using STORAGE_URL override"
fi

# Auto-select database based on study name if not explicitly provided
if [ -z "$DB_PATH" ]; then
    if [[ "$STUDY_NAME" == *"Isotherm"* ]]; then
        DB_PATH="sqlite:///optuna_study_isotherm.db"
        echo "Auto-selected Isotherm database"
    elif [[ "$STUDY_NAME" == *"Depression"* ]] || [[ "$STUDY_NAME" == *"Cone"* ]]; then
        DB_PATH="sqlite:///optuna_study_depression.db"
        echo "Auto-selected Depression database"
    else
        DB_PATH="sqlite:///optuna_study.db"
        echo "Using default database"
    fi
fi

# --- Setup ---
# Create directories
mkdir -p ./slurm_jobs/logs
mkdir -p ./power_logs

# Load modules
module purge
module load cuda/12.4.1

# Activate virtualenv
source /home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona/bin/activate

# --- Job Info ---
echo "============================================================================"
echo "SLURM Job Information"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Study Name: $STUDY_NAME"
echo "Database: $DB_PATH"
if [ -n "$CSV_FILE" ]; then
    echo "CSV File: $CSV_FILE"
fi
echo "============================================================================"

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

# Show GPU info
nvidia-smi

# --- Start Power Monitoring ---
POWER_LOG_DIR="./power_logs/final_train_${SLURM_JOB_ID}"
echo "Starting power monitoring..."
echo "Power logs will be saved to: $POWER_LOG_DIR"

python monitoring/power_monitor.py \
    --output-dir "$POWER_LOG_DIR" \
    --interval "$POWER_INTERVAL" \
    --filter python &

POWER_MONITOR_PID=$!
echo "Power monitor started with PID: $POWER_MONITOR_PID"
sleep 2

# --- Run Final Training ---
echo "============================================================================"
echo "Starting final model training..."
echo "============================================================================"

# Build command
TRAIN_CMD="python scripts/train_final_model.py --study-name \"$STUDY_NAME\" --storage-url \"$DB_PATH\""

# Add CSV file if provided
if [ -n "$CSV_FILE" ]; then
    TRAIN_CMD="$TRAIN_CMD --csv-file \"$CSV_FILE\""
fi

echo "Command: $TRAIN_CMD"
eval $TRAIN_CMD

TRAIN_EXIT_CODE=$?

# --- Stop Power Monitoring ---
echo "============================================================================"
echo "Stopping power monitoring..."
echo "============================================================================"
kill -SIGINT $POWER_MONITOR_PID
wait $POWER_MONITOR_PID 2>/dev/null

echo "Power logs saved to: $POWER_LOG_DIR"

# --- Summary ---
echo "============================================================================"
echo "Job Completed"
echo "============================================================================"
echo "Exit Code: $TRAIN_EXIT_CODE"
echo "Study Name: $STUDY_NAME"
echo "Power Logs: $POWER_LOG_DIR"
echo ""
echo "Next steps:"
echo "  - Analyze power: python monitoring/analyze_power_logs.py $POWER_LOG_DIR"
echo "============================================================================"

exit $TRAIN_EXIT_CODE
