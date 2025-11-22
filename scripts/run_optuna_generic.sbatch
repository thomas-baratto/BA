#!/bin/bash
#SBATCH --job-name=optuna_generic
#SBATCH --output=./slurm_jobs/optuna_%A_%a.out
#SBATCH --error=./slurm_jobs/optuna_%A_%a.err
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -w argon-gtx

# ============================================================================
# Generic Optuna Hyperparameter Tuning with Power Monitoring
# ============================================================================
# Usage:
#   sbatch --export=CSV_FILE=./data/Clean_Results_Isotherm.csv,TARGET=all scripts/run_optuna_generic.sbatch
#   sbatch --export=CSV_FILE=./data/Depression_cones.csv,TARGET=Cone scripts/run_optuna_generic.sbatch
#
# Optional parameters:
#   N_TRIALS=320           # Number of trials per GPU (default: 320)
#   POWER_INTERVAL=2.0     # Power sampling interval (default: 2.0)
#   DB_PATH=sqlite:///optuna_study.db  # Legacy database path (auto-selected if unset)
#   OPTUNA_STORAGE_URL=postgresql+psycopg2://user:pass@host/db  # RDB storage override
#   STUDY_NAME=custom_study  # Custom study name (auto-generated if not set)
# ============================================================================

# --- Default Parameters ---
CSV_FILE=${CSV_FILE:-./data/Clean_Results_Isotherm.csv}
TARGET=${TARGET:-all}
N_TRIALS=${N_TRIALS:-320}
POWER_INTERVAL=${POWER_INTERVAL:-2.0}

# RDB storage overrides take precedence
if [ -n "$OPTUNA_STORAGE_URL" ]; then
    DB_PATH="$OPTUNA_STORAGE_URL"
    echo "Using OPTUNA_STORAGE_URL override"
elif [ -n "$STORAGE_URL" ]; then
    DB_PATH="$STORAGE_URL"
    echo "Using STORAGE_URL override"
fi

# Auto-select database based on CSV file if not explicitly provided
if [ -z "$DB_PATH" ]; then
    if [[ "$CSV_FILE" == *"Clean_Results_Isotherm"* ]]; then
        DB_PATH="sqlite:///optuna_study_isotherm.db"
        echo "Auto-selected Isotherm database"
    elif [[ "$CSV_FILE" == *"Depression_cones"* ]]; then
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
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "CSV File: $CSV_FILE"
echo "Target: $TARGET"
echo "Number of Trials: $N_TRIALS"
echo "Database: $DB_PATH"
if [ -n "$STUDY_NAME" ]; then
    echo "Study Name: $STUDY_NAME"
fi
echo "============================================================================"

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

# Show GPU info
nvidia-smi

# --- Determine GPU to use based on array task ID ---
export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# --- Generate study name if not provided ---
DATASET_NAME=$(basename "$CSV_FILE" .csv)
if [ -z "$STUDY_NAME" ]; then
    if [ "$TARGET" == "all" ]; then
        STUDY_NAME="nn_study_${DATASET_NAME}"
    else
        TARGET_CLEAN=$(echo "$TARGET" | tr ',' '_')
        STUDY_NAME="nn_study_${DATASET_NAME}_${TARGET_CLEAN}"
    fi
fi

echo "Study Name: $STUDY_NAME"

# --- Start Power Monitoring ---
POWER_LOG_DIR="./power_logs/optuna_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
echo "Starting power monitoring..."
echo "Power logs will be saved to: $POWER_LOG_DIR"

python power_monitor.py \
    --output-dir "$POWER_LOG_DIR" \
    --interval "$POWER_INTERVAL" \
    --filter python &

POWER_MONITOR_PID=$!
echo "Power monitor started with PID: $POWER_MONITOR_PID"
sleep 2

# --- Run Optuna ---
echo "============================================================================"
echo "Starting Optuna optimization..."
echo "============================================================================"

OPTUNA_CMD="python run_optuna.py \
    --csv-file \"$CSV_FILE\" \
    --target \"$TARGET\" \
    --n-trials $N_TRIALS \
    --storage-url \"$DB_PATH\" \
    --study-name \"$STUDY_NAME\""

echo "Command: $OPTUNA_CMD"
eval $OPTUNA_CMD

OPTUNA_EXIT_CODE=$?

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
echo "Exit Code: $OPTUNA_EXIT_CODE"
echo "Study Name: $STUDY_NAME"
echo "Database: $DB_PATH"
echo "Power Logs: $POWER_LOG_DIR"
echo ""
echo "Next steps:"
echo "  - View results: optuna-dashboard $DB_PATH"
echo "  - Analyze power: python analyze_power_logs.py $POWER_LOG_DIR"
echo "  - Train final model: python train_final_model.py --study-name $STUDY_NAME --storage-url $DB_PATH"
echo "============================================================================"

exit $OPTUNA_EXIT_CODE
