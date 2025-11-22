#!/bin/bash
#SBATCH --job-name=optuna_isotherm
#SBATCH --output=./slurm_jobs/optuna_isotherm_%A_%a.out
#SBATCH --error=./slurm_jobs/optuna_isotherm_%A_%a.err
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH -w argon-gtx

# ============================================================================
# Optuna Hyperparameter Tuning for Isotherm Dataset
# ============================================================================
# Uses 4 parallel tasks to avoid SQLite database contention
# ============================================================================

# --- Parameters ---
CSV_FILE="./data/Clean_Results_Isotherm.csv"
TARGET="all"
N_TRIALS=${N_TRIALS:-320}
POWER_INTERVAL=${POWER_INTERVAL:-2.0}
DB_PATH="sqlite:///optuna_study_isotherm.db"
STUDY_NAME="nn_study_Clean_Results_Isotherm"

if [ -n "$OPTUNA_STORAGE_URL" ]; then
    DB_PATH="$OPTUNA_STORAGE_URL"
    echo "Using OPTUNA_STORAGE_URL override"
elif [ -n "$STORAGE_URL" ]; then
    DB_PATH="$STORAGE_URL"
    echo "Using STORAGE_URL override"
fi

# --- Setup ---
mkdir -p ./slurm_jobs
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
echo "Dataset: Clean Results Isotherm"
echo "CSV File: $CSV_FILE"
echo "Target: $TARGET"
echo "Number of Trials: $N_TRIALS"
echo "Database: $DB_PATH"
echo "Study Name: $STUDY_NAME"
echo "============================================================================"

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

# Show GPU info
nvidia-smi

# --- Map array task ID to GPU (2 tasks per GPU) ---
GPU_ID=$((SLURM_ARRAY_TASK_ID % 7))
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Array Task ID: $SLURM_ARRAY_TASK_ID -> Using GPU: $CUDA_VISIBLE_DEVICES"

# --- Start Power Monitoring ---
POWER_LOG_DIR="./power_logs/optuna_isotherm_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
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

python run_optuna.py \
    --csv-file "$CSV_FILE" \
    --target "$TARGET" \
    --n-trials $N_TRIALS \
    --storage-url "$DB_PATH" \
    --study-name "$STUDY_NAME"

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
