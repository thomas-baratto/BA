#!/bin/bash
#SBATCH --job-name=optuna_depression
#SBATCH --output=./slurm_jobs/optuna_depression_%j.out
#SBATCH --error=./slurm_jobs/optuna_depression_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:3
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -w argon-gtx

# ============================================================================
# Optuna Hyperparameter Tuning for Depression Cones Dataset (n_jobs mode)
# ============================================================================
# Uses 3 GPUs with n_jobs=3 for parallel trials within single job
# ============================================================================

# --- Parameters ---
CSV_FILE="./data/Depression_cones.csv"
TARGET="Cone"
N_TRIALS=${N_TRIALS:-320}
N_JOBS=3
POWER_INTERVAL=${POWER_INTERVAL:-2.0}
DB_PATH="sqlite:///optuna_study_depression.db"
STUDY_NAME="nn_study_Depression_cones_Cone"

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
echo "Node: $SLURMD_NODENAME"
echo "Dataset: Depression Cones"
echo "CSV File: $CSV_FILE"
echo "Target: $TARGET"
echo "Number of Trials: $N_TRIALS"
echo "Parallel Jobs: $N_JOBS"
echo "Database: $DB_PATH"
echo "Study Name: $STUDY_NAME"
echo "============================================================================"

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

# Show GPU info
nvidia-smi

# --- Set CUDA_VISIBLE_DEVICES to use GPUs 0-2 ---
export CUDA_VISIBLE_DEVICES=0,1,2
export OPTUNA_N_JOBS=$N_JOBS
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Optuna n_jobs: $OPTUNA_N_JOBS"

# --- Start Power Monitoring ---
POWER_LOG_DIR="./power_logs/optuna_depression_job_${SLURM_JOB_ID}"
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
echo "Starting Optuna optimization with n_jobs=$N_JOBS..."
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
