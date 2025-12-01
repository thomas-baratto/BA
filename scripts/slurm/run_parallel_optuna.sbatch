#!/bin/bash
#SBATCH --job-name=optuna_parallel
#SBATCH --output=./slurm_jobs/optuna_parallel_%j.out
#SBATCH --error=./slurm_jobs/optuna_parallel_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH -w argon-gtx

# ============================================================================
# Parallel Optuna Optimization - 16 background processes sharing 7 GPUs
# ============================================================================

# Parameters
CSV_FILE="./data/Clean_Results_Isotherm.csv"
TARGET="all"
N_TRIALS=320
DB_PATH="sqlite:///optuna_study_isotherm.db"
STUDY_NAME="nn_study_Clean_Results_Isotherm"

if [ -n "$OPTUNA_STORAGE_URL" ]; then
    DB_PATH="$OPTUNA_STORAGE_URL"
    echo "Using OPTUNA_STORAGE_URL override"
elif [ -n "$STORAGE_URL" ]; then
    DB_PATH="$STORAGE_URL"
    echo "Using STORAGE_URL override"
fi

# Setup
mkdir -p ./slurm_jobs
mkdir -p ./logs
mkdir -p ./power_logs

# Load modules
module purge
module load cuda/12.4.1

# Activate virtualenv
source /home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona/bin/activate

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

echo "============================================================================"
echo "SLURM Job Information"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Number of tasks: $SLURM_NTASKS"
echo "Dataset: Clean Results Isotherm"
echo "CSV File: $CSV_FILE"
echo "Target: $TARGET"
echo "Number of Trials: $N_TRIALS"
echo "Database: $DB_PATH"
echo "Study Name: $STUDY_NAME"
echo "============================================================================"

# Show GPU info
nvidia-smi

# --- Start Power Monitoring ---
POWER_LOG_DIR="./power_logs/optuna_parallel_job_${SLURM_JOB_ID}"
POWER_INTERVAL=2.0
echo "Starting power monitoring..."
echo "Power logs will be saved to: $POWER_LOG_DIR"

python monitoring/power_monitor.py \
    --output-dir "$POWER_LOG_DIR" \
    --interval "$POWER_INTERVAL" \
    --filter python &

POWER_MONITOR_PID=$!
echo "Power monitor started with PID: $POWER_MONITOR_PID"
sleep 2

echo "============================================================================"
echo "Starting 16 parallel Optuna workers..."
echo "============================================================================"

# Launch 16 workers in parallel as background processes
for i in {0..15}; do
    # Map task to GPU (2-3 tasks per GPU)
    GPU_ID=$((i % 7))
    LOG_FILE="./logs/optuna_worker_${i}.log"
    
    echo "Launching worker $i on GPU $GPU_ID"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/run_optuna.py \
        --csv-file "$CSV_FILE" \
        --target "$TARGET" \
        --optuna-trials $N_TRIALS \
        --storage-url "$DB_PATH" \
        --study-name "$STUDY_NAME" \
        --optuna-workers 1 \
        --run-tag "worker_${i}" > "$LOG_FILE" 2>&1 &
    
    # Small delay to stagger startup
    sleep 1
done

echo "All workers launched. Waiting for completion..."
wait

# --- Stop Power Monitoring ---
echo "============================================================================"
echo "Stopping power monitoring..."
echo "============================================================================"
kill -SIGINT $POWER_MONITOR_PID
wait $POWER_MONITOR_PID 2>/dev/null

echo "Power logs saved to: $POWER_LOG_DIR"

echo "============================================================================"
echo "All workers finished!"
echo "============================================================================"
