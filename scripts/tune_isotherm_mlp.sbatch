#!/bin/bash
#SBATCH --job-name=optuna_mlp_iso_56x
#SBATCH --output=slurm_jobs/optuna_mlp_iso_%j_%a.out
#SBATCH --error=slurm_jobs/optuna_mlp_iso_%j_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=56          # CRITICAL: 56 independent worker processes (tasks)
#SBATCH --cpus-per-task=1    # CRITICAL: 1 dedicated CPU core per process (total 56 cores used)
#SBATCH --gpus=7             # Request all 7 GPUs for 56 tasks to share
#SBATCH --time=2-00:00:00
#SBATCH --mem=240G
#SBATCH -w argon-gtx

# --- Environment setup ---
set -euo pipefail

module purge
module load cuda/12.4.1 || true
source /home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona/bin/activate
VENV_PYTHON="${VENV_PYTHON:-$(which python 2>/dev/null || echo /home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona/bin/python)}"
export VENV_PYTHON

# Crucial for preventing contention on the single allocated CPU core per task:
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
GLOBAL_RUN_DIR="runs/global_run_${SLURM_JOB_ID}"
mkdir -p slurm_jobs "$GLOBAL_RUN_DIR"

# --- 1. Global Resource Monitoring Daemon (Only starts on Task 0) ---
# This background process tracks the usage of all 7 physical GPUs independently of the workers.
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "Starting global resource monitor on all 7 GPUs..."
    GPU_MONITOR_LOG="${GLOBAL_RUN_DIR}/gpu_monitor.log"
    
    # Logs (comma separated): Unix Timestamp, GPU Index, Utilization (%), Memory Used (MiB), Memory Total (MiB)
    (while true; do 
        # The output is joined by semicolons for each GPU (e.g., 0,50,4000;1,75,6000;...)
        echo "$(date +%s), $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,nounits,noheader | tr '\n' ';')" >> "$GPU_MONITOR_LOG"
        sleep 5
    done) &
    GPU_MONITOR_PID=$!
    echo "GPU Monitor PID: $GPU_MONITOR_PID, Log: $GPU_MONITOR_LOG"
    
    # Ensure the monitor is killed when the job finishes (even on failure)
    trap "echo 'Killing GPU monitor PID $GPU_MONITOR_PID'; kill $GPU_MONITOR_PID" EXIT
fi

# --- 2. CRITICAL GPU CORE MAPPING LOGIC (Runs on all 56 tasks) ---
# This pins 8 Python processes to each physical GPU (56 tasks / 7 GPUs = 8 workers/GPU).
export CUDA_DEVICE_ID=$(( $SLURM_PROCID % 7 ))
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID

# --- User-adjustable knobs ---
CSV_FILE=${CSV_FILE:-"./data/Clean_Results_Isotherm.csv"}
TARGET=${TARGET:-"all"}
TOTAL_TRIALS=${TOTAL_TRIALS:-10000}
TRIALS_PER_WORKER=$(( (TOTAL_TRIALS + 55) / 56 )) 

# Using RDB for distributed coordination
DB_URL=${DB_URL:-"mysql+mysqlconnector://root:strong_password@127.0.0.1:3307/optuna_study?ssl=false"}
STUDY_NAME=${STUDY_NAME:-"nn_study_isotherm_bulk"}

# --- Diagnostics for current worker ---
echo "===================================================================="
echo "Optuna HPO Worker $SLURM_PROCID (PID: $$) is launching."
echo "Assigned GPU Index: $CUDA_DEVICE_ID (8 workers sharing this physical GPU)"
echo "Worker's Trial Share: $TRIALS_PER_WORKER"
echo "===================================================================="

# --- 3. Execute the Optuna Worker Script ---
# IMPORTANT: The Python scripts (run_optuna.py and optuna_objective.py) MUST 
# be updated to use DataLoader(num_workers=0) and study.optimize(n_jobs=1) 
# to prevent oversubscribing the single CPU core allocated per task.
srun --ntasks=1 --exclusive $VENV_PYTHON run_optuna.py \
    --csv-file "${CSV_FILE}" \
    --target "${TARGET}" \
    --storage-url "${DB_URL}" \
    --study-name "${STUDY_NAME}" \
    --optuna-trials "${TRIALS_PER_WORKER}" \
    --optuna-workers 1 \
    --run-tag "worker${SLURM_PROCID}"

echo "Worker $SLURM_PROCID finished."

# --- 4. Final job completion and TensorBoard Generation (Only Task 0) ---
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "Waiting for all 56 workers to finalize..."
    # Sleep briefly to ensure the monitor catches the very end of the job
    sleep 10 
    
    echo "===================================================================="
    echo "Tuning finished. Generating TensorBoard summary from RDB..."
    
    # Generate TensorBoard logs of the Optuna trials (loss, params)
    $VENV_PYTHON -m optuna dashboard --storage "$DB_URL" --study "$STUDY_NAME" --export-tensorboard "$GLOBAL_RUN_DIR/tensorboard_hpo" 2>&1 || true
    
    echo ""
    echo "HPO FINISHED: Total Trials: $TOTAL_TRIALS (Approx)."
    echo "--------------------------------------------------------------------"
    echo "Raw System GPU Usage Log: ${GPU_MONITOR_LOG}"
    echo "To view HPO results (TPE progress, loss history, parameter importance):"
    echo "tensorboard --logdir=${GLOBAL_RUN_DIR}/tensorboard_hpo"
    echo "===================================================================="
fi