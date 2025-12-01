#!/bin/bash
#SBATCH --job-name=optuna_mlp_journal
#SBATCH --output=slurm_jobs/optuna_mlp_journal_%j_%a.out
#SBATCH --error=slurm_jobs/optuna_mlp_journal_%j_%a.err
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
VENV_BIN=$(dirname "$VENV_PYTHON")
export VENV_PYTHON

# Crucial for GIL-free parallelization and preventing contention:
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
GLOBAL_RUN_DIR="runs/global_run_${SLURM_JOB_ID}"
mkdir -p slurm_jobs "$GLOBAL_RUN_DIR"

# --- Journal Storage Setup ---
JOURNAL_STORAGE_PATH="${GLOBAL_RUN_DIR}/optuna_journal_storage"
mkdir -p "$JOURNAL_STORAGE_PATH" 
echo "Using Optuna Journal Storage at: $JOURNAL_STORAGE_PATH"



# --- 2. Synchronous Tuning Execution (BLOCKING STEP) ---

echo "===================================================================="
echo "Starting 56 concurrent Optuna workers (one block, synchronous wait)."
echo "===================================================================="

# --- User-adjustable knobs ---
CSV_FILE=${CSV_FILE:-"./data/Depression_cones.csv"}
TARGET=${TARGET:-"Cone"}
TOTAL_TRIALS=${TOTAL_TRIALS:-10000}
TRIALS_PER_WORKER=$(( (TOTAL_TRIALS + 55) / 56 )) 
STUDY_NAME=${STUDY_NAME:-"depression_cones_mlp_journal_study"}

# Launch the 56 tasks. 
srun --ntasks=56 \
     bash -c '
         set -euo pipefail;
         
         # Worker-specific environment variables inside the srun job step
         PROC_ID=$SLURM_PROCID
         GPU_ID=$(( PROC_ID % 7 ))
         
         export CUDA_DEVICE_ID=$GPU_ID
         export CUDA_VISIBLE_DEVICES=$GPU_ID
         
         # Inherit shell variables from the submission script
         VENV_PYTHON="'"$VENV_PYTHON"'"
         JOURNAL_STORAGE_PATH="'"$JOURNAL_STORAGE_PATH"'"
         STUDY_NAME="'"$STUDY_NAME"'"
         CSV_FILE="'"$CSV_FILE"'"
         TARGET="'"$TARGET"'"
         TRIALS_PER_WORKER="'"$TRIALS_PER_WORKER"'"
         
         echo "[Worker $PROC_ID] Assigned GPU $GPU_ID. Starting $TRIALS_PER_WORKER trials."
         
         $VENV_PYTHON run_optuna.py \
             --csv-file "$CSV_FILE" \
             --target "$TARGET" \
             --storage-path "$JOURNAL_STORAGE_PATH" \
             --study-name "$STUDY_NAME" \
             --optuna-trials "$TRIALS_PER_WORKER" \
             --optuna-workers 1 \
             --run-tag "worker$PROC_ID"
             
         # CRITICAL FIX: Force flush kernel I/O buffers for journal.log immediately before process exit
         sync
             
         echo "[Worker $PROC_ID] Finished and Synced."
     '
# The master script now pauses here until all 56 workers exit gracefully.

# --- 3. Final Export and Cleanup (Master Task) ---
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "===================================================================="
    echo "All 56 workers finalized. Starting TensorBoard Export."
    
    # Secondary system sync and wait (redundant, but safe)
    echo "Final file system sync and wait (guaranteeing write completion)..."
    sync
    sleep 10 # Provide a large buffer for NFS/cache visibility
    
    # --- NO POLLING: The failure was due to attempting to read while writing.
    # We rely on the srun block + inner syncs + external wait buffer.
    
    echo "Tuning finished. Generating Optuna TensorBoard Summary..."
    
    OPTUNA_TB_DIR="${GLOBAL_RUN_DIR}/tensorboard_hpo_summary"
    
    # Use the custom utility to load the study and log results
    $VENV_PYTHON export_tb_summary.py \
        --storage-path "${JOURNAL_STORAGE_PATH}" \
        --study-name "${STUDY_NAME}" \
        --output-dir "${OPTUNA_TB_DIR}" 2>&1 || true
    
    echo ""
    echo "HPO FINISHED: Total Trials: $TOTAL_TRIALS (Approx)."
    echo "--------------------------------------------------------------------"
    echo "TensorBoard Logs Path: ${GLOBAL_RUN_DIR}"
    echo "To view REAL-TIME GPU/CPU Usage + FINAL HPO Trial Metrics:"
    echo "tensorboard --logdir=${GLOBAL_RUN_DIR}"
    echo "===================================================================="
fi