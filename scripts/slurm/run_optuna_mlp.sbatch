#!/bin/bash
#SBATCH --job-name=optuna_mlp
#SBATCH --output=slurm_jobs/optuna_mlp_%j_%a.out
#SBATCH --error=slurm_jobs/optuna_mlp_%j_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --cpus-per-task=1
#SBATCH --gpus=7
#SBATCH --time=2-00:00:00
#SBATCH --mem=240G
#SBATCH -w argon-gtx

set -euo pipefail

module purge
module load cuda/12.4.1 || true
source /home/barattts/lavoltabuona/BA/.venv/env/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_jobs

GLOBAL_RUN_DIR="runs/global_run_${SLURM_JOB_ID}"
JOURNAL_STORAGE_PATH="${GLOBAL_RUN_DIR}/optuna_journal_storage"
mkdir -p "$JOURNAL_STORAGE_PATH"

CSV_FILE=${CSV_FILE:-"./data/Clean_Results_Isotherm.csv"}
TARGET=${TARGET:-"all"}
TOTAL_TRIALS=${TOTAL_TRIALS:-10000}
STUDY_NAME=${STUDY_NAME:-"nn_study_mlp_journal"}
USE_AREA_ROOT=${USE_AREA_ROOT:-"0"}
TRIALS_PER_WORKER=$(( (TOTAL_TRIALS + 55) / 56 ))

AREA_ROOT_FLAG=""
if [ "$USE_AREA_ROOT" = "1" ]; then
    AREA_ROOT_FLAG="--use-area-root"
fi

echo "Starting Optuna MLP tuning"
echo "Study: $STUDY_NAME"
echo "CSV: $CSV_FILE"
echo "Target: $TARGET"
echo "Trials total: $TOTAL_TRIALS"
echo "Use sqrt(Area): $USE_AREA_ROOT"

srun --ntasks=56 bash -c '
    set -euo pipefail

    PROC_ID=$SLURM_PROCID
    GPU_ID=$(( PROC_ID % 7 ))

    export CUDA_DEVICE_ID=$GPU_ID
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    python scripts/training/run_optuna.py \
        --csv-file "'"$CSV_FILE"'" \
        --target "'"$TARGET"'" \
        --storage-path "'"$JOURNAL_STORAGE_PATH"'" \
        --study-name "'"$STUDY_NAME"'" \
        --optuna-trials "'"$TRIALS_PER_WORKER"'" \
        --optuna-workers 1 \
        --run-tag "worker${PROC_ID}" \
        '"$AREA_ROOT_FLAG"'

    sync
'

echo "Generating Optuna TensorBoard summary"
python optimization/export_tb_summary.py \
    --storage-path "$JOURNAL_STORAGE_PATH" \
    --study-name "$STUDY_NAME" \
    --output-dir "${GLOBAL_RUN_DIR}/tensorboard_hpo_summary" || true

echo "Done. Journal path: ${JOURNAL_STORAGE_PATH}/journal.log"
