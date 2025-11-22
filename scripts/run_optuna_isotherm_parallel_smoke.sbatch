#!/bin/bash
#SBATCH --job-name=optuna_iso_smoke
#SBATCH --output=slurm_jobs/optuna_iso_smoke_%j.out
#SBATCH --error=slurm_jobs/optuna_iso_smoke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:7
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH -w argon-gtx

# Lightweight dry-run that mirrors the parallel setup from run_optuna_isotherm_parallel.sbatch
# Launches up to 2 workers (1 GPU each) for a handful of trials, then trains the final model

set -euo pipefail

VENV_PATH=${VENV_PATH:-"/home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona"}

CSV_FILE=${CSV_FILE:-"./data/Clean_Results_Isotherm.csv"}
TARGET=${TARGET:-"all"}
CPUS_PER_WORKER=${CPUS_PER_WORKER:-4}
STUDY_NAME=${STUDY_NAME:-"nn_study_isotherm_smoke"}
DB_BASENAME=${DB_BASENAME:-"optuna_isotherm_smoke.db"}
DEFAULT_DB_URL="sqlite:///runs/${DB_BASENAME}"

if [ -n "${OPTUNA_STORAGE_URL:-}" ]; then
    DB_URL="$OPTUNA_STORAGE_URL"
elif [ -n "${DB_URL:-}" ]; then
    DB_URL="$DB_URL"
elif [ -n "${STORAGE_URL:-}" ]; then
    DB_URL="$STORAGE_URL"
else
    DB_URL="$DEFAULT_DB_URL"
fi
export DB_URL STUDY_NAME
GPU_IDS=(0 1 2 3 4 5 6)
DEFAULT_WORKER_COUNT=${#GPU_IDS[@]}
if [[ -z "${WORKER_COUNT:-}" ]] || (( WORKER_COUNT <= 0 )); then
    WORKER_COUNT=$DEFAULT_WORKER_COUNT
fi
if (( WORKER_COUNT > ${#GPU_IDS[@]} )); then
    WORKER_COUNT=${#GPU_IDS[@]}
fi
TRIALS_PER_WORKER=1
TOTAL_TRIALS=$((WORKER_COUNT * TRIALS_PER_WORKER))

module purge
source "$VENV_PATH/bin/activate"
module load cuda/12.4.1 || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_jobs runs

case "$DB_URL" in
    sqlite:////*)
        DB_FILE="/${DB_URL#sqlite:////}"
        ;;
    sqlite:///*)
        DB_FILE="$SLURM_SUBMIT_DIR/${DB_URL#sqlite:///}"
        ;;
    *)
        DB_FILE=""
        ;;
esac

if [[ -n "$DB_FILE" ]]; then
    DB_DIR="$(dirname "$DB_FILE")"
    mkdir -p "$DB_DIR"
    export DB_FILE
    python - <<'PY'
import os
import sqlite3

db_file = os.environ['DB_FILE']
conn = sqlite3.connect(db_file)
conn.execute('PRAGMA journal_mode=WAL;')
conn.close()
PY
fi

python - <<'PY'
import os
import optuna

storage = os.environ['DB_URL']
study_name = os.environ['STUDY_NAME']
try:
    optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True,
    )
except Exception as exc:  # pragma: no cover
    print(f"[WARN] Unable to pre-create Optuna study {study_name}: {exc}")
PY

cat <<EOM
================ SMOKE TEST ================
Study: $STUDY_NAME
Storage: $DB_URL
 Trials total: $TOTAL_TRIALS (per worker: $TRIALS_PER_WORKER)
 CSV: $CSV_FILE
 Target: $TARGET
============================================
EOM

nvidia-smi

declare -a worker_pids=()
for idx in $(seq 0 $((WORKER_COUNT - 1))); do
    gpu_id=${GPU_IDS[$idx]}
    run_tag="smoke_w${idx}"
    echo "[Smoke] Worker ${idx} -> GPU ${gpu_id}, ${TRIALS_PER_WORKER} trials"
    srun --exclusive -N1 -n1 --cpus-per-task=${CPUS_PER_WORKER} --gres=gpu:1 \
        bash -c "set -euo pipefail; source ${VENV_PATH}/bin/activate; export CUDA_VISIBLE_DEVICES=${gpu_id}; \
        python run_optuna.py \\
            --csv-file ${CSV_FILE} \\
            --target ${TARGET} \\
            --storage-url ${DB_URL} \\
            --study-name ${STUDY_NAME} \\
            --optuna-trials ${TRIALS_PER_WORKER} \\
            --optuna-workers 1 \\
            --run-tag ${run_tag} \\
            --disable-power-monitor" &
    worker_pids+=("$!")
    sleep 1
done

for pid in "${worker_pids[@]}"; do
    wait "$pid"
done

echo "Smoke job finished. Inspect optuna-dashboard ${DB_URL}"
