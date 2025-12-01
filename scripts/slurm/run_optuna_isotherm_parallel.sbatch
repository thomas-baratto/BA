#!/bin/bash
#SBATCH --job-name=optuna_iso_bulk
#SBATCH --output=slurm_jobs/optuna_iso_bulk_%j.out
#SBATCH --error=slurm_jobs/optuna_iso_bulk_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:7
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH -w argon-gtx

# Massive Optuna tuning for Clean_Results_Isotherm.csv
# - Launches one worker per GPU (max 7) to stay within SQLite's connection limits
# - Each worker runs `run_optuna.py` on a disjoint subset of trials
# - After all workers finish, launch `train_final_model.py` separately to materialize weights

set -euo pipefail


# --- Environment setup ---
module purge
module load cuda/12.4.1 || true
source /home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona/bin/activate
# Prefer the venv python. Don't call /usr/bin/python (that bypasses the venv).
# Allow overriding via env `VENV_PYTHON` when submitting the job.
VENV_PYTHON="${VENV_PYTHON:-$(which python 2>/dev/null || echo /home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona/bin/python)}"
export VENV_PYTHON
echo "Activated venv; using python: $($VENV_PYTHON -V 2>&1 || echo 'python not found')"
# Quick optuna import check to make failures obvious in the sbatch .err file
if ! $VENV_PYTHON -c "import optuna" >/dev/null 2>&1; then
    echo "Warning: 'optuna' not importable from $VENV_PYTHON" >&2
    echo "You can install it with: $VENV_PYTHON -m pip install optuna" >&2
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_jobs runs

# --- User-adjustable knobs (override via env vars before sbatch) ---
CSV_FILE=${CSV_FILE:-"./data/Clean_Results_Isotherm.csv"}
TARGET=${TARGET:-"all"}
TOTAL_TRIALS=${TOTAL_TRIALS:-10000}
WORKER_COUNT=${WORKER_COUNT:-7}  # One worker per GPU to minimize SQLite contention
OPTUNA_NJOBS=${OPTUNA_NJOBS:-4}  # Trials per worker (1=sequential, 2+=parallel threads)
CPUS_PER_WORKER=${CPUS_PER_WORKER:-4}
POWER_INTERVAL=${POWER_INTERVAL:-1.0}
STUDY_NAME=${STUDY_NAME:-"nn_study_isotherm_bulk"}
DB_BASENAME=${DB_BASENAME:-"optuna_isotherm_bulk.db"}
DEFAULT_DB_URL="sqlite:///runs/${DB_BASENAME}"
# Allow RDB overrides via OPTUNA_STORAGE_URL / DB_URL / STORAGE_URL
if [ -n "${OPTUNA_STORAGE_URL:-}" ]; then
    DB_URL="$OPTUNA_STORAGE_URL"
elif [ -n "${DB_URL:-}" ]; then
    DB_URL="$DB_URL"
elif [ -n "${STORAGE_URL:-}" ]; then
    DB_URL="$STORAGE_URL"
else
    DB_URL="$DEFAULT_DB_URL"
fi

GPU_IDS=(0 1 2 3 4 5 6)
if (( WORKER_COUNT > ${#GPU_IDS[@]} )); then
    WORKER_COUNT=${#GPU_IDS[@]}
fi
TRIALS_PER_WORKER=$(( (TOTAL_TRIALS + WORKER_COUNT - 1) / WORKER_COUNT ))


# Ensure SQLite backends live on a shared path and enable WAL to reduce writer lock times.
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
import sqlite3
import os

db_file = os.environ['DB_FILE']
conn = sqlite3.connect(db_file)
conn.execute('PRAGMA journal_mode=WAL;')
conn.close()
PY
fi

cat <<EOM
====================================================================
Optuna bulk tuning job
====================================================================
Node: $SLURMD_NODENAME
GPUs requested: ${GPU_IDS[*]}
Workers: $WORKER_COUNT (trials/worker: $TRIALS_PER_WORKER)
Total target trials: $TOTAL_TRIALS
Optuna storage: $DB_URL
Study name: $STUDY_NAME
CSV file: $CSV_FILE
Target labels: $TARGET
====================================================================
EOM

nvidia-smi

declare -a worker_pids=()
for idx in $(seq 0 $((WORKER_COUNT - 1))); do
    gpu_id=${GPU_IDS[$idx]}
    run_tag="worker${idx}"
    echo "[Launch] Worker ${idx} -> GPU ${gpu_id}, ${TRIALS_PER_WORKER} trials"
    srun --exclusive -N1 -n1 --cpus-per-task=${CPUS_PER_WORKER} --gres=gpu:1 \
        bash -c "set -euo pipefail; export CUDA_VISIBLE_DEVICES=${gpu_id}; \
        $VENV_PYTHON run_optuna.py \
            --csv-file ${CSV_FILE} \
            --target ${TARGET} \
            --storage-url ${DB_URL} \
            --study-name ${STUDY_NAME} \
            --optuna-trials ${TRIALS_PER_WORKER} \
            --optuna-workers ${OPTUNA_NJOBS} \
            --run-tag ${run_tag} \
            --disable-power-monitor \
            --power-interval ${POWER_INTERVAL} \
            --power-filter python" &
    worker_pids+=("$!")
    sleep 2  # small stagger to reduce contention at start
done

echo "Waiting for all tuning workers to finish..."
for pid in "${worker_pids[@]}"; do
    wait "$pid"
done

echo "===================================================================="
echo "Bulk tuning finished. Best study stored at: ${DB_URL}"
echo "Inspect with: optuna-dashboard ${DB_URL}"
echo "To materialize the final model, run: train_final_model.py --study-name ${STUDY_NAME} --storage-url ${DB_URL}"
echo ""
echo "TensorBoard logs disabled (objective function configured without logging)."
echo "To view worker run metadata: ls -la runs/run_*_worker*/"
echo "====================================================================="
