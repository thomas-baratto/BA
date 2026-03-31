#!/bin/bash

# --- Job Configuration ---
#SBATCH --job-name="random_sweep"
#SBATCH --output=./slurm_jobs/slurm-sweep-%j.out
#SBATCH -w argon-gtx

# --- Resource Allocation ---
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:7
#SBATCH --mem=350G
#SBATCH --time=0-03:00:00           # 3 hours — full-parallel launcher

# --- Environment Setup ---
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "---"

mkdir -p ./slurm_jobs

module purge
module load cuda/12.4.1

source /home/barattts/lavoltabuona/BA/.venv/env/bin/activate

PROJECT_ROOT="/home/barattts/lavoltabuona/BA"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

JOB_ID_STR=${SLURM_JOB_ID:-"manual_$(date +%Y%m%d_%H%M%S)"}
# Use node-local scratch (fast SSD, avoids filling NFS home)
SCRATCH_DIR="/data/scratch/barattts"
mkdir -p "$SCRATCH_DIR"
BASE_RUN_DIR="${SCRATCH_DIR}/run_sweep_random_${JOB_ID_STR}"
mkdir -p "$BASE_RUN_DIR"
# Where to copy final results back on NFS (visible from login node)
FINAL_RUN_DIR="${PROJECT_ROOT}/runs/run_sweep_random_${JOB_ID_STR}"

echo "Results will be saved in: $BASE_RUN_DIR (node-local scratch)"
echo "Final results will be copied to: $FINAL_RUN_DIR"
echo "Mode: --no-save-model (metrics only, retrain winners separately)"

# ============================================================
# Fixed parameters (not swept)
# ============================================================
FEATURE_SCALER=robust
LABEL_SCALER=robust
RSC_PROB=0.3
MAX_PARALLEL=7          # One job per GPU (7 working GPUs on argon-gtx)

# ============================================================
# Sweep ranges — edit these to control the sweep
# ============================================================
HIDDEN_SIZES=(500 1000 1500)
LAYER_COUNTS=(1 2 3)
BLOCK_COUNTS=(3 5 8)
ENSEMBLE_COUNTS=(10 20)
ACTIVATIONS=(ReLU GELU ELU)

# Models to sweep (comment out any you want to skip)
MODELS_SIMPLE=(ELM)                             # Only n_hidden matters
MODELS_DEEP=(dRVFL)                             # n_hidden × n_layers
MODELS_ENSEMBLE=(edRVFL edRVFL-SC esc-edRVFL)   # n_hidden × n_layers × n_ensemble
MODELS_RESIDUAL=(SResdRVFL)                     # n_hidden × n_blocks

DATASETS=(cone isotherm)

# ============================================================
# Build the full command list
# ============================================================
declare -a commands

for dataset in "${DATASETS[@]}"; do
    # Extra flags per dataset
    EXTRA=""
    if [ "$dataset" = "isotherm" ]; then
        EXTRA="--use-area-root"
    fi

    COMMON="--no-save-model --disable-power-monitor --feature-scaler $FEATURE_SCALER --label-scaler $LABEL_SCALER --base-dir \"$BASE_RUN_DIR\" $EXTRA"

    for act in "${ACTIVATIONS[@]}"; do
        ACT_FLAG="--activation $act"

        # --- ELM: sweep n_hidden only ---
        for nh in "${HIDDEN_SIZES[@]}"; do
            commands+=("python scripts/training/train_random_models.py --model ELM --dataset $dataset --n-hidden $nh $ACT_FLAG $COMMON")
        done

        # --- dRVFL: sweep n_hidden × n_layers ---
        for nh in "${HIDDEN_SIZES[@]}"; do
            for nl in "${LAYER_COUNTS[@]}"; do
                commands+=("python scripts/training/train_random_models.py --model dRVFL --dataset $dataset --n-hidden $nh --n-layers $nl $ACT_FLAG $COMMON")
            done
        done

        # --- Ensemble models: sweep n_hidden × n_layers × n_ensemble ---
        for model in "${MODELS_ENSEMBLE[@]}"; do
            if [ "$model" = "edRVFL-SC" ]; then
                # Sweep both skip-connection modes for edRVFL-SC
                SC_MODES=(dense random)
            else
                SC_MODES=("")
            fi

            for sc_mode in "${SC_MODES[@]}"; do
                SC_FLAGS=""
                if [ -n "$sc_mode" ]; then
                    SC_FLAGS="--sc-mode $sc_mode"
                fi

                for nh in "${HIDDEN_SIZES[@]}"; do
                    for nl in "${LAYER_COUNTS[@]}"; do
                        for ne in "${ENSEMBLE_COUNTS[@]}"; do
                            commands+=("python scripts/training/train_random_models.py --model $model --dataset $dataset --n-hidden $nh --n-layers $nl --n-ensemble $ne $SC_FLAGS $ACT_FLAG $COMMON")
                        done
                    done
                done
            done
        done

        # --- SResdRVFL: sweep n_hidden × n_blocks (1 layer per block, direct link) ---
        for nh in "${HIDDEN_SIZES[@]}"; do
            for nb in "${BLOCK_COUNTS[@]}"; do
                commands+=("python scripts/training/train_random_models.py --model SResdRVFL --dataset $dataset --n-hidden $nh --n-layers 1 --n-blocks $nb --direct-link $ACT_FLAG $COMMON")
            done
        done
    done
done

TOTAL=${#commands[@]}
echo "======================================"
echo "Parameter Sweep: $TOTAL total runs"
echo "  Hidden sizes:    ${HIDDEN_SIZES[*]}"
echo "  Layer counts:    ${LAYER_COUNTS[*]}"
echo "  Block counts:    ${BLOCK_COUNTS[*]}"
echo "  Ensemble counts: ${ENSEMBLE_COUNTS[*]}"
echo "  Activations:     ${ACTIVATIONS[*]}"
echo "  Datasets:        ${DATASETS[*]}"
echo "======================================"

# ============================================================
# Launch parallel workers (no model saving — metrics only)
# Power monitor runs as a single job-level background process
# (per-run monitors are disabled — runs are too short for useful data)
# ============================================================
echo "Deploying $TOTAL sweep runs (--no-save-model, metrics only)..."

export KEEP_RUN_ARTIFACTS=0

POWER_LOG_DIR="${BASE_RUN_DIR}/power_monitor"
mkdir -p "$POWER_LOG_DIR"
echo "Starting job-level power monitor → $POWER_LOG_DIR"
python monitoring/power_monitor.py \
    --output-dir "$POWER_LOG_DIR" \
    --interval 1.0 \
    --filter python &
POWER_PID=$!

python scripts/sweep/launch_sweep_workers.py $MAX_PARALLEL "${commands[@]}"

echo "Stopping power monitor (PID=$POWER_PID)..."
kill $POWER_PID 2>/dev/null && wait $POWER_PID 2>/dev/null || true

echo "======================================"
echo "All $TOTAL sweep runs completed."
echo "======================================"

# ── Summarize ─────────────────────────────────────────────────
echo "Generating summary table..."
python scripts/analysis/summarize_results.py --run-dir "$BASE_RUN_DIR"

# ── Select knee-point winners (4 models) ─────────────────────
echo ""
echo "Selecting knee-point winners from Pareto frontiers..."
python scripts/analysis/select_knee_points.py \
    --summary-csv "$BASE_RUN_DIR/summary_table.csv" \
    --run-dir "$BASE_RUN_DIR" \
    --retrain-base "$FINAL_RUN_DIR" \
    --output-csv "$BASE_RUN_DIR/knee_point_winners.csv"

# ── Copy lightweight results to NFS ──────────────────────────
echo ""
echo "Copying lightweight results to NFS..."
mkdir -p "$FINAL_RUN_DIR"

# Copy summary table and knee point winners
cp "$BASE_RUN_DIR/summary_table.csv" "$FINAL_RUN_DIR/"
cp "$BASE_RUN_DIR/knee_point_winners.csv" "$FINAL_RUN_DIR/"

# Copy power monitor data
if [ -d "$BASE_RUN_DIR/power_monitor" ]; then
    cp -r "$BASE_RUN_DIR/power_monitor" "$FINAL_RUN_DIR/"
    echo "Copied power_monitor data to $FINAL_RUN_DIR/power_monitor/"
fi

# Copy only the JSON result files (tiny) from each run subfolder
find "$BASE_RUN_DIR" -mindepth 2 -maxdepth 2 -name "results_*.json" | while read src; do
    rel=$(realpath --relative-to="$BASE_RUN_DIR" "$src")
    dst="$FINAL_RUN_DIR/$rel"
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
done

echo "Copied to: $FINAL_RUN_DIR"
du -sh "$FINAL_RUN_DIR"

# ── Retrain the 4 frontier winners (with model saving) ──────
echo ""
echo "Retraining frontier winners (with full artifact saving)..."

while IFS=',' read -r dataset frontier model folder time nrmse kge fsize cmd; do
    # Skip header
    [ "$dataset" = "dataset" ] && continue

    echo ""
    echo "  Retraining: $dataset / $frontier → $model"
    echo "    CMD: $cmd"
    eval "$cmd"
done < "$BASE_RUN_DIR/knee_point_winners.csv"

echo ""
echo "======================================"
echo "Sweep + retrain complete."
echo "======================================"

# ── Cleanup scratch ──────────────────────────────────────────
echo "Cleaning up node-local scratch..."
rm -rf "$BASE_RUN_DIR"

echo ""
echo "Results in: $FINAL_RUN_DIR"
echo "  - summary_table.csv    : metrics for ALL ${TOTAL} models"
echo "  - knee_point_winners.csv: the 4 selected frontier winners"
echo "  - <dataset>_<frontier>_winner/  : retrained model artifacts"
echo ""
echo "Next steps:"
echo "  PYTHONPATH=. python scripts/analysis/select_knee_points.py --summary-csv $FINAL_RUN_DIR/summary_table.csv --run-dir $FINAL_RUN_DIR"
