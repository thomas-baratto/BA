#!/bin/bash

# --- Job Configuration ---
#SBATCH --job-name="random_sweep"
#SBATCH --output=./slurm_jobs/slurm-sweep-%j.out
#SBATCH -w argon-gtx

# --- Resource Allocation ---
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:7
#SBATCH --mem=240G
#SBATCH --time=0-12:00:00           # 12 hours — sweep is longer

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
BASE_RUN_DIR="runs/run_sweep_random_${JOB_ID_STR}"
mkdir -p "$BASE_RUN_DIR"

echo "Results will be saved in: $BASE_RUN_DIR"

# ============================================================
# Fixed parameters (not swept)
# ============================================================
FEATURE_SCALER=robust
LABEL_SCALER=robust
RSC_PROB=0.3
MAX_PARALLEL=7          # One job per GPU

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

    COMMON="--feature-scaler $FEATURE_SCALER --label-scaler $LABEL_SCALER --base-dir \"$BASE_RUN_DIR\" --no-save-model $EXTRA"

    for act in "${ACTIVATIONS[@]}"; do
        ACT_FLAG="--activation $act"

        # --- ELM: sweep n_hidden only ---
        for nh in "${HIDDEN_SIZES[@]}"; do
            commands+=("python scripts/train_random_models.py --model ELM --dataset $dataset --n-hidden $nh $ACT_FLAG $COMMON")
        done

        # --- dRVFL: sweep n_hidden × n_layers ---
        for nh in "${HIDDEN_SIZES[@]}"; do
            for nl in "${LAYER_COUNTS[@]}"; do
                commands+=("python scripts/train_random_models.py --model dRVFL --dataset $dataset --n-hidden $nh --n-layers $nl $ACT_FLAG $COMMON")
            done
        done

        # --- Ensemble models: sweep n_hidden × n_layers × n_ensemble ---
        for model in "${MODELS_ENSEMBLE[@]}"; do
            SC_FLAGS=""
            if [ "$model" = "edRVFL-SC" ]; then
                SC_FLAGS="--sc-mode dense"
            fi

            for nh in "${HIDDEN_SIZES[@]}"; do
                for nl in "${LAYER_COUNTS[@]}"; do
                    for ne in "${ENSEMBLE_COUNTS[@]}"; do
                        commands+=("python scripts/train_random_models.py --model $model --dataset $dataset --n-hidden $nh --n-layers $nl --n-ensemble $ne $SC_FLAGS $ACT_FLAG $COMMON")
                    done
                done
            done
        done

        # --- SResdRVFL: sweep n_hidden × n_blocks (1 layer per block, direct link) ---
        for nh in "${HIDDEN_SIZES[@]}"; do
            for nb in "${BLOCK_COUNTS[@]}"; do
                commands+=("python scripts/train_random_models.py --model SResdRVFL --dataset $dataset --n-hidden $nh --n-layers 1 --n-blocks $nb --direct-link $ACT_FLAG $COMMON")
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
# Launch in batches of MAX_PARALLEL
# ============================================================
running=0
for i in "${!commands[@]}"; do
    GPU_ID=$((running % MAX_PARALLEL))
    echo "[${i}/${TOTAL}] GPU $GPU_ID: ${commands[$i]}"
    CUDA_VISIBLE_DEVICES=$GPU_ID eval "${commands[$i]}" &
    running=$((running + 1))

    # When we hit MAX_PARALLEL, wait for the batch to finish
    if [ $running -ge $MAX_PARALLEL ]; then
        wait
        running=0
        echo "--- Batch complete, starting next batch ---"
    fi

    sleep 1
done

# Wait for any remaining jobs
wait

echo "======================================"
echo "All $TOTAL sweep runs completed."
echo "======================================"

echo "Generating summary table..."
python scripts/summarize_results.py --run-dir "$BASE_RUN_DIR"

echo "Sweep finished. Results in: $BASE_RUN_DIR"
