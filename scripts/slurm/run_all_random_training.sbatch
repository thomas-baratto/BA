#!/bin/bash

# --- Job Configuration ---
#SBATCH --job-name="random_training_"                  # Name of job
#SBATCH --output=./slurm_jobs/slurm-random-%j.out  # Output file, %j expands to job ID
#SBATCH -w argon-gtx                            # Request the specific node 'argon-gtx'

# --- Resource Allocation ---
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Single task
#SBATCH --cpus-per-task=28          # 28 CPUs
#SBATCH --gres=gpu:7                # Request 7 GPUs
#SBATCH --mem=240G                  # Memory allocation
#SBATCH --time=0-04:00:00           # 4 hours max

# --- Environment Setup ---
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "---"

# Create directory for slurm job outputs if it doesn't exist
mkdir -p ./slurm_jobs

# Load the modules
module purge
module load cuda/12.4.1         

# Activate the virtualenv
source /home/barattts/lavoltabuona/BA/.venv/env/bin/activate

# Navigate to the project root directory
PROJECT_ROOT="/home/barattts/lavoltabuona/BA"
cd "$PROJECT_ROOT"

# Add project root to PYTHONPATH so imports work
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Define an overarching folder for this entire suite of experiments
JOB_ID_STR=${SLURM_JOB_ID:-"manual_$(date +%Y%m%d_%H%M%S)"}
BASE_RUN_DIR="runs/run_training_random_${JOB_ID_STR}"
mkdir -p "$BASE_RUN_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Results will be saved in: $BASE_RUN_DIR"

# ============================================================
# Configurable Parameters — adjust these before submitting
# ============================================================
N_HIDDEN=1000           # Hidden neurons per layer
N_LAYERS=2              # Hidden layers (dRVFL / edRVFL / edRVFL-SC / esc-edRVFL)
N_ENSEMBLE=20           # Ensemble members (edRVFL / edRVFL-SC / esc-edRVFL)
N_BLOCKS=5              # Residual blocks (SResdRVFL)
N_LAYERS_PER_BLOCK=1    # Layers inside each residual block (SResdRVFL)
RSC_PROB=0.3            # Random skip-connection probability
FEATURE_SCALER=robust
LABEL_SCALER=robust

echo "======================================"
echo "Parameters:"
echo "  n_hidden=$N_HIDDEN  n_layers=$N_LAYERS  n_ensemble=$N_ENSEMBLE"
echo "  n_blocks=$N_BLOCKS  n_layers_per_block=$N_LAYERS_PER_BLOCK"
echo "  rsc_prob=$RSC_PROB  feature_scaler=$FEATURE_SCALER  label_scaler=$LABEL_SCALER"
echo "======================================"
echo "Preparing to run Experiments on CONE and ISOTHERM Datasets in Parallel"
echo "======================================"

# Common flags shared by every command
COMMON="--n-hidden $N_HIDDEN --feature-scaler $FEATURE_SCALER --label-scaler $LABEL_SCALER --base-dir \"$BASE_RUN_DIR\""

# Array of all training commands
commands=(
    # ---------- CONE ----------
    "python scripts/train_random_models.py --model ELM          --dataset cone $COMMON"
    "python scripts/train_random_models.py --model dRVFL         --dataset cone $COMMON --n-layers $N_LAYERS"
    "python scripts/train_random_models.py --model edRVFL        --dataset cone $COMMON --n-layers $N_LAYERS --n-ensemble $N_ENSEMBLE"
    "python scripts/train_random_models.py --model edRVFL-SC     --dataset cone $COMMON --n-layers $N_LAYERS --n-ensemble $N_ENSEMBLE --sc-mode dense"
    "python scripts/train_random_models.py --model edRVFL-SC     --dataset cone $COMMON --n-layers $N_LAYERS --n-ensemble $N_ENSEMBLE --sc-mode random --rsc-prob $RSC_PROB"
    "python scripts/train_random_models.py --model esc-edRVFL    --dataset cone $COMMON --n-layers $N_LAYERS --n-ensemble $N_ENSEMBLE"
    "python scripts/train_random_models.py --model SResdRVFL     --dataset cone $COMMON --n-layers $N_LAYERS_PER_BLOCK --n-blocks $N_BLOCKS --direct-link"

    # ---------- ISOTHERM ----------
    "python scripts/train_random_models.py --model ELM          --dataset isotherm $COMMON --use-area-root"
    "python scripts/train_random_models.py --model dRVFL         --dataset isotherm $COMMON --n-layers $N_LAYERS --use-area-root"
    "python scripts/train_random_models.py --model edRVFL        --dataset isotherm $COMMON --n-layers $N_LAYERS --n-ensemble $N_ENSEMBLE --use-area-root"
    "python scripts/train_random_models.py --model edRVFL-SC     --dataset isotherm $COMMON --n-layers $N_LAYERS --n-ensemble $N_ENSEMBLE --sc-mode dense --use-area-root"
    "python scripts/train_random_models.py --model edRVFL-SC     --dataset isotherm $COMMON --n-layers $N_LAYERS --n-ensemble $N_ENSEMBLE --sc-mode random --rsc-prob $RSC_PROB --use-area-root"
    "python scripts/train_random_models.py --model esc-edRVFL    --dataset isotherm $COMMON --n-layers $N_LAYERS --n-ensemble $N_ENSEMBLE --use-area-root"
    "python scripts/train_random_models.py --model SResdRVFL     --dataset isotherm $COMMON --n-layers $N_LAYERS_PER_BLOCK --n-blocks $N_BLOCKS --direct-link --use-area-root"
)

# Launch jobs in parallel
for i in "${!commands[@]}"; do
    GPU_ID=$((i % 7))
    echo "Launching job $i on GPU $GPU_ID: ${commands[$i]}"
    CUDA_VISIBLE_DEVICES=$GPU_ID eval "${commands[$i]}" &
    sleep 2  # stagger startup to avoid simultaneous heavy IO/compilation
done

echo "Waiting for all experiments to complete..."
wait

echo "Generating summary table..."
python scripts/summarize_results.py --run-dir "$BASE_RUN_DIR"

echo "All random model experiments completed"
