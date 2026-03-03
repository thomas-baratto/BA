#!/bin/bash

# --- Job Configuration ---
#SBATCH --job-name="rbf_training_"                  # Name of job
#SBATCH --output=./slurm_jobs/slurm-rbf-%j.out  # Output file, %j expands to job ID
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
BASE_RUN_DIR="runs/run_training_rbf_${JOB_ID_STR}"
mkdir -p "$BASE_RUN_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Results will be saved in: $BASE_RUN_DIR"

echo "======================================"
echo "Preparing to run Experiments on CONE and ISOTHERM Datasets in Parallel"
echo "======================================"

# Array of all training commands
commands=(
    "python scripts/train_random_models.py --model ELM --dataset cone --activation rbf --n-hidden 1000 --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model dRVFL --dataset cone --activation rbf --n-hidden 250 --n-layers 10 --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model edRVFL --dataset cone --activation rbf --n-hidden 250 --n-layers 5 --n-ensemble 20 --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model edRVFL-SC --dataset cone --activation rbf --n-hidden 250 --n-layers 5 --n-ensemble 20 --sc-mode dense --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model edRVFL-SC --dataset cone --activation rbf --n-hidden 250 --n-layers 5 --n-ensemble 20 --sc-mode random --rsc-prob 0.3 --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model esc-edRVFL --dataset cone --activation rbf --n-hidden 250 --n-layers 5 --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model SResdRVFL --dataset cone --activation rbf --n-hidden 200 --n-blocks 15 --direct-link --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model NF-RVFL --dataset cone --activation rbf --n-hidden 500 --nf-variation K --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model FELM --dataset cone --basis-type rbf --n-hidden 1000 --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
   
    "python scripts/train_random_models.py --model ELM --dataset isotherm --activation rbf --n-hidden 1000 --use-area-root --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model dRVFL --dataset isotherm --activation rbf --n-hidden 250 --n-layers 10 --use-area-root --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model edRVFL --dataset isotherm --activation rbf --n-hidden 250 --n-layers 5 --n-ensemble 20 --use-area-root --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model edRVFL-SC --dataset isotherm --activation rbf --n-hidden 250 --n-layers 5 --n-ensemble 20 --use-area-root --sc-mode dense --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model edRVFL-SC --dataset isotherm --activation rbf --n-hidden 250 --n-layers 5 --n-ensemble 20 --use-area-root --sc-mode random --rsc-prob 0.3 --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model esc-edRVFL --dataset isotherm --activation rbf --n-hidden 250 --n-layers 5 --use-area-root --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model SResdRVFL --dataset isotherm --activation rbf --n-hidden 200 --n-blocks 15 --use-area-root --direct-link --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model NF-RVFL --dataset isotherm --activation rbf --n-hidden 500 --use-area-root --nf-variation K --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
    "python scripts/train_random_models.py --model FELM --dataset isotherm --basis-type rbf --n-hidden 1000 --use-area-root --base-dir \"$BASE_RUN_DIR\" --feature-scaler robust --label-scaler robust"
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
