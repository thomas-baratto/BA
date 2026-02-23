#!/bin/bash

# --- Job Configuration ---
#SBATCH --job-name="elm_best_train"             # Name of job
#SBATCH --output=./slurm_jobs/slurm-elm-best-%j.out  # Output file
#SBATCH -w argon-gtx                            # Request the specific node 'argon-gtx'

# --- Resource Allocation ---
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Single task
#SBATCH --cpus-per-task=4           # 4 CPUs
#SBATCH --mem=32G                   # Memory allocation
#SBATCH --time=0-01:00:00           # 1 hour max

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

# --- Run the Program ---
echo "Starting ELM training with best hyperparameters..."

# Navigate to the project root
cd /home/barattts/lavoltabuona/BA

# Add current directory to PYTHONPATH so that 'core' module can be imported
export PYTHONPATH=$PYTHONPATH:.

# Run with best hyperparameters: n_hidden=7000, activation=GELU, alpha=0.01
python -u scripts/train_elm.py \
    --target all \
    --n-hidden 7000 \
    --activation GELU \
    --alpha 0.01 \
    --feature-scaler standard \
    --label-scaler standard \
    --no-cuda

echo "ELM training completed"
