#!/bin/bash

# --- Job Configuration ---
#SBATCH --job-name="elm_grid_BA"                # Name of job
#SBATCH --output=./slurm_jobs/slurm-elm-grid-%j.out  # Output file
#SBATCH -w argon-gtx                            # Request the specific node 'argon-gtx'

# --- Resource Allocation ---
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Single task
#SBATCH --cpus-per-task=4           # 4 CPUs
#SBATCH --mem=32G                   # Memory allocation
#SBATCH --time=0-03:00:00           # 3 hours max (grid search should finish in ~30-60 minutes)

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
source /home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona/bin/activate

# --- Run the Program ---
echo "Starting ELM grid search..."

# Navigate to the directory where you submitted the job
cd $SLURM_SUBMIT_DIR

# Run ELM grid search
# Grid will test:
#   - n_hidden: [500, 1000, 2000, 3000, 5000, 7000]
#   - activation: ['ReLU', 'LeakyReLU', 'GELU', 'ELU']
#   - alpha: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# Total: 6 × 4 × 5 = 120 combinations
# Expected time: ~30-60 minutes

python -u grid_search_elm.py \
    --target all \
    --feature-scaler standard \
    --label-scaler standard \
    --no-cuda

echo "ELM grid search completed"
