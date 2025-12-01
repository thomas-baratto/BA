#!/bin/bash

# --- Job Configuration ---
#SBATCH --job-name="elm_BA"                     # Name of job
#SBATCH --output=./slurm_jobs/slurm-elm-%j.out  # Output file, %j expands to job ID
#SBATCH -w argon-gtx                            # Request the specific node 'argon-gtx'

# --- Resource Allocation ---
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Single task
#SBATCH --cpus-per-task=4           # 4 CPUs for data loading
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --mem=32G                   # Memory allocation
#SBATCH --time=0-02:00:00           # 2 hours max (ELM is very fast)

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
echo "Starting ELM training..."

# Navigate to the directory where you submitted the job
cd $SLURM_SUBMIT_DIR

# Show GPU info
nvidia-smi

# Run ELM training for all targets
# You can customize these parameters:
#   --target: Area, Iso_distance, Iso_width, or all
#   --n-hidden: number of hidden neurons (default: 2000)
#   --activation: ReLU, LeakyReLU, GELU, ELU (default: ReLU)
#   --alpha: regularization strength (default: 1e-3)
#   --feature-scaler: minmax, standard, robust, quantile (default: standard)
#   --label-scaler: minmax, standard, robust, quantile (default: standard)

python scripts/train_elm.py \
    --target all \
    --n-hidden 3000 \
    --activation GELU \
    --alpha 1e-3 \
    --feature-scaler standard \
    --label-scaler standard

echo "ELM training completed"
