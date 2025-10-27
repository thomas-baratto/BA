#!/bin/bash

# --- Job Configuration ---
#SBATCH --job-name="optuna_BA_barattts"        # Name for your job
#SBATCH --output=slurm-job-%j.out   # Output file, %j expands to job ID
#SBATCH -w argon-tesla2             # Request the specific node 'argon-tesla2'

# --- Resource Allocation ---
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task (python script)
#SBATCH --cpus-per-task=8           # Give 8 CPUs to that task (for dataloaders, etc.)
#SBATCH --gpus=1                    # Request 1 GPU (argon-tesla2 has 2)
#SBATCH --time=12:00:00             # 12 hours

# --- Environment Setup ---
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "---"

# TODO: Load the modules your cluster provides
# These are EXAMPLES. You MUST check your cluster's documentation.
module purge
module load anaconda/2023.03  # Example for loading Conda
module load cuda/12.2.2         # Example for loading the correct CUDA version

# If using a virtualenv (venv):
source /home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona/bin/activate

# --- Run the Program ---
echo "Starting Python script..."

# Navigate to the directory where you submitted the job
cd $SLURM_SUBMIT_DIR

# Run your main Optuna script
# We just use 'python', not 'srun', because it's a single-task job.
python run_optuna.py

echo "Job finished."