#!/bin/bash

# --- Job Configuration ---
#SBATCH --job-name="optuna_BA_barattts"        # Name of job
#SBATCH --output=./slurm_jobs/slurm-job-%j.out   # Output file, %j expands to job ID
#SBATCH -w argon-gtx                           # Request the specific node 'argon-gtx'

# --- Resource Allocation ---
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task (python script)
#SBATCH --cpus-per-task=4           # Give 4 CPUs to that task
#SBATCH --gpus=1                    # Request 1 GPU (argon-gtx has 8 GPUs)
#SBATCH --time=2-00:00:00             # 24 hours

# --- Environment Setup ---
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "---"

# Create directory for slurm job outputs if it doesn't exist
mkdir -p ./slurm_jobs

# Create directory for Optuna study database if it doesn't exist
mkdir -p ./runs

# Load the modules
module purge
module load cuda/12.4.1         
# Activate the virtualenv
source /home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona/bin/activate

# --- Run the Program ---
echo "Starting Python script..."

# Navigate to the directory where you submitted the job
cd $SLURM_SUBMIT_DIR

# Show GPU info
nvidia-smi

# --- Run your main Optuna script ---
# Pass all arguments to the Python script
echo "Running: python run_optuna.py $@"
python run_optuna.py $@

EXIT_CODE=$?
echo "Job finished with exit code: $EXIT_CODE"
exit $EXIT_CODE