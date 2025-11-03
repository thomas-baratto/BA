#!/bin/bash

# --- Job Configuration ---
#SBATCH --job-name="optuna_BA_parallel"          # Name for the wholejob
#SBATCH --output=./slurm_jobs/slurm-job-%A_%a.out # Output file
                                               # %A = Main Job ID, %a = Task ID
#SBATCH -w argon-gtx                           # Request the specific node 'argon-gtx'

# --- Job Array ---
#SBATCH --array=0-3                            # Run 4 tasks, with IDs 0, 1, 2, 3

# --- Resource Allocation (PER TASK) ---
# Each of the 4 tasks in the array will get these resources
# Total resources used: 4 GPUs, 16 CPUs. (argon-gtx has 8 GPUs & 56 CPUs)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=24:00:00                         # 24 hours (for each task)

# --- Environment Setup ---
echo "--- SLURM JOB ---"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job started on $(hostname)"
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
echo "Starting Python script..."

# Navigate to the directory where you submitted the job
cd $SLURM_SUBMIT_DIR

# --- Map Array ID to Target ---
# 1. Define a bash array with your targets
TARGETS=("all" "Area" "Iso_distance" "Iso_width")

# 2. Get the target for this specific task ID
CURRENT_TARGET=${TARGETS[$SLURM_ARRAY_TASK_ID]}

echo "This task is running: python run_optuna.py --target $CURRENT_TARGET"

# 3. Run your main Optuna script with the selected target
python run_optuna.py --target $CURRENT_TARGET

echo "---"
echo "Task $SLURM_ARRAY_TASK_ID ($CURRENT_TARGET) finished."