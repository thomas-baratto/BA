#!/bin/bash
#SBATCH --job-name=optuna_training_first_11
#SBATCH --output=logs/output_%j.txt         # Standard output
#SBATCH --error=logs/error_%j.txt           # Standard error
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=24:00:00                     # Max runtime
#SBATCH --gres=gpu:8                        # Request eight GPU
#SBATCH --partition=gpu                    # Adjust if cluster uses a different GPU partition

# Optional: load required modules
module load python/3.10 cuda/12.2

# Optional: activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

# Make sure logs directory exists
mkdir -p logs

# Run your training script
python your_script.py