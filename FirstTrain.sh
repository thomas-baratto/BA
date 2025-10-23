#!/bin/bash
#SBATCH --job-name=optuna_training_first_11
#SBATCH --output=logs/output_%j.txt         # Standard output
#SBATCH --error=logs/error_%j.txt           # Standard error
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00                     # Max runtimepw
#SBATCH --gres=gpu:8                        # Request eight GPU

# Optional: load required modules
module load python/3.10 cuda/12.2
source env/firstTestenv/bin/activate

# Make sure logs directory exists
mkdir -p logs

# Run your training script
python FirstTrainingLabel11.py