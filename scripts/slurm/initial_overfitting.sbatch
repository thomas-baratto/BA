#!/bin/bash
#SBATCH --job-name=initial_overfitting
#SBATCH --output=initial_overfitting.out
#SBATCH --error=initial_overfitting.err
#SBATCH -w argon-gtx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=4G


module purge
module load cuda/12.4.1

source .venv/env/bin/activate

python initial_overfitting/main.py