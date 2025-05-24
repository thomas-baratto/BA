#!/bin/bash
#SBATCH --job-name=mkdir_test
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --mem=1G

# Create logs folder if it doesn't exist
mkdir -p logs

# Create a directory
mkdir -p test_output

# Write a file inside it
echo "Hello from SLURM job $SLURM_JOB_ID" > test_output/hello.txt

echo "Done."