#!/bin/bash
# Script to run training on all targets using previous best Optuna parameters
# This skips hyperparameter optimization and uses the best params from existing studies

echo "======================================"
echo "Running training with --skip-optuna"
echo "Using best parameters from previous Optuna studies"
echo "======================================"
echo ""

# Array of all target labels
targets=("Area" "Iso_distance" "Iso_width" "all")

# Submit SLURM jobs for each target
for target in "${targets[@]}"; do
    echo "Submitting job for target: $target"
    sbatch run.sbatch --target "$target" --skip-optuna
    echo "  Job submitted!"
    echo ""
    sleep 1  # Small delay to avoid overwhelming the scheduler
done

echo "======================================"
echo "All 4 jobs submitted successfully!"
echo "======================================"
echo ""
echo "Check job status with: squeue -u $USER"
echo "View output logs in: ./slurm_jobs/"
echo ""
echo "Targets submitted:"
for target in "${targets[@]}"; do
    echo "  - $target"
done
