#!/bin/bash

# ============================================================================
# Parallel Execution Launcher for MLP Retraining
# Submits both Dataset jobs as distinct Slurm allocations so they run
# simultaneously in hardware-parallel instead of sequentially.
# ============================================================================

echo "============================================================================"
echo "Launching MLPs to Slurm..."
echo "============================================================================"

# Submit MLP Retraining for Isotherm
sbatch --export=ALL,DATASET=isotherm scripts/slurm/train_mlp_metrics.sbatch

# Submit MLP Retraining for Cone
sbatch --export=ALL,DATASET=cone scripts/slurm/train_mlp_metrics.sbatch

echo "============================================================================"
echo "Both MLPs submitted successfully to run in parallel!"
echo "Use 'squeue -u \$USER' to monitor them."
echo "============================================================================"
