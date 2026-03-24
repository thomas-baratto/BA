#!/bin/bash

# ============================================================================
# Parallel Execution Launcher for argon-gtx
# Employs perfectly matched resource division:
#   - MLP Isotherm: 4 CPU, 1 GPU, 32GB RAM
#   - MLP Cone:     4 CPU, 1 GPU, 32GB RAM
#   - Random Sweep: 20 CPU, 6 GPUs, 240GB RAM
# Total: 28 CPUs, 8 GPUs (Maximizes Node)
# ============================================================================

echo "============================================================================"
echo "Launching Parallel Jobs to argon-gtx"
echo "============================================================================"

# Submit MLP Retraining for Isotherm
echo "Submitting MLP Retraining (Dataset: Isotherm)..."
sbatch --export=ALL,DATASET=isotherm scripts/slurm/train_mlp_metrics.sbatch

# Submit MLP Retraining for Cone
echo "Submitting MLP Retraining (Dataset: Cone)..."
sbatch --export=ALL,DATASET=cone scripts/slurm/train_mlp_metrics.sbatch

# Submit Random Model Sweep
echo "Submitting Random Parameter Sweep (Isotherm & Cone)..."
sbatch scripts/slurm/sweep_random_params.sbatch

echo "============================================================================"
echo "All 3 jobs submitted to SLURM. They should begin running concurrently."
echo "Use 'squeue -u \$USER' to monitor them."
echo "============================================================================"
