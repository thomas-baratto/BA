#!/bin/bash
#SBATCH --job-name=test_power_monitor
#SBATCH --output=./slurm_jobs/test_power_monitor_%j.out
#SBATCH --error=./slurm_jobs/test_power_monitor_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -w argon-gtx

# ============================================================================
# Test Power Monitor Functionality
# ============================================================================

# Setup
mkdir -p ./slurm_jobs
mkdir -p ./power_logs

# Load modules
module purge
module load cuda/12.4.1

# Activate virtualenv
source /home/barattts/lavoltabuona/BA/.venv/daicheelavoltabuona/bin/activate

# Navigate to submission directory
cd $SLURM_SUBMIT_DIR

echo "============================================================================"
echo "Power Monitor Test"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "============================================================================"

# Show GPU info
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,power.draw,power.limit --format=csv

# Test power monitor for 30 seconds
POWER_LOG_DIR="./power_logs/test_power_monitor_${SLURM_JOB_ID}"
POWER_INTERVAL=1.0
DURATION=30

echo ""
echo "Starting power monitor test..."
echo "Duration: ${DURATION} seconds"
echo "Sample interval: ${POWER_INTERVAL} seconds"
echo "Output directory: $POWER_LOG_DIR"
echo ""

# Start power monitor in background
python power_monitor.py \
    --output-dir "$POWER_LOG_DIR" \
    --interval "$POWER_INTERVAL" \
    --filter python &

POWER_MONITOR_PID=$!
echo "Power monitor started with PID: $POWER_MONITOR_PID"
sleep 2

# Run a GPU workload to test power monitoring
echo "Running GPU workload on GPU 0 for ${DURATION} seconds..."
CUDA_VISIBLE_DEVICES=0 python -c "
import torch
import time

# Create some GPU workload
device = torch.device('cuda')
print(f'Using device: {device}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

# Run matrix operations for ${DURATION} seconds
start = time.time()
while time.time() - start < ${DURATION}:
    # Large matrix multiplication to stress GPU
    x = torch.randn(4000, 4000, device=device)
    y = torch.randn(4000, 4000, device=device)
    z = torch.matmul(x, y)
    torch.cuda.synchronize()

print('GPU workload complete')
"

# Stop power monitor
echo "Stopping power monitor..."
kill -SIGINT $POWER_MONITOR_PID
wait $POWER_MONITOR_PID 2>/dev/null

echo ""
echo "============================================================================"
echo "Power monitor test complete!"
echo "============================================================================"
echo "Check results in: $POWER_LOG_DIR"
echo ""
echo "Files created:"
ls -lh "$POWER_LOG_DIR"

echo ""
echo "Summary file contents:"
cat "$POWER_LOG_DIR"/power_summary_*.json

echo ""
echo "============================================================================"
