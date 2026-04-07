#!/bin/bash
#SBATCH --job-name=seed_sweep
#SBATCH --output=./slurm_jobs/seed_sweep_%j.out
#SBATCH --error=./slurm_jobs/seed_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:7
#SBATCH --mem=350G
#SBATCH --time=0-02:00:00
#SBATCH -w argon-gtx

# ============================================================================
# Seed Sweep: run 3 random model winners with N seeds each (default 100)
# ============================================================================
# Measures performance variance across random initialisations.
# Uses --no-save-model to suppress model.pkl + test_predictions.npz (disk safe).
# All metrics JSONs and multi_seed_summary.json are preserved.
#
# Usage:
#   sbatch scripts/slurm/seed_sweep.sbatch                  # 100 seeds (default)
#   sbatch --export=N_SEEDS=200 scripts/slurm/seed_sweep.sbatch
# ============================================================================

N_SEEDS="${N_SEEDS:-100}"
OUT_ROOT="runs/seed_sweep_${SLURM_JOB_ID}"

# --- Setup ---
mkdir -p ./slurm_jobs

module purge
module load cuda/12.4.1

source /home/barattts/lavoltabuona/BA/.venv/env/bin/activate
export PYTHONPATH=.

# --- Job Info ---
echo "============================================================================"
echo "SEED SWEEP — SLURM Job Information"
echo "============================================================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "N_SEEDS:  $N_SEEDS"
echo "Output:   $OUT_ROOT"
echo "============================================================================"

cd $SLURM_SUBMIT_DIR
nvidia-smi

mkdir -p "$OUT_ROOT"

# ============================================================================
# Winner 1: cone / edRVFL-SC
# ============================================================================
CUDA_VISIBLE_DEVICES=0,1 python scripts/training/train_random_models.py \
    --model edRVFL-SC \
    --dataset cone \
    --feature-scaler robust \
    --label-scaler robust \
    --n-hidden 1000 \
    --activation GELU \
    --alpha 0.001 \
    --gamma 1.0 \
    --n-layers 3 \
    --n-ensemble 10 \
    --sc-mode dense \
    --rsc-prob 0.5 \
    --n-folds 5 \
    --n-blocks 5 \
    --random-state 42 \
    --n-seeds "$N_SEEDS" \
    --no-save-model \
    --disable-power-monitor \
    --output-dir "$OUT_ROOT/cone_edRVFL-SC" \
    &> "$OUT_ROOT/cone_edRVFL-SC.log" &
PID_CONE=$!
echo "Started cone/edRVFL-SC (PID=$PID_CONE) on GPU 0,1"

# ============================================================================
# Winner 2: isotherm_nRMSE / SResdRVFL
# ============================================================================
CUDA_VISIBLE_DEVICES=2,3 python scripts/training/train_random_models.py \
    --model SResdRVFL \
    --dataset isotherm \
    --feature-scaler robust \
    --label-scaler robust \
    --use-area-root \
    --n-hidden 1500 \
    --activation GELU \
    --alpha 0.001 \
    --gamma 1.0 \
    --n-layers 1 \
    --n-ensemble 10 \
    --sc-mode dense \
    --rsc-prob 0.5 \
    --n-folds 5 \
    --n-blocks 8 \
    --direct-link \
    --random-state 42 \
    --n-seeds "$N_SEEDS" \
    --no-save-model \
    --disable-power-monitor \
    --output-dir "$OUT_ROOT/isotherm_SResdRVFL" \
    &> "$OUT_ROOT/isotherm_SResdRVFL.log" &
PID_ISO_NRMSE=$!
echo "Started isotherm_nRMSE/SResdRVFL (PID=$PID_ISO_NRMSE) on GPU 2,3"

# ============================================================================
# Winner 3: isotherm_KGE / dRVFL
# ============================================================================
CUDA_VISIBLE_DEVICES=4,5 python scripts/training/train_random_models.py \
    --model dRVFL \
    --dataset isotherm \
    --feature-scaler robust \
    --label-scaler robust \
    --use-area-root \
    --n-hidden 1500 \
    --activation ELU \
    --alpha 0.001 \
    --gamma 1.0 \
    --n-layers 1 \
    --n-ensemble 10 \
    --sc-mode dense \
    --rsc-prob 0.5 \
    --n-folds 5 \
    --n-blocks 5 \
    --random-state 42 \
    --n-seeds "$N_SEEDS" \
    --no-save-model \
    --disable-power-monitor \
    --output-dir "$OUT_ROOT/isotherm_dRVFL" \
    &> "$OUT_ROOT/isotherm_dRVFL.log" &
PID_ISO_KGE=$!
echo "Started isotherm_KGE/dRVFL (PID=$PID_ISO_KGE) on GPU 4,5"

# ============================================================================
# Wait for all to finish
# ============================================================================
echo "Waiting for all 3 winners to finish..."
wait $PID_CONE
EC_CONE=$?
echo "cone/edRVFL-SC finished (exit=$EC_CONE)"

wait $PID_ISO_NRMSE
EC_ISO_NRMSE=$?
echo "isotherm_nRMSE/SResdRVFL finished (exit=$EC_ISO_NRMSE)"

wait $PID_ISO_KGE
EC_ISO_KGE=$?
echo "isotherm_KGE/dRVFL finished (exit=$EC_ISO_KGE)"

# ============================================================================
# Post-processing: run analysis
# ============================================================================
echo "============================================================================"
echo "Running seed sweep analysis..."
echo "============================================================================"
python scripts/analysis/analyze_seed_sweep.py --sweep-dir "$OUT_ROOT"
EC_ANALYSIS=$?

echo "============================================================================"
echo "Disk usage:"
du -sh "$OUT_ROOT"
du -sh "$OUT_ROOT"/*/
echo "============================================================================"
echo "Seed sweep completed."
echo "  cone:            exit=$EC_CONE"
echo "  isotherm_nRMSE:  exit=$EC_ISO_NRMSE"
echo "  isotherm_KGE:    exit=$EC_ISO_KGE"
echo "  analysis:        exit=$EC_ANALYSIS"
echo "============================================================================"

# Exit with non-zero if any training failed
if [ $EC_CONE -ne 0 ] || [ $EC_ISO_NRMSE -ne 0 ] || [ $EC_ISO_KGE -ne 0 ]; then
    exit 1
fi
exit 0
