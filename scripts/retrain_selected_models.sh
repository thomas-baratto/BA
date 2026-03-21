#!/bin/bash
#
# Retrain selected best and efficient models from sweep with model saving enabled.
# Usage: ./scripts/retrain_selected_models.sh <original_sweep_run_dir> [--continue-from-csv <csv_file>]
#

set -e  # Exit on error

SWEEP_RUN_DIR="${1:?Error: sweep run directory required. Usage: $0 <sweep_run_dir> [--continue-from-csv <csv_file>]}"
SELECTED_CSV="${2:-${SWEEP_RUN_DIR}/selected_for_retrain.csv}"

if [ ! -d "$SWEEP_RUN_DIR" ]; then
    echo "Error: Sweep run directory not found: $SWEEP_RUN_DIR"
    exit 1
fi

# If CSV not already generated, generate it now
if [ ! -f "$SELECTED_CSV" ]; then
    echo "Generating selected_for_retrain.csv..."
    SUMMARY_CSV="${SWEEP_RUN_DIR}/summary_table.csv"
    if [ ! -f "$SUMMARY_CSV" ]; then
        echo "Error: summary_table.csv not found in $SWEEP_RUN_DIR"
        echo "Run: python scripts/summarize_results.py --run-dir $SWEEP_RUN_DIR"
        exit 1
    fi
    python scripts/select_models_for_retrain.py --summary-csv "$SUMMARY_CSV" --output-csv "$SELECTED_CSV"
fi

echo ""
echo "Retraining selected models from: $SELECTED_CSV"
echo "===================="

# Read CSV and submit jobs
JOB_IDS=()
COUNT=0

while IFS=',' read -r Dataset Model RMSE Time Folder N_Hidden Layers N_Ensemble Blocks Activation SelectionReason; do
    # Skip header
    if [ "$Dataset" == "Dataset" ]; then
        continue
    fi
    
    COUNT=$((COUNT + 1))
    
    # Extract dataset-specific prefix (e.g., "cone_esc-edRVFL" -> "cone", "isotherm_ELM" -> "isotherm")
    DATASET_PREFIX="${Dataset%_*}"
    
    echo ""
    echo "Job $COUNT: $Model on $Dataset (RMSE=$RMSE, selection=$SelectionReason)"
    echo "  Hyperparameters: n_hidden=$N_Hidden, n_layers=$Layers, n_ensemble=$N_Ensemble"
    echo "                   blocks=$Blocks, activation=$Activation"
    
    # Submit retrain job with NO_SAVE_MODEL=0 to enable model saving
    # The run_random_model.sbatch script will:
    #   1. Load scalers.pkl from sweep run (to ensure consistency)
    #   2. Train new model with same hyperparameters
    #   3. Save model.pkl, predictions, results.json
    JOB_CMD="sbatch \
        --export=NO_SAVE_MODEL=0,DATASET=${DATASET_PREFIX},MODEL=${Model},\
N_HIDDEN=${N_Hidden},N_LAYERS=${Layers},N_ENSEMBLE=${N_Ensemble},\
BLOCKS=${Blocks},ACTIVATION=${Activation},\
SCALERS_DIR=${SWEEP_RUN_DIR} \
        scripts/slurm/run_random_model.sbatch"
    
    echo "  Command: $JOB_CMD"
    JOB_OUTPUT=$($JOB_CMD)
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
    JOB_IDS+=("$JOB_ID")
    echo "  Submitted: Job ID $JOB_ID"
    
done < "$SELECTED_CSV"

echo ""
echo "===================="
echo "Submitted $COUNT retrain jobs:"
for i in "${!JOB_IDS[@]}"; do
    echo "  [$(($i + 1))] Job ID: ${JOB_IDS[$i]}"
done

echo ""
echo "Monitor progress with: squeue --me"
echo "View logs with: tail -f slurm_jobs/retrain_*.err"
echo ""
echo "Once complete, run:"
echo "  python scripts/package_models.py \\"
echo "    --random-summary ${SWEEP_RUN_DIR}/summary_table.csv \\"
echo "    --random-run-dir ${SWEEP_RUN_DIR} \\"
echo "    --include-efficient"
