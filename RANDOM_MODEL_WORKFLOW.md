# Random Model Sweep → Retrain → Package Workflow

This workflow allows you to run a large hyperparameter sweep of random network models (ELM, RVFL variants), select the best-performing and most-efficient models, retrain them with model saving enabled, and package them for deployment—just like the MLP models.

## Overview

**4 Steps:**

1. **Run Sweep** — Submit all hyperparameter combinations (7 models × datasets, parallelized across 7 GPUs)
2. **Select Models** — Extract top-k by accuracy (RMSE) and efficiency (RMSE/Time ratio) per dataset
3. **Retrain Selected** — Rerun selected models with `NO_SAVE_MODEL=0` to save artifacts
4. **Package** — Copy artifacts to deployment directory with manifest

---

## Step 1: Run the Hyperparameter Sweep

The sweep submits 7 random network model architectures (ELM, dRVFL, edRVFL, edRVFL-SC, esc-edRVFL, SResdRVFL) with various hyperparameters, testing on both **isotherm** and **cone** datasets.

**Important:** Set `KEEP_RUN_ARTIFACTS=1` to preserve all run artifacts (model weights, scalers, predictions). By default, they are deleted after summarization to save space.

```bash
cd /home/barattts/lavoltabuona/BA

sbatch --export=KEEP_RUN_ARTIFACTS=1 scripts/slurm/sweep_random_params.sbatch
```

This will submit multiple jobs running in parallel (max 7 at a time due to GPU limit). You can monitor progress:

```bash
squeue -u $USER  # Check job status
tail -f slurm_jobs/sweep_random_*.err  # Watch logs (tail multiple files)
```

**Expected run time:**  
- ~2–4 hours depending on queue wait time and node load
- Each individual model trains in ~2–5 minutes

---

## Step 2: Generate Summary and Select Models

Once the sweep completes, regenerate the summary table (aggregates all run results):

```bash
PYTHONPATH=. python scripts/summarize_results.py --run-dir runs/run_sweep_random_XXXX
```

This creates `runs/run_sweep_random_XXXX/summary_table.csv` with columns:

```
Model,Dataset,RMSE,MAE,MSE,MAPE,R2,nRMSE,KGE,Time(s),Folder,N_Hidden,Layers,N_Ensemble,Blocks,Activation
```

Then **select the best and most efficient models** for retraining:

```bash
python scripts/select_models_for_retrain.py \
  --summary-csv runs/run_sweep_random_XXXX/summary_table.csv \
  --top-k 3
```

This outputs `runs/run_sweep_random_XXXX/selected_for_retrain.csv` with:
- Top 3 models by **lowest RMSE** (best accuracy) per dataset
- Top 3 models by **efficiency** (lowest RMSE/Time ratio) per dataset  
- Duplicates merged (same model in both categories listed once)

**Example output:**

```
Dataset,Model,RMSE,Time(s),Folder,N_Hidden,Layers,N_Ensemble,Blocks,Activation,selection_reason
cone,cone_edRVFL-SC,0.0234,185.2,cone_edRVFL-SC_5821,256,3,10,5,elu,best_rmse
cone,cone_esc-edRVFL,0.0241,92.1,cone_esc-edRVFL_5842,512,2,8,3,tanh,best_efficiency
isotherm,isotherm_edRVFL,0.0118,210.5,isotherm_edRVFL_5803,128,2,5,2,relu,best_rmse
...
```

---

## Step 3: Retrain Selected Models

The retrain step re-executes the selected models with `NO_SAVE_MODEL=0`, which ensures:
- Models are saved as `.pkl` files
- Scalers are saved
- Predictions and metrics are saved as `.json` and `.npz`

### Option A: Automatic (using provided script)

```bash
bash scripts/retrain_selected_models.sh runs/run_sweep_random_XXXX
```

This script will:
1. Load `selected_for_retrain.csv`
2. Extract hyperparameters for each model
3. Submit SLURM jobs for each with `NO_SAVE_MODEL=0`
4. Print job IDs and next instructions

### Option B: Manual (more control)

For each row in `selected_for_retrain.csv`, submit a job:

```bash
sbatch \
  --export=NO_SAVE_MODEL=0,DATASET=cone,MODEL=cone_esc-edRVFL,\
N_HIDDEN=256,N_LAYERS=3,N_ENSEMBLE=10,BLOCKS=5,ACTIVATION=elu,\
SCALERS_DIR=runs/run_sweep_random_XXXX \
  scripts/slurm/run_random_model.sbatch
```

**Key exports:**
- `NO_SAVE_MODEL=0` — **Required**; enables saving of model.pkl
- `DATASET` — "cone" or "isotherm"
- `MODEL` — Full model name (e.g., "cone_esc-edRVFL")
- `N_HIDDEN`, `N_LAYERS`, `N_ENSEMBLE`, `BLOCKS`, `ACTIVATION` — From CSV
- `SCALERS_DIR` — Path to original sweep run (ensures consistent preprocessing)

**Monitor retraining:**

```bash
squeue -u $USER
tail -f slurm_jobs/final_train_*.err  # Or whichever job IDs were submitted
```

**Expected run time:**  
- ~2–5 minutes per model
- Total: (number of selected models) × 2–5 minutes

---

## Step 4: Package Models

Once retraining completes, consolidate the best-performing and best-efficiency models into a deployment package:

```bash
python scripts/package_models.py \
  --random-summary runs/run_sweep_random_XXXX/summary_table.csv \
  --random-run-dir runs/run_sweep_random_XXXX \
  --include-efficient
```

This script will:
1. Identify top-performing model per dataset (by RMSE)
2. Identify top-efficiency model per dataset (by RMSE/Time ratio)
3. Copy artifacts to `data/good_runs/packages/`:
   - `model.pkl` (trained weights)
   - `scalers.pkl` (preprocessing)
   - `model_config.json` (feature/label names for inference)
4. Generate `manifest.json` with model metadata and efficiency ranking tags

**Output structure:**

```
data/good_runs/packages/
  cone_esc-edRVFL_20260321/
    model.pkl
    scalers.pkl
    model_config.json
  cone_edRVFL-SC_20260321/  (tagged: best_efficiency)
    model.pkl
    scalers.pkl
    model_config.json
  isotherm_edRVFL_20260321/
    model.pkl
    scalers.pkl
    model_config.json
  ...
  manifest.json
```

---

## Complete Workflow (Copy-Paste)

```bash
#!/bin/bash
set -e

SWEEP_RUN="runs/run_sweep_random_1004"  # Replace with your sweep run directory

echo "Step 1: Running sweep..."
sbatch --export=KEEP_RUN_ARTIFACTS=1 scripts/slurm/sweep_random_params.sbatch

echo "Waiting for sweep to complete..."
echo "Check status with: squeue -u \$USER"
echo "When complete, continue with Step 2."

# [WAIT FOR SWEEP TO COMPLETE]

echo ""
echo "Step 2: Summarizing and selecting models..."
PYTHONPATH=. python scripts/summarize_results.py --run-dir "$SWEEP_RUN"
python scripts/select_models_for_retrain.py --summary-csv "$SWEEP_RUN/summary_table.csv" --top-k 3

echo ""
echo "Step 3: Retraining selected models..."
bash scripts/retrain_selected_models.sh "$SWEEP_RUN"

echo "Waiting for retrain jobs to complete..."
echo "Check status with: squeue -u \$USER"
echo "When complete, proceed to Step 4."

# [WAIT FOR RETRAIN TO COMPLETE]

echo ""
echo "Step 4: Packaging models..."
python scripts/package_models.py \
  --random-summary "$SWEEP_RUN/summary_table.csv" \
  --random-run-dir "$SWEEP_RUN" \
  --include-efficient

echo ""
echo "✓ Workflow complete! Packaged models in data/good_runs/packages/"
```

---

## Troubleshooting

### Sweep doesn't submit jobs
- Check that `scripts/slurm/sweep_random_params.sbatch` exists
- Verify you have SLURM access: `sinfo` should show available partitions
- Check queue: `squeue` — if queue full, wait for jobs to complete

### Models not saving during retrain
- Verify `NO_SAVE_MODEL=0` is exported (case-sensitive)
- Check job logs: `tail slurm_jobs/final_train_*.err` for errors
- Verify selected CSV has correct hyperparameter values (especially `BLOCKS` = number of blocks for esc-edRVFL variants)

### Scalers mismatch during retrain
- Ensure `SCALERS_DIR` points to the original sweep run directory
- The script `run_random_model.sbatch` loads scalers.pkl from `$SCALERS_DIR` to maintain preprocessing consistency

### Package step fails
- Verify that retrain completed and `model.pkl` files exist in retrain run directories
- Check `--random-run-dir` points to correct sweep directory
- Ensure `data/good_runs/` directory exists: `mkdir -p data/good_runs/packages`

---

## Key Files Reference

| Script | Purpose |
|--------|---------|
| `scripts/slurm/sweep_random_params.sbatch` | Parameterized sweep launcher (7 jobs in parallel) |
| `scripts/slurm/run_random_model.sbatch` | Single-model trainer (called by sweep and retrain) |
| `scripts/select_models_for_retrain.py` | **Your generated script** — selects best & efficient models |
| `scripts/retrain_selected_models.sh` | **Your generated script** — loops through CSV and submits retrain jobs |
| `scripts/summarize_results.py` | Aggregates all run results into summary_table.csv |
| `scripts/package_models.py` | Consolidates trained artifacts for deployment |
| `scripts/train_mlp_with_metrics.py` | Reference: MLP training (uses same packaging pipeline) |

---

## Notes

- **MLP vs Random Models**: MLP models are trained once via Optuna, then packaged. Random models require a full sweep → select → retrain → package pipeline because we want to test many architecture combinations upfront.
- **Efficiency Metric**: `RMSE / Time(s)` represents error reduction per second. Higher is better (lower error faster). Use this to select fast-training models that trade minimal accuracy for speed.
- **Scalers Consistency**: All retraining uses scalers from the original sweep to ensure preprocessing consistency. This is critical for fair model comparison.
- **Artifacts**: Summary table, model weights, scalers, hyperparameters, and predictions are all preserved after retraining (unless `KEEP_RUN_ARTIFACTS=0` is set).
