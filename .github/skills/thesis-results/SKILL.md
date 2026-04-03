---
name: thesis-results
description: "Use when: implementing BA scripts, models, or analysis to produce results needed by the thesis. Covers thesis gaps requiring BA work (missing plots, baseline comparisons, inference benchmarking, new analysis scripts), SLURM job submission constraints, power monitoring integration, artifact structure, and the full mapping from thesis TODOs to BA implementation tasks."
argument-hint: "Describe what thesis result you need — e.g. 'generate missing power plots', 'add inference benchmarking', 'regenerate Optuna parallel coordinate plots'"
---

# Thesis Results from BA Pipeline

Produce experimental results, plots, and analysis artifacts in `BA/` that the thesis references or still needs.

## When to Use

- The thesis has a TODO, placeholder, or commented-out figure that requires BA work
- A thesis section says something is "not included" or "left for future work" that can now be implemented
- You need to create or modify a BA analysis script to generate thesis-ready output
- You need to run training, sweeps, or evaluation that requires SLURM submission

## Critical Constraints

### Execution Environment

| Task type | Where to run | How |
|-----------|-------------|-----|
| Analysis scripts (plotting, metrics, tables) | **Local** (macOS) | `PYTHONPATH=. .venv/bin/python scripts/analysis/<script>.py` |
| Inference / prediction | **Local** | `PYTHONPATH=. .venv/bin/python scripts/deployment/predict.py` |
| Retrain random models (deterministic, CPU) | **Local** | `PYTHONPATH=. .venv/bin/python scripts/deployment/retrain_random_models.py` |
| MLP training (GPU required) | **SLURM only** | `sbatch scripts/slurm/train_mlp_metrics.sbatch` |
| Optuna HPO | **SLURM only** | `sbatch scripts/slurm/run_optuna_*.sbatch` |
| Random model sweep (467 configs) | **SLURM only** | `sbatch scripts/slurm/sweep_random_params.sbatch` |
| Power monitoring | **SLURM only** | Integrated into SLURM scripts; requires nvidia-smi |

**Server details:**
- Remote path: `/home/barattts/lavoltabuona/BA/`
- Venv on server: `.venv/env/bin/activate` (note: `env/` subdirectory, not `.venv/bin/`)
- Local venv: `.venv/bin/python`
- Node: `argon-gtx` (7× GTX 1080 Ti, 56 CPUs, 754 GB RAM)
- Module: `cuda/12.4.1`
- Scratch disk: `/data/scratch/barattts/` (node-local SSD, copy back post-job)

### Python Environment

- **Always** use `.venv/bin/python` and `.venv/bin/pip` locally
- **Always** set `PYTHONPATH=.` when running scripts from project root
- **Never** use bare `python` or `pip`
- **Never** run formatters via terminal (VS Code Format on Save handles it)

### Model Artifacts

Random model `.pkl` files are excluded from git (too large).
To recreate them locally:
```bash
PYTHONPATH=. .venv/bin/python scripts/deployment/retrain_random_models.py
```
This is deterministic (seed=42) and produces bit-identical models on CPU.
All other artifacts (`.pt`, `model_config.json`, `scalers.pkl`, `results_*.json`) are tracked.

### Metrics

Use functions from `core/metrics.py` — never reimplement.
Key function: `compute_regression_metrics(y_true, y_pred)` → dict with r2, rmse, nrmse, kge, mae, mape, residual stats.
For per-label metrics, loop over label columns and call `compute_regression_metrics` on each column slice.

### Output Conventions

- Thesis-ready plots: PDF to `docs/plots/` or `thesis/graphics/plots/`
- Results JSON: `artifacts/models/<model_type>/<dataset>/`
- LaTeX tables: `docs/tables/`
- Use `core/thesis_style.py` → `apply_thesis_style()` + `save_fig()` for consistent plot styling

## Thesis Gap Tracker

### Completed

| Item | Script | Artifact |
|------|--------|----------|
| Böttcher analytical baseline | `scripts/analysis/evaluate_boettcher_baseline.py` | `artifacts/models/baseline/boettcher/results_boettcher_baseline.json` |

### Open — Actionable Locally

| Priority | Thesis gap | What to implement | Target output |
|----------|-----------|-------------------|---------------|
| **HIGH** | Inference latency not measured (Limitation §6.2, Future Work §6.3) | New script `scripts/analysis/benchmark_inference.py` — load each model, time N predictions, report per-sample latency | JSON + summary table |
| **HIGH** | Baseline comparison is qualitative only (§5.2) | Update thesis text — baseline is now quantitative (results JSON exists) | Thesis LaTeX update |
| **MEDIUM** | 11 missing power/energy plots in appendix | Regenerate from monitoring CSVs if raw `power_log_*.csv` / `power_summary_*.json` exist. Scripts: `plot_power_analysis.py`, `plot_optuna_study.py` | PDFs for appendix |
| **MEDIUM** | Missing `generic_rvfl.pdf` architecture diagram (thesis line 375) | Extend `visualize_architecture.py` or create manually in TikZ | PDF in `thesis/graphics/plots/architecture/` |
| **LOW** | 2 missing Optuna parallel coordinate plots | Run `plot_optuna_study.py` with journal logs | PDFs in `thesis/graphics/plots/optuna_*/` |

### Open — Requires SLURM / Server

| Priority | Thesis gap | What to implement | SLURM script |
|----------|-----------|-------------------|--------------|
| **LOW** | Continuous RaNN optimization (Future Work §6.3) | Optuna study for RaNN hyperparameters | New `.sbatch` needed |
| **LOW** | Missing power monitoring data | Re-run training with monitoring if raw CSVs are lost | Existing SLURM scripts already integrate `monitoring/power_monitor.py` |

### Out of Scope

These are acknowledged as future work in the thesis and not feasible before the deadline:
- CNN / QCNN comparison (different output representation)
- Transfer to heterogeneous/transient settings (needs new datasets)
- Hybrid RaNN + gradient fine-tuning (new model class, research-level)
- Cross-validation (impractical: 467 configs × k folds + 10k Optuna trials)
- BayesValidRox integration (optional per Ausschreibung, steep learning curve)

## Analysis Scripts Reference

| Script | What it produces | Runs locally? |
|--------|-----------------|---------------|
| `evaluate_boettcher_baseline.py` | Böttcher vs NN metrics on Isotherm==1 test set | Yes |
| `compare_models.py` | Console table of MLP vs random performance | Yes |
| `generate_model_comparison.py` | Full markdown report with embedded plots | Yes |
| `csv_to_latex.py` | Summary CSV → LaTeX booktabs table | Yes |
| `generate_all_plots.py` | Orchestrates all plot scripts (staleness-based) | Yes |
| `plot_pareto_frontiers.py` | 4 Pareto plots (2 datasets × 2 metrics) | Yes |
| `plot_optuna_study.py` | Optuna history, importance, slices, contours, parallel coords | Yes (needs journal.log) |
| `plot_power_analysis.py` | Power timelines, distributions, utilization | Yes (needs power_log CSVs) |
| `plot_mlp_resources.py` | CPU/RAM/GPU memory from resource_usage.json | Yes |
| `plot_initial_overfitting.py` | Overfitting study learning curves | Yes |
| `summarize_results.py` | Aggregate results JSONs → summary_table.csv | Yes |
| `visualize_architecture.py` | TikZ neural network diagrams | Yes |
| `select_knee_points.py` | Pareto knee-point model selection | Yes |
| `pareto_manager.py` | Pareto frontier computation from sweep CSV | Yes |

## SLURM Job Patterns

All SLURM scripts follow this template. Use existing scripts in `scripts/slurm/` as reference.

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --output=./slurm_jobs/<name>_%j.out
#SBATCH --error=./slurm_jobs/<name>_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:7
#SBATCH --mem=754G
#SBATCH --time=1-00:00:00
#SBATCH -w argon-gtx

mkdir -p ./slurm_jobs
module purge && module load cuda/12.4.1
source /home/barattts/lavoltabuona/BA/.venv/env/bin/activate
export PYTHONPATH=.
cd $SLURM_SUBMIT_DIR

# Power monitoring (background)
python monitoring/power_monitor.py --output-dir "$OUTPUT_DIR/power_monitor" &
MONITOR_PID=$!

# ... actual work ...

kill $MONITOR_PID 2>/dev/null
```

## Workflow

1. **Check the gap tracker** above to identify what's needed
2. **Determine execution environment** — local or SLURM? (see table)
3. **If local:** implement the script, run it, verify outputs, save artifacts
4. **If SLURM:** implement script + `.sbatch` wrapper, instruct user to `sbatch` it
5. **Update the gap tracker** — move completed items, add artifact paths
6. **Notify thesis side** — the `ba-data` skill in the thesis workspace knows how to extract and format BA artifacts as LaTeX
