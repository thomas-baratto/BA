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
| Seed sweep (N seeds × 3 winners) | **SLURM only** | `sbatch scripts/slurm/seed_sweep.sbatch` (default 100; use `--export=N_SEEDS=4096`) |
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

## Ausschreibung Requirements Checklist

From [BA/Ausschreibung.pdf](../../Ausschreibung.pdf) — primary deliverables and their status:

| Requirement | Status | Notes |
|-------------|--------|-------|
| **Compare MLP to ELM (+ RaNN variants)** on heat plume prediction | ✅ Done | 7 RaNN architectures compared against MLP on both datasets |
| **Thorough analysis**: metrics, data, training/inference times, model size | ✅ Done | Full analysis in §5.1–5.5 incl. inference latency (§5.3.1) |
| **Literature research**: MLPs, ELMs, random feature methods, RaNNs | ✅ Done | Ch. 2 (Background) + Ch. 3 (Related Work) fully written |
| **Overfit to 1, then 10 data points** to validate architecture | ✅ Done | §4.3.1 Initial Overfitting Tests, plots in appendix |
| **Optimize hyperparameters** (Optuna suggested) | ✅ Done | 10k+ Optuna trials for MLP, 467-config grid search for RaNN |
| **Baseline: Böttcher regression formula** for comparison | ✅ Done | `evaluate_boettcher_baseline.py` run; results in `artifacts/models/baseline/boettcher/results_boettcher_baseline.json`; thesis §5.2 has two quantitative tables |
| **Submit code via git** | ✅ Done | Full repository on GitHub |
| **Submit trained models via DaRUS** with DOI | ✅ Done | DOI: 10.18419/DARUS-5815, cited in thesis. 5 individually-zipped models + code zip uploaded to DaRUS with `models/` directory structure. Packaging script: `scripts/package_darus.sh`. |
| **Clean packaging** of best model for practitioners | ✅ Done | `ba-predict` CLI, `predict.py`, `INFERENCE_GUIDE.md`. Release branch is GitHub default. DaRUS has individual model zips for selective download. |
| **Clean, documented, reproducible code** with testing | ✅ Done | pytest suite, README, docstrings, type hints |
| **Optional**: Compare against models in Maheshwari et al. [5] | ✅ Discussed | Qualitative comparison in §5.6 Discussion |
| **Optional**: Compare against 1HP-CNN from Pelzer [3] | ✅ Discussed | Qualitative comparison in §5.6 Discussion |
| **Optional**: BayesValidRox | ❌ Skipped | Out of scope (acknowledged) |

## Thesis Gap Tracker

> Last updated: 2026-04-04. All thesis text is written (111 pages, zero warnings).
> Remaining gaps are 3 missing Optuna cone power plots (appendix only) and final proofreading.

### Thesis Content Status

All chapters have complete prose — no stubs or placeholder text remain.
The thesis compiles cleanly at 111 pages with no undefined references or warnings.

### Completed

| Item | Evidence |
|------|----------|
| All thesis chapters written (Intro, Background, Related Work, Methodology, Results, Conclusion) | Full prose in all sections |
| Abstract written | Lines 84–99 of main-english.tex |
| MLP trained on both datasets | `artifacts/models/mlp/cone/`, `artifacts/models/mlp/isotherm/` |
| RaNN sweep (467 configs) | `runs/run_sweep_random_1053/` |
| Pareto front analysis + plots | 4 PDFs in `thesis/graphics/plots/pareto/` |
| Optuna study plots (6 per dataset) | `thesis/graphics/plots/optuna_cone/` and `optuna_isotherm/` (history, importance, slices, contours, trial durations, parallel coords) |
| MLP power monitoring plots | `thesis/graphics/plots/power/mlp_cone/` (4 PDFs), `mlp_isotherm/` (4 PDFs) |
| Random winner power plots | `thesis/graphics/plots/power/random_winners/` (6 PDFs) |
| Architecture diagrams (8 of 8) | `thesis/graphics/plots/architecture/` — MLP, ELM, RVFL, dRVFL, edRVFL, edRVFL-SC, esc-edRVFL, SResdRVFL. Also `docs/plots/architecture/` has 5 model-specific diagrams. |
| Böttcher baseline evaluated | `artifacts/models/baseline/boettcher/results_boettcher_baseline.json` — metrics for all_disp (n=7140) + matched (n=454) subsets |
| Böttcher comparison in thesis | §5.2 with `tab:baseline-all` and `tab:baseline-matched` (R², KGE, MAPE, MAE, RMSE) |
| Inference latency benchmarked | `artifacts/models/benchmark_inference_results.json` + `docs/tables/inference_benchmark.tex` + `docs/tables/inference_scaling.tex` |
| Inference latency in thesis | §5.3.1 with `tab:inference-latency` and `tab:inference-scaling` |
| Deployment pipeline | `predict.py`, `package_models.py`, `retrain_random_models.py`, `INFERENCE_GUIDE.md` |
| LaTeX comparison table | `docs/tables/model_comparison.tex` |
| Limitations reduced to 4 | Removed stale items about missing baseline and inference data |
| Future Work reduced to 4 | Removed completed items (baseline comparison, inference benchmarking) |
| Critical numeric fixes (2026-04-04) | tab:results restructured to per-label format (11 rows) with 6dp precision from canonical JSONs; hidden layer counts corrected (5 iso, 2 cone) at 5 locations; training times reconciled (735s cone, 4572s iso); R²=0.995→0.991, KGE=0.991→0.993 |
| Böttcher baseline + inference in Abstract | Two new sentences on baseline negative R² and MLP inference <3 µs/sample |
| Böttcher baseline + inference in Contributions | New contribution item for baseline comparison; inference latency added to evaluation item |
| Böttcher baseline + inference in Discussion | Two new paragraphs: "Analytical baseline" and "Inference latency" |
| Böttcher baseline + inference in Summary | Two new enumerated findings: items 6 (baseline) and 7 (inference latency) |
| Trial count clarification | Abstract: "thousands of trials per dataset"; Summary: explicit per-dataset counts (6,800 iso, 10,000 cone) |
| Raw RVFL → \gls{rvfl} in edRVFL-SC caption | Style consistency fix |
| generic_rvfl.pdf uncommented | Figure block fully uncommented with caption and label |
| Parallel coordinate plots uncommented | Both cone and isotherm parallel coordinate figures uncommented (files existed) |
| Seed sweep infrastructure | `scripts/slurm/seed_sweep.sbatch` + `scripts/analysis/analyze_seed_sweep.py` + `--n-seeds` flag in `train_random_models.py`; runs N seeds for 3 RaNN winners (cone/edRVFL-SC, iso/SResdRVFL, iso/dRVFL); produces `multi_seed_summary.json`, LaTeX table, box plots |
| **4096-seed sweep (Job 1069)** | `runs/seed_sweep_1069/` — IN PROGRESS on argon-gtx, 4096 seeds (1–4096). isotherm_dRVFL done; others running. **PRIORITY: integrate results into thesis when complete.** |

### Open — Actionable Locally

| Priority | Gap | Status |
|----------|-----|--------|
| *Resolved* | Optuna cone energy/gpu/worker plots | Intentionally skipped; only power_timeline and utilization_timeline included (exist in BA/docs/plots/power/optuna_cone/) |

### Open — Requires SLURM / Server

| Priority | Gap | Notes |
|----------|-----|-------|
| **HIGH** | 4096-seed sweep results (Job 1069) | Running on argon-gtx. When done: (1) run `analyze_seed_sweep.py`, (2) update `server-data.md`, (3) add variance stats + box plots to §5.1, (4) revise §6.2 fourth limitation ("single seed" → now addressed with 4096 seeds) |
| **LOW** | Continuous RaNN optimization (Future Work §6.3) | New Optuna study for RaNN hyperparameters — acknowledged as future work |

### Out of Scope

Acknowledged as future work in the thesis — not feasible before the April 14 deadline:
- CNN / QCNN comparison (different output representation)
- Transfer to heterogeneous/transient settings (needs new datasets)
- Hybrid RaNN + gradient fine-tuning (new model class, research-level)
- BayesValidRox integration (optional per Ausschreibung, skipped)

## Analysis Scripts Reference

| Script | What it produces | Runs locally? |
|--------|-----------------|---------------|
| `evaluate_boettcher_baseline.py` | Böttcher vs NN metrics on Isotherm==1 test set | Yes |
| `benchmark_inference.py` | Per-model inference latency + batch scaling | Yes |
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
| `analyze_seed_sweep.py` | Multi-seed sweep LaTeX table + box plots | Yes (needs `multi_seed_summary.json`) |

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
