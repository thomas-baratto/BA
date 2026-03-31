"""Central registry of plot groups for the orchestrator.

Each group declares:
  - outputs:      glob patterns (relative to project root) for produced files
  - dependencies: file paths/globs that the outputs depend on
  - generate:     callable(project_root, force) that produces the plots
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PY = str(PROJECT_ROOT / ".venv" / "env" / "bin" / "python")


@dataclass
class PlotGroup:
    """A logical group of related plots."""

    name: str
    description: str
    outputs: list[str]
    dependencies: list[str]
    generate: Callable[[Path], None]
    # Extra CLI args the user might want to override
    extra_args: dict[str, str] = field(default_factory=dict)


# ── Generator helpers ───────────────────────────────────────────────────────


def _run_script(script: str, *args: str) -> None:
    """Run a project script with PYTHONPATH=. and the project venv."""
    cmd = [_PY, "-m", script, *args]
    env = {"PYTHONPATH": str(PROJECT_ROOT), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
    if result.returncode != 0:
        print(f"  ⚠  {script} exited with code {result.returncode}", file=sys.stderr)


def _gen_model_comparison(root: Path) -> None:
    """Generate model comparison plots + markdown report."""
    _run_script(
        "scripts.analysis.generate_model_comparison",
        "--sweep-dir",
        "runs/run_sweep_random_1053",
        "--output-dir",
        "docs/plots",
    )


def _gen_pareto(root: Path) -> None:
    _run_script(
        "scripts.analysis.plot_pareto_frontiers",
        "--summary-csv",
        "runs/run_sweep_random_1053/summary_table.csv",
        "--knee-csv",
        "runs/run_sweep_random_1053/knee_point_winners.csv",
        "--output-dir",
        "docs/plots/pareto",
    )


def _gen_mlp_resources(root: Path) -> None:
    _run_script(
        "scripts.analysis.plot_mlp_resources",
        "--output-dir",
        "docs/plots/power",
    )


def _gen_optuna_isotherm(root: Path) -> None:
    _run_script(
        "scripts.analysis.plot_optuna_study",
        "--journal",
        "runs/global_run_1049/optuna_journal_storage/journal.log",
        "--output-dir",
        "docs/plots/optuna_isotherm",
    )


def _gen_optuna_cone(root: Path) -> None:
    _run_script(
        "scripts.analysis.plot_optuna_study",
        "--journal",
        "runs/global_run_1054/optuna_journal_storage/journal.log",
        "--output-dir",
        "docs/plots/optuna_cone",
    )


def _gen_power_optuna(root: Path) -> None:
    _run_script(
        "scripts.analysis.plot_power_analysis",
        "--run-type",
        "optuna",
        "--output-dir",
        "docs/plots/power/optuna_isotherm",
    )


def _gen_power_optuna_cone(root: Path) -> None:
    _run_script(
        "scripts.analysis.plot_power_analysis",
        "--run-pattern",
        "run_20260328-19265*_Cone_worker*",
        "--output-dir",
        "docs/plots/power/optuna_cone",
        "--run-label",
        "Cone HPO Sweep",
        "--prefix",
        "",
    )


def _gen_power_random(root: Path) -> None:
    _run_script(
        "scripts.analysis.plot_power_analysis",
        "--run-type",
        "random",
        "--output-dir",
        "docs/plots/power",
    )


def _gen_initial_overfitting(root: Path) -> None:
    _run_script("scripts.analysis.plot_initial_overfitting")


# ── Registry ────────────────────────────────────────────────────────────────

PLOT_GROUPS: dict[str, PlotGroup] = {
    "model_comparison": PlotGroup(
        name="model_comparison",
        description="Regression + residual plots for all 5 models, MODEL_COMPARISON.md",
        outputs=[
            "docs/plots/mlp/regression_*.png",
            "docs/plots/mlp/residuals_*.png",
            "docs/plots/random/regression_*.png",
            "docs/plots/random/residuals_*.png",
            "docs/MODEL_COMPARISON.md",
        ],
        dependencies=[
            "artifacts/models/mlp/cone/best_model.pt",
            "artifacts/models/mlp/isotherm/best_model.pt",
            "artifacts/models/random/cone/winner/model.pkl",
            "artifacts/models/random/isotherm/nRMSE_winner/model.pkl",
            "artifacts/models/random/isotherm/KGE_winner/model.pkl",
            "data/Clean_Results_Isotherm.csv",
            "data/Depression_cones.csv",
            "runs/run_sweep_random_1053/summary_table.csv",
        ],
        generate=_gen_model_comparison,
    ),
    "pareto": PlotGroup(
        name="pareto",
        description="Pareto frontier plots (2 datasets × 2 metrics)",
        outputs=[
            "docs/plots/pareto/pareto_*.png",
        ],
        dependencies=[
            "runs/run_sweep_random_1053/summary_table.csv",
            "runs/run_sweep_random_1053/knee_point_winners.csv",
        ],
        generate=_gen_pareto,
    ),
    "mlp_resources": PlotGroup(
        name="mlp_resources",
        description="GPU memory, CPU/RAM, GPU utilisation for MLP training",
        outputs=[
            "docs/plots/power/mlp_cone/mlp_*.png",
            "docs/plots/power/mlp_isotherm/mlp_*.png",
        ],
        dependencies=[
            "artifacts/models/mlp/cone/resources/resource_usage.json",
            "artifacts/models/mlp/isotherm/resources/resource_usage.json",
        ],
        generate=_gen_mlp_resources,
    ),
    "optuna_isotherm": PlotGroup(
        name="optuna_isotherm",
        description="Optuna study plots for the isotherm dataset",
        outputs=[
            "docs/plots/optuna_isotherm/optuna_*.png",
        ],
        dependencies=[
            "runs/global_run_1049/optuna_journal_storage/journal.log",
        ],
        generate=_gen_optuna_isotherm,
    ),
    "optuna_cone": PlotGroup(
        name="optuna_cone",
        description="Optuna study plots for the cone dataset",
        outputs=[
            "docs/plots/optuna_cone/optuna_*.png",
        ],
        dependencies=[
            "runs/global_run_1054/optuna_journal_storage/journal.log",
        ],
        generate=_gen_optuna_cone,
    ),
    "power_optuna": PlotGroup(
        name="power_optuna",
        description="Power/energy analysis for Optuna HPO sweep (isotherm)",
        outputs=[
            "docs/plots/power/optuna_isotherm/optuna_*.png",
        ],
        dependencies=[
            "runs/run_20260325-19291*/power_monitor/power_log_*.csv",
        ],
        generate=_gen_power_optuna,
    ),
    "power_optuna_cone": PlotGroup(
        name="power_optuna_cone",
        description="Power/energy analysis for Optuna HPO sweep (cone)",
        outputs=[
            "docs/plots/power/optuna_cone/*.png",
        ],
        dependencies=[
            "runs/run_20260328-19265*_Cone_worker*/power_monitor/power_log_*.csv",
        ],
        generate=_gen_power_optuna_cone,
    ),
    "power_random": PlotGroup(
        name="power_random",
        description="Power/energy analysis for random model sweep",
        outputs=[
            "docs/plots/power/random_*.png",
        ],
        dependencies=[
            "runs/run_sweep_random_*/power_monitor/power_log_*.csv",
        ],
        generate=_gen_power_random,
    ),
    "initial_overfitting": PlotGroup(
        name="initial_overfitting",
        description="Overfitting proof-of-concept: MLP + ELM on 1 & 10 samples",
        outputs=[
            "docs/plots/initial_overfitting/cone/*.png",
            "docs/plots/initial_overfitting/isotherm/*.png",
        ],
        dependencies=[
            "data/Clean_Results_Isotherm.csv",
            "data/Depression_cones.csv",
        ],
        generate=_gen_initial_overfitting,
    ),
}
