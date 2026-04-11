---
name: thesis-plots
description: "Use when: creating, editing, or reviewing matplotlib plots for the thesis. Enforces thesis style guide (SciencePlots base, colour-blind palette, PDF output, consistent dimensions). Covers figure creation, styling, colour palette, save workflow, and output paths."
argument-hint: "Describe the plot — e.g. 'scatter plot of predicted vs true for cone dataset', 'Pareto frontier with model colours'"
---

# Thesis-Quality Plotting

Create consistent, publication-ready matplotlib plots that match the thesis style guide.

## When to Use

- Creating any new plot or figure for the thesis
- Editing an existing analysis script that produces figures
- Reviewing whether a plot follows thesis conventions
- Adding a plot to a new or existing analysis script

## Single Source of Truth

All style configuration lives in `core/thesis_style.py`. Never hardcode colours, font sizes, figure dimensions, or DPI — import them from that module.

## Required Boilerplate

Every script that produces plots must start with:

```python
from core.thesis_style import apply_thesis_style, COLORS, save_fig
apply_thesis_style()
```

If importing from `core/plotting.py`, the style is applied automatically on import — no extra call needed.

## Step-by-Step Procedure

### 1. Apply the Style

Call `apply_thesis_style()` once at module level (or rely on `core/plotting.py` doing it). This sets:
- SciencePlots `science` + `no-latex` + `grid` base
- Serif fonts (Times-like), 10pt base, axis labels 11pt, title 12pt
- Colour cycle from `COLORS` palette
- Grid alpha 0.4, linewidth 0.5
- savefig DPI 300, bbox tight

### 2. Choose Figure Size

Import the appropriate constant from `core/thesis_style`:

| Constant | Dimensions | Use Case |
|----------|-----------|----------|
| `FIG_SINGLE` | 5.5 × 4.0 in | Default single-column figure |
| `FIG_SQUARE` | 5.5 × 5.5 in | Regression / scatter plots |
| `FIG_WIDE` | 7.2 × 4.5 in | Wide comparison, bar charts, timelines |
| `FIG_TALL` | 7.2 × 8.0 in | Multi-panel or tall figures |

```python
from core.thesis_style import FIG_SINGLE, FIG_SQUARE, FIG_WIDE
fig, ax = plt.subplots(figsize=FIG_SINGLE)
```

### 3. Use the Colour Palette

**General-purpose colours** — import `COLORS`:

| Key | Hex | Role |
|-----|-----|------|
| `primary` | `#2c7bb6` | Main scatter, primary data |
| `secondary` | `#d7191c` | Ideal lines, reference |
| `accent1` | `#fdae61` | Residuals, orange accent |
| `accent2` | `#abd9e9` | Light blue, secondary scatter |
| `accent3` | `#1a9641` | Green, good/pass indicators |
| `grid` | `#cccccc` | Grid lines (auto-set) |
| `text` | `#333333` | Text colour (auto-set) |

**Model-type colours** — import `MODEL_COLORS` for Pareto/comparison plots:

| Key | Hex |
|-----|-----|
| `ELM` | `#d62728` |
| `SResdRVFL` | `#bcbd22` |
| `dRVFL` | `#2ca02c` |
| `edRVFL` | `#00bfff` |
| `edRVFL-SC` | `#1f77b4` |
| `esc-edRVFL` | `#e377c2` |
| `MLP` | `#ff7f0e` |

```python
from core.thesis_style import COLORS, MODEL_COLORS
ax.scatter(x, y, color=COLORS["primary"])
ax.bar(models, scores, color=[MODEL_COLORS[m] for m in models])
```

Never hardcode hex values — always reference the dictionaries.

### 4. Label Axes with Units

Use `label_with_unit()` to build axis labels with LaTeX-formatted units:

```python
from core.thesis_style import label_with_unit
ax.set_xlabel(label_with_unit("True Area"))       # → "True Area ($\mathrm{m^2}$)"
ax.set_ylabel(label_with_unit("Predicted Cone"))   # → "Predicted Cone ($\mathrm{m}$)"
```

Units are looked up automatically from `core/metrics.LABEL_UNITS`. Pass an explicit unit to override:

```python
ax.set_ylabel(label_with_unit("Residuals", "m²"))
```

### 5. Save as PDF

Always use `save_fig()` — never call `fig.savefig()` directly:

```python
from core.thesis_style import save_fig
save_fig(fig, output_dir / "regression_cone")
```

Key behaviour:
- Default format: **PDF only** (`formats=("pdf",)`)
- Creates parent directories automatically
- Closes the figure after saving (pass `close=False` to keep it open)
- Uses DPI=300 and `bbox_inches="tight"`
- Extension is replaced per format, so pass the path without extension or with any extension

To also save PNG (e.g. for documentation):
```python
save_fig(fig, path, formats=("pdf", "png"))
```

### 6. Place Output Files

| Destination | When |
|-------------|------|
| `thesis/graphics/plots/<category>/` | Figures referenced by `\includegraphics` in thesis |
| `docs/plots/` | Intermediate or documentation figures in BA repo |
| Script-specific `--output-dir` arg | When the script accepts an output path |

The thesis `\includegraphics` paths omit the extension — LaTeX auto-discovers `.pdf`. Example:
```latex
\includegraphics[width=0.7\linewidth]{graphics/plots/mlp/cone/regression_cone}
```

## Checklist

Before finalising any plot:

- [ ] `apply_thesis_style()` called (or inherited from `core/plotting.py`)
- [ ] Figure size from `FIG_SINGLE` / `FIG_SQUARE` / `FIG_WIDE` / `FIG_TALL`
- [ ] Colours from `COLORS` or `MODEL_COLORS` — no hardcoded hex
- [ ] Axis labels use `label_with_unit()` where applicable
- [ ] Saved via `save_fig()` producing PDF
- [ ] Output path is correct for thesis or docs
- [ ] No `plt.show()` in scripts (only in notebooks/debug)
- [ ] Legend uses `ax.legend()` — positioned automatically or with `loc=`

## Common Patterns

### Regression Scatter (predicted vs actual)

```python
fig, ax = plt.subplots(figsize=FIG_SQUARE)
ax.scatter(true, pred, alpha=0.4, s=12, color=COLORS["primary"], edgecolors="none")
lims = [min(true.min(), pred.min()) * 0.95, max(true.max(), pred.max()) * 1.05]
ax.plot(lims, lims, color=COLORS["secondary"], ls="--", lw=1.5, label=r"Ideal ($y = x$)")
ax.set_xlabel(label_with_unit(f"True {label}"))
ax.set_ylabel(label_with_unit(f"Predicted {label}"))
ax.set_aspect("equal", adjustable="box")
ax.legend()
save_fig(fig, out / f"regression_{label}")
```

### Loss Curve (training + validation, dual axis for test)

Training and validation loss on left y-axis (log scale), test loss on right y-axis when
the scale differs significantly. Always include both train and val; add test only when
available.

```python
fig, ax = plt.subplots(figsize=FIG_WIDE)
ax.semilogy(epochs, train_losses, label="Training Loss", color=COLORS["primary"])
ax.semilogy(epochs, val_losses, label="Validation Loss", color=COLORS["accent1"], ls="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (log scale)")

# If test loss is on a different scale, use a secondary y-axis
if test_losses is not None:
    ax2 = ax.twinx()
    ax2.semilogy(epochs, test_losses, label="Test Loss",
                 color=COLORS["secondary"], ls=":", lw=1.2)
    ax2.set_ylabel("Test Loss (log scale)")
    ax2.legend(loc="center right")

ax.legend(loc="upper right")
ax.set_title(f"Training Progress — {dataset}")
save_fig(fig, out / f"loss_{dataset}")
```

Key rules for loss plots:
- Always show both training and validation curves
- Use `semilogy` for loss (values span orders of magnitude)
- Use `COLORS["primary"]` for train, `COLORS["accent1"]` (dashed) for val, `COLORS["secondary"]` (dotted) for test
- When test loss has a very different scale, use `ax.twinx()` for a right y-axis
- Add a secondary x-axis for wall-clock time when `epoch_times` are available (see `plot_training_curves.py` in `scripts/analysis/`)

### Bar Chart Comparing Models

```python
fig, ax = plt.subplots(figsize=FIG_WIDE)
bars = ax.bar(model_names, scores, color=[MODEL_COLORS.get(m, COLORS["primary"]) for m in model_names])
ax.set_ylabel(label_with_unit("RMSE", unit))
ax.set_title("Model Comparison")
save_fig(fig, out / "model_comparison")
```

### Heatmap (e.g. correlation matrix, confusion-style)

```python
fig, ax = plt.subplots(figsize=FIG_SQUARE)
im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, shrink=0.8)

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)

# Annotate cells
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(matrix[i, j]) > 0.6 else COLORS["text"], fontsize=8)

ax.set_title("Feature Correlation")
save_fig(fig, out / "correlation_heatmap")
```

### Subplots / Multi-Panel Figures

**Two panels side-by-side** (matches thesis `0.48\linewidth` subfigure layout):

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_WIDE)
# ... plot on ax1 and ax2 ...
fig.tight_layout()
save_fig(fig, out / "comparison_two_panel")
```

**2×2 grid with shared axes**:

```python
fig, axes = plt.subplots(2, 2, figsize=FIG_TALL, sharex=True, sharey=True)
for ax, label in zip(axes.flat, labels):
    ax.scatter(true[label], pred[label], alpha=0.4, s=12, color=COLORS["primary"])
    ax.set_title(label)
fig.supxlabel("True Value")
fig.supylabel("Predicted Value")
fig.tight_layout()
save_fig(fig, out / "grid_regression")
```

**Vertical stack** (e.g. multiple loss components):

```python
fig, axes = plt.subplots(3, 1, figsize=FIG_TALL, sharex=True)
for ax, (name, data) in zip(axes, series.items()):
    ax.plot(epochs, data, color=COLORS["primary"])
    ax.set_ylabel(name)
axes[-1].set_xlabel("Epoch")
fig.tight_layout()
save_fig(fig, out / "loss_components")
```

Rules for multi-panel figures:
- Always call `fig.tight_layout()` or pass `constrained_layout=True` to avoid label overlap
- Use `sharex`/`sharey` when panels share an axis to reduce clutter
- Use `fig.supxlabel()` / `fig.supylabel()` for shared axis labels
- Match thesis subfigure layout: 2 panels → `FIG_WIDE`, 2×2 or vertical stacks → `FIG_TALL`

### Existing Plot Functions

Before writing a new plot from scratch, check if `core/plotting.py` already has what you need:

| Function | Purpose |
|----------|---------|
| `create_regression_plots()` | Predicted vs actual scatter with R² and RMSE |
| `create_residual_plots()` | Residuals vs true values |
| `create_qq_plots()` | Q-Q normality check on residuals |
| `create_scatter_plot()` | Simple scatter with ideal line |

> **Note:** `plot_results()` and `plot_split_metric_bars()` still exist in `core/plotting.py`
> but are **no longer called during training**. Training now saves `.npz` data files;
> all plot generation happens in the analysis pipeline (`scripts/analysis/`).

Dedicated analysis scripts (in `scripts/analysis/`):

| Script | Purpose |
|--------|----------|
| `plot_training_curves.py` | Loss curves, QQ plots, metric bar charts from `.npz` data |
| `plot_distributions.py` | Feature/label distribution histograms from `.npz` data |
| `plot_initial_overfitting.py` | ELM/MLP overfitting study learning curves |
| `plot_pareto_frontiers.py` | Pareto efficiency plots (accuracy vs. training time) |
| `plot_optuna_study.py` | Hyperparameter optimization visualizations |
| `plot_mlp_resources.py` | GPU/CPU resource consumption plots |
| `plot_power_analysis.py` | Energy/power consumption plots |

These already follow all conventions and call `save_fig()`.
