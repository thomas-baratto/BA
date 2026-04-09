"""Thesis-quality matplotlib style configuration.

Based on the SciencePlots ``science`` style (standard in CS/engineering
papers) with thesis-specific overrides for figure size, colour palette,
and grid.  Requires ``pip install SciencePlots``.

The ``science`` base provides:
  • Serif (Times-like) font family with proper math rendering
  • Thin axes and tick styling matching IEEE / Springer / ACM conventions
  • Clean legend and label defaults

Usage
-----
    from core.thesis_style import apply_thesis_style, COLORS, save_fig
    apply_thesis_style()
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ── Colour palette (colour-blind safe, print-friendly) ──────────────────────
COLORS = {
    "primary": "#2c7bb6",       # steel blue  – main scatter
    "secondary": "#d7191c",     # red         – ideal lines, reference
    "accent1": "#fdae61",       # orange      – residuals
    "accent2": "#abd9e9",       # light blue  – secondary scatter
    "accent3": "#1a9641",       # green       – good/pass
    "grid": "#cccccc",
    "text": "#333333",
}

# Model-type colours (for Pareto / comparison plots)
MODEL_COLORS = {
    "ELM": "#d62728",
    "SResdRVFL": "#bcbd22",
    "dRVFL": "#2ca02c",
    "edRVFL": "#00bfff",
    "edRVFL-SC": "#1f77b4",
    "esc-edRVFL": "#e377c2",
    "MLP": "#ff7f0e",
}

# ── Consistent figure dimensions ────────────────────────────────────────────
# Single-column thesis figure: ~5.5 in wide; double-column ~7.2 in.
FIG_SINGLE = (5.5, 4.0)
FIG_SQUARE = (5.5, 5.5)
FIG_WIDE = (7.2, 4.5)
FIG_TALL = (7.2, 8.0)

DPI = 300  # thesis-quality raster resolution


def apply_thesis_style() -> None:
    """Apply a publication-quality matplotlib style based on ``science``.

    Uses the SciencePlots ``science`` + ``no-latex`` + ``grid`` styles as
    the foundation (IEEE/ACM-standard fonts and layout), then layers
    thesis-specific overrides for figure size, colours, and DPI.
    """
    # ── Base: SciencePlots "science" style (serif fonts, clean layout) ──
    import scienceplots  # noqa: F401 – registers styles on import
    plt.style.use(["science", "no-latex", "grid"])

    # ── Thesis overrides on top of the science base ─────────────────────
    overrides = {
        # ── Figure dimensions (wider than IEEE single-column default) ───
        "figure.figsize": FIG_SINGLE,
        "figure.dpi": 100,          # screen preview; savefig uses DPI
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # ── Font sizes (scaled up from science 8pt for thesis A4 page) ──
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # ── Colour cycle ────────────────────────────────────────────────
        "axes.prop_cycle": mpl.cycler(
            color=[
                COLORS["primary"],
                COLORS["secondary"],
                COLORS["accent1"],
                COLORS["accent3"],
                COLORS["accent2"],
            ]
        ),
        # ── Grid (lighter than science default) ─────────────────────────
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
        # ── Lines / scatter ─────────────────────────────────────────────
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "scatter.marker": "o",
        # ── Legend ──────────────────────────────────────────────────────
        "legend.framealpha": 0.9,
        "legend.edgecolor": COLORS["grid"],
    }
    mpl.rcParams.update(overrides)


def label_with_unit(name: str, unit: str | None = None) -> str:
    """Build an axis label like ``'True Area [m²]'``.

    Looks up the unit from ``LABEL_UNITS`` by trying the full *name* first,
    then each word in *name* (so ``"True Area"`` finds the unit for ``"Area"``).
    An explicit *unit* argument overrides the lookup.
    """
    from core.metrics import LABEL_UNITS

    if unit is None:
        # Try full name first, then each word (right-to-left so the noun wins)
        unit = LABEL_UNITS.get(name, "")
        if not unit:
            for word in reversed(name.split()):
                unit = LABEL_UNITS.get(word, "")
                if unit:
                    break

    if not unit:
        return name
    return f"{name} [{unit}]"


def save_fig(
    fig: Figure,
    path: str | Path,
    *,
    close: bool = True,
    formats: tuple[str, ...] = ("pdf",),
) -> None:
    """Save a figure in multiple formats (PDF + PNG by default).

    Parameters
    ----------
    fig : matplotlib Figure
    path : Output path.  The suffix is replaced for each format, so passing
        ``"plots/regression_cone.png"`` will produce both
        ``regression_cone.pdf`` and ``regression_cone.png``.
    close : Whether to close the figure after saving.
    formats : Tuple of file extensions to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt, dpi=DPI, bbox_inches="tight")

    if close:
        plt.close(fig)
