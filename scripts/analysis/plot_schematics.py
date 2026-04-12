"""Generate schematic diagrams of thermal plume and depression cone.

Produces two thesis-quality PDF figures:
  1. Thermal plume (plan view) with Area, Iso_distance, and Iso_width labelled
  2. Depression cone (cross-section) with Cone (max drawdown) labelled

Usage:
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/plot_schematics.py
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.special import exp1

from core.thesis_style import COLORS, FIG_SINGLE, apply_thesis_style, save_fig

apply_thesis_style()

OUTPUT_DIR = Path(__file__).resolve().parents[2] / ".." / "thesis" / "graphics" / "plots"


# ─── Thermal plume (plan view) ──────────────────────────────────────────────


def _plume_outline(n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) arrays tracing a smooth elongated plume boundary.

    Uses cos + cos² for the x-component, which gives smooth asymmetric
    elongation without any derivative discontinuities (no bumps).
    Plain sin for width keeps the profile perfectly smooth.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=True)
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    # Smooth asymmetric x: downstream 0.70, upstream 0.30 (no np.where kink)
    x = 0.50 * cos_t + 0.20 * cos_t**2

    # Smooth symmetric width
    y = sin_t * 0.40

    return x, y


def plot_thermal_plume() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    # ── Dimensions ──
    iso_dist = 8.5   # well → downstream tip
    well_x = 1.5     # plume extends this far upstream of well
    iso_width = 3.4  # maximum transverse width

    # ── Build plume shape ──
    xp, yp = _plume_outline()
    x_min_raw, x_max_raw = xp.min(), xp.max()
    total_raw = x_max_raw - x_min_raw
    total_vis = well_x + iso_dist
    xp = (xp - x_min_raw) / total_raw * total_vis - well_x
    y_range = yp.max() - yp.min()
    yp = yp / y_range * iso_width

    plume_tip_x = xp.max()
    plume_back_x = xp.min()

    # ── Filled plume ──
    ax.fill(xp, yp, color=COLORS["accent2"], alpha=0.35, zorder=2)
    ax.plot(xp, yp, color=COLORS["primary"], lw=1.6, zorder=3)

    # ── Injection well ──
    ax.plot(0, 0, "s", color=COLORS["text"], ms=6, zorder=5, markeredgecolor="k")
    ax.annotate(
        "Injection well",
        xy=(0, 0),
        xytext=(-1.8, 2.3),
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=COLORS["text"], lw=0.9,
                        connectionstyle="arc3,rad=-0.15"),
        color=COLORS["text"],
        zorder=6,
    )

    # ── Iso_distance — horizontal dimension below plume ──
    # Thin extension lines from well and tip down to dimension line
    y_dim = -2.8
    ext_gap = 0.15
    for xx in [0, plume_tip_x]:
        # Find plume bottom at this x
        lower_mask = yp < 0
        x_lower = xp[lower_mask]
        y_lower = yp[lower_mask]
        sort_idx = np.argsort(x_lower)
        bottom = np.interp(xx, x_lower[sort_idx], y_lower[sort_idx], left=-0.3, right=-0.3)
        ax.plot([xx, xx], [bottom - ext_gap, y_dim + ext_gap],
                ls="-", color=COLORS["text"], lw=0.5, zorder=3)

    ax.annotate(
        "", xy=(plume_tip_x, y_dim), xytext=(0, y_dim),
        arrowprops=dict(arrowstyle="<->", color=COLORS["text"], lw=1.1,
                        shrinkA=0, shrinkB=0),
        zorder=4,
    )
    ax.text(
        plume_tip_x / 2, y_dim - 0.3,
        "Iso_distance",
        ha="center", va="top", fontsize=10,
        fontstyle="italic", color=COLORS["text"],
        zorder=6,
    )

    # ── Iso_width — vertical dimension to the right of plume ──
    # Find the actual widest point
    idx_max = np.argmax(yp)
    hw = yp[idx_max]
    x_at_widest = xp[idx_max]

    # Place vertical dimension line to the right of the plume
    dim_x = plume_tip_x + 0.8

    # Extension lines from widest-point boundary to dimension line
    ax.plot([x_at_widest + 0.15, dim_x + 0.15], [hw, hw],
            ls="-", color=COLORS["text"], lw=0.5, zorder=3)
    ax.plot([x_at_widest + 0.15, dim_x + 0.15], [-hw, -hw],
            ls="-", color=COLORS["text"], lw=0.5, zorder=3)

    ax.annotate(
        "", xy=(dim_x, hw), xytext=(dim_x, -hw),
        arrowprops=dict(arrowstyle="<->", color=COLORS["text"], lw=1.1,
                        shrinkA=0, shrinkB=0),
        zorder=4,
    )
    ax.text(
        dim_x + 0.3, 0,
        "Iso_width",
        ha="left", va="center", fontsize=10,
        fontstyle="italic", color=COLORS["text"],
        rotation=90,
        zorder=6,
    )

    # ── Area label (inside the plume) ──
    ax.text(
        plume_tip_x * 0.45, 0,
        "Area",
        ha="center", va="center", fontsize=11,
        fontstyle="italic", fontweight="bold",
        color=COLORS["primary"],
        zorder=6,
    )

    # ── Δ T isotherm label along the boundary ──
    ax.text(
        plume_tip_x * 0.62, hw * 0.38,
        r"$\Delta T$ isotherm",
        ha="center", va="bottom", fontsize=8.5,
        color=COLORS["primary"], rotation=-6,
        zorder=6,
    )

    # ── Groundwater flow arrow ──
    flow_y = hw + 1.2
    ax.annotate(
        "", xy=(plume_tip_x - 0.5, flow_y), xytext=(plume_tip_x - 3.5, flow_y),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["text"], lw=1.5,
                        mutation_scale=14),
        zorder=4,
    )
    ax.text(
        plume_tip_x - 2.0, flow_y + 0.35,
        "Groundwater flow",
        ha="center", va="bottom", fontsize=9,
        color=COLORS["text"],
        zorder=6,
    )

    # ── Axes — minimal, schematic ──
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlim(plume_back_x - 1.5, dim_x + 2.5)
    ax.set_ylim(-3.8, flow_y + 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    save_fig(fig, OUTPUT_DIR / "thermal_plume_schematic")
    print(f"  Saved: {OUTPUT_DIR / 'thermal_plume_schematic.pdf'}")


# ─── Depression cone (cross-section) ────────────────────────────────────────


def _theis_drawdown(
    r: np.ndarray,
    Q: float = 0.02,
    T: float = 5e-3,
    S: float = 1e-4,
    t: float = 86400 * 30,
) -> np.ndarray:
    """Compute Theis drawdown s(r) for given parameters."""
    u = r**2 * S / (4 * T * t)
    return Q / (4 * np.pi * T) * exp1(u)


def plot_depression_cone() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    # ── Drawdown curve ──
    r = np.linspace(0.3, 120, 500)
    s = _theis_drawdown(r)

    # Normalise for visual clarity
    s_max = s.max()
    s_norm = s / s_max  # 0..1
    r_norm = r / r.max()

    # Work in display coordinates: x = distance from well, y = depth below water table
    water_table_y = 0
    max_depth = -4.0  # visual depth of cone at the well
    width = 10.0  # visual half-extent

    x_plot = np.concatenate([-r_norm[::-1] * width, r_norm * width])
    y_plot = np.concatenate([-(s_norm[::-1]) * abs(max_depth), -(s_norm) * abs(max_depth)])

    # ── Static water table (dashed) ──
    ax.hlines(
        y=water_table_y,
        xmin=-width - 0.5,
        xmax=width + 0.5,
        color=COLORS["primary"],
        lw=1.3,
        ls="--",
        zorder=3,
    )
    ax.text(
        width + 0.7,
        water_table_y,
        "Static water table",
        fontsize=9,
        va="center",
        color=COLORS["primary"],
    )

    # ── Drawdown curve (depressed water table) ──
    ax.fill_between(
        x_plot,
        water_table_y,
        y_plot,
        color=COLORS["accent2"],
        alpha=0.35,
        zorder=2,
    )
    ax.plot(x_plot, y_plot, color=COLORS["primary"], lw=1.6, zorder=4)

    # ── Well marker ──
    ax.plot(0, 0, "s", color=COLORS["text"], ms=6, zorder=5, markeredgecolor="k")
    ax.annotate(
        "Extraction well",
        xy=(0, 0),
        xytext=(-3.0, 1.8),
        fontsize=9,
        ha="center",
        arrowprops=dict(
            arrowstyle="-|>", color=COLORS["text"], lw=0.9,
            connectionstyle="arc3,rad=-0.15",
        ),
        color=COLORS["text"],
        zorder=6,
    )

    # ── Cone dimension arrow ──
    cone_depth = y_plot.min()

    # Extension lines from water table and cone bottom out to dimension line
    dim_x = -width - 0.8  # place to the LEFT of the diagram
    ext_gap = 0.12
    # Left edge of the drawdown curve at water table and at cone bottom
    ax.plot([dim_x - 0.1, -ext_gap], [water_table_y, water_table_y],
            ls="-", color=COLORS["text"], lw=0.5, zorder=3)
    ax.plot([dim_x - 0.1, -ext_gap], [cone_depth, cone_depth],
            ls="-", color=COLORS["text"], lw=0.5, zorder=3)

    ax.annotate(
        "",
        xy=(dim_x, cone_depth),
        xytext=(dim_x, water_table_y),
        arrowprops=dict(arrowstyle="<->", color=COLORS["text"], lw=1.1,
                        shrinkA=0, shrinkB=0),
        zorder=6,
    )
    ax.text(
        dim_x - 0.3,
        (water_table_y + cone_depth) / 2,
        "Cone",
        ha="right",
        va="center",
        fontsize=10,
        fontstyle="italic",
        color=COLORS["text"],
        zorder=6,
    )

    # ── Axes — minimal, schematic (matching thermal plume style) ──
    ax.grid(False)
    ax.set_xlim(-width - 2.5, width + 5.0)
    ax.set_ylim(max_depth - 1.0, 2.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    save_fig(fig, OUTPUT_DIR / "depression_cone_schematic")
    print(f"  Saved: {OUTPUT_DIR / 'depression_cone_schematic.pdf'}")


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating schematic diagrams...")
    plot_thermal_plume()
    plot_depression_cone()
    print("Done.")
