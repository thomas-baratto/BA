"""Generate standalone TikZ (.tex) neural-network architecture diagrams.

Reads model_config.json files and produces compilable LaTeX documents
with publication-quality architecture diagrams.

Design:
  - Input/output layers draw individual circle nodes (meaningful labels).
  - Hidden layers use rectangular tensor blocks (no spaghetti connections).
  - Single thick arrows between layers instead of all-to-all node connections.
  - Random-frozen weights shown as dashed gray edges; learned as solid bold.
  - Ensemble uses stacked 3D planes (offset copies) instead of bounding box.
  - Direct links routed as arcs above the network to avoid crossover.

Usage:
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/visualize_architecture.py \
        --config artifacts/models/mlp/isotherm/model_config.json
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/visualize_architecture.py \
        --all-artifacts --compile
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINNERS_PATH = PROJECT_ROOT / "config" / "random_model_winners.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "models"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "plots" / "architecture"

MAX_DRAWN_NODES = 5  # input/output nodes before inserting ellipsis

TIKZ_COLORS = {
    "inputcol": "0.17, 0.48, 0.71",
    "hiddencol": "1.00, 0.50, 0.06",
    "outputcol": "0.84, 0.15, 0.16",
    "frozencol": "0.17, 0.63, 0.17",
    "ensemblecol": "0.00, 0.75, 1.00",
    "residualcol": "0.74, 0.74, 0.13",
    "directcol": "0.55, 0.22, 0.08",
    "learnedcol": "0.12, 0.30, 0.60",
}


def _build_preamble() -> str:
    """Build the LaTeX standalone preamble with styles."""
    color_defs = "\n".join(
        r"\definecolor{" + n + "}{rgb}{" + v + "}"
        for n, v in TIKZ_COLORS.items()
    )
    return (
        r"\documentclass[border=12pt,tikz]{standalone}" "\n"
        r"\usepackage[T1]{fontenc}" "\n"
        r"\usepackage{tikz}" "\n"
        r"\usetikzlibrary{positioning, arrows.meta, calc, fit,"
        r" backgrounds, decorations.pathreplacing, shapes.geometric}" "\n\n"
        + color_defs + "\n\n"
        # ── node styles ──
        r"\tikzset{" "\n"
        r"  ionode/.style={circle, draw, minimum size=6.5mm, inner sep=0pt," "\n"
        r"                 font=\scriptsize, line width=0.45pt}," "\n"
        r"  input node/.style={ionode, fill=inputcol!20, draw=inputcol!70}," "\n"
        r"  output node/.style={ionode, fill=outputcol!20, draw=outputcol!70}," "\n"
        # tensor block (hidden layer)
        r"  tensor/.style={rectangle, draw, rounded corners=2pt," "\n"
        r"                 minimum width=12mm, minimum height=25mm," "\n"
        r"                 inner sep=3pt, font=\scriptsize, align=center," "\n"
        r"                 line width=0.6pt}," "\n"
        r"  hidden block/.style={tensor, fill=hiddencol!15, draw=hiddencol!70}," "\n"
        r"  frozen block/.style={tensor, fill=frozencol!12, draw=frozencol!60}," "\n"
        # sum / avg
        r"  sum node/.style={circle, draw, minimum size=7mm, inner sep=0pt," "\n"
        r"                   font=\normalsize, line width=0.5pt}," "\n"
        r"  avg node/.style={circle, draw, minimum size=7mm, inner sep=0pt," "\n"
        r"                   font=\scriptsize, line width=0.5pt," "\n"
        r"                   fill=ensemblecol!12}," "\n"
        # misc
        r"  dots node/.style={font=\Large, inner sep=0pt}," "\n"
        r"  annot/.style={font=\scriptsize, text=black!65}," "\n"
        r"  lbl/.style={font=\scriptsize\bfseries," "\n"
        r"              text=black!80, align=center}," "\n"
        # edges
        r"  arr/.style={-{Stealth[length=2.5pt]}, line width=0.55pt, black!50}," "\n"
        r"  random arr/.style={arr, densely dashed, frozencol!70}," "\n"
        r"  learned arr/.style={arr, line width=0.9pt, learnedcol}," "\n"
        r"  direct arr/.style={-{Stealth[length=3pt]}, line width=0.7pt," "\n"
        r"                     directcol, densely dotted}," "\n"
        # boxes
        r"  resbox/.style={draw=residualcol!55, dashed, rounded corners=3pt," "\n"
        r"                 fill=residualcol!6, line width=0.55pt}," "\n"
        r"}" "\n"
    )


PREAMBLE = _build_preamble()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    # Don't escape inside $...$ math delimiters (preserves subscripts like $x_1$)
    if s.startswith("$") and s.endswith("$"):
        return s
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _io_positions(n: int, spacing: float = 0.9):
    """Return list of (index_or_None, y) for drawing IO nodes, centred on 0."""
    indices, has_ell = _split_indices(n)
    n_slots = len(indices) + (1 if has_ell else 0)
    top = (n_slots - 1) * spacing / 2
    ys: list[tuple[int | None, float]] = []
    si = 0
    ell_done = False
    half = MAX_DRAWN_NODES // 2
    for i in range(n_slots):
        y = top - i * spacing
        if has_ell and not ell_done and si == half:
            ys.append((None, y))  # ellipsis slot
            ell_done = True
        else:
            ys.append((indices[si], y))
            si += 1
    return ys


def _split_indices(n: int):
    if n <= MAX_DRAWN_NODES:
        return list(range(n)), False
    top = MAX_DRAWN_NODES // 2
    bot = MAX_DRAWN_NODES - top
    return list(range(top)) + list(range(n - bot, n)), True


def _draw_io_column(
    lines: list[str], prefix: str, x: float, names: list[str],
    style: str, spacing: float = 0.9, label_side: str = "left",
    label_names: list[str] | None = None,
) -> list[str]:
    """Draw input/output circle nodes.  Returns list of TikZ node ids."""
    n = len(names)
    slots = _io_positions(n, spacing)
    node_ids: list[str] = []
    real_i = 0
    for idx, y in slots:
        if idx is None:
            lines.append(
                "    " + r"\node[dots node] (" + prefix + "_dots) at ("
                + f"{x:.2f},{y:.2f}" + r") {$\vdots$};"
            )
            continue
        nid = f"{prefix}_{real_i}"
        lines.append(
            "    " + r"\node[" + style + "] (" + nid + ") at ("
            + f"{x:.2f},{y:.2f}" + ") {};"
        )
        if label_names:
            lbl = _esc(label_names[idx])
            side = "left" if label_side == "left" else "right"
            lines.append(
                "    " + r"\node[annot, " + side + "=2.5mm of " + nid
                + "] {" + lbl + "};"
            )
        node_ids.append(nid)
        real_i += 1
    return node_ids


def _col_header(lines: list[str], x: float, top_y: float, text: str):
    """Add a header label above a column."""
    lines.append(
        "    " + r"\node[lbl, above=5mm] at ("
        + f"{x:.2f},{top_y:.2f}" + ") {" + text + "};"
    )


_SUPPRESS_LEGEND = False


def _draw_legend(lines: list[str], entries: list[tuple[str, str]]) -> None:
    """Append a compact legend box anchored below-right of the diagram.

    *entries*: list of (tikz_arrow_style, label_text) pairs.
    """
    if not entries or _SUPPRESS_LEGEND:
        return
    lines.append("")
    lines.append("    % legend")
    lines.append(
        "    " + r"\begin{scope}[shift={($(current bounding box.south east)"
        + r"+(0.3,-0.6)$)}]"
    )
    # background frame
    row_h = 0.4
    pad = 0.15
    h = len(entries) * row_h + 2 * pad
    w = 3.6
    lines.append(
        "    " + r"\fill[white, rounded corners=2pt, draw=black!25,"
        + r" line width=0.35pt]"
        + f" (0,0) rectangle (-{w:.1f},-{h:.2f});"
    )
    for i, (style, label) in enumerate(entries):
        y = -(pad + row_h * i + row_h / 2)
        x0 = -(w - pad)
        x1 = x0 + 0.7
        xt = x1 + 0.2
        lines.append(
            "    " + r"\draw[" + style + "] ("
            + f"{x0:.2f},{y:.2f}) -- ({x1:.2f},{y:.2f});"
        )
        lines.append(
            "    " + r"\node[anchor=west, font=\scriptsize] at ("
            + f"{xt:.2f},{y:.2f}" + ") {" + label + "};"
        )
    lines.append("    " + r"\end{scope}")


def _draw_title(lines: list[str], display_name: str, dataset: str) -> None:
    """Append a bold title centred above the diagram."""
    title_text = display_name
    if dataset:
        title_text += " (" + _esc(dataset.capitalize()) + ")"
    lines.append("")
    lines.append("    % title")
    lines.append(
        "    " + r"\node[font=\bfseries, above=8mm]"
        + r" at (current bounding box.north) {" + title_text + "};"
    )


# ── MLP renderer ─────────────────────────────────────────────────────────────

def render_mlp(cfg: dict[str, Any]) -> str:
    features = cfg["feature_names"]
    labels = cfg["label_names"]
    # NeuralNetwork builds 1 "input projection" layer + nr_hidden_layers extra
    # hidden layers, so the total number of hidden blocks is nr_hidden_layers + 1.
    n_hidden = cfg["nr_hidden_layers"] + 1
    n_neurons = cfg["nr_neurons"]
    activation = cfg.get("activation_name", "ReLU")
    dropout = cfg.get("dropout_rate", 0.0)
    use_bn = cfg.get("use_batchnorm", False)

    x_gap = 2.8
    lines: list[str] = []

    # ── Input ────────────────────────────────────────────────────────────
    x_in = 0.0
    in_ids = _draw_io_column(
        lines, "in", x_in, features, "input node",
        label_side="left", label_names=features,
    )
    top_y = _io_positions(len(features))[0][1]
    _col_header(lines, x_in, top_y, r"Input\\" + str(len(features)))

    # ── Hidden tensor blocks ─────────────────────────────────────────────
    block_ids: list[str] = []
    for li in range(n_hidden):
        bx = x_in + (li + 1) * x_gap
        bid = f"H{li}"
        sub_lines: list[str] = [str(n_neurons)]
        sub_lines.append(activation)
        if use_bn:
            sub_lines.append("BN")
        if dropout > 0.01:
            sub_lines.append("drop " + f"{dropout:.0%}".replace("%", r"\%"))
        inner_text = r"\\".join(sub_lines)
        lines.append(
            "    " + r"\node[hidden block] (" + bid + ") at ("
            + f"{bx:.2f}" + ",0) {" + inner_text + "};"
        )
        lines.append(
            "    " + r"\node[lbl, above=5mm] at (" + bid
            + ".north) {Hidden " + str(li + 1) + "};"
        )
        block_ids.append(bid)

    # ── Output ───────────────────────────────────────────────────────────
    x_out = x_in + (n_hidden + 1) * x_gap
    out_ids = _draw_io_column(
        lines, "out", x_out, labels, "output node",
        label_side="right", label_names=labels,
    )
    top_y = _io_positions(len(labels))[0][1]
    _col_header(lines, x_out, top_y, r"Output\\" + str(len(labels)))

    # ── Arrows ───────────────────────────────────────────────────────────
    lines.append("")
    lines.append("    % arrows")
    if block_ids:
        for nid in in_ids:
            lines.append(
                "    " + r"\draw[arr] (" + nid + ") -- ("
                + block_ids[0] + ".west);"
            )
        for i in range(len(block_ids) - 1):
            lines.append(
                "    " + r"\draw[arr] (" + block_ids[i] + ".east)"
                + " -- (" + block_ids[i + 1] + ".west);"
            )
        for nid in out_ids:
            lines.append(
                "    " + r"\draw[arr] (" + block_ids[-1] + ".east)"
                + " -- (" + nid + ");"
            )

    return "\n".join(lines)


# ── ELM renderer ─────────────────────────────────────────────────────────────

def render_elm(cfg: dict[str, Any], hp: dict[str, Any]) -> str:
    features = cfg["feature_names"]
    labels = cfg["label_names"]
    n_hidden = hp.get("n_hidden", 500)
    activation = hp.get("activation", "ReLU")

    x_gap = 3.2
    lines: list[str] = []

    # input
    in_ids = _draw_io_column(
        lines, "in", 0.0, features, "input node",
        label_side="left", label_names=features,
    )
    top_y = _io_positions(len(features))[0][1]
    _col_header(lines, 0.0, top_y, r"Input\\" + str(len(features)))

    # hidden
    bx = x_gap
    inner = str(n_hidden) + r"\\" + activation
    lines.append(
        "    " + r"\node[frozen block] (H0) at ("
        + f"{bx:.2f}" + ",0) {" + inner + "};"
    )
    lines.append("    " + r"\node[lbl, above=5mm] at (H0.north) {Random frozen};")

    # output
    x_out = 2 * x_gap
    out_ids = _draw_io_column(
        lines, "out", x_out, labels, "output node",
        label_side="right", label_names=labels,
    )
    top_y = _io_positions(len(labels))[0][1]
    _col_header(lines, x_out, top_y, r"Output\\" + str(len(labels)))
    lines.append("    " + r"\node[annot, below=3mm] at (H0.south) {Ridge regression};")

    # arrows
    lines.append("")
    for nid in in_ids:
        lines.append("    " + r"\draw[random arr] (" + nid + ") -- (H0.west);")
    for nid in out_ids:
        lines.append("    " + r"\draw[learned arr] (H0.east) -- (" + nid + ");")

    # legend
    _draw_legend(lines, [
        ("random arr", "Random frozen"),
        ("learned arr", "Learned (Ridge)"),
    ])

    return "\n".join(lines)


# ── dRVFL renderer ───────────────────────────────────────────────────────────

def render_drvfl(cfg: dict[str, Any], hp: dict[str, Any]) -> str:
    features = cfg["feature_names"]
    labels = cfg["label_names"]
    n_hidden = hp.get("n_hidden", 1000)
    n_layers = hp.get("n_layers", 1)
    activation = hp.get("activation", "GELU")
    direct_link = hp.get("direct_link", True)

    x_gap = 3.0
    lines: list[str] = []

    # input
    in_ids = _draw_io_column(
        lines, "in", 0.0, features, "input node",
        label_side="left", label_names=features,
    )
    top_y = _io_positions(len(features))[0][1]
    _col_header(lines, 0.0, top_y, r"Input\\" + str(len(features)))

    # hidden blocks
    block_ids: list[str] = []
    for li in range(n_layers):
        bx = (li + 1) * x_gap
        bid = f"H{li}"
        inner = str(n_hidden) + r"\\" + activation
        lines.append(
            "    " + r"\node[frozen block] (" + bid + ") at ("
            + f"{bx:.2f}" + ",0) {" + inner + "};"
        )
        lines.append(
            "    " + r"\node[lbl, above=5mm] at (" + bid
            + ".north) {Random layer " + str(li + 1) + "};"
        )
        block_ids.append(bid)

    # concat node
    concat_x = (n_layers + 1) * x_gap - 1.0
    lines.append(
        "    " + r"\node[sum node, diamond, minimum size=5mm,"
        + r" inner sep=0pt, font=\tiny]"
        + " (cat) at (" + f"{concat_x:.2f}" + ",0) {cat};"
    )

    # output
    x_out = (n_layers + 1) * x_gap + 0.5
    out_ids = _draw_io_column(
        lines, "out", x_out, labels, "output node",
        label_side="right", label_names=labels,
    )
    top_y = _io_positions(len(labels))[0][1]
    _col_header(lines, x_out, top_y, r"Output\\" + str(len(labels)))
    lines.append(
        "    " + r"\node[annot, below=3mm] at (cat.south) {Ridge regression};"
    )

    # arrows
    lines.append("")
    for nid in in_ids:
        lines.append(
            "    " + r"\draw[random arr] (" + nid + ") -- ("
            + block_ids[0] + ".west);"
        )
    for i in range(len(block_ids) - 1):
        lines.append(
            "    " + r"\draw[random arr] (" + block_ids[i] + ".east)"
            + " -- (" + block_ids[i + 1] + ".west);"
        )
    lines.append(
        "    " + r"\draw[random arr] (" + block_ids[-1] + ".east) -- (cat.west);"
    )
    for nid in out_ids:
        lines.append(
            "    " + r"\draw[learned arr] (cat.east) -- (" + nid + ");"
        )

    # direct link
    if direct_link:
        lines.append("")
        lines.append("    % direct link (input bypass)")
        arc_loose = max(70, 55 + n_layers * 5)
        arc_tight = 180 - arc_loose
        lines.append(
            "    " + r"\draw[direct arr]"
            + " (in_0.north) to[out=" + str(arc_loose)
            + ", in=" + str(arc_tight) + "] (cat.north);"
        )
        lift = 0.8 + n_layers * 0.35
        lines.append(
            "    " + r"\node[annot, above=1mm] at"
            + " ($0.5*(in_0.north)+0.5*(cat.north)+(0,"
            + f"{lift:.2f}" + ")$)"
            + r" {\textit{direct link}};"
        )

    # legend
    legend_entries = [
        ("random arr", "Random frozen"),
        ("learned arr", "Learned (Ridge)"),
    ]
    if direct_link:
        legend_entries.append(("direct arr", "Direct link"))
    _draw_legend(lines, legend_entries)

    return "\n".join(lines)


# ── edRVFL / edRVFL-SC — stacked 3D ensemble ────────────────────────────────

def render_edrvfl(cfg: dict[str, Any], hp: dict[str, Any]) -> str:
    features = cfg["feature_names"]
    labels = cfg["label_names"]
    n_hidden = hp.get("n_hidden", 1000)
    n_layers = hp.get("n_layers", 1)
    n_ensemble = hp.get("n_ensemble", 10)
    activation = hp.get("activation", "GELU")
    direct_link = hp.get("direct_link", False)
    model_name = hp.get("model", cfg.get("model_name", "edRVFL"))
    sc_mode = hp.get("sc_mode", "")

    x_gap = 3.0
    lines: list[str] = []

    # input
    in_ids = _draw_io_column(
        lines, "in", 0.0, features, "input node",
        label_side="left", label_names=features,
    )
    top_y = _io_positions(len(features))[0][1]
    _col_header(lines, 0.0, top_y, r"Input\\" + str(len(features)))

    # ── stacked ensemble planes (drawn back-to-front) ────────────────────
    n_planes = min(n_ensemble, 4)  # draw at most 4 stacked copies
    stack_dx, stack_dy = 0.22, 0.18

    # draw shadow copies (back to front, skip foreground)
    for pi in range(n_planes - 1, 0, -1):
        ox = pi * stack_dx
        oy = pi * stack_dy
        opacity = 0.25
        for li in range(n_layers):
            bx = (li + 1) * x_gap + ox
            by = oy
            lines.append(
                "    " + r"\node[frozen block, opacity=" + f"{opacity}"
                + ", fill opacity=" + f"{opacity}" + "]"
                + " at (" + f"{bx:.2f},{by:.2f}" + ") {};"
            )

    # foreground hidden blocks
    block_ids: list[str] = []
    for li in range(n_layers):
        bx = (li + 1) * x_gap
        bid = f"H{li}"
        inner = str(n_hidden) + r"\\" + activation
        lines.append(
            "    " + r"\node[frozen block] (" + bid + ") at ("
            + f"{bx:.2f}" + ",0) {" + inner + "};"
        )
        label_text = "Random layer " + str(li + 1)
        lines.append(
            "    " + r"\node[lbl, above=5mm] at (" + bid
            + ".north) {" + label_text + "};"
        )
        block_ids.append(bid)

    # ensemble label
    stack_top_y = (n_planes - 1) * stack_dy
    mid_block_x = sum((li + 1) * x_gap for li in range(n_layers)) / n_layers
    sc_suffix = ""
    if "SC" in model_name or sc_mode:
        sc_suffix = ", SC=" + sc_mode if sc_mode else ", SC"
    kfold_suffix = ""
    n_folds = hp.get("n_folds", 0)
    if "esc" in model_name.lower() and n_folds:
        kfold_suffix = r", " + str(n_folds) + "-fold CV"
    lines.append(
        "    " + r"\node[lbl, above=8mm] at"
        + " (" + f"{mid_block_x:.2f},{stack_top_y + 1.5:.2f}" + ")"
        + r" {$\times$" + str(n_ensemble) + " ensemble" + sc_suffix
        + kfold_suffix + "};"
    )

    # averaging node
    avg_x = n_layers * x_gap + x_gap - 0.6
    lines.append(
        "    " + r"\node[avg node] (avg) at ("
        + f"{avg_x:.2f}" + r",0) {\tiny avg};"
    )

    # output
    x_out = avg_x + 2.0
    out_ids = _draw_io_column(
        lines, "out", x_out, labels, "output node",
        label_side="right", label_names=labels,
    )
    top_y = _io_positions(len(labels))[0][1]
    _col_header(lines, x_out, top_y, r"Output\\" + str(len(labels)))

    # arrows
    lines.append("")
    for nid in in_ids:
        lines.append(
            "    " + r"\draw[random arr] (" + nid + ") -- ("
            + block_ids[0] + ".west);"
        )
    for i in range(len(block_ids) - 1):
        lines.append(
            "    " + r"\draw[random arr] (" + block_ids[i] + ".east)"
            + " -- (" + block_ids[i + 1] + ".west);"
        )
    lines.append(
        "    " + r"\draw[learned arr] (" + block_ids[-1] + ".east) -- (avg);"
    )
    for nid in out_ids:
        lines.append("    " + r"\draw[arr] (avg) -- (" + nid + ");")

    # direct link
    if direct_link:
        lines.append("")
        lines.append(
            "    " + r"\draw[direct arr] (in_0.north)"
            + " to[out=55, in=125] (avg.north);"
        )
        lines.append(
            "    " + r"\node[annot, above=1mm] at"
            + " ($0.5*(in_0.north)+0.5*(avg.north)+(0,0.9)$)"
            + r" {\textit{direct link}};"
        )

    # legend
    legend_entries = [
        ("random arr", "Random frozen"),
        ("learned arr", "Learned (Ridge)"),
    ]
    if direct_link:
        legend_entries.append(("direct arr", "Direct link"))
    _draw_legend(lines, legend_entries)

    return "\n".join(lines)


# ── SResdRVFL ────────────────────────────────────────────────────────────────

def render_sresdrvfl(cfg: dict[str, Any], hp: dict[str, Any]) -> str:
    features = cfg["feature_names"]
    labels = cfg["label_names"]
    n_hidden = hp.get("n_hidden", 1500)
    n_layers = hp.get("n_layers", 1)
    n_blocks = hp.get("n_blocks", 5)
    activation = hp.get("activation", "GELU")
    direct_link = hp.get("direct_link", True)

    bw = 2.2   # block width
    bg = 1.3   # block gap
    bh = 2.8   # block full height
    lines: list[str] = []

    # ── Input ────────────────────────────────────────────────────────────
    in_ids = _draw_io_column(
        lines, "in", 0.0, features, "input node",
        spacing=0.75, label_side="left", label_names=features,
    )
    top_y = _io_positions(len(features), 0.75)[0][1]
    _col_header(lines, 0.0, top_y, r"Input\\" + str(len(features)))

    # ── Residual blocks ──────────────────────────────────────────────────
    drawn_blocks = min(n_blocks, 3)
    sum_names: list[str] = []
    block_centers: list[float] = []

    for bi in range(drawn_blocks):
        bl = 2.5 + bi * (bw + bg)
        br = bl + bw
        bc = (bl + br) / 2
        block_centers.append(bc)

        lines.append(
            "    " + r"\draw[resbox] (" + f"{bl:.2f},-{bh / 2:.1f}"
            + ") rectangle (" + f"{br:.2f},{bh / 2:.1f}" + ");"
        )

        block_info = str(n_layers) + r"$\times$" + str(n_hidden)
        dl_str = ", DL" if direct_link else ""
        lines.append(
            "    " + r"\node[lbl] at (" + f"{bc:.2f},{bh / 2 - 0.5:.2f}"
            + ") {Block " + str(bi + 1) + "};"
        )
        lines.append(
            "    " + r"\node[annot] at (" + f"{bc:.2f}" + ",0.15)"
            + " {" + block_info + dl_str + "};"
        )
        lines.append(
            "    " + r"\node[annot] at (" + f"{bc:.2f}" + ",-0.25)"
            + " {" + activation + "};"
        )
        lines.append(
            "    " + r"\node[annot] at (" + f"{bc:.2f}" + ",-0.65)"
            + " {Ridge};"
        )

        # sum node below
        sn = "sum" + str(bi)
        sum_y = -bh / 2 - 0.9
        lines.append(
            "    " + r"\node[sum node] (" + sn + ") at ("
            + f"{bc:.2f},{sum_y:.2f}" + r") {$+$};"
        )
        lines.append(
            "    " + r"\draw[arr] (" + f"{bc:.2f},-{bh / 2:.1f}"
            + ") -- (" + sn + ");"
        )
        sum_names.append(sn)

    # ellipsis
    if n_blocks > drawn_blocks:
        ex = 2.5 + drawn_blocks * (bw + bg) - bg / 2
        lines.append(
            "    " + r"\node[font=\Large] at (" + f"{ex:.2f}" + r",0) {$\cdots$};"
        )
        lines.append(
            "    " + r"\node[annot] at (" + f"{ex:.2f}" + ",-0.5)"
            + " {(" + str(n_blocks) + " blocks)};"
        )

    # ── Input -> block 1 ─────────────────────────────────────────────────
    lines.append("")
    lines.append("    % input to block 1")
    first_bl = 2.5
    for nid in in_ids:
        lines.append(
            "    " + r"\draw[arr] (" + nid + ") -- ("
            + f"{first_bl:.2f}" + ",0);"
        )

    # ── Residual flow arrows between blocks ──────────────────────────────
    lines.append("")
    lines.append("    % residual flow")
    res_y = bh / 2 + 0.5
    for bi in range(drawn_blocks - 1):
        br = 2.5 + bi * (bw + bg) + bw
        nl = 2.5 + (bi + 1) * (bw + bg)
        lines.append(
            "    " + r"\draw[-{Stealth[length=2.5pt]}, residualcol!80,"
            + " line width=0.5pt]"
            + " (" + f"{br:.2f}" + ",0) -- ++(0.15,0)"
            + " |- (" + f"{br + 0.15:.2f},{res_y:.2f}" + ")"
            + " -- (" + f"{nl - 0.15:.2f},{res_y:.2f}" + ")"
            + " |- (" + f"{nl:.2f}" + ",0);"
        )
    if drawn_blocks > 1:
        mid_res = (block_centers[0] + block_centers[1]) / 2
        lines.append(
            "    " + r"\node[annot, above=1pt] at (" + f"{mid_res:.2f},{res_y:.2f}"
            + r") {\textit{residual}};"
        )

    # ── Final Sigma ──────────────────────────────────────────────────────
    sum_y = -bh / 2 - 0.9
    if n_blocks > drawn_blocks:
        sigma_x = 2.5 + drawn_blocks * (bw + bg) + 0.8
    else:
        sigma_x = block_centers[-1] + bw / 2 + 1.2
    lines.append(
        "    " + r"\node[sum node, minimum size=8mm] (sigma)"
        + " at (" + f"{sigma_x:.2f},{sum_y:.2f}" + r") {$\Sigma$};"
    )
    for sn in sum_names:
        lines.append("    " + r"\draw[arr] (" + sn + ") -- (sigma);")

    # ── Output ───────────────────────────────────────────────────────────
    x_out = sigma_x + 2.5
    out_ids = _draw_io_column(
        lines, "out", x_out, labels, "output node",
        spacing=0.75, label_side="right", label_names=labels,
    )
    top_y = _io_positions(len(labels), 0.75)[0][1]
    _col_header(lines, x_out, top_y, r"Output\\" + str(len(labels)))
    for nid in out_ids:
        lines.append("    " + r"\draw[arr] (sigma) -- (" + nid + ");")

    # legend
    _draw_legend(lines, [
        ("arr", "Data flow"),
        ("-{Stealth[length=2.5pt]}, residualcol!80, line width=0.5pt", "Residual connection"),
    ])

    return "\n".join(lines)


# ── RBF renderer ─────────────────────────────────────────────────────────────

def render_rbf(cfg: dict[str, Any], hp: dict[str, Any]) -> str:
    features = cfg["feature_names"]
    labels = cfg["label_names"]
    n_hidden = hp.get("n_hidden", 500)

    x_gap = 3.2
    lines: list[str] = []

    # input
    in_ids = _draw_io_column(
        lines, "in", 0.0, features, "input node",
        label_side="left", label_names=features,
    )
    top_y = _io_positions(len(features))[0][1]
    _col_header(lines, 0.0, top_y, r"Input\\" + str(len(features)))

    # hidden
    bx = x_gap
    inner = str(n_hidden) + r"\\Gaussian\\RBF"
    lines.append(
        "    " + r"\node[frozen block] (H0) at ("
        + f"{bx:.2f}" + ",0) {" + inner + "};"
    )
    lines.append("    " + r"\node[lbl, above=5mm] at (H0.north) {RBF centers};")

    # output
    x_out = 2 * x_gap
    out_ids = _draw_io_column(
        lines, "out", x_out, labels, "output node",
        label_side="right", label_names=labels,
    )
    top_y = _io_positions(len(labels))[0][1]
    _col_header(lines, x_out, top_y, r"Output\\" + str(len(labels)))

    # arrows
    lines.append("")
    for nid in in_ids:
        lines.append("    " + r"\draw[random arr] (" + nid + ") -- (H0.west);")
    for nid in out_ids:
        lines.append("    " + r"\draw[learned arr] (H0.east) -- (" + nid + ");")

    # legend
    _draw_legend(lines, [
        ("random arr", "Random frozen"),
        ("learned arr", "Learned (Ridge)"),
    ])

    return "\n".join(lines)


# ── Config loading & enrichment ──────────────────────────────────────────────

def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_winners(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def find_winner_hparams(
    cfg: dict[str, Any], winners: dict[str, Any],
) -> dict[str, Any]:
    model_name = cfg.get("model_name", "")
    dataset = cfg.get("dataset", "")
    for _key, entry in winners.items():
        if entry.get("model") == model_name and entry.get("dataset") == dataset:
            return entry
    return {}


# ── Dispatch ─────────────────────────────────────────────────────────────────

RANDOM_RENDERERS = {
    "ELM": render_elm,
    "RVFL": render_drvfl,
    "dRVFL": render_drvfl,
    "edRVFL": render_edrvfl,
    "edRVFL-SC": render_edrvfl,
    "esc-edRVFL": render_edrvfl,
    "SResdRVFL": render_sresdrvfl,
    "RBF": render_rbf,
}


# ── Generic (schematic) configs ──────────────────────────────────────────────

_GENERIC_FEATURES = ["$x_1$", "$x_2$", "$x_3$", "$x_4$"]
_GENERIC_LABELS = ["$y_1$", "$y_2$", "$y_3$"]

_GENERIC_CFG_BASE: dict[str, Any] = {
    "feature_names": _GENERIC_FEATURES,
    "label_names": _GENERIC_LABELS,
    "input_size": len(_GENERIC_FEATURES),
    "output_size": len(_GENERIC_LABELS),
}

GENERIC_CONFIGS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "MLP": (
        {
            **_GENERIC_CFG_BASE,
            "model_type": "mlp",
            "nr_hidden_layers": 2,
            "nr_neurons": 64,
            "activation_name": "ReLU",
            "dropout_rate": 0.0,
            "use_batchnorm": False,
        },
        {},
    ),
    "ELM": (
        {**_GENERIC_CFG_BASE, "model_type": "random", "model_name": "ELM"},
        {"n_hidden": 100, "activation": "ReLU"},
    ),
    "RVFL": (
        {**_GENERIC_CFG_BASE, "model_type": "random", "model_name": "RVFL"},
        {
            "n_hidden": 100, "n_layers": 1, "activation": "ReLU",
            "direct_link": True,
        },
    ),
    "dRVFL": (
        {**_GENERIC_CFG_BASE, "model_type": "random", "model_name": "dRVFL"},
        {
            "n_hidden": 100, "n_layers": 2, "activation": "ReLU",
            "direct_link": True,
        },
    ),
    "edRVFL": (
        {**_GENERIC_CFG_BASE, "model_type": "random", "model_name": "edRVFL"},
        {
            "n_hidden": 100, "n_layers": 2, "n_ensemble": 5,
            "activation": "ReLU", "direct_link": False,
            "model": "edRVFL",
        },
    ),
    "edRVFL-SC": (
        {**_GENERIC_CFG_BASE, "model_type": "random", "model_name": "edRVFL-SC"},
        {
            "n_hidden": 100, "n_layers": 2, "n_ensemble": 5,
            "activation": "ReLU", "direct_link": False,
            "sc_mode": "dense", "model": "edRVFL-SC",
        },
    ),
    "esc-edRVFL": (
        {**_GENERIC_CFG_BASE, "model_type": "random", "model_name": "esc-edRVFL"},
        {
            "n_hidden": 100, "n_layers": 2, "n_ensemble": 5,
            "activation": "ReLU", "direct_link": False,
            "n_folds": 5, "model": "esc-edRVFL",
        },
    ),
    "SResdRVFL": (
        {**_GENERIC_CFG_BASE, "model_type": "random", "model_name": "SResdRVFL"},
        {
            "n_hidden": 100, "n_layers": 2, "n_blocks": 2,
            "activation": "ReLU", "direct_link": True,
        },
    ),
}


def render(
    cfg: dict[str, Any],
    winners: dict[str, Any],
    config_path: Path | None = None,
    *,
    generic: bool = False,
    generic_hp: dict[str, Any] | None = None,
) -> str:
    is_random = cfg.get("model_type") == "random"
    if is_random:
        model_name = cfg.get("model_name", "")
        renderer = RANDOM_RENDERERS.get(model_name)
        if renderer is None:
            print("Warning: unknown random model '" + model_name + "', falling back to ELM")
            renderer = render_elm
        if generic and generic_hp is not None:
            hp = generic_hp
        else:
            hp = find_winner_hparams(cfg, winners)
            if not hp:
                print("Warning: no winner hparams for " + model_name + "/" + str(cfg.get("dataset")))
                hp = {}
        body_lines: list[str] = []
        body_lines.append(renderer(cfg, hp))
        display_name = model_name
    else:
        body_lines = []
        body_lines.append(render_mlp(cfg))
        display_name = "MLP"

    # title suppressed — each diagram is included in the thesis with a \caption

    body = "\n".join(body_lines)

    return (
        PREAMBLE
        + "\n"
        + r"\begin{document}" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + body + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"\end{document}" + "\n"
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def discover_configs(base_dir: Path) -> list[Path]:
    return sorted(base_dir.rglob("model_config.json"))


def output_path_for(config_path: Path, output_dir: Path) -> Path:
    rel = config_path.parent.relative_to(ARTIFACTS_DIR)
    stem = "_".join(rel.parts)
    return output_dir / (stem + ".tex")


def main():
    parser = argparse.ArgumentParser(
        description="Generate TikZ architecture diagrams from model configs.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="Path to model_config.json")
    group.add_argument(
        "--all-artifacts", action="store_true",
        help="Discover all model_config.json under artifacts/models/",
    )
    group.add_argument(
        "--generic", action="store_true",
        help="Generate one schematic diagram per architecture type",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--winners-config", type=Path, default=WINNERS_PATH,
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Run pdflatex on generated .tex files",
    )
    parser.add_argument(
        "--no-legend", action="store_true",
        help="Suppress per-diagram legend boxes",
    )
    parser.add_argument(
        "--legend-only", action="store_true",
        help="Generate a standalone legend PDF (combine with --generic)",
    )
    args = parser.parse_args()

    global _SUPPRESS_LEGEND
    _SUPPRESS_LEGEND = args.no_legend or args.legend_only

    winners = load_winners(args.winners_config)

    if args.generic:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if args.legend_only:
            _generate_legend_only(OUTPUT_DIR, compile_pdf=args.compile)
            return
        for model_key, (cfg, hp) in GENERIC_CONFIGS.items():
            stem = "generic_" + model_key.lower().replace("-", "_")
            out = OUTPUT_DIR / (stem + ".tex")
            tex = render(cfg, winners, generic=True, generic_hp=hp)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(tex)
            print("  generic " + model_key + " -> " + str(out))
            if args.compile:
                _compile_tex(out)
    elif args.all_artifacts:
        configs = discover_configs(ARTIFACTS_DIR)
        if not configs:
            print("No model_config.json found under " + str(ARTIFACTS_DIR))
            sys.exit(1)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for cp in configs:
            out = output_path_for(cp, OUTPUT_DIR)
            _generate_one(cp, out, winners, compile_pdf=args.compile)
    else:
        cfg_path = args.config
        if not cfg_path.exists():
            print("Config not found: " + str(cfg_path))
            sys.exit(1)
        if args.output:
            out = args.output
        else:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stem = "_".join(cfg_path.parent.parts[-2:])
            out = OUTPUT_DIR / (stem + ".tex")
        _generate_one(cfg_path, out, winners, compile_pdf=args.compile)


def _generate_one(
    config_path: Path, output_path: Path, winners: dict, *, compile_pdf: bool,
):
    cfg = load_config(config_path)
    tex = render(cfg, winners, config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex)
    print("  " + str(config_path) + " -> " + str(output_path))
    if compile_pdf:
        _compile_tex(output_path)


def _compile_tex(tex_path: Path):
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-output-directory", str(tex_path.parent), str(tex_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print("  Compiled -> " + str(tex_path.with_suffix(".pdf")))
        else:
            print("  pdflatex failed for " + str(tex_path))
            for line in result.stdout.strip().split("\n")[-10:]:
                print("    " + line)
    except FileNotFoundError:
        print("  pdflatex not found")
    except subprocess.TimeoutExpired:
        print("  pdflatex timed out for " + str(tex_path))


def _generate_legend_only(output_dir: Path, *, compile_pdf: bool) -> None:
    """Generate a standalone legend PDF with all arrow styles."""
    entries = [
        ("random arr", "Random frozen weights"),
        ("learned arr", "Learned weights (Ridge)"),
        ("direct arr", "Direct link"),
        ("arr", "Data flow"),
        ("-{Stealth[length=2.5pt]}, residualcol!80, line width=0.5pt",
         "Residual connection"),
    ]
    row_h = 0.45
    pad = 0.2
    h = len(entries) * row_h + 2 * pad
    w = 4.0

    lines: list[str] = []
    lines.append(r"\fill[white, rounded corners=2pt, draw=black!25,"
                 r" line width=0.35pt]"
                 f" (0,0) rectangle ({w:.1f},-{h:.2f});")
    for i, (style, label) in enumerate(entries):
        y = -(pad + row_h * i + row_h / 2)
        x0 = pad
        x1 = x0 + 0.8
        xt = x1 + 0.2
        lines.append(
            r"\draw[" + style + "] ("
            + f"{x0:.2f},{y:.2f}) -- ({x1:.2f},{y:.2f});"
        )
        lines.append(
            r"\node[anchor=west, font=\scriptsize] at ("
            + f"{xt:.2f},{y:.2f}" + ") {" + label + "};"
        )

    body = "\n    ".join(lines)
    tex = (
        PREAMBLE + "\n"
        + r"\begin{document}" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + "    " + body + "\n"
        + r"\end{tikzpicture}" + "\n"
        + r"\end{document}" + "\n"
    )
    out = output_dir / "generic_legend.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print("  legend -> " + str(out))
    if compile_pdf:
        _compile_tex(out)


if __name__ == "__main__":
    main()
