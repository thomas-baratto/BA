#!/usr/bin/env python3
"""Makefile-style plot orchestrator with dependency-based staleness checks.

Checks whether source files are newer than outputs and only regenerates
stale plot groups.  Supports ``--force``, ``--only``, ``--dry-run``, and
``--list`` for flexible control.

Usage
-----
    # Show status of all groups
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/generate_all_plots.py --list

    # Dry-run: show what *would* be regenerated
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/generate_all_plots.py --dry-run

    # Regenerate only stale groups
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/generate_all_plots.py

    # Force regenerate everything
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/generate_all_plots.py --force

    # Regenerate specific groups
    PYTHONPATH=. .venv/env/bin/python scripts/analysis/generate_all_plots.py --only pareto mlp_resources
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.plot_config import PLOT_GROUPS, PlotGroup


# ── Staleness logic ─────────────────────────────────────────────────────────


def _resolve_globs(patterns: list[str], root: Path) -> list[Path]:
    """Expand a list of glob patterns relative to *root* into concrete paths."""
    found: list[Path] = []
    for pat in patterns:
        found.extend(Path(p) for p in glob.glob(str(root / pat)))
    return found


def _newest_mtime(paths: list[Path]) -> float:
    """Return the newest mtime among *paths*, or 0.0 if none exist."""
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes) if mtimes else 0.0


def _oldest_mtime(paths: list[Path]) -> float:
    """Return the oldest mtime among *paths*, or 0.0 if none exist."""
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return min(mtimes) if mtimes else 0.0


def check_staleness(group: PlotGroup, root: Path) -> tuple[bool, str]:
    """Return (is_stale, reason) for a plot group.

    A group is stale when:
      - It has no existing output files, OR
      - Any dependency file is newer than the oldest output file.
    """
    outputs = _resolve_globs(group.outputs, root)
    deps = _resolve_globs(group.dependencies, root)

    if not outputs:
        return True, "no outputs found"

    if not deps:
        return False, "no dependency files found (cannot check)"

    oldest_out = _oldest_mtime(outputs)
    newest_dep = _newest_mtime(deps)

    if newest_dep > oldest_out:
        return True, "dependencies newer than outputs"

    return False, "up to date"


# ── Display helpers ─────────────────────────────────────────────────────────

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _status_icon(stale: bool) -> str:
    return f"{_YELLOW}STALE{_RESET}" if stale else f"{_GREEN}OK{_RESET}"


def list_groups(root: Path) -> None:
    """Print all groups with their staleness status."""
    print(f"\n{'Group':<22} {'Status':<18} {'Reason'}")
    print("-" * 65)
    for name, group in PLOT_GROUPS.items():
        stale, reason = check_staleness(group, root)
        icon = _status_icon(stale)
        outputs = _resolve_globs(group.outputs, root)
        print(f"  {name:<20} {icon:<27} {reason}")
        print(f"    {group.description}")
        print(f"    {len(outputs)} output file(s)")
    print()


def dry_run(groups: dict[str, PlotGroup], root: Path, force: bool) -> None:
    """Show what would be regenerated without doing it."""
    print(f"\n{_BOLD}Dry run{_RESET} (no files will be modified):\n")
    any_work = False
    for name, group in groups.items():
        stale, reason = check_staleness(group, root)
        if force or stale:
            any_work = True
            tag = "FORCE" if (not stale and force) else "STALE"
            print(f"  [{_YELLOW}{tag}{_RESET}] {name}: {reason}")
        else:
            print(f"  [{_GREEN}SKIP{_RESET}] {name}: {reason}")
    if not any_work:
        print(f"\n  All groups are up to date. Use {_BOLD}--force{_RESET} to regenerate.")
    print()


# ── Main ────────────────────────────────────────────────────────────────────


def run(
    groups: dict[str, PlotGroup],
    root: Path,
    *,
    force: bool = False,
) -> None:
    """Run generation for stale (or forced) groups."""
    generated = 0
    skipped = 0

    for name, group in groups.items():
        stale, reason = check_staleness(group, root)

        if not force and not stale:
            print(f"  [{_GREEN}SKIP{_RESET}] {name} — {reason}")
            skipped += 1
            continue

        tag = "FORCE" if (not stale and force) else "STALE"
        print(f"\n  [{_YELLOW}{tag}{_RESET}] Generating {name}...")
        t0 = time.time()
        group.generate(root)
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s")
        generated += 1

    print(f"\n{_BOLD}Summary:{_RESET} {generated} generated, {skipped} skipped")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Makefile-style plot orchestrator — regenerate only stale plots.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all groups regardless of staleness.",
    )
    p.add_argument(
        "--only",
        nargs="+",
        metavar="GROUP",
        help="Only process these groups (space-separated names).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be regenerated without running generators.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        dest="list_groups",
        help="List all groups with their current staleness status.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT

    if args.list_groups:
        list_groups(root)
        return

    # Filter to requested groups
    if args.only:
        unknown = set(args.only) - set(PLOT_GROUPS)
        if unknown:
            print(f"Error: unknown group(s): {', '.join(sorted(unknown))}")
            print(f"Available: {', '.join(PLOT_GROUPS)}")
            sys.exit(1)
        groups = {k: PLOT_GROUPS[k] for k in args.only}
    else:
        groups = PLOT_GROUPS

    if args.dry_run:
        dry_run(groups, root, args.force)
        return

    run(groups, root, force=args.force)


if __name__ == "__main__":
    main()
