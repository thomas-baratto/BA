#!/usr/bin/env python3
"""
Manage dynamic Pareto frontiers for random sweep runs.

Two frontiers are maintained per dataset:
 - (Time(s), 1 - KGE)    -> both to MINIMIZE (lower time, higher KGE)
 - (Time(s), nRMSE)      -> both to MINIMIZE

Given a run directory that contains a `summary_table.csv` (as produced by
`scripts/analysis/summarize_results.py`), this module computes the frontier members
and can prune run subdirectories that are not on either frontier.
"""
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
from typing import Set, List, Tuple


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Return boolean mask of pareto-efficient (non-dominated) rows.

    This function treats smaller values as better (minimization).
    """
    if costs.size == 0:
        return np.array([], dtype=bool)

    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # An item is dominated if there exists another item strictly better
            # in all objectives. We keep items that are NOT strictly dominated.
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
    return is_efficient


class ParetoManager:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")

    def _load_summary(self) -> pd.DataFrame:
        csv = self.run_dir / "summary_table.csv"
        if not csv.exists():
            return pd.DataFrame()
        df = pd.read_csv(csv)
        return df

    def compute_frontier_folders(self) -> Set[str]:
        """Compute union of folders that lie on either of the two frontiers.

        Returns a set of folder names (not full paths).
        """
        df = self._load_summary()
        if df.empty:
            return set()

        keep_folders = set()

        for dataset in df['Dataset'].unique():
            sub = df[df['Dataset'] == dataset].copy()

            # Ensure relevant columns exist
            if 'Time(s)' not in sub.columns:
                continue

            # FRONTIER A: minimize [Time(s), 1-KGE] -> requires KGE
            a_df = sub.dropna(subset=['Time(s)', 'KGE'])
            if not a_df.empty:
                costs_a = np.vstack([a_df['Time(s)'].astype(float).values, (1.0 - a_df['KGE'].astype(float).values)])
                costs_a = costs_a.T
                mask_a = is_pareto_efficient(costs_a)
                folders_a = set(a_df.loc[mask_a, 'Folder'].astype(str).values)
                keep_folders.update(folders_a)

            # FRONTIER B: minimize [Time(s), nRMSE] -> requires nRMSE
            b_df = sub.dropna(subset=['Time(s)', 'nRMSE'])
            if not b_df.empty:
                costs_b = np.vstack([b_df['Time(s)'].astype(float).values, b_df['nRMSE'].astype(float).values])
                costs_b = costs_b.T
                mask_b = is_pareto_efficient(costs_b)
                folders_b = set(b_df.loc[mask_b, 'Folder'].astype(str).values)
                keep_folders.update(folders_b)

        return keep_folders

    def prune_non_frontier(self, dry_run: bool = False, keep_env_var: str = 'KEEP_RUN_ARTIFACTS') -> List[Tuple[Path, str]]:
        """Delete large model artifacts from subdirectories NOT on either frontier.

        Keeps all JSON summary files (results_*.json, multi_seed_summary.json) so they can
        be used for visualization/analysis. Only deletes heavy binary files:
          - model.pkl (the trained model)
          - test_predictions.npz (prediction arrays)
          - seed_* subdirectories (contains per-seed models and predictions)

        If the environment variable `keep_env_var` is set to "1", pruning is skipped.
        Returns a list of (Path, artifact_type) tuples that were removed.
        """
        if os.environ.get(keep_env_var, '0') == '1':
            return []

        keep = self.compute_frontier_folders()
        removed = []

        # Files to delete from non-frontier runs (model + inference artifacts)
        heavy_artifacts = ['model.pkl', 'test_predictions.npz', 'scalers.pkl',
                           'model_config.json', 'artifact_manifest.json']

        for child in self.run_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if name in keep:
                continue
            # safety: only prune directories that contain a results_*.json file
            results_files = list(child.glob('results_*.json'))
            if not results_files:
                continue

            # Delete heavy binary artifacts but keep JSON summaries
            for artifact in heavy_artifacts:
                artifact_path = child / artifact
                if artifact_path.exists():
                    if dry_run:
                        removed.append((artifact_path, 'file'))
                        continue
                    try:
                        artifact_path.unlink()
                        removed.append((artifact_path, 'file'))
                    except Exception:
                        pass

            # Delete seed_* subdirectories (per-seed models)
            for seed_dir in child.glob('seed_*'):
                if not seed_dir.is_dir():
                    continue
                if dry_run:
                    removed.append((seed_dir, 'dir'))
                    continue
                try:
                    shutil.rmtree(seed_dir)
                    removed.append((seed_dir, 'dir'))
                except Exception:
                    pass

        return removed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--prune', action='store_true', help='Delete model artifacts from non-frontier run folders (keep JSON summaries)')
    args = parser.parse_args()

    mgr = ParetoManager(args.run_dir)
    keep = mgr.compute_frontier_folders()
    print(f"Keeping {len(keep)} folders on Pareto frontiers:")
    for k in sorted(keep):
        print(f"  {k}")

    if args.prune:
        removed = mgr.prune_non_frontier(dry_run=args.dry_run)
        print(f"Removed {len(removed)} artifacts (dry_run={args.dry_run}):")
        for path, atype in removed:
            print(f"  [{atype}] {path.relative_to(args.run_dir)}")
