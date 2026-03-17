#!/usr/bin/env python3
"""Package trained MLP and random models into a structured deployable directory.

This script is intentionally file-based: it copies already-produced artifacts.
It does not retrain models.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _write_manifest(dst_dir: Path, payload: dict) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dst_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _package_mlp(dataset: str, src_dir: Path, output_root: Path) -> tuple[bool, str]:
    if not src_dir.exists():
        return False, f"MLP source directory does not exist: {src_dir}"

    dst_dir = output_root / "mlp" / dataset / src_dir.name
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = {
        "best_model.pt": _copy_if_exists(src_dir / "best_model.pt", dst_dir / "best_model.pt"),
        "model_config.json": _copy_if_exists(src_dir / "model_config.json", dst_dir / "model_config.json"),
        "results.json": _copy_if_exists(src_dir / "results.json", dst_dir / "results.json"),
        "stats/metrics_summary.txt": _copy_if_exists(
            src_dir / "stats" / "metrics_summary.txt",
            dst_dir / "stats" / "metrics_summary.txt",
        ),
    }

    if not copied["best_model.pt"] or not copied["model_config.json"] or not copied["results.json"]:
        return False, f"MLP package incomplete for {dataset}: missing required files in {src_dir}"

    _write_manifest(
        dst_dir,
        {
            "packaged_at": datetime.now(timezone.utc).isoformat(),
            "model_family": "mlp",
            "dataset": dataset,
            "source_dir": str(src_dir),
            "files": copied,
        },
    )
    return True, f"Packaged MLP ({dataset}) -> {dst_dir}"


def _rank_random(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ranked_rows = []
    for dataset, ds_df in df.groupby("Dataset"):
        ds_df = ds_df.sort_values(by=["R2", "RMSE"], ascending=[False, True])
        ranked_rows.append(ds_df.head(top_k))
    if not ranked_rows:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(ranked_rows, ignore_index=True)


def _safe_model_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))


def _package_random_best(
    random_summary_csv: Path,
    random_run_dir: Path,
    output_root: Path,
    top_k_per_dataset: int,
) -> list[str]:
    if not random_summary_csv.exists():
        return [f"Random summary not found: {random_summary_csv}"]
    if not random_run_dir.exists():
        return [f"Random run dir not found: {random_run_dir}"]

    df = pd.read_csv(random_summary_csv)
    required_cols = {"Dataset", "Model", "Folder", "R2", "RMSE"}
    missing = required_cols - set(df.columns)
    if missing:
        return [f"Random summary missing required columns: {sorted(missing)}"]

    selected = _rank_random(df, top_k_per_dataset)
    messages: list[str] = []

    for _, row in selected.iterrows():
        dataset = str(row["Dataset"])
        model = str(row["Model"])
        folder = str(row["Folder"])

        src_dir = random_run_dir / folder
        if not src_dir.exists():
            messages.append(f"Skipped random package (missing folder): {src_dir}")
            continue

        dst_dir = output_root / "random" / dataset / _safe_model_name(model) / folder
        dst_dir.mkdir(parents=True, exist_ok=True)

        # One results_*.json is expected.
        results_files: Iterable[Path] = src_dir.glob("results_*.json")
        results_files = list(results_files)
        copied_results = False
        if results_files:
            copied_results = _copy_if_exists(results_files[0], dst_dir / results_files[0].name)

        copied_model = _copy_if_exists(src_dir / "model.pkl", dst_dir / "model.pkl")
        copied_npz = _copy_if_exists(src_dir / "test_predictions.npz", dst_dir / "test_predictions.npz")

        if not copied_model:
            messages.append(
                f"Skipped random package (model.pkl missing, likely --no-save-model run): {src_dir}"
            )
            continue

        manifest = {
            "packaged_at": datetime.now(timezone.utc).isoformat(),
            "model_family": "random",
            "dataset": dataset,
            "model": model,
            "source_dir": str(src_dir),
            "summary_row": {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()},
            "files": {
                "model.pkl": copied_model,
                "results_json": copied_results,
                "test_predictions.npz": copied_npz,
            },
        }
        _write_manifest(dst_dir, manifest)
        messages.append(f"Packaged random ({dataset}, {model}) -> {dst_dir}")

    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package trained MLP and random models")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/good_runs/packages"),
        help="Destination directory for packaged artifacts",
    )

    parser.add_argument("--mlp-isotherm-dir", type=Path, default=None)
    parser.add_argument("--mlp-cone-dir", type=Path, default=None)

    parser.add_argument(
        "--random-summary",
        type=Path,
        default=None,
        help="Path to sweep summary_table.csv",
    )
    parser.add_argument(
        "--random-run-dir",
        type=Path,
        default=None,
        help="Run directory containing random model subfolders referenced by summary_table.csv",
    )
    parser.add_argument(
        "--random-top-k-per-dataset",
        type=int,
        default=1,
        help="How many best random rows per dataset to package",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []

    if args.mlp_isotherm_dir is not None:
        ok, msg = _package_mlp("isotherm", args.mlp_isotherm_dir, args.output_root)
        messages.append(msg)
        if not ok:
            print(f"WARNING: {msg}")

    if args.mlp_cone_dir is not None:
        ok, msg = _package_mlp("cone", args.mlp_cone_dir, args.output_root)
        messages.append(msg)
        if not ok:
            print(f"WARNING: {msg}")

    if args.random_summary is not None and args.random_run_dir is not None:
        messages.extend(
            _package_random_best(
                random_summary_csv=args.random_summary,
                random_run_dir=args.random_run_dir,
                output_root=args.output_root,
                top_k_per_dataset=max(1, args.random_top_k_per_dataset),
            )
        )

    if not messages:
        print("Nothing to package. Pass MLP and/or random inputs.")
        return

    print("\nPackaging summary:")
    for msg in messages:
        print(f"- {msg}")


if __name__ == "__main__":
    main()
