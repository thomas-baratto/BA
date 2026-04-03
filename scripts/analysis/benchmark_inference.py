#!/usr/bin/env python3
"""Benchmark inference latency for all deployed models.

Measures forward-pass-only and full-pipeline timing for each model
across the real test set and synthetic batches of varying size.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/analysis/benchmark_inference.py
    PYTHONPATH=. .venv/bin/python scripts/analysis/benchmark_inference.py --warmup 10 --repeats 200
"""

import argparse
import json
import logging
import os
import platform
import time
from typing import Any

import warnings

import numpy as np
import torch

from config.datasets import DATASET_CONFIGS, DEFAULT_MODEL_DIRS
from core.data_loader import load_data
from core.inference import load_model_and_scalers, make_predictions, preprocess_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — all 5 models to benchmark
# ---------------------------------------------------------------------------
MODELS_TO_BENCHMARK = [
    {
        "name": "MLP",
        "dataset": "cone",
        "model_key": "mlp",
        "model_dir": DEFAULT_MODEL_DIRS["cone"]["mlp"],
    },
    {
        "name": "MLP",
        "dataset": "isotherm",
        "model_key": "mlp",
        "model_dir": DEFAULT_MODEL_DIRS["isotherm"]["mlp"],
    },
    {
        "name": "edRVFL-SC",
        "dataset": "cone",
        "model_key": "random",
        "model_dir": DEFAULT_MODEL_DIRS["cone"]["random"],
    },
    {
        "name": "SResdRVFL",
        "dataset": "isotherm",
        "model_key": "random:nRMSE",
        "model_dir": DEFAULT_MODEL_DIRS["isotherm"]["random:nRMSE"],
    },
    {
        "name": "dRVFL",
        "dataset": "isotherm",
        "model_key": "random:KGE",
        "model_dir": DEFAULT_MODEL_DIRS["isotherm"]["random:KGE"],
    },
]

SYNTHETIC_BATCH_SIZES = [1, 10, 100, 1000]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def _time_fn(fn, warmup: int, repeats: int) -> dict[str, float]:
    """Time *fn()* over *repeats* iterations after *warmup* discards.

    Returns dict with keys: mean_ns, std_ns, median_ns, min_ns, max_ns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        for _ in range(warmup):
            fn()

        times_ns: list[int] = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            times_ns.append(t1 - t0)

    arr = np.array(times_ns, dtype=np.float64)
    return {
        "mean_ns": float(arr.mean()),
        "std_ns": float(arr.std()),
        "median_ns": float(np.median(arr)),
        "min_ns": float(arr.min()),
        "max_ns": float(arr.max()),
        "n_repeats": repeats,
    }


def _ns_to_ms(ns: float) -> float:
    return ns / 1e6


def _ns_to_us(ns: float) -> float:
    return ns / 1e3


# ---------------------------------------------------------------------------
# Benchmark a single model
# ---------------------------------------------------------------------------
def benchmark_model(
    spec: dict,
    test_data: dict[str, tuple[np.ndarray, np.ndarray]],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Run benchmarks for a single model specification.

    Returns a results dict with forward-pass and full-pipeline timings
    for each batch size (synthetic + real test set).
    """
    model_dir = spec["model_dir"]
    dataset = spec["dataset"]

    if not os.path.isdir(model_dir):
        logger.warning("Model dir missing: %s — skipping %s/%s", model_dir, spec["name"], dataset)
        return {"error": f"Model directory not found: {model_dir}"}

    # Check for random .pkl presence
    pkl_path = os.path.join(model_dir, "model.pkl")
    pt_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(pkl_path) and not os.path.exists(pt_path):
        msg = (
            f"No model artifact in {model_dir}. "
            "For random models run: PYTHONPATH=. .venv/bin/python "
            "scripts/deployment/retrain_random_models.py"
        )
        logger.warning(msg)
        return {"error": msg}

    # Load model once for forward-pass benchmarks
    trained_model, feat_scaler, lbl_scaler = load_model_and_scalers(model_dir)
    config = trained_model.config
    use_log = config.get("use_log", False)
    use_area_root = config.get("use_area_root", False)

    n_features = config["input_size"]

    # Get real test data
    X_test_raw, _ = test_data[dataset]
    n_test = X_test_raw.shape[0]

    # Pre-scale the real test set for forward-pass benchmarks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        X_test_scaled = preprocess_features(X_test_raw, feat_scaler, apply_log=use_log)

    results: dict[str, Any] = {
        "model_name": spec["name"],
        "model_key": spec["model_key"],
        "dataset": dataset,
        "model_dir": model_dir,
        "n_features": n_features,
        "n_test_samples": n_test,
        "benchmarks": {},
    }

    # --- Build batch list: synthetic sizes + real test set ---
    batches: list[tuple[str, np.ndarray, np.ndarray]] = []

    for n in SYNTHETIC_BATCH_SIZES:
        rng = np.random.default_rng(42)
        # Use abs() so log1p does not encounter negative values
        X_synth_raw = np.abs(rng.standard_normal((n, n_features))).astype(np.float32) + 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X_synth_scaled = preprocess_features(
                X_synth_raw.copy(), feat_scaler, apply_log=use_log
            )
        batches.append((f"synthetic_{n}", X_synth_raw, X_synth_scaled))

    batches.append((f"test_set_{n_test}", X_test_raw, X_test_scaled))

    for batch_label, X_raw, X_scaled in batches:
        n_samples = X_scaled.shape[0]
        # Scale down repeats for large batches to keep runtime reasonable
        effective_repeats = max(10, repeats // max(1, n_samples // 500))
        effective_warmup = max(2, warmup // max(1, n_samples // 500))
        logger.info(
            "  %s/%s — batch=%s (n=%d, repeats=%d)",
            spec["name"], dataset, batch_label, n_samples, effective_repeats,
        )

        # --- 1. Forward pass only (model already loaded, data pre-scaled) ---
        fwd_timing = _time_fn(
            lambda _X=X_scaled: trained_model.predict(_X, inverse_transform=False),
            warmup=effective_warmup,
            repeats=effective_repeats,
        )
        fwd_timing["per_sample_mean_ns"] = fwd_timing["mean_ns"] / n_samples

        # --- 2. Full pipeline (includes preprocessing + postprocessing) ---
        def _full_pipeline(_X=X_raw):
            return make_predictions(
                trained_model,
                _X.copy(),
                feat_scaler,
                lbl_scaler,
                apply_feature_log=use_log,
                apply_inverse_transform=True,
                apply_label_expm1=use_log,
                use_area_root=use_area_root,
                label_names=config.get("label_names"),
            )

        pipe_timing = _time_fn(_full_pipeline, warmup=effective_warmup, repeats=effective_repeats)
        pipe_timing["per_sample_mean_ns"] = pipe_timing["mean_ns"] / n_samples

        results["benchmarks"][batch_label] = {
            "n_samples": n_samples,
            "forward_pass": fwd_timing,
            "full_pipeline": pipe_timing,
        }

    return results


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
def _system_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": str(os.cpu_count()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_available": str(torch.cuda.is_available()),
        "device_used": "cpu",
    }


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------
def _generate_latex_table(all_results: list[dict], output_path: str) -> None:
    """Write a booktabs LaTeX table comparing test-set inference latency."""
    lines = [
        r"% Auto-generated by scripts/analysis/benchmark_inference.py",
        r"% Do not edit manually.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Inference latency on the test set (CPU).}",
        r"  \label{tab:inference-latency}",
        r"  \begin{tabular}{ll r rr rr}",
        r"    \toprule",
        (
            r"    Model & Dataset & $n$ & "
            r"\multicolumn{2}{c}{Forward Pass} & "
            r"\multicolumn{2}{c}{Full Pipeline} \\"
        ),
        (
            r"    & & & "
            r"Total (\si{\milli\second}) & Per-sample (\si{\micro\second}) & "
            r"Total (\si{\milli\second}) & Per-sample (\si{\micro\second}) \\"
        ),
        r"    \midrule",
    ]

    for res in all_results:
        if "error" in res:
            continue
        # Find the test_set benchmark
        test_key = None
        for key in res["benchmarks"]:
            if key.startswith("test_set_"):
                test_key = key
                break
        if test_key is None:
            continue

        bench = res["benchmarks"][test_key]
        n = bench["n_samples"]
        fwd = bench["forward_pass"]
        pipe = bench["full_pipeline"]

        fwd_total_ms = _ns_to_ms(fwd["mean_ns"])
        fwd_per_us = _ns_to_us(fwd["per_sample_mean_ns"])
        pipe_total_ms = _ns_to_ms(pipe["mean_ns"])
        pipe_per_us = _ns_to_us(pipe["per_sample_mean_ns"])

        lines.append(
            f"    {res['model_name']:<12s} & {res['dataset']:<10s} & {n} & "
            f"{fwd_total_ms:.3f} & {fwd_per_us:.2f} & "
            f"{pipe_total_ms:.3f} & {pipe_per_us:.2f} \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("LaTeX table written to %s", output_path)


def _generate_scaling_latex_table(all_results: list[dict], output_path: str) -> None:
    """Write a booktabs table showing how latency scales with batch size."""
    lines = [
        r"% Auto-generated by scripts/analysis/benchmark_inference.py",
        r"% Do not edit manually.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Forward-pass latency scaling with batch size (CPU).}",
        r"  \label{tab:inference-scaling}",
        r"  \begin{tabular}{ll rrrr r}",
        r"    \toprule",
        (
            r"    Model & Dataset & "
            + " & ".join(f"$n={s}$" for s in SYNTHETIC_BATCH_SIZES)
            + r" & Test set \\"
        ),
        r"    & & "
        + " & ".join([r"\multicolumn{1}{c}{(\si{\micro\second/sample})}"] * (len(SYNTHETIC_BATCH_SIZES) + 1))
        + r" \\",
        r"    \midrule",
    ]

    for res in all_results:
        if "error" in res:
            continue
        cells = []
        for n in SYNTHETIC_BATCH_SIZES:
            key = f"synthetic_{n}"
            if key in res["benchmarks"]:
                us = _ns_to_us(res["benchmarks"][key]["forward_pass"]["per_sample_mean_ns"])
                cells.append(f"{us:.2f}")
            else:
                cells.append("--")

        # Test set
        test_key = next((k for k in res["benchmarks"] if k.startswith("test_set_")), None)
        if test_key:
            us = _ns_to_us(res["benchmarks"][test_key]["forward_pass"]["per_sample_mean_ns"])
            cells.append(f"{us:.2f}")
        else:
            cells.append("--")

        lines.append(
            f"    {res['model_name']:<12s} & {res['dataset']:<10s} & "
            + " & ".join(cells)
            + r" \\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Scaling LaTeX table written to %s", output_path)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def _print_summary(all_results: list[dict]) -> None:
    header = f"{'Model':<14s} {'Dataset':<10s} {'n':>5s}  {'Fwd(ms)':>10s} {'Fwd/smp(µs)':>12s}  {'Pipe(ms)':>10s} {'Pipe/smp(µs)':>13s}"
    logger.info("\n%s\n%s", header, "-" * len(header))
    for res in all_results:
        if "error" in res:
            logger.info("%-14s %-10s  ERROR: %s", res.get("model_name", "?"), res.get("dataset", "?"), res["error"])
            continue
        test_key = next((k for k in res["benchmarks"] if k.startswith("test_set_")), None)
        if test_key is None:
            continue
        bench = res["benchmarks"][test_key]
        n = bench["n_samples"]
        fwd = bench["forward_pass"]
        pipe = bench["full_pipeline"]
        logger.info(
            "%-14s %-10s %5d  %10.3f %12.2f  %10.3f %13.2f",
            res["model_name"],
            res["dataset"],
            n,
            _ns_to_ms(fwd["mean_ns"]),
            _ns_to_us(fwd["per_sample_mean_ns"]),
            _ns_to_ms(pipe["mean_ns"]),
            _ns_to_us(pipe["per_sample_mean_ns"]),
        )


# ---------------------------------------------------------------------------
# Load test data once for both datasets
# ---------------------------------------------------------------------------
def _load_test_data() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load raw (unscaled) test splits for each dataset.

    Returns dict mapping dataset name → (X_test_raw, y_test_raw).
    """
    out = {}
    for ds_name, ds_cfg in DATASET_CONFIGS.items():
        logger.info("Loading %s test data from %s", ds_name, ds_cfg["csv_file"])
        X_train, X_test, _, y_train, y_test, _ = load_data(
            csv_file=ds_cfg["csv_file"],
            feature_cols=ds_cfg["features"],
            label_cols=ds_cfg["labels"],
            test_size=0.3,
            random_state=42,
            use_log=False,       # Keep raw for benchmarking
            use_area_root=False,  # Keep raw
            plots=False,
        )
        out[ds_name] = (X_test, y_test)
        logger.info("  %s test set: %d samples, %d features", ds_name, X_test.shape[0], X_test.shape[1])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference latency for deployed models.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    parser.add_argument("--repeats", type=int, default=100, help="Timed iterations (default: 100)")
    parser.add_argument(
        "--output-json",
        default="artifacts/models/benchmark_inference_results.json",
        help="Path for JSON results",
    )
    parser.add_argument(
        "--output-latex",
        default="docs/tables/inference_benchmark.tex",
        help="Path for main LaTeX table",
    )
    parser.add_argument(
        "--output-latex-scaling",
        default="docs/tables/inference_scaling.tex",
        help="Path for batch-scaling LaTeX table",
    )
    args = parser.parse_args()

    logger.info("Inference benchmark — warmup=%d, repeats=%d", args.warmup, args.repeats)
    logger.info("System: %s", json.dumps(_system_info(), indent=2))

    # Load test data once (shared across models of the same dataset)
    test_data = _load_test_data()

    all_results: list[dict] = []
    for spec in MODELS_TO_BENCHMARK:
        logger.info("Benchmarking %s on %s ...", spec["name"], spec["dataset"])
        result = benchmark_model(spec, test_data, warmup=args.warmup, repeats=args.repeats)
        all_results.append(result)

    # Console summary
    _print_summary(all_results)

    # JSON output
    output = {
        "system_info": _system_info(),
        "settings": {"warmup": args.warmup, "repeats": args.repeats},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": all_results,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("JSON results written to %s", args.output_json)

    # LaTeX tables
    _generate_latex_table(all_results, args.output_latex)
    _generate_scaling_latex_table(all_results, args.output_latex_scaling)


if __name__ == "__main__":
    main()
