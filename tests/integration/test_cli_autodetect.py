import subprocess
import pytest
from pathlib import Path

def test_cli_autodetect_cone():
    """Verify that 'cone' dataset is auto-detected from sample_cone.csv."""
    cmd = [
        "python3", "predict.py",
        "--input", "data/sample_cone.csv",
        "--model", "mlp",
        "--no-report"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Auto-detecting dataset type..." in result.stdout
    assert "Detected dataset: cone" in result.stdout
    assert "Using model: models/mlp-cone" in result.stdout

def test_cli_autodetect_isotherm():
    """Verify that 'isotherm' dataset is auto-detected from sample_isotherm.csv."""
    cmd = [
        "python3", "predict.py",
        "--input", "data/sample_isotherm.csv",
        "--model", "mlp",
        "--no-report"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Auto-detecting dataset type..." in result.stdout
    assert "Detected dataset: isotherm" in result.stdout
    assert "Using model: models/mlp-isotherm" in result.stdout

def test_cli_manual_override():
    """Verify that manual --dataset flag overrides auto-detection."""
    cmd = [
        "python3", "predict.py",
        "--input", "data/sample_cone.csv",
        "--dataset", "cone",
        "--model", "mlp",
        "--no-report"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0
    # Should NOT see "Auto-detecting"
    assert "Auto-detecting dataset type..." not in result.stdout
    assert "Using model: models/mlp-cone" in result.stdout

def test_cli_ambiguous_failure(tmp_path):
    """Verify failure when CSV doesn't match any known dataset columns."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("random_col1,random_col2\n1,2")
    
    cmd = [
        "python3", "predict.py",
        "--input", str(bad_csv),
        "--model", "mlp",
        "--no-report"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode != 0
    assert "Error: Could not automatically detect dataset type" in result.stdout
