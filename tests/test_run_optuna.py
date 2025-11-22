"""Tests aligned with the refactored run_optuna entry point."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import optuna
import pytest

from optuna_config import (
    parse_args,
    validate_target_labels,
)
import run_optuna


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run_optuna.py'])
    args = parse_args()

    assert args.target == 'all'
    assert args.csv_file == './data/Clean_Results_Isotherm.csv'
    assert args.storage_url is None
    assert args.optuna_trials == 10000


def test_parse_args_db_alias(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run_optuna.py', '--storage-url', 'sqlite:///foo.db'])
    args = parse_args()

    assert args.storage_url == 'sqlite:///foo.db'


def test_validate_target_labels_accepts_valid():
    validate_target_labels(['Area'])
    validate_target_labels(['Area', 'Iso_distance'])


def test_validate_target_labels_raises_for_invalid():
    with pytest.raises(ValueError):
        validate_target_labels(['invalid'])


def _write_minimal_csv(csv_path: Path):
    csv_path.write_text(
        'Flow_well,Temp_diff,kW_well,Hydr_gradient,Hydr_conductivity,Aqu_thickness,'
        'Long_dispersivity,Trans_dispersivity,Isotherm,Area,Iso_distance,Iso_width\n'
        '1,2,3,4,5,6,7,8,9,10,11,12\n',
        encoding='utf-8'
    )
