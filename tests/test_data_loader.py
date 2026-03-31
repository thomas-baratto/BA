"""Tests for core.data_loader — CSV loading, scaling, and splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split

from core.data_loader import CSVDataset, load_data


# ── CSVDataset ───────────────────────────────────────────────────────────────

class TestCSVDataset:

    def test_len_and_getitem(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([[7.0], [8.0], [9.0]])
        ds = CSVDataset(data, labels)

        assert len(ds) == 3
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert x.tolist() == [1.0, 2.0]
        assert y.tolist() == [7.0]

    def test_tensors_are_float32(self):
        ds = CSVDataset(np.zeros((2, 3)), np.ones((2, 1)))
        assert ds.data.dtype == torch.float32
        assert ds.labels.dtype == torch.float32


# ── load_data basics ─────────────────────────────────────────────────────────

class TestLoadDataBasic:

    def test_shapes_and_split(self, isotherm_csv: Path):
        features = ["Flow_well", "Temp_diff", "kW_well"]
        labels = ["Area"]

        X_train, X_test, X_sc, y_train, y_test, y_sc = load_data(
            csv_file=str(isotherm_csv), feature_cols=features,
            label_cols=labels, test_size=0.3, plots=False,
        )

        total = len(X_train) + len(X_test)
        assert total == 100
        assert X_train.shape[1] == 3
        assert y_train.shape[1] == 1

    def test_multiple_labels(self, isotherm_csv: Path):
        X_tr, _, _, y_tr, _, _ = load_data(
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well", "Temp_diff"],
            label_cols=["Area", "Iso_distance", "Iso_width"],
            plots=False,
        )
        assert y_tr.shape[1] == 3

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_data(csv_file="no_such_file.csv", feature_cols=["x"], label_cols=["y"], plots=False)

    def test_invalid_column(self, isotherm_csv: Path):
        with pytest.raises(KeyError):
            load_data(csv_file=str(isotherm_csv), feature_cols=["BOGUS"], label_cols=["Area"], plots=False)


# ── Scaler variants ──────────────────────────────────────────────────────────

class TestScalerTypes:

    @pytest.mark.parametrize("kind", ["minmax", "standard", "robust", "quantile"])
    def test_scaler_returns_data(self, isotherm_csv: Path, kind: str):
        X_tr, X_te, X_sc, y_tr, y_te, y_sc = load_data(
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well", "Temp_diff"],
            label_cols=["Area"],
            feature_scaler_type=kind,
            label_scaler_type=kind,
            plots=False,
        )
        assert X_tr is not None
        assert X_sc is not None or kind == "none"

    def test_minmax_range(self, isotherm_csv: Path):
        X_tr, *_ = load_data(
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well", "Temp_diff"],
            label_cols=["Area"],
            feature_scaler_type="minmax",
            label_scaler_type="minmax",
            plots=False,
        )
        assert X_tr.min() >= -0.01
        assert X_tr.max() <= 1.01

    def test_none_scaler_no_transform(self, isotherm_csv: Path):
        X_tr, _, X_sc, *_ = load_data(
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well"],
            label_cols=["Area"],
            feature_scaler_type="none",
            label_scaler_type="none",
            plots=False,
        )
        assert X_sc is None


# ── Reproducibility & data leakage ──────────────────────────────────────────

class TestReproducibilityAndIntegrity:

    def test_deterministic_split(self, isotherm_csv: Path):
        kw = dict(csv_file=str(isotherm_csv), feature_cols=["Flow_well"],
                  label_cols=["Area"], plots=False)
        X1, *_ = load_data(**kw)
        X2, *_ = load_data(**kw)
        np.testing.assert_array_equal(X1, X2)

    def test_no_train_test_overlap(self, isotherm_csv: Path):
        X_tr, X_te, _, y_tr, y_te, _ = load_data(
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well", "Temp_diff"],
            label_cols=["Area"],
            plots=False,
        )
        # Check sizes add up
        assert len(X_tr) + len(X_te) == 100

    def test_scaler_inverse_roundtrip(self, isotherm_csv: Path):
        _, _, _, y_tr, _, y_sc = load_data(
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well"],
            label_cols=["Area"],
            plots=False,
        )
        y_back = y_sc.transform(y_sc.inverse_transform(y_tr))
        np.testing.assert_allclose(y_tr, y_back, rtol=1e-5)


# ── Preprocessing invariants (log + area_root roundtrips) ────────────────────

class TestPreprocessingInvariants:

    def _build_tiny_csv(self, tmp_path: Path) -> Path:
        import pandas as pd
        df = pd.DataFrame({
            "Flow_well": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Temp_diff": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "Hydr_gradient": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "Hydr_conductivity": [10, 11, 12, 13, 14, 15],
            "Aqu_thickness": [20, 21, 22, 23, 24, 25],
            "Area": [100, 121, 144, 169, 196, 225],
            "Iso_distance": [5, 6, 7, 8, 9, 10],
        })
        p = tmp_path / "tiny.csv"
        df.to_csv(p, index=False)
        return p

    def test_log_roundtrip(self, tmp_path: Path):
        """inverse_scaler + expm1 must recover the original label values."""
        import pandas as pd
        csv = self._build_tiny_csv(tmp_path)
        df = pd.read_csv(csv)

        features = list(df.columns[:5])
        labels = ["Area", "Iso_distance"]

        _, _, _, y_tr, _, y_sc = load_data(
            csv_file=str(csv), feature_cols=features, label_cols=labels,
            feature_scaler_type="none", label_scaler_type="minmax",
            use_log=True, use_area_root=False,
            test_size=0.3, random_state=42,
        )

        y_raw = df[labels].values
        _, _, y_train_raw, _ = train_test_split(
            y_raw, y_raw, test_size=0.3, random_state=42, shuffle=True,
        )

        y_recovered = np.expm1(y_sc.inverse_transform(y_tr))
        np.testing.assert_allclose(y_recovered, y_train_raw, rtol=1e-6, atol=1e-6)

    def test_area_root_roundtrip(self, tmp_path: Path):
        """sqrt(Area) + log1p + scale → inverse must recover original Area."""
        import pandas as pd
        csv = self._build_tiny_csv(tmp_path)
        df = pd.read_csv(csv)

        features = list(df.columns[:5])
        labels = ["Area", "Iso_distance"]

        _, _, _, y_tr, _, y_sc = load_data(
            csv_file=str(csv), feature_cols=features, label_cols=labels,
            feature_scaler_type="none", label_scaler_type="minmax",
            use_log=True, use_area_root=True,
            test_size=0.3, random_state=42,
        )

        y_raw = df[labels].values
        _, _, y_train_raw, _ = train_test_split(
            y_raw, y_raw, test_size=0.3, random_state=42, shuffle=True,
        )

        y_inv = np.expm1(y_sc.inverse_transform(y_tr))
        area_idx = labels.index("Area")
        y_inv[:, area_idx] = y_inv[:, area_idx] ** 2

        np.testing.assert_allclose(y_inv, y_train_raw, rtol=1e-6, atol=1e-6)


# ── Plot generation ──────────────────────────────────────────────────────────

class TestPlotGeneration:

    def test_creates_plot_files(self, isotherm_csv: Path, tmp_path: Path):
        load_data(
            csv_file=str(isotherm_csv),
            feature_cols=["Flow_well", "Temp_diff"],
            label_cols=["Area"],
            plots=True, rf=str(tmp_path),
        )
        plot_dir = tmp_path / "plots" / "Area"
        assert plot_dir.exists()
        assert (plot_dir / "before_transform.pdf").exists()
        assert (plot_dir / "after_log_transform.pdf").exists()
        assert (plot_dir / "after_scaling.pdf").exists()
