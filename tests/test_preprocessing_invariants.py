"""Preprocessing invariants for data loader transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.data_loader import load_data


def _build_csv(csv_path):
    df = pd.DataFrame(
        {
            'Flow_well': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'Temp_diff': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            'Hydr_gradient': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'Hydr_conductivity': [10, 11, 12, 13, 14, 15],
            'Aqu_thickness': [20, 21, 22, 23, 24, 25],
            'Area': [100, 121, 144, 169, 196, 225],
            'Iso_distance': [5, 6, 7, 8, 9, 10],
        }
    )
    df.to_csv(csv_path, index=False)
    return df


def test_load_data_log_roundtrip_consistency(tmp_path):
    csv_path = tmp_path / 'toy.csv'
    df = _build_csv(csv_path)

    feature_cols = ['Flow_well', 'Temp_diff', 'Hydr_gradient', 'Hydr_conductivity', 'Aqu_thickness']
    label_cols = ['Area', 'Iso_distance']

    X_train, X_test, X_scaler, y_train, y_test, y_scaler = load_data(
        csv_file=str(csv_path),
        feature_cols=feature_cols,
        label_cols=label_cols,
        feature_scaler_type='none',
        label_scaler_type='minmax',
        use_log=True,
        use_area_root=False,
        test_size=0.3,
        random_state=42,
    )

    # Manual split/transformation for labels should match loader's scaled train labels.
    y_raw = df[label_cols].values
    _, _, y_train_raw, _ = train_test_split(y_raw, y_raw, test_size=0.3, random_state=42, shuffle=True)
    y_train_log = np.log1p(y_train_raw)
    y_train_expected = y_scaler.transform(y_train_log)

    np.testing.assert_allclose(y_train, y_train_expected)

    # Inverse scaler + expm1 should recover raw labels (up to floating precision).
    y_train_recovered = np.expm1(y_scaler.inverse_transform(y_train))
    np.testing.assert_allclose(y_train_recovered, y_train_raw, rtol=1e-6, atol=1e-6)


def test_load_data_area_root_inverse_consistency(tmp_path):
    csv_path = tmp_path / 'toy.csv'
    df = _build_csv(csv_path)

    feature_cols = ['Flow_well', 'Temp_diff', 'Hydr_gradient', 'Hydr_conductivity', 'Aqu_thickness']
    label_cols = ['Area', 'Iso_distance']

    _, _, _, y_train, _, y_scaler = load_data(
        csv_file=str(csv_path),
        feature_cols=feature_cols,
        label_cols=label_cols,
        feature_scaler_type='none',
        label_scaler_type='minmax',
        use_log=True,
        use_area_root=True,
        test_size=0.3,
        random_state=42,
    )

    y_raw = df[label_cols].values
    _, _, y_train_raw, _ = train_test_split(y_raw, y_raw, test_size=0.3, random_state=42, shuffle=True)

    # Inverse logic for Area when use_area_root=True is: expm1(...) then square Area.
    y_inv = np.expm1(y_scaler.inverse_transform(y_train))
    area_idx = label_cols.index('Area')
    y_inv[:, area_idx] = y_inv[:, area_idx] ** 2

    np.testing.assert_allclose(y_inv, y_train_raw, rtol=1e-6, atol=1e-6)
