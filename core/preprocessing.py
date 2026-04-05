"""Preprocessing utilities for inference — engineered feature computation."""

import pandas as pd

# Registry of engineered features: name → (compute_fn, required_raw_columns)
ENGINEERED_FEATURES: dict[str, tuple] = {
    "Transmissivity": (
        lambda df: df["Hydr_conductivity"] * df["Aqu_thickness"],
        ["Hydr_conductivity", "Aqu_thickness"],
    ),
    "Darcy_velocity": (
        lambda df: df["Hydr_conductivity"] * df["Hydr_gradient"],
        ["Hydr_conductivity", "Hydr_gradient"],
    ),
    "Q_over_T": (
        lambda df: df["Flow_well"]
        / (df["Hydr_conductivity"] * df["Aqu_thickness"]),
        ["Flow_well", "Hydr_conductivity", "Aqu_thickness"],
    ),
}


def compute_engineered_features(
    df: pd.DataFrame, required_features: list[str]
) -> pd.DataFrame:
    """Add any missing engineered columns to *df* (in-place) if they can be
    derived from columns already present.

    Only columns listed in *required_features* that are (a) missing from *df*
    and (b) registered in ENGINEERED_FEATURES will be computed.

    Returns the (possibly augmented) DataFrame.
    """
    for feat in required_features:
        if feat in df.columns:
            continue
        if feat not in ENGINEERED_FEATURES:
            continue
        compute_fn, raw_cols = ENGINEERED_FEATURES[feat]
        missing_raw = [c for c in raw_cols if c not in df.columns]
        if missing_raw:
            raise ValueError(
                f"Cannot compute '{feat}': missing raw columns {missing_raw}"
            )
        df[feat] = compute_fn(df)
    return df
