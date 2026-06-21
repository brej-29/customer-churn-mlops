"""Tests for churn/features.py — ChurnFeatureEngineer and build_preprocessor()."""

import numpy as np
import pandas as pd
import pytest

from churn.data import ALL_FEATURES, get_splits, load_clean_telco
from churn.features import (
    CT_CATEGORICAL,
    CT_NUMERIC,
    ENGINEERED_FEATURES,
    SERVICE_COLS,
    ChurnFeatureEngineer,
    build_preprocessor,
)

_csv_present = pytest.mark.skipif(
    not __import__("pathlib").Path("data/raw/telco_churn.csv").exists(),
    reason="data/raw/telco_churn.csv not present",
)


# ---------------------------------------------------------------------------
# Minimal DataFrame factory
# ---------------------------------------------------------------------------


def _row(**overrides) -> pd.DataFrame:
    """One-row DataFrame with valid defaults for all feature columns."""
    defaults: dict = {
        "tenure": 12.0,
        "MonthlyCharges": 60.0,
        "TotalCharges": 720.0,
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "No",
        "MultipleLines": "No",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


def _fit_transform(df: pd.DataFrame) -> pd.DataFrame:
    fe = ChurnFeatureEngineer()
    fe.fit(df)
    return fe.transform(df)


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------


def test_constants_lengths():
    assert len(CT_NUMERIC) == 6      # 3 original + 3 engineered numeric
    assert len(CT_CATEGORICAL) == 17  # 16 original + tenure_bucket
    assert len(ENGINEERED_FEATURES) == 4
    assert len(SERVICE_COLS) == 8


def test_engineered_feature_names():
    assert "tenure_bucket" in ENGINEERED_FEATURES
    assert "num_services" in ENGINEERED_FEATURES
    assert "avg_monthly_spend" in ENGINEERED_FEATURES
    assert "spend_growth_ratio" in ENGINEERED_FEATURES


# ---------------------------------------------------------------------------
# ChurnFeatureEngineer — num_services
# ---------------------------------------------------------------------------


def test_num_services_all_inactive():
    """'No', 'No phone service', 'No internet service' all count as 0."""
    df = _row(
        PhoneService="No",
        MultipleLines="No phone service",
        InternetService="No",
        OnlineSecurity="No internet service",
        OnlineBackup="No internet service",
        DeviceProtection="No internet service",
        TechSupport="No internet service",
        StreamingTV="No internet service",
        StreamingMovies="No internet service",
    )
    out = _fit_transform(df)
    assert out["num_services"].iloc[0] == 0


def test_num_services_phone_and_internet_only():
    """PhoneService=Yes + InternetService=DSL (not No) → 2."""
    df = _row(PhoneService="Yes", InternetService="DSL")
    out = _fit_transform(df)
    assert out["num_services"].iloc[0] == 2


def test_num_services_several_yes_plus_fiber():
    """Three 'Yes' service add-ons + Fiber optic internet → 4."""
    df = _row(
        PhoneService="Yes",
        MultipleLines="Yes",
        InternetService="Fiber optic",
        OnlineSecurity="Yes",
        OnlineBackup="No internet service",
        DeviceProtection="No internet service",
        TechSupport="No internet service",
        StreamingTV="No internet service",
        StreamingMovies="No internet service",
    )
    out = _fit_transform(df)
    assert out["num_services"].iloc[0] == 4


def test_num_services_all_yes_and_fiber():
    """All 8 service cols 'Yes' + Fiber optic → 9."""
    df = _row(
        PhoneService="Yes",
        MultipleLines="Yes",
        InternetService="Fiber optic",
        OnlineSecurity="Yes",
        OnlineBackup="Yes",
        DeviceProtection="Yes",
        TechSupport="Yes",
        StreamingTV="Yes",
        StreamingMovies="Yes",
    )
    out = _fit_transform(df)
    assert out["num_services"].iloc[0] == 9


# ---------------------------------------------------------------------------
# ChurnFeatureEngineer — tenure_bucket boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tenure,expected", [
    (0,  "0-12"),   # leftmost edge: include_lowest makes [0,12]
    (12, "0-12"),   # right edge of first bin
    (13, "12-24"),  # just above first bin
    (24, "12-24"),  # right edge of second bin
    (25, "24-48"),
    (48, "24-48"),  # right edge of third bin
    (49, "48-60"),
    (60, "48-60"),  # right edge of fourth bin
    (61, "60-72"),
    (72, "60-72"),  # maximum right edge
])
def test_tenure_bucket_boundaries(tenure, expected):
    df = _row(
        tenure=float(tenure),
        TotalCharges=float(tenure) * 60.0,
    )
    out = _fit_transform(df)
    assert out["tenure_bucket"].iloc[0] == expected, (
        f"tenure={tenure}: expected '{expected}', got '{out['tenure_bucket'].iloc[0]}'"
    )


# ---------------------------------------------------------------------------
# ChurnFeatureEngineer — avg_monthly_spend
# ---------------------------------------------------------------------------


def test_avg_monthly_spend_tenure_gt_zero():
    df = _row(tenure=12.0, TotalCharges=720.0, MonthlyCharges=60.0)
    out = _fit_transform(df)
    assert out["avg_monthly_spend"].iloc[0] == pytest.approx(60.0)


def test_avg_monthly_spend_tenure_zero_uses_monthly_charges():
    df = _row(tenure=0.0, TotalCharges=0.0, MonthlyCharges=50.0)
    out = _fit_transform(df)
    assert out["avg_monthly_spend"].iloc[0] == pytest.approx(50.0)


def test_avg_monthly_spend_non_round():
    df = _row(tenure=6.0, TotalCharges=360.0, MonthlyCharges=80.0)
    out = _fit_transform(df)
    assert out["avg_monthly_spend"].iloc[0] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# ChurnFeatureEngineer — spend_growth_ratio
# ---------------------------------------------------------------------------


def test_spend_growth_ratio_stable_customer():
    """MonthlyCharges == historical avg → ratio == 1.0."""
    df = _row(tenure=12.0, TotalCharges=720.0, MonthlyCharges=60.0)
    out = _fit_transform(df)
    assert out["spend_growth_ratio"].iloc[0] == pytest.approx(1.0)


def test_spend_growth_ratio_price_increase():
    """Current charge > historical avg → ratio > 1."""
    df = _row(tenure=6.0, TotalCharges=360.0, MonthlyCharges=80.0)
    out = _fit_transform(df)
    # avg=60, ratio=80/60
    assert out["spend_growth_ratio"].iloc[0] == pytest.approx(80.0 / 60.0, rel=1e-5)


def test_spend_growth_ratio_tenure_zero_equals_one():
    """For tenure==0, avg==MonthlyCharges so ratio is always 1.0 (when MC>0)."""
    df = _row(tenure=0.0, TotalCharges=0.0, MonthlyCharges=50.0)
    out = _fit_transform(df)
    assert out["spend_growth_ratio"].iloc[0] == pytest.approx(1.0)


def test_spend_growth_ratio_guard_zero_avg():
    """avg_monthly_spend==0 (MC=0, tenure=0) must not divide → ratio==1.0."""
    df = _row(tenure=0.0, TotalCharges=0.0, MonthlyCharges=0.0)
    out = _fit_transform(df)
    assert out["avg_monthly_spend"].iloc[0] == pytest.approx(0.0)
    assert out["spend_growth_ratio"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ChurnFeatureEngineer — no mutation of input
# ---------------------------------------------------------------------------


def test_transform_does_not_mutate_input():
    df = _row(tenure=12.0, TotalCharges=720.0, MonthlyCharges=60.0)
    original_cols = df.columns.tolist()
    original_vals = df.iloc[0].to_dict()

    fe = ChurnFeatureEngineer()
    fe.fit(df)
    _ = fe.transform(df)

    assert df.columns.tolist() == original_cols
    for col, val in original_vals.items():
        assert df.iloc[0][col] == val, f"Column '{col}' was mutated"


# ---------------------------------------------------------------------------
# ChurnFeatureEngineer — get_feature_names_out
# ---------------------------------------------------------------------------


def test_get_feature_names_out_length():
    df = _row()
    fe = ChurnFeatureEngineer()
    fe.fit(df)
    names = fe.get_feature_names_out()
    assert len(names) == len(ALL_FEATURES) + len(ENGINEERED_FEATURES)


def test_get_feature_names_out_contains_engineered():
    df = _row()
    fe = ChurnFeatureEngineer()
    fe.fit(df)
    names = fe.get_feature_names_out()
    for eng in ENGINEERED_FEATURES:
        assert eng in names


# ---------------------------------------------------------------------------
# build_preprocessor — fit and transform
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_preprocessor():
    if not __import__("pathlib").Path("data/raw/telco_churn.csv").exists():
        pytest.skip("data/raw/telco_churn.csv not present")
    X_train, _, _, _ = get_splits()
    pp = build_preprocessor()
    pp.fit(X_train)
    return pp, X_train


@_csv_present
def test_preprocessor_train_output_is_2d_numeric_no_nan(fitted_preprocessor):
    pp, X_train = fitted_preprocessor
    result = pp.transform(X_train)
    assert result.ndim == 2
    assert result.dtype.kind == "f"  # floating point
    assert not np.isnan(result).any()


@_csv_present
def test_preprocessor_test_output_same_cols_no_nan(fitted_preprocessor):
    pp, X_train = fitted_preprocessor
    _, X_test, _, _ = get_splits()
    train_result = pp.transform(X_train)
    test_result = pp.transform(X_test)
    assert test_result.shape[1] == train_result.shape[1]
    assert not np.isnan(test_result).any()


@_csv_present
def test_get_feature_names_out_matches_width(fitted_preprocessor):
    pp, X_train = fitted_preprocessor
    result = pp.transform(X_train)
    names = pp.get_feature_names_out()
    assert len(names) == result.shape[1]


# ---------------------------------------------------------------------------
# Leakage check: scaler statistics are derived from train only
# ---------------------------------------------------------------------------


@_csv_present
def test_scaler_stats_are_train_derived():
    X_train, _, _, _ = get_splits()
    df_full = load_clean_telco(save=False)
    X_full = df_full[ALL_FEATURES]

    pp_train = build_preprocessor()
    pp_train.fit(X_train)
    scaler_train = pp_train.named_steps["ct"].named_transformers_["num"].named_steps["scaler"]
    train_mean = scaler_train.mean_[0]  # mean of first numeric col (tenure)

    pp_full = build_preprocessor()
    pp_full.fit(X_full)
    scaler_full = pp_full.named_steps["ct"].named_transformers_["num"].named_steps["scaler"]
    full_mean = scaler_full.mean_[0]

    assert train_mean != pytest.approx(full_mean, rel=1e-6), (
        "Scaler means are identical for train-fit vs full-fit — possible data leakage"
    )


@_csv_present
def test_transform_does_not_refit_scaler(fitted_preprocessor):
    """Calling transform on X_test must not change the fitted scaler's parameters."""
    pp, X_train = fitted_preprocessor
    _, X_test, _, _ = get_splits()
    scaler = pp.named_steps["ct"].named_transformers_["num"].named_steps["scaler"]
    mean_before = scaler.mean_.copy()
    pp.transform(X_test)
    np.testing.assert_array_equal(scaler.mean_, mean_before)


# ---------------------------------------------------------------------------
# handle_unknown: unseen category → zeros, no exception
# ---------------------------------------------------------------------------


@_csv_present
def test_handle_unknown_unseen_category_no_raise(fitted_preprocessor):
    pp, _ = fitted_preprocessor
    # Create a single row with a PaymentMethod value never seen in training.
    novel_row = _row(PaymentMethod="Bitcoin_unseen_9999")
    result = pp.transform(novel_row)
    assert result.ndim == 2
    assert not np.isnan(result).any()


@_csv_present
def test_handle_unknown_unseen_category_yields_zeros(fitted_preprocessor):
    pp, _ = fitted_preprocessor
    novel_row = _row(PaymentMethod="Bitcoin_unseen_9999")
    result = pp.transform(novel_row)

    feature_names = pp.get_feature_names_out()
    payment_indices = [
        i for i, name in enumerate(feature_names) if "PaymentMethod" in name
    ]
    assert payment_indices, "No PaymentMethod columns found in feature names"
    assert all(result[0, i] == 0.0 for i in payment_indices), (
        "Unseen category should produce all-zero OHE columns"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@_csv_present
def test_preprocessor_deterministic(fitted_preprocessor):
    pp, X_train = fitted_preprocessor
    out_a = pp.transform(X_train)
    out_b = pp.transform(X_train)
    np.testing.assert_array_equal(out_a, out_b)
