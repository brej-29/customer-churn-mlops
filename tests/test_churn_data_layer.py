"""Tests for the churn.data cleaning and splitting layer (Tier 1 rebuild, step 1)."""

import pandas as pd
import pytest

from churn.data import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    TELCO_CLEAN_SHAPE,
    get_splits,
    load_clean_telco,
    load_telco_raw,
)

CSV_PRESENT = pytest.importorskip  # sentinel — actual skip below

_csv_present = pytest.mark.skipif(
    not __import__("pathlib").Path("data/raw/telco_churn.csv").exists(),
    reason="data/raw/telco_churn.csv not present",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_df():
    return load_telco_raw()


@pytest.fixture(scope="module")
def clean_df():
    return load_clean_telco(save=False)


@pytest.fixture(scope="module")
def splits():
    return get_splits()


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------


def test_schema_constants():
    assert TARGET == "Churn"
    assert "tenure" in NUMERIC_FEATURES
    assert "MonthlyCharges" in NUMERIC_FEATURES
    assert "TotalCharges" in NUMERIC_FEATURES
    assert len(NUMERIC_FEATURES) == 3
    assert len(CATEGORICAL_FEATURES) == 16
    assert len(ALL_FEATURES) == 19
    # customerID and Churn must not appear in features
    assert "customerID" not in ALL_FEATURES
    assert TARGET not in ALL_FEATURES
    # SeniorCitizen is listed as categorical (handled uniformly downstream)
    assert "SeniorCitizen" in CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Cleaning — shape and columns
# ---------------------------------------------------------------------------


@_csv_present
def test_clean_shape(clean_df):
    assert clean_df.shape == TELCO_CLEAN_SHAPE, (
        f"Expected {TELCO_CLEAN_SHAPE}, got {clean_df.shape}"
    )


@_csv_present
def test_customer_id_absent(clean_df):
    assert "customerID" not in clean_df.columns


@_csv_present
def test_total_charges_is_float_no_nulls(clean_df):
    assert clean_df["TotalCharges"].dtype == "float64"
    assert clean_df["TotalCharges"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Cleaning — TotalCharges blanks → 0.0, all tenure==0
# ---------------------------------------------------------------------------


@_csv_present
def test_total_charges_blanks_were_tenure_zero(raw_df, clean_df):
    """The 11 rows that were blank in TotalCharges must all have had tenure==0
    and must now be 0.0 in the cleaned frame."""
    tc_numeric = pd.to_numeric(raw_df["TotalCharges"], errors="coerce")
    blank_idx = raw_df.index[tc_numeric.isna()]

    assert len(blank_idx) == 11, f"Expected 11 blanks, found {len(blank_idx)}"

    # All blanks had tenure == 0 in the raw data
    assert (raw_df.loc[blank_idx, "tenure"] == 0).all(), (
        "Some blank-TotalCharges rows had tenure != 0"
    )

    # In the cleaned frame those rows are now 0.0
    assert (clean_df.loc[blank_idx, "TotalCharges"] == 0.0).all(), (
        "Blank rows not filled with 0.0 in cleaned frame"
    )


# ---------------------------------------------------------------------------
# Cleaning — target encoding and nulls
# ---------------------------------------------------------------------------


@_csv_present
def test_churn_values_binary(clean_df):
    assert set(clean_df[TARGET].unique()) == {0, 1}


@_csv_present
def test_no_nulls_anywhere(clean_df):
    null_totals = clean_df.isnull().sum()
    cols_with_nulls = null_totals[null_totals > 0]
    assert cols_with_nulls.empty, (
        f"Null values found after cleaning: {cols_with_nulls.to_dict()}"
    )


# ---------------------------------------------------------------------------
# Cleaning — dtype contracts
# ---------------------------------------------------------------------------


@_csv_present
def test_numeric_feature_dtypes(clean_df):
    for col in NUMERIC_FEATURES:
        assert clean_df[col].dtype == "float64", (
            f"{col} should be float64, got {clean_df[col].dtype}"
        )


@_csv_present
def test_target_dtype_is_int(clean_df):
    assert pd.api.types.is_integer_dtype(clean_df[TARGET])


# ---------------------------------------------------------------------------
# Split — determinism
# ---------------------------------------------------------------------------


@_csv_present
def test_splits_are_deterministic():
    X_train_a, X_test_a, y_train_a, y_test_a = get_splits()
    X_train_b, X_test_b, y_train_b, y_test_b = get_splits()

    assert X_train_a.index.tolist() == X_train_b.index.tolist()
    assert X_test_a.index.tolist() == X_test_b.index.tolist()
    assert y_train_a.tolist() == y_train_b.tolist()
    assert y_test_a.tolist() == y_test_b.tolist()


# ---------------------------------------------------------------------------
# Split — sizes and stratification
# ---------------------------------------------------------------------------


@_csv_present
def test_split_sizes(splits):
    X_train, X_test, y_train, y_test = splits
    assert len(X_train) == 5634, f"Expected 5634 train rows, got {len(X_train)}"
    assert len(X_test) == 1409, f"Expected 1409 test rows, got {len(X_test)}"


@_csv_present
def test_split_stratification(splits, clean_df):
    X_train, X_test, y_train, y_test = splits
    overall_rate = clean_df[TARGET].mean()
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    tol = 0.01  # within 1 percentage point
    assert abs(train_rate - overall_rate) < tol, (
        f"Train churn rate {train_rate:.4f} deviates from overall {overall_rate:.4f} by >{tol}"
    )
    assert abs(test_rate - overall_rate) < tol, (
        f"Test churn rate {test_rate:.4f} deviates from overall {overall_rate:.4f} by >{tol}"
    )


@_csv_present
def test_split_indices_do_not_overlap(splits):
    X_train, X_test, _, _ = splits
    overlap = set(X_train.index) & set(X_test.index)
    assert not overlap, f"Train and test indices overlap: {len(overlap)} shared rows"


@_csv_present
def test_x_y_alignment(splits):
    X_train, X_test, y_train, y_test = splits
    assert X_train.index.tolist() == y_train.index.tolist()
    assert X_test.index.tolist() == y_test.index.tolist()
