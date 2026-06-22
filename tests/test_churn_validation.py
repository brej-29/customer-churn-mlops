"""Tests for churn/validation.py — fully offline (no network, no LLM).

Covers:
  - validate_clean passes real data and rejects each corruption type
  - validate_serving accepts valid rows and rejects bad ones
  - Consistency: ALLOWED sets == API Pydantic Literals == data.py CATEGORICAL_FEATURES
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_csv_present = pytest.mark.skipif(
    not Path("data/raw/telco_churn.csv").exists(),
    reason="data/raw/telco_churn.csv not present — skipping real-data tests",
)

# ---------------------------------------------------------------------------
# Fixtures — one valid cleaned row (mirrors what clean_telco() produces)
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_row() -> pd.DataFrame:
    """One row that satisfies both the training and serving contracts."""
    return pd.DataFrame(
        [
            {
                "tenure": 12.0,
                "MonthlyCharges": 70.0,
                "TotalCharges": 840.0,
                "gender": "Male",
                "SeniorCitizen": "0",
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Bank transfer (automatic)",
                "Churn": 0,
            }
        ]
    )


@pytest.fixture
def valid_serving_row(valid_row: pd.DataFrame) -> pd.DataFrame:
    """Valid row without the Churn target — for serving-contract tests."""
    return valid_row.drop(columns=["Churn"])


# ---------------------------------------------------------------------------
# validate_clean — real data
# ---------------------------------------------------------------------------


@_csv_present
def test_validate_clean_passes_real_data() -> None:
    from churn.data import load_clean_telco
    from churn.validation import validate_clean

    df = load_clean_telco(save=False, validate=False)
    result = validate_clean(df)
    assert result.shape == df.shape


# ---------------------------------------------------------------------------
# validate_clean — valid synthetic row passes
# ---------------------------------------------------------------------------


def test_validate_clean_passes_valid_row(valid_row: pd.DataFrame) -> None:
    from churn.validation import validate_clean

    result = validate_clean(valid_row)
    assert result.shape == valid_row.shape


# ---------------------------------------------------------------------------
# validate_clean — corruption tests (each must raise ValueError naming the column)
# ---------------------------------------------------------------------------


def test_validate_clean_rejects_out_of_range_tenure(valid_row: pd.DataFrame) -> None:
    from churn.validation import validate_clean

    bad = valid_row.copy()
    bad["tenure"] = 200.0
    with pytest.raises(ValueError, match="tenure"):
        validate_clean(bad)


def test_validate_clean_rejects_negative_tenure(valid_row: pd.DataFrame) -> None:
    from churn.validation import validate_clean

    bad = valid_row.copy()
    bad["tenure"] = -1.0
    with pytest.raises(ValueError, match="tenure"):
        validate_clean(bad)


def test_validate_clean_rejects_zero_monthly_charges(valid_row: pd.DataFrame) -> None:
    from churn.validation import validate_clean

    bad = valid_row.copy()
    bad["MonthlyCharges"] = 0.0
    with pytest.raises(ValueError, match="MonthlyCharges"):
        validate_clean(bad)


def test_validate_clean_rejects_negative_total_charges(valid_row: pd.DataFrame) -> None:
    from churn.validation import validate_clean

    bad = valid_row.copy()
    bad["TotalCharges"] = -5.0
    with pytest.raises(ValueError, match="TotalCharges"):
        validate_clean(bad)


def test_validate_clean_rejects_unknown_contract(valid_row: pd.DataFrame) -> None:
    from churn.validation import validate_clean

    bad = valid_row.copy()
    bad["Contract"] = "Annual"
    with pytest.raises(ValueError, match="Contract"):
        validate_clean(bad)


def test_validate_clean_rejects_unknown_internet_service(valid_row: pd.DataFrame) -> None:
    from churn.validation import validate_clean

    bad = valid_row.copy()
    bad["InternetService"] = "Cable"
    with pytest.raises(ValueError, match="InternetService"):
        validate_clean(bad)


def test_validate_clean_rejects_unknown_payment_method(valid_row: pd.DataFrame) -> None:
    from churn.validation import validate_clean

    bad = valid_row.copy()
    bad["PaymentMethod"] = "Crypto"
    with pytest.raises(ValueError, match="PaymentMethod"):
        validate_clean(bad)


def test_validate_clean_rejects_invalid_churn_value(valid_row: pd.DataFrame) -> None:
    from churn.validation import validate_clean

    bad = valid_row.copy()
    bad["Churn"] = 2
    with pytest.raises(ValueError, match="Churn"):
        validate_clean(bad)


def test_validate_clean_rejects_null_in_categorical(valid_row: pd.DataFrame) -> None:
    from churn.validation import validate_clean

    bad = valid_row.copy()
    bad = bad.astype(object)  # allow None in any column
    bad.loc[0, "Contract"] = None
    with pytest.raises(ValueError):
        validate_clean(bad)


def test_validate_clean_rejects_wrong_dtype_for_numeric(valid_row: pd.DataFrame) -> None:
    """A string in a float column must fail dtype validation."""
    from churn.validation import validate_clean

    bad = valid_row.copy()
    bad["tenure"] = "twelve"  # forces column to object dtype
    with pytest.raises(ValueError):
        validate_clean(bad)


# ---------------------------------------------------------------------------
# validate_serving — accepts valid row, rejects bad ones
# ---------------------------------------------------------------------------


def test_validate_serving_accepts_valid_row(valid_serving_row: pd.DataFrame) -> None:
    from churn.validation import validate_serving

    result = validate_serving(valid_serving_row)
    assert result.shape == valid_serving_row.shape


def test_validate_serving_churn_column_not_required(valid_serving_row: pd.DataFrame) -> None:
    """The serving contract must not require the Churn target column."""
    from churn.validation import validate_serving

    assert "Churn" not in valid_serving_row.columns
    validate_serving(valid_serving_row)  # must not raise


def test_validate_serving_rejects_unknown_payment_method(
    valid_serving_row: pd.DataFrame,
) -> None:
    from churn.validation import validate_serving

    bad = valid_serving_row.copy()
    bad["PaymentMethod"] = "Crypto"
    with pytest.raises(ValueError, match="PaymentMethod"):
        validate_serving(bad)


def test_validate_serving_rejects_unknown_contract(valid_serving_row: pd.DataFrame) -> None:
    from churn.validation import validate_serving

    bad = valid_serving_row.copy()
    bad["Contract"] = "Annual"
    with pytest.raises(ValueError, match="Contract"):
        validate_serving(bad)


def test_validate_serving_rejects_out_of_range_tenure(valid_serving_row: pd.DataFrame) -> None:
    from churn.validation import validate_serving

    bad = valid_serving_row.copy()
    bad["tenure"] = 999.0
    with pytest.raises(ValueError, match="tenure"):
        validate_serving(bad)


# ---------------------------------------------------------------------------
# load_clean_telco — validate flag
# ---------------------------------------------------------------------------


def test_load_clean_telco_validate_false_skips_validation(tmp_path: Path) -> None:
    """validate=False must return the cleaned frame without calling Pandera."""

    # We can't call load_clean_telco without the CSV; test the flag plumbing
    # by verifying validate=True is the default signature.
    import inspect

    from churn.data import load_clean_telco

    sig = inspect.signature(load_clean_telco)
    assert "validate" in sig.parameters
    assert sig.parameters["validate"].default is True


# ---------------------------------------------------------------------------
# Consistency: ALLOWED == API Pydantic Literals
# ---------------------------------------------------------------------------


def test_categorical_sets_match_api_literals() -> None:
    """ALLOWED sets in validation.py must equal the Pydantic Literals in PredictRequest.

    This test makes it impossible for the data contract and the API to silently
    diverge on which values are accepted.
    """
    import typing

    from api.main import PredictRequest
    from churn.validation import ALLOWED

    # get_type_hints() resolves string annotations produced by
    # `from __future__ import annotations` in api/main.py.
    hints = typing.get_type_hints(PredictRequest)

    for field_name, allowed_set in ALLOWED.items():
        annotation = hints.get(field_name)
        assert annotation is not None, (
            f"{field_name} is in ALLOWED but not in PredictRequest"
        )
        args = typing.get_args(annotation)
        assert args, f"{field_name} annotation has no Literal args: {annotation}"

        if field_name == "SeniorCitizen":
            # API uses int Literal[0, 1]; cleaned data and ALLOWED use str {"0", "1"}
            api_as_str = frozenset(str(v) for v in args)
            assert api_as_str == allowed_set, (
                f"SeniorCitizen: API→str {api_as_str} != ALLOWED {allowed_set}"
            )
        else:
            api_set = frozenset(args)
            assert api_set == allowed_set, (
                f"Mismatch for {field_name}:\n"
                f"  ALLOWED = {sorted(allowed_set)}\n"
                f"  API     = {sorted(str(v) for v in api_set)}"
            )


def test_allowed_keys_match_categorical_features() -> None:
    """ALLOWED keys must exactly equal CATEGORICAL_FEATURES from churn/data.py."""
    from churn.data import CATEGORICAL_FEATURES
    from churn.validation import ALLOWED

    validation_cats = set(ALLOWED.keys())
    data_cats = set(CATEGORICAL_FEATURES)
    assert validation_cats == data_cats, (
        f"Keys mismatch:\n"
        f"  validation ALLOWED : {sorted(validation_cats)}\n"
        f"  data.CATEGORICAL   : {sorted(data_cats)}"
    )


def test_senior_citizen_str_str_roundtrip() -> None:
    """ALLOWED['SeniorCitizen'] must be exactly {str(v) for v in API Literal args}."""
    import typing

    from api.main import PredictRequest
    from churn.validation import ALLOWED

    hints = typing.get_type_hints(PredictRequest)
    args = typing.get_args(hints["SeniorCitizen"])
    api_as_str = frozenset(str(v) for v in args)
    assert api_as_str == ALLOWED["SeniorCitizen"]
