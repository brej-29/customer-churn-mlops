"""Pandera data-validation contracts for the Telco Customer Churn dataset.

Both the training contract (cleaned frame + target) and the serving contract
(19 feature columns, no target) derive from a single source of truth: the
``ALLOWED`` mapping of categorical feature → permitted value set.

Usage
-----
Training pipeline  : ``validate_clean(df)``   — call after ``clean_telco()``
Serving / batch    : ``validate_serving(df)`` — call before the scoring pipeline
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

# ---------------------------------------------------------------------------
# Single source of truth — allowed categorical values
# ---------------------------------------------------------------------------

_YES_NO: frozenset[str] = frozenset({"No", "Yes"})
_YES_NO_NS: frozenset[str] = frozenset({"No", "No internet service", "Yes"})
_YES_NO_NPS: frozenset[str] = frozenset({"No", "No phone service", "Yes"})

# ALLOWED is the one place to update when the Telco vocabulary changes.
# TRAIN_SCHEMA and SERVE_SCHEMA both derive their isin() checks from it, so
# train and serve contracts can never silently diverge on categorical domains.
# The consistency test in tests/test_churn_validation.py asserts that each
# set here equals the Pydantic Literal in api/main.py::PredictRequest.
ALLOWED: dict[str, frozenset[str]] = {
    "gender": frozenset({"Female", "Male"}),
    "SeniorCitizen": frozenset({"0", "1"}),  # str after clean_telco(); int at API boundary
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": _YES_NO_NPS,
    "InternetService": frozenset({"DSL", "Fiber optic", "No"}),
    "OnlineSecurity": _YES_NO_NS,
    "OnlineBackup": _YES_NO_NS,
    "DeviceProtection": _YES_NO_NS,
    "TechSupport": _YES_NO_NS,
    "StreamingTV": _YES_NO_NS,
    "StreamingMovies": _YES_NO_NS,
    "Contract": frozenset({"Month-to-month", "One year", "Two year"}),
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": frozenset({
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    }),
}

# ---------------------------------------------------------------------------
# Schema builders — shared by both contracts
# ---------------------------------------------------------------------------


def _cat_col(allowed: frozenset[str]) -> pa.Column:
    return pa.Column(dtype=str, checks=pa.Check.isin(list(allowed)), nullable=False)


def _feature_columns() -> dict[str, pa.Column]:
    """Build the 19 feature columns that appear in both training and serving schemas."""
    cols: dict[str, pa.Column] = {
        "tenure": pa.Column(
            dtype=float,
            checks=[pa.Check.ge(0), pa.Check.le(72)],
            nullable=False,
        ),
        "MonthlyCharges": pa.Column(
            dtype=float,
            checks=pa.Check.gt(0),
            nullable=False,
        ),
        "TotalCharges": pa.Column(
            dtype=float,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
    }
    for feature, allowed in ALLOWED.items():
        cols[feature] = _cat_col(allowed)
    return cols


# ---------------------------------------------------------------------------
# Exported schemas
# ---------------------------------------------------------------------------

_TRAIN_COLS: dict[str, pa.Column] = _feature_columns()
_TRAIN_COLS["Churn"] = pa.Column(dtype=int, checks=pa.Check.isin([0, 1]), nullable=False)

TRAIN_SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema(
    columns=_TRAIN_COLS,
    coerce=False,
    strict=False,  # extra columns are allowed; all listed columns are required
)

SERVE_SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema(
    columns=_feature_columns(),
    coerce=False,
    strict=False,
)

# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------


def _format_errors(exc: pa.errors.SchemaErrors) -> str:
    try:
        cols = exc.failure_cases["column"].dropna().unique().tolist()
        detail = exc.failure_cases[["column", "check", "failure_case"]].to_string(index=False)
        return f"column(s) {cols}:\n{detail}"
    except Exception:
        return str(exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the post-``clean_telco`` DataFrame against the training contract.

    Runs a lazy (collect-all-errors) Pandera check. On violation, raises
    ``ValueError`` naming every offending column and failure case. Returns
    ``df`` unchanged on success.
    """
    try:
        return TRAIN_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValueError(
            f"validate_clean: {len(exc.failure_cases)} violation(s) in {_format_errors(exc)}"
        ) from exc
    except pa.errors.SchemaError as exc:
        raise ValueError(f"validate_clean: {exc}") from exc


def validate_serving(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a serving-input DataFrame (19 features, no Churn target).

    Reuses the same ``ALLOWED`` sets as the training contract. Suitable for
    batch scoring and drift-monitoring pipelines.
    """
    try:
        return SERVE_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValueError(
            f"validate_serving: {len(exc.failure_cases)} violation(s) in {_format_errors(exc)}"
        ) from exc
    except pa.errors.SchemaError as exc:
        raise ValueError(f"validate_serving: {exc}") from exc
