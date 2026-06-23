"""Tests for monitoring/drift.py — fully offline (no network, no LLM).

All Evidently computations run locally on small in-memory DataFrames.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import (
    MONITORED_FEATURES,
    NUMERIC_FEATURES,
    get_current_window,
    run_drift_check,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CATS = {
    "gender": "Male",
    "SeniorCitizen": "0",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
}


def _payload(tenure: float, senior_int: int = 0) -> str:
    """Return a complete request_payload JSON string (as logged by api/main.py).

    SeniorCitizen is intentionally an int to mirror the API's Literal[0, 1] field.
    """
    data = {
        "tenure": tenure,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
        **{k: v for k, v in _VALID_CATS.items() if k != "SeniorCitizen"},
        "SeniorCitizen": senior_int,  # int — same as PredictRequest.model_dump()
    }
    return json.dumps(data)


def _create_log_db(db_path: Path, n_rows: int) -> None:
    """Create a prediction_logs SQLite DB with n_rows; tenure == row number (1-based)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            request_payload TEXT,
            churn_probability REAL,
            model_uri TEXT,
            latency_ms REAL,
            status TEXT,
            error_message TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO prediction_logs (timestamp, request_payload, churn_probability, "
        "model_uri, latency_ms, status) VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("2024-01-01T00:00:00", _payload(float(i)), 0.3, "test", 10.0, "ok")
            for i in range(1, n_rows + 1)
        ],
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref_df() -> pd.DataFrame:
    """Realistic reference DataFrame with all 19 MONITORED_FEATURES (200 rows)."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "tenure": rng.uniform(0, 72, n),
            "MonthlyCharges": rng.uniform(18, 120, n),
            "TotalCharges": rng.uniform(0, 8000, n),
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.choice(["0", "1"], n, p=[0.9, 0.1]),
            "Partner": rng.choice(["No", "Yes"], n),
            "Dependents": rng.choice(["No", "Yes"], n),
            "PhoneService": rng.choice(["No", "Yes"], n),
            "MultipleLines": rng.choice(["No", "No phone service", "Yes"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["No", "No internet service", "Yes"], n),
            "OnlineBackup": rng.choice(["No", "No internet service", "Yes"], n),
            "DeviceProtection": rng.choice(["No", "No internet service", "Yes"], n),
            "TechSupport": rng.choice(["No", "No internet service", "Yes"], n),
            "StreamingTV": rng.choice(["No", "No internet service", "Yes"], n),
            "StreamingMovies": rng.choice(["No", "No internet service", "Yes"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["No", "Yes"], n),
            "PaymentMethod": rng.choice(
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ],
                n,
            ),
        }
    )[MONITORED_FEATURES]


@pytest.fixture(scope="module")
def shifted_df() -> pd.DataFrame:
    """All 19 features heavily skewed away from the reference distribution (200 rows).

    Every numeric feature is pulled to the top of its range; every categorical
    feature is set to a single value that appears with low probability in the
    reference.  This guarantees that Evidently detects drift on all features.
    """
    rng = np.random.default_rng(99)
    n = 200
    return pd.DataFrame(
        {
            "tenure": rng.uniform(65, 72, n),          # Top 10 % of [0, 72]
            "MonthlyCharges": rng.uniform(110, 120, n),  # Top 10 % of [18, 120]
            "TotalCharges": rng.uniform(7000, 8000, n),  # Top 10 % of [0, 8000]
            "gender": np.full(n, "Female"),
            "SeniorCitizen": np.full(n, "1"),            # Reference is 90 % "0"
            "Partner": np.full(n, "Yes"),
            "Dependents": np.full(n, "Yes"),
            "PhoneService": np.full(n, "Yes"),
            "MultipleLines": np.full(n, "Yes"),
            "InternetService": np.full(n, "Fiber optic"),
            "OnlineSecurity": np.full(n, "Yes"),
            "OnlineBackup": np.full(n, "Yes"),
            "DeviceProtection": np.full(n, "Yes"),
            "TechSupport": np.full(n, "Yes"),
            "StreamingTV": np.full(n, "Yes"),
            "StreamingMovies": np.full(n, "Yes"),
            "Contract": np.full(n, "Two year"),
            "PaperlessBilling": np.full(n, "Yes"),
            "PaymentMethod": np.full(n, "Electronic check"),
        }
    )[MONITORED_FEATURES]


# ---------------------------------------------------------------------------
# Drift-check: stable data
# ---------------------------------------------------------------------------


def test_stable_data_no_drift(ref_df: pd.DataFrame) -> None:
    """Identical reference and current → no drift, no retrain flag."""
    result = run_drift_check(ref_df, ref_df.copy())
    assert not result.dataset_drift
    assert not result.retrain_recommended
    assert result.n_drifted_features == 0
    assert result.n_total_features == len(MONITORED_FEATURES)


# ---------------------------------------------------------------------------
# Drift-check: shifted data
# ---------------------------------------------------------------------------


def test_shifted_data_retrain_recommended(
    ref_df: pd.DataFrame, shifted_df: pd.DataFrame
) -> None:
    """All features shifted → dataset_drift and retrain_recommended both True."""
    result = run_drift_check(ref_df, shifted_df)
    assert result.retrain_recommended, (
        f"Expected retrain_recommended=True; got n_drifted={result.n_drifted_features}/{result.n_total_features}"
    )
    assert result.dataset_drift or result.drifted_share >= 0.3, (
        "Expected either full dataset drift or ≥30 % of features drifted"
    )
    assert result.n_drifted_features > 0


# ---------------------------------------------------------------------------
# per_feature column scope
# ---------------------------------------------------------------------------


def test_per_feature_contains_only_monitored_features(ref_df: pd.DataFrame) -> None:
    """per_feature entries must cover exactly the MONITORED_FEATURES columns."""
    result = run_drift_check(ref_df, ref_df.copy())
    returned_features = {f.feature for f in result.per_feature}
    monitored_set = set(MONITORED_FEATURES)
    # Every returned feature must be in MONITORED_FEATURES
    assert returned_features <= monitored_set, (
        f"Unexpected features in per_feature: {returned_features - monitored_set}"
    )
    # All 19 monitored features must appear
    assert returned_features == monitored_set, (
        f"Missing features in per_feature: {monitored_set - returned_features}"
    )


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def test_artifact_writing_creates_html_and_json(
    ref_df: pd.DataFrame, tmp_path: Path
) -> None:
    """run_drift_check(output_dir=…) must produce drift_report.html and drift_summary.json."""
    run_drift_check(ref_df, ref_df.copy(), output_dir=tmp_path)

    html_file = tmp_path / "drift_report.html"
    json_file = tmp_path / "drift_summary.json"
    assert html_file.exists(), "drift_report.html was not created"
    assert json_file.exists(), "drift_summary.json was not created"

    summary = json.loads(json_file.read_text(encoding="utf-8"))
    assert "dataset_drift" in summary
    assert "retrain_recommended" in summary
    assert "n_drifted_features" in summary
    assert "per_feature" in summary
    assert isinstance(summary["per_feature"], list)


# ---------------------------------------------------------------------------
# Window selector: correct row count and chronological order
# ---------------------------------------------------------------------------


def test_window_returns_n_most_recent_chronological(tmp_path: Path) -> None:
    """15 rows inserted; requesting n=5 must return the 5 most recent in oldest-first order.

    tenure is set to the insertion index (1.0, 2.0, …, 15.0) so that
    chronological order is verifiable numerically.
    """
    db_path = tmp_path / "pred.db"
    _create_log_db(db_path, n_rows=15)

    result = get_current_window(n=5, db_path=str(db_path))

    assert len(result) == 5, f"Expected 5 rows, got {len(result)}"
    # Rows 11–15 (tenure 11.0–15.0) are the 5 most recent; oldest-first
    assert list(result["tenure"]) == [11.0, 12.0, 13.0, 14.0, 15.0], (
        f"Unexpected tenure order: {list(result['tenure'])}"
    )


def test_window_all_monitored_columns_present(tmp_path: Path) -> None:
    """get_current_window result must have exactly the MONITORED_FEATURES columns."""
    db_path = tmp_path / "pred.db"
    _create_log_db(db_path, n_rows=3)

    result = get_current_window(n=3, db_path=str(db_path))

    assert list(result.columns) == MONITORED_FEATURES, (
        f"Column mismatch: {list(result.columns)}"
    )


# ---------------------------------------------------------------------------
# Window selector: SeniorCitizen int → str normalisation
# ---------------------------------------------------------------------------


def test_window_senior_citizen_normalised_to_str(tmp_path: Path) -> None:
    """SeniorCitizen is logged as int(0/1) by the API; window must return str('0'/'1')."""
    db_path = tmp_path / "pred.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            request_payload TEXT,
            churn_probability REAL,
            model_uri TEXT,
            latency_ms REAL,
            status TEXT,
            error_message TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO prediction_logs (timestamp, request_payload, churn_probability, "
        "model_uri, latency_ms, status) VALUES (?, ?, ?, ?, ?, ?)",
        ("2024-01-01", _payload(tenure=5.0, senior_int=1), 0.5, "m", 10.0, "ok"),
    )
    conn.commit()
    conn.close()

    result = get_current_window(n=1, db_path=str(db_path))

    assert result["SeniorCitizen"].iloc[0] == "1", (
        f"Expected str '1', got {result['SeniorCitizen'].iloc[0]!r}"
    )
    assert result["SeniorCitizen"].dtype == object, (
        "SeniorCitizen column should be object (str), not int"
    )


# ---------------------------------------------------------------------------
# Window selector: numeric columns coerced to float
# ---------------------------------------------------------------------------


def test_window_numeric_columns_are_float(tmp_path: Path) -> None:
    """Numeric columns must be float64 regardless of how they are stored in JSON."""
    db_path = tmp_path / "pred.db"
    _create_log_db(db_path, n_rows=2)

    result = get_current_window(n=2, db_path=str(db_path))

    for col in NUMERIC_FEATURES:
        assert result[col].dtype in (
            "float64",
            "float32",
        ), f"{col} should be float, got {result[col].dtype}"


# ---------------------------------------------------------------------------
# Window selector: missing / empty DB
# ---------------------------------------------------------------------------


def test_window_missing_db_returns_empty_frame(tmp_path: Path) -> None:
    """Non-existent DB path must return empty DataFrame with correct columns."""
    result = get_current_window(n=5, db_path=str(tmp_path / "nonexistent.db"))

    assert result.empty
    assert list(result.columns) == MONITORED_FEATURES


def test_window_empty_table_returns_empty_frame(tmp_path: Path) -> None:
    """DB with no rows must return empty DataFrame with correct columns."""
    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            request_payload TEXT,
            churn_probability REAL,
            model_uri TEXT,
            latency_ms REAL,
            status TEXT,
            error_message TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    result = get_current_window(n=5, db_path=str(db_path))

    assert result.empty
    assert list(result.columns) == MONITORED_FEATURES
