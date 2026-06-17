"""Tests for the new churn.config and churn.data modules (Tier 1 rebuild, setup step)."""

import pytest

from churn.config import Settings
from churn.data import TELCO_EXPECTED_SHAPE, load_telco_raw


def test_settings_defaults():
    s = Settings()
    assert s.random_seed == 42
    assert str(s.telco_csv_path) == r"data\raw\telco_churn.csv" or str(
        s.telco_csv_path
    ) == "data/raw/telco_churn.csv"
    assert str(s.mlflow_tracking_uri) == "file:./mlruns"


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("CHURN_RANDOM_SEED", "99")
    monkeypatch.setenv("CHURN_MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
    s = Settings()
    assert s.random_seed == 99
    assert s.mlflow_tracking_uri == "sqlite:///mlruns.db"


def test_load_telco_raw_missing_file(tmp_path):
    absent = tmp_path / "no_such_file.csv"
    with pytest.raises(FileNotFoundError, match="Telco CSV not found"):
        load_telco_raw(csv_path=absent)


def test_load_telco_raw_error_message_is_actionable(tmp_path):
    absent = tmp_path / "no_such_file.csv"
    with pytest.raises(FileNotFoundError, match="kaggle"):
        load_telco_raw(csv_path=absent)


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/raw/telco_churn.csv").exists(),
    reason="data/raw/telco_churn.csv not present — skipping shape assertion",
)
def test_load_telco_raw_shape():
    df = load_telco_raw()
    assert df.shape == TELCO_EXPECTED_SHAPE, (
        f"Expected shape {TELCO_EXPECTED_SHAPE}, got {df.shape}"
    )
