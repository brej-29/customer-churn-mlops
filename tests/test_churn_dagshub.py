"""Offline tests for DagsHub / DVC / MLflow config wiring.

All tests run without network access. No DagsHub credentials required.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# MLflow tracking URI — env var vs default
# ---------------------------------------------------------------------------


def test_mlflow_uri_reads_bare_env_var(monkeypatch: object) -> None:
    """MLFLOW_TRACKING_URI env var (bare, no CHURN_ prefix) must be reflected in Settings."""
    import os

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/brej-29/customer-churn-mlops.mlflow")
    monkeypatch.delenv("CHURN_MLFLOW_TRACKING_URI", raising=False)

    # Ensure load_dotenv does not pull in a .env file that could override
    monkeypatch.setattr(os, "environ", {**os.environ})

    from churn.config import Settings

    s = Settings()
    assert s.mlflow_tracking_uri == "https://dagshub.com/brej-29/customer-churn-mlops.mlflow"


def test_mlflow_uri_default_is_sqlite(monkeypatch: object) -> None:
    """When neither MLFLOW_TRACKING_URI nor CHURN_MLFLOW_TRACKING_URI is set, default to SQLite."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("CHURN_MLFLOW_TRACKING_URI", raising=False)

    from churn.config import Settings

    s = Settings()
    assert s.mlflow_tracking_uri == "sqlite:///mlruns.db"


def test_mlflow_uri_churn_prefix_still_works(monkeypatch: object) -> None:
    """CHURN_MLFLOW_TRACKING_URI (pydantic-settings prefix) must also be honoured."""
    monkeypatch.setenv("CHURN_MLFLOW_TRACKING_URI", "sqlite:///custom.db")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    from churn.config import Settings

    s = Settings()
    assert s.mlflow_tracking_uri == "sqlite:///custom.db"


# ---------------------------------------------------------------------------
# DVC dataset tracking
# ---------------------------------------------------------------------------


def test_dvc_pointer_file_exists() -> None:
    """data/raw/telco_churn.csv.dvc must exist after dvc add."""
    assert Path("data/raw/telco_churn.csv.dvc").exists(), (
        "DVC pointer file not found. Run: dvc add data/raw/telco_churn.csv"
    )


def test_dvc_pointer_has_md5_and_path() -> None:
    """The .dvc pointer must contain an md5 hash and a path field."""
    import yaml  # bundled with DVC, always available

    pointer = yaml.safe_load(Path("data/raw/telco_churn.csv.dvc").read_text())
    out = pointer["outs"][0]
    assert "md5" in out, "DVC pointer missing md5 field"
    assert out["path"] == "telco_churn.csv", f"Unexpected path in .dvc: {out['path']}"


def test_raw_csv_not_git_tracked() -> None:
    """After git rm --cached, the raw CSV must no longer appear in git ls-files."""
    result = subprocess.run(
        ["git", "ls-files", "data/raw/telco_churn.csv"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "", (
        "data/raw/telco_churn.csv is still git-tracked. Run: git rm --cached data/raw/telco_churn.csv"
    )


def test_raw_csv_still_on_disk() -> None:
    """The raw CSV must still exist on disk so local tests and training keep passing."""
    assert Path("data/raw/telco_churn.csv").exists(), (
        "data/raw/telco_churn.csv is missing from disk. Run: dvc pull"
    )


# ---------------------------------------------------------------------------
# DVC config — no secrets committed
# ---------------------------------------------------------------------------


def test_dvc_config_exists() -> None:
    """.dvc/config must exist (committed; contains non-secret remote config)."""
    assert Path(".dvc/config").exists()


def test_dvc_config_has_no_secrets() -> None:
    """.dvc/config must not contain access keys or tokens."""
    config_text = Path(".dvc/config").read_text()
    forbidden = ("access_key_id", "secret_access_key", "password", "token")
    for keyword in forbidden:
        assert keyword not in config_text.lower(), (
            f".dvc/config contains '{keyword}' — secrets must go in .dvc/config.local"
        )


def test_dvc_config_has_dagshub_endpoint() -> None:
    """.dvc/config must point at the DagsHub S3-compatible endpoint."""
    config_text = Path(".dvc/config").read_text()
    assert "dagshub.com" in config_text, "DagsHub endpoint not found in .dvc/config"
    assert "s3://dvc" in config_text, "S3 bucket URL not found in .dvc/config"
