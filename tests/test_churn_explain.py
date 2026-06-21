"""Tests for churn/explain.py — SHAP explainability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_csv_present = pytest.mark.skipif(
    not Path("data/raw/telco_churn.csv").exists(),
    reason="data/raw/telco_churn.csv not present",
)

_FAST = dict(sample_size=300, log_to_mlflow=False)

N_FEATURES = 54  # 6 numeric + 48 OHE categorical


# ---------------------------------------------------------------------------
# Module-scoped fixture — compute SHAP once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shap_result(tmp_path_factory):
    """Run compute_shap once at fast settings; reused by all shape/key tests."""
    if not Path("data/raw/telco_churn.csv").exists():
        pytest.skip("data/raw/telco_churn.csv not present")
    from churn.explain import compute_shap
    return compute_shap(**_FAST, reports_dir=tmp_path_factory.mktemp("shap"))


# ---------------------------------------------------------------------------
# compute_shap — shape, no NaN, keys
# ---------------------------------------------------------------------------


@_csv_present
def test_compute_shap_returns_required_keys(shap_result):
    assert set(shap_result.keys()) == {"shap_values", "feature_names", "importance_df", "importance_agg"}


@_csv_present
def test_compute_shap_values_shape(shap_result):
    sv = shap_result["shap_values"]
    assert sv.ndim == 2, f"Expected 2D shap_values, got shape {sv.shape}"
    assert sv.shape[1] == N_FEATURES, f"Expected {N_FEATURES} features, got {sv.shape[1]}"
    assert sv.shape[0] <= 300, f"Sample rows exceed sample_size=300: {sv.shape[0]}"


@_csv_present
def test_compute_shap_no_nans(shap_result):
    assert not np.isnan(shap_result["shap_values"]).any(), "SHAP values contain NaN"


@_csv_present
def test_compute_shap_feature_names_length(shap_result):
    assert len(shap_result["feature_names"]) == N_FEATURES


@_csv_present
def test_compute_shap_importance_df_length(shap_result):
    assert len(shap_result["importance_df"]) == N_FEATURES


@_csv_present
def test_compute_shap_importance_df_sorted(shap_result):
    vals = shap_result["importance_df"]["mean_abs_shap"].values
    assert (np.diff(vals) <= 0).all(), "importance_df is not sorted descending"


@_csv_present
def test_compute_shap_importance_df_non_negative(shap_result):
    assert (shap_result["importance_df"]["mean_abs_shap"] >= 0).all()


@_csv_present
def test_compute_shap_importance_agg_non_empty(shap_result):
    assert len(shap_result["importance_agg"]) > 0


@_csv_present
def test_compute_shap_importance_agg_fewer_features(shap_result):
    """Aggregating OHE columns must give fewer entries than N_FEATURES."""
    assert len(shap_result["importance_agg"]) < N_FEATURES


@_csv_present
def test_compute_shap_feature_names_prefixes(shap_result):
    """All transformed feature names must start with 'num__' or 'cat__'."""
    for name in shap_result["feature_names"]:
        assert name.startswith("num__") or name.startswith("cat__"), (
            f"Unexpected feature name prefix: {name!r}"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@_csv_present
def test_compute_shap_deterministic(tmp_path):
    from churn.explain import compute_shap

    a = compute_shap(**_FAST, reports_dir=tmp_path / "ra")
    b = compute_shap(**_FAST, reports_dir=tmp_path / "rb")
    np.testing.assert_allclose(
        a["shap_values"], b["shap_values"], atol=1e-6,
        err_msg="compute_shap is not deterministic across two calls with the same args",
    )


# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------


@_csv_present
def test_compute_shap_mlflow_integration(tmp_path):
    import mlflow

    from churn.explain import compute_shap

    tracking_uri = f"sqlite:///{tmp_path / 'mlruns_shap.db'}"
    compute_shap(
        sample_size=200,
        log_to_mlflow=True,
        tracking_uri=tracking_uri,
        experiment_name="test-explain",
        reports_dir=tmp_path / "r",
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test-explain")
    assert experiment is not None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 1

    run = runs[0]
    # At least one SHAP rank metric must be logged
    shap_metrics = [k for k in run.data.metrics if k.startswith("shap_rank")]
    assert len(shap_metrics) > 0, (
        f"No shap_rank* metrics logged: {list(run.data.metrics.keys())}"
    )
    # Importance CSV artifact should exist under "shap/"
    model_files = client.list_artifacts(run.info.run_id, "shap")
    assert len(model_files) > 0, "No artifacts logged under shap/"


# ---------------------------------------------------------------------------
# _parse_original_name unit tests
# ---------------------------------------------------------------------------


def test_parse_original_name_numeric():
    from churn.explain import _parse_original_name

    assert _parse_original_name("num__tenure") == "tenure"
    assert _parse_original_name("num__MonthlyCharges") == "MonthlyCharges"
    assert _parse_original_name("num__spend_growth_ratio") == "spend_growth_ratio"


def test_parse_original_name_categorical():
    from churn.explain import _parse_original_name

    assert _parse_original_name("cat__gender_Female") == "gender"
    assert _parse_original_name("cat__PaymentMethod_Electronic check") == "PaymentMethod"
    assert _parse_original_name("cat__tenure_bucket_0-12") == "tenure_bucket"
    assert _parse_original_name("cat__Contract_Month-to-month") == "Contract"
