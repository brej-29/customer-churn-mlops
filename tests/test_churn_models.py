"""Tests for churn/models.py — leaderboard, pipeline factory, MLflow integration."""

from __future__ import annotations

import math

import mlflow
import pandas as pd
import pytest

from churn.models import (
    build_model_pipeline,
    get_candidate_models,
    run_leaderboard,
)

_csv_present = pytest.mark.skipif(
    not __import__("pathlib").Path("data/raw/telco_churn.csv").exists(),
    reason="data/raw/telco_churn.csv not present",
)

# ── fast leaderboard parameters used by almost every test ────────────────────
_FAST = dict(cv=2, sample_frac=0.15, log_to_mlflow=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fast_leaderboard():
    """Run the leaderboard once with all candidates; cache for the module."""
    if not __import__("pathlib").Path("data/raw/telco_churn.csv").exists():
        pytest.skip("data/raw/telco_churn.csv not present")
    return run_leaderboard(**_FAST)


# ---------------------------------------------------------------------------
# build_model_pipeline
# ---------------------------------------------------------------------------


def test_build_model_pipeline_has_two_steps():
    from sklearn.dummy import DummyClassifier

    pipe = build_model_pipeline(DummyClassifier())
    assert list(pipe.named_steps.keys()) == ["preprocessor", "model"]


def test_build_model_pipeline_model_step_is_estimator():
    from sklearn.dummy import DummyClassifier

    est = DummyClassifier()
    pipe = build_model_pipeline(est)
    assert pipe.named_steps["model"] is est


# ---------------------------------------------------------------------------
# get_candidate_models
# ---------------------------------------------------------------------------


def test_candidate_models_keys():
    models = get_candidate_models()
    expected = {"dummy", "logreg", "xgboost", "lightgbm", "catboost", "mlp"}
    assert set(models.keys()) == expected


def test_candidate_models_are_unfitted():
    from sklearn.utils.validation import check_is_fitted

    for name, est in get_candidate_models().items():
        try:
            check_is_fitted(est)
            raise AssertionError(f"Model '{name}' appears already fitted")
        except Exception as exc:
            # NotFittedError is the expected outcome
            if "NotFittedError" not in type(exc).__name__:
                raise


# ---------------------------------------------------------------------------
# run_leaderboard — shape and schema
# ---------------------------------------------------------------------------


@_csv_present
def test_leaderboard_has_one_row_per_model(fast_leaderboard):
    lb = fast_leaderboard
    expected_models = set(get_candidate_models().keys())
    assert set(lb["model"]) == expected_models


@_csv_present
def test_leaderboard_has_required_columns(fast_leaderboard):
    required = {
        "model", "pr_auc_mean", "pr_auc_std",
        "roc_auc_mean", "roc_auc_std",
        "brier_mean", "brier_std", "fit_time_mean",
    }
    assert required.issubset(set(fast_leaderboard.columns))


@_csv_present
def test_leaderboard_sorted_by_pr_auc_descending(fast_leaderboard):
    scores = fast_leaderboard["pr_auc_mean"].tolist()
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# run_leaderboard — metric validity
# ---------------------------------------------------------------------------


@_csv_present
def test_leaderboard_no_nans(fast_leaderboard):
    numeric_cols = fast_leaderboard.select_dtypes("number").columns
    assert not fast_leaderboard[numeric_cols].isna().any().any()


@_csv_present
def test_leaderboard_pr_auc_in_range(fast_leaderboard):
    for _, row in fast_leaderboard.iterrows():
        assert 0.0 <= row["pr_auc_mean"] <= 1.0, (
            f"{row['model']} pr_auc_mean={row['pr_auc_mean']}"
        )
        assert 0.0 <= row["pr_auc_std"], (
            f"{row['model']} pr_auc_std negative"
        )


@_csv_present
def test_leaderboard_roc_auc_in_range(fast_leaderboard):
    for _, row in fast_leaderboard.iterrows():
        assert 0.0 <= row["roc_auc_mean"] <= 1.0


@_csv_present
def test_leaderboard_brier_in_range(fast_leaderboard):
    for _, row in fast_leaderboard.iterrows():
        assert 0.0 <= row["brier_mean"] <= 1.0


@_csv_present
def test_leaderboard_all_metrics_finite(fast_leaderboard):
    numeric_cols = fast_leaderboard.select_dtypes("number").columns
    for col in numeric_cols:
        for val in fast_leaderboard[col]:
            assert math.isfinite(val), f"Non-finite value in column '{col}': {val}"


@_csv_present
def test_leaderboard_fit_time_positive(fast_leaderboard):
    assert (fast_leaderboard["fit_time_mean"] > 0).all()


# ---------------------------------------------------------------------------
# Sanity gate: XGBoost beats Dummy on PR-AUC
# ---------------------------------------------------------------------------


@_csv_present
def test_xgboost_beats_dummy_pr_auc(fast_leaderboard):
    lb = fast_leaderboard.set_index("model")
    xgb_score   = lb.loc["xgboost",  "pr_auc_mean"]
    dummy_score = lb.loc["dummy",    "pr_auc_mean"]
    margin = 0.10  # XGBoost should exceed Dummy by at least 10 pp
    assert xgb_score > dummy_score + margin, (
        f"XGBoost PR-AUC {xgb_score:.4f} does not exceed Dummy {dummy_score:.4f} by {margin}"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@_csv_present
def test_leaderboard_is_deterministic():
    lb_a = run_leaderboard(**_FAST)
    lb_b = run_leaderboard(**_FAST)
    # Exclude fit_time_mean — it is wall-clock time and cannot be deterministic.
    metric_cols = [
        "model", "pr_auc_mean", "pr_auc_std",
        "roc_auc_mean", "roc_auc_std", "brier_mean", "brier_std",
    ]
    lb_a = lb_a[metric_cols].sort_values("model").reset_index(drop=True)
    lb_b = lb_b[metric_cols].sort_values("model").reset_index(drop=True)
    pd.testing.assert_frame_equal(lb_a, lb_b, atol=1e-6)


# ---------------------------------------------------------------------------
# Subset-model run (smoke test without catboost for speed in CI)
# ---------------------------------------------------------------------------


@_csv_present
def test_leaderboard_subset_models():
    from collections import OrderedDict

    from sklearn.dummy import DummyClassifier
    from xgboost import XGBClassifier

    from churn.models import SEED

    subset = OrderedDict([
        ("dummy",   DummyClassifier(strategy="prior")),
        ("xgboost", XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1)),
    ])
    lb = run_leaderboard(cv=2, sample_frac=0.10, log_to_mlflow=False, models=subset)
    assert len(lb) == 2
    assert set(lb["model"]) == {"dummy", "xgboost"}


# ---------------------------------------------------------------------------
# MLflow integration: temp tracking URI, parent + nested runs, metrics logged
# ---------------------------------------------------------------------------


@_csv_present
def test_mlflow_logging_creates_parent_and_child_runs(tmp_path):
    from collections import OrderedDict

    from sklearn.dummy import DummyClassifier
    from xgboost import XGBClassifier

    from churn.models import SEED

    tracking_uri = f"sqlite:///{tmp_path / 'mlruns_test.db'}"

    subset = OrderedDict([
        ("dummy",   DummyClassifier(strategy="prior")),
        ("xgboost", XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1)),
    ])

    lb = run_leaderboard(
        cv=2,
        sample_frac=0.10,
        log_to_mlflow=True,
        tracking_uri=tracking_uri,
        experiment_name="test-leaderboard",
        models=subset,
    )

    # Verify structure in the temp tracking store
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test-leaderboard")
    assert experiment is not None, "Experiment was not created in temp tracking store"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time ASC"],
    )
    assert len(runs) >= 3, (
        f"Expected at least 3 runs (1 parent + 2 children), got {len(runs)}"
    )

    # Find a child run and check that pr_auc_mean was logged
    child_runs = [r for r in runs if r.data.tags.get("mlflow.parentRunId")]
    assert child_runs, "No nested child runs found"

    logged_metrics = child_runs[0].data.metrics
    assert "pr_auc_mean" in logged_metrics, (
        f"pr_auc_mean not found in child run metrics: {list(logged_metrics.keys())}"
    )

    # Leaderboard result should still be valid
    assert set(lb["model"]) == {"dummy", "xgboost"}
    assert (lb["pr_auc_mean"].between(0, 1)).all()
