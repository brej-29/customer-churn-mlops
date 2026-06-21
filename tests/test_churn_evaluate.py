"""Tests for churn/evaluate.py — helpers and build_final_model orchestrator."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from churn.evaluate import (
    FinalModelResult,
    assess_calibration,
    build_final_model,
    select_threshold_by_cost,
)

_csv_present = pytest.mark.skipif(
    not Path("data/raw/telco_churn.csv").exists(),
    reason="data/raw/telco_churn.csv not present",
)

# Fast parameters: small CV, 25% subsample — keeps nested CV manageable.
_FAST = dict(cv=2, sample_frac=0.25, log_to_mlflow=False)


# ---------------------------------------------------------------------------
# select_threshold_by_cost
# ---------------------------------------------------------------------------


def _cost_fixture():
    """Synthetic example with known cost-optimal threshold.

    proba:  [0.9, 0.8, 0.6, 0.4, 0.35, 0.3, 0.2, 0.1]
    y_true: [1,   1,   1,   0,   0,    0,   0,   0  ]  (3 pos, 5 neg)

    With fn_cost=5, fp_cost=1:
      t ≤ 0.35 → FN=1, cost floor reached when FP also 0 → no improvement below
      t in (0.35, 0.60] → pred=[1,1,1], FN=0, FP=0, cost=0
      t in (0.60, 0.80] → FN=1, FP=0, cost=5
      t > 0.90          → FN=3, FP=0, cost=15
    The cost-minimising region is t ∈ (0.35, 0.60] with cost=0.
    """
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.6, 0.4, 0.35, 0.3, 0.2, 0.1])
    return y, p


def test_select_threshold_returns_threshold_in_range():
    y, p = _cost_fixture()
    result = select_threshold_by_cost(y, p)
    assert 0.0 < result["threshold"] < 1.0


def test_select_threshold_optimal_cost_is_zero():
    """With perfectly separable data the cost minimum must be 0."""
    y, p = _cost_fixture()
    result = select_threshold_by_cost(y, p, fn_cost=5.0, fp_cost=1.0)
    assert result["costs"].min() == pytest.approx(0.0)


def test_select_threshold_optimal_region():
    """Cost-optimal threshold lands in the (0.35, 0.60] zero-cost region."""
    y, p = _cost_fixture()
    result = select_threshold_by_cost(y, p, fn_cost=5.0, fp_cost=1.0)
    # argmin returns the FIRST minimum in the sweep, which is the smallest
    # threshold that achieves cost=0. With n_thresholds=200 the grid step is
    # ~0.005, so the optimal should sit in (0.35, 0.61).
    assert 0.35 < result["threshold"] < 0.61


def test_select_threshold_returns_required_keys():
    y, p = _cost_fixture()
    result = select_threshold_by_cost(y, p)
    assert {"threshold", "thresholds", "costs", "f1_threshold"} == set(result.keys())


def test_select_threshold_arrays_same_length():
    y, p = _cost_fixture()
    result = select_threshold_by_cost(y, p, n_thresholds=50)
    assert len(result["thresholds"]) == len(result["costs"]) == 50


def test_select_threshold_costs_non_negative():
    y, p = _cost_fixture()
    result = select_threshold_by_cost(y, p)
    assert (result["costs"] >= 0).all()


def test_select_threshold_f1_threshold_in_range():
    y, p = _cost_fixture()
    result = select_threshold_by_cost(y, p)
    assert 0.0 < result["f1_threshold"] < 1.0


def test_select_threshold_higher_fn_cost_lowers_threshold():
    """Raising fn_cost (more expensive FNs) should shift the threshold down.

    Use overlapping distributions so the optimal threshold is not trivially 0
    or 1.
    y_true: [1,1,1,0,0,0,0,0]
    proba:  [0.9,0.7,0.4,0.8,0.6,0.35,0.2,0.1]
    (Some negatives score above some positives — real overlap.)

    With fn_cost=1, fp_cost=10 → strongly penalise FP → prefer high threshold.
    With fn_cost=10, fp_cost=1 → strongly penalise FN → prefer low threshold.
    """
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.7, 0.4, 0.8, 0.6, 0.35, 0.2, 0.1])

    low_fn = select_threshold_by_cost(y, p, fn_cost=10.0, fp_cost=1.0)
    high_fn_cost_is_low_fp = select_threshold_by_cost(y, p, fn_cost=1.0, fp_cost=10.0)
    assert low_fn["threshold"] < high_fn_cost_is_low_fp["threshold"]


# ---------------------------------------------------------------------------
# assess_calibration
# ---------------------------------------------------------------------------


def test_assess_calibration_brier_in_range():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200)
    p = rng.random(200)
    result = assess_calibration(y, p)
    assert 0.0 <= result["brier"] <= 1.0


def test_assess_calibration_brier_finite():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=100)
    p = rng.random(100)
    result = assess_calibration(y, p)
    assert math.isfinite(result["brier"])


def test_assess_calibration_returns_required_keys():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=100)
    p = rng.random(100)
    result = assess_calibration(y, p)
    assert {"brier", "bin_centers", "frac_pos", "bin_counts"} == set(result.keys())


def test_assess_calibration_array_shapes_consistent():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=300)
    p = rng.random(300)
    result = assess_calibration(y, p, n_bins=10)
    # All three arrays should have the same length (# non-empty bins)
    n = len(result["bin_centers"])
    assert n == len(result["frac_pos"]) == len(result["bin_counts"])
    assert 1 <= n <= 10  # at most n_bins non-empty bins


def test_assess_calibration_perfect_model_low_brier():
    """A perfectly calibrated model should have near-zero Brier."""
    y = np.array([0] * 100 + [1] * 100)
    p = np.array([0.0] * 100 + [1.0] * 100)
    result = assess_calibration(y, p)
    assert result["brier"] < 0.01


def test_assess_calibration_frac_pos_in_range():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, size=200)
    p = rng.random(200)
    result = assess_calibration(y, p)
    assert (result["frac_pos"] >= 0.0).all()
    assert (result["frac_pos"] <= 1.0).all()


# ---------------------------------------------------------------------------
# build_final_model — end-to-end
# ---------------------------------------------------------------------------


@_csv_present
def test_build_final_model_returns_result(tmp_path):
    result = build_final_model(**_FAST, threshold_out=tmp_path / "t.json")
    assert isinstance(result, FinalModelResult)


@_csv_present
def test_build_final_model_threshold_in_range(tmp_path):
    result = build_final_model(**_FAST, threshold_out=tmp_path / "t.json")
    assert 0.0 < result.threshold < 1.0


@_csv_present
def test_build_final_model_calibration_method_string(tmp_path):
    result = build_final_model(**_FAST, threshold_out=tmp_path / "t.json")
    assert result.calibration_method in ("uncalibrated", "isotonic")


@_csv_present
def test_build_final_model_test_metrics_valid(tmp_path):
    result = build_final_model(**_FAST, threshold_out=tmp_path / "t.json")
    m = result.test_metrics
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["brier"] <= 1.0
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0


@_csv_present
def test_build_final_model_test_metrics_finite(tmp_path):
    result = build_final_model(**_FAST, threshold_out=tmp_path / "t.json")
    for k, v in result.test_metrics.items():
        if k == "confusion_matrix":
            continue
        assert math.isfinite(v), f"Non-finite metric '{k}': {v}"


@_csv_present
def test_build_final_model_confusion_matrix_shape(tmp_path):
    result = build_final_model(**_FAST, threshold_out=tmp_path / "t.json")
    cm = result.test_metrics["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2


@_csv_present
def test_build_final_model_threshold_json_written(tmp_path):
    tpath = tmp_path / "threshold.json"
    build_final_model(**_FAST, threshold_out=tpath)
    assert tpath.exists()
    data = json.loads(tpath.read_text())
    assert "threshold" in data
    assert "fn_cost" in data
    assert "fp_cost" in data
    assert "calibration_method" in data
    assert 0.0 < data["threshold"] < 1.0


@_csv_present
def test_build_final_model_model_can_predict(tmp_path):
    """The returned model must be fit and produce valid probabilities on X_test."""
    from churn.data import get_splits

    result = build_final_model(**_FAST, threshold_out=tmp_path / "t.json")
    _, X_test, _, y_test = get_splits()
    proba = result.model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert (proba >= 0).all() and (proba <= 1).all()


@_csv_present
def test_build_final_model_brier_oof_fields(tmp_path):
    result = build_final_model(**_FAST, threshold_out=tmp_path / "t.json")
    assert 0.0 <= result.uncal_brier_oof <= 1.0
    assert 0.0 <= result.cal_brier_oof <= 1.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@_csv_present
def test_build_final_model_deterministic(tmp_path):
    a = build_final_model(**_FAST, threshold_out=tmp_path / "a.json")
    b = build_final_model(**_FAST, threshold_out=tmp_path / "b.json")
    assert abs(a.threshold - b.threshold) < 1e-9
    assert abs(a.test_metrics["pr_auc"] - b.test_metrics["pr_auc"]) < 1e-6


# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------


@_csv_present
def test_build_final_model_mlflow_integration(tmp_path):
    import mlflow

    tracking_uri = f"sqlite:///{tmp_path / 'mlruns_eval.db'}"
    tpath = tmp_path / "threshold.json"

    result = build_final_model(
        cv=2,
        sample_frac=0.15,
        log_to_mlflow=True,
        tracking_uri=tracking_uri,
        experiment_name="test-final",
        threshold_out=tpath,
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test-final")
    assert experiment is not None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 1

    run = runs[0]
    assert "test_pr_auc" in run.data.metrics, (
        f"test_pr_auc not logged: {list(run.data.metrics.keys())}"
    )
    assert "threshold" in run.data.metrics
    assert "calibration_method" in run.data.params

    # With MLflow 3.x name= parameter, the model is stored at path "final_model/"
    # but doesn't appear in the root list_artifacts listing — query it explicitly.
    model_files = client.list_artifacts(run.info.run_id, "final_model")
    assert len(model_files) > 0, (
        f"No files found under final_model/ for run {run.info.run_id}"
    )
    # Result should still be valid
    assert 0.0 <= result.test_metrics["pr_auc"] <= 1.0
