"""Tests for churn/fairness.py — subgroup fairness analysis."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.fairness import subgroup_metrics

_csv_present = pytest.mark.skipif(
    not Path("data/raw/telco_churn.csv").exists(),
    reason="data/raw/telco_churn.csv not present",
)

# Fast build_final_model settings forwarded via build_final_model_kwargs.
_FAST_KWARGS = dict(cv=2, sample_frac=0.25)

_EXPECTED_COLUMNS = {
    "group", "n", "positive_rate",
    "recall", "fpr", "precision", "selection_rate", "roc_auc",
}


# ---------------------------------------------------------------------------
# subgroup_metrics — unit tests on synthetic data
# ---------------------------------------------------------------------------


def _synthetic_two_groups():
    """Synthetic example with a known recall disparity.

    Group A (label="A"): 10 positives, model predicts all correctly → recall=1.0
    Group B (label="B"): 10 positives, model misses half → recall=0.5
    No negatives in either group so FPR is undefined (NaN).
    """
    y_true = np.array([1] * 10 + [1] * 10)
    # A: all predicted positive; B: first 5 predicted positive, last 5 negative
    y_pred = np.array([1] * 10 + [1] * 5 + [0] * 5)
    proba = np.array([0.9] * 10 + [0.8] * 5 + [0.2] * 5)
    groups = pd.Series(["A"] * 10 + ["B"] * 10)
    return y_true, y_pred, proba, groups


def test_subgroup_metrics_returns_dataframe():
    y_true, y_pred, proba, groups = _synthetic_two_groups()
    result = subgroup_metrics(y_true, y_pred, proba, groups)
    assert isinstance(result, pd.DataFrame)


def test_subgroup_metrics_has_required_columns():
    y_true, y_pred, proba, groups = _synthetic_two_groups()
    result = subgroup_metrics(y_true, y_pred, proba, groups)
    assert _EXPECTED_COLUMNS.issubset(set(result.columns)), (
        f"Missing columns: {_EXPECTED_COLUMNS - set(result.columns)}"
    )


def test_subgroup_metrics_one_row_per_group():
    y_true, y_pred, proba, groups = _synthetic_two_groups()
    result = subgroup_metrics(y_true, y_pred, proba, groups)
    assert len(result) == 2


def test_subgroup_metrics_group_a_recall_is_one():
    """Group A: all 10 positives predicted positive → recall must be 1.0."""
    y_true, y_pred, proba, groups = _synthetic_two_groups()
    result = subgroup_metrics(y_true, y_pred, proba, groups)
    row_a = result[result["group"] == "A"].iloc[0]
    assert row_a["recall"] == pytest.approx(1.0)


def test_subgroup_metrics_group_b_recall_is_half():
    """Group B: 5/10 positives correctly predicted → recall must be 0.5."""
    y_true, y_pred, proba, groups = _synthetic_two_groups()
    result = subgroup_metrics(y_true, y_pred, proba, groups)
    row_b = result[result["group"] == "B"].iloc[0]
    assert row_b["recall"] == pytest.approx(0.5)


def test_subgroup_metrics_fpr_nan_when_no_negatives():
    """Groups A and B have no negatives → FPR must be NaN."""
    y_true, y_pred, proba, groups = _synthetic_two_groups()
    result = subgroup_metrics(y_true, y_pred, proba, groups)
    assert result["fpr"].isna().all(), "Expected all FPR to be NaN (no negatives)"


def test_subgroup_metrics_group_with_no_positives_no_crash():
    """A group with zero positives must not raise; recall must be NaN."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    proba = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.7])
    groups = pd.Series(["no_pos", "no_pos", "no_pos", "has_pos", "has_pos", "has_pos"])

    result = subgroup_metrics(y_true, y_pred, proba, groups)
    no_pos_row = result[result["group"] == "no_pos"].iloc[0]
    assert math.isnan(no_pos_row["recall"]), "recall should be NaN when no positives"
    assert not math.isnan(no_pos_row["fpr"]), "fpr should be defined when negatives exist"


def test_subgroup_metrics_group_with_no_negatives_fpr_nan():
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1])
    proba = np.array([0.9, 0.8, 0.2, 0.1, 0.6])
    groups = pd.Series(["only_pos", "only_pos", "has_neg", "has_neg", "has_neg"])

    result = subgroup_metrics(y_true, y_pred, proba, groups)
    only_pos = result[result["group"] == "only_pos"].iloc[0]
    assert math.isnan(only_pos["fpr"]), "fpr should be NaN when no negatives"


def test_subgroup_metrics_roc_auc_nan_single_class():
    """ROC-AUC must be NaN for a group that has only one class present."""
    # only_pos: indices 0,1 — both positive, no negatives → NaN ROC-AUC
    # has_both: indices 2,3,4,5 — one positive + three negatives → defined ROC-AUC
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0])
    proba = np.array([0.9, 0.4, 0.8, 0.2, 0.6, 0.1])
    groups = pd.Series(["only_pos", "only_pos", "has_both", "has_both", "has_both", "has_both"])

    result = subgroup_metrics(y_true, y_pred, proba, groups)
    only_pos = result[result["group"] == "only_pos"].iloc[0]
    assert math.isnan(only_pos["roc_auc"]), "ROC-AUC must be NaN when no negatives"
    has_both = result[result["group"] == "has_both"].iloc[0]
    assert not math.isnan(has_both["roc_auc"]), "ROC-AUC must be defined when both classes present"


def test_subgroup_metrics_n_sums_to_total():
    y_true, y_pred, proba, groups = _synthetic_two_groups()
    result = subgroup_metrics(y_true, y_pred, proba, groups)
    assert result["n"].sum() == len(y_true)


def test_subgroup_metrics_positive_rate_in_range():
    y_true, y_pred, proba, groups = _synthetic_two_groups()
    result = subgroup_metrics(y_true, y_pred, proba, groups)
    assert (result["positive_rate"].dropna().between(0, 1)).all()


def test_subgroup_metrics_selection_rate_in_range():
    y_true, y_pred, proba, groups = _synthetic_two_groups()
    result = subgroup_metrics(y_true, y_pred, proba, groups)
    assert (result["selection_rate"].dropna().between(0, 1)).all()


def test_subgroup_metrics_disparity_direction():
    """Verify that A has higher recall than B in the synthetic example."""
    y_true, y_pred, proba, groups = _synthetic_two_groups()
    result = subgroup_metrics(y_true, y_pred, proba, groups)
    recall_a = result[result["group"] == "A"]["recall"].values[0]
    recall_b = result[result["group"] == "B"]["recall"].values[0]
    assert recall_a > recall_b


# ---------------------------------------------------------------------------
# run_fairness_analysis — integration tests
# ---------------------------------------------------------------------------


@_csv_present
def test_run_fairness_returns_dict(tmp_path):
    from churn.fairness import run_fairness_analysis

    out = run_fairness_analysis(
        sensitive_features=["gender"],
        log_to_mlflow=False,
        build_final_model_kwargs=_FAST_KWARGS,
        reports_dir=tmp_path / "r",
    )
    assert isinstance(out, dict)
    assert set(out.keys()) == {"tables", "disparities", "model_result"}


@_csv_present
def test_run_fairness_tables_have_correct_columns(tmp_path):
    from churn.fairness import run_fairness_analysis

    out = run_fairness_analysis(
        sensitive_features=["gender", "SeniorCitizen"],
        log_to_mlflow=False,
        build_final_model_kwargs=_FAST_KWARGS,
        reports_dir=tmp_path / "r",
    )
    for feat, tbl in out["tables"].items():
        assert _EXPECTED_COLUMNS.issubset(set(tbl.columns)), (
            f"Feature '{feat}' table missing columns: {_EXPECTED_COLUMNS - set(tbl.columns)}"
        )
        assert "low_confidence" in tbl.columns


@_csv_present
def test_run_fairness_disparities_has_expected_columns(tmp_path):
    from churn.fairness import run_fairness_analysis

    out = run_fairness_analysis(
        sensitive_features=["gender"],
        log_to_mlflow=False,
        build_final_model_kwargs=_FAST_KWARGS,
        reports_dir=tmp_path / "r",
    )
    disp_cols = {"feature", "n_groups", "recall_min", "recall_max", "recall_gap",
                 "fpr_min", "fpr_max", "fpr_gap"}
    assert disp_cols.issubset(set(out["disparities"].columns))


@_csv_present
def test_run_fairness_one_disparity_row_per_feature(tmp_path):
    from churn.fairness import run_fairness_analysis

    feats = ["gender", "SeniorCitizen"]
    out = run_fairness_analysis(
        sensitive_features=feats,
        log_to_mlflow=False,
        build_final_model_kwargs=_FAST_KWARGS,
        reports_dir=tmp_path / "r",
    )
    assert len(out["disparities"]) == len(feats)


@_csv_present
def test_run_fairness_disparity_gaps_non_negative(tmp_path):
    from churn.fairness import run_fairness_analysis

    out = run_fairness_analysis(
        sensitive_features=["gender", "Partner"],
        log_to_mlflow=False,
        build_final_model_kwargs=_FAST_KWARGS,
        reports_dir=tmp_path / "r",
    )
    for _, row in out["disparities"].iterrows():
        if not math.isnan(row["recall_gap"]):
            assert row["recall_gap"] >= 0
        if not math.isnan(row["fpr_gap"]):
            assert row["fpr_gap"] >= 0


@_csv_present
def test_run_fairness_tables_keyed_by_feature(tmp_path):
    from churn.fairness import run_fairness_analysis

    feats = ["gender", "Dependents"]
    out = run_fairness_analysis(
        sensitive_features=feats,
        log_to_mlflow=False,
        build_final_model_kwargs=_FAST_KWARGS,
        reports_dir=tmp_path / "r",
    )
    assert set(out["tables"].keys()) == set(feats)


# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------


@_csv_present
def test_run_fairness_mlflow_integration(tmp_path):
    import mlflow

    from churn.fairness import run_fairness_analysis

    tracking_uri = f"sqlite:///{tmp_path / 'mlruns_fair.db'}"
    run_fairness_analysis(
        sensitive_features=["gender"],
        log_to_mlflow=True,
        tracking_uri=tracking_uri,
        experiment_name="test-fairness",
        build_final_model_kwargs=dict(cv=2, sample_frac=0.15),
        reports_dir=tmp_path / "r",
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test-fairness")
    assert experiment is not None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 1

    run = runs[0]
    # Recall gap for gender must be logged
    assert "gender_recall_gap" in run.data.metrics, (
        f"gender_recall_gap not in metrics: {list(run.data.metrics.keys())}"
    )
    # Fairness artifacts should be logged
    fairness_files = client.list_artifacts(run.info.run_id, "fairness")
    assert len(fairness_files) > 0, "No artifacts under fairness/"
