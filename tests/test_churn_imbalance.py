"""Tests for the Step 4 imbalance experiment: build_imbalance_pipeline + run_imbalance_experiment."""

from __future__ import annotations

import math

import mlflow
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from churn.models import (
    _IMBALANCE_STRATEGIES,
    build_imbalance_pipeline,
    run_imbalance_experiment,
)

_csv_present = pytest.mark.skipif(
    not __import__("pathlib").Path("data/raw/telco_churn.csv").exists(),
    reason="data/raw/telco_churn.csv not present",
)

_FAST = dict(cv=2, sample_frac=0.2, log_to_mlflow=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fast_imbalance(tmp_path_factory):
    """Run the experiment once at fast settings; cache for the module."""
    if not __import__("pathlib").Path("data/raw/telco_churn.csv").exists():
        pytest.skip("data/raw/telco_churn.csv not present")
    return run_imbalance_experiment(**_FAST, reports_dir=tmp_path_factory.mktemp("imb"))


# ---------------------------------------------------------------------------
# run_imbalance_experiment — shape, schema, metric validity
# ---------------------------------------------------------------------------


@_csv_present
def test_imbalance_has_one_row_per_strategy(fast_imbalance):
    assert set(fast_imbalance["strategy"]) == set(_IMBALANCE_STRATEGIES)
    assert len(fast_imbalance) == len(_IMBALANCE_STRATEGIES)


@_csv_present
def test_imbalance_has_required_columns(fast_imbalance):
    required = {
        "strategy", "pr_auc_mean", "pr_auc_std",
        "roc_auc_mean", "roc_auc_std",
        "brier_mean", "brier_std", "fit_time_mean",
    }
    assert required.issubset(set(fast_imbalance.columns))


@_csv_present
def test_imbalance_sorted_by_pr_auc_descending(fast_imbalance):
    scores = fast_imbalance["pr_auc_mean"].tolist()
    assert scores == sorted(scores, reverse=True)


@_csv_present
def test_imbalance_no_nans(fast_imbalance):
    numeric_cols = fast_imbalance.select_dtypes("number").columns
    assert not fast_imbalance[numeric_cols].isna().any().any()


@_csv_present
def test_imbalance_all_metrics_finite(fast_imbalance):
    numeric_cols = fast_imbalance.select_dtypes("number").columns
    for col in numeric_cols:
        for val in fast_imbalance[col]:
            assert math.isfinite(val), f"Non-finite in '{col}': {val}"


@_csv_present
def test_imbalance_pr_auc_in_range(fast_imbalance):
    for _, row in fast_imbalance.iterrows():
        assert 0.0 <= row["pr_auc_mean"] <= 1.0
        assert row["pr_auc_std"] >= 0.0


@_csv_present
def test_imbalance_roc_auc_in_range(fast_imbalance):
    for _, row in fast_imbalance.iterrows():
        assert 0.0 <= row["roc_auc_mean"] <= 1.0


@_csv_present
def test_imbalance_brier_in_range(fast_imbalance):
    for _, row in fast_imbalance.iterrows():
        assert 0.0 <= row["brier_mean"] <= 1.0


@_csv_present
def test_all_strategies_beat_base_rate(fast_imbalance):
    """Every strategy must exceed the base-rate PR-AUC floor (~0.265)."""
    base_rate_floor = 0.265
    for _, row in fast_imbalance.iterrows():
        assert row["pr_auc_mean"] > base_rate_floor, (
            f"Strategy '{row['strategy']}' PR-AUC {row['pr_auc_mean']:.4f} "
            f"does not exceed base-rate floor {base_rate_floor}"
        )


# ---------------------------------------------------------------------------
# SMOTENC correctness: synthetic categoricals stay within original value sets
# ---------------------------------------------------------------------------


@_csv_present
def test_smotenc_categoricals_validity():
    """After SMOTENC fit_resample on a post-FE frame, each categorical column
    contains ONLY values observed in the original training data.  This proves
    SMOTENC (not plain SMOTE) is operating: plain SMOTE would interpolate
    numeric codes and produce fractional / unseen category values."""
    from imblearn.over_sampling import SMOTENC

    from churn.data import get_splits
    from churn.features import CT_CATEGORICAL, ChurnFeatureEngineer

    X_train, _, y_train, _ = get_splits()
    # Keep ~10% for speed; still gives >140 minority samples (above k=5).
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train, train_size=0.1, stratify=y_train, random_state=42
    )

    fe = ChurnFeatureEngineer()
    X_fe = fe.fit_transform(X_sub)

    # Record the original category universe per column.
    orig_cats = {col: set(X_fe[col].astype(str).unique()) for col in CT_CATEGORICAL}

    smotenc = SMOTENC(categorical_features=CT_CATEGORICAL, random_state=42)
    X_res, _ = smotenc.fit_resample(X_fe, y_sub)

    for col in CT_CATEGORICAL:
        resampled_cats = set(X_res[col].astype(str).unique())
        novel = resampled_cats - orig_cats[col]
        assert not novel, (
            f"SMOTENC introduced novel values in column '{col}': {novel}"
        )


# ---------------------------------------------------------------------------
# No leakage at inference: sampler does NOT resample during predict
# ---------------------------------------------------------------------------


@_csv_present
def test_no_leakage_at_inference():
    """Fitting a 'smotenc' pipeline on (X, y) and calling predict/predict_proba
    on X must return exactly len(X) outputs — the sampler must not inflate the
    output at inference time."""
    from churn.data import get_splits
    from churn.models import SEED

    X_train, _, y_train, _ = get_splits()
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train, train_size=0.1, stratify=y_train, random_state=SEED
    )

    n_neg = int((y_sub == 0).sum())
    n_pos = int((y_sub == 1).sum())
    spw = float(n_neg / n_pos)

    pipe = build_imbalance_pipeline("smotenc", spw=spw)
    pipe.fit(X_sub, y_sub)

    preds = pipe.predict(X_sub)
    probas = pipe.predict_proba(X_sub)

    assert len(preds) == len(X_sub), (
        f"predict returned {len(preds)} outputs for {len(X_sub)} inputs"
    )
    assert len(probas) == len(X_sub), (
        f"predict_proba returned {len(probas)} outputs for {len(X_sub)} inputs"
    )


# ---------------------------------------------------------------------------
# Pipeline structure tests (no data needed)
# ---------------------------------------------------------------------------


def test_structure_none_is_sklearn_pipeline():
    from sklearn.pipeline import Pipeline as SkPipeline

    pipe = build_imbalance_pipeline("none")
    assert isinstance(pipe, SkPipeline)
    # imblearn Pipeline is a subclass — type() check confirms it's pure sklearn
    assert type(pipe) is SkPipeline


def test_structure_scale_pos_weight_is_sklearn_pipeline():
    from sklearn.pipeline import Pipeline as SkPipeline

    pipe = build_imbalance_pipeline("scale_pos_weight", spw=2.77)
    assert type(pipe) is SkPipeline


def test_structure_scale_pos_weight_has_positive_spw():
    pipe = build_imbalance_pipeline("scale_pos_weight", spw=2.77)
    xgb = pipe.named_steps["model"]
    assert xgb.scale_pos_weight > 0
    assert xgb.scale_pos_weight == pytest.approx(2.77)


def test_structure_smotenc_is_imblearn_pipeline():
    from imblearn.pipeline import Pipeline as ImbPipeline

    pipe = build_imbalance_pipeline("smotenc")
    assert isinstance(pipe, ImbPipeline)
    assert "smotenc" in pipe.named_steps
    from imblearn.over_sampling import SMOTENC as _SMOTENC

    assert isinstance(pipe.named_steps["smotenc"], _SMOTENC)


def test_structure_smotenc_pipeline_step_order():
    """imblearn pipeline for smotenc must be [fe, smotenc, ct, model]."""
    pipe = build_imbalance_pipeline("smotenc")
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["fe", "smotenc", "ct", "model"]


def test_structure_none_has_no_sampler():
    """'none' pipeline must not contain any imblearn sampler."""
    from imblearn.base import BaseSampler

    pipe = build_imbalance_pipeline("none")
    for _, step in pipe.steps:
        assert not isinstance(step, BaseSampler), (
            f"'none' pipeline unexpectedly contains sampler: {type(step).__name__}"
        )


def test_structure_invalid_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_imbalance_pipeline("bad_strategy")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@_csv_present
def test_imbalance_deterministic(tmp_path):
    a = run_imbalance_experiment(**_FAST, reports_dir=tmp_path / "ra")
    b = run_imbalance_experiment(**_FAST, reports_dir=tmp_path / "rb")
    metric_cols = [
        "strategy", "pr_auc_mean", "pr_auc_std",
        "roc_auc_mean", "roc_auc_std", "brier_mean", "brier_std",
    ]
    a = a[metric_cols].sort_values("strategy").reset_index(drop=True)
    b = b[metric_cols].sort_values("strategy").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------


@_csv_present
def test_imbalance_mlflow_integration(tmp_path):
    """log_to_mlflow=True must create a parent run + one nested run per strategy
    with pr_auc_mean logged on each child."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlruns_imb.db'}"

    result = run_imbalance_experiment(
        cv=2,
        sample_frac=0.1,
        log_to_mlflow=True,
        tracking_uri=tracking_uri,
        experiment_name="test-imbalance",
        strategies=["none", "scale_pos_weight"],
        reports_dir=tmp_path / "r",
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test-imbalance")
    assert experiment is not None, "Experiment was not created"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time ASC"],
    )
    # Expect: 1 parent + 2 children (one per strategy) + possible extra artifact runs
    assert len(runs) >= 3, (
        f"Expected ≥ 3 runs (1 parent + 2 children), got {len(runs)}"
    )

    child_runs = [r for r in runs if r.data.tags.get("mlflow.parentRunId")]
    assert child_runs, "No nested child runs found"
    assert "pr_auc_mean" in child_runs[0].data.metrics, (
        f"pr_auc_mean not logged: {list(child_runs[0].data.metrics.keys())}"
    )

    # Return value must still be a valid DataFrame
    assert set(result["strategy"]) == {"none", "scale_pos_weight"}
    assert (result["pr_auc_mean"].between(0, 1)).all()
