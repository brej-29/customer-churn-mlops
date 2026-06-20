"""Tests for churn/registry.py — champion/challenger gate."""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.exceptions
import pytest
from sklearn.dummy import DummyClassifier

from churn.registry import register_with_gate, should_promote

_csv_present = pytest.mark.skipif(
    not Path("data/raw/telco_churn.csv").exists(),
    reason="data/raw/telco_churn.csv not present",
)


# ---------------------------------------------------------------------------
# Helper: log a trivial sklearn model and return its runs:/ URI
# ---------------------------------------------------------------------------


def _log_dummy(tracking_uri: str, experiment: str = "test-registry") -> str:
    """Log a DummyClassifier to a temp MLflow run; return the model URI."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as r:
        m = DummyClassifier(strategy="most_frequent").fit([[0], [1]], [0, 1])
        # Use artifact_path (traditional API) so the model is at a well-known path.
        mlflow.sklearn.log_model(sk_model=m, artifact_path="model")
        return f"runs:/{r.info.run_id}/model"


# ---------------------------------------------------------------------------
# should_promote — pure unit tests (no I/O)
# ---------------------------------------------------------------------------


def test_should_promote_no_champion():
    """None champion → always promote (first registration)."""
    assert should_promote(0.66, None) is True


def test_should_promote_better_candidate():
    """Strictly better candidate → promote."""
    assert should_promote(0.67, 0.66) is True


def test_should_promote_worse_candidate():
    """Worse candidate → do not promote."""
    assert should_promote(0.65, 0.66) is False


def test_should_promote_equal_candidate():
    """Equal candidate with min_improvement=0 → not promoted (must be strictly better)."""
    assert should_promote(0.66, 0.66) is False


def test_should_promote_min_improvement_not_met():
    """Improvement 0.001 < min_improvement=0.005 → not promoted."""
    assert should_promote(0.661, 0.66, min_improvement=0.005) is False


def test_should_promote_min_improvement_met():
    """Improvement 0.01 > min_improvement=0.005 → promote."""
    assert should_promote(0.67, 0.66, min_improvement=0.005) is True


def test_should_promote_exactly_at_boundary():
    """Improvement exactly equals min_improvement → not promoted (strictly greater required)."""
    assert should_promote(0.665, 0.66, min_improvement=0.005) is False


def test_should_promote_zero_improvement_with_margin():
    """No improvement at all with positive min_improvement → not promoted."""
    assert should_promote(0.50, 0.66, min_improvement=0.001) is False


# ---------------------------------------------------------------------------
# register_with_gate — integration tests against a temp tracking URI
# ---------------------------------------------------------------------------


def test_first_registration_gets_champion_alias(tmp_path):
    """The very first registered version must receive the 'champion' alias."""
    tracking_uri = f"sqlite:///{tmp_path / 'reg.db'}"
    uri = _log_dummy(tracking_uri)

    result = register_with_gate(
        model_uri=uri,
        candidate_metric=0.66,
        model_name="test-model",
        threshold=0.35,
        calibration_method="uncalibrated",
        tracking_uri=tracking_uri,
    )

    assert result["alias"] == "champion"
    assert result["promoted"] is True
    assert result["is_first_registration"] is True
    assert result["champion_metric"] is None


def test_first_registration_version_is_one(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'reg.db'}"
    uri = _log_dummy(tracking_uri)
    result = register_with_gate(
        uri, 0.66, "test-model", 0.35, "uncalibrated", tracking_uri=tracking_uri
    )
    assert result["new_version"] == "1"


def test_worse_candidate_gets_challenger_alias(tmp_path):
    """A worse candidate must receive 'challenger', not 'champion'."""
    tracking_uri = f"sqlite:///{tmp_path / 'reg.db'}"
    uri1 = _log_dummy(tracking_uri, "test-registry")
    register_with_gate(uri1, 0.66, "test-model", 0.35, "uncalibrated", tracking_uri=tracking_uri)

    uri2 = _log_dummy(tracking_uri, "test-registry")
    result2 = register_with_gate(
        uri2, 0.65, "test-model", 0.35, "uncalibrated", tracking_uri=tracking_uri
    )

    assert result2["alias"] == "challenger"
    assert result2["promoted"] is False


def test_worse_candidate_champion_alias_unchanged(tmp_path):
    """When a worse candidate is registered, 'champion' must still point to version 1.

    This is the key test that proves the legacy unconditional-promotion bug is fixed.
    """
    tracking_uri = f"sqlite:///{tmp_path / 'reg.db'}"
    uri1 = _log_dummy(tracking_uri)
    r1 = register_with_gate(
        uri1, 0.66, "test-model", 0.35, "uncalibrated", tracking_uri=tracking_uri
    )

    uri2 = _log_dummy(tracking_uri)
    register_with_gate(
        uri2, 0.65, "test-model", 0.35, "uncalibrated", tracking_uri=tracking_uri
    )

    # Champion alias must still resolve to version 1 (the better model).
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    champ = client.get_model_version_by_alias("test-model", "champion")
    assert str(champ.version) == str(r1["new_version"]), (
        f"Champion moved to {champ.version!r} but should still be {r1['new_version']!r}"
    )


def test_better_candidate_displaces_champion(tmp_path):
    """A strictly better candidate must take the 'champion' alias."""
    tracking_uri = f"sqlite:///{tmp_path / 'reg.db'}"
    uri1 = _log_dummy(tracking_uri)
    register_with_gate(uri1, 0.66, "test-model", 0.35, "uncalibrated", tracking_uri=tracking_uri)

    uri2 = _log_dummy(tracking_uri)
    r2 = register_with_gate(
        uri2, 0.70, "test-model", 0.35, "uncalibrated", tracking_uri=tracking_uri
    )

    assert r2["alias"] == "champion"
    assert r2["promoted"] is True

    # Verify via client
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    champ = client.get_model_version_by_alias("test-model", "champion")
    assert str(champ.version) == str(r2["new_version"])


def test_version_tags_are_written(tmp_path):
    """test_pr_auc, threshold, calibration_method, registered_at must all be tagged."""
    tracking_uri = f"sqlite:///{tmp_path / 'reg.db'}"
    uri = _log_dummy(tracking_uri)
    result = register_with_gate(
        uri, 0.66, "test-model", 0.174, "isotonic", tracking_uri=tracking_uri
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version("test-model", result["new_version"])

    assert mv.tags.get("test_pr_auc") == str(0.66)
    assert mv.tags.get("threshold") == str(0.174)
    assert mv.tags.get("calibration_method") == "isotonic"
    assert "registered_at" in mv.tags and mv.tags["registered_at"] != ""


def test_candidate_metric_recorded_in_result(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'reg.db'}"
    uri = _log_dummy(tracking_uri)
    result = register_with_gate(
        uri, 0.6597, "test-model", 0.35, "uncalibrated", tracking_uri=tracking_uri
    )
    assert result["candidate_metric"] == pytest.approx(0.6597)


def test_champion_metric_recorded_in_second_result(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'reg.db'}"
    uri1 = _log_dummy(tracking_uri)
    register_with_gate(uri1, 0.66, "test-model", 0.35, "uncalibrated", tracking_uri=tracking_uri)

    uri2 = _log_dummy(tracking_uri)
    result2 = register_with_gate(
        uri2, 0.65, "test-model", 0.35, "uncalibrated", tracking_uri=tracking_uri
    )
    assert result2["champion_metric"] == pytest.approx(0.66)


def test_min_improvement_blocks_marginal_gain(tmp_path):
    """A candidate that beats the champion by less than min_improvement is a challenger."""
    tracking_uri = f"sqlite:///{tmp_path / 'reg.db'}"
    uri1 = _log_dummy(tracking_uri)
    register_with_gate(uri1, 0.66, "test-model", 0.35, "uncalibrated", tracking_uri=tracking_uri)

    uri2 = _log_dummy(tracking_uri)
    result2 = register_with_gate(
        uri2, 0.661, "test-model", 0.35, "uncalibrated",
        min_improvement=0.005, tracking_uri=tracking_uri,
    )
    assert result2["alias"] == "challenger"


def test_result_dict_has_required_keys(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'reg.db'}"
    uri = _log_dummy(tracking_uri)
    result = register_with_gate(
        uri, 0.66, "test-model", 0.35, "uncalibrated", tracking_uri=tracking_uri
    )
    required = {
        "model_name", "new_version", "alias", "promoted",
        "candidate_metric", "champion_metric",
        "is_first_registration", "min_improvement",
    }
    assert required.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# FinalModelResult.run_id / model_uri — evaluate.py additive fields
# ---------------------------------------------------------------------------


@_csv_present
def test_final_model_result_has_run_id_when_logged(tmp_path):
    from churn.evaluate import build_final_model

    tracking_uri = f"sqlite:///{tmp_path / 'eval.db'}"
    result = build_final_model(
        cv=2, sample_frac=0.15,
        log_to_mlflow=True,
        tracking_uri=tracking_uri,
        experiment_name="test-run-id",
        threshold_out=tmp_path / "t.json",
    )
    assert result.run_id is not None
    assert isinstance(result.run_id, str) and len(result.run_id) > 0


@_csv_present
def test_final_model_result_model_uri_format(tmp_path):
    from churn.evaluate import build_final_model

    tracking_uri = f"sqlite:///{tmp_path / 'eval.db'}"
    result = build_final_model(
        cv=2, sample_frac=0.15,
        log_to_mlflow=True,
        tracking_uri=tracking_uri,
        experiment_name="test-model-uri",
        threshold_out=tmp_path / "t.json",
    )
    assert result.model_uri is not None
    assert result.model_uri.startswith("runs:/")
    assert "final_model" in result.model_uri
    assert result.run_id in result.model_uri


@_csv_present
def test_final_model_result_no_log_has_none_fields(tmp_path):
    from churn.evaluate import build_final_model

    result = build_final_model(
        cv=2, sample_frac=0.15,
        log_to_mlflow=False,
        threshold_out=tmp_path / "t.json",
    )
    assert result.run_id is None
    assert result.model_uri is None
