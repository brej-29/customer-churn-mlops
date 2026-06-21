"""Tests for churn/tuning.py — suggest_xgb_params + tune_xgboost."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlflow
import optuna
import pytest
from optuna.samplers import TPESampler
from optuna.trial import FixedTrial

from churn.tuning import (
    _TUNED_PARAM_KEYS,
    _XGB_FIXED,
    suggest_xgb_params,
    tune_xgboost,
)

_csv_present = pytest.mark.skipif(
    not Path("data/raw/telco_churn.csv").exists(),
    reason="data/raw/telco_churn.csv not present",
)

# Fast settings: 5 trials, 2-fold CV, 20% subsample — runs in ~30 s.
_FAST = dict(n_trials=5, cv=2, sample_frac=0.2, log_to_mlflow=False)


# ---------------------------------------------------------------------------
# suggest_xgb_params — keys and bounds
# ---------------------------------------------------------------------------


def test_suggest_xgb_params_returns_all_keys():
    trial = FixedTrial(
        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "gamma": 1.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
    )
    params = suggest_xgb_params(trial)
    assert set(params.keys()) == set(_TUNED_PARAM_KEYS)


def test_suggest_xgb_params_fixed_trial_values():
    """FixedTrial echo: the function must return the exact values given."""
    fixed_vals = {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "gamma": 2.5,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
    }
    trial = FixedTrial(fixed_vals)
    params = suggest_xgb_params(trial)
    for k, v in fixed_vals.items():
        assert params[k] == pytest.approx(v), f"{k}: expected {v}, got {params[k]}"


def test_suggest_xgb_params_sampled_bounds():
    """TPE-sampled params stay within every specified bound across 10 trials."""
    study = optuna.create_study(sampler=TPESampler(seed=0))

    def _obj(trial: optuna.Trial) -> float:
        p = suggest_xgb_params(trial)
        assert 100 <= p["n_estimators"] <= 1000, p["n_estimators"]
        assert 0.01 <= p["learning_rate"] <= 0.3, p["learning_rate"]
        assert 3 <= p["max_depth"] <= 10, p["max_depth"]
        assert 1 <= p["min_child_weight"] <= 10, p["min_child_weight"]
        assert 0.6 <= p["subsample"] <= 1.0, p["subsample"]
        assert 0.6 <= p["colsample_bytree"] <= 1.0, p["colsample_bytree"]
        assert 0.0 <= p["gamma"] <= 5.0, p["gamma"]
        assert 1e-3 <= p["reg_alpha"] <= 10.0, p["reg_alpha"]
        assert 1e-3 <= p["reg_lambda"] <= 10.0, p["reg_lambda"]
        return 0.5

    study.optimize(_obj, n_trials=10)


# ---------------------------------------------------------------------------
# _XGB_FIXED constant
# ---------------------------------------------------------------------------


def test_xgb_fixed_has_required_keys():
    required = {"objective", "eval_metric", "tree_method", "random_state", "n_jobs"}
    assert required.issubset(set(_XGB_FIXED.keys()))


def test_xgb_fixed_objective():
    assert _XGB_FIXED["objective"] == "binary:logistic"


def test_xgb_fixed_no_scale_pos_weight():
    # Step 4 showed scale_pos_weight harms calibration; must not appear.
    assert "scale_pos_weight" not in _XGB_FIXED


# ---------------------------------------------------------------------------
# tune_xgboost — return value and best_params
# ---------------------------------------------------------------------------


@_csv_present
def test_tune_returns_study(tmp_path):
    study = tune_xgboost(**_FAST, params_out=tmp_path / "p.json", reports_dir=tmp_path / "r")
    assert isinstance(study, optuna.Study)


@_csv_present
def test_tune_best_value_in_range(tmp_path):
    study = tune_xgboost(**_FAST, params_out=tmp_path / "p.json", reports_dir=tmp_path / "r")
    assert 0.0 <= study.best_value <= 1.0
    assert math.isfinite(study.best_value)


@_csv_present
def test_tune_best_params_keys(tmp_path):
    study = tune_xgboost(**_FAST, params_out=tmp_path / "p.json", reports_dir=tmp_path / "r")
    assert set(study.best_params.keys()) == set(_TUNED_PARAM_KEYS)


@_csv_present
def test_tune_expected_trials(tmp_path):
    study = tune_xgboost(**_FAST, params_out=tmp_path / "p.json", reports_dir=tmp_path / "r")
    assert len(study.trials) == _FAST["n_trials"]


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


@_csv_present
def test_tune_json_written(tmp_path):
    params_path = tmp_path / "params.json"
    tune_xgboost(**_FAST, params_out=params_path, reports_dir=tmp_path / "r")
    assert params_path.exists()


@_csv_present
def test_tune_json_contains_tuned_keys(tmp_path):
    params_path = tmp_path / "params.json"
    tune_xgboost(**_FAST, params_out=params_path, reports_dir=tmp_path / "r")
    data = json.loads(params_path.read_text())
    for k in _TUNED_PARAM_KEYS:
        assert k in data, f"Tuned key '{k}' missing from JSON"


@_csv_present
def test_tune_json_contains_fixed_keys(tmp_path):
    params_path = tmp_path / "params.json"
    tune_xgboost(**_FAST, params_out=params_path, reports_dir=tmp_path / "r")
    data = json.loads(params_path.read_text())
    for k in _XGB_FIXED:
        assert k in data, f"Fixed key '{k}' missing from JSON"


# ---------------------------------------------------------------------------
# Sanity: best_value must beat the base-rate PR-AUC floor
# ---------------------------------------------------------------------------


@_csv_present
def test_tune_beats_base_rate(tmp_path):
    study = tune_xgboost(**_FAST, params_out=tmp_path / "p.json", reports_dir=tmp_path / "r")
    assert study.best_value > 0.265, (
        f"best_value {study.best_value:.4f} does not exceed base-rate floor 0.265"
    )


# ---------------------------------------------------------------------------
# Reproducibility: same args → same best_value (seeded TPE + sequential CV)
# ---------------------------------------------------------------------------


@_csv_present
def test_tune_deterministic(tmp_path):
    a = tune_xgboost(**_FAST, params_out=tmp_path / "a.json", reports_dir=tmp_path / "r")
    b = tune_xgboost(**_FAST, params_out=tmp_path / "b.json", reports_dir=tmp_path / "r")
    assert abs(a.best_value - b.best_value) < 1e-6, (
        f"Non-deterministic: {a.best_value:.8f} vs {b.best_value:.8f}"
    )


# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------


@_csv_present
def test_tune_mlflow_creates_run(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlruns_tune.db'}"
    params_path = tmp_path / "params.json"

    tune_xgboost(
        n_trials=3,
        cv=2,
        sample_frac=0.1,
        log_to_mlflow=True,
        tracking_uri=tracking_uri,
        experiment_name="test-tuning",
        params_out=params_path,
        reports_dir=tmp_path / "r",
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test-tuning")
    assert experiment is not None, "Experiment was not created"

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 1, "No runs found"

    run = runs[0]
    assert "best_cv_pr_auc" in run.data.metrics, (
        f"best_cv_pr_auc not in metrics: {list(run.data.metrics.keys())}"
    )
    # At least one best_ param must be logged
    best_keys = [k for k in run.data.params if k.startswith("best_")]
    assert best_keys, f"No best_* params logged: {list(run.data.params.keys())}"
