"""Model pipeline factory, candidate registry, and cross-validated leaderboard."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.config import settings
from churn.data import get_splits
from churn.features import build_preprocessor

SEED: int = settings.random_seed

# Metrics reported in the leaderboard.
# neg_brier_score is negated by sklearn; we flip the sign when assembling results.
_SCORERS: dict[str, str] = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "brier": "neg_brier_score",
}

# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def build_model_pipeline(estimator) -> Pipeline:
    """Return a full end-to-end Pipeline: preprocessor then *estimator*.

    Reused by every later step (tuning, training, serving) so the same
    preprocessing is always applied in the same order.
    """
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])


# ---------------------------------------------------------------------------
# Candidate model registry
# ---------------------------------------------------------------------------


def get_candidate_models() -> OrderedDict:
    """Return an ordered dict of {name: unfitted estimator}.

    Rules: library defaults + fixed seed + quiet logging only.
    No class_weight / scale_pos_weight / resampling — this is the pure
    out-of-the-box baseline comparison. Imbalance handling comes later.

    MLPClassifier is the optional neural baseline. On tabular data it is
    expected to underperform GBMs; that result is itself informative.
    LogisticRegression gets max_iter=2000 to ensure convergence on 54 features.
    """
    return OrderedDict([
        ("dummy",    DummyClassifier(strategy="prior")),
        ("logreg",   LogisticRegression(max_iter=2000, random_state=SEED)),
        ("xgboost",  XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1)),
        ("lightgbm", LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1)),
        ("catboost", CatBoostClassifier(random_seed=SEED, verbose=0)),
        ("mlp",      MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=400,
            early_stopping=True,
            random_state=SEED,
        )),
    ])


# ---------------------------------------------------------------------------
# Cross-validated leaderboard
# ---------------------------------------------------------------------------


def run_leaderboard(
    cv: int = 5,
    sample_frac: Optional[float] = None,
    log_to_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "churn-leaderboard",
    models: Optional[dict] = None,
) -> pd.DataFrame:
    """Cross-validate every candidate on the TRAIN split; return a sorted leaderboard.

    Parameters
    ----------
    cv:
        Number of StratifiedKFold folds.
    sample_frac:
        If set, take a stratified subsample of X_train for fast testing.
    log_to_mlflow:
        Whether to log runs to MLflow. Set False in tests to keep them fast.
    tracking_uri:
        Override the MLflow tracking URI (defaults to settings.mlflow_tracking_uri).
        Pass a tmp_path string in tests to avoid touching the real store.
    experiment_name:
        MLflow experiment name for the parent run.
    models:
        Override the candidate dict (subset for targeted tests).

    Returns
    -------
    DataFrame sorted by pr_auc_mean descending, one row per model.
    Also writes reports/leaderboard.csv.
    """
    X_train, _, y_train, _ = get_splits()

    # Optional stratified subsample for fast CI / unit testing
    if sample_frac is not None:
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train,
            train_size=sample_frac,
            stratify=y_train,
            random_state=SEED,
        )

    candidates: dict = models if models is not None else get_candidate_models()
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

    rows: list[dict] = []

    # --- MLflow parent run setup ---
    if log_to_mlflow:
        uri = tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        parent_run = mlflow.start_run(run_name="leaderboard")
        parent_run_id = parent_run.info.run_id
    else:
        parent_run_id = None

    try:
        for name, estimator in candidates.items():
            pipe = build_model_pipeline(estimator)

            t0 = time.perf_counter()
            cv_results = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv_splitter,
                scoring={k: v for k, v in zip(_SCORERS.keys(), _SCORERS.values())},
                return_train_score=False,
                n_jobs=1,  # avoid nested parallelism conflicts
            )
            elapsed = time.perf_counter() - t0

            pr_auc_scores  = cv_results["test_pr_auc"]
            roc_auc_scores = cv_results["test_roc_auc"]
            # neg_brier_score → flip sign so 0 is perfect, 1 is worst
            brier_scores = -cv_results["test_brier"]

            row = {
                "model":          name,
                "pr_auc_mean":    float(np.mean(pr_auc_scores)),
                "pr_auc_std":     float(np.std(pr_auc_scores)),
                "roc_auc_mean":   float(np.mean(roc_auc_scores)),
                "roc_auc_std":    float(np.std(roc_auc_scores)),
                "brier_mean":     float(np.mean(brier_scores)),
                "brier_std":      float(np.std(brier_scores)),
                "fit_time_mean":  float(elapsed / cv),
            }
            rows.append(row)

            if log_to_mlflow:
                with mlflow.start_run(
                    run_name=name,
                    nested=True,
                    parent_run_id=parent_run_id,
                ):
                    mlflow.log_param("model_name", name)
                    mlflow.log_param("cv_folds", cv)
                    mlflow.log_param("sample_frac", sample_frac)
                    mlflow.log_metric("pr_auc_mean",  row["pr_auc_mean"])
                    mlflow.log_metric("pr_auc_std",   row["pr_auc_std"])
                    mlflow.log_metric("roc_auc_mean", row["roc_auc_mean"])
                    mlflow.log_metric("roc_auc_std",  row["roc_auc_std"])
                    mlflow.log_metric("brier_mean",   row["brier_mean"])
                    mlflow.log_metric("brier_std",    row["brier_std"])
                    mlflow.log_metric("fit_time_mean_s", row["fit_time_mean"])

    finally:
        if log_to_mlflow:
            mlflow.end_run()  # close parent run

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values("pr_auc_mean", ascending=False)
        .reset_index(drop=True)
    )

    # Persist leaderboard artifact
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    leaderboard_path = reports_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    if log_to_mlflow:
        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_id=parent_run_id):
            mlflow.log_artifact(str(leaderboard_path), artifact_path="leaderboard")
        # log leaderboard metadata as JSON on parent for easy API access
        with mlflow.start_run(run_id=parent_run_id):
            mlflow.log_dict(
                json.loads(leaderboard.to_json(orient="records")),
                "leaderboard/summary.json",
            )

    return leaderboard
