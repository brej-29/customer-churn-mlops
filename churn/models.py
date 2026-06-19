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
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.config import settings
from churn.data import get_splits
from churn.features import CT_CATEGORICAL, build_preprocessor

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


# ---------------------------------------------------------------------------
# Imbalance experiment
# ---------------------------------------------------------------------------

# Ordered from simplest to most complex — used for tiebreaking in recommendation.
_IMBALANCE_STRATEGIES: list[str] = ["none", "scale_pos_weight", "smotenc"]

# Note: "smotenc_tomek" (SMOTETomek with SMOTENC) is intentionally omitted.
# SMOTETomek's TomekLinks component receives the SMOTENC output, which is a
# mixed-type DataFrame (string categoricals). TomekLinks uses NearestNeighbors
# and requires a numeric array; it fails on string columns. To keep the pipeline
# correct and readable, the Tomek undersampling is excluded here.


def build_imbalance_pipeline(strategy: str, spw: float = 1.0) -> Pipeline | ImbPipeline:
    """Return a full end-to-end pipeline for the given imbalance strategy.

    All strategies use XGBClassifier with the SAME hyperparameters as the
    leaderboard baseline (eval_metric='logloss', random_state=SEED, n_jobs=-1).
    The only variable is how class imbalance is addressed.

    Parameters
    ----------
    strategy : {"none", "scale_pos_weight", "smotenc"}
        none
            Plain preprocessor + XGBClassifier. Reference baseline.
        scale_pos_weight
            Plain preprocessor + XGBClassifier(..., scale_pos_weight=spw).
            Stratified K-fold preserves the class ratio in every fold, so a
            single train-level spw is representative across folds.
        smotenc
            imblearn Pipeline: [fe → SMOTENC → ct → XGBClassifier].
            SMOTENC is categorical-aware: synthetic minority samples for the
            categorical columns are drawn from the observed category set, not
            interpolated numerically. The sampler fires only during fit (on
            each fold's training portion); the validation fold is always
            un-resampled, so CV scores reflect real held-out performance.
    spw : float
        Positive-class weight = #negatives / #positives on y_train (≈2.77).
        Ignored by "none" and "smotenc".
    """
    def _xgb() -> XGBClassifier:
        return XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1)

    if strategy == "none":
        return build_model_pipeline(_xgb())

    if strategy == "scale_pos_weight":
        return build_model_pipeline(
            XGBClassifier(
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1,
                scale_pos_weight=spw,
            )
        )

    if strategy == "smotenc":
        # Pull the two inner steps out of a fresh preprocessor so each
        # pipeline call gets independent, unfitted transformer objects.
        prep = build_preprocessor()
        fe_step = prep.named_steps["fe"]
        ct_step = prep.named_steps["ct"]
        # CT_CATEGORICAL lists the 17 columns that are categorical in the
        # post-FE frame (16 originals + tenure_bucket). SMOTENC draws
        # synthetic values for those columns from observed category sets;
        # the 6 numeric columns are interpolated normally.
        smotenc = SMOTENC(categorical_features=CT_CATEGORICAL, random_state=SEED)
        return ImbPipeline([
            ("fe", fe_step),
            ("smotenc", smotenc),
            ("ct", ct_step),
            ("model", _xgb()),
        ])

    raise ValueError(
        f"Unknown strategy {strategy!r}. Choose from: {_IMBALANCE_STRATEGIES}"
    )


def run_imbalance_experiment(
    cv: int = 5,
    sample_frac: Optional[float] = None,
    log_to_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "churn-imbalance",
    strategies: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Cross-validate XGBoost under different imbalance-handling strategies.

    Resampling occurs INSIDE each CV fold (imblearn Pipeline applies the
    sampler only on each fold's training portion). Validation-fold metrics
    reflect un-resampled data throughout.

    Parameters
    ----------
    cv : int
        Number of StratifiedKFold folds.
    sample_frac : float | None
        Stratified subsample fraction for fast testing (None = full train set).
    log_to_mlflow : bool
        Whether to log runs to MLflow.
    tracking_uri : str | None
        Override tracking URI (defaults to settings.mlflow_tracking_uri).
    experiment_name : str
        MLflow experiment name.
    strategies : list[str] | None
        Subset of _IMBALANCE_STRATEGIES to run (None = all).

    Returns
    -------
    DataFrame sorted by pr_auc_mean descending, one row per strategy.
    Also writes reports/imbalance_experiment.csv.
    """
    X_train, _, y_train, _ = get_splits()

    if sample_frac is not None:
        from sklearn.model_selection import train_test_split as _tts

        X_train, _, y_train, _ = _tts(
            X_train, y_train,
            train_size=sample_frac,
            stratify=y_train,
            random_state=SEED,
        )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = float(n_neg / n_pos)

    run_strategies = strategies if strategies is not None else _IMBALANCE_STRATEGIES
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

    rows: list[dict] = []

    if log_to_mlflow:
        uri = tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        parent_run = mlflow.start_run(run_name="imbalance-experiment")
        parent_run_id = parent_run.info.run_id
    else:
        parent_run_id = None

    try:
        for strategy in run_strategies:
            pipe = build_imbalance_pipeline(strategy, spw=spw)

            t0 = time.perf_counter()
            cv_results = cross_validate(
                pipe,
                X_train,
                y_train,
                cv=cv_splitter,
                scoring={k: v for k, v in _SCORERS.items()},
                return_train_score=False,
                n_jobs=1,
            )
            elapsed = time.perf_counter() - t0

            row = {
                "strategy": strategy,
                "pr_auc_mean": float(np.mean(cv_results["test_pr_auc"])),
                "pr_auc_std": float(np.std(cv_results["test_pr_auc"])),
                "roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
                "roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
                "brier_mean": float(np.mean(-cv_results["test_brier"])),
                "brier_std": float(np.std(-cv_results["test_brier"])),
                "fit_time_mean": float(elapsed / cv),
            }
            rows.append(row)

            if log_to_mlflow:
                with mlflow.start_run(
                    run_name=strategy,
                    nested=True,
                    parent_run_id=parent_run_id,
                ):
                    mlflow.log_param("strategy", strategy)
                    mlflow.log_param("spw", round(spw, 4))
                    mlflow.log_param("cv_folds", cv)
                    mlflow.log_param("sample_frac", sample_frac)
                    mlflow.log_metric("pr_auc_mean", row["pr_auc_mean"])
                    mlflow.log_metric("pr_auc_std", row["pr_auc_std"])
                    mlflow.log_metric("roc_auc_mean", row["roc_auc_mean"])
                    mlflow.log_metric("roc_auc_std", row["roc_auc_std"])
                    mlflow.log_metric("brier_mean", row["brier_mean"])
                    mlflow.log_metric("brier_std", row["brier_std"])

    finally:
        if log_to_mlflow:
            mlflow.end_run()

    result = (
        pd.DataFrame(rows)
        .sort_values("pr_auc_mean", ascending=False)
        .reset_index(drop=True)
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    result_path = reports_dir / "imbalance_experiment.csv"
    result.to_csv(result_path, index=False)

    # --- Recommendation ---
    # Rule: pick best PR-AUC. If any simpler strategy is within 1 std of the
    # best, prefer it if its Brier is no worse (simpler = less preprocessing,
    # less variance at serving time).
    best_pr = result["pr_auc_mean"].max()
    best_pr_std = result.loc[result["pr_auc_mean"].idxmax(), "pr_auc_std"]
    simplicity = {s: i for i, s in enumerate(_IMBALANCE_STRATEGIES)}

    candidates = result[result["pr_auc_mean"] >= best_pr - best_pr_std].copy()
    candidates["_rank"] = candidates["strategy"].map(
        lambda s: (candidates["brier_mean"].min() - candidates.loc[
            candidates["strategy"] == s, "brier_mean"
        ].values[0], -simplicity.get(s, 99))
    )
    # Sort: lower Brier first, then simpler strategy
    candidates = candidates.sort_values(
        ["brier_mean", "strategy"],
        key=lambda col: col if col.name == "brier_mean" else col.map(simplicity),
    )
    recommended = candidates.iloc[0]["strategy"]
    rec_row = candidates.iloc[0]
    best_row = result.loc[result["pr_auc_mean"].idxmax()]

    if recommended == best_row["strategy"]:
        rationale = (
            f"'{recommended}' has the best PR-AUC "
            f"({rec_row['pr_auc_mean']:.4f} ± {rec_row['pr_auc_std']:.4f})."
        )
    else:
        pr_gap = best_row["pr_auc_mean"] - rec_row["pr_auc_mean"]
        rationale = (
            f"PR-AUC gap vs best ({pr_gap:.4f}) is within 1 std "
            f"({best_pr_std:.4f}); '{recommended}' chosen for better/equal "
            f"calibration (Brier {rec_row['brier_mean']:.4f} vs "
            f"{best_row['brier_mean']:.4f}) and lower complexity."
        )

    print("\n=== Imbalance Experiment Results ===")
    print(result.to_string(index=False))
    print(f"\nRecommended strategy : {recommended}")
    print(f"Rationale            : {rationale}")
    print(f"scale_pos_weight used: {spw:.4f}  ({n_neg} neg / {n_pos} pos)")

    if log_to_mlflow:
        uri = tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_id=parent_run_id):
            mlflow.log_artifact(str(result_path), artifact_path="imbalance")
            mlflow.log_dict(
                json.loads(result.to_json(orient="records")),
                "imbalance/summary.json",
            )
            mlflow.log_param("recommended_strategy", recommended)

    return result
