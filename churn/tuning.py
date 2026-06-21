"""XGBoost hyperparameter tuning with Optuna."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from churn.config import settings
from churn.data import get_splits
from churn.models import SEED, build_model_pipeline

# ---------------------------------------------------------------------------
# Fixed XGBoost parameters (not tuned — identical to leaderboard baseline)
# ---------------------------------------------------------------------------

_XGB_FIXED: dict = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",   # histogram-based splits: fast and deterministic
    "random_state": SEED,
    "n_jobs": -1,
}

# Tuned-parameter names used by tests to validate output completeness.
_TUNED_PARAM_KEYS: list[str] = [
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "gamma",
    "reg_alpha",
    "reg_lambda",
]


# ---------------------------------------------------------------------------
# Committed search space
# ---------------------------------------------------------------------------


def suggest_xgb_params(trial: optuna.Trial) -> dict:
    """Sample XGBoost hyperparameters from the committed search space.

    Search-space rationale
    ----------------------
    n_estimators [100, 1000]
        Number of boosting rounds. More rounds can capture finer patterns but
        risk overfitting; early stopping (Step 6) will refine the final count.
        A wide range lets TPE explore the lr / n_estimators tradeoff freely.

    learning_rate [0.01, 0.3]  log-uniform
        Step size per round. Log-uniform sampling puts equal mass on each
        order of magnitude (0.01–0.03, 0.03–0.1, 0.1–0.3), matching the
        known sensitivity profile: low lr + many rounds generalises well but
        the useful range spans two decades.

    max_depth [3, 10]
        Individual tree depth. Shallow trees (3–6) are well-regularised on
        tabular data; upper bound of 10 allows TPE to explore if signal
        warrants it. Works in tandem with min_child_weight.

    min_child_weight [1, 10]
        Minimum Hessian weight in a leaf. Higher values prevent splits with
        sparse support — effectively a per-leaf regularizer that complements
        max_depth for controlling overfitting on the minority churn class.

    subsample [0.6, 1.0]
        Row subsampling fraction per round (stochastic gradient boosting).
        Introduces variance reduction analogous to bagging; 0.6 floor avoids
        discarding too much signal from the 5634-row training set.

    colsample_bytree [0.6, 1.0]
        Column subsampling per tree. Decorrelates base learners and reduces
        feature-selection bias; 0.6 floor keeps most of the 54 features in
        contention each round.

    gamma [0.0, 5.0]
        Minimum loss-reduction required to make a split. A pure structural
        regularizer: 0 disables it; higher values enforce conservative splits
        as an independent lever from max_depth.

    reg_alpha [1e-3, 10.0]  log-uniform
        L1 regularisation on leaf weights. Drives sparsity; log-uniform
        prior covers several orders of magnitude around the XGBoost default
        of 0.

    reg_lambda [1e-3, 10.0]  log-uniform
        L2 regularisation on leaf weights. XGBoost default is 1.0; log-
        uniform prior lets TPE explore both stronger and weaker regularisation
        within the same framework as reg_alpha.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_importances(study: optuna.Study) -> dict:
    """Return param importances; fall back to MDI if fANOVA fails."""
    try:
        return optuna.importance.get_param_importances(study)
    except RuntimeError:
        # fANOVA raises RuntimeError on zero variance (e.g., very few trials
        # or all-identical objective values). MDI is more robust.
        return optuna.importance.get_param_importances(
            study,
            evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator(),
        )


def _save_plots(study: optuna.Study, reports_dir: Path) -> list[Path]:
    """Save optimization-history and param-importance plots. Returns saved paths."""
    saved: list[Path] = []
    try:
        # Import matplotlib inside the function so the Agg backend can be set
        # before pyplot is loaded, avoiding backend conflicts with callers.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import optuna.visualization.matplotlib as opt_mpl

        history_path = reports_dir / "xgb_optimization_history.png"
        ax = opt_mpl.plot_optimization_history(study)
        ax.get_figure().savefig(history_path, bbox_inches="tight", dpi=120)
        plt.close("all")
        saved.append(history_path)

        importance_path = reports_dir / "xgb_param_importances.png"
        ax2 = opt_mpl.plot_param_importances(study)
        ax2.get_figure().savefig(importance_path, bbox_inches="tight", dpi=120)
        plt.close("all")
        saved.append(importance_path)
    except Exception:
        # Plots are supplementary; never let them abort a tuning run.
        pass
    return saved


# ---------------------------------------------------------------------------
# Tuning entry point
# ---------------------------------------------------------------------------


def tune_xgboost(
    n_trials: int = 60,
    cv: int = 5,
    sample_frac: Optional[float] = None,
    log_to_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "churn-tuning",
    params_out: str | Path = "reports/best_xgb_params.json",
    timeout: Optional[float] = None,
    reports_dir: Optional[Path] = None,
) -> optuna.Study:
    """Tune XGBoost hyperparameters with Optuna on the TRAIN split.

    Strategy is fixed to 'none' (no resampling, no scale_pos_weight) as
    selected in Step 4.  n_estimators is a tuned hyperparameter; early
    stopping is applied at final-fit time in Step 6 where it does not need
    to compose with a cross-validating preprocessor pipeline.

    The returned study's best_value is an optimistic (CV-selected) estimate
    of PR-AUC.  The untouched test split evaluated in Step 6 is the honest
    guard against the study having overfit the CV folds.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials.
    cv : int
        Stratified K-fold folds. Matches the leaderboard CV so best_value
        is directly comparable to the untuned baseline (0.6216).
    sample_frac : float | None
        Stratified subsample fraction for fast testing (None = full train).
    log_to_mlflow : bool
        Whether to log to MLflow.
    tracking_uri : str | None
        Override MLflow tracking URI.
    experiment_name : str
        MLflow experiment name.
    params_out : str | Path
        JSON output path for best params (tuned + fixed combined).
    timeout : float | None
        Stop after this many seconds even if n_trials not reached.

    Returns
    -------
    optuna.Study
        Completed study with .best_params and .best_value populated.
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

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_xgb_params(trial)
        pipe = build_model_pipeline(XGBClassifier(**params, **_XGB_FIXED))
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring="average_precision",
            n_jobs=1,  # sequential for reproducibility under fixed seeds
        )
        return float(np.mean(scores["test_score"]))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)

    # --- Persist best params (tuned + fixed combined) ---
    params_path = Path(params_out)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    best_combined = {**study.best_params, **_XGB_FIXED}
    params_path.write_text(json.dumps(best_combined, indent=2))

    # --- Compute importances ---
    _reports = Path(reports_dir) if reports_dir is not None else Path("reports")
    _reports.mkdir(exist_ok=True)

    importances = _compute_importances(study)

    importances_json = _reports / "xgb_param_importances.json"
    importances_json.write_text(
        json.dumps({k: float(v) for k, v in importances.items()}, indent=2)
    )

    # --- Save plots (non-critical) ---
    plot_paths = _save_plots(study, _reports)

    # --- Print summary ---
    print("\n=== XGBoost Tuning Results ===")
    print(f"Trials run        : {len(study.trials)}")
    print(f"Best CV PR-AUC    : {study.best_value:.6f}")
    print("Untuned baseline  : 0.621568  (leaderboard Step 3)")
    print(f"Lift              : {study.best_value - 0.621568:+.6f}")
    print("\nBest hyperparameters:")
    for k in _TUNED_PARAM_KEYS:
        print(f"  {k:22s} = {study.best_params[k]}")
    print("\nParameter importances (fANOVA):")
    for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:22s} : {v:.4f}")

    # --- MLflow ---
    if log_to_mlflow:
        uri = tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="xgb-tuning"):
            # Search-space metadata
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv_folds", cv)
            mlflow.log_param("sample_frac", sample_frac)
            mlflow.log_param("strategy", "none")
            mlflow.log_param(
                "search_space",
                "n_estimators[100-1000], lr[0.01-0.3 log], max_depth[3-10], "
                "min_child_weight[1-10], subsample[0.6-1.0], "
                "colsample_bytree[0.6-1.0], gamma[0.0-5.0], "
                "reg_alpha[1e-3-10 log], reg_lambda[1e-3-10 log]",
            )
            # Best result
            mlflow.log_metric("best_cv_pr_auc", study.best_value)
            # Best hyperparameters
            for k, v in study.best_params.items():
                mlflow.log_param(f"best_{k}", v)
            # Importances as metrics
            for k, v in importances.items():
                mlflow.log_metric(f"importance_{k}", float(v))
            # Artifacts
            mlflow.log_artifact(str(params_path), artifact_path="params")
            mlflow.log_artifact(str(importances_json), artifact_path="params")
            for p in plot_paths:
                mlflow.log_artifact(str(p), artifact_path="plots")

    return study
