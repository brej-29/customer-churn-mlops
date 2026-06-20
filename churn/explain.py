"""SHAP-based explainability for the tuned XGBoost model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from churn.config import settings
from churn.data import get_splits
from churn.features import CT_CATEGORICAL
from churn.models import SEED, build_model_pipeline

_DEFAULT_PARAMS_PATH: Path = Path("reports/best_xgb_params.json")


def _get_feature_names(pipeline) -> list[str]:
    """Return the 54 transformed feature names from the fitted pipeline's ColumnTransformer."""
    ct = pipeline.named_steps["preprocessor"].named_steps["ct"]
    return list(ct.get_feature_names_out())


def _parse_original_name(fname: str) -> str:
    """Map a transformed feature name back to its original column name.

    Numeric features come as "num__<col>" → "<col>".
    OHE features come as "cat__<col>_<value>" → "<col>".
    CT_CATEGORICAL is sorted longest-first so that "tenure_bucket" beats any
    shorter prefix that might accidentally match.
    """
    if fname.startswith("num__"):
        return fname[5:]
    if fname.startswith("cat__"):
        part = fname[5:]
        for col in sorted(CT_CATEGORICAL, key=len, reverse=True):
            if part.startswith(col + "_") or part == col:
                return col
    return fname


def _aggregate_to_original(
    mean_abs_shap: np.ndarray, feature_names: list[str]
) -> pd.Series:
    """Sum mean|SHAP| values of OHE columns back to their original column names."""
    series = pd.Series(mean_abs_shap, index=feature_names)
    grouped = series.groupby(series.index.map(_parse_original_name)).sum()
    return grouped.sort_values(ascending=False)


def _mlflow_key(s: str) -> str:
    """Sanitize a string to a valid MLflow metric/param key."""
    return re.sub(r"[^a-zA-Z0-9_.\-/]", "_", s)[:250]


def compute_shap(
    sample_size: int = 1000,
    log_to_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "churn-explain",
    params_path: str | Path = _DEFAULT_PARAMS_PATH,
) -> dict:
    """Compute SHAP values for the tuned XGBoost model on a stratified TRAIN sample.

    Uses shap.TreeExplainer on the uncalibrated tuned XGBClassifier step.
    Calibration (isotonic) is a monotonic post-transform: it rescales model
    scores without reordering them, so SHAP attributions on the log-odds output
    of the uncalibrated model fully describe which features drive each prediction.
    TreeExplainer cannot read CalibratedClassifierCV directly.

    Parameters
    ----------
    sample_size : int
        Maximum number of TRAIN rows to use for SHAP computation. A stratified
        subsample is taken when X_train has more rows than this.
    log_to_mlflow : bool
        Whether to log plots and the importance table to MLflow.
    tracking_uri : str | None
        Override MLflow tracking URI.
    experiment_name : str
        MLflow experiment name.
    params_path : str | Path
        Path to the JSON of tuned XGBoost params (from Step 5).

    Returns
    -------
    dict with keys:
        shap_values    : ndarray of shape (n_sample, 54) — log-odds SHAP values
        feature_names  : list[str] of length 54 — transformed column names
        importance_df  : DataFrame sorted by mean_abs_shap descending (54 rows)
        importance_agg : Series sorted descending — importance by original feature
    """
    X_train, _, y_train, _ = get_splits()

    # Load tuned params and fit UNCALIBRATED pipeline on full TRAIN.
    with open(Path(params_path)) as f:
        all_params = json.load(f)

    tuned_pipe = build_model_pipeline(XGBClassifier(**all_params))
    tuned_pipe.fit(X_train, y_train)

    # Stratified sample (up to sample_size rows) from X_train.
    n = min(sample_size, len(X_train))
    if n < len(X_train):
        X_sample, _, _, _ = train_test_split(
            X_train, y_train,
            train_size=n,
            stratify=y_train,
            random_state=SEED,
        )
    else:
        X_sample = X_train.copy()

    # Transform sample to the 54-column preprocessed matrix.
    preprocessor = tuned_pipe.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_sample)

    feature_names = _get_feature_names(tuned_pipe)

    # SHAP TreeExplainer on the fitted XGBClassifier.
    # shap_values is (n_sample, 54) for binary classification log-odds output.
    xgb_model = tuned_pipe.named_steps["model"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed)

    # Some shap versions return a list of arrays for binary classifiers.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Global importance: mean absolute SHAP per feature.
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    importance_agg = _aggregate_to_original(mean_abs, feature_names)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Plots — non-critical; wrapped in try/except.
    plot_paths: list[Path] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Beeswarm summary plot (shows direction of effect, top 20 features).
        shap.summary_plot(
            shap_values, X_transformed,
            feature_names=feature_names,
            max_display=20,
            show=False,
        )
        beeswarm_path = reports_dir / "shap_beeswarm.png"
        plt.savefig(beeswarm_path, bbox_inches="tight", dpi=120)
        plt.close("all")
        plot_paths.append(beeswarm_path)

        # Bar plot — mean|SHAP| magnitude only (top 20).
        shap.summary_plot(
            shap_values, X_transformed,
            feature_names=feature_names,
            max_display=20,
            plot_type="bar",
            show=False,
        )
        bar_path = reports_dir / "shap_bar.png"
        plt.savefig(bar_path, bbox_inches="tight", dpi=120)
        plt.close("all")
        plot_paths.append(bar_path)

        # Aggregated bar plot — OHE columns collapsed to original feature.
        fig, ax = plt.subplots(figsize=(8, 6))
        importance_agg.head(20).sort_values().plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("SHAP importance — original features (OHE aggregated)")
        agg_path = reports_dir / "shap_bar_aggregated.png"
        fig.savefig(agg_path, bbox_inches="tight", dpi=120)
        plt.close("all")
        plot_paths.append(agg_path)

    except Exception:
        pass

    # Save importance table.
    importance_csv = reports_dir / "shap_importance.csv"
    importance_df.to_csv(importance_csv, index=False)

    # Print summary.
    print("\n=== Top 10 SHAP Feature Importances (mean|SHAP|, transformed features) ===")
    print(importance_df.head(10).to_string(index=False))
    print("\n=== Top 10 by Original Feature (OHE aggregated) ===")
    print(importance_agg.head(10).to_string())

    # MLflow logging.
    if log_to_mlflow:
        uri = tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="shap-explain"):
            mlflow.log_param("sample_size", len(X_sample))
            mlflow.log_param("n_features", len(feature_names))
            for rank, (_, row) in enumerate(importance_df.head(10).iterrows(), start=1):
                key = _mlflow_key(f"shap_rank{rank:02d}_{row['feature']}")
                mlflow.log_metric(key, float(row["mean_abs_shap"]))
            mlflow.log_artifact(str(importance_csv), artifact_path="shap")
            for p in plot_paths:
                mlflow.log_artifact(str(p), artifact_path="shap")

    return {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "importance_df": importance_df,
        "importance_agg": importance_agg,
    }
