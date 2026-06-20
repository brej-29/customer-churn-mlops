"""Subgroup fairness analysis for the Telco churn model."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from churn.config import settings
from churn.data import get_splits
from churn.evaluate import build_final_model


def subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    sensitive_series: pd.Series,
) -> pd.DataFrame:
    """Compute per-subgroup fairness metrics for a binary classifier.

    All rates are computed within each group independently. Undefined rates
    (e.g., recall when a group has no positives) are returned as NaN so that
    downstream disparity calculations can simply call .dropna().

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (0/1).
    y_pred : array-like of shape (n,)
        Predicted binary labels at the chosen threshold.
    proba : array-like of shape (n,)
        Predicted positive-class probabilities (for ROC-AUC).
    sensitive_series : pd.Series of shape (n,)
        Sensitive attribute values; one element per sample.

    Returns
    -------
    DataFrame with one row per unique group value and columns:
        group           str  — group label
        n               int  — group size
        positive_rate   float — churn base rate (n_pos / n)
        recall          float — TPR = TP / (TP + FN); NaN if no positives
        fpr             float — FP / (FP + TN); NaN if no negatives
        precision       float — TP / (TP + FP); NaN if no predicted positives
        selection_rate  float — predicted-positive rate
        roc_auc         float — per-group ROC-AUC; NaN if only one class present
        low_confidence  bool  — True if n < 30 (set by caller when desired)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba = np.asarray(proba)
    groups = np.asarray(sensitive_series)

    rows: list[dict] = []
    for group in sorted(np.unique(groups), key=str):
        mask = groups == group
        yt = y_true[mask]
        yp = y_pred[mask]
        pr = proba[mask]

        n = int(mask.sum())
        n_pos = int(yt.sum())
        n_neg = n - n_pos

        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())

        rows.append({
            "group": str(group),
            "n": n,
            "positive_rate": float(n_pos / n) if n > 0 else float("nan"),
            "recall": float(tp / n_pos) if n_pos > 0 else float("nan"),
            "fpr": float(fp / n_neg) if n_neg > 0 else float("nan"),
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan"),
            "selection_rate": float(yp.sum() / n) if n > 0 else float("nan"),
            "roc_auc": (
                float(roc_auc_score(yt, pr))
                if (n_pos > 0 and n_neg > 0)
                else float("nan")
            ),
        })

    return pd.DataFrame(rows)


def run_fairness_analysis(
    sensitive_features: Sequence[str] = (
        "gender", "SeniorCitizen", "Partner", "Dependents"
    ),
    log_to_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "churn-fairness",
    build_final_model_kwargs: Optional[dict] = None,
) -> dict:
    """Evaluate subgroup fairness on the TEST set using the final fitted model.

    Calls build_final_model() (with log_to_mlflow=False) to obtain the fitted
    isotonic-calibrated XGBoost pipeline and the cost-optimal threshold.  The
    test set is used for inference only — consistent with the evaluation protocol
    in Step 6 where the test split is touched exactly once.

    Disparity flags
    ---------------
    recall_gap  = max(group recall) - min(group recall)
    fpr_gap     = max(group FPR) - min(group FPR)
    Groups with n < 30 are flagged as low_confidence in their table.

    Parameters
    ----------
    sensitive_features : sequence of str
        Column names in X_test to use as sensitive attributes.
    log_to_mlflow : bool
        Whether to log per-feature tables and disparity summary to MLflow.
    tracking_uri : str | None
        Override MLflow tracking URI.
    experiment_name : str
        MLflow experiment name.
    build_final_model_kwargs : dict | None
        Extra kwargs forwarded to build_final_model() (e.g., cv=2, sample_frac
        for fast tests). log_to_mlflow is always forced False.

    Returns
    -------
    dict with keys:
        tables      : dict[str, DataFrame] — per-feature subgroup metric tables
        disparities : DataFrame — recall_gap and fpr_gap per sensitive feature
        model_result: FinalModelResult from build_final_model
    """
    kwargs = {**(build_final_model_kwargs or {}), "log_to_mlflow": False}
    result = build_final_model(**kwargs)

    _, X_test, _, y_test = get_splits()
    y_test_arr = np.asarray(y_test)
    proba = result.model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= result.threshold).astype(int)

    tables: dict[str, pd.DataFrame] = {}
    disparity_rows: list[dict] = []

    for feat in sensitive_features:
        if feat not in X_test.columns:
            continue

        tbl = subgroup_metrics(y_test_arr, y_pred, proba, X_test[feat])
        tbl["low_confidence"] = tbl["n"] < 30
        tables[feat] = tbl

        recall_vals = tbl["recall"].dropna()
        fpr_vals = tbl["fpr"].dropna()

        recall_gap = (
            float(recall_vals.max() - recall_vals.min())
            if len(recall_vals) >= 2
            else float("nan")
        )
        fpr_gap = (
            float(fpr_vals.max() - fpr_vals.min())
            if len(fpr_vals) >= 2
            else float("nan")
        )

        disparity_rows.append({
            "feature": feat,
            "n_groups": len(tbl),
            "recall_min": float(recall_vals.min()) if len(recall_vals) else float("nan"),
            "recall_max": float(recall_vals.max()) if len(recall_vals) else float("nan"),
            "recall_gap": recall_gap,
            "fpr_min": float(fpr_vals.min()) if len(fpr_vals) else float("nan"),
            "fpr_max": float(fpr_vals.max()) if len(fpr_vals) else float("nan"),
            "fpr_gap": fpr_gap,
        })

        print(f"\n=== Subgroup Analysis: {feat} ===")
        print(tbl.to_string(index=False))
        _rg = f"{recall_gap:.4f}" if not np.isnan(recall_gap) else "N/A"
        _fg = f"{fpr_gap:.4f}" if not np.isnan(fpr_gap) else "N/A"
        print(f"  recall_gap : {_rg}")
        print(f"  fpr_gap    : {_fg}")

    disparities = pd.DataFrame(disparity_rows)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    saved_paths: list[Path] = []
    for feat, tbl in tables.items():
        p = reports_dir / f"fairness_{feat}.csv"
        tbl.to_csv(p, index=False)
        saved_paths.append(p)

    disp_path = reports_dir / "fairness_disparities.csv"
    disparities.to_csv(disp_path, index=False)
    saved_paths.append(disp_path)

    if log_to_mlflow:
        uri = tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="fairness-analysis"):
            mlflow.log_param("sensitive_features", ",".join(sensitive_features))
            mlflow.log_param("threshold", result.threshold)
            for _, row in disparities.iterrows():
                feat = row["feature"]
                if not np.isnan(row["recall_gap"]):
                    mlflow.log_metric(f"{feat}_recall_gap", row["recall_gap"])
                if not np.isnan(row["fpr_gap"]):
                    mlflow.log_metric(f"{feat}_fpr_gap", row["fpr_gap"])
            for p in saved_paths:
                mlflow.log_artifact(str(p), artifact_path="fairness")

    return {
        "tables": tables,
        "disparities": disparities,
        "model_result": result,
    }
