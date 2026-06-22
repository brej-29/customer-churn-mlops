"""Final-model calibration, threshold selection, and test-set evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from churn.config import settings
from churn.data import get_splits
from churn.models import SEED, build_model_pipeline

# Path to tuned params produced by Step 5.
_DEFAULT_PARAMS_PATH: Path = Path("reports/best_xgb_params.json")


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclass
class FinalModelResult:
    """Container for all outputs of build_final_model."""

    model: Any                    # fitted final estimator (pipeline or cal. wrapper)
    threshold: float              # chosen decision threshold (cost-optimised)
    calibration_method: str       # "uncalibrated" or "isotonic"
    test_metrics: dict            # all test-set metrics
    uncal_brier_oof: float        # OOF Brier for uncalibrated model
    cal_brier_oof: float          # OOF Brier for isotonic-calibrated model
    threshold_details: dict = field(default_factory=dict)  # cost curve info
    # Set when log_to_mlflow=True; used by registry.py to register the model.
    run_id: Optional[str] = None
    model_uri: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: calibration assessment
# ---------------------------------------------------------------------------


def assess_calibration(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute Brier score and reliability-curve data for a set of probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (0/1).
    proba : array-like of shape (n,)
        Predicted positive-class probabilities.
    n_bins : int
        Number of equal-width bins for the reliability curve.

    Returns
    -------
    dict with keys:
        brier       : float — Brier score (lower = better calibration, range [0,1])
        bin_centers : ndarray — mean predicted probability per non-empty bin
        frac_pos    : ndarray — fraction of positives per non-empty bin
        bin_counts  : ndarray — number of samples per non-empty bin
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    brier = float(brier_score_loss(y_true, proba))

    frac_pos, mean_pred = calibration_curve(
        y_true, proba, n_bins=n_bins, strategy="uniform"
    )

    # Replicate sklearn's binning to get per-bin sample counts aligned with
    # the (frac_pos, mean_pred) arrays (which skip empty bins).
    bins_edges = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    bin_ids = np.searchsorted(bins_edges[1:-1], proba)
    bin_totals = np.bincount(bin_ids, minlength=n_bins)
    bin_counts = bin_totals[bin_totals > 0]

    return {
        "brier": brier,
        "bin_centers": mean_pred,
        "frac_pos": frac_pos,
        "bin_counts": bin_counts,
    }


# ---------------------------------------------------------------------------
# Helper: cost-based threshold selection
# ---------------------------------------------------------------------------


def select_threshold_by_cost(
    y_true: np.ndarray,
    proba: np.ndarray,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    n_thresholds: int = 200,
) -> dict:
    """Select the decision threshold that minimises expected cost.

    Cost model (stated assumption — replace with real business numbers):
        expected_cost = fn_count * fn_cost + fp_count * fp_cost

    A 5:1 ratio (fn_cost=5, fp_cost=1) reflects: a missed churner who leaves
    costs roughly 5× the expense of a wasted retention offer sent to a loyal
    customer. This ratio is a planning assumption, not an empirical estimate.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    proba : array-like
        Predicted positive-class probabilities.
    fn_cost : float
        Cost of a false negative (missed churner).
    fp_cost : float
        Cost of a false positive (unnecessary retention offer).
    n_thresholds : int
        Number of threshold candidates in (0, 1) exclusive.

    Returns
    -------
    dict with keys:
        threshold    : float — cost-minimising threshold
        thresholds   : ndarray — all candidate thresholds
        costs        : ndarray — expected cost at each threshold
        f1_threshold : float — F1-maximising threshold (for comparison)
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    # Sweep thresholds strictly inside (0, 1).
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1]

    costs = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        y_pred = (proba >= t).astype(int)
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        costs[i] = fn * fn_cost + fp * fp_cost

    best_idx = int(np.argmin(costs))
    best_threshold = float(thresholds[best_idx])

    # F1-optimal threshold: maximise harmonic mean of precision and recall.
    f1_scores = np.array(
        [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in thresholds]
    )
    f1_threshold = float(thresholds[int(np.argmax(f1_scores))])

    return {
        "threshold": best_threshold,
        "thresholds": thresholds,
        "costs": costs,
        "f1_threshold": f1_threshold,
    }


# ---------------------------------------------------------------------------
# Plotting helpers (non-critical; all wrapped in try/except)
# ---------------------------------------------------------------------------


def _plot_reliability(
    y_train: np.ndarray,
    oof_uncal: np.ndarray,
    oof_cal: np.ndarray,
    reports_dir: Path,
    n_bins: int = 10,
) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

        for proba, label, color in [
            (oof_uncal, "Uncalibrated", "tab:blue"),
            (oof_cal, "Isotonic", "tab:orange"),
        ]:
            fp_, mp_ = calibration_curve(y_train, proba, n_bins=n_bins, strategy="uniform")
            ax.plot(mp_, fp_, "o-", color=color, label=label)

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Reliability diagram (OOF, TRAIN)")
        ax.legend()
        out = reports_dir / "reliability_plot.png"
        fig.savefig(out, bbox_inches="tight", dpi=120)
        plt.close("all")
        return out
    except Exception:
        return None


def _plot_cost_curve(
    cost_result: dict,
    reports_dir: Path,
    fn_cost: float,
    fp_cost: float,
) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        thresholds = cost_result["thresholds"]
        costs = cost_result["costs"]
        chosen = cost_result["threshold"]
        f1_thr = cost_result["f1_threshold"]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(thresholds, costs, color="tab:blue", label="Expected cost")
        ax.axvline(chosen, color="tab:red", linestyle="--",
                   label=f"Cost-optimal  (t={chosen:.3f})")
        ax.axvline(f1_thr, color="tab:green", linestyle=":",
                   label=f"F1-optimal   (t={f1_thr:.3f})")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(f"Cost  (FN×{fn_cost} + FP×{fp_cost})")
        ax.set_title("Cost vs. threshold (OOF TRAIN probabilities)")
        ax.legend()
        out = reports_dir / "cost_vs_threshold.png"
        fig.savefig(out, bbox_inches="tight", dpi=120)
        plt.close("all")
        return out
    except Exception:
        return None


def _plot_pr_curve(
    y_test: np.ndarray,
    test_proba: np.ndarray,
    threshold: float,
    pr_auc: float,
    reports_dir: Path,
) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        prec, rec, _ = precision_recall_curve(y_test, test_proba)
        # Point on the curve closest to the chosen threshold
        y_pred_t = (test_proba >= threshold).astype(int)
        pt_prec = precision_score(y_test, y_pred_t, zero_division=0)
        pt_rec = recall_score(y_test, y_pred_t, zero_division=0)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(rec, prec, color="tab:blue", label=f"PR curve (AUC={pr_auc:.4f})")
        ax.scatter([pt_rec], [pt_prec], color="tab:red", zorder=5,
                   label=f"Chosen threshold {threshold:.3f}")
        ax.axhline(y_test.mean(), linestyle="--", color="gray",
                   label=f"Baseline (prevalence {y_test.mean():.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curve (TEST set)")
        ax.legend()
        out = reports_dir / "pr_curve.png"
        fig.savefig(out, bbox_inches="tight", dpi=120)
        plt.close("all")
        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def build_final_model(
    cv: int = 5,
    sample_frac: Optional[float] = None,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    log_to_mlflow: bool = True,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "churn-final",
    threshold_out: str | Path = "reports/threshold.json",
    params_path: str | Path = _DEFAULT_PARAMS_PATH,
) -> FinalModelResult:
    """Calibrate, select threshold, fit final model, evaluate ONCE on test set.

    Calibration-method selection and threshold choice are made entirely on
    out-of-fold TRAIN predictions — the test split is never touched until
    the final single evaluation at the end.

    Steps
    -----
    1. Load tuned XGBoost params (from Step 5) and build the tuned pipeline.
    2. Calibration comparison (OOF on TRAIN):
       a. Uncalibrated OOF probabilities via cross_val_predict.
       b. Isotonic-calibrated OOF via cross_val_predict over
          CalibratedClassifierCV (nested CV: outer cv-fold × inner cv-fold).
       c. Choose the model with the lower OOF Brier score.
    3. Threshold selection (OOF of the chosen model):
       Sweep 200 thresholds, minimise expected cost = FN×fn_cost + FP×fp_cost.
       Record the F1-optimal threshold for contrast.  Save to threshold_out.
    4. Final fit: fit the chosen model on the FULL TRAIN set.
    5. Test evaluation (single, final touch of the test split):
       PR-AUC, ROC-AUC, Brier, precision/recall/F1, confusion matrix.
    6. (Optional) MLflow logging with a logged sklearn model.

    Note: best_value from Step 5 (0.6700) is an optimistic CV-selected
    estimate. The test PR-AUC here is the honest generalisation measure.

    Parameters
    ----------
    cv : int
        CV folds for both OOF cross_val_predict and CalibratedClassifierCV.
    sample_frac : float | None
        Subsample TRAIN for fast testing (None = full train set).
    fn_cost, fp_cost : float
        Cost-ratio assumption for threshold selection (default 5:1).
    log_to_mlflow : bool
        Whether to log to MLflow.
    tracking_uri : str | None
        Override MLflow tracking URI.
    experiment_name : str
        MLflow experiment name.
    threshold_out : str | Path
        Path for the JSON file with the chosen threshold + cost ratio.
    params_path : str | Path
        Path to the tuned-params JSON from Step 5.
    """
    # ── 0. Data ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = get_splits()
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)

    if sample_frac is not None:
        from sklearn.model_selection import train_test_split as _tts

        X_train, _, y_train, _ = _tts(
            X_train, y_train,
            train_size=sample_frac,
            stratify=y_train,
            random_state=SEED,
        )
        y_train_arr = np.asarray(y_train)

    # ── 1. Load tuned params and build base pipeline ──────────────────────
    with open(Path(params_path)) as f:
        all_params = json.load(f)

    # XGBClassifier accepts both tuned and fixed params directly.
    tuned_pipe = build_model_pipeline(XGBClassifier(**all_params))

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

    # ── 2a. Uncalibrated OOF ─────────────────────────────────────────────
    print("Computing uncalibrated OOF probabilities...")
    oof_uncal = cross_val_predict(
        clone(tuned_pipe), X_train, y_train,
        cv=cv_splitter, method="predict_proba", n_jobs=1,
    )
    uncal_info = assess_calibration(y_train_arr, oof_uncal[:, 1])

    # ── 2b. Isotonic-calibrated OOF (nested CV) ───────────────────────────
    print("Computing isotonic-calibrated OOF probabilities (nested CV)...")
    cal_wrapper = CalibratedClassifierCV(
        estimator=clone(tuned_pipe), method="isotonic", cv=cv,
    )
    oof_cal = cross_val_predict(
        cal_wrapper, X_train, y_train,
        cv=cv_splitter, method="predict_proba", n_jobs=1,
    )
    cal_info = assess_calibration(y_train_arr, oof_cal[:, 1])

    # ── 2c. Decision: choose the model with lower OOF Brier ──────────────
    use_calibration = cal_info["brier"] < uncal_info["brier"]
    calibration_method = "isotonic" if use_calibration else "uncalibrated"
    chosen_oof_proba = oof_cal[:, 1] if use_calibration else oof_uncal[:, 1]

    print("\nCalibration assessment (OOF Brier, TRAIN):")
    print(f"  Uncalibrated : {uncal_info['brier']:.6f}")
    print(f"  Isotonic     : {cal_info['brier']:.6f}")
    print(f"  Decision     : {calibration_method}")

    # ── 3. Threshold selection on chosen OOF probabilities ───────────────
    cost_result = select_threshold_by_cost(
        y_train_arr, chosen_oof_proba,
        fn_cost=fn_cost, fp_cost=fp_cost,
    )
    threshold = cost_result["threshold"]

    print("\nThreshold selection (5:1 cost, OOF TRAIN):")
    print(f"  Cost-optimal threshold : {threshold:.4f}")
    print(f"  F1-optimal threshold   : {cost_result['f1_threshold']:.4f}")

    # Save threshold + metadata to JSON
    threshold_path = Path(threshold_out)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_payload = {
        "threshold": threshold,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "cost_ratio": f"{fn_cost:.0f}:{fp_cost:.0f}",
        "f1_threshold": cost_result["f1_threshold"],
        "calibration_method": calibration_method,
    }
    threshold_path.write_text(json.dumps(threshold_payload, indent=2))

    # ── 4. Final fit on full TRAIN ────────────────────────────────────────
    print(f"\nFitting final model ({calibration_method}) on full TRAIN set...")
    if use_calibration:
        final_model = CalibratedClassifierCV(
            estimator=clone(tuned_pipe), method="isotonic", cv=cv,
        )
    else:
        final_model = clone(tuned_pipe)

    final_model.fit(X_train, y_train)

    # ── 5. Test evaluation (single, final use of the test split) ─────────
    print("Evaluating on TEST set (first and only time)...")
    test_proba = final_model.predict_proba(X_test)[:, 1]
    y_pred_test = (test_proba >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_test_arr, test_proba))
    roc_auc = float(roc_auc_score(y_test_arr, test_proba))
    brier_test = float(brier_score_loss(y_test_arr, test_proba))
    prec = float(precision_score(y_test_arr, y_pred_test, zero_division=0))
    rec = float(recall_score(y_test_arr, y_pred_test, zero_division=0))
    f1 = float(f1_score(y_test_arr, y_pred_test, zero_division=0))
    cm = confusion_matrix(y_test_arr, y_pred_test).tolist()

    test_metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier_test,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "threshold": threshold,
    }

    # ── 6. Plots ──────────────────────────────────────────────────────────
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    plot_paths: list[Path] = []
    rel_plot = _plot_reliability(y_train_arr, oof_uncal[:, 1], oof_cal[:, 1], reports_dir)
    if rel_plot:
        plot_paths.append(rel_plot)
    cost_plot = _plot_cost_curve(cost_result, reports_dir, fn_cost, fp_cost)
    if cost_plot:
        plot_paths.append(cost_plot)
    pr_plot = _plot_pr_curve(y_test_arr, test_proba, threshold, pr_auc, reports_dir)
    if pr_plot:
        plot_paths.append(pr_plot)

    # ── 7. Print summary ──────────────────────────────────────────────────
    print("\n=== Final Model - Test-Set Results ===")
    print(f"Calibration      : {calibration_method}")
    print(f"Threshold        : {threshold:.4f}  (5:1 cost,  F1-optimal={cost_result['f1_threshold']:.4f})")
    print(f"PR-AUC (test)    : {pr_auc:.6f}  [CV best was 0.670034 -> gap {pr_auc - 0.670034:+.6f}]")
    print(f"ROC-AUC (test)   : {roc_auc:.6f}")
    print(f"Brier (test)     : {brier_test:.6f}")
    print(f"Precision        : {prec:.4f}")
    print(f"Recall           : {rec:.4f}")
    print(f"F1               : {f1:.4f}")
    tn, fp_count, fn_count, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"Confusion matrix : TN={tn}  FP={fp_count}  FN={fn_count}  TP={tp}")

    # ── 8. MLflow ─────────────────────────────────────────────────────────
    run_id_logged: Optional[str] = None
    model_uri_logged: Optional[str] = None

    if log_to_mlflow:
        uri = tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="final-model") as _active_run:
            run_id_logged = _active_run.info.run_id
            # Calibration + threshold
            mlflow.log_param("calibration_method", calibration_method)
            mlflow.log_param("fn_cost", fn_cost)
            mlflow.log_param("fp_cost", fp_cost)
            mlflow.log_param("cost_ratio", f"{fn_cost:.0f}:{fp_cost:.0f}")
            mlflow.log_metric("threshold", threshold)
            mlflow.log_metric("f1_threshold", cost_result["f1_threshold"])
            # Calibration Brier scores (OOF)
            mlflow.log_metric("oof_brier_uncalibrated", uncal_info["brier"])
            mlflow.log_metric("oof_brier_isotonic", cal_info["brier"])
            # Test metrics
            mlflow.log_metric("test_pr_auc", pr_auc)
            mlflow.log_metric("test_roc_auc", roc_auc)
            mlflow.log_metric("test_brier", brier_test)
            mlflow.log_metric("test_precision", prec)
            mlflow.log_metric("test_recall", rec)
            mlflow.log_metric("test_f1", f1)
            mlflow.log_metric("test_tn", tn)
            mlflow.log_metric("test_fp", fp_count)
            mlflow.log_metric("test_fn", fn_count)
            mlflow.log_metric("test_tp", tp)
            # Plot artifacts
            for p in plot_paths:
                mlflow.log_artifact(str(p), artifact_path="plots")
            # Threshold JSON
            mlflow.log_artifact(str(threshold_path), artifact_path="params")
            # Logged model with signature (consumed by Steps 8–9)
            signature = infer_signature(
                X_train, final_model.predict_proba(X_train.head(5))
            )
            # artifact_path= is used (not name=) for compatibility with hosted
            # MLflow backends (e.g. DagsHub) that pre-date the MLflow 3.x
            # LoggedModel API (/api/2.0/mlflow/logged-models/search).  Using
            # name= causes the model artifact to be silently skipped on those
            # servers.  model_info.model_uri is captured so register_model gets
            # the actual storage URI rather than a manually-constructed string.
            model_info = mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="final_model",
                signature=signature,
                input_example=X_train.head(5),
                skops_trusted_types=[
                    "churn.features.ChurnFeatureEngineer",
                    "numpy.dtype",
                    "sklearn.calibration._CalibratedClassifier",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                ],
            )
        model_uri_logged = model_info.model_uri

    return FinalModelResult(
        model=final_model,
        threshold=threshold,
        calibration_method=calibration_method,
        test_metrics=test_metrics,
        uncal_brier_oof=uncal_info["brier"],
        cal_brier_oof=cal_info["brier"],
        threshold_details=cost_result,
        run_id=run_id_logged,
        model_uri=model_uri_logged,
    )
