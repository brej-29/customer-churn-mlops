"""Model registration with a champion/challenger promotion gate.

The core insight: legacy code set the "champion" alias unconditionally on every
run, so a worse model could silently displace a better one.  Here, a new version
only becomes champion if it strictly beats the current champion's held-out
PR-AUC (plus an optional minimum-improvement margin).  A model that fails the
gate is registered as a "challenger" and the champion alias is left untouched.

Note on repeated gating against the same test set
--------------------------------------------------
The test PR-AUC used as the gate metric was computed ONCE when build_final_model
ran.  If you call register_champion() repeatedly with the same test split (e.g.,
after every re-training sweep), the champion is selected by the metric that
performed best on that specific held-out sample.  Over many evaluations this
erodes the independence of the test set.  In production, gate on a fresh
holdout or a recent-data window that was not used during training.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import mlflow
import mlflow.exceptions

from churn.config import settings
from churn.evaluate import build_final_model  # noqa: F401  (re-exported via register_champion)

# ---------------------------------------------------------------------------
# Pure gate function — independently unit-testable, no I/O
# ---------------------------------------------------------------------------


def should_promote(
    candidate_metric: float,
    champion_metric: Optional[float],
    min_improvement: float = 0.0,
) -> bool:
    """Decide whether a candidate model should become the new champion.

    The metric is PR-AUC (higher is better).

    Parameters
    ----------
    candidate_metric : float
        PR-AUC of the candidate model on the held-out test set.
    champion_metric : float | None
        PR-AUC of the current champion, read from its MLflow version tag.
        ``None`` means no champion exists yet (first registration) → promote.
    min_improvement : float
        Minimum required improvement over the champion to promote.  Defaults
        to 0.0 (any improvement promotes).  A small positive value (e.g. 0.005)
        avoids promoting on noise when the candidate barely beats the champion.

    Returns
    -------
    bool
        True → assign "champion" alias to the candidate.
        False → assign "challenger" alias; current champion is unchanged.
    """
    if champion_metric is None:
        return True  # first registration: always promote
    return candidate_metric > champion_metric + min_improvement


# ---------------------------------------------------------------------------
# Registration + gate
# ---------------------------------------------------------------------------


def register_with_gate(
    model_uri: str,
    candidate_metric: float,
    model_name: str,
    threshold: float,
    calibration_method: str,
    min_improvement: float = 0.0,
    tracking_uri: Optional[str] = None,
) -> dict:
    """Register a logged model under *model_name* and apply the champion gate.

    Steps
    -----
    1. Register the model at *model_uri* → new version number.
    2. Tag the new version with test_pr_auc, threshold, calibration_method,
       and a UTC timestamp (registered_at).
    3. Read the current "champion" alias (if any) and its test_pr_auc tag.
    4. Call should_promote() to decide the alias.
       - Promoted → set "champion" alias to new version.
       - Not promoted → set "challenger" alias to new version; champion stays.
    5. Return a dict describing the decision.

    Parameters
    ----------
    model_uri : str
        URI of the logged model, e.g. ``"runs:/<run_id>/final_model"``.
    candidate_metric : float
        Test-set PR-AUC of the candidate.
    model_name : str
        Registered model name in the MLflow Model Registry.
    threshold : float
        Decision threshold stored as a version tag for traceability.
    calibration_method : str
        "isotonic" or "uncalibrated", stored as a version tag.
    min_improvement : float
        Forwarded to should_promote().
    tracking_uri : str | None
        Override MLflow tracking URI (default: settings.mlflow_tracking_uri).

    Returns
    -------
    dict with keys:
        model_name, new_version, alias, promoted, candidate_metric,
        champion_metric, is_first_registration, min_improvement
    """
    uri = tracking_uri or settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient()

    # ── 1. Register new version ───────────────────────────────────────────
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    new_version = str(mv.version)

    # ── 2. Tag new version ────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).isoformat()
    client.set_model_version_tag(model_name, new_version, "test_pr_auc", str(candidate_metric))
    client.set_model_version_tag(model_name, new_version, "threshold", str(threshold))
    client.set_model_version_tag(model_name, new_version, "calibration_method", calibration_method)
    client.set_model_version_tag(model_name, new_version, "registered_at", timestamp)

    # ── 3. Read current champion metric ───────────────────────────────────
    champion_metric: Optional[float] = None
    is_first = False
    try:
        champ_mv = client.get_model_version_by_alias(model_name, "champion")
        tag_val = champ_mv.tags.get("test_pr_auc")
        champion_metric = float(tag_val) if tag_val is not None else None
    except mlflow.exceptions.MlflowException:
        # No "champion" alias exists yet — this is the first registration.
        is_first = True

    # ── 4. Gate decision ──────────────────────────────────────────────────
    promoted = should_promote(candidate_metric, champion_metric, min_improvement)
    alias = "champion" if promoted else "challenger"
    client.set_registered_model_alias(model_name, alias, new_version)

    # ── 5. Print summary ──────────────────────────────────────────────────
    print(f"\n=== Model Registration: {model_name} v{new_version} ===")
    print(f"Candidate PR-AUC : {candidate_metric:.6f}")
    if champion_metric is not None:
        delta = candidate_metric - champion_metric
        print(f"Champion PR-AUC  : {champion_metric:.6f}  (delta {delta:+.6f}, min_improvement={min_improvement})")
    else:
        print("Champion PR-AUC  : None  (first registration)")
    print(f"Decision         : {alias.upper()}")

    return {
        "model_name": model_name,
        "new_version": new_version,
        "alias": alias,
        "promoted": promoted,
        "candidate_metric": candidate_metric,
        "champion_metric": champion_metric,
        "is_first_registration": is_first,
        "min_improvement": min_improvement,
    }


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def register_champion(
    model_name: str = "customer-churn-xgboost",
    min_improvement: float = 0.0,
    tracking_uri: Optional[str] = None,
    **build_kwargs,
) -> dict:
    """Build, log, and conditionally register the final churn model.

    Calls build_final_model() with log_to_mlflow=True (forced) to ensure the
    model is logged and a model_uri is available, then delegates to
    register_with_gate() for the champion/challenger decision.

    Parameters
    ----------
    model_name : str
        Registered model name in the MLflow Model Registry.
    min_improvement : float
        Minimum PR-AUC improvement required to unseat the current champion.
    tracking_uri : str | None
        MLflow tracking URI override.
    **build_kwargs
        Extra keyword arguments forwarded to build_final_model() (e.g., cv=5,
        fn_cost=5.0).  log_to_mlflow is always forced True.

    Returns
    -------
    dict — same structure as register_with_gate().
    """
    uri = tracking_uri or settings.mlflow_tracking_uri
    # Always log to MLflow so the model artifact exists for registration.
    build_kwargs.pop("log_to_mlflow", None)
    result = build_final_model(log_to_mlflow=True, tracking_uri=uri, **build_kwargs)

    if result.model_uri is None:
        raise RuntimeError(
            "build_final_model did not populate model_uri — "
            "check that log_to_mlflow=True is honoured."
        )

    return register_with_gate(
        model_uri=result.model_uri,
        candidate_metric=result.test_metrics["pr_auc"],
        model_name=model_name,
        threshold=result.threshold,
        calibration_method=result.calibration_method,
        min_improvement=min_improvement,
        tracking_uri=uri,
    )
