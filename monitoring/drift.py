"""Drift detection for the Telco Customer Churn prediction service.

Source of truth for which features to monitor: ``churn.data``
(NUMERIC_FEATURES, CATEGORICAL_FEATURES) — the same 19 columns the
Pandera contract in ``churn.validation.ALLOWED`` constrains.  Monitoring
and the data contract therefore can never diverge on the feature set.

Reference distribution
----------------------
X_train from the canonical ``get_splits()`` call (seed-fixed via
``settings.random_seed``), saved to ``reports/drift_reference.parquet``
by ``build_reference()``.  Re-run after every model retrain so the
baseline tracks the latest training distribution.

Current window
--------------
The ``DRIFT_WINDOW`` most recent prediction inputs from ``logs/predictions.db``
(same SQLite table written by ``api/main.py``).

DRIFT_WINDOW bug (fixed here)
------------------------------
The original ``monitoring/generate_drift_report.py`` had three issues:

1. **Wrong schema** — used ``FEATURE_COLUMNS`` from ``training.preprocess``
   (6 synthetic columns: tenure, monthly_charges, contract_type …) instead
   of the 19 real Telco columns.  Logged payloads use the real schema, so 5
   of 6 synthetic columns were filled with zeros; drift comparison was
   meaningless.

2. **Wrong reference** — reference was generated from synthetic data, not
   from ``X_train``.

3. **Row order** — ``ORDER BY id DESC LIMIT n`` returns rows newest-first;
   the result was reversed-chronological.

This module fixes all three.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from churn.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of recent predictions to include in the current window.
# 500 is the minimum sample that provides stable per-feature test statistics
# while being recent enough to reflect the current serving distribution.
DRIFT_WINDOW: int = int(os.getenv("DRIFT_WINDOW", "500"))

# Retrain when this share of the 19 features has drifted.
# 0.30 = flag for retrain once >= 30 % of features shift, even if Evidently's
# dataset-level threshold (>=50 % by default) hasn't been crossed.
DRIFT_FEATURE_THRESHOLD: float = float(os.getenv("DRIFT_FEATURE_THRESHOLD", "0.3"))

LOG_DB_PATH: str = os.getenv("LOG_DB_PATH", "logs/predictions.db")
REFERENCE_PATH: Path = Path(
    os.getenv("DRIFT_REFERENCE_PATH", "reports/drift_reference.parquet")
)

# The 19 feature columns to monitor — derived from churn.data so that this
# list always matches the Pandera contract in churn.validation.
MONITORED_FEATURES: list[str] = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FeatureDriftResult:
    feature: str
    drifted: bool
    score: Optional[float] = None  # p-value or distance metric (method-dependent)


@dataclass
class DriftCheckResult:
    dataset_drift: bool
    n_drifted_features: int
    n_total_features: int
    drifted_share: float
    per_feature: list[FeatureDriftResult] = field(default_factory=list)
    retrain_recommended: bool = False
    retrain_reason: str = ""


# ---------------------------------------------------------------------------
# Reference distribution
# ---------------------------------------------------------------------------


def build_reference(force_rebuild: bool = False) -> pd.DataFrame:
    """Build or load the training-split reference DataFrame.

    On first call (or when ``force_rebuild=True``) this calls ``get_splits()``
    to obtain X_train, restricts it to ``MONITORED_FEATURES``, and persists the
    result to ``reports/drift_reference.parquet``.  Subsequent calls load from
    disk.  The canonical seed in ``settings.random_seed`` (used by get_splits)
    ensures reproducibility.

    Parameters
    ----------
    force_rebuild:
        Ignore any cached file and rebuild from the training data.
    """
    if not force_rebuild and REFERENCE_PATH.is_file():
        return pd.read_parquet(REFERENCE_PATH)

    from churn.data import get_splits  # noqa: PLC0415

    X_train, _, _, _ = get_splits()
    ref = X_train[MONITORED_FEATURES].copy()

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(REFERENCE_PATH, index=False)
    logger.info("Drift reference saved to %s (%d rows)", REFERENCE_PATH, len(ref))
    return ref


# ---------------------------------------------------------------------------
# Current window — fixed selector
# ---------------------------------------------------------------------------


def get_current_window(
    n: int = DRIFT_WINDOW,
    db_path: str = LOG_DB_PATH,
) -> pd.DataFrame:
    """Return the ``n`` most recent logged prediction inputs in chronological order.

    Reads ``request_payload`` from the ``prediction_logs`` SQLite table (the
    same DB written by ``api/main.py``).

    Normalisation applied:
    * **SeniorCitizen** is logged as int (0/1) by ``PredictRequest.model_dump()``.
      Converted to str "0"/"1" to match the training reference (``clean_telco``
      casts all categoricals to str).
    * Numeric columns are coerced to float64.

    Returns an empty DataFrame with the correct columns when the DB is absent
    or empty.
    """
    empty = pd.DataFrame(columns=MONITORED_FEATURES)
    db_file = Path(db_path)
    if not db_file.is_file():
        return empty

    try:
        conn = sqlite3.connect(str(db_file))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT request_payload
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(n),),
        )
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.Error:
        logger.exception("Failed to read prediction logs for drift window.")
        return empty

    if not rows:
        return empty

    payloads = []
    for row in rows:
        try:
            payload = json.loads(row["request_payload"])
        except Exception:  # noqa: BLE001
            continue
        # Normalise SeniorCitizen: logged as int, reference has str "0"/"1"
        if "SeniorCitizen" in payload:
            payload["SeniorCitizen"] = str(int(payload["SeniorCitizen"]))
        payloads.append(payload)

    if not payloads:
        return empty

    df = pd.DataFrame(payloads)
    # Reverse: we queried newest-first; chronological order is oldest-first
    df = df.iloc[::-1].reset_index(drop=True)

    # Ensure all monitored columns exist
    for col in MONITORED_FEATURES:
        if col not in df.columns:
            df[col] = pd.NA

    # Coerce numeric columns (payload may deliver ints for tenure=0 etc.)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[MONITORED_FEATURES]


# ---------------------------------------------------------------------------
# Evidently result parser (v2 metric schema)
# ---------------------------------------------------------------------------


def _parse_evidently_snapshot(result_dict: dict) -> tuple[bool, int, list[FeatureDriftResult]]:
    """Parse an Evidently 0.7.21 Snapshot JSON dict.

    Evidently 0.7 v2 metrics structure::

        {
          "metrics": [
            {
              "metric_name": "DriftedColumnsCount(drift_share=0.5)",
              "config": {"type": "…:DriftedColumnsCount", "drift_share": 0.5},
              "value": {"count": 3.0, "share": 0.158}
            },
            {
              "metric_name": "ValueDrift(column=tenure,method=K-S p_value,…)",
              "config": {"type": "…:ValueDrift", "column": "tenure",
                         "method": "K-S p_value", "threshold": 0.05},
              "value": 0.312   # p-value; drift when < threshold
            },
            …
          ]
        }

    Returns
    -------
    (dataset_drift, n_drifted, per_feature)
    """
    n_drifted = 0
    dataset_drift = False
    per_feature: list[FeatureDriftResult] = []

    for metric in result_dict.get("metrics", []):
        config = metric.get("config", {})
        metric_type = config.get("type", "")
        value = metric.get("value")

        if "DriftedColumnsCount" in metric_type and isinstance(value, dict):
            n_drifted = int(round(value.get("count", 0)))
            share = float(value.get("share", 0.0))
            drift_share_threshold = float(config.get("drift_share", 0.5))
            dataset_drift = share >= drift_share_threshold

        elif "ValueDrift" in metric_type and value is not None:
            col_name = config.get("column")
            threshold = float(config.get("threshold", 0.05))
            if col_name:
                try:
                    score = float(value)
                    drifted = score < threshold
                except (TypeError, ValueError):
                    score, drifted = None, False
                per_feature.append(
                    FeatureDriftResult(feature=col_name, drifted=drifted, score=score)
                )

    return dataset_drift, n_drifted, per_feature


# ---------------------------------------------------------------------------
# Main drift-check entry point
# ---------------------------------------------------------------------------


def run_drift_check(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> DriftCheckResult:
    """Run Evidently DataDriftPreset over the 19 contract feature columns.

    Parameters
    ----------
    reference_df:
        Training-split features (obtain via ``build_reference()``).
    current_df:
        Recent prediction inputs (obtain via ``get_current_window()``).
    output_dir:
        If provided, write ``drift_report.html`` and ``drift_summary.json``
        to this directory.  ``None`` (default) skips artifact writing.

    Returns
    -------
    DriftCheckResult
        ``dataset_drift``: True when Evidently's DriftedColumnsCount share
        ≥ its internal threshold (0.5 by default).

        ``retrain_recommended``: True when ``dataset_drift`` is True **OR**
        ``drifted_share >= DRIFT_FEATURE_THRESHOLD`` (default 0.30).  This
        lower secondary threshold catches partial drift that hasn't yet crossed
        Evidently's 50 % dataset-level bar.
    """
    ref = reference_df[MONITORED_FEATURES].reset_index(drop=True)
    cur = current_df[MONITORED_FEATURES].reset_index(drop=True)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref, current_data=cur)

    try:
        result_dict = json.loads(snapshot.json())
    except Exception:  # noqa: BLE001
        logger.exception("Failed to parse Evidently snapshot JSON.")
        result_dict = {}

    dataset_drift, n_drifted, per_feature = _parse_evidently_snapshot(result_dict)
    n_total = len(MONITORED_FEATURES)
    drifted_share = n_drifted / n_total if n_total > 0 else 0.0

    retrain_recommended = dataset_drift or (drifted_share >= DRIFT_FEATURE_THRESHOLD)
    if dataset_drift:
        retrain_reason = (
            f"Evidently dataset drift: {n_drifted}/{n_total} features drifted "
            f"({drifted_share:.0%} >= dataset-level threshold)"
        )
    elif drifted_share >= DRIFT_FEATURE_THRESHOLD:
        retrain_reason = (
            f"{n_drifted}/{n_total} features drifted "
            f"({drifted_share:.0%} >= retrain threshold {DRIFT_FEATURE_THRESHOLD:.0%})"
        )
    else:
        retrain_reason = f"no significant drift ({n_drifted}/{n_total} features, {drifted_share:.0%})"

    result = DriftCheckResult(
        dataset_drift=dataset_drift,
        n_drifted_features=n_drifted,
        n_total_features=n_total,
        drifted_share=drifted_share,
        per_feature=per_feature,
        retrain_recommended=retrain_recommended,
        retrain_reason=retrain_reason,
    )

    if output_dir is not None:
        _write_artifacts(snapshot, result, Path(output_dir))

    return result


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def _write_artifacts(
    snapshot: object,
    check_result: DriftCheckResult,
    out_dir: Path,
) -> tuple[str, str]:
    """Write Evidently HTML report and a machine-readable JSON summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "drift_report.html"
    json_path = out_dir / "drift_summary.json"

    # Evidently HTML report
    if hasattr(snapshot, "save_html"):
        snapshot.save_html(str(html_path))
    elif hasattr(snapshot, "json"):
        html_path.write_text(
            "<html><body><pre>"
            + json.dumps(json.loads(snapshot.json()), indent=2)
            + "</pre></body></html>",
            encoding="utf-8",
        )

    # Machine-readable drift summary (DriftCheckResult)
    summary = {
        "dataset_drift": check_result.dataset_drift,
        "n_drifted_features": check_result.n_drifted_features,
        "n_total_features": check_result.n_total_features,
        "drifted_share": check_result.drifted_share,
        "retrain_recommended": check_result.retrain_recommended,
        "retrain_reason": check_result.retrain_reason,
        "per_feature": [
            {"feature": f.feature, "drifted": f.drifted, "score": f.score}
            for f in check_result.per_feature
        ],
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(
        "Drift artifacts written: html=%s json=%s retrain=%s",
        html_path,
        json_path,
        check_result.retrain_recommended,
    )
    return str(html_path), str(json_path)
