import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from mlflow.exceptions import MlflowException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from training.preprocess import FEATURE_COLUMNS as TRAINING_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "CustomerChurnModel"
DEFAULT_MODEL_ALIAS = "champion"
DEFAULT_MODEL_URI = f"models:/{DEFAULT_MODEL_NAME}@{DEFAULT_MODEL_ALIAS}"
DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_LOG_DB_PATH = "logs/predictions.db"

CHURN_MODEL_URI = os.getenv("CHURN_MODEL_URI", DEFAULT_MODEL_URI)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
LOG_DB_PATH = os.getenv("LOG_DB_PATH", DEFAULT_LOG_DB_PATH)
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
}

FEATURE_COLUMNS = TRAINING_FEATURE_COLUMNS

INT_FEATURE_COLUMNS = [
    "tenure",
    "contract_type",
    "has_internet",
    "support_calls",
    "is_senior",
]

FLOAT_FEATURE_COLUMNS = ["monthly_charges"]


def load_model() -> Any:
    model_uri = CHURN_MODEL_URI
    if not model_uri:
        return None

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Loaded MLflow model from {model_uri}")
        return model
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: Failed to load MLflow model from {model_uri}: {exc}")
        return None


def coerce_to_int64(field_name: str, value: Any) -> int:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be a numeric value (received {value!r})",
        ) from exc

    if not np.isfinite(numeric_value):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be a finite numeric value (received {value!r})",
        )

    if abs(numeric_value - round(numeric_value)) > 1e-9:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be an integer value (received {numeric_value})",
        )

    return int(round(numeric_value))


def build_features(payload: "PredictRequest") -> pd.DataFrame:
    """Build a single-row DataFrame with the exact dtypes expected by the MLflow model."""
    data: Dict[str, Any] = {}
    for column in FEATURE_COLUMNS:
        if column in INT_FEATURE_COLUMNS:
            raw_value = getattr(payload, column)
            data[column] = coerce_to_int64(column, raw_value)
        elif column in FLOAT_FEATURE_COLUMNS:
            raw_value = getattr(payload, column)
            try:
                data[column] = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"{column} must be a floating-point numeric value "
                        f"(received {raw_value!r})"
                    ),
                ) from exc
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unexpected feature column '{column}' in API configuration.",
            )

    features = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    features = features.astype(
        {
            "tenure": "int64",
            "contract_type": "int64",
            "has_internet": "int64",
            "support_calls": "int64",
            "is_senior": "int64",
            "monthly_charges": "float64",
        }
    )
    return features


def _get_db_connection() -> sqlite3.Connection:
    db_dir = os.path.dirname(LOG_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(LOG_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            request_payload TEXT NOT NULL,
            churn_probability REAL,
            model_uri TEXT,
            latency_ms REAL,
            status TEXT NOT NULL,
            error_message TEXT
        )
        """
    )
    return conn


def log_prediction(
    *,
    request_payload: Dict[str, Any],
    churn_probability: Optional[float],
    model_uri: str,
    latency_ms: float,
    status: str,
    error_message: Optional[str],
) -> None:
    """Persist a single prediction event to SQLite.

    Logging failures must never crash the API.
    """
    try:
        conn = _get_db_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO prediction_logs (
                    timestamp,
                    request_payload,
                    churn_probability,
                    model_uri,
                    latency_ms,
                    status,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(timespec="seconds"),
                    json.dumps(request_payload),
                    float(churn_probability) if churn_probability is not None else None,
                    model_uri,
                    float(latency_ms),
                    status,
                    error_message,
                ),
            )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to log prediction event.")
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def fetch_recent_logs(limit: int) -> List[Dict[str, Any]]:
    """Return the most recent prediction logs, newest first."""
    try:
        conn = _get_db_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT
                timestamp,
                request_payload,
                churn_probability,
                model_uri,
                latency_ms,
                status,
                error_message
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cursor.fetchall()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to fetch recent prediction logs.")
        return []
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    logs: List[Dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["request_payload"])
        except Exception:  # noqa: BLE001
            payload = {}

        logs.append(
            {
                "timestamp": row["timestamp"],
                "request_payload": payload,
                "churn_probability": row["churn_probability"],
                "model_uri": row["model_uri"],
                "latency_ms": row["latency_ms"],
                "status": row["status"],
                "error_message": row["error_message"],
            }
        )

    return logs


def compute_stats(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute simple summary statistics over recent prediction logs."""
    total = len(logs)
    success_count = sum(1 for log in logs if log.get("status") == "success")
    failure_count = total - success_count
    success_rate = float(success_count / total) if total > 0 else 0.0

    latencies = [
        float(log["latency_ms"])
        for log in logs
        if log.get("latency_ms") is not None
    ]
    probabilities = [
        float(log["churn_probability"])
        for log in logs
        if log.get("churn_probability") is not None
    ]

    if latencies:
        arr = np.asarray(latencies, dtype="float64")
        latency_p50 = float(np.percentile(arr, 50))
        latency_p95 = float(np.percentile(arr, 95))
        latency_avg = float(arr.mean())
    else:
        latency_p50 = latency_p95 = latency_avg = 0.0

    avg_probability = float(np.mean(probabilities)) if probabilities else 0.0
    last_model_uri = logs[0].get("model_uri") if logs else None

    return {
        "count": total,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_rate,
        "latency_p50_ms": latency_p50,
        "latency_p95_ms": latency_p95,
        "latency_avg_ms": latency_avg,
        "avg_churn_probability": avg_probability,
        "last_model_uri": last_model_uri,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)

if PROMETHEUS_ENABLED:
    Instrumentator().instrument(app).expose(app)


class PredictRequest(BaseModel):
    tenure: float
    monthly_charges: float
    contract_type: float
    has_internet: float
    support_calls: float
    is_senior: float


class PredictResponse(BaseModel):
    churn_probability: float


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats(limit: int = Query(100, ge=1, le=10_000)) -> Dict[str, Any]:
    logs = fetch_recent_logs(limit)
    return compute_stats(logs)


@app.get("/recent")
def recent(limit: int = Query(20, ge=1, le=1_000)) -> List[Dict[str, Any]]:
    return fetch_recent_logs(limit)


@app.post("/predict", response_model=PredictResponse)
def predict(
    request_data: PredictRequest,
    request: Request,
    response: Response,
) -> PredictResponse:
    model = getattr(request.app.state, "model", None)

    start_time = time.perf_counter()
    probability: Optional[float] = None
    status = "success"
    error_message: Optional[str] = None

    try:
        features = build_features(request_data)

        if model is None:
            tenure = int(features.loc[0, "tenure"])
            monthly_charges = float(features.loc[0, "monthly_charges"])
            contract_type = int(features.loc[0, "contract_type"])
            support_calls = int(features.loc[0, "support_calls"])
            is_senior = int(features.loc[0, "is_senior"])

            score = 0.2
            if tenure < 12:
                score += 0.2
            if monthly_charges > 80:
                score += 0.1
            if support_calls >= 3:
                score += 0.1
            if contract_type == 0:
                score += 0.1
            if is_senior == 1:
                score += 0.05
            probability = max(0.0, min(0.95, score))
        else:
            try:
                raw_pred = model.predict(features)
            except MlflowException as exc:
                logger.exception("MLflow model prediction failed.")
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Model prediction failed due to input schema or model error: "
                        f"{exc}"
                    ),
                ) from exc
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error during model prediction.")
                raise HTTPException(
                    status_code=500,
                    detail="Unexpected error during model prediction.",
                ) from exc

            if isinstance(raw_pred, (list, tuple, np.ndarray)):
                arr = np.asarray(raw_pred)
                if arr.ndim == 0:
                    probability = float(arr)
                elif arr.ndim == 1:
                    probability = float(arr[0])
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    probability = float(arr[0, 0])
                elif arr.ndim == 2 and arr.shape[1] == 2:
                    probability = float(arr[0, 1])
                else:
                    probability = float(arr.ravel()[0])
            else:
                probability = float(raw_pred)

            probability = float(np.clip(probability, 0.0, 1.0))
    except HTTPException as exc:
        status = "fail"
        error_message = str(exc.detail)
        raise
    except Exception:  # noqa: BLE001
        status = "fail"
        error_message = "Unexpected error during prediction."
        logger.exception("Unhandled error in /predict.")
        raise
    finally:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Model-Latency-ms"] = f"{latency_ms:.3f}"
        log_prediction(
            request_payload=request_data.dict(),
            churn_probability=probability,
            model_uri=CHURN_MODEL_URI,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
        )

    return PredictResponse(churn_probability=float(probability))