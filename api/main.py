"""Churn prediction API — serves the registered champion pipeline from the MLflow Model Registry."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from mlflow.tracking import MlflowClient
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from churn.config import settings
from churn.data import ALL_FEATURES

logger = logging.getLogger(__name__)

CHAMPION_MODEL_NAME = "customer-churn-xgboost"
CHAMPION_ALIAS = "champion"
_THRESHOLD_FALLBACK_PATH = "reports/threshold.json"

LOG_DB_PATH = os.getenv("LOG_DB_PATH", "logs/predictions.db")
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "true").lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Registry loader — injectable in tests via monkeypatch
# ---------------------------------------------------------------------------


def load_champion_model(tracking_uri: Optional[str] = None) -> tuple[Any, float, str]:
    """Load champion pipeline, threshold, and version from the MLflow Model Registry.

    Returns (model, threshold, version_str). Falls back to reports/threshold.json
    if the registered version's threshold tag is absent. Raises on any failure;
    the lifespan caller is responsible for error handling.
    """
    uri = tracking_uri or settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    mv = client.get_model_version_by_alias(CHAMPION_MODEL_NAME, CHAMPION_ALIAS)
    version = str(mv.version)
    tag_val = mv.tags.get("threshold")
    if tag_val is not None:
        threshold = float(tag_val)
    else:
        with open(_THRESHOLD_FALLBACK_PATH) as f:
            threshold = json.load(f)["threshold"]
        logger.warning(
            "threshold tag absent from %s v%s; using fallback %s",
            CHAMPION_MODEL_NAME, version, _THRESHOLD_FALLBACK_PATH,
        )

    model_uri = f"models:/{CHAMPION_MODEL_NAME}@{CHAMPION_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)
    print(
        f"Loaded {CHAMPION_MODEL_NAME} v{version}"
        f" (threshold={threshold:.4f}) from {model_uri}"
    )
    return model, threshold, version


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, threshold, version = load_champion_model()
        app.state.model = model
        app.state.threshold = threshold
        app.state.model_version = version
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load champion model at startup: %s", exc)
        app.state.model = None
        app.state.threshold = None
        app.state.model_version = None
    yield


app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)

if PROMETHEUS_ENABLED:
    Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    # Numeric features — pipeline expects float64
    tenure: float = Field(ge=0.0, description="Months with the company")
    MonthlyCharges: float = Field(ge=0.0, description="Current monthly bill")
    TotalCharges: float = Field(ge=0.0, description="Total charges to date")

    # SeniorCitizen is 0/1 in the raw data; pipeline stores it as string "0"/"1"
    SeniorCitizen: Literal[0, 1]

    # Categorical features — values must match the Telco training vocabulary exactly
    gender: Literal["Female", "Male"]
    Partner: Literal["No", "Yes"]
    Dependents: Literal["No", "Yes"]
    PhoneService: Literal["No", "Yes"]
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "No internet service", "Yes"]
    OnlineBackup: Literal["No", "No internet service", "Yes"]
    DeviceProtection: Literal["No", "No internet service", "Yes"]
    TechSupport: Literal["No", "No internet service", "Yes"]
    StreamingTV: Literal["No", "No internet service", "Yes"]
    StreamingMovies: Literal["No", "No internet service", "Yes"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["No", "Yes"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold: float
    model_version: str


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


def build_features(payload: PredictRequest) -> pd.DataFrame:
    """Build a single-row DataFrame matching the pipeline's expected raw input columns."""
    row: Dict[str, Any] = {
        "tenure": float(payload.tenure),
        "MonthlyCharges": float(payload.MonthlyCharges),
        "TotalCharges": float(payload.TotalCharges),
        # SeniorCitizen: pipeline was trained on "0"/"1" strings (clean_telco casts all cats)
        "gender": payload.gender,
        "SeniorCitizen": str(payload.SeniorCitizen),
        "Partner": payload.Partner,
        "Dependents": payload.Dependents,
        "PhoneService": payload.PhoneService,
        "MultipleLines": payload.MultipleLines,
        "InternetService": payload.InternetService,
        "OnlineSecurity": payload.OnlineSecurity,
        "OnlineBackup": payload.OnlineBackup,
        "DeviceProtection": payload.DeviceProtection,
        "TechSupport": payload.TechSupport,
        "StreamingTV": payload.StreamingTV,
        "StreamingMovies": payload.StreamingMovies,
        "Contract": payload.Contract,
        "PaperlessBilling": payload.PaperlessBilling,
        "PaymentMethod": payload.PaymentMethod,
    }
    return pd.DataFrame([row], columns=ALL_FEATURES)


# ---------------------------------------------------------------------------
# Prediction logging
# ---------------------------------------------------------------------------


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
    """Persist a prediction event to SQLite. Failures are swallowed so the API never crashes."""
    try:
        conn = _get_db_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO prediction_logs (
                    timestamp, request_payload, churn_probability,
                    model_uri, latency_ms, status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
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
    try:
        conn = _get_db_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT timestamp, request_payload, churn_probability,
                   model_uri, latency_ms, status, error_message
            FROM prediction_logs
            ORDER BY id DESC LIMIT ?
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
    total = len(logs)
    success_count = sum(1 for log in logs if log.get("status") == "success")
    failure_count = total - success_count
    success_rate = float(success_count / total) if total > 0 else 0.0

    latencies = [
        float(log["latency_ms"]) for log in logs if log.get("latency_ms") is not None
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health(request: Request) -> Dict[str, Any]:
    model = getattr(request.app.state, "model", None)
    version = getattr(request.app.state, "model_version", None)
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": version,
    }


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
    threshold = getattr(request.app.state, "threshold", 0.5)
    model_version = getattr(request.app.state, "model_version", "unknown")
    model_uri = f"models:/{CHAMPION_MODEL_NAME}@{CHAMPION_ALIAS}"

    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded.")

    start_time = time.perf_counter()
    probability: Optional[float] = None
    status = "success"
    error_message: Optional[str] = None

    try:
        features = build_features(request_data)
        proba_arr = model.predict_proba(features)
        probability = float(np.clip(proba_arr[0, 1], 0.0, 1.0))
        churn_prediction = bool(probability >= threshold)
    except Exception as exc:  # noqa: BLE001
        status = "fail"
        error_message = str(exc)
        logger.exception("Unexpected error in /predict.")
        raise HTTPException(
            status_code=500, detail="Unexpected error during prediction."
        ) from exc
    finally:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Model-Latency-ms"] = f"{latency_ms:.3f}"
        log_prediction(
            request_payload=request_data.model_dump(),
            churn_probability=probability,
            model_uri=model_uri,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
        )

    return PredictResponse(
        churn_probability=probability,
        churn_prediction=churn_prediction,
        threshold=threshold,
        model_version=str(model_version),
    )
