import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from mlflow.exceptions import MlflowException
from pydantic import BaseModel

from training.preprocess import FEATURE_COLUMNS as TRAINING_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "CustomerChurnModel"
DEFAULT_MODEL_ALIAS = "champion"
DEFAULT_MODEL_URI = f"models:/{DEFAULT_MODEL_NAME}@{DEFAULT_MODEL_ALIAS}"
DEFAULT_TRACKING_URI = "file:./mlruns"

CHURN_MODEL_URI = os.getenv("CHURN_MODEL_URI", DEFAULT_MODEL_URI)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)

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
    data = {}
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)


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
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request_data: PredictRequest, request: Request) -> PredictResponse:
    model = getattr(request.app.state, "model", None)

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

    return PredictResponse(churn_probability=probability)