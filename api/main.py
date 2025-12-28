import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel

DEFAULT_MODEL_NAME = "CustomerChurnModel"
DEFAULT_MODEL_ALIAS = "champion"
DEFAULT_MODEL_URI = f"models:/{DEFAULT_MODEL_NAME}@{DEFAULT_MODEL_ALIAS}"
DEFAULT_TRACKING_URI = "file:./mlruns"

CHURN_MODEL_URI = os.getenv("CHURN_MODEL_URI", DEFAULT_MODEL_URI)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)


class PredictRequest(BaseModel):
    tenure: float
    monthly_charges: float
    contract_type: int
    has_internet: int
    support_calls: int
    is_senior: int


class PredictResponse(BaseModel):
    churn_probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request_data: PredictRequest, request: Request) -> PredictResponse:
    model = getattr(request.app.state, "model", None)

    features = pd.DataFrame(
        [
            {
                "tenure": request_data.tenure,
                "monthly_charges": request_data.monthly_charges,
                "contract_type": request_data.contract_type,
                "has_internet": request_data.has_internet,
                "support_calls": request_data.support_calls,
                "is_senior": request_data.is_senior,
            }
        ]
    )

    if model is None:
        score = 0.2
        if request_data.tenure < 12:
            score += 0.2
        if request_data.monthly_charges > 80:
            score += 0.1
        if request_data.support_calls >= 3:
            score += 0.1
        if request_data.contract_type == 0:
            score += 0.1
        if request_data.is_senior == 1:
            score += 0.05
        probability = max(0.0, min(0.95, score))
    else:
        raw_pred = model.predict(features)
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