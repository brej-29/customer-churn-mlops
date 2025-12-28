from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier

app = FastAPI(title="Customer Churn Prediction API")

MODEL_PATH = Path("models/xgboost_churn_model.json")
_model: Optional[XGBClassifier] = None


class PredictRequest(BaseModel):
    tenure: float
    monthly_charges: float
    contract_type: int
    has_internet: int
    support_calls: int
    is_senior: int


class PredictResponse(BaseModel):
    churn_probability: float


def load_model() -> Optional[XGBClassifier]:
    global _model
    if _model is None and MODEL_PATH.exists():
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        _model = model
    return _model


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = load_model()

    features = np.array(
        [
            [
                request.tenure,
                request.monthly_charges,
                request.contract_type,
                request.has_internet,
                request.support_calls,
                request.is_senior,
            ]
        ],
        dtype=float,
    )

    if model is None:
        score = 0.2
        if request.tenure < 12:
            score += 0.2
        if request.monthly_charges > 80:
            score += 0.1
        if request.support_calls >= 3:
            score += 0.1
        if request.contract_type == 0:
            score += 0.1
        if request.is_senior == 1:
            score += 0.05
        probability = max(0.0, min(0.95, score))
    else:
        probability = float(model.predict_proba(features)[0, 1])

    return PredictResponse(churn_probability=probability)