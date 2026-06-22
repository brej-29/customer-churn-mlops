"""Tests for api/main.py — Telco churn prediction API.

All tests use real sklearn pipelines (no model=None heuristic). The champion
model is not loaded from the MLflow registry during tests; load_champion_model
is monkeypatched so tests run offline and fast.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_csv_present = pytest.mark.skipif(
    not Path("data/raw/telco_churn.csv").exists(),
    reason="data/raw/telco_churn.csv not present",
)

# ---------------------------------------------------------------------------
# Canonical valid payload — a mid-tenure DSL customer unlikely to churn
# ---------------------------------------------------------------------------

VALID_PAYLOAD: dict = {
    "tenure": 24,
    "MonthlyCharges": 65.0,
    "TotalCharges": 1560.0,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Bank transfer (automatic)",
}

# ---------------------------------------------------------------------------
# Shared session fixtures — train once per pytest session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _tuned_params() -> dict:
    with open("reports/best_xgb_params.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def fast_pipeline(_tuned_params):
    """Real sklearn Pipeline with n_estimators=50 — trains in ~1 s for API tests."""
    from xgboost import XGBClassifier

    from churn.data import get_splits
    from churn.models import build_model_pipeline

    X_train, _, y_train, _ = get_splits()
    params = {**_tuned_params, "n_estimators": 50}
    pipeline = build_model_pipeline(XGBClassifier(**params))
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# Function-scoped fixtures — use monkeypatch to bypass registry loading
# ---------------------------------------------------------------------------


@pytest.fixture
def client_real(fast_pipeline, monkeypatch):
    """TestClient whose lifespan loads the fast_pipeline instead of the registry model."""
    import api.main as api_module

    monkeypatch.setattr(
        api_module,
        "load_champion_model",
        lambda tracking_uri=None: (fast_pipeline, 0.1741, "test-v1"),
    )
    with TestClient(api_module.app) as client:
        yield client


@pytest.fixture
def client_no_model(monkeypatch):
    """TestClient whose lifespan fails to load a model — app.state.model is None."""
    import api.main as api_module

    def _fail(tracking_uri=None):
        raise RuntimeError("No champion model in registry (test)")

    monkeypatch.setattr(api_module, "load_champion_model", _fail)
    with TestClient(api_module.app) as client:
        yield client


# ---------------------------------------------------------------------------
# 1. Model-quality gate — self-contained, no registry dependency
# ---------------------------------------------------------------------------


@_csv_present
def test_model_quality_gate_pr_auc(_tuned_params):
    """Tuned uncalibrated XGBoost must achieve >= 0.60 PR-AUC on the held-out test set.

    This test makes CI meaningful: a broken pipeline or data change that drops
    model quality below the floor will fail here before reaching production.
    """
    from sklearn.metrics import average_precision_score
    from xgboost import XGBClassifier

    from churn.data import get_splits
    from churn.models import build_model_pipeline

    X_train, X_test, y_train, y_test = get_splits()
    pipeline = build_model_pipeline(XGBClassifier(**_tuned_params))
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, proba))

    assert pr_auc >= 0.60, (
        f"Model quality gate FAILED: test PR-AUC {pr_auc:.4f} < floor 0.60"
    )
    print(f"\nModel quality gate: test PR-AUC = {pr_auc:.4f}")


# ---------------------------------------------------------------------------
# 2. Happy path — real model, valid payload
# ---------------------------------------------------------------------------


@_csv_present
def test_predict_happy_path_200(client_real):
    response = client_real.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


@_csv_present
def test_predict_probability_in_unit_interval(client_real):
    data = client_real.post("/predict", json=VALID_PAYLOAD).json()
    assert 0.0 <= data["churn_probability"] <= 1.0


@_csv_present
def test_predict_prediction_consistent_with_threshold(client_real):
    """churn_prediction must match (probability >= threshold)."""
    data = client_real.post("/predict", json=VALID_PAYLOAD).json()
    expected = data["churn_probability"] >= data["threshold"]
    assert data["churn_prediction"] == expected


@_csv_present
def test_predict_response_contains_required_fields(client_real):
    data = client_real.post("/predict", json=VALID_PAYLOAD).json()
    assert {"churn_probability", "churn_prediction", "threshold", "model_version"}.issubset(
        data.keys()
    )


@_csv_present
def test_predict_threshold_matches_loaded_value(client_real):
    """Threshold in response must equal the one loaded from the registry (0.1741)."""
    data = client_real.post("/predict", json=VALID_PAYLOAD).json()
    assert abs(data["threshold"] - 0.1741) < 1e-6


@_csv_present
def test_predict_model_version_in_response(client_real):
    data = client_real.post("/predict", json=VALID_PAYLOAD).json()
    assert data["model_version"] == "test-v1"


@_csv_present
def test_predict_latency_header_present(client_real):
    response = client_real.post("/predict", json=VALID_PAYLOAD)
    assert "X-Model-Latency-ms" in response.headers
    assert float(response.headers["X-Model-Latency-ms"]) >= 0.0


# ---------------------------------------------------------------------------
# 3. Determinism — same payload, same result
# ---------------------------------------------------------------------------


@_csv_present
def test_predict_determinism(client_real):
    r1 = client_real.post("/predict", json=VALID_PAYLOAD).json()
    r2 = client_real.post("/predict", json=VALID_PAYLOAD).json()
    assert r1["churn_probability"] == r2["churn_probability"]
    assert r1["churn_prediction"] == r2["churn_prediction"]


# ---------------------------------------------------------------------------
# 4. Validation — 422 on invalid input (Pydantic catches these before the handler)
# ---------------------------------------------------------------------------


def test_predict_422_bad_contract_category():
    """An unrecognised Contract value must return 422 via Pydantic validation."""
    import api.main as api_module

    client = TestClient(api_module.app)
    bad = {**VALID_PAYLOAD, "Contract": "enterprise"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_422_negative_tenure():
    """tenure < 0 violates the ge=0 constraint → 422."""
    import api.main as api_module

    client = TestClient(api_module.app)
    bad = {**VALID_PAYLOAD, "tenure": -1}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_422_negative_monthly_charges():
    import api.main as api_module

    client = TestClient(api_module.app)
    bad = {**VALID_PAYLOAD, "MonthlyCharges": -5.0}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_422_bad_gender():
    import api.main as api_module

    client = TestClient(api_module.app)
    bad = {**VALID_PAYLOAD, "gender": "Unknown"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_422_senior_citizen_invalid():
    """SeniorCitizen only accepts 0 or 1."""
    import api.main as api_module

    client = TestClient(api_module.app)
    bad = {**VALID_PAYLOAD, "SeniorCitizen": 2}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_422_missing_field():
    import api.main as api_module

    client = TestClient(api_module.app)
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 5. 503 when no model is loaded
# ---------------------------------------------------------------------------


@_csv_present
def test_predict_503_no_model(client_no_model):
    response = client_no_model.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503
    assert "No model loaded" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 6. /health endpoint
# ---------------------------------------------------------------------------


@_csv_present
def test_health_model_loaded_true(client_real):
    data = client_real.get("/health").json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == "test-v1"


@_csv_present
def test_health_model_loaded_false(client_no_model):
    data = client_no_model.get("/health").json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False


def test_health_always_200():
    """Health endpoint returns 200 regardless of model state."""
    import api.main as api_module

    client = TestClient(api_module.app)
    response = client.get("/health")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# 7. Schema contract — regression guard against field drift
# ---------------------------------------------------------------------------


def test_schema_contract_fields_match_all_features():
    """PredictRequest must expose exactly the 19 columns the pipeline expects."""
    from api.main import PredictRequest
    from churn.data import ALL_FEATURES

    api_fields = set(PredictRequest.model_fields.keys())
    expected = set(ALL_FEATURES)
    assert api_fields == expected, (
        f"Schema drift detected!\n"
        f"  Extra in API:    {api_fields - expected}\n"
        f"  Missing from API:{expected - api_fields}"
    )


# ---------------------------------------------------------------------------
# 8. Observability — /stats and /recent
# ---------------------------------------------------------------------------


@_csv_present
def test_stats_endpoint_reflects_prediction(client_real):
    client_real.post("/predict", json=VALID_PAYLOAD)
    stats = client_real.get("/stats").json()
    for key in [
        "count", "success_count", "failure_count", "success_rate",
        "latency_p50_ms", "latency_p95_ms", "latency_avg_ms",
        "avg_churn_probability", "last_model_uri",
    ]:
        assert key in stats
    assert stats["count"] >= 1


@_csv_present
def test_recent_endpoint_returns_prediction_fields(client_real):
    client_real.post("/predict", json=VALID_PAYLOAD)
    logs = client_real.get("/recent").json()
    assert isinstance(logs, list) and len(logs) >= 1
    log = logs[0]
    for key in ["timestamp", "request_payload", "churn_probability", "latency_ms", "status"]:
        assert key in log


# ---------------------------------------------------------------------------
# 9. /explain endpoint
# ---------------------------------------------------------------------------

_EXPLAIN_FIELDS = {
    "churn_probability", "threshold", "model_version",
    "risk_level", "summary", "key_factors", "recommended_action",
    "citations", "provider", "ungrounded_factors",
}

# Shared mock ChurnExplanation returned by the stubbed _get_explanation.
def _mock_explanation(features, calibrated_prob, top_k=5):
    from churn.genai.explainer import ChurnExplanation
    expl = ChurnExplanation(
        risk_level="high" if calibrated_prob > 0.5 else "low",
        summary="Mock summary for testing.",
        key_factors=["Contract: Month-to-month"],
        recommended_action="Offer a discount.",
        citations=[],
    )
    return expl, {"provider": "mock-llm", "model": "mock", "probability": calibrated_prob, "ungrounded_factors": []}


@_csv_present
def test_explain_200_schema(client_real, monkeypatch):
    """POST /explain returns 200 with all required fields."""
    import api.main as api_module
    monkeypatch.setattr(api_module, "_get_explanation", _mock_explanation)
    response = client_real.post("/explain", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert _EXPLAIN_FIELDS.issubset(response.json().keys())


@_csv_present
def test_explain_probability_matches_predict(client_real, monkeypatch):
    """churn_probability in /explain must equal /predict for the same input."""
    import api.main as api_module
    monkeypatch.setattr(api_module, "_get_explanation", _mock_explanation)
    predict_prob = client_real.post("/predict", json=VALID_PAYLOAD).json()["churn_probability"]
    explain_prob = client_real.post("/explain", json=VALID_PAYLOAD).json()["churn_probability"]
    assert abs(predict_prob - explain_prob) < 1e-6


@_csv_present
def test_explain_threshold_and_version_match_predict(client_real, monkeypatch):
    """threshold and model_version in /explain must match /predict."""
    import api.main as api_module
    monkeypatch.setattr(api_module, "_get_explanation", _mock_explanation)
    pred = client_real.post("/predict", json=VALID_PAYLOAD).json()
    expl = client_real.post("/explain", json=VALID_PAYLOAD).json()
    assert abs(pred["threshold"] - expl["threshold"]) < 1e-6
    assert pred["model_version"] == expl["model_version"]


@_csv_present
def test_explain_provider_field_returned(client_real, monkeypatch):
    """provider field must be present and equal what _get_explanation returns."""
    import api.main as api_module
    monkeypatch.setattr(api_module, "_get_explanation", _mock_explanation)
    data = client_real.post("/explain", json=VALID_PAYLOAD).json()
    assert data["provider"] == "mock-llm"


@_csv_present
def test_explain_llm_failure_returns_200_with_fallback(client_real, monkeypatch):
    """When _get_explanation raises, /explain must return 200 with provider='fallback'."""
    import api.main as api_module

    def _failing(features, calibrated_prob, top_k=5):
        raise RuntimeError("Simulated LLM failure")

    monkeypatch.setattr(api_module, "_get_explanation", _failing)
    response = client_real.post("/explain", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "fallback"
    assert _EXPLAIN_FIELDS.issubset(data.keys())


@_csv_present
def test_explain_fallback_probability_still_correct(client_real, monkeypatch):
    """Even when LLM fails, /explain must return the champion's calibrated probability."""
    import api.main as api_module

    def _failing(features, calibrated_prob, top_k=5):
        raise RuntimeError("Simulated LLM failure")

    monkeypatch.setattr(api_module, "_get_explanation", _failing)
    predict_prob = client_real.post("/predict", json=VALID_PAYLOAD).json()["churn_probability"]
    explain_prob = client_real.post("/explain", json=VALID_PAYLOAD).json()["churn_probability"]
    assert abs(predict_prob - explain_prob) < 1e-6


@_csv_present
def test_explain_503_no_model(client_no_model):
    """When no champion model is loaded, /explain returns 503."""
    response = client_no_model.post("/explain", json=VALID_PAYLOAD)
    assert response.status_code == 503
    assert "No model loaded" in response.json()["detail"]


@_csv_present
def test_explain_ungrounded_factors_is_list(client_real, monkeypatch):
    """ungrounded_factors must always be a list (empty is fine)."""
    import api.main as api_module
    monkeypatch.setattr(api_module, "_get_explanation", _mock_explanation)
    data = client_real.post("/explain", json=VALID_PAYLOAD).json()
    assert isinstance(data["ungrounded_factors"], list)


@_csv_present
def test_explain_key_factors_is_list(client_real, monkeypatch):
    import api.main as api_module
    monkeypatch.setattr(api_module, "_get_explanation", _mock_explanation)
    data = client_real.post("/explain", json=VALID_PAYLOAD).json()
    assert isinstance(data["key_factors"], list)


def test_explain_422_bad_input():
    """Invalid payload to /explain returns 422 (Pydantic validation)."""
    import api.main as api_module
    client = TestClient(api_module.app)
    bad = {**VALID_PAYLOAD, "Contract": "enterprise"}
    response = client.post("/explain", json=bad)
    assert response.status_code == 422
