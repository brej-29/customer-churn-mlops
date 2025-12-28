from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
client.app.state.model = None  # Ensure tests do not depend on an external MLflow model


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_accepts_ints():
    payload = {
        "tenure": 12,
        "monthly_charges": 70.0,
        "contract_type": 0,
        "has_internet": 1,
        "support_calls": 2,
        "is_senior": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    # Latency header should be present and numeric.
    assert "X-Model-Latency-ms" in response.headers
    assert float(response.headers["X-Model-Latency-ms"]) >= 0.0


def test_predict_accepts_integer_floats():
    payload = {
        "tenure": 12.0,
        "monthly_charges": 70.0,
        "contract_type": 0.0,
        "has_internet": 1.0,
        "support_calls": 2.0,
        "is_senior": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_rejects_non_integer_float():
    payload = {
        "tenure": 12.7,
        "monthly_charges": 70.0,
        "contract_type": 0,
        "has_internet": 1,
        "support_calls": 2,
        "is_senior": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "tenure must be an integer value" in str(data["detail"])


def test_missing_field():
    payload = {
        # "tenure" is omitted on purpose
        "monthly_charges": 70.0,
        "contract_type": 0,
        "has_internet": 1,
        "support_calls": 2,
        "is_senior": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert any(
        isinstance(err, dict) and err.get("loc", [None])[-1] == "tenure"
        for err in data["detail"]
    )


def test_stats_endpoint_reflects_prediction():
    payload = {
        "tenure": 6,
        "monthly_charges": 90.0,
        "contract_type": 0,
        "has_internet": 1,
        "support_calls": 3,
        "is_senior": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    stats_response = client.get("/stats")
    assert stats_response.status_code == 200
    stats = stats_response.json()

    for key in [
        "count",
        "success_count",
        "failure_count",
        "success_rate",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_avg_ms",
        "avg_churn_probability",
        "last_model_uri",
    ]:
        assert key in stats

    assert isinstance(stats["count"], int)
    assert stats["count"] >= 1
    assert isinstance(stats["latency_p50_ms"], (int, float))
    assert isinstance(stats["latency_p95_ms"], (int, float))
    assert isinstance(stats["latency_avg_ms"], (int, float))


def test_recent_endpoint_returns_logs():
    payload = {
        "tenure": 10,
        "monthly_charges": 60.0,
        "contract_type": 1,
        "has_internet": 0,
        "support_calls": 1,
        "is_senior": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    recent_response = client.get("/recent")
    assert recent_response.status_code == 200
    logs = recent_response.json()
    assert isinstance(logs, list)
    assert logs
    log = logs[0]
    assert "timestamp" in log
    assert "request_payload" in log
    assert "churn_probability" in log
    assert "latency_ms" in log
    assert "status" in log