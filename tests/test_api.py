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