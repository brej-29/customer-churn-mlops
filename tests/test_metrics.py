from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_metrics_endpoint_exposes_http_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    # The default instrumentator exposes HTTP metrics such as http_requests_total.
    assert "http" in body.lower()