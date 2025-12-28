import random

from locust import HttpUser, between, task


def random_payload() -> dict:
    """Generate a realistic random payload for the /predict endpoint."""
    tenure = random.randint(0, 72)
    monthly_charges = random.uniform(20.0, 120.0)
    contract_type = random.choice([0, 1, 2])  # month-to-month, one-year, two-year
    has_internet = random.choice([0, 1])
    support_calls = random.randint(0, 10)
    is_senior = random.choice([0, 1])

    return {
        "tenure": float(tenure),
        "monthly_charges": float(monthly_charges),
        "contract_type": float(contract_type),
        "has_internet": float(has_internet),
        "support_calls": float(support_calls),
        "is_senior": float(is_senior),
    }


class ChurnUser(HttpUser):
    """Locust user that exercises the churn prediction API."""

    wait_time = between(0.5, 2.0)

    @task(1)
    def health_check(self) -> None:
        self.client.get("/health")

    @task(5)
    def predict_churn(self) -> None:
        payload = random_payload()
        self.client.post("/predict", json=payload)