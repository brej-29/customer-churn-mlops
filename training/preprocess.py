import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "tenure",
    "monthly_charges",
    "contract_type",
    "has_internet",
    "support_calls",
    "is_senior",
]

TARGET_COLUMN = "churn"


def generate_synthetic_churn_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(0, 72, size=n_samples)
    monthly_charges = rng.uniform(20.0, 120.0, size=n_samples)
    contract_type = rng.integers(0, 3, size=n_samples)
    has_internet = rng.integers(0, 2, size=n_samples)
    support_calls = rng.integers(0, 11, size=n_samples)
    is_senior = rng.integers(0, 2, size=n_samples)

    logits = (
        -2.0
        + 0.03 * (monthly_charges - 70.0)
        - 0.04 * tenure
        + 0.4 * (contract_type == 0).astype(float)
        + 0.3 * (support_calls >= 3).astype(float)
        + 0.4 * is_senior
    )
    proba = 1.0 / (1.0 + np.exp(-logits))
    churn = rng.binomial(1, proba)

    df = pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract_type": contract_type,
            "has_internet": has_internet,
            "support_calls": support_calls,
            "is_senior": is_senior,
            "churn": churn,
        }
    )
    return df


def get_train_test_data(
    n_samples: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = generate_synthetic_churn_data(n_samples=n_samples, random_state=random_state)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)