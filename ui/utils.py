from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


INTEGER_FEATURES = [
    "tenure",
    "contract_type",
    "has_internet",
    "support_calls",
    "is_senior",
]

FLOAT_FEATURES = ["monthly_charges"]


@dataclass
class CustomerFeatures:
    tenure: float
    monthly_charges: float
    contract_type: float
    has_internet: float
    support_calls: float
    is_senior: float


def _coerce_int_feature(name: str, value: Any) -> int:
    if isinstance(value, bool):
        # Avoid treating booleans as integers.
        raise ValueError(f"{name} must be an integer, not a boolean.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric (received {value!r}).") from exc

    if not float(numeric_value).is_integer():
        raise ValueError(f"{name} must be an integer value (received {numeric_value}).")

    return int(numeric_value)


def build_payload(features: CustomerFeatures) -> Dict[str, float]:
    """Build the JSON payload expected by the FastAPI /predict endpoint.

    This mirrors the API typing:
    - integer features are accepted as ints or floats that represent ints (e.g. 12.0)
    - monthly_charges is coerced to float.
    """
    data: Dict[str, float] = {}

    for field in INTEGER_FEATURES:
        raw_value = getattr(features, field)
        data[field] = float(_coerce_int_feature(field, raw_value))

    for field in FLOAT_FEATURES:
        raw_value = getattr(features, field)
        try:
            data[field] = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field} must be a floating-point numeric value (received {raw_value!r})."
            ) from exc

    return data


def classify_churn_risk(
    churn_probability: float,
    low_threshold: float = 0.33,
    high_threshold: float = 0.66,
) -> Tuple[str, str]:
    """Return a human-readable churn risk bucket and explanation.

    Buckets:
    - Low:    p < low_threshold
    - Medium: low_threshold <= p < high_threshold
    - High:   p >= high_threshold
    """
    p = float(churn_probability)

    if p < low_threshold:
        return "Low", (
            f"Low risk (p &lt; {low_threshold:.2f}). Customer is unlikely to churn under "
            "current conditions."
        )
    if p < high_threshold:
        return "Medium", (
            f"Medium risk ({low_threshold:.2f} ≤ p &lt; {high_threshold:.2f}). "
            "Customer shows some warning signals and is worth monitoring."
        )
    return "High", (
        f"High risk (p ≥ {high_threshold:.2f}). Customer has multiple churn drivers "
        "and likely needs proactive retention actions."
    )