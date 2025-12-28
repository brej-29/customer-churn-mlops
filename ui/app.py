import os
from typing import Any, Dict, Optional

import numpy as np
import requests
import streamlit as st
from requests.exceptions import RequestException

from training.preprocess import FEATURE_COLUMNS, generate_synthetic_churn_data
from training.train import EXPERIMENT_NAME, MODEL_NAME
from ui.utils import CustomerFeatures, build_payload, classify_churn_risk

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def get_api_status(api_base_url: str) -> bool:
    try:
        response = requests.get(api_base_url.rstrip("/") + "/health", timeout=2)
        return response.status_code == 200
    except RequestException:
        return False


def call_predict(api_base_url: str, payload: Dict[str, float]) -> Optional[float]:
    try:
        response = requests.post(
            api_base_url.rstrip("/") + "/predict",
            json=payload,
            timeout=5,
        )
        response.raise_for_status()
    except RequestException as exc:
        st.error(f"Error calling API: {exc}")
        return None

    data: Dict[str, Any] = response.json()
    probability = data.get("churn_probability")
    if probability is None:
        st.error("API response did not contain 'churn_probability'.")
        return None

    try:
        return float(probability)
    except (TypeError, ValueError):
        st.error("API returned a non-numeric churn_probability.")
        return None


@st.cache_data(show_spinner=False)
def compute_permutation_importance(api_base_url: str, n_samples: int = 64) -> Optional[Dict[str, float]]:
    """Approximate feature importance by permutation using the live API model."""
    try:
        df = generate_synthetic_churn_data(n_samples=n_samples, random_state=123)
    except Exception:
        return None

    def _predict_for_df(dataframe) -> Optional[np.ndarray]:
        probs = []
        for _, row in dataframe.iterrows():
            features = CustomerFeatures(
                tenure=row["tenure"],
                monthly_charges=row["monthly_charges"],
                contract_type=row["contract_type"],
                has_internet=row["has_internet"],
                support_calls=row["support_calls"],
                is_senior=row["is_senior"],
            )
            payload = build_payload(features)
            p = call_predict(api_base_url, payload)
            if p is None:
                return None
            probs.append(p)
        return np.asarray(probs, dtype=float)

    baseline = _predict_for_df(df[FEATURE_COLUMNS])
    if baseline is None:
        return None

    importances: Dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        shuffled = df.copy()
        shuffled[col] = np.random.permutation(shuffled[col].values)
        perturbed = _predict_for_df(shuffled[FEATURE_COLUMNS])
        if perturbed is None:
            return None
        importances[col] = float(np.abs(baseline - perturbed).mean())

    total = sum(importances.values())
    if total > 0:
        for k in list(importances.keys()):
            importances[k] = importances[k] / total

    return importances


st.set_page_config(
    page_title="Customer Churn MLOps Demo",
    page_icon="📉",
    layout="wide",
)

st.title("Customer Churn Prediction Demo")

st.markdown(
    """
**What is customer churn?**  
Customer churn is when an existing customer cancels their subscription or stops using a service.
Telecom companies care deeply about churn because acquiring a new customer is usually much more
expensive than retaining an existing one.

**What does this app predict?**  
For a single customer, the model estimates the **probability that they will churn** in the near term.

**What data is used?**  
This demo uses a **synthetic telecom-style dataset** generated locally by the code in
`training/preprocess.py`. Each row represents a customer with:

- `tenure` – how many months they have been a customer  
- `monthly_charges` – their current monthly bill amount  
- `contract_type` – whether the contract is month-to-month, one-year, or two-year  
- `has_internet` – whether they have internet service  
- `support_calls` – number of support calls in the last month  
- `is_senior` – whether the customer is a senior citizen
"""
)

st.info(
    "Disclaimer: This is a demo pipeline with a **synthetic dataset**. "
    "It is **not** a production telecom churn model."
)

# Sidebar: API config and inputs
st.sidebar.header("API & Model Settings")

api_url = st.sidebar.text_input("FastAPI base URL", value=DEFAULT_API_URL)

api_ok = get_api_status(api_url)
status_color = "#2e7d32" if api_ok else "#c62828"
status_text = "Online" if api_ok else "Offline"

st.sidebar.markdown(
    f"""
**API status:**  
<span style="color:{status_color}; font-weight:bold;">● {status_text}</span>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Customer Features")

if "sample_clicked" not in st.session_state:
    st.session_state.sample_clicked = False

def set_sample_customer():
    st.session_state.tenure = 3
    st.session_state.monthly_charges = 95.0
    st.session_state.contract_display = "Month-to-month"
    st.session_state.has_internet = "Yes"
    st.session_state.support_calls = 4
    st.session_state.is_senior = "Yes"


if st.sidebar.button("Try a sample customer"):
    set_sample_customer()
    st.session_state.sample_clicked = True

tenure = st.sidebar.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=st.session_state.get("tenure", 12),
    step=1,
    key="tenure",
)
monthly_charges = st.sidebar.number_input(
    "Monthly charges",
    min_value=0.0,
    max_value=200.0,
    value=st.session_state.get("monthly_charges", 70.0),
    step=1.0,
    key="monthly_charges",
)

contract_display_to_value = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}
contract_display = st.sidebar.selectbox(
    "Contract type",
    list(contract_display_to_value.keys()),
    index=list(contract_display_to_value.keys()).index(
        st.session_state.get("contract_display", "Month-to-month")
    ),
    key="contract_display",
)
contract_type = contract_display_to_value[contract_display]

has_internet = st.sidebar.selectbox(
    "Has internet service?",
    ["No", "Yes"],
    index=["No", "Yes"].index(st.session_state.get("has_internet", "Yes")),
    key="has_internet",
)
has_internet_value = 1 if has_internet == "Yes" else 0

support_calls = st.sidebar.number_input(
    "Support calls in last month",
    min_value=0,
    max_value=20,
    value=st.session_state.get("support_calls", 1),
    step=1,
    key="support_calls",
)

is_senior = st.sidebar.selectbox(
    "Is senior citizen?",
    ["No", "Yes"],
    index=["No", "Yes"].index(st.session_state.get("is_senior", "No")),
    key="is_senior",
)
is_senior_value = 1 if is_senior == "Yes" else 0

# Main layout
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Prediction")

    if not api_ok:
        st.warning(
            "API appears to be offline. Start the FastAPI container or service, "
            "then refresh this page."
        )

    if st.button("Predict churn risk", disabled=not api_ok):
        features = CustomerFeatures(
            tenure=tenure,
            monthly_charges=monthly_charges,
            contract_type=contract_type,
            has_internet=has_internet_value,
            support_calls=support_calls,
            is_senior=is_senior_value,
        )

        try:
            payload = build_payload(features)
        except ValueError as exc:
            st.error(str(exc))
        else:
            probability = call_predict(api_url, payload)

            if probability is not None:
                bucket, explanation = classify_churn_risk(probability)

                st.metric(
                    label="Churn probability",
                    value=f"{probability:.3f}",
                    delta=None,
                )
                st.write(f"**Risk bucket:** {bucket}")
                st.caption(explanation)

                st.progress(min(max(probability, 0.0), 1.0))

with col_right:
    st.subheader("What drives churn?")

    if api_ok:
        with st.spinner("Computing feature importance from the live model..."):
            importances = compute_permutation_importance(api_url)
    else:
        importances = None

    if importances:
        import pandas as pd

        imp_df = (
            pd.DataFrame(
                {"feature": list(importances.keys()), "importance": list(importances.values())}
            )
            .sort_values("importance", ascending=True)
        )

        st.bar_chart(
            imp_df.set_index("feature"),
            height=250,
        )

        st.caption(
            "Feature importance is estimated via permutation on synthetic customers. "
            "Higher values indicate features that move predicted churn the most when shuffled."
        )
    elif api_ok:
        st.info(
            "Feature importance is not available yet. Try running a few predictions and reload "
            "the page, or check that the API is reachable from this UI container."
        )
    else:
        st.info("Start the API to compute feature importance.")

st.markdown("---")

st.subheader("Model & MLOps details")

model_uri = os.getenv("CHURN_MODEL_URI", f"models:/{MODEL_NAME}@champion")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

st.markdown(
    f"""
- **Current model URI (`CHURN_MODEL_URI`)**: `{model_uri}`
- **MLflow experiment name**: `{EXPERIMENT_NAME}`
- **Registered model name**: `{MODEL_NAME}`
- **MLflow tracking server**: `{tracking_uri}`
- **MLflow UI**: [http://localhost:5000](http://localhost:5000)
  - Open the **`{EXPERIMENT_NAME}`** experiment to inspect recent runs.
  - Look under **Models → `{MODEL_NAME}`** to see model versions and aliases (e.g. `champion`).
"""
)