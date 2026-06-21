"""Streamlit UI for the Telco Customer Churn prediction demo."""
import os
from typing import Any, Dict, Optional

import requests
import streamlit as st
from requests.exceptions import RequestException

from ui.utils import classify_churn_risk

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def get_api_health(api_base_url: str) -> Dict[str, Any]:
    try:
        response = requests.get(api_base_url.rstrip("/") + "/health", timeout=2)
        if response.status_code == 200:
            return response.json()
    except RequestException:
        pass
    return {"status": "offline", "model_loaded": False, "model_version": None}


def call_predict(api_base_url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        response = requests.post(
            api_base_url.rstrip("/") + "/predict",
            json=payload,
            timeout=10,
        )
    except RequestException as exc:
        st.error(f"Error calling API: {exc}")
        return None

    if response.status_code == 503:
        st.error("The model is not loaded on the API server. Check that the champion model is registered.")
        return None
    if response.status_code == 422:
        st.error(f"Validation error: {response.json().get('detail', response.text)}")
        return None
    if not response.ok:
        st.error(f"API error {response.status_code}: {response.text}")
        return None

    return response.json()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn MLOps Demo",
    page_icon="",
    layout="wide",
)

st.title("Customer Churn Prediction Demo")
st.markdown(
    """
**IBM Telco dataset** — 7 043 customers, 26.5 % churn rate.
The champion model is a calibrated XGBoost pipeline registered in the MLflow Model Registry.
Enter customer attributes and click **Predict churn risk** to score the customer.
"""
)

# ---------------------------------------------------------------------------
# Sidebar — API config and status
# ---------------------------------------------------------------------------

st.sidebar.header("API Settings")
api_url = st.sidebar.text_input("FastAPI base URL", value=DEFAULT_API_URL)

health = get_api_health(api_url)
api_ok = health.get("status") == "ok"
model_loaded = health.get("model_loaded", False)
model_version = health.get("model_version")

if api_ok and model_loaded:
    status_color, status_text = "#2e7d32", f"Online  |  model v{model_version}"
elif api_ok:
    status_color, status_text = "#e65100", "Online — no model loaded"
else:
    status_color, status_text = "#c62828", "Offline"

st.sidebar.markdown(
    f'<span style="color:{status_color}; font-weight:bold;">● {status_text}</span>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — customer inputs
# ---------------------------------------------------------------------------

st.sidebar.subheader("Account")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"], index=0)
senior = st.sidebar.selectbox("Senior citizen", ["No (0)", "Yes (1)"], index=0)
senior_value = 1 if senior.startswith("Yes") else 0
partner = st.sidebar.selectbox("Partner", ["No", "Yes"], index=0)
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"], index=0)

st.sidebar.subheader("Billing")
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
monthly_charges = st.sidebar.number_input(
    "Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0
)
total_charges = st.sidebar.number_input(
    "Total charges ($)", min_value=0.0, max_value=10000.0,
    value=round(float(tenure) * 70.0, 2), step=1.0,
)
contract = st.sidebar.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"], index=0
)
paperless_billing = st.sidebar.selectbox("Paperless billing", ["No", "Yes"], index=1)
payment_method = st.sidebar.selectbox(
    "Payment method",
    ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
    index=2,
)

st.sidebar.subheader("Phone")
phone_service = st.sidebar.selectbox("Phone service", ["No", "Yes"], index=1)
multiple_lines = st.sidebar.selectbox(
    "Multiple lines", ["No", "No phone service", "Yes"], index=0
)

st.sidebar.subheader("Internet")
internet_service = st.sidebar.selectbox(
    "Internet service", ["DSL", "Fiber optic", "No"], index=1
)
online_security = st.sidebar.selectbox(
    "Online security", ["No", "No internet service", "Yes"], index=0
)
online_backup = st.sidebar.selectbox(
    "Online backup", ["No", "No internet service", "Yes"], index=0
)
device_protection = st.sidebar.selectbox(
    "Device protection", ["No", "No internet service", "Yes"], index=0
)
tech_support = st.sidebar.selectbox(
    "Tech support", ["No", "No internet service", "Yes"], index=0
)
streaming_tv = st.sidebar.selectbox(
    "Streaming TV", ["No", "No internet service", "Yes"], index=0
)
streaming_movies = st.sidebar.selectbox(
    "Streaming movies", ["No", "No internet service", "Yes"], index=0
)

# ---------------------------------------------------------------------------
# Main content — prediction
# ---------------------------------------------------------------------------

if not api_ok:
    st.warning("API appears offline. Start the FastAPI service, then refresh.")

predict_clicked = st.button("Predict churn risk", disabled=not (api_ok and model_loaded))

if predict_clicked:
    payload = {
        "tenure": float(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "gender": gender,
        "SeniorCitizen": senior_value,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
    }

    with st.spinner("Scoring customer..."):
        result = call_predict(api_url, payload)

    if result is not None:
        prob = float(result["churn_probability"])
        pred = bool(result["churn_prediction"])
        thr = float(result["threshold"])
        ver = result.get("model_version", "?")

        bucket, explanation = classify_churn_risk(prob)

        col_prob, col_pred, col_ver = st.columns(3)
        col_prob.metric("Churn probability", f"{prob:.3f}")
        col_pred.metric("Prediction", "CHURN" if pred else "STAY")
        col_ver.metric("Model version", ver)

        st.progress(min(max(prob, 0.0), 1.0))

        if pred:
            st.error(
                f"This customer is predicted to **churn** "
                f"(p = {prob:.3f} >= threshold {thr:.4f})."
            )
        else:
            st.success(
                f"This customer is predicted to **stay** "
                f"(p = {prob:.3f} < threshold {thr:.4f})."
            )

        st.write(f"**Risk bucket:** {bucket}")
        st.caption(explanation)

# ---------------------------------------------------------------------------
# Model info footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Model & MLOps details")
st.markdown(
    f"""
- **Registered model**: `customer-churn-xgboost@champion`
- **Tracking URI**: `{os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlruns.db')}`
- **Decision threshold**: loaded from the champion version's `threshold` tag
  (fallback: `reports/threshold.json`)
- **Calibration**: isotonic (selected on OOF Brier score)
"""
)
