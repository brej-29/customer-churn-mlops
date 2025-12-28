import requests
import streamlit as st
from requests.exceptions import RequestException

DEFAULT_API_URL = "http://localhost:8000"

st.title("Customer Churn Prediction")

st.sidebar.header("API Settings")
api_url = st.sidebar.text_input("FastAPI base URL", value=DEFAULT_API_URL)

st.header("Customer Features")

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12,
    step=1,
)
monthly_charges = st.number_input(
    "Monthly charges", min_value=0.0, max_value=200.0, value=70.0, step=1.0
)

contract_display_to_value = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}
contract_display = st.selectbox("Contract type", list(contract_display_to_value.keys()))
contract_type = contract_display_to_value[contract_display]

has_internet = st.selectbox("Has internet service?", ["No", "Yes"])
has_internet_value = 1 if has_internet == "Yes" else 0

support_calls = st.number_input(
    "Support calls in last month", min_value=0, max_value=20, value=1, step=1
)

is_senior = st.selectbox("Is senior citizen?", ["No", "Yes"])
is_senior_value = 1 if is_senior == "Yes" else 0

if st.button("Predict"):
    # Basic client-side validation in case the widgets are reconfigured to allow floats.
    integer_fields = [("tenure", tenure), ("support_calls", support_calls)]
    non_integer_fields = [
        (name, value)
        for name, value in integer_fields
        if isinstance(value, float) and not float(value).is_integer()
    ]
    if non_integer_fields:
        for field_name, value in non_integer_fields:
            st.warning(
                f"{field_name.replace('_', ' ')} must be an integer value (received {value})."
            )
        st.stop()

    payload = {
        "tenure": int(tenure),
        "monthly_charges": float(monthly_charges),
        "contract_type": int(contract_type),
        "has_internet": int(has_internet_value),
        "support_calls": int(support_calls),
        "is_senior": int(is_senior_value),
    }

    try:
        response = requests.post(
            api_url.rstrip("/") + "/predict",
            json=payload,
            timeout=5,
        )
        response.raise_for_status()
    except RequestException as exc:
        st.error(f"Error calling API: {exc}")
    else:
        data = response.json()
        probability = data.get("churn_probability")
        if probability is not None:
            st.success(f"Estimated churn probability: {probability:.3f}")
        else:
            st.error("API response did not contain 'churn_probability'.")