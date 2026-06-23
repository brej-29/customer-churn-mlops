"""Streamlit UI for the Telco Customer Churn prediction demo."""
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is on sys.path so ui.utils resolves regardless of how
# Streamlit invokes this script (locally: adds project root; Docker: PYTHONPATH=/app).
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import streamlit as st
from requests.exceptions import RequestException

from ui.utils import classify_churn_risk

# ---------------------------------------------------------------------------
# Constants — valid option lists must match the API's Pydantic Literal fields
# ---------------------------------------------------------------------------

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:7860")

YES_NO = ["No", "Yes"]
YES_NO_NOPHONE = ["No", "No phone service", "Yes"]
YES_NO_NOINTERNET = ["No", "No internet service", "Yes"]
GENDER_OPTIONS = ["Female", "Male"]
SENIOR_OPTIONS = ["No", "Yes"]
INTERNET_OPTIONS = ["DSL", "Fiber optic", "No"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAYMENT_OPTIONS = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]

RISK_COLORS = {"Low": "#2e7d32", "Medium": "#e65100", "High": "#c62828"}
RISK_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

# ---------------------------------------------------------------------------
# Example customer presets
# ---------------------------------------------------------------------------

_HIGH_RISK: Dict[str, Any] = {
    # New month-to-month customer, high bill, fiber, no add-ons, electronic check
    "tenure": 2,
    "MonthlyCharges": 85.0,
    "TotalCharges": 170.0,
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

_LOW_RISK: Dict[str, Any] = {
    # Long-tenured two-year contract customer, DSL, support add-ons, autopay
    "tenure": 48,
    "MonthlyCharges": 55.0,
    "TotalCharges": 2640.0,
    "gender": "Male",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "Yes",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
}

_DEFAULTS: Dict[str, Any] = {
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


def _apply_preset(preset: Dict[str, Any]) -> None:
    for field, value in preset.items():
        st.session_state[f"inp_{field}"] = value
    # Clear previous result so the new preset starts fresh
    for key in ("last_result", "last_payload", "explanation"):
        st.session_state.pop(key, None)


def _init_state() -> None:
    for field, value in _DEFAULTS.items():
        key = f"inp_{field}"
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def get_api_health(api_base_url: str) -> Dict[str, Any]:
    try:
        r = requests.get(api_base_url.rstrip("/") + "/health", timeout=2)
        if r.status_code == 200:
            return r.json()
    except RequestException:
        pass
    return {"status": "offline", "model_loaded": False, "model_version": None}


def call_predict(api_base_url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(api_base_url.rstrip("/") + "/predict", json=payload, timeout=10)
    except RequestException as exc:
        st.error(f"Error calling API: {exc}")
        return None
    if r.status_code == 503:
        st.error("Model not loaded on the API server — check the champion is registered.")
        return None
    if r.status_code == 422:
        st.error(f"Validation error: {r.json().get('detail', r.text)}")
        return None
    if not r.ok:
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    return r.json()


def call_explain(api_base_url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(api_base_url.rstrip("/") + "/explain", json=payload, timeout=60)
    except RequestException as exc:
        st.error(f"Error calling /explain: {exc}")
        return None
    if r.status_code == 503:
        st.error("Model not loaded on the API server.")
        return None
    if not r.ok:
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    return r.json()


# ---------------------------------------------------------------------------
# Payload builder — field names must match PredictRequest exactly
# ---------------------------------------------------------------------------


def _build_payload() -> Dict[str, Any]:
    """Read widget values from session state and assemble the /predict payload.

    SeniorCitizen is stored as "No"/"Yes" for display but converted to
    integer 0/1 per the API's Pydantic Literal[0, 1] constraint.
    """
    s = st.session_state
    return {
        "tenure": float(s["inp_tenure"]),
        "MonthlyCharges": float(s["inp_MonthlyCharges"]),
        "TotalCharges": float(s["inp_TotalCharges"]),
        "gender": s["inp_gender"],
        "SeniorCitizen": 1 if s["inp_SeniorCitizen"] == "Yes" else 0,
        "Partner": s["inp_Partner"],
        "Dependents": s["inp_Dependents"],
        "PhoneService": s["inp_PhoneService"],
        "MultipleLines": s["inp_MultipleLines"],
        "InternetService": s["inp_InternetService"],
        "OnlineSecurity": s["inp_OnlineSecurity"],
        "OnlineBackup": s["inp_OnlineBackup"],
        "DeviceProtection": s["inp_DeviceProtection"],
        "TechSupport": s["inp_TechSupport"],
        "StreamingTV": s["inp_StreamingTV"],
        "StreamingMovies": s["inp_StreamingMovies"],
        "Contract": s["inp_Contract"],
        "PaperlessBilling": s["inp_PaperlessBilling"],
        "PaymentMethod": s["inp_PaymentMethod"],
    }


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Churn Risk Predictor",
    page_icon="📉",
    layout="wide",
)

_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("API Settings")
    api_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL)

    health = get_api_health(api_url)
    api_ok = health.get("status") == "ok"
    model_loaded = health.get("model_loaded", False)
    model_version = health.get("model_version")

    if api_ok and model_loaded:
        status_color, status_text = "#2e7d32", f"Online · model v{model_version}"
    elif api_ok:
        status_color, status_text = "#e65100", "Online — no model loaded"
    else:
        status_color, status_text = "#c62828", "Offline"

    st.markdown(
        f'<span style="color:{status_color};font-weight:bold;">● {status_text}</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Example customers ────────────────────────────────────────────────────
    st.subheader("Try an example customer")
    st.caption("Fills the form instantly — then click Predict.")
    col_a, col_b = st.columns(2)
    if col_a.button("🔴 High risk", use_container_width=True, help="Month-to-month, new customer, fiber, no add-ons"):
        _apply_preset(_HIGH_RISK)
    if col_b.button("🟢 Low risk", use_container_width=True, help="Two-year contract, long tenure, support add-ons, autopay"):
        _apply_preset(_LOW_RISK)

    st.divider()

    # ── Account ──────────────────────────────────────────────────────────────
    st.subheader("Account")
    st.number_input(
        "Tenure (months)",
        min_value=0, max_value=72, step=1,
        key="inp_tenure",
        help="How many months this customer has been with the company. "
             "New customers (< 6 months) churn at much higher rates.",
    )
    st.selectbox(
        "Contract type",
        CONTRACT_OPTIONS,
        key="inp_Contract",
        help="Month-to-month customers have no lock-in and churn far more. "
             "Two-year contract customers are the most loyal.",
    )
    st.selectbox(
        "Paperless billing",
        YES_NO,
        key="inp_PaperlessBilling",
        help="Whether the customer receives bills by email rather than post.",
    )
    st.selectbox(
        "Payment method",
        PAYMENT_OPTIONS,
        key="inp_PaymentMethod",
        help="Electronic check is the highest-churn payment method — it requires "
             "active effort each month. Automatic bank transfer or credit card = autopay = stickier.",
    )

    # ── Demographics ─────────────────────────────────────────────────────────
    st.subheader("Demographics")
    st.selectbox(
        "Gender",
        GENDER_OPTIONS,
        key="inp_gender",
        help="Customer gender. Minimal predictive power in this dataset.",
    )
    st.selectbox(
        "Senior citizen",
        SENIOR_OPTIONS,
        key="inp_SeniorCitizen",
        help="Whether the customer is 65 or older.",
    )
    st.selectbox(
        "Has a partner",
        YES_NO,
        key="inp_Partner",
        help="Whether the customer lives with a partner. Single customers churn slightly more.",
    )
    st.selectbox(
        "Has dependents",
        YES_NO,
        key="inp_Dependents",
        help="Whether the customer has children or other dependents. "
             "Customers with dependents are more likely to stay.",
    )

    # ── Services ─────────────────────────────────────────────────────────────
    st.subheader("Services")
    st.selectbox("Phone service", YES_NO, key="inp_PhoneService",
                 help="Whether the customer has a phone line.")
    st.selectbox("Multiple lines", YES_NO_NOPHONE, key="inp_MultipleLines",
                 help="Whether the customer has multiple phone lines.")
    st.selectbox(
        "Internet service",
        INTERNET_OPTIONS,
        key="inp_InternetService",
        help="Type of internet connection. Fiber optic comes with higher bills "
             "and also higher churn — likely due to competitive alternatives.",
    )
    st.selectbox(
        "Online security add-on",
        YES_NO_NOINTERNET,
        key="inp_OnlineSecurity",
        help="One of the strongest retention signals — customers without security "
             "add-ons churn significantly more.",
    )
    st.selectbox("Online backup", YES_NO_NOINTERNET, key="inp_OnlineBackup",
                 help="Cloud backup service. More add-ons = stickier customer.")
    st.selectbox("Device protection", YES_NO_NOINTERNET, key="inp_DeviceProtection",
                 help="Device insurance plan.")
    st.selectbox(
        "Tech support",
        YES_NO_NOINTERNET,
        key="inp_TechSupport",
        help="Dedicated tech support subscription. Strong retention signal — "
             "customers with support help churn significantly less.",
    )
    st.selectbox("Streaming TV", YES_NO_NOINTERNET, key="inp_StreamingTV",
                 help="Whether the customer streams TV through the service.")
    st.selectbox("Streaming movies", YES_NO_NOINTERNET, key="inp_StreamingMovies",
                 help="Whether the customer streams movies through the service.")

    # ── Billing ──────────────────────────────────────────────────────────────
    st.subheader("Billing")
    st.number_input(
        "Monthly charges ($)",
        min_value=0.0, max_value=200.0, step=1.0, format="%.2f",
        key="inp_MonthlyCharges",
        help="Current monthly bill in dollars. Higher bills increase churn risk, "
             "especially on month-to-month contracts.",
    )
    st.number_input(
        "Total charges to date ($)",
        min_value=0.0, max_value=10000.0, step=10.0, format="%.2f",
        key="inp_TotalCharges",
        help="Cumulative charges since the customer joined (roughly tenure × monthly). "
             "Set to 0 for brand-new customers.",
    )


# ---------------------------------------------------------------------------
# Main area — intro
# ---------------------------------------------------------------------------

st.title("📉 Customer Churn Risk Predictor")
st.markdown(
    "Telecom companies lose revenue every time a customer cancels — and **winning back "
    "a lost customer costs 5–10× more than keeping one**. This demo predicts whether a "
    "customer is likely to churn based on their account details, then optionally generates "
    "a plain-language explanation of the key risk drivers with a recommended retention action. "
    "Fill in the sidebar or click an example customer to get started."
)

with st.expander("How it works — technical detail", expanded=False):
    st.markdown(
        """
**ML pipeline (Tier 1)**
Trained on the IBM Telco Customer Churn dataset — 7 043 customers, 26.5 % churn rate.
A five-model leaderboard (logistic regression, XGBoost, LightGBM, CatBoost, MLP) was
evaluated with 5-fold stratified CV scored on **PR-AUC** (the right metric for imbalanced
classification — ROC-AUC inflates because it rewards easy true-negative predictions).
XGBoost was selected and tuned with Optuna (60 trials, TPE sampler). The model is wrapped
in an **isotonic calibration** layer so probabilities are meaningful, then a
**cost-optimal decision threshold (0.174)** is selected by minimising `5 × FN + 1 × FP`
(missing a churner costs 5× more than a wasted retention offer).
Final test results: **PR-AUC 0.660 · ROC-AUC 0.848 · Recall 0.872 · Precision 0.462**.

**GenAI explanation layer (Tier 2)**
The `/explain` endpoint computes **SHAP feature attributions** (TreeExplainer on the
uncalibrated XGBoost — exact, no sampling, fast enough for live serving) and passes
the top-5 drivers into a retrieval-augmented prompt. Eight hand-curated retention
playbooks are indexed with `all-MiniLM-L6-v2` sentence embeddings in a FAISS flat index;
the most relevant playbooks are retrieved by cosine similarity and injected as context.
A `CONSTRAINT` line in every prompt lists the allowed driver feature names, banning the
LLM from citing features not supported by SHAP. Faithfulness is measured offline with a
two-tier checker (exact string match → cosine similarity ≥ 0.5); live score on Gemini:
**0.900 (45/50)**. Primary LLM: `gemini-2.5-flash-lite`; fallback: Groq
`llama-3.1-8b-instant`; deterministic rule-based fallback if both are unavailable.

**MLOps stack (Tier 3)**
Models are versioned in an **MLflow registry on DagsHub**. A champion/challenger gate
only promotes a new model if its PR-AUC exceeds the current champion by ≥ 0.01.
A **three-job GitHub Actions CI pipeline** runs ruff + 399 offline tests, builds and
audits both Docker images, and re-evaluates the registered champion (PR-AUC ≥ 0.60 gate
or build fails). **Evidently drift monitoring** compares recent prediction inputs against
the training distribution; `retrain_recommended = True` when ≥ 30 % of the 19 features
drift. Deployed as two HF Docker Spaces: this Streamlit UI (830 MB, zero ML packages)
and the FastAPI backend (4.9 GB, CPU-only torch, embedding model pre-baked at build time).

[View source on GitHub →](https://github.com/brej-29/customer-churn-mlops)
"""
    )

st.divider()

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

if not api_ok:
    st.warning("API is offline. Start the FastAPI service and refresh the page.")

if st.button(
    "Predict churn risk",
    type="primary",
    disabled=not (api_ok and model_loaded),
):
    payload = _build_payload()
    with st.spinner("Scoring customer..."):
        result = call_predict(api_url, payload)
    if result is not None:
        st.session_state["last_payload"] = payload
        st.session_state["last_result"] = result
        st.session_state["explanation"] = None

# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

result = st.session_state.get("last_result")
if result is not None:
    prob = float(result["churn_probability"])
    pred = bool(result["churn_prediction"])
    thr = float(result["threshold"])
    ver = result.get("model_version", "?")

    bucket, _ = classify_churn_risk(prob)
    color = RISK_COLORS.get(bucket, "#555")
    emoji = RISK_EMOJI.get(bucket, "")

    col_verdict, col_numbers = st.columns([3, 2])

    with col_verdict:
        verdict_text = "Likely to **churn**" if pred else "Likely to **stay**"
        badge = (
            f'<span style="background:{color};color:#fff;padding:4px 12px;'
            f'border-radius:4px;font-size:0.85em;font-weight:bold;">'
            f"{emoji} {bucket} risk</span>"
        )
        st.markdown(f"### {verdict_text} &nbsp; {badge}", unsafe_allow_html=True)
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(
            f"The model assigns a **{prob:.1%} churn probability** to this customer. "
            f"We flag anyone above **{thr:.1%}** for retention outreach because "
            f"missing a churner costs 5× more than a wasted retention offer."
        )

    with col_numbers:
        st.metric("Churn probability", f"{prob:.1%}")
        st.metric("Decision threshold", f"{thr:.1%}")
        st.metric("Model version", f"v{ver}")

    st.divider()

    # ── Explain ──────────────────────────────────────────────────────────────
    if st.button(
        "Explain this prediction",
        disabled=not (api_ok and model_loaded),
        help="Calls /explain: computes SHAP drivers, retrieves relevant retention "
             "playbooks, and asks the LLM for a plain-language narrative.",
    ):
        with st.spinner("Generating explanation (SHAP + RAG + LLM)..."):
            expl = call_explain(api_url, st.session_state.get("last_payload", {}))
        st.session_state["explanation"] = expl

    expl_data = st.session_state.get("explanation")
    if expl_data is not None:
        provider = expl_data.get("provider", "unknown")
        rl = expl_data.get("risk_level", bucket.lower())
        rl_cap = rl.capitalize()
        rl_color = RISK_COLORS.get(rl_cap, "#555")
        rl_emoji = RISK_EMOJI.get(rl_cap, "")

        st.subheader("Explanation")

        prov_note = (
            "Deterministic rule-based explanation (LLM unavailable)."
            if provider == "fallback"
            else f"Generated by: `{provider}`"
        )
        st.markdown(
            f'<span style="color:{rl_color};font-weight:bold;">'
            f"{rl_emoji} Risk level: {rl.upper()}</span> &nbsp;·&nbsp; "
            + prov_note,
            unsafe_allow_html=True,
        )

        st.markdown("**Summary**")
        st.write(expl_data.get("summary", ""))

        key_factors = expl_data.get("key_factors", [])
        if key_factors:
            st.markdown("**Key churn drivers**")
            for factor in key_factors:
                st.markdown(f"- {factor}")

        st.markdown("**Recommended action**")
        st.info(expl_data.get("recommended_action", ""))

        citations = expl_data.get("citations", [])
        if citations:
            st.markdown("**Retention playbook sources**")
            for cite in citations:
                st.markdown(f"- `{cite}`")

        ungrounded = expl_data.get("ungrounded_factors", [])
        if ungrounded:
            st.warning(
                f"Grounding note: {len(ungrounded)} key factor(s) may not map directly "
                f"to a SHAP driver: {', '.join(ungrounded)}"
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "local")
st.caption(
    f"Model: `customer-churn-xgboost@champion` · "
    f"Tracking: `{tracking_uri}` · "
    "[GitHub](https://github.com/brej-29/customer-churn-mlops)"
)
