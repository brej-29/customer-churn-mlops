---
title: Customer Churn UI
emoji: 🔮
colorFrom: green
colorTo: teal
sdk: docker
app_port: 7860
pinned: false
---

# Customer Churn Prediction UI

Streamlit interface for the Customer Churn API Space.
Enter customer attributes and get an instant churn-risk score, plus an optional
SHAP-grounded LLM explanation with retention-playbook citations.

This container contains **no ML code** — it calls the API over HTTP.

## Required Space Variable

Set this under **Settings → Variables** (not Secrets — not sensitive) in the HF Space panel.

| Variable | Value |
|---|---|
| `API_BASE_URL` | Full URL of the API Space, e.g. `https://<YOUR_HF_USERNAME>-churn-api.hf.space` |

Deploy the **API Space first** to get the URL, then set this variable before starting the UI Space.

## Source

GitHub: [brej-29/customer-churn-mlops](https://github.com/brej-29/customer-churn-mlops)
Branch: `tier3-deployment`
