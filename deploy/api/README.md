---
title: Customer Churn API
emoji: 📉
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Customer Churn Prediction API

FastAPI service that scores telecom customers for churn risk.
Loads a calibrated XGBoost champion from a DagsHub-hosted MLflow registry at startup.
Optionally generates SHAP-grounded LLM explanations via Gemini (Groq fallback).

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service status + model load flag |
| `/predict` | POST | Churn probability + binary prediction at cost-optimal threshold |
| `/explain` | POST | SHAP top-5 drivers + RAG playbook context + LLM narrative |
| `/stats` | GET | Prediction-log aggregates (count, latency p50/p95, avg prob) |
| `/docs` | GET | Interactive Swagger UI |

## Required Space Secrets

Set these under **Settings → Repository secrets** in the HF Space panel.

| Secret | Required | Value |
|---|---|---|
| `MLFLOW_TRACKING_URI` | **Yes** | `https://dagshub.com/<dagshub-user>/customer-churn-mlops.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | **Yes** | Your DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | **Yes** | Your DagsHub access token |
| `GEMINI_API_KEY` | Optional | Enables `/explain` with `gemini-2.5-flash-lite` |
| `GROQ_API_KEY` | Optional | Groq fallback LLM for `/explain` |

If no LLM key is set, `/explain` returns a deterministic rule-based explanation (`provider: fallback`).

## Known limitations (free tier)

- **Cold start**: Space sleeps after 48 h of inactivity; restart re-downloads the champion
  from DagsHub MLflow (~30 s).
- **Prediction log**: SQLite resets on container restart — fine for a demo.
- **`/explain` SHAP**: requires `data/raw/telco_churn.csv` in the build context
  (see [deploy/HF_DEPLOY.md](../../deploy/HF_DEPLOY.md) optional step).

## Source

GitHub: [brej-29/customer-churn-mlops](https://github.com/brej-29/customer-churn-mlops)
Branch: `tier3-deployment`
