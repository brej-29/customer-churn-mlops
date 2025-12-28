<div align="center">
  <h1>Customer Churn MLOps Starter</h1>
  <p><i>Production-style starter for telecom churn — XGBoost Training → MLflow Tracking/Registry → FastAPI Model Serving → Streamlit UI → Dockerized End-to-End Demo</i></p>
</div>

<br>

<div align="center">
  <img alt="CI" src="https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg">
  <img alt="Language" src="https://img.shields.io/badge/Language-Python-blue">
  <img alt="Framework" src="https://img.shields.io/badge/Framework-FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="UI" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="MLOps" src="https://img.shields.io/badge/MLOps-MLflow%20Tracking%20%2B%20Registry-orange">
  <img alt="Deployment" src="https://img.shields.io/badge/Infra-Docker%20Compose-2496ED?logo=docker&logoColor=white">
  </br>
  <a href="https://github.com/OWNER/REPO" target="_blank">
    <button style="background-color: #0f766e; color: white; padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-weight: bold;">
        REPO LINK
    </button>
  </a>
</div>

<div align="center">
  <br>
  <b>Built with:</b>
  <br><br>
  <code>Python</code> |
  <code>XGBoost</code> |
  <code>scikit-learn</code> |
  <code>MLflow</code> |
  <code>FastAPI</code> |
  <code>Streamlit</code> |
  <code>Docker</code> |
  <code>pytest</code>
</div>

---

## Project Structure

## Project Structure

```text
.
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI app (/health, /predict)
│   └── Dockerfile        # FastAPI Docker image
├── data/
│   └── .gitkeep          # Placeholder for raw/processed data
├── models/
│   └── .gitkeep          # Optional local model artifacts
├── training/
│   ├── __init__.py
│   ├── preprocess.py     # Synthetic data generation & train/test split
│   └── train.py          # XGBoost training + MLflow logging + Model Registry
├── ui/
│   ├── app.py            # Streamlit UI that calls the FastAPI /predict endpoint
│   └── Dockerfile        # Streamlit Docker image
├── monitoring/
│   ├── __init__.py
│   └── generate_drift_report.py  # Evidently drift report generation
├── loadtest/
│   └── locustfile.py     # Locust load test for /health and /predict
├── notebooks/
│   ├── 01_eda.ipynb      # EDA using the same synthetic data pipeline
│   └── README.md
├── tests/
│   ├── test_api.py       # API unit tests
│   ├── test_metrics.py   # /metrics scrape endpoint tests
│   ├── test_monitoring_drift.py  # Drift report generation tests
│   └── test_ui_utils.py  # UI payload / typing tests
├── docker-compose.yml    # End-to-end demo (MLflow + API + UI + trainer)
├── requirements.txt
├── requirements-dev.txt  # Optional notebook/EDA/load-test dependencies
├── LICENSE
└── README.md
```

---

## Architecture (high level)

```text
[ Streamlit UI ]  -->  [ FastAPI /predict ]  -->  [ MLflow Tracking & Model Registry ]
       |                          |                        |
       |                          |                        +--> [ Model artifacts (mlartifacts/) ]
       |                          |
       |                          +--> [ MLflow Tracking DB (SQLite: mlruns.db) ]
       |
       +--> [ Users explore churn risk & drivers ]
```

---

## Setup

1. **Create and activate a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # On macOS/Linux
   # .venv\Scripts\activate         # On Windows (PowerShell or cmd)
   ```

2. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 1. Run Training (XGBoost + MLflow + Model Registry)

The training script:

- Generates a synthetic churn dataset (no external data download required)
- Trains an XGBoost classifier
- Logs hyperparameters and metrics to **MLflow**
- Logs the model with an input example and inferred signature so the schema is stored
- Registers the model in the **MLflow Model Registry** as `CustomerChurnModel` (when a registry-enabled tracking URI is used)

From the project root:

```bash
python -m training.train
```

After it completes, you should see:

- An MLflow run recorded under experiment `customer-churn`
- A new or updated registered model version named `CustomerChurnModel` (when the Model Registry is available)
- Console output showing the run-level model URI and a suggested serving URI such as `models:/CustomerChurnModel@champion` (alias-based) or `models:/CustomerChurnModel/Production` (stage-based)

### MLflow tracking URI and UI

Both training and the API look at these environment variables:

- `MLFLOW_TRACKING_URI`  
  - Where MLflow logs runs and (optionally) hosts the Model Registry  
  - Default: `file:./mlruns` (local filesystem tracking)
- `CHURN_MODEL_URI`  
  - The model URI that the API will load with `mlflow.pyfunc.load_model`  
  - Default: `models:/CustomerChurnModel@champion`

#### Option A: Quick local UI (no Model Registry)

If you just want to browse local runs stored under `./mlruns`:

```bash
mlflow ui
```

This starts a UI at `http://127.0.0.1:5000`, where you can see runs, metrics, and artifacts.

#### Option B: Local MLflow server with Model Registry

To use the Model Registry locally, run an MLflow server backed by SQLite:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

mlflow server \
  --backend-store-uri sqlite:///mlruns.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

Then open `http://127.0.0.1:5000` in your browser to access both the MLflow UI and the Model Registry.

After running `python -m training.train` with this server running, you should see new runs and registered model versions under the name `CustomerChurnModel`.

---

## 2. Run the FastAPI Service

The FastAPI app:

- Loads a model once at startup using FastAPI's lifespan mechanism
- Uses `mlflow.pyfunc.load_model(CHURN_MODEL_URI)` to load the churn model
- Keeps the loaded model in `app.state.model` and reuses it for all requests
- Falls back to a simple heuristic only if the MLflow model cannot be loaded (useful for local development/tests)

By default, the app looks for:

- `MLFLOW_TRACKING_URI` – must point at the same MLflow tracking server used during training
- `CHURN_MODEL_URI` – defaults to `models:/CustomerChurnModel@champion`

A typical setup using the local MLflow server from above:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export CHURN_MODEL_URI=models:/CustomerChurnModel@champion  # or a specific version (e.g. models:/CustomerChurnModel/3) or stage (e.g. models:/CustomerChurnModel/Production)

uvicorn api.main:app --reload
```

- Health check: `GET http://localhost:8000/health` → `{"status": "ok"}`
- Prediction: `POST http://localhost:8000/predict` with JSON body, for example:

  ```json
  {
    "tenure": 12,
    "monthly_charges": 70.0,
    "contract_type": 0,
    "has_internet": 1,
    "support_calls": 2,
    "is_senior": 0
  }
  ```

The response will contain:

```json
{
  "churn_probability": 0.42
}
```

(The exact value will differ.)

Note on input types:

- The model expects integer features (`tenure`, `contract_type`, `has_internet`, `support_calls`, `is_senior`) and a floating-point feature (`monthly_charges`).
- The API will accept both integers and floats that represent integer values (for example, `12` and `12.0`) for the integer features and will safely coerce them before calling the MLflow model.
- If a non-integer float is sent for an integer feature (for example, `12.7`), the API will return a 422 response with a clear validation error instead of a 500 error.

### How serving picks the model

By default this starter uses an MLflow **model alias** to decide which version the API serves:

- The training script registers each new model version as `CustomerChurnModel`.
- When the Model Registry is available, it automatically updates the alias `champion`
  to point at the latest registered version.
- The FastAPI service loads whatever `CHURN_MODEL_URI` points at, and by default that is
  `models:/CustomerChurnModel@champion`.

This has a few practical consequences:

- Retrain and re-register a model, and the next time you (re)start the API it will
  serve the new "champion" version.
- To roll back, change the alias in the MLflow UI/CLI to point at an older version
  (no code changes required).
- You can override the alias used during training by setting `CHURN_MODEL_ALIAS`
  (for example, `staging`), and you can override what the API serves by setting
  `CHURN_MODEL_URI` explicitly (for example, `models:/CustomerChurnModel@staging`).

If you prefer stage-based URIs instead of aliases, you can:

- Promote a model version to the `Production` stage in the MLflow UI, and
- Set `CHURN_MODEL_URI=models:/CustomerChurnModel/Production` for the API.

---

## 3. Run the Streamlit UI

The Streamlit app is a thin UI layer over the FastAPI API.

With the API running on `http://localhost:8000`, start the UI:

```bash
streamlit run ui/app.py
```

Then open the URL that Streamlit prints (typically `http://localhost:8501`).

The UI lets you:

- Enter customer attributes (tenure, monthly charges, contract type, etc.)
- Click **“Predict”** to send a request to the FastAPI `/predict` endpoint
- See the estimated churn probability
- Inspect an approximate feature importance chart and textual explanation of what
  tends to drive churn in this synthetic dataset
- See which MLflow experiment and model URI are currently configured

You can change the API base URL from the sidebar if needed.

---

## 4. Tests and Lint

### Run tests locally

```bash
python -m pytest -q
```

This runs the same tests as CI and currently exercises:

- `/health`
- `/predict` (ensures it returns a probability between 0 and 1)
- `/stats` and `/recent` (prediction logging and summaries)
- `/metrics` (Prometheus scrape endpoint)
- Drift report generation on a small synthetic sample

You can also run without `-q` for more verbose output:

```bash
pytest
```

### Run lint locally (ruff)

Install dependencies (including `ruff`) if you haven't already:

```bash
pip install -r requirements.txt
```

Then run ruff checks against the codebase:

```bash
ruff check .
```

To automatically format the code, run:

```bash
ruff format .
```

If you want to verify formatting without changing files, use:

```bash
ruff format --check .
```

CI runs `ruff check .` (lint-only). It's good practice to run both lint and formatting locally before pushing.

### Optional: quick syntax check

A quick syntax check of all Python files (also run in CI):

```bash
python -m compileall .
```

---

## 5. Typical Local Workflow (without Docker)

1. **Train a model** (or retrain when you change features/hyperparameters):

   ```bash
   python -m training.train
   ```

2. **Start an MLflow server** (optional but recommended for registry features):

   ```bash
   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000  # PowerShell: $env:MLFLOW_TRACKING_URI='http://127.0.0.1:5000'
   mlflow server \
     --backend-store-uri sqlite:///mlruns.db \
     --default-artifact-root ./mlartifacts \
     --host 127.0.0.1 \
     --port 5000
   ```

3. **Start the API** (serves the latest model):

   ```bash
   uvicorn api.main:app --reload
   ```

4. **Start the UI** (calls the API):

   ```bash
   streamlit run ui/app.py
   ```

5. **Run tests** before committing:

   ```bash
   pytest
   ```

---

## 6. Run the end-to-end demo with Docker

This repository is Dockerized so you can run the full stack (MLflow + API + UI)
with a single command.

### Build and start all services

From the project root:

```bash
docker compose up --build
```

This starts:

- `mlflow` – MLflow tracking server with a SQLite backend store (`mlruns.db`)
  and local artifact root (`mlartifacts/`)
- `api` – FastAPI service, configured to talk to the `mlflow` service
- `ui` – Streamlit app that talks to the `api` service

Once everything is healthy, you can visit:

- MLflow UI: http://localhost:5000
- FastAPI docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

### Run a one-off training job in Docker

To train and register a new model against the same MLflow server used by
`docker-compose`:

```bash
docker compose run --rm trainer
```

This will:

- Generate synthetic churn data
- Train the XGBoost model
- Log metrics and artifacts to MLflow
- Attempt to register/alias the model as `CustomerChurnModel` (e.g. `champion`)

### Stop and clean up

To stop the running services:

```bash
docker compose down
```

To remove containers, networks, and anonymous volumes:

```bash
docker compose down -v
```

Persisted MLflow data is stored in:

- `mlruns.db` – SQLite database for the tracking backend
- `mlartifacts/` – local directory used as the artifact store

---

## 7. Observability & Monitoring

This starter includes basic observability features so you can inspect how the
model behaves over time and under load.

### Prediction logging and latency

Every successful or failed `/predict` call is logged to a lightweight SQLite
database (by default `logs/predictions.db`). For each request the API stores:

- `timestamp` – when the prediction was made (UTC)
- `request_payload` – the full feature payload sent by the client
- `churn_probability` – model output (or `null` on failure)
- `model_uri` – value of `CHURN_MODEL_URI` used by the service
- `latency_ms` – end-to-end processing time for `/predict`
- `status` – `"success"` or `"fail"`
- `error_message` – error summary for failed requests

Latency is also exposed per-request via the `X-Model-Latency-ms` HTTP response
header for `/predict`.

Two convenience endpoints make it easy to inspect recent traffic:

- `GET /stats?limit=100` – summary over the last *N* requests:
  - `count`, `success_rate`, `latency_p50_ms`, `latency_p95_ms`,
    `latency_avg_ms`, `avg_churn_probability`, `last_model_uri`
- `GET /recent?limit=20` – raw log records (most recent first), including the
  full payload for each request

These are intended for local monitoring and demos; no redaction or
authentication is applied.

### Prometheus /metrics

The FastAPI app is instrumented with
`prometheus-fastapi-instrumentator` and exposes a `/metrics` endpoint suitable
for scraping by Prometheus:

- Endpoint: `GET http://localhost:8000/metrics`
- Includes standard HTTP metrics such as request counts, durations, and status
  codes per path and method.

A minimal Prometheus scrape configuration might look like:

```yaml
scrape_configs:
  - job_name: "churn-api"
    static_configs:
      - targets: ["host.docker.internal:8000"]  # or "localhost:8000"
```

You can disable metrics instrumentation entirely by setting
`PROMETHEUS_ENABLED=false` before starting the API.

### Drift monitoring with Evidently

Data drift occurs when the distribution of incoming data changes compared to
what the model saw during training. This can erode model performance over time
even if the code and model weights do not change.

This project includes a simple drift report based on **Evidently**:

- Reference data:
  - The training dataset is materialized to `data/churn_reference.csv`
    (auto-generated if missing).
- Current data:
  - The most recent prediction inputs from the SQLite log
    (`logs/predictions.db`), using a configurable window (`DRIFT_WINDOW`).

Under the hood, the monitoring script uses the Evidently v0.7+ API:

```python
from evidently import Report
from evidently.presets import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=reference_df, current_data=current_df)
snapshot.save_html("drift_report.html")
snapshot.save_json("drift_report.json")
```

To generate a drift report locally:

1. Ensure you have some prediction logs (for example, by exercising the UI or
   calling `/predict` directly).
2. From the project root, run:

   ```bash
   python -m monitoring.generate_drift_report
   ```

3. Outputs (by default):

   - HTML report: `reports/drift_report.html`
   - JSON snapshot: `reports/drift_report.json`

Open `reports/drift_report.html` in a browser to view Evidently's diagnostics,
including feature-level drift scores and summary charts.

You can customize behavior via environment variables:

- `DRIFT_WINDOW` – number of most recent predictions to treat as \"current\" data
- `REPORT_OUTPUT_DIR` – where to write the HTML/JSON artifacts
- `REFERENCE_DATA_PATH` – path to the reference dataset used during training

### Load testing with Locust

To exercise the API under concurrent load, you can use **Locust**:

1. Install dev dependencies (including Locust):

   ```bash
   pip install -r requirements-dev.txt
   ```

2. Start the API (and MLflow if desired) as usual, e.g.:

   ```bash
   uvicorn api.main:app --reload
   ```

3. In a separate terminal, run Locust from the project root:

   ```bash
   locust -f loadtest/locustfile.py --host http://localhost:8000
   ```

4. Open the Locust web UI (typically `http://localhost:8089`), set the number
   of users and spawn rate, and start the test.

The default Locust user behavior:

- Periodically calls `GET /health` to verify service availability
- Sends randomized customer payloads to `POST /predict`

This gives you a quick sense of API throughput, latency distribution, and error
rates, especially when used together with `/metrics` and the `/stats` endpoint.

---

## Configuration

These environment variables control observability and monitoring behaviour:

| Variable              | Default                             | Description                                                                 |
|-----------------------|-------------------------------------|-----------------------------------------------------------------------------|
| `MLFLOW_TRACKING_URI` | `file:./mlruns`                     | MLflow tracking server used by training and the API.                        |
| `CHURN_MODEL_URI`     | `models:/CustomerChurnModel@champion` | MLflow model URI loaded by the API.                                        |
| `LOG_DB_PATH`         | `logs/predictions.db`              | SQLite database used to store `/predict` logs.                              |
| `DRIFT_WINDOW`        | `500`                               | Number of most recent prediction logs used as \"current\" data for drift.    |
| `REFERENCE_DATA_PATH` | `data/churn_reference.csv`         | Path to the reference dataset used for drift analysis.                      |
| `REPORT_OUTPUT_DIR`   | `reports`                          | Output directory for drift reports (`drift_report.html` / `.json`).         |
| `PROMETHEUS_ENABLED`  | `true`                             | Enable Prometheus instrumentation and the `/metrics` endpoint when `true`.  |

You can override any of these by setting the corresponding environment variable
before starting the API or running the monitoring scripts.

---

## 8. Notes and Next Steps

- The dataset is synthetic but structured to resemble telecom churn patterns and is entirely local.
- The MLflow integration is minimal by design and can be extended with:
  - Custom tracking URIs
  - Model registry
  - Multiple experiments / runs comparison
- The FastAPI service loads its model from MLflow using `mlflow.pyfunc.load_model(CHURN_MODEL_URI)` and,
  by default, targets the `champion` alias (`models:/CustomerChurnModel@champion`).  
  You can evolve this to:
  - Pin to a specific MLflow run or model version
  - Use different aliases (for example, `staging` vs `champion`)
  - Integrate more sophisticated deployment flows

This starter is intended as a foundation for a more complete MLOps pipeline—feel free to extend it with CI, Docker, cloud storage, and orchestration as needed.
