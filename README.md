# Customer Churn MLOps Starter

Production-style starter project for a telecom customer churn prediction pipeline using:

- **XGBoost** for modeling
- **MLflow** for experiment tracking
- **FastAPI** for serving predictions
- **Streamlit** for a simple UI

Everything runs locally and is intended to be easy to extend.

---

## Project Structure

```text
.
├── api/
│   ├── __init__.py
│   └── main.py           # FastAPI app (/health, /predict)
├── data/
│   └── .gitkeep          # Placeholder for raw/processed data
├── models/
│   └── .gitkeep          # Optional local model artifacts
├── training/
│   ├── __init__.py
│   ├── preprocess.py     # Synthetic data generation & train/test split
│   └── train.py          # XGBoost training + MLflow logging + Model Registry
├── ui/
│   └── app.py            # Streamlit UI that calls the FastAPI /predict endpoint
├── tests/
│   └── test_api.py       # Basic API tests
├── requirements.txt
├── LICENSE
└── README.md
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
- Console output showing the run-level model URI and a suggested serving URI such as `models:/CustomerChurnModel/Production`

### MLflow tracking URI and UI

Both training and the API look at these environment variables:

- `MLFLOW_TRACKING_URI`  
  - Where MLflow logs runs and (optionally) hosts the Model Registry  
  - Default: `file:./mlruns` (local filesystem tracking)
- `CHURN_MODEL_URI`  
  - The model URI that the API will load with `mlflow.pyfunc.load_model`  
  - Default: `models:/CustomerChurnModel/Production`

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
- `CHURN_MODEL_URI` – defaults to `models:/CustomerChurnModel/Production`

A typical setup using the local MLflow server from above:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export CHURN_MODEL_URI=models:/CustomerChurnModel/Production  # or a specific version, e.g. models:/CustomerChurnModel/1

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

## 5. Typical Local Workflow

1. **Train a model** (or retrain when you change features/hyperparameters):

   ```bash
   python -m training.train
   ```

2. **Start the API** (serves the latest model):

   ```bash
   uvicorn api.main:app --reload
   ```

3. **Start the UI** (calls the API):

   ```bash
   streamlit run ui/app.py
   ```

4. **Run tests** before committing:

   ```bash
   pytest
   ```

---

## 6. Notes and Next Steps

- The dataset is synthetic but structured to resemble telecom churn patterns and is entirely local.
- The MLflow integration is minimal by design and can be extended with:
  - Custom tracking URIs
  - Model registry
  - Multiple experiments / runs comparison
- The FastAPI service currently loads a single “latest” model from `models/xgboost_churn_model.json`.  
  You can evolve this to:
  - Load a model from a specific MLflow run
  - Pull models from a registry
  - Support versioned deployments

This starter is intended as a foundation for a more complete MLOps pipeline—feel free to extend it with CI, Docker, cloud storage, and orchestration as needed.
