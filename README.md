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
│   └── .gitkeep          # Trained model artifacts (saved by training)
├── training/
│   ├── __init__.py
│   ├── preprocess.py     # Synthetic data generation & train/test split
│   └── train.py          # XGBoost training + MLflow logging
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

## 1. Run Training (XGBoost + MLflow)

The training script:

- Generates a synthetic churn dataset (no external data download required)
- Trains an XGBoost classifier
- Logs hyperparameters, metrics, and the model artifact to **MLflow**
- Saves the latest trained model to `models/xgboost_churn_model.json`

From the project root:

```bash
python -m training.train
```

After it completes, you should see:

- An MLflow run recorded under experiment `churn_xgboost`
- A model file at `models/xgboost_churn_model.json`

### Optional: Browse MLflow UI

In a separate terminal (from the project root):

```bash
mlflow ui
```

By default, this starts a UI at `http://127.0.0.1:5000`, where you can see runs, metrics, and artifacts.

---

## 2. Run the FastAPI Service

The FastAPI app loads the latest XGBoost model from `models/xgboost_churn_model.json` if present.  
If no model is found, it falls back to a simple heuristic so the `/predict` endpoint still responds.

Start the API with **uvicorn**:

```bash
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

If you want to verify formatting without changing files (as CI does), use:

```bash
ruff format --check .
```

CI will fail if any of these checks fail, so it's good practice to run them before pushing.

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
