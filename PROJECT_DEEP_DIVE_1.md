# PROJECT_DEEP_DIVE — Customer Churn MLOps Starter

## 1) TL;DR
- What it is: A local end-to-end **customer churn prediction** starter: train an XGBoost model, track/register it with MLflow, serve predictions via FastAPI, and demo it via a Streamlit UI.
- Who it’s for: ML / MLOps learners who want a small, “production-style” reference project that includes training + model registry + an inference API + a UI.
- Why it matters: It demonstrates a realistic loop—**train → track metrics → register/version → serve**—and the operational seams you’d discuss in interviews (versioning, schema validation, fallbacks, CI).
- Repo evidence:
  - `README.md` (project overview, local run flow)
  - `training/train.py` (`train`)
  - `training/preprocess.py` (`generate_synthetic_churn_data`, `get_train_test_data`)
  - `api/main.py` (`load_model`, `build_features`, `/predict`, `/health`)
  - `ui/app.py` (Streamlit UI → POST `/predict`)
  - `.github/workflows/ci.yml` (ruff + compileall + pytest)

## 2) What problem does it solve?
- Problem statement: Given a small set of customer attributes (tenure, monthly charges, contract type, etc.), estimate the probability that a customer will churn.
- Success criteria:
  - Training produces a binary classifier and logs **parameters + metrics** to MLflow.
  - The API returns a bounded probability `0.0 ≤ churn_probability ≤ 1.0` for valid inputs.
  - A “happy path” demo is possible locally: train → start API → use UI to predict.
- Non-goals:
  - Real telecom data ingestion and ETL (the dataset is synthetic and generated in code).
  - Production-grade authentication/authorization, rate limiting, and monitoring.
- Evidence:
  - `training/preprocess.py` (`TARGET_COLUMN = "churn"`, feature generation)
  - `training/train.py` (metrics logged: `accuracy`, `roc_auc`, `log_loss`)
  - `api/main.py` (`PredictResponse.churn_probability`, `/predict`)
  - `README.md` (“synthetic churn dataset”, local workflow)

## 3) How it works (user perspective)
- Primary workflow:
  - Train a model locally (`python -m training.train`) and (optionally) run an MLflow UI/server.
  - Start the API (`uvicorn api.main:app --reload`), which loads the model from MLflow on startup.
  - Start the UI (`streamlit run ui/app.py`), enter features, click “Predict”, and see the churn probability.
- Inputs/outputs:
  - Input: JSON body with 6 numeric fields: `tenure`, `monthly_charges`, `contract_type`, `has_internet`, `support_calls`, `is_senior`.
  - Output: JSON body `{ "churn_probability": <float> }`.
- Demo script (2–3 minutes):
  1. (Optional) Run MLflow UI (file-based): `mlflow ui`
  2. Train: `python -m training.train`
  3. Start API: `uvicorn api.main:app --reload`
  4. Start UI: `streamlit run ui/app.py`
  5. In the Streamlit page, click “Predict” and narrate: “UI → API → MLflow model → probability response”.
- Evidence:
  - `README.md` (commands and endpoints)
  - `ui/app.py` (`requests.post(... + "/predict")`)
  - `api/main.py` (`@app.post("/predict")`, `PredictRequest`, `PredictResponse`)

## 4) Tech stack (with evidence)
- Languages: Python.
- Frameworks:
  - FastAPI (inference API)
  - Streamlit (demo UI)
- Key libraries:
  - XGBoost (model training)
  - scikit-learn (train/test split + metrics)
  - MLflow (experiment tracking + model registry integration + model loading)
  - pandas/numpy (data generation, feature frame creation, numeric handling)
  - pytest (tests), ruff (lint)
- Data/storage:
  - Synthetic dataset generated at runtime (not persisted as a dataset file by default).
  - MLflow file-based tracking by default (`file:./mlruns`), with an optional SQLite-backed MLflow server described in docs.
  - Local MLflow artifact directories exist in the repo (`mlruns/` tracked; `mlartifacts/` and `mlruns.db` appear as local state in this checkout).
- Tooling (CI, lint, formatting):
  - GitHub Actions workflow runs `ruff check .`, `python -m compileall .`, and `python -m pytest -q` for Python 3.10 and 3.11.
- Evidence:
  - `requirements.txt` (dependencies)
  - `.github/workflows/ci.yml` (CI steps and Python versions)
  - `README.md` (MLflow tracking URI defaults and commands)
  - `mlruns/` (local MLflow file-store directory)

## 5) Repo map
- Top-level folders explained:
  - `api/`: FastAPI service that loads an MLflow model and serves `/health` + `/predict`.
  - `training/`: Synthetic data generation + train/test split + training and MLflow logging/registry.
  - `ui/`: Streamlit UI that calls the FastAPI `/predict` endpoint.
  - `tests/`: Pytest tests for API behavior.
  - `data/`: Placeholder directory for future raw/processed datasets (currently `.gitkeep`).
  - `models/`: Placeholder for local model artifacts (currently `.gitkeep`).
  - `mlruns/`: MLflow local tracking directory (file-store default).
  - `mlartifacts/`, `mlruns.db`: Local MLflow server artifact root and SQLite backend store (present in this working tree; not required by code, but referenced by docs).
- Key files to read first:
  - `README.md` (how to run + model registry explanation)
  - `training/preprocess.py` (features/label + synthetic data)
  - `training/train.py` (model, metrics, MLflow logging + registry alias)
  - `api/main.py` (inference schema, input validation, model loading, prediction)
  - `ui/app.py` (demo UI client)
  - `.github/workflows/ci.yml` (quality gates)
- Evidence:
  - `README.md` (project structure section)
  - `api/main.py`, `training/train.py`, `training/preprocess.py`, `ui/app.py`

## 6) Architecture
- Components:
  - Training job (`training/train.py`) creates data, trains XGBoost, logs to MLflow, and (if available) registers a model + updates an alias.
  - Model store / registry (MLflow): either file-based runs (`file:./mlruns`) or an MLflow server with a registry backend (example: SQLite + `./mlartifacts`).
  - Inference API (FastAPI): loads an MLflow model once at startup (lifespan) and serves predictions.
  - UI (Streamlit): captures feature inputs and calls the API.
- Data flow:
  - Training: synthetic generator → `train_test_split` → model fit → compute metrics → MLflow log params/metrics/model → optional registry + alias update.
  - Serving: API startup → `mlflow.pyfunc.load_model(CHURN_MODEL_URI)` → request → validate/coerce inputs → build 1-row DataFrame with expected dtypes → `model.predict(...)` → return probability.
- Diagram(s):
  - Component diagram:
    ```text
    +-------------------+          +----------------------+
    |  training/train.py|          |      ui/app.py       |
    | (train + log/reg) |          | (Streamlit UI)       |
    +---------+---------+          +----------+-----------+
              |                               |
              | MLflow tracking/registry      | HTTP POST /predict
              v                               v
    +-------------------+          +----------------------+
    |      MLflow       |<---------|     api/main.py      |
    |  runs + artifacts  |  load    | (FastAPI inference) |
    |  registry (opt.)   |  model   |  /health /predict   |
    +-------------------+          +----------------------+
    ```
  - Prediction request flow:
    ```text
    Streamlit UI -> POST /predict -> build_features() -> model.predict(df) -> churn_probability
                     (422 on bad ints)   (MLflow pyfunc)      (clipped 0..1)
    ```
- Evidence:
  - `training/train.py` (`mlflow.log_params`, `mlflow.log_metrics`, `mlflow.xgboost.log_model`, `mlflow.register_model`, `MlflowClient.set_registered_model_alias`)
  - `api/main.py` (`lifespan`, `load_model`, `build_features`, `coerce_to_int64`, `/predict`)
  - `ui/app.py` (API call to `/predict`)
  - `README.md` (MLflow server option, alias-based serving description)

## 7) Pipeline(s)
### 7.1 Local run
- Prereqs:
  - Python installed (CI runs on Python 3.10 and 3.11; other versions are unknown from repo).
  - Ability to install Python packages from `requirements.txt`.
- Setup steps:
  - Create a virtual environment, then `pip install -r requirements.txt`.
  - Decide how you want MLflow to run:
    - Simple file-based tracking (default): uses `file:./mlruns`
    - Optional MLflow server for Model Registry (example uses SQLite + `./mlartifacts`)
- Commands:
  - Train: `python -m training.train`
  - API: `uvicorn api.main:app --reload`
  - UI: `streamlit run ui/app.py`
  - Optional MLflow UI (file-based): `mlflow ui`
  - Optional MLflow server (registry-enabled, per docs): `mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000`
- Common errors & fixes:
  - API can’t load the model (serves fallback heuristic instead):
    - Confirm `CHURN_MODEL_URI` points to a valid MLflow model URI and that the tracking URI matches training.
    - If you are using file-based tracking, registry features may be unavailable; consider using the MLflow server option.
  - Input validation errors (HTTP 422):
    - Integer-like fields reject non-integer floats (e.g., `12.7` for `tenure`).
- Evidence:
  - `README.md` (setup + commands + MLflow server example + env vars)
  - `requirements.txt` (dependencies required to run)
  - `api/main.py` (`DEFAULT_TRACKING_URI`, fallback path when `model is None`, integer coercion/422)

### 7.2 Build / Test
- How to run tests:
  - `python -m pytest -q` (as in CI)
- Lint/format:
  - `ruff check .` (as in CI)
  - `ruff format .` (described in docs; CI currently runs lint only)
- Evidence:
  - `.github/workflows/ci.yml` (ruff check, compileall, pytest)
  - `pytest.ini` (test discovery and pythonpath)
  - `tests/test_api.py` (tests cover `/health` + `/predict` behavior)
  - `README.md` (local lint/test commands)

### 7.3 CI/CD (if present)
- What runs on PR/push:
  - Lint (ruff), syntax compile (`compileall`), and tests (pytest) on Ubuntu for Python 3.10 and 3.11.
- Deploy steps:
  - Unknown from repo (no deployment workflow, Dockerfile, or infrastructure definitions found).
- Evidence:
  - `.github/workflows/ci.yml`

## 8) ML Deep Dive (ONLY if ML exists)
- Task type: Binary classification (predict `churn` ∈ {0,1}) with a probability-like output.
- Dataset:
  - Where it comes from: Generated synthetically at training time (no external download).
  - Schema/labels/features:
    - Features: `tenure`, `monthly_charges`, `contract_type`, `has_internet`, `support_calls`, `is_senior`
    - Label: `churn`
  - Any preprocessing/cleaning steps: No file-based cleaning; generator produces numeric features and a probabilistic churn label via a logistic transformation.
  - Train/val/test split approach: `train_test_split(..., stratify=y, test_size=0.2, random_state=42)` (train/test only; no explicit validation split).
- Model & training:
  - Algorithm: `xgboost.XGBClassifier` with `objective="binary:logistic"` and `eval_metric="logloss"`.
  - Hyperparameters are set directly in code (e.g., `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, etc.).
  - Reproducibility hooks: fixed `random_state=42` in both data split and model; full determinism across hardware/threads is best inference (needs confirmation).
- Evaluation & metrics:
  - Metrics computed and logged: `accuracy`, `roc_auc`, `log_loss`.
  - Why these metrics make sense:
    - `roc_auc` evaluates ranking quality for churn risk (threshold-independent).
    - `log_loss` evaluates probability calibration/quality (important when returning a probability).
    - `accuracy` is an easy-to-explain baseline but is threshold- and class-balance-sensitive.
  - Implementation: metrics computed on test split from `get_train_test_data`.
- Inference:
  - Serving path: FastAPI loads an MLflow PyFunc model and calls `model.predict(...)` on a single-row pandas DataFrame.
  - Input schema handling:
    - API enforces integer semantics for specific columns by coercing numeric values and rejecting non-integer floats (422).
    - API explicitly casts dtypes to match what the model expects (`int64` for integer features and `float64` for `monthly_charges`).
- Artifacts/versioning:
  - Tracking: MLflow experiment name `customer-churn`.
  - Registry name: `CustomerChurnModel`.
  - Default serving strategy: alias-based URI (default `models:/CustomerChurnModel@champion`); training attempts to update the alias after registering a new version.
  - If registry/alias isn’t available, training catches MLflow exceptions and continues.
- Risks (leakage/bias):
  - Leakage: Low risk in current code because features are generated separately from the label; still a general risk if future features incorporate post-churn information (unknown from repo).
  - Bias: `is_senior` is explicitly used as a feature; there is no fairness analysis/mitigation code (gap).
- Evidence:
  - `training/preprocess.py` (`FEATURE_COLUMNS`, `TARGET_COLUMN`, `generate_synthetic_churn_data`, `get_train_test_data`)
  - `training/train.py` (`XGBClassifier(...)`, metrics, `mlflow.*` logging/registration, alias update)
  - `api/main.py` (`FEATURE_COLUMNS` imported from training, dtype casting, coercion rules, `mlflow.pyfunc.load_model`)
  - `tests/test_api.py` (integer float acceptance and rejection behavior)

## 9) Deployment & operations
- Is it deployable today?
  - Partially: it runs as a local service via Uvicorn, with CI quality gates.
  - Not production-ready by default: no auth, no containerization/IaC, and limited observability.
- Docker/containers:
  - Unknown from repo (no `Dockerfile`/compose files found).
- Cloud/IaC:
  - Unknown from repo (no Terraform/Pulumi/CloudFormation/Kubernetes manifests found).
- Env vars:
  - `MLFLOW_TRACKING_URI`: where training logs and where the API looks for the MLflow tracking server/store (default `file:./mlruns`).
  - `CHURN_MODEL_URI`: the MLflow model URI the API loads (default `models:/CustomerChurnModel@champion`).
  - `CHURN_MODEL_ALIAS`: which alias training updates after registering a model version (default `champion`).
  - Note: The repo does not define a `.env` template; values and environment management are “best inference (needs confirmation)” for your target deploy environment.
- Observability:
  - API uses Python logging and logs exceptions on prediction failures; there’s no structured logging config, metrics endpoint, tracing, or external error reporting wiring.
- Evidence:
  - `api/main.py` (env vars, logging calls, model loading, error handling)
  - `training/train.py` (env vars and defaults, alias update)
  - `README.md` (env var descriptions and example setup)

## 10) Major decisions & rationale (evidence-based)

### Decision: Use a synthetic dataset generated in-code
- Evidence:
  - `training/preprocess.py` (`generate_synthetic_churn_data`)
  - `README.md` (“synthetic churn dataset”, “no external data download required”)
- Likely rationale: Remove external dependencies so the project is runnable anywhere and focused on the MLOps wiring rather than data acquisition. (Grounded: the generator exists and docs emphasize local-only setup.)
- Tradeoffs:
  - Pros: reproducible demo; no data licensing/PII risk; quick iterations.
  - Cons: not representative of real churn feature distributions; limits realism of evaluation and bias analysis.
- Alternatives:
  - Use an open telecom churn dataset checked into `data/` (or downloaded via script).
  - Add a data ingestion layer with schema validation (e.g., Great Expectations) and store data artifacts with versioning (DVC/lakeFS).
- Improvements now:
  - Add a `data/README.md` describing a real dataset option + expected schema; add a small sample CSV and/or a download script.

### Decision: Use XGBoost for churn classification
- Evidence:
  - `training/train.py` (`XGBClassifier(... objective="binary:logistic" ...)`)
  - `requirements.txt` (`xgboost`)
  - `README.md` (“XGBoost for modeling”)
- Likely rationale: XGBoost is a strong baseline for tabular churn problems: fast, high-performing, and explainable enough for interviews.
- Tradeoffs:
  - Pros: strong tabular performance; interpretable via feature importance/SHAP (not implemented here).
  - Cons: requires careful calibration/monitoring; can overfit without proper validation; GPU/CPU differences can affect training.
- Alternatives:
  - Logistic regression or random forest for a simpler baseline.
  - LightGBM or CatBoost for competing GBDT implementations.
- Improvements now:
  - Add cross-validation or a validation split; add calibration (Platt/isotonic) and log calibration curves.

### Decision: Use MLflow for tracking + (optional) Model Registry + alias-based serving
- Evidence:
  - `training/train.py` (`mlflow.set_experiment`, `mlflow.log_params`, `mlflow.log_metrics`, `mlflow.register_model`, `MlflowClient.set_registered_model_alias`)
  - `api/main.py` (`mlflow.pyfunc.load_model`, `CHURN_MODEL_URI` default `models:/...@champion`)
  - `README.md` (tracking URI defaults and alias-based “champion” description)
- Likely rationale: MLflow is a lightweight way to demonstrate experiment tracking and a production-like model promotion mechanism (alias or stage).
- Tradeoffs:
  - Pros: versioned artifacts + metrics; clean separation between training and serving; easy rollback by repointing alias.
  - Cons: requires consistent tracking URIs; registry features depend on MLflow backend; local file store lacks registry.
- Alternatives:
  - Save model artifacts to `models/` and load locally (simpler, less MLOps).
  - Use a different registry/tracker (Weights & Biases, SageMaker Model Registry, Vertex AI Model Registry).
- Improvements now:
  - Add documentation for “dev vs prod” tracking URIs; add a small script to list/select model versions for `CHURN_MODEL_URI`.

### Decision: Validate and coerce request inputs instead of requiring strict integer types in the API schema
- Evidence:
  - `api/main.py` (`PredictRequest` fields typed as `float`, `coerce_to_int64`, `build_features`)
  - `tests/test_api.py` (`test_predict_accepts_integer_floats`, `test_predict_rejects_non_integer_float`)
  - `README.md` (describes integer-float coercion behavior)
- Likely rationale: Real clients often send `12.0` for “integer” fields; coercion improves UX while still preventing invalid values like `12.7`.
- Tradeoffs:
  - Pros: robust API boundary; fewer 500s; clearer 422 errors.
  - Cons: more custom validation logic to maintain; schema says “float” even when semantic type is int.
- Alternatives:
  - Use strict `int` types in Pydantic and require exact integers from clients.
  - Define custom Pydantic validators to accept int-like floats while keeping int types.
- Improvements now:
  - Move coercion into Pydantic validators (clearer schema); add OpenAPI examples and error examples.

### Decision: Provide a fallback heuristic when the MLflow model isn’t available
- Evidence:
  - `api/main.py` (fallback branch when `model is None`)
  - `tests/test_api.py` (`client.app.state.model = None`)
  - `README.md` (mentions fallback heuristic for local dev/tests)
- Likely rationale: Keep the API and tests usable without standing up MLflow or training a model first.
- Tradeoffs:
  - Pros: “always-on” demo; tests don’t require MLflow.
  - Cons: Risk of accidentally running fallback in production; heuristic may mislead users.
- Alternatives:
  - Fail hard on startup if model load fails.
  - Expose a `/model_status` endpoint and make `/predict` return 503 when no model is loaded.
- Improvements now:
  - Add an explicit env flag like `ALLOW_FALLBACK=true/false` and default it to false for production.

## 11) Interview kit
### 11.1 Pitch scripts
- 30 sec: “This is a small churn prediction MLOps starter. It trains an XGBoost churn model on a synthetic dataset, logs metrics and a model artifact to MLflow, optionally registers it in the MLflow Model Registry, and serves predictions via a FastAPI endpoint. A Streamlit UI calls the API so you can demo the full train-to-serve loop locally.”
- 2 min: “I designed it to show the full lifecycle: data generation → train/test split → XGBoost training → evaluation with ROC AUC/log loss/accuracy → MLflow logging and optional model registry. For serving, a FastAPI app loads the model once at startup from an MLflow URI (defaulting to a ‘champion’ alias) and validates inputs by coercing integer-like values and rejecting invalid floats. There’s also a fallback heuristic so local demos/tests don’t depend on MLflow. CI enforces linting and tests on multiple Python versions.”
- 5 min: “Start with the architecture: a training module, an MLflow tracking/registry backend, an inference API, and a UI client. Walk through training code: synthetic data, split, XGBoost hyperparameters, metrics, MLflow model signature logging, and registry aliasing. Then serving: startup model load, schema alignment using pandas dtypes, error handling (422 vs 500), and tests. Close with tradeoffs and what I’d add next: real dataset ingestion, Docker/IaC, auth, monitoring, and better evaluation (CV, calibration).”

### 11.2 Questions & strong answers (project-specific)
1. What exactly is the ML task here?
   - Binary classification predicting `churn` and returning a probability-like score. Evidence: `training/preprocess.py` (`TARGET_COLUMN`), `training/train.py` (`objective="binary:logistic"`), `api/main.py` (`PredictResponse.churn_probability`).
2. How do you generate data, and why?
   - Data is generated synthetically with a logistic relationship to features so the project has no external data dependency. Evidence: `training/preprocess.py` (`generate_synthetic_churn_data`), `README.md` (synthetic/no-download).
3. Which features does the model use?
   - `tenure`, `monthly_charges`, `contract_type`, `has_internet`, `support_calls`, `is_senior`. Evidence: `training/preprocess.py` (`FEATURE_COLUMNS`), `api/main.py` (`FEATURE_COLUMNS = TRAINING_FEATURE_COLUMNS`).
4. How do you prevent training/serving skew (feature order/type mismatches)?
   - The API imports the canonical `FEATURE_COLUMNS` from training and builds a DataFrame in that order; it also casts dtypes explicitly before calling the MLflow model. Evidence: `api/main.py` (`FEATURE_COLUMNS`, `build_features`, `features.astype(...)`).
5. What metrics do you use and why?
   - Accuracy, ROC AUC, and log loss; AUC is threshold-independent, log loss evaluates probability quality, and accuracy is a simple baseline. Evidence: `training/train.py` (metric computations/logging).
6. How is the model version chosen for serving?
   - The API loads from `CHURN_MODEL_URI`, which defaults to `models:/CustomerChurnModel@champion`; training tries to update that alias to the latest registered version when the registry is available. Evidence: `api/main.py` (`DEFAULT_MODEL_URI`), `training/train.py` (alias update), `README.md` (serving picks model by alias).
7. What happens if MLflow or the model registry isn’t available?
   - Training catches MLflow registry failures and continues; the API may fail to load a model and will fall back to a heuristic probability function. Evidence: `training/train.py` (`except MlflowException` around register/alias), `api/main.py` (`load_model` exception handling, `if model is None` fallback).
8. How do you handle bad inputs at the API boundary?
   - Integer-like features accept `12` and `12.0` but reject `12.7` with HTTP 422; unexpected model prediction errors are 500. Evidence: `api/main.py` (`coerce_to_int64`, exception handling), `tests/test_api.py` (422 cases).
9. Why does the request schema use floats even for integer fields?
   - It’s a design choice to accept int-like floats from clients and coerce them safely while still rejecting invalid fractional values. Evidence: `api/main.py` (`PredictRequest` types, `coerce_to_int64`), `README.md` (input types note).
10. How do you test the system without depending on external services?
   - Tests set `app.state.model = None` to force the deterministic fallback path and validate endpoint behavior. Evidence: `tests/test_api.py` (model override), `api/main.py` (fallback branch).
11. Where are artifacts stored and how would you manage them in production?
   - By default, runs go to `file:./mlruns`. For a registry-backed setup, docs show an MLflow server with SQLite backend and `./mlartifacts` as artifact root. A production setup is unknown from repo, but would typically use a managed backend store + object storage for artifacts. Evidence: `README.md` (tracking URI options), `training/train.py` (`DEFAULT_TRACKING_URI`).
12. What are the biggest operational gaps right now?
   - No auth, no rate limiting, no containers/IaC, no metrics/tracing, and minimal model monitoring. Evidence: `api/main.py` (no auth middleware; basic logging only), repo root (no Docker/IaC files).
13. How would you add monitoring for model drift?
   - Best inference (needs confirmation): log input feature distributions and prediction distributions, compare to training baselines, and alert on shifts; also capture ground truth when available. To confirm feasibility, add logging hooks and a store for inference logs. Evidence to extend: `api/main.py` (central predict function is the hook point).
14. What would you change to make this reproducible for a team?
   - Pin dependency versions, add a lockfile, and document a consistent MLflow backend. Evidence: `requirements.txt` (unpinned), `README.md` (multiple MLflow modes).
15. How would you secure this API in production?
   - Add auth (e.g., JWT), input rate limiting, and secrets management; ensure `CHURN_MODEL_URI`/tracking URIs are controlled and audited. Production security is unknown from repo today. Evidence: `api/main.py` (no auth; env-driven configuration).

### 11.3 System design follow-ups
- Scaling:
  - Scale the API horizontally (stateless once model is loaded) and consider a dedicated model-serving layer if model load is heavy; add caching if repeated requests occur. Best inference (needs confirmation): ensure model loading is done once per worker (Uvicorn/Gunicorn config).
- Reliability:
  - Add explicit “model loaded” health checks and fail-fast startup for production; add retry/backoff around MLflow if used remotely; remove or gate fallback heuristic in prod.
- Security:
  - Add auth, rate limiting, request size limits, and dependency scanning; lock down MLflow endpoints and artifact access.
- Cost:
  - Use smaller instance sizes for low-throughput demos; choose backend storage that matches retention needs; avoid keeping large `mlruns/` in git.
- Evidence:
  - `api/main.py` (startup load and fallback behavior)
  - `README.md` (alias-based promotion/rollback concept)

### 11.4 Failure modes & debugging stories
- Model fails to load at startup → detect via logs/health + model status → fix by setting `MLFLOW_TRACKING_URI`/`CHURN_MODEL_URI` correctly and ensuring the registry/model version exists.
  - Evidence: `api/main.py` (`load_model` prints warning; model stored on `app.state.model`), `README.md` (env vars).
- Client sends non-integer floats for integer fields → detect via 422 responses and error detail → fix by client-side validation or clearer API schema/validators.
  - Evidence: `api/main.py` (`coerce_to_int64`), `tests/test_api.py` (422 assertion), `ui/app.py` (client-side integer check).
- MLflow registry operations fail during training → detect via printed MLflow exception message → fix by using a registry-enabled tracking URI (MLflow server + backend store) or skipping registry and serving by run URI.
  - Evidence: `training/train.py` (register/alias try/except), `README.md` (server option).
- Prediction output shape changes (e.g., model returns 2D probabilities) → detect via wrong value extraction → fix by robust handling of prediction array shapes (already partially handled).
  - Evidence: `api/main.py` (array shape handling for 1D/2D predictions).
- CI fails due to lint or type/schema assumptions → detect in GitHub Actions logs → fix by running `ruff check .` and `pytest` locally and updating code/tests consistently.
  - Evidence: `.github/workflows/ci.yml` (CI steps), `README.md` (lint/test commands).

### 11.5 “Built with AI” — how to answer confidently
- Suggested answer:
  - “I used AI as an accelerator, but I owned the engineering: I validated the pipeline end-to-end, ensured the API schema matched the training features, added tests for tricky input cases (like int-like floats), and set up CI to enforce linting and tests. I can explain the tradeoffs I made—like using MLflow aliases for rollout/rollback—and what I’d improve for production.”
- Evidence of ownership (tests, reviews, changes, understanding):
  - Concrete touchpoints to cite in conversation: input coercion and error handling in `api/main.py`, regression tests in `tests/test_api.py`, CI gates in `.github/workflows/ci.yml`, and the MLflow registry/alias logic in `training/train.py`.

### 11.6 Resume bullets (accurate)
- Built an end-to-end churn prediction starter (XGBoost + MLflow) with reproducible training, evaluation (ROC AUC/log loss/accuracy), and model artifact logging. Evidence: `training/train.py`.
- Implemented a FastAPI inference service that loads models from MLflow via a configurable model URI/alias and enforces input schema/dtypes to prevent serving-time errors. Evidence: `api/main.py`.
- Delivered a Streamlit UI that demos the ML service through a real API call path (UI → API → model → response). Evidence: `ui/app.py`.
- Added CI with linting (ruff), bytecode compilation, and pytest to enforce baseline quality across multiple Python versions. Evidence: `.github/workflows/ci.yml`.

### 11.7 Gaps to close (checklist)
- [ ] Pin dependency versions (and/or add a lockfile) for reproducibility. Evidence: `requirements.txt` (unpinned).
- [ ] Add Docker support (`Dockerfile`, compose) and document production runtime config. Unknown from repo today.
- [ ] Add API authentication and rate limiting for production. Unknown from repo today.
- [ ] Add a model status endpoint and consider removing/gating fallback heuristic in production. Evidence: `api/main.py` (fallback).
- [ ] Add richer evaluation (CV, calibration, threshold tuning) and log artifacts (confusion matrix, ROC curve). Evidence: `training/train.py` (single split; logs scalar metrics only).
- [ ] Add monitoring hooks for drift and model/version telemetry. Best inference (needs confirmation).

## 12) Appendices
- Glossary of key modules/functions:
  - `training/preprocess.py`
    - `FEATURE_COLUMNS`: canonical feature list used by both training and serving.
    - `generate_synthetic_churn_data(...)`: generates synthetic churn dataset.
    - `get_train_test_data(...)`: returns train/test split.
  - `training/train.py`
    - `train()`: trains XGBoost model; logs params/metrics/model to MLflow; attempts registry + alias update.
  - `api/main.py`
    - `load_model()`: loads MLflow model from `CHURN_MODEL_URI`.
    - `coerce_to_int64(...)`: enforces integer semantics for selected fields.
    - `build_features(...)`: builds a 1-row DataFrame with explicit dtypes.
    - `/predict`: main inference endpoint; fallback heuristic if model is missing.
  - `ui/app.py`: Streamlit UI; calls `/predict`.
- Commands reference:
  - `pip install -r requirements.txt`
  - `python -m training.train`
  - `uvicorn api.main:app --reload`
  - `streamlit run ui/app.py`
  - `python -m pytest -q`
  - `ruff check .`
  - `python -m compileall .`
- Open questions (Unknown from repo + how to confirm)
  - Is there an intended real dataset and schema beyond the synthetic generator?
    - How to confirm: check project issue tracker/notes; add/inspect a `data/` schema doc; look for planned ingestion scripts.
  - How should the service be deployed (containers, cloud, serverless)?
    - How to confirm: look for missing `Dockerfile`/IaC in future branches; ask the author; add deployment docs.
  - What is the expected SLO (latency/throughput) and how will inference be monitored?
    - How to confirm: add metrics/tracing, load test, and define latency/error budgets.
  - Should fallback heuristic ever run outside tests/dev?
    - How to confirm: define env-based policy and document it; add a runtime check in the API.

