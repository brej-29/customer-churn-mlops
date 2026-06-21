# REPO AUDIT — Customer Churn MLOps

**Audit date:** 2026-06-17  
**Auditor:** Forensic read-only pass (no code modified, no pipeline executed)

---

## A) Repo Snapshot

### Directory tree (≤ 3 levels, build/cache dirs excluded)

```
d:/Project/customer-churn-mlops/
├── .dockerignore
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── LICENSE
├── README.md
├── PROJECT_DEEP_DIVE_1.md        ← untracked / not in git
├── api/
│   ├── Dockerfile
│   ├── __init__.py
│   └── main.py
├── data/
│   └── .gitkeep
├── docker-compose.yml
├── loadtest/
│   └── locustfile.py
├── mlruns.db                     ← untracked (generated locally)
├── mlartifacts/                  ← untracked (generated locally)
├── models/
│   └── .gitkeep
├── monitoring/
│   ├── __init__.py
│   └── generate_drift_report.py
├── notebooks/
│   ├── README.md
│   └── 01_eda.ipynb
├── pytest.ini
├── requirements-dev.txt
├── requirements.txt
├── tests/
│   ├── test_api.py
│   ├── test_metrics.py
│   ├── test_monitoring_drift.py
│   └── test_ui_utils.py
├── training/
│   ├── __init__.py
│   ├── preprocess.py
│   └── train.py
└── ui/
    ├── Dockerfile
    ├── app.py
    └── utils.py
```

**Tracked file count:** 31 files (excluding .git, `__pycache__`, mlartifacts, mlruns, .ruff_cache, .pytest_cache)

### Commit history

| Stat | Value |
|------|-------|
| Total commits | 26 |
| First commit | 2025-12-28 (eb4c3d6 "Initial commit") |
| Last commit | 2025-12-28 (19a8bbb "Docker file, Monitoring and Readme changes") |
| **All 26 commits land on a single calendar day.** | This is visible to any recruiter who runs `git log`. |

### Stack

- **Dependency manager:** pip + `requirements.txt` — **no lockfile of any kind** (no `uv.lock`, `poetry.lock`, `Pipfile.lock`, `requirements.txt` hash pins)
- **Python version constraint:** NONE — no `pyproject.toml`, no `.python-version`, no `python_requires`. CI uses 3.10/3.11 matrix but the repo itself imposes no constraint.
- **Key versions (floating):** `requirements.txt:1-15` — `xgboost`, `mlflow`, `fastapi`, `pydantic`, `streamlit`, `evidently` — all unpinned.

### One-line reproducibility verdict

> **MARGINAL.** A stranger can clone and `pip install -r requirements.txt && python -m training.train` and it will probably work today, but floating dependencies mean it can silently break tomorrow; there is no lockfile, no Python version pin, no committed data file, and no instructions for the prerequisite MLflow server step.

---

## B) Component Inventory Table

| Component | Status | Evidence (file:line or value) | Note |
|-----------|--------|-------------------------------|------|
| Reproducibility / clone-and-run | WEAK | `requirements.txt` all unpinned | Works today, no guarantee tomorrow |
| Dependency lockfile | MISSING | No `*.lock`, no `pyproject.toml` | — |
| Python version pin | MISSING | No `.python-version`, no `pyproject.toml` | CI uses 3.10/3.11 matrix only |
| Deterministic random seed | PRESENT & SOLID | `preprocess.py:17` seed=42, `preprocess.py:56` stratify+seed=42, `train.py:35` random_state=42 | All three sites seeded |
| Modular code structure | PRESENT & SOLID | `training/`, `api/`, `ui/`, `monitoring/` as importable packages | — |
| Committed notebook outputs | MISSING | `notebooks/01_eda.ipynb` — all 16 cells have 0 saved outputs | Notebook was never run and committed |
| Synthetic data generation | PRESENT & SOLID | `training/preprocess.py:17-49` | Fixed seed, deterministic |
| `has_internet` feature signal | WEAK | `preprocess.py:27-34` logit formula — `has_internet` absent | Pure noise feature; generates but never uses it |
| Data schema / validation layer | MISSING | No Pandera, no Great Expectations, no schema file | — |
| Data versioning (DVC) | MISSING | No `.dvc`, no DVC config | — |
| Dedicated validation split | MISSING | `preprocess.py:52-60` — only train/test, no val set | No hold-out for hyperparameter selection |
| sklearn Pipeline / ColumnTransformer | MISSING | No pipeline object anywhere | Raw DataFrame → XGBoost directly |
| Train/serve preprocessing skew | PRESENT & SOLID | `api/main.py:18` imports `FEATURE_COLUMNS` from training | Same definition; no reimplementation |
| Feature scaling / encoding / imputation | MISSING | No scaler, no encoder, no imputer | XGBoost tolerates this; no justification documented |
| XGBoost hyperparameters | PRESENT & SOLID | `train.py:27-37` (quoted in section C) | Hardcoded, not tuned |
| Hyperparameter tuning | MISSING | No Optuna, GridSearch, RandomSearch anywhere | — |
| Class imbalance handling | MISSING | No `scale_pos_weight`, no resampling, no `class_weight` | Class ratio unexamined |
| Cross-validation | MISSING | Single 80/20 split only | — |
| Early stopping / eval_set | MISSING | `train.py:56` — bare `model.fit(X_train, y_train)` | — |
| Model artifact format | PRESENT BUT WEAK | `train.py:77-82` — XGBoost Booster saved via `mlflow.xgboost.log_model(model.get_booster(), ...)` | Saves raw Booster, not sklearn wrapper; signature inferred from sklearn's `predict_proba` — mismatch |
| Metrics logged | PRESENT BUT WEAK | `train.py:65-71` — accuracy, roc_auc, log_loss only | Missing F1, precision, recall, PR-AUC, Brier, confusion matrix |
| Decision threshold | WEAK | `train.py:59` — hardcoded 0.5 | Unjustified; not optimized |
| Model calibration | MISSING | No calibration curve, no Brier score | — |
| SHAP / feature importance | MISSING | Not logged to MLflow; permutation importance is UI-only, not saved | — |
| Baseline comparison | PRESENT BUT WEAK | `notebooks/01_eda.ipynb:cell-15` — logistic regression present but not in training pipeline, notebook never run | Not usable as a gating criterion |
| MLflow tracking backend | PRESENT BUT WEAK | `train.py:16` `file:./mlruns`; compose uses SQLite `mlruns.db:9` | Local only; not portable |
| MLflow logged artifacts | PRESENT BUT WEAK | params, metrics, model, signature, input_example logged; no tags, no `mlflow.log_artifact("requirements.txt")`, no conda/pip env explicitly | Incomplete run metadata |
| Champion alias promotion | PRESENT BUT WEAK | `train.py:102-113` — unconditional; every run becomes champion | No gating criterion; any retrain silently overwrites champion |
| MLflow portability | MISSING | `mlruns.db` and `mlartifacts/` are local, untracked; no remote tracking URI | Pipeline breaks if cloned to a new machine |
| FastAPI endpoints | PRESENT & SOLID | `/health`, `/predict`, `/stats`, `/recent`, `/metrics` (Prometheus) | Well-structured |
| Model loading strategy | PRESENT BUT WEAK | `api/main.py:296-299` lifespan load at startup; falls back to heuristic if load fails | Heuristic (`api/main.py:353-371`) lives in prod endpoint — confusing for users and tests |
| Pydantic request model | PRESENT BUT WEAK | `api/main.py:308-318` all 6 features typed as `float`; coercion to int64 in `build_features()` | Features declared as float but conceptually int; confusing contract |
| Input range validation | MISSING | API accepts `tenure=-999`, `contract_type=99`; no range checks beyond Pydantic type validation | — |
| Batch prediction | MISSING | `/predict` is single-record only | — |
| Model version in response | MISSING | `PredictResponse` returns only `churn_probability`; version not surfaced | — |
| Prediction logging | PRESENT & SOLID | SQLite log at `logs/predictions.db`; `/stats` and `/recent` endpoints | Good observability primitive |
| Prometheus metrics | PRESENT & SOLID | `api/main.py:304-305` `prometheus-fastapi-instrumentator` | — |
| Streamlit demo | PRESENT & SOLID | `ui/app.py` — calls live API, configurable URL, "sample customer" button | — |
| Streamlit permutation importance | PRESENT BUT WEAK | `ui/app.py:50-93` — 64 samples × 7 features = 448 sequential API calls on page load | No seed for `np.random.permutation` (line 82); blocking and slow |
| Hardcoded MLflow URL in UI | PRESENT BUT WEAK | `ui/app.py:308` `"http://localhost:5000"` hardcoded in markdown display | Not broken but non-configurable display |
| Test count | PRESENT BUT WEAK | 4 files, 12 test functions | No model quality tests; all API tests use heuristic fallback |
| Tests hit real model | MISSING | `test_api.py:6` `client.app.state.model = None` — every test bypasses MLflow model | — |
| Model quality gate test | MISSING | No test asserts ROC-AUC > threshold | — |
| Input schema regression test | MISSING | No test adds a new feature without updating the API schema | — |
| Test coverage measurement | MISSING | No `--cov` flag in pytest.ini or CI step | — |
| CI workflow | PRESENT BUT WEAK | `.github/workflows/ci.yml` — lint + test only | No Docker build, no training run, no CD |
| Docker images | PRESENT BUT WEAK | `api/Dockerfile`, `ui/Dockerfile` — single-stage, python:3.11-slim | No multi-stage; copies entire repo into image |
| Docker trainer service | PRESENT BUT WEAK | `docker-compose.yml:64-74` reuses api Dockerfile | No separate trainer image |
| MLflow Docker service | PRESENT BUT WEAK | `docker-compose.yml:5-29` installs mlflow at runtime via pip | No pinned version; slow cold start |
| Live deployment | MISSING | No live URL, no cloud deploy config, entirely local | — |
| Config / env vars | PRESENT & SOLID | All tuneable via env vars with sane defaults | Well designed |
| Hardcoded relative paths | PRESENT BUT WEAK | `train.py:16`, `api/main.py:26`, `monitoring/generate_drift_report.py:17-19` — all relative | Break if cwd is not repo root |
| Secrets in repo | MISSING (GOOD) | `grep -r "password\|secret\|api_key"` — no hits | Clean |
| README badge placeholders | PRESENT BUT WEAK | `README.md:9` `OWNER/REPO` never replaced | Badge is broken |
| GenAI / LLM surface | MISSING | No LLM, no embeddings, no RAG, no agent code anywhere | Explicit gap for GenAI Engineer roles |

---

## C) Must-Capture Facts

### Exact feature list

From `training/preprocess.py:5-12`:

```python
FEATURE_COLUMNS = [
    "tenure",          # int, range [0, 71]
    "monthly_charges", # float, range [20.0, 120.0)
    "contract_type",   # int, range {0, 1, 2}  (0=month-to-month, 1=one-year, 2=two-year)
    "has_internet",    # int, range {0, 1}  ← NOT in churn logit formula; pure noise feature
    "support_calls",   # int, range [0, 10]
    "is_senior",       # int, range {0, 1}
]
TARGET_COLUMN = "churn"  # binary, binomial sample from logistic function
```

**Logit formula** (`preprocess.py:27-34`) — verbatim:

```python
logits = (
    -2.0
    + 0.03 * (monthly_charges - 70.0)
    - 0.04 * tenure
    + 0.4  * (contract_type == 0).astype(float)
    + 0.3  * (support_calls >= 3).astype(float)
    + 0.4  * is_senior
)
```

`has_internet` is absent from this formula.

### Exact XGBoost parameters

From `training/train.py:27-37`:

```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
```

No `scale_pos_weight`. No `use_label_encoder`. No `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda` set explicitly (XGBoost defaults apply). No `early_stopping_rounds` or `eval_set`.

### Exact metrics logged

From `training/train.py:61-71`:

```python
accuracy = accuracy_score(y_test, y_pred)          # threshold 0.5, hardcoded
roc_auc  = roc_auc_score(y_test, y_pred_proba)
loss     = log_loss(y_test, y_pred_proba)

mlflow.log_metrics({
    "accuracy": float(accuracy),
    "roc_auc":  float(roc_auc),
    "log_loss": float(loss),
})
```

**Not computed, not logged:** F1, precision, recall, PR-AUC, Brier score, confusion matrix, per-class metrics, calibration curve.

### Champion-promotion mechanism

From `training/train.py:99-113`:

```python
client = MlflowClient()
model_alias = os.getenv("CHURN_MODEL_ALIAS", DEFAULT_MODEL_ALIAS)   # default: "champion"

if hasattr(client, "set_registered_model_alias"):
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=model_alias,
        version=registered_model.version,
    )
```

**There is no gating criterion.** Every successful training run unconditionally sets the `champion` alias to the newest registered version. There is no comparison against the current champion's metrics, no human approval step, no staged rollout.

### Current deployment state

**Nothing is deployed.** The project runs locally or via `docker compose up`. There is no live URL, no cloud configuration, no Kubernetes manifest, no Heroku/Render/Railway config, no GitHub Actions deployment step. `mlruns.db` and `mlartifacts/` are local, untracked files that exist only on the author's machine.

---

## D) Shallow-Answer & Interview-Risk Flags

- **"Why `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`?"** — Hardcoded in `train.py:27-37` with no comment, no ablation, no hyperparameter search. The owner cannot justify these values.

- **"Did you tune any hyperparameters?"** — No. No Optuna, GridSearch, or RandomSearch anywhere in the codebase. The values appear to be defaults or rough guesses.

- **"How did you handle class imbalance?"** — Not handled. `scale_pos_weight` is absent. The owner cannot state the class imbalance ratio because it is never computed or logged.

- **"Why is the decision threshold 0.5?"** — `train.py:59` hardcodes `>= 0.5` with no justification. No threshold sweep, no precision-recall tradeoff analysis, no business cost function.

- **"What is `has_internet` actually doing in your model?"** — `has_internet` is in `FEATURE_COLUMNS` and passed to XGBoost, but the churn label (`preprocess.py:27-34`) does not depend on it at all. The model is being trained on a known-noise feature. The owner needs to explain this design choice or acknowledge it is unintentional.

- **"What happens if you retrain — does the API automatically serve a worse model?"** — Yes. `train.py:102-113` unconditionally sets the `champion` alias on every run. There is no metric-gated promotion. Any retrain replaces the champion regardless of performance.

- **"Why no cross-validation?"** — Single 80/20 split. With 1,000 samples, variance in reported metrics is meaningful. No k-fold, no stratified repeated CV.

- **"How do you know the model is better than a naive baseline?"** — The logistic regression baseline lives in `notebooks/01_eda.ipynb:cell-15` but was never run (zero saved outputs) and is never used as a gating criterion in the training pipeline.

- **"Do your tests actually test the model?"** — No. `test_api.py:6` sets `model = None` before every test, so all tests exercise the heuristic fallback, not the XGBoost model. The model itself is never tested.

- **"What does the `XGBClassifier.get_booster()` call at `train.py:78` imply?"** — The sklearn wrapper is used for fitting and `predict_proba` (for signature inference), but the raw XGBoost Booster is what gets saved via `mlflow.xgboost.log_model`. This creates a subtle inconsistency: the saved model's `pyfunc.predict()` calls `Booster.predict()` (probabilities for `binary:logistic`), while the signature was inferred from the sklearn `predict_proba` output. This works in practice but the owner should be able to explain the distinction.

- **"Why does the API `PredictRequest` declare integer fields as `float` (`api/main.py:308-313`)?"** — All 6 Pydantic fields are `float`. The integer semantics are enforced by a custom `coerce_to_int64` function (`main.py:67-88`). The owner must explain why not use `int` directly and what this buys.

- **"Is there any LLM or GenAI component?"** — No. None. This is a significant gap for any GenAI Engineer or ML Engineer role where modern systems integrate LLM-based explanations, RAG retrieval, or agentic workflows.

- **"What is `DRIFT_WINDOW=500` based on?"** — `monitoring/generate_drift_report.py:19` hardcodes 500 with no comment. The dataset only has 1,000 training samples. The owner cannot justify this window size.

- **"How do you handle model rollback?"** — Manually, by pointing the `champion` alias at an older version in the MLflow UI. There is no automated rollback trigger based on live metrics, and the drift monitor is not connected to the champion promotion flow in any way.

- **"Is this deployed anywhere?"** — No. The readme's CI badge points to `OWNER/REPO` (a placeholder, `README.md:9`), which is broken. Nothing is deployed. For a "portfolio project," absence of a live demo URL is a real recruiter friction point.

- **"How reproducible is your pipeline?"** — Moderately. The random seeds are fixed, but all dependencies are floating (`requirements.txt:1-15`), so the pipeline may produce different results after a pip upgrade.

- **"Why did the entire project happen in one day?"** — All 26 commits share the date 2025-12-28. This is visible in `git log` and signals to a recruiter that the project was scaffolded rapidly rather than organically developed.

---

## E) Gap Analysis Ranked by Impact

| Rank | Gap | Why It Matters | Effort |
|------|-----|----------------|--------|
| 1 | **No live deployment** | Portfolio projects with no demo URL carry much less weight. Recruiters cannot click anything. Free tiers on Render/Railway/Fly.io exist; the main blocker is the local mlruns dependency. | M |
| 2 | **No LLM / GenAI surface** | The target roles include GenAI Engineer. The repo has zero LLM code. Adding even a lightweight component (natural-language churn explanation via an LLM, RAG over customer history, an agent loop) fundamentally changes the story. | L |
| 3 | **Unconditional champion promotion** | Every retrain silently replaces the champion. A recruiter asking "how do you prevent a regression?" has no good answer. Requires adding a metric comparison before `set_registered_model_alias`. | S |
| 4 | **No hyperparameter tuning** | Hardcoded hyperparameters cannot be defended in a technical screen. Even a small Optuna/Hyperopt study with a committed `search_space.py` demonstrates the skill. | M |
| 5 | **No model quality tests** | Tests never touch the real model. A pytest fixture that trains on the synthetic data and asserts `roc_auc > 0.75` would make CI meaningful. Currently CI only verifies the heuristic fallback. | S |
| 6 | **Floating dependencies** | No lockfile means silent breakage. Converting to `uv` or adding `pip-compile`-generated `requirements.lock` is a 15-minute fix that signals engineering discipline. | S |
| 7 | **No real / public dataset** | Synthetic data is a known portfolio weakness. Replacing with a public dataset (Telco Customer Churn on Kaggle, UCI ML Repo) would eliminate the biggest "yeah but it's fake" objection. | M |
| 8 | **Thin metrics logged** | Only accuracy, ROC-AUC, log_loss. PR-AUC, F1, confusion matrix, Brier score, and SHAP values are expected in any ML Engineer evaluation discussion. | S |
| 9 | **`has_internet` is a noise feature** | The churn logit does not depend on `has_internet`. Either remove it or add a coefficient to make it meaningful. As-is, it is a latent bug waiting to be discovered in an interview. | S |
| 10 | **No cross-validation** | Single split on 1,000 samples gives high-variance metrics. k-fold or at minimum a repeated stratified split, plus standard deviation reporting, is baseline good practice. | S |
| 11 | **Class imbalance unexamined** | `scale_pos_weight` is not computed or set. The owner cannot state the imbalance ratio. Even logging it as a metric would allow a justification. | S |
| 12 | **No Python version pin / lockfile** | Complements gap 6. Adding `pyproject.toml` with `requires-python = ">=3.10,<3.12"` takes minutes. | S |
| 13 | **CI does not build Docker image** | The Dockerfile is untested in CI. A `docker build` step would catch image breakage before it reaches compose testing. | S |
| 14 | **Permutation importance is slow and unseeded** | `ui/app.py:50-93` makes 448 sequential API calls on page load. Should be pre-computed and cached, not recalculated live. `np.random.permutation` at line 82 has no seed — results change on every page refresh. | S |
| 15 | **All 26 commits in one day** | Signals a rapid scaffold rather than ongoing development. Even splitting into meaningful feature branches with real commit history would help perception. | M |

**Effort key:** S = Small (< 2 hours), M = Medium (half-day to a day), L = Large (multi-day)

---

## F) Executive Summary

The repository is a coherent, runnable MLOps scaffold that correctly wires together the major components — XGBoost training, MLflow experiment tracking, FastAPI serving, Streamlit UI, Evidently drift monitoring, Prometheus metrics, and Dockerized compose. The code quality is clean, the module structure is sensible, and the environment variable configuration is well-designed. However, nearly every component is implemented at the minimum viable level, and several contain specific flaws that would surface immediately under interview questioning. The most damaging issues are: (1) the champion promotion is unconditional — every retrain overwrites the champion with zero gating, making the MLflow integration misleading rather than production-like; (2) all tests bypass the real model via `model = None`, so CI does not actually validate that the ML component works; (3) there is no live deployment and the README CI badge is a broken placeholder, meaning the project has no "you can try it now" proof point; (4) there is no LLM or GenAI component whatsoever, which is a hard gap for GenAI Engineer roles; and (5) all 26 commits land on a single calendar day, which signals to a recruiter that this was scaffolded rather than built. Fixing the champion-gating logic, adding a model-quality test, pinning dependencies, and deploying to a free tier would convert this from a demo skeleton into a defensible portfolio piece. Adding even a minimal LLM-based explanation layer would make it relevant to GenAI roles.