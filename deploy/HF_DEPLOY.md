# Hugging Face Docker Spaces — Deploy Runbook

**Project**: customer-churn-mlops · Branch: `tier3-deployment`
**Approach**: Each HF Space is its own git repo. Push the entire GitHub repo there with the
relevant Dockerfile renamed to `Dockerfile` (HF Docker Spaces always build from `Dockerfile`
at the repo root). The main GitHub repo keeps `Dockerfile.api` and `Dockerfile.ui` as-is.

Deploy the **API Space first** — you need its public URL before setting the UI Space variable.

---

## Prerequisites

```bash
# Install the HF CLI (one-time)
pip install huggingface_hub

# Log in with your HF write token
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens (type: write)
```

Set a shell variable for your HF username (used in all commands below):

```bash
HF_USER=<YOUR_HF_USERNAME>   # e.g.  HF_USER=brej-29
```

---

## Step 1 — Create the HF Spaces (web UI, one-time)

1. Go to https://huggingface.co/new-space
2. **API Space**:
   - Space name: `churn-api`
   - SDK: **Docker**
   - Visibility: Public
   - Create Space → you now have `https://huggingface.co/spaces/${HF_USER}/churn-api`
3. Repeat for **UI Space**:
   - Space name: `churn-ui`
   - SDK: **Docker**
   - Visibility: Public

---

## Step 2 — Set API Space Secrets

In the HF Space panel for `churn-api`: **Settings → Repository secrets → New secret**

| Secret name | Value |
|---|---|
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/<dagshub-user>/customer-churn-mlops.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | Your DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | Your DagsHub access token |
| `GEMINI_API_KEY` | _(optional)_ Your Gemini API key — enables `/explain` |
| `GROQ_API_KEY` | _(optional)_ Your Groq API key — Groq LLM fallback |

> **No secrets go in any committed file.** The API reads all of these from env at runtime.

---

## Step 3 — Deploy the API Space

This creates a temp directory, copies the GitHub repo, renames `Dockerfile.api` → `Dockerfile`,
substitutes the Space-specific README, and pushes to HF.

```bash
# From the root of your local customer-churn-mlops checkout (tier3-deployment branch)
REPO_ROOT=$(pwd)   # must be the project root

# Clone the (empty) HF Space repo to a temp location
TMPDIR=$(mktemp -d)
git clone "https://huggingface.co/spaces/${HF_USER}/churn-api" "$TMPDIR/churn-api"
cd "$TMPDIR/churn-api"

# Copy the full project (exclude git, venv, runtime artifacts)
rsync -a --exclude='.git' --exclude='.venv' --exclude='mlruns' \
         --exclude='logs' --exclude='*.pyc' --exclude='__pycache__' \
         --exclude='.env' \
         "$REPO_ROOT"/ .

# HF Docker Spaces require a file named exactly 'Dockerfile' at the repo root
cp Dockerfile.api Dockerfile

# Use the Space-specific README (YAML header + docs)
cp deploy/api/README.md README.md

# --- Optional: bake in the training CSV for full /explain support ---
# /predict works without the CSV. /explain needs telco_churn.csv to
# compute SHAP drivers at runtime. To enable it:
#
#   cp "$REPO_ROOT/data/raw/telco_churn.csv" data/raw/telco_churn.csv
#
# Then add this line to the Dockerfile just before USER churn:
#   COPY data/raw/telco_churn.csv data/raw/telco_churn.csv
#
# The CSV is ~2.5 MB and is NOT sensitive — safe to include in the Space.
# -------------------------------------------------------------------

git add .
git commit -m "deploy: customer-churn-mlops API Space (tier3-deployment)"

# Push (HF authenticates with the token from `huggingface-cli login`)
git push

cd "$REPO_ROOT"
rm -rf "$TMPDIR/churn-api"
```

**Expected**: HF triggers a Docker build (shows live build logs in the Space UI).
Build takes ~10–20 min (first time; installs CPU-only torch + sentence-transformers).
The image is ~4.9 GB; HF caches layers so subsequent pushes are faster.

---

## Step 4 — Get the API Space public URL

HF Space URLs follow the pattern:

```
https://<owner>-<space-name>.hf.space
```

For your API Space:

```
API_SPACE_URL=https://${HF_USER}-churn-api.hf.space
```

Confirm it is live (wait for the build to finish):

```bash
curl "${API_SPACE_URL}/health"
# Expected: {"status":"ok","model_loaded":true,"model_version":"1"}
# (model_loaded: false means the MLflow secrets are missing or wrong)
```

Also check Swagger: open `${API_SPACE_URL}/docs` in a browser.

---

## Step 5 — Set UI Space Variable

In the HF Space panel for `churn-ui`: **Settings → Variables → New variable**

| Variable | Value |
|---|---|
| `API_BASE_URL` | `https://${HF_USER}-churn-api.hf.space` ← exact value from Step 4 |

> `API_BASE_URL` is a **Variable** (not a Secret) because the API URL is not sensitive.

---

## Step 6 — Deploy the UI Space

```bash
REPO_ROOT=$(pwd)   # must be the project root

TMPDIR=$(mktemp -d)
git clone "https://huggingface.co/spaces/${HF_USER}/churn-ui" "$TMPDIR/churn-ui"
cd "$TMPDIR/churn-ui"

rsync -a --exclude='.git' --exclude='.venv' --exclude='mlruns' \
         --exclude='logs' --exclude='*.pyc' --exclude='__pycache__' \
         --exclude='.env' \
         "$REPO_ROOT"/ .

# HF Docker Spaces require a file named exactly 'Dockerfile' at the repo root
cp Dockerfile.ui Dockerfile

# Use the Space-specific README
cp deploy/ui/README.md README.md

git add .
git commit -m "deploy: customer-churn-mlops UI Space (tier3-deployment)"
git push

cd "$REPO_ROOT"
rm -rf "$TMPDIR/churn-ui"
```

**Expected**: Build takes ~3–5 min (Streamlit + requests only, 830 MB image).
Open `https://${HF_USER}-churn-ui.hf.space` in a browser — the sidebar should show
`● Online | model v1` if the API Space is running and `API_BASE_URL` is set correctly.

---

## Step 7 — Update README.md with live URLs

Once both Spaces are confirmed live, edit [README.md](../README.md) lines 18–19:

```markdown
| **Interactive UI** | [🚀 Live demo](https://<HF_USER>-churn-ui.hf.space) |
| **API (Swagger)**  | [📡 Swagger](https://<HF_USER>-churn-api.hf.space/docs) |
```

---

## Free-Tier Realities

| Behaviour | Detail |
|---|---|
| **48 h sleep** | Free Spaces sleep after 48 h of inactivity. On wake, the container restarts and re-pulls `@champion` from DagsHub MLflow (~30 s cold start). |
| **Prediction log resets** | `logs/predictions.db` (SQLite) lives inside the container; it resets on restart. Acceptable for a demo. |
| **No GPU** | CPU-only torch is intentional — the sentence-transformers model and XGBoost inference both run fine on CPU. |
| **Storage** | HF free tier has a 50 GB Space storage limit. The API image is ~4.9 GB. |

---

## Re-deploy after code changes

```bash
# Re-clone → rsync → rename → push  (same steps as above)
# HF rebuilds only the changed layers. If only app code changed (no
# new deps), the venv layer is cached and the build takes ~2 min.
```

---

## Secrets reference (never committed)

| Where | Name | Set as |
|---|---|---|
| API HF Space | `MLFLOW_TRACKING_URI` | HF Secret |
| API HF Space | `MLFLOW_TRACKING_USERNAME` | HF Secret |
| API HF Space | `MLFLOW_TRACKING_PASSWORD` | HF Secret |
| API HF Space | `GEMINI_API_KEY` | HF Secret |
| API HF Space | `GROQ_API_KEY` | HF Secret |
| UI HF Space | `API_BASE_URL` | HF Variable (not sensitive) |
| GitHub Actions | `DAGSHUB_USER`, `DAGSHUB_TOKEN` | GitHub repository secret |
| GitHub Actions | `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | GitHub repository secret |
