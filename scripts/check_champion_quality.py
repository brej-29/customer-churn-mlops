"""Model-quality gate: load the registered champion and assert PR-AUC >= floor.

Run locally (with DagsHub creds in env):
    uv run python scripts/check_champion_quality.py

Run in CI via the model-quality-gate job (MLFLOW_* vars injected from secrets).
Exits 0 on pass, 1 on failure so the CI job fails the PR.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is importable when the script is run as a file
# (pytest adds "." via pythonpath; direct invocation does not).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlflow
import mlflow.sklearn
from sklearn.metrics import average_precision_score

from churn.config import settings
from churn.data import get_splits

CHAMPION_URI = "models:/customer-churn-xgboost@champion"
PR_AUC_FLOOR = 0.60


def main() -> int:
    uri = os.environ.get("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
    mlflow.set_tracking_uri(uri)

    print(f"Tracking URI : {uri}")
    print(f"Loading      : {CHAMPION_URI}")

    try:
        model = mlflow.sklearn.load_model(CHAMPION_URI)
    except Exception as exc:
        print(f"ERROR: could not load champion model — {exc}", file=sys.stderr)
        return 1

    _, X_test, _, y_test = get_splits()
    proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, proba)

    status = "PASS" if pr_auc >= PR_AUC_FLOOR else "FAIL"
    print(f"PR-AUC       : {pr_auc:.6f}  (floor={PR_AUC_FLOOR})  [{status}]")

    if pr_auc < PR_AUC_FLOOR:
        print(
            f"FAIL: champion PR-AUC {pr_auc:.4f} is below the floor {PR_AUC_FLOOR}",
            file=sys.stderr,
        )
        return 1

    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
