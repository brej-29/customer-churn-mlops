"""Faithfulness evaluation for SHAP-grounded LLM explanations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from churn.genai.explainer import ChurnExplanation, Driver

# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class FaithfulnessResult:
    faithful: bool
    unsupported_factors: list[str]


@dataclass
class FaithfulnessViolation:
    sample_idx: int
    unsupported_factors: list[str]


@dataclass
class FaithfulnessEvalResult:
    n_samples: int
    n_faithful: int
    faithfulness_score: float
    violations: list[FaithfulnessViolation] = field(default_factory=list)


# ── String normalization ───────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, split camelCase, keep alphanumeric and spaces."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _factor_matches_driver(key_factor: str, driver: Driver) -> bool:
    """Return True if the key_factor text is grounded in the given driver.

    Two-pass lightweight string check:
      1. Normalized driver feature name appears as substring in the key factor.
      2. Any long word (>7 chars) from the driver feature name appears in the
         key factor — handles multi-word features where only part is cited.

    Minimum word length of 8 prevents common short words ("service", "support")
    from creating false-positive matches against unrelated factors.

    For stronger grounding use embedding similarity or an LLM judge.
    """
    kf_norm = _normalize(key_factor)
    feat_norm = _normalize(driver.feature)

    if feat_norm in kf_norm:
        return True

    feat_words = [w for w in feat_norm.split() if len(w) > 7]
    return bool(feat_words) and any(word in kf_norm for word in feat_words)


# ── Per-explanation faithfulness check ────────────────────────────────────────

def check_faithfulness(
    explanation: ChurnExplanation,
    drivers: list[Driver],
) -> FaithfulnessResult:
    """Verify that every key_factor references a real SHAP driver.

    Each entry in explanation.key_factors is matched against the provided
    Driver objects (the actual SHAP results) using normalized substring and
    keyword overlap against the driver feature names. Factors with no match
    are flagged as unsupported — potentially hallucinated.

    Returns FaithfulnessResult with faithful=True only when every key_factor
    is matched by at least one driver.
    """
    unsupported: list[str] = []
    for factor in explanation.key_factors:
        if not any(_factor_matches_driver(factor, d) for d in drivers):
            unsupported.append(factor)
    return FaithfulnessResult(
        faithful=len(unsupported) == 0,
        unsupported_factors=unsupported,
    )


# ── Aggregate faithfulness eval ────────────────────────────────────────────────

def run_faithfulness_eval(
    n_samples: int = 30,
    sample_frac: float | None = None,
    log_to_mlflow: bool = False,
    top_k: int = 5,
    random_state: int = 42,
    _explain_fn: Callable | None = None,
    _drivers_fn: Callable | None = None,
    _test_data: Any = None,
) -> FaithfulnessEvalResult:
    """Sample held-out customers and measure the fraction of faithful explanations.

    Parameters
    ----------
    n_samples :
        Number of customers to evaluate from the test split.
    sample_frac :
        Alternative to n_samples: fraction of the test set.
    log_to_mlflow :
        Log faithfulness_score as an MLflow metric when True.
    top_k :
        SHAP driver depth.
    random_state :
        Seed for reproducible sampling.
    _explain_fn :
        Injection point: callable(features, top_k=...) -> (ChurnExplanation, meta).
        Defaults to explain_prediction. Inject a mock in tests to avoid LLM calls.
    _drivers_fn :
        Injection point: callable(features, top_k=...) -> DriverResult.
        Defaults to per_prediction_drivers.
    _test_data :
        Inject a pd.DataFrame for offline tests; skips get_splits() when provided.

    Returns
    -------
    FaithfulnessEvalResult with faithfulness_score in [0, 1], n_faithful,
    n_samples evaluated, and a violations list.
    """
    from churn.genai.explainer import explain_prediction, per_prediction_drivers

    if _explain_fn is None:
        _explain_fn = explain_prediction
    if _drivers_fn is None:
        _drivers_fn = per_prediction_drivers

    if _test_data is not None:
        sampled = _test_data
    else:
        from churn.data import get_splits

        _, X_test, _, _ = get_splits()
        if sample_frac is not None:
            sampled = X_test.sample(frac=sample_frac, random_state=random_state)
        else:
            n = min(n_samples, len(X_test))
            sampled = X_test.sample(n=n, random_state=random_state)

    violations: list[FaithfulnessViolation] = []
    n_faithful = 0
    n_evaluated = 0

    for idx, (_, row) in enumerate(sampled.iterrows()):
        features = row.to_dict()
        try:
            dr = _drivers_fn(features, top_k=top_k)
            expl, _ = _explain_fn(features, top_k=top_k)
            result = check_faithfulness(expl, dr.drivers)
            n_evaluated += 1
            if result.faithful:
                n_faithful += 1
            else:
                violations.append(
                    FaithfulnessViolation(
                        sample_idx=idx,
                        unsupported_factors=result.unsupported_factors,
                    )
                )
        except Exception:
            continue

    score = n_faithful / n_evaluated if n_evaluated > 0 else 0.0

    if log_to_mlflow:
        try:
            import mlflow  # noqa: PLC0415

            mlflow.log_metric("faithfulness_score", score)
            mlflow.log_metric("n_faithful", n_faithful)
            mlflow.log_metric("n_samples", n_evaluated)
        except Exception:
            pass

    return FaithfulnessEvalResult(
        n_samples=n_evaluated,
        n_faithful=n_faithful,
        faithfulness_score=score,
        violations=violations,
    )
