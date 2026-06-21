"""Faithfulness evaluation for SHAP-grounded LLM explanations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from churn.genai.explainer import ChurnExplanation, Driver

# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class FactorResult:
    """Match details for a single key_factor from the LLM explanation."""

    factor: str
    match_type: Literal["exact", "semantic", "none"]
    matched_driver: str | None = None
    similarity: float | None = None  # cosine score for "semantic" / "none" tiers
    reason: str | None = None        # for "none": category from _categorize_unsupported


@dataclass
class FaithfulnessResult:
    faithful: bool
    unsupported_factors: list[str]
    factor_results: list[FactorResult] = field(default_factory=list)


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
    n_exact: int = 0
    n_semantic: int = 0
    n_unsupported: int = 0
    semantic_recoveries: list[dict] = field(default_factory=list)
    residual_categories: dict = field(default_factory=dict)
    random_state: int = 42


# ── Readable feature name helpers ──────────────────────────────────────────────

_ABBREV_MAP: dict[str, str] = {
    "avg": "average",
    "num": "number",
    "pct": "percent",
    "amt": "amount",
    "qty": "quantity",
}


def _readable_feature_name(feature: str) -> str:
    """Expand camelCase/snake_case feature name to lowercase spaced words.

    Applies abbreviation expansion so that feature identifiers used in model
    code map to the natural language words evaluators and LLMs use.

    Examples:
      MonthlyCharges    → "monthly charges"
      avg_monthly_spend → "average monthly spend"
      PaymentMethod     → "payment method"
      OnlineSecurity    → "online security"
      tenure            → "tenure"
    """
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", feature)
    text = re.sub(r"_", " ", text)
    text = text.lower().strip()
    return " ".join(_ABBREV_MAP.get(w, w) for w in text.split())


# ── Tier 1 — string/keyword matching ──────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, split camelCase, keep alphanumeric and spaces."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _factor_matches_driver(key_factor: str, driver: Driver) -> bool:
    """Tier 1: match key_factor against the driver's readable feature name.

    Uses _readable_feature_name (camelCase + snake_case + abbreviation expansion)
    so feature identifiers map to the natural language evaluators use.
    Example: avg_monthly_spend → "average monthly spend" enables Tier 1 to
    match "Average monthly spend" without a Tier 2 embedding call.

    Two-pass check:
      1. Readable feature name appears as substring in the normalized key factor.
      2. Any long word (>7 chars) from the feature name appears in the factor.
    """
    kf_norm = _normalize(key_factor)
    feat_norm = _readable_feature_name(driver.feature)

    if feat_norm in kf_norm:
        return True

    feat_words = [w for w in feat_norm.split() if len(w) > 7]
    return bool(feat_words) and any(word in kf_norm for word in feat_words)


# ── Tier 2 — embedding similarity ─────────────────────────────────────────────

def _driver_description(driver: Driver) -> str:
    """Human-readable embedding text for a driver.

    Uses the readable feature name (camelCase/snake_case/abbreviation expanded)
    and appends categorical values for semantic context. Numeric values are
    excluded — they dilute cosine similarity without adding semantic signal,
    which caused near-paraphrases like "High monthly charge" to fall below the
    0.5 threshold when the description included the raw charge amount.

    Examples:
      MonthlyCharges  / 89.50              → "monthly charges"
      PaymentMethod   / "Electronic check" → "payment method electronic check"
      avg_monthly_spend / 65.0             → "average monthly spend"
    """
    readable = _readable_feature_name(driver.feature)
    val = driver.value
    if isinstance(val, str):
        return f"{readable} {val.lower()}"
    return readable


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def _semantic_max_similarity(
    embedder: Callable,
    factor: str,
    drivers: list[Driver],
) -> tuple[float, int]:
    """Return (max cosine similarity, best driver index) for the factor vs drivers."""
    if not drivers:
        return 0.0, 0

    texts = [factor] + [_driver_description(d) for d in drivers]
    raw = np.array(embedder(texts), dtype=np.float32)
    vecs = _l2_normalize(raw)
    factor_vec = vecs[0]
    driver_vecs = vecs[1:]
    sims = [float(np.dot(factor_vec, dv)) for dv in driver_vecs]
    best_idx = int(np.argmax(sims))
    return sims[best_idx], best_idx


# ── Residual categorization ────────────────────────────────────────────────────

def _categorize_unsupported(factor: str, drivers: list[Driver]) -> str:
    """Heuristic label for a factor that passed neither Tier 1 nor Tier 2.

    This is a review aid, not ground truth. Categories:
      "likely-paraphrase" : shares a meaningful word (>3 chars) with a driver's
                            readable feature name — probably a rephrasing the
                            string/embedding checks narrowly missed.
      "value-narration"   : shares a meaningful word with a driver's value text —
                            the LLM described a value in business language rather
                            than citing the feature name (e.g. "autopay" for a
                            PaymentMethod driver whose value is "Electronic check").
      "other/ungrounded"  : no word overlap with any driver; may be hallucinated
                            or a RAG-playbook concept bleeding into key_factors.
    """
    kf_words = {w for w in _normalize(factor).split() if len(w) > 3}
    if not kf_words:
        return "other/ungrounded"

    for d in drivers:
        feat_words = set(_readable_feature_name(d.feature).split())
        if kf_words & feat_words:
            return "likely-paraphrase"

    for d in drivers:
        val_words = {w for w in _normalize(str(d.value)).split() if len(w) > 3}
        if kf_words & val_words:
            return "value-narration"

    return "other/ungrounded"


# ── Per-explanation faithfulness check ────────────────────────────────────────

def check_faithfulness(
    explanation: ChurnExplanation,
    drivers: list[Driver],
    embedder: Callable | None = None,
    sim_threshold: float = 0.5,
) -> FaithfulnessResult:
    """Two-tier faithfulness check: does every key_factor reference a real SHAP driver?

    Tier 1 — string match (zero compute):
        Normalized substring/keyword overlap against the driver's readable feature
        name (camelCase/snake_case/abbreviation expanded via _readable_feature_name).
        Assigns match_type "exact". Abbreviation expansion means avg_monthly_spend
        now matches "Average monthly spend" in Tier 1 rather than requiring Tier 2.

    Tier 2 — semantic similarity (runs only for Tier-1 failures):
        Embeds the key_factor and each driver's human-readable description
        (readable feature name + categorical value; numeric values excluded to
        avoid diluting similarity). Uses the injected embedder (default: the RAG
        sentence-transformers model from churn.genai.rag). Assigns match_type
        "semantic" when max cosine similarity >= sim_threshold (default 0.5).
        Excluding numeric values allows near-paraphrases like "High monthly charge"
        to recover against MonthlyCharges without raising the threshold.

    Tier 1+2 failures receive match_type "none" and are classified by
    _categorize_unsupported as "likely-paraphrase", "value-narration", or
    "other/ungrounded" (review aid only — not ground truth). Only "none" factors
    appear in unsupported_factors and count as real violations.

    Parameters
    ----------
    sim_threshold :
        Cosine similarity threshold for Tier 2 (default 0.5). A fully rigorous
        check would use an LLM judge over the raw SHAP values and customer features.
    """
    if embedder is None:
        try:
            from churn.genai.rag import _get_default_embedder

            embedder = _get_default_embedder()
        except Exception:
            embedder = None

    factor_results: list[FactorResult] = []
    unsupported: list[str] = []

    for factor in explanation.key_factors:
        # Tier 1: string/keyword match against readable feature names
        if any(_factor_matches_driver(factor, d) for d in drivers):
            factor_results.append(FactorResult(factor=factor, match_type="exact"))
            continue

        # Tier 2: embedding similarity for Tier-1 failures
        if embedder is not None:
            max_sim, best_idx = _semantic_max_similarity(embedder, factor, drivers)
            if max_sim >= sim_threshold:
                factor_results.append(
                    FactorResult(
                        factor=factor,
                        match_type="semantic",
                        matched_driver=drivers[best_idx].feature if drivers else None,
                        similarity=round(max_sim, 4),
                    )
                )
                continue
            else:
                reason = _categorize_unsupported(factor, drivers)
                factor_results.append(
                    FactorResult(
                        factor=factor,
                        match_type="none",
                        similarity=round(max_sim, 4),
                        reason=reason,
                    )
                )
        else:
            reason = _categorize_unsupported(factor, drivers)
            factor_results.append(FactorResult(factor=factor, match_type="none", reason=reason))

        unsupported.append(factor)

    return FaithfulnessResult(
        faithful=len(unsupported) == 0,
        unsupported_factors=unsupported,
        factor_results=factor_results,
    )


# ── Aggregate faithfulness eval ────────────────────────────────────────────────

def run_faithfulness_eval(
    n_samples: int = 30,
    sample_frac: float | None = None,
    log_to_mlflow: bool = False,
    top_k: int = 5,
    random_state: int = 42,
    sim_threshold: float = 0.5,
    _explain_fn: Callable | None = None,
    _drivers_fn: Callable | None = None,
    _test_data: Any = None,
    _embedder: Callable | None = None,
) -> FaithfulnessEvalResult:
    """Sample held-out customers and measure the fraction of faithful explanations.

    Parameters
    ----------
    n_samples : int
        Number of customers to evaluate from the test split.
    sample_frac : float | None
        Alternative: fraction of the test set to sample.
    log_to_mlflow : bool
        Log metrics as MLflow metrics when True.
    top_k : int
        SHAP driver depth.
    random_state : int
        Seed for reproducible customer sampling; stored in the returned result
        so the exact sample can be reproduced.
    sim_threshold : float
        Cosine similarity threshold for Tier 2 semantic matching (default 0.5).
    _explain_fn : Callable | None
        Injection point: callable(features, top_k=...) → (ChurnExplanation, meta).
        Defaults to explain_prediction. Inject a mock in tests to avoid LLM calls.
    _drivers_fn : Callable | None
        Injection point: callable(features, top_k=...) → DriverResult.
        Defaults to per_prediction_drivers.
    _test_data : Any
        Inject a pd.DataFrame for offline tests; skips get_splits() when provided.
    _embedder : Callable | None
        Injection point for the Tier 2 embedder. None uses the default RAG
        sentence-transformers model. Inject a fake callable in tests.

    Returns
    -------
    FaithfulnessEvalResult with:
      - faithfulness_score, n_faithful, n_samples, random_state
      - n_exact / n_semantic / n_unsupported: factor-level breakdown
      - semantic_recoveries: per-factor detail for all Tier 2 recoveries
        (factor text, matched driver feature, cosine similarity)
      - residual_categories: count by "likely-paraphrase" / "value-narration" /
        "other/ungrounded" — a review aid for threshold tuning and audit
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
    semantic_recoveries: list[dict] = []
    residual_categories: dict[str, int] = {
        "likely-paraphrase": 0,
        "value-narration": 0,
        "other/ungrounded": 0,
    }
    n_faithful = 0
    n_evaluated = 0
    n_exact = 0
    n_semantic = 0
    n_unsupported = 0

    for idx, (_, row) in enumerate(sampled.iterrows()):
        features = row.to_dict()
        try:
            dr = _drivers_fn(features, top_k=top_k)
            expl, _ = _explain_fn(features, top_k=top_k)
            result = check_faithfulness(
                expl, dr.drivers, embedder=_embedder, sim_threshold=sim_threshold
            )
            n_evaluated += 1

            for fr in result.factor_results:
                if fr.match_type == "exact":
                    n_exact += 1
                elif fr.match_type == "semantic":
                    n_semantic += 1
                    semantic_recoveries.append(
                        {
                            "sample_idx": idx,
                            "factor": fr.factor,
                            "matched_driver": fr.matched_driver,
                            "similarity": fr.similarity,
                        }
                    )
                else:
                    n_unsupported += 1
                    if fr.reason:
                        residual_categories[fr.reason] = (
                            residual_categories.get(fr.reason, 0) + 1
                        )

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
            mlflow.log_metric("n_exact", n_exact)
            mlflow.log_metric("n_semantic", n_semantic)
            mlflow.log_metric("n_unsupported", n_unsupported)
        except Exception:
            pass

    return FaithfulnessEvalResult(
        n_samples=n_evaluated,
        n_faithful=n_faithful,
        faithfulness_score=score,
        violations=violations,
        n_exact=n_exact,
        n_semantic=n_semantic,
        n_unsupported=n_unsupported,
        semantic_recoveries=semantic_recoveries,
        residual_categories=residual_categories,
        random_state=random_state,
    )
