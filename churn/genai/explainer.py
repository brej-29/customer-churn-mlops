"""Per-prediction SHAP driver computation and LLM-grounded churn explanation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from churn.config import settings
from churn.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, get_splits
from churn.explain import _get_feature_names, _parse_original_name
from churn.genai.client import complete_with_fallback
from churn.models import build_model_pipeline

_PARAMS_PATH = Path("reports/best_xgb_params.json")


# ── Schema ─────────────────────────────────────────────────────────────────────

class ChurnExplanation(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    summary: str
    key_factors: list[str]
    recommended_action: str
    citations: list[str] = Field(default_factory=list)  # playbook sources; empty = no RAG


@dataclass
class Driver:
    feature: str             # original feature name (OHE aggregated)
    value: Any               # customer's raw value for this feature
    shap_contribution: float # aggregated SHAP contribution
    direction: str           # "increases" | "decreases"


@dataclass
class DriverResult:
    probability: float
    drivers: list[Driver]


# ── Pipeline / explainer cache ─────────────────────────────────────────────────

_CACHE: dict[str, Any] = {}


def _build_pipeline_and_explainer() -> tuple:
    """Build and cache the uncalibrated tuned XGBoost pipeline + SHAP explainer.

    Reads tuned params from reports/best_xgb_params.json (written by tune_xgboost).
    Fitted once; subsequent calls return the cached objects.
    """
    if _CACHE:
        return _CACHE["pipeline"], _CACHE["explainer"], _CACHE["feature_names"]

    import shap
    from xgboost import XGBClassifier

    params = json.loads(_PARAMS_PATH.read_text())
    pipeline = build_model_pipeline(XGBClassifier(**params))

    X_train, _, y_train, _ = get_splits()
    pipeline.fit(X_train, y_train)

    feature_names = _get_feature_names(pipeline)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])

    _CACHE["pipeline"] = pipeline
    _CACHE["explainer"] = explainer
    _CACHE["feature_names"] = feature_names
    return pipeline, explainer, feature_names


# ── Per-prediction SHAP drivers ────────────────────────────────────────────────

def per_prediction_drivers(features: dict[str, Any], top_k: int = 5) -> DriverResult:
    """Compute churn probability and top signed SHAP drivers for one customer.

    Parameters
    ----------
    features :
        Dict of original Telco column names → values (before feature engineering).
        Must include all columns the ChurnFeatureEngineer expects.
    top_k :
        Number of drivers to return, sorted by |SHAP contribution| descending.
    """
    pipeline, explainer, feature_names = _build_pipeline_and_explainer()

    row = pd.DataFrame([features])
    # Match the dtype coercion applied by clean_telco so the fitted OHE
    # sees the same types it saw during training (SeniorCitizen is a string "0"/"1").
    for col in CATEGORICAL_FEATURES:
        if col in row.columns:
            row[col] = row[col].astype(str)
    for col in NUMERIC_FEATURES:
        if col in row.columns:
            row[col] = row[col].astype("float64")

    prob = float(pipeline.predict_proba(row)[0, 1])

    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(row)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    sv = shap_values[0]  # 1-D: one SHAP per transformed feature

    # Customer values after feature engineering (includes engineered cols like tenure_bucket)
    row_fe_dict = preprocessor.named_steps["fe"].transform(row).iloc[0].to_dict()

    # Aggregate SHAP contributions by original feature name
    agg: dict[str, float] = {}
    for fname, sval in zip(feature_names, sv):
        orig = _parse_original_name(fname)
        agg[orig] = agg.get(orig, 0.0) + float(sval)

    sorted_feats = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]

    drivers: list[Driver] = []
    for feat, shap_val in sorted_feats:
        raw_val = row_fe_dict.get(feat)
        if isinstance(raw_val, (np.integer, np.floating)):
            raw_val = raw_val.item()
        drivers.append(Driver(
            feature=feat,
            value=raw_val,
            shap_contribution=round(shap_val, 4),
            direction="increases" if shap_val > 0 else "decreases",
        ))
    return DriverResult(probability=prob, drivers=drivers)


# ── Driver value formatting ─────────────────────────────────────────────────────

_DOLLAR_FEATURES: frozenset[str] = frozenset(
    {"MonthlyCharges", "TotalCharges", "avg_monthly_spend"}
)


def _format_driver_value(feature: str, value: Any) -> str:
    """Return a human-readable, unit-annotated string for a driver's raw value."""
    if value is None:
        return "N/A"
    if feature == "tenure":
        try:
            return f"{int(round(float(value)))} months"
        except (TypeError, ValueError):
            return str(value)
    if feature in _DOLLAR_FEATURES:
        try:
            return f"${float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


# ── Prompt builders ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a customer retention analyst. Given SHAP-based churn risk drivers for one customer,
explain why they are at risk of churning.

STRICT RULES:
- key_factors MUST contain only short phrases that name or directly paraphrase one of the
  provided SHAP driver features. Every entry must map to a driver in the list below.
  Do NOT introduce payment habits (autopay, auto-pay), customer value scores, loyalty
  concepts, engagement metrics, or any term not explicitly listed as a SHAP driver.
  Retrieved playbook tactics are for recommended_action — never for key_factors.
- summary explains churn risk using the listed SHAP drivers only.
- recommended_action is one concrete retention action.
- citations must be an empty list [].

Respond with a JSON object with exactly these keys:
  risk_level        : "low" | "medium" | "high"
  summary           : 2-3 sentences explaining the main churn drivers for this specific customer
  key_factors       : list of short phrases — ONE per SHAP driver, directly referencing that
                      driver's feature name or a close paraphrase (e.g. "Month-to-month contract"
                      for Contract, "High monthly bill" for MonthlyCharges). No invented factors.
  recommended_action: one concrete, specific retention action
  citations         : empty list []

Thresholds: high when probability > 0.5, medium for 0.3-0.5, low below 0.3.\
"""

_SYSTEM_PROMPT_RAG = """\
You are a customer retention analyst. Given SHAP-based churn risk drivers and retrieved
retention playbook tactics, explain why this customer is at risk and recommend an action.

STRICT RULES:
- key_factors MUST contain only short phrases that name or directly paraphrase one of the
  provided SHAP driver features. Every entry must map to a driver in the list below.
  Do NOT introduce payment habits (autopay, auto-pay), customer value scores, loyalty
  concepts, engagement metrics, or any playbook tactic as a key factor.
  Playbook tactics belong ONLY in recommended_action.
- summary explains churn risk using the listed SHAP drivers only.
- recommended_action is where retrieved playbook tactics go; base it on the retrieved tactics.
- Cite the exact source filenames you used in the citations field.

Respond with a JSON object with exactly these keys:
  risk_level        : "low" | "medium" | "high"
  summary           : 2-3 sentences explaining the main churn drivers for this specific customer
  key_factors       : list of short phrases — ONE per SHAP driver, directly referencing that
                      driver's feature name or a close paraphrase. No playbook concepts here.
  recommended_action: concrete retention action grounded in the retrieved tactics
  citations         : list of source filenames used, e.g. ["contract_upgrade.md"]

Thresholds: high when probability > 0.5, medium for 0.3-0.5, low below 0.3.\
"""


def _build_rag_query(drivers: list[Driver]) -> str:
    """Build a semantic search query from the top SHAP drivers."""
    return " ".join(f"{d.feature} {d.value}" for d in drivers[:3])


def _build_user_message(
    prob: float,
    drivers: list[Driver],
    retrieved: list | None = None,
) -> str:
    lines = [
        f"Churn probability: {prob:.1%}",
        "",
        "Top SHAP risk drivers (positive = increases churn risk):",
    ]
    for i, d in enumerate(drivers, 1):
        sign = "+" if d.shap_contribution > 0 else ""
        formatted = _format_driver_value(d.feature, d.value)
        lines.append(
            f"  {i}. {d.feature}: customer value = {formatted}"
            f" | SHAP = {sign}{d.shap_contribution:.4f} ({d.direction} churn risk)"
        )

    lines += ["", "Units: tenure is in months; MonthlyCharges, TotalCharges, and avg_monthly_spend are USD amounts."]

    driver_names = ", ".join(d.feature for d in drivers)
    lines += [
        "",
        f"CONSTRAINT: key_factors must reference ONLY these driver features (or close paraphrases): {driver_names}.",
        "Do NOT add autopay, customer value, loyalty, long-term commitment, onboarding, or any term not in that list.",
    ]

    if retrieved:
        lines += ["", "Retrieved retention playbook tactics (ground your recommendation in these):"]
        for r in retrieved:
            lines += [
                "---",
                f"Source: {r.citation}",
                f"Title: {r.entry.title}",
                r.entry.text[:600].rstrip(),
                "",
            ]
        lines.append("List the source filenames you used in the citations field.")
        lines.append("Reminder: playbook concepts go in recommended_action, NOT in key_factors.")

    lines += ["", "Return JSON only."]
    return "\n".join(lines)


# ── Deterministic fallback ──────────────────────────────────────────────────────

def _fallback_explanation(
    prob: float,
    drivers: list[Driver],
    retrieved: list | None = None,
) -> ChurnExplanation:
    """Build a rule-based explanation from SHAP drivers; no LLM required."""
    risk_level: Literal["low", "medium", "high"] = (
        "high" if prob > 0.5 else "medium" if prob > 0.3 else "low"
    )
    pos = [d for d in drivers if d.shap_contribution > 0]
    neg = [d for d in drivers if d.shap_contribution < 0]

    parts: list[str] = [f"Customer has an estimated {prob:.0%} churn probability."]
    if pos:
        driver_str = " and ".join(f"{d.feature} ({d.value!r})" for d in pos[:2])
        parts.append(f"Key risk factors include {driver_str}.")
    if neg:
        parts.append(f"{neg[0].feature} ({neg[0].value!r}) reduces churn risk.")

    key_factors = [
        f"{d.feature}: {d.value!r} (SHAP {d.shap_contribution:+.4f})"
        for d in drivers[:3]
    ]
    action_map: dict[str, str] = {
        "high": "Immediately contact the customer with a personalised retention offer.",
        "medium": "Schedule a proactive check-in and review service usage patterns.",
        "low": "Maintain standard engagement and monitor for behavioural changes.",
    }
    citations = [r.citation for r in (retrieved or [])[:2]]
    return ChurnExplanation(
        risk_level=risk_level,
        summary=" ".join(parts),
        key_factors=key_factors,
        recommended_action=action_map[risk_level],
        citations=citations,
    )


# ── Main entry point ────────────────────────────────────────────────────────────

def explain_prediction(
    features: dict[str, Any],
    top_k: int = 5,
    use_rag: bool | None = None,
    _rag_embedder: Any = None,
    _calibrated_probability: float | None = None,
) -> tuple[ChurnExplanation, dict[str, Any]]:
    """Generate a grounded churn explanation for one customer.

    Parameters
    ----------
    features :
        Dict of original Telco column names → values.
    top_k :
        Number of SHAP drivers to surface and ground the explanation in.
    use_rag :
        Whether to retrieve playbook tactics before calling the LLM.
        None (default) = auto-detect: True when ``settings.rag_corpus_path`` is
        non-empty, False otherwise.
    _rag_embedder :
        Inject a custom embedder (for tests). When provided, the module-level
        FAISS cache is bypassed and a fresh index is built with this embedder.
    _calibrated_probability :
        When provided, use this probability for risk_level and the prompt headline
        instead of the uncalibrated SHAP model's probability. Pass the champion
        model's calibrated probability here so the explanation's risk_level is
        consistent with what /predict returns.

    Returns
    -------
    (explanation, metadata)
        explanation : validated ChurnExplanation (includes citations list).
        metadata    : dict with keys ``provider``, ``model``, ``probability``,
                      ``shap_probability`` (when calibrated differs), and
                      ``ungrounded_factors`` (key_factors that failed Tier-1 check).
    """
    result = per_prediction_drivers(features, top_k=top_k)
    shap_prob, drivers = result.probability, result.drivers

    # Use calibrated probability for prompt + risk_level when the champion provides it.
    prob = _calibrated_probability if _calibrated_probability is not None else shap_prob
    meta: dict[str, Any] = {"probability": prob}
    if _calibrated_probability is not None:
        meta["shap_probability"] = shap_prob

    # ── RAG retrieval ─────────────────────────────────────────────────────────
    retrieved: list = []
    _use_rag = use_rag
    if _use_rag is None:
        corpus = settings.rag_corpus_path
        _use_rag = corpus.exists() and any(corpus.glob("*.md"))

    if _use_rag:
        try:
            from churn.genai.rag import get_rag_components, retrieve  # noqa: PLC0415

            entries, index = get_rag_components(embedder=_rag_embedder)
            if entries and index is not None:
                query = _build_rag_query(drivers)
                retrieved = retrieve(
                    query, entries, index, top_k=3, embedder=_rag_embedder
                )
        except Exception:
            retrieved = []

    # ── LLM or fallback path ─────────────────────────────────────────────────
    if not settings.explanation_enabled:
        meta |= {"provider": "fallback", "model": "fallback", "ungrounded_factors": []}
        return _fallback_explanation(prob, drivers, retrieved=retrieved), meta

    system = _SYSTEM_PROMPT_RAG if retrieved else _SYSTEM_PROMPT
    user = _build_user_message(prob, drivers, retrieved=retrieved)
    try:
        completion = complete_with_fallback(system, user)
        raw = json.loads(completion.text)
        explanation = ChurnExplanation(**raw)
        meta |= {"provider": completion.provider, "model": completion.model}
    except Exception:
        meta |= {"provider": "fallback", "model": "fallback"}
        explanation = _fallback_explanation(prob, drivers, retrieved=retrieved)

    # Lightweight Tier-1 grounding check — surfaces any key_factor that doesn't
    # reference a SHAP driver. Informational only; nothing is dropped.
    try:
        from churn.genai.eval import _factor_matches_driver  # noqa: PLC0415

        ungrounded = [
            f for f in explanation.key_factors
            if not any(_factor_matches_driver(f, d) for d in drivers)
        ]
    except Exception:
        ungrounded = []
    meta["ungrounded_factors"] = ungrounded

    return explanation, meta
