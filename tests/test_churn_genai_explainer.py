"""Tests for churn/genai/explainer.py.

Offline: LLM calls are always mocked.
SHAP/pipeline tests are skipped when data/raw/telco_churn.csv or
reports/best_xgb_params.json are absent (same pattern as test_churn_explain.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from churn.genai.client import CompletionResult, LLMError
from churn.genai.explainer import (
    ChurnExplanation,
    Driver,
    DriverResult,
    _fallback_explanation,
    explain_prediction,
    per_prediction_drivers,
)

_data_present = pytest.mark.skipif(
    not Path("data/raw/telco_churn.csv").exists()
    or not Path("reports/best_xgb_params.json").exists(),
    reason="Telco CSV or tuned XGBoost params not present",
)

# Representative high-risk customer: short tenure, month-to-month, fibre optic.
_SAMPLE: dict = {
    "gender": "Male",
    "SeniorCitizen": "0",  # clean_telco coerces all CATEGORICAL_FEATURES to str
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.50,
    "TotalCharges": 447.50,
}

_CANNED_LLM_JSON = json.dumps({
    "risk_level": "high",
    "summary": (
        "Customer is at high churn risk due to a month-to-month contract and short tenure. "
        "Fibre optic service increases switching options. "
        "Immediate outreach is strongly recommended."
    ),
    "key_factors": ["Contract: Month-to-month", "tenure: 5 months", "InternetService: Fiber optic"],
    "recommended_action": "Offer a 12-month contract with a 15% loyalty discount.",
})


# ── Module-scoped SHAP fixture ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def driver_result() -> DriverResult:
    """Compute SHAP drivers once per test session; reused by all driver tests."""
    if not Path("data/raw/telco_churn.csv").exists():
        pytest.skip("data/raw/telco_churn.csv not present")
    if not Path("reports/best_xgb_params.json").exists():
        pytest.skip("reports/best_xgb_params.json not present")
    return per_prediction_drivers(_SAMPLE, top_k=5)


# ── _fallback_explanation — no data required ──────────────────────────────────

def _make_drivers(contributions: list[float]) -> list[Driver]:
    return [
        Driver(
            feature=f"feat{i}", value=i,
            shap_contribution=c,
            direction="increases" if c > 0 else "decreases",
        )
        for i, c in enumerate(contributions)
    ]


def test_fallback_returns_valid_churn_explanation():
    expl = _fallback_explanation(0.65, _make_drivers([0.4, 0.2, -0.1]))
    assert isinstance(expl, ChurnExplanation)
    ChurnExplanation.model_validate(expl.model_dump())


def test_fallback_risk_level_high():
    expl = _fallback_explanation(0.72, _make_drivers([0.3, 0.2]))
    assert expl.risk_level == "high"


def test_fallback_risk_level_medium():
    expl = _fallback_explanation(0.40, _make_drivers([0.1]))
    assert expl.risk_level == "medium"


def test_fallback_risk_level_low():
    expl = _fallback_explanation(0.10, _make_drivers([-0.2]))
    assert expl.risk_level == "low"


def test_fallback_key_factors_come_from_drivers():
    drivers = _make_drivers([0.5, 0.3, 0.1])
    expl = _fallback_explanation(0.6, drivers)
    assert len(expl.key_factors) == 3
    # Each factor should reference its driver's feature name
    assert all("feat" in f for f in expl.key_factors)


def test_fallback_summary_mentions_positive_drivers():
    drivers = _make_drivers([0.4, -0.1])
    expl = _fallback_explanation(0.55, drivers)
    assert "feat0" in expl.summary  # top positive driver appears


def test_fallback_summary_mentions_negative_driver():
    drivers = _make_drivers([0.4, -0.3])
    expl = _fallback_explanation(0.55, drivers)
    assert "feat1" in expl.summary  # negative driver reduces risk — mentioned


def test_fallback_recommended_action_non_empty():
    expl = _fallback_explanation(0.2, _make_drivers([-0.1]))
    assert isinstance(expl.recommended_action, str) and len(expl.recommended_action) > 0


# ── per_prediction_drivers ─────────────────────────────────────────────────────

@_data_present
def test_per_prediction_probability_in_range(driver_result):
    assert 0.0 <= driver_result.probability <= 1.0


@_data_present
def test_per_prediction_returns_top_k(driver_result):
    assert len(driver_result.drivers) == 5


@_data_present
def test_per_prediction_no_ohe_prefixes(driver_result):
    for d in driver_result.drivers:
        assert not d.feature.startswith("num__"), f"OHE prefix leaked: {d.feature!r}"
        assert not d.feature.startswith("cat__"), f"OHE prefix leaked: {d.feature!r}"


@_data_present
def test_per_prediction_direction_matches_sign(driver_result):
    for d in driver_result.drivers:
        expected = "increases" if d.shap_contribution > 0 else "decreases"
        assert d.direction == expected, (
            f"{d.feature}: shap={d.shap_contribution}, direction={d.direction}"
        )


@_data_present
def test_per_prediction_customer_values_not_none(driver_result):
    for d in driver_result.drivers:
        assert d.value is not None, f"Driver {d.feature!r} has None customer value"


@_data_present
def test_per_prediction_deterministic():
    r1 = per_prediction_drivers(_SAMPLE, top_k=5)
    r2 = per_prediction_drivers(_SAMPLE, top_k=5)
    assert r1.probability == r2.probability
    assert [d.feature for d in r1.drivers] == [d.feature for d in r2.drivers]
    assert [d.shap_contribution for d in r1.drivers] == [d.shap_contribution for d in r2.drivers]


@_data_present
def test_per_prediction_top_k_respected():
    result = per_prediction_drivers(_SAMPLE, top_k=3)
    assert len(result.drivers) == 3


# ── explain_prediction — LLM mocked ──────────────────────────────────────────

@_data_present
def test_explain_prediction_returns_valid_schema(monkeypatch):
    import churn.config
    monkeypatch.setattr(churn.config.settings, "explanation_enabled", True)
    monkeypatch.setattr(
        "churn.genai.explainer.complete_with_fallback",
        lambda *a, **kw: CompletionResult(
            text=_CANNED_LLM_JSON, provider="gemini", model="gemini-2.5-flash-lite"
        ),
    )
    expl, meta = explain_prediction(_SAMPLE)

    assert isinstance(expl, ChurnExplanation)
    assert expl.risk_level in ("low", "medium", "high")
    assert isinstance(expl.summary, str) and len(expl.summary) > 10
    assert isinstance(expl.key_factors, list) and len(expl.key_factors) >= 1
    assert isinstance(expl.recommended_action, str)
    assert meta["provider"] == "gemini"
    assert meta["model"] == "gemini-2.5-flash-lite"
    assert "probability" in meta


@_data_present
def test_explain_prediction_prompt_contains_driver_names(monkeypatch):
    """The user message sent to the LLM must include the real SHAP driver names."""
    import churn.config
    monkeypatch.setattr(churn.config.settings, "explanation_enabled", True)
    captured: dict = {}

    def mock_complete(system, user, **kw):
        captured["system"] = system
        captured["user"] = user
        return CompletionResult(
            text=_CANNED_LLM_JSON, provider="groq", model="llama-3.1-8b-instant"
        )

    monkeypatch.setattr("churn.genai.explainer.complete_with_fallback", mock_complete)
    explain_prediction(_SAMPLE)

    # Top SHAP driver name must appear verbatim in the user message
    real = per_prediction_drivers(_SAMPLE, top_k=5)
    top_feature = real.drivers[0].feature
    assert top_feature in captured["user"], (
        f"Top driver {top_feature!r} not found in user prompt"
    )
    assert "probability" in captured["user"].lower()
    assert "SHAP" in captured["user"]


@_data_present
def test_explain_prediction_fallback_when_disabled(monkeypatch):
    import churn.config
    monkeypatch.setattr(churn.config.settings, "explanation_enabled", False)

    expl, meta = explain_prediction(_SAMPLE)

    assert isinstance(expl, ChurnExplanation)
    assert meta["provider"] == "fallback"
    assert meta["model"] == "fallback"
    assert "probability" in meta


@_data_present
def test_explain_prediction_fallback_when_llm_fails(monkeypatch):
    import churn.config
    monkeypatch.setattr(churn.config.settings, "explanation_enabled", True)

    def _raise(*a, **kw):
        raise LLMError("both providers failed")

    monkeypatch.setattr("churn.genai.explainer.complete_with_fallback", _raise)

    expl, meta = explain_prediction(_SAMPLE)

    assert isinstance(expl, ChurnExplanation)
    assert meta["provider"] == "fallback"


@_data_present
def test_explain_prediction_fallback_valid_schema_on_llm_failure(monkeypatch):
    """Fallback explanation must satisfy the ChurnExplanation schema."""
    import churn.config
    monkeypatch.setattr(churn.config.settings, "explanation_enabled", True)
    monkeypatch.setattr(
        "churn.genai.explainer.complete_with_fallback",
        lambda *a, **kw: (_ for _ in ()).throw(LLMError("fail")),
    )

    expl, _ = explain_prediction(_SAMPLE)
    ChurnExplanation.model_validate(expl.model_dump())


# ── RAG path — LLM mocked, fake embedder injected ─────────────────────────────

_CANNED_LLM_JSON_RAG = json.dumps({
    "risk_level": "high",
    "summary": (
        "Customer has a high churn probability driven by a month-to-month contract "
        "and short tenure. Immediate intervention is recommended."
    ),
    "key_factors": ["Contract: Month-to-month", "tenure: 5 months"],
    "recommended_action": (
        "Offer a 12-month contract with a 15% discount to reduce switching cost."
    ),
    "citations": ["contract_upgrade.md"],
})


@_data_present
def test_explain_prediction_rag_prompt_includes_tactics(monkeypatch, fake_embedder):
    """Retrieved tactic sources must appear in the user message sent to the LLM."""
    import churn.config
    monkeypatch.setattr(churn.config.settings, "explanation_enabled", True)

    captured: dict = {}

    def mock_complete(system, user, **kw):
        captured["system"] = system
        captured["user"] = user
        return CompletionResult(
            text=_CANNED_LLM_JSON_RAG, provider="gemini", model="gemini-2.5-flash-lite"
        )

    monkeypatch.setattr("churn.genai.explainer.complete_with_fallback", mock_complete)
    explain_prediction(_SAMPLE, use_rag=True, _rag_embedder=fake_embedder)

    assert "Source:" in captured["user"], "RAG context 'Source:' not in user prompt"
    assert ".md" in captured["user"], "No .md citation in user prompt"
    assert "Title:" in captured["user"], "No 'Title:' in RAG context block"


@_data_present
def test_explain_prediction_rag_uses_rag_system_prompt(monkeypatch, fake_embedder):
    """System prompt should mention 'playbook' when RAG context is present."""
    import churn.config
    monkeypatch.setattr(churn.config.settings, "explanation_enabled", True)

    captured: dict = {}

    def mock_complete(system, user, **kw):
        captured["system"] = system
        return CompletionResult(text=_CANNED_LLM_JSON_RAG, provider="gemini", model="test")

    monkeypatch.setattr("churn.genai.explainer.complete_with_fallback", mock_complete)
    explain_prediction(_SAMPLE, use_rag=True, _rag_embedder=fake_embedder)

    assert "playbook" in captured["system"].lower()


@_data_present
def test_explain_prediction_rag_citations_from_llm(monkeypatch, fake_embedder):
    """ChurnExplanation.citations reflects what the LLM returns."""
    import churn.config
    monkeypatch.setattr(churn.config.settings, "explanation_enabled", True)
    monkeypatch.setattr(
        "churn.genai.explainer.complete_with_fallback",
        lambda *a, **kw: CompletionResult(
            text=_CANNED_LLM_JSON_RAG, provider="gemini", model="test"
        ),
    )

    expl, _ = explain_prediction(_SAMPLE, use_rag=True, _rag_embedder=fake_embedder)

    assert isinstance(expl.citations, list)
    assert "contract_upgrade.md" in expl.citations


@_data_present
def test_explain_prediction_rag_citations_are_md_filenames(monkeypatch, fake_embedder):
    """All citations should be .md filenames from the corpus."""
    import churn.config
    monkeypatch.setattr(churn.config.settings, "explanation_enabled", True)
    monkeypatch.setattr(
        "churn.genai.explainer.complete_with_fallback",
        lambda *a, **kw: CompletionResult(
            text=_CANNED_LLM_JSON_RAG, provider="gemini", model="test"
        ),
    )

    expl, _ = explain_prediction(_SAMPLE, use_rag=True, _rag_embedder=fake_embedder)
    for c in expl.citations:
        assert c.endswith(".md"), f"Citation is not an .md filename: {c!r}"


@_data_present
def test_explain_prediction_fallback_with_rag_has_citations(monkeypatch, fake_embedder):
    """Deterministic fallback populates citations from retrieved tactics."""
    import churn.config
    monkeypatch.setattr(churn.config.settings, "explanation_enabled", False)

    expl, meta = explain_prediction(_SAMPLE, use_rag=True, _rag_embedder=fake_embedder)

    assert isinstance(expl, ChurnExplanation)
    assert meta["provider"] == "fallback"
    assert isinstance(expl.citations, list)
    assert len(expl.citations) > 0, "Fallback with RAG should have at least one citation"
    for c in expl.citations:
        assert c.endswith(".md")


@_data_present
def test_explain_prediction_fallback_valid_schema_with_rag(monkeypatch, fake_embedder):
    """Fallback + RAG explanation must pass full schema validation."""
    import churn.config
    monkeypatch.setattr(churn.config.settings, "explanation_enabled", False)

    expl, _ = explain_prediction(_SAMPLE, use_rag=True, _rag_embedder=fake_embedder)
    ChurnExplanation.model_validate(expl.model_dump())
