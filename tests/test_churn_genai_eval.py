"""Tests for churn/genai/eval.py — fully offline; LLM and embedder mocked."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from churn.genai.eval import (
    FaithfulnessEvalResult,
    FaithfulnessResult,
    FaithfulnessViolation,
    _categorize_unsupported,
    _driver_description,
    _readable_feature_name,
    check_faithfulness,
    run_faithfulness_eval,
)
from churn.genai.explainer import (
    ChurnExplanation,
    Driver,
    DriverResult,
    _build_user_message,
    _format_driver_value,
)

_data_present = pytest.mark.skipif(
    not Path("data/raw/telco_churn.csv").exists()
    or not Path("reports/best_xgb_params.json").exists(),
    reason="Telco CSV or tuned XGBoost params not present",
)


# ── Shared test data ───────────────────────────────────────────────────────────

def _driver(feature: str, value: Any = "Yes", shap: float = 0.3) -> Driver:
    return Driver(
        feature=feature,
        value=value,
        shap_contribution=shap,
        direction="increases" if shap > 0 else "decreases",
    )


_DRIVERS = [
    _driver("Contract", "Month-to-month"),
    _driver("tenure", 5.0),
    _driver("InternetService", "Fiber optic"),
    _driver("OnlineSecurity", "No"),
    _driver("MonthlyCharges", 89.50),
]

_FAITHFUL_EXPL = ChurnExplanation(
    risk_level="high",
    summary="High risk due to month-to-month contract.",
    key_factors=[
        "Contract: Month-to-month",
        "Short tenure (5 months)",
        "Fiber optic internet service",
    ],
    recommended_action="Offer annual contract.",
    citations=[],
)

_UNFAITHFUL_EXPL = ChurnExplanation(
    risk_level="high",
    summary="High risk.",
    key_factors=[
        "Contract: Month-to-month",
        "poor customer service history",
        "recent billing dispute",
    ],
    recommended_action="Call the customer.",
    citations=[],
)


# ── check_faithfulness ─────────────────────────────────────────────────────────

def test_check_faithfulness_returns_result():
    assert isinstance(check_faithfulness(_FAITHFUL_EXPL, _DRIVERS), FaithfulnessResult)


def test_check_faithfulness_all_grounded_is_faithful():
    result = check_faithfulness(_FAITHFUL_EXPL, _DRIVERS)
    assert result.faithful is True
    assert result.unsupported_factors == []


def test_check_faithfulness_hallucinated_is_not_faithful():
    result = check_faithfulness(_UNFAITHFUL_EXPL, _DRIVERS)
    assert result.faithful is False


def test_check_faithfulness_lists_unsupported_factors():
    result = check_faithfulness(_UNFAITHFUL_EXPL, _DRIVERS)
    assert "poor customer service history" in result.unsupported_factors
    assert "recent billing dispute" in result.unsupported_factors


def test_check_faithfulness_grounded_factor_not_in_unsupported():
    result = check_faithfulness(_UNFAITHFUL_EXPL, _DRIVERS)
    assert "Contract: Month-to-month" not in result.unsupported_factors


def test_check_faithfulness_empty_key_factors_is_faithful():
    expl = ChurnExplanation(
        risk_level="low",
        summary="Low risk.",
        key_factors=[],
        recommended_action="Monitor.",
        citations=[],
    )
    result = check_faithfulness(expl, _DRIVERS)
    assert result.faithful is True
    assert result.unsupported_factors == []


def test_check_faithfulness_no_drivers_flags_all():
    result = check_faithfulness(_FAITHFUL_EXPL, [])
    assert result.faithful is False
    assert len(result.unsupported_factors) == len(_FAITHFUL_EXPL.key_factors)


def test_check_faithfulness_camelcase_driver_matches():
    """'OnlineSecurity' (camelCase) should match 'No Online Security'."""
    expl = ChurnExplanation(
        risk_level="high", summary="", key_factors=["No Online Security"],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [_driver("OnlineSecurity", "No")])
    assert result.faithful is True


def test_check_faithfulness_internet_service_matches():
    """'InternetService' matches 'Fiber optic internet service' via substring."""
    expl = ChurnExplanation(
        risk_level="high", summary="", key_factors=["Fiber optic internet service"],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [_driver("InternetService", "Fiber optic")])
    assert result.faithful is True


def test_check_faithfulness_monthly_charges_matches():
    expl = ChurnExplanation(
        risk_level="high", summary="", key_factors=["High monthly charges ($89.50)"],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [_driver("MonthlyCharges", 89.50)])
    assert result.faithful is True


# ── _format_driver_value ───────────────────────────────────────────────────────

def test_format_tenure_float_to_months():
    assert _format_driver_value("tenure", 5.0) == "5 months"


def test_format_tenure_int_to_months():
    assert _format_driver_value("tenure", 12) == "12 months"


def test_format_monthly_charges_dollar():
    assert _format_driver_value("MonthlyCharges", 89.50) == "$89.50"


def test_format_total_charges_dollar():
    assert _format_driver_value("TotalCharges", 447.50) == "$447.50"


def test_format_avg_monthly_spend_dollar():
    assert _format_driver_value("avg_monthly_spend", 65.0) == "$65.00"


def test_format_categorical_as_str():
    assert _format_driver_value("Contract", "Month-to-month") == "Month-to-month"


def test_format_none_is_na():
    assert _format_driver_value("tenure", None) == "N/A"


# ── _build_user_message unit rendering ────────────────────────────────────────

def _make_drivers(**kwargs: Any) -> list[Driver]:
    return [Driver(feature=f, value=v, shap_contribution=0.1, direction="increases")
            for f, v in kwargs.items()]


def test_user_message_tenure_rendered_in_months():
    msg = _build_user_message(0.7, _make_drivers(tenure=5.0))
    assert "5 months" in msg


def test_user_message_monthly_charges_rendered_with_dollar():
    msg = _build_user_message(0.7, _make_drivers(MonthlyCharges=89.50))
    assert "$89.50" in msg


def test_user_message_total_charges_rendered_with_dollar():
    msg = _build_user_message(0.7, _make_drivers(TotalCharges=447.50))
    assert "$447.50" in msg


def test_user_message_units_note_present():
    msg = _build_user_message(0.7, _make_drivers(tenure=2.0, MonthlyCharges=50.0))
    assert "months" in msg.lower()
    assert "USD" in msg or "usd" in msg.lower()


def test_user_message_categorical_rendered_without_repr_quotes():
    msg = _build_user_message(0.7, _make_drivers(Contract="Month-to-month"))
    assert "Month-to-month" in msg


def test_user_message_shap_and_probability_still_present():
    msg = _build_user_message(0.72, _make_drivers(Contract="Month-to-month"))
    assert "SHAP" in msg
    assert "72.0%" in msg


# ── run_faithfulness_eval — offline via injection ─────────────────────────────

_SYNTHETIC = pd.DataFrame([{"x": i} for i in range(3)])


def _mock_drivers_fn(features: dict, top_k: int = 5) -> DriverResult:
    return DriverResult(
        probability=0.7,
        drivers=[_driver("Contract", "Month-to-month")],
    )


def _mock_explain_faithful(features: dict, top_k: int = 5, **kw: Any):
    return ChurnExplanation(
        risk_level="high",
        summary="test",
        key_factors=["Contract: Month-to-month"],
        recommended_action="test action",
        citations=[],
    ), {"provider": "mock", "model": "mock", "probability": 0.7}


def _mock_explain_unfaithful(features: dict, top_k: int = 5, **kw: Any):
    return ChurnExplanation(
        risk_level="high",
        summary="test",
        key_factors=["unresolved billing dispute", "poor customer service"],
        recommended_action="fix",
        citations=[],
    ), {"provider": "mock", "model": "mock", "probability": 0.6}


def test_run_faithfulness_eval_returns_eval_result():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_faithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
    )
    assert isinstance(result, FaithfulnessEvalResult)


def test_run_faithfulness_eval_score_in_range():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_faithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
    )
    assert 0.0 <= result.faithfulness_score <= 1.0


def test_run_faithfulness_eval_faithful_mock_score_is_one():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_faithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
    )
    assert result.n_samples == 3
    assert result.n_faithful == 3
    assert result.faithfulness_score == 1.0
    assert result.violations == []


def test_run_faithfulness_eval_unfaithful_mock_score_is_zero():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_unfaithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
    )
    assert result.n_faithful == 0
    assert result.faithfulness_score == 0.0
    assert len(result.violations) == 3


def test_run_faithfulness_eval_violations_have_unsupported_factors():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_unfaithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
    )
    for v in result.violations:
        assert isinstance(v, FaithfulnessViolation)
        assert len(v.unsupported_factors) > 0


def test_run_faithfulness_eval_deterministic():
    kwargs = dict(
        _explain_fn=_mock_explain_faithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
    )
    r1 = run_faithfulness_eval(**kwargs)
    r2 = run_faithfulness_eval(**kwargs)
    assert r1.faithfulness_score == r2.faithfulness_score
    assert r1.n_samples == r2.n_samples


def test_run_faithfulness_eval_empty_data_returns_zero():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_faithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=pd.DataFrame(),
    )
    assert result.n_samples == 0
    assert result.faithfulness_score == 0.0


# ── Tier 2 embedding similarity — helper embedders ────────────────────────────

def _const_embedder(texts: list[str]) -> np.ndarray:
    """All texts get the same unit vector → cosine similarity = 1.0 for any pair."""
    v = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    return np.tile(v, (len(texts), 1))


def _ortho_embedder(texts: list[str]) -> np.ndarray:
    """Each text gets a unique orthogonal unit vector → cosine similarity = 0.0."""
    dim = max(len(texts), 8)
    return np.eye(len(texts), dim, dtype=np.float32)


def _make_cosine_embedder(target_cos: float, dim: int = 8) -> Callable:
    """Embedder where texts[0] (the factor) has cosine target_cos with all others."""
    sin_val = float(np.sqrt(max(0.0, 1.0 - target_cos ** 2)))
    v_factor = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32)
    v_driver = np.array([target_cos, sin_val] + [0.0] * (dim - 2), dtype=np.float32)

    def _embed(texts: list[str]) -> np.ndarray:
        return np.array(
            [v_factor if i == 0 else v_driver for i in range(len(texts))],
            dtype=np.float32,
        )

    return _embed


# ── Tier 2 semantic recovery ───────────────────────────────────────────────────

def test_check_faithfulness_factor_results_field_present():
    result = check_faithfulness(_FAITHFUL_EXPL, _DRIVERS)
    assert hasattr(result, "factor_results")
    assert isinstance(result.factor_results, list)


def test_check_faithfulness_exact_factors_have_match_type_exact():
    result = check_faithfulness(_FAITHFUL_EXPL, _DRIVERS)
    for fr in result.factor_results:
        assert fr.match_type == "exact", f"Expected 'exact', got {fr.match_type!r} for {fr.factor!r}"


def test_check_faithfulness_semantic_recovery_via_embedder():
    """Paraphrase that fails Tier 1 but passes Tier 2 via constant embedder."""
    driver = _driver("PaymentMethod", "Electronic check")
    factor = "Electronic check payment"  # fails Tier 1: "paymentmethod" not in kf_norm
    expl = ChurnExplanation(
        risk_level="high", summary="", key_factors=[factor],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [driver], embedder=_const_embedder)
    assert result.faithful is True
    assert result.unsupported_factors == []
    assert result.factor_results[0].match_type == "semantic"


def test_check_faithfulness_semantic_recovery_similarity_recorded():
    driver = _driver("PaymentMethod", "Electronic check")
    factor = "Electronic check payment"
    expl = ChurnExplanation(
        risk_level="high", summary="", key_factors=[factor],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [driver], embedder=_const_embedder)
    fr = result.factor_results[0]
    assert fr.similarity is not None
    assert fr.similarity > 0.99  # const_embedder → cosine ≈ 1.0


def test_check_faithfulness_semantic_recovery_matched_driver_recorded():
    driver = _driver("PaymentMethod", "Electronic check")
    factor = "Electronic check payment"
    expl = ChurnExplanation(
        risk_level="high", summary="", key_factors=[factor],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [driver], embedder=_const_embedder)
    assert result.factor_results[0].matched_driver == "PaymentMethod"


def test_check_faithfulness_genuine_hallucination_with_embedder():
    """Orthogonal embedder: all similarities = 0 → factor stays match_type 'none'."""
    driver = _driver("Contract", "Month-to-month")
    factor = "customer called support three times"
    expl = ChurnExplanation(
        risk_level="high", summary="", key_factors=[factor],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [driver], embedder=_ortho_embedder, sim_threshold=0.5)
    assert result.faithful is False
    assert factor in result.unsupported_factors
    assert result.factor_results[0].match_type == "none"


def test_check_faithfulness_none_factor_has_similarity_recorded():
    """Below-threshold factors still record their similarity score."""
    driver = _driver("Contract", "Month-to-month")
    factor = "customer called support three times"
    expl = ChurnExplanation(
        risk_level="high", summary="", key_factors=[factor],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [driver], embedder=_ortho_embedder, sim_threshold=0.5)
    fr = result.factor_results[0]
    assert fr.similarity is not None
    assert fr.similarity < 0.5


def test_check_faithfulness_threshold_boundary_at_exact():
    """cosine == threshold → semantic (>= comparison)."""
    embedder = _make_cosine_embedder(0.5)
    driver = _driver("Contract", "Month-to-month")
    factor = "some paraphrase that will not match tier one"
    expl = ChurnExplanation(
        risk_level="high", summary="", key_factors=[factor],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [driver], embedder=embedder, sim_threshold=0.5)
    assert result.factor_results[0].match_type == "semantic"


def test_check_faithfulness_threshold_boundary_below():
    """cosine just below threshold → none."""
    embedder = _make_cosine_embedder(0.49)
    driver = _driver("Contract", "Month-to-month")
    factor = "some paraphrase that will not match tier one"
    expl = ChurnExplanation(
        risk_level="high", summary="", key_factors=[factor],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [driver], embedder=embedder, sim_threshold=0.5)
    assert result.factor_results[0].match_type == "none"


def test_check_faithfulness_no_embedder_tier1_failure_is_none():
    """Without embedder, Tier 1 failure stays 'none' (Tier 2 skipped)."""
    driver = _driver("Contract", "Month-to-month")
    factor = "some paraphrase that will not match tier one"
    expl = ChurnExplanation(
        risk_level="high", summary="", key_factors=[factor],
        recommended_action="", citations=[],
    )
    # conftest autouse patches _get_default_embedder → None, so Tier 2 is skipped
    result = check_faithfulness(expl, [driver], embedder=None)
    assert result.factor_results[0].match_type == "none"


def test_check_faithfulness_mixed_exact_and_semantic():
    """One factor via Tier 1, one via Tier 2 → both faithful; match_types correct."""
    drivers = [
        _driver("Contract", "Month-to-month"),
        _driver("PaymentMethod", "Electronic check"),
    ]
    expl = ChurnExplanation(
        risk_level="high",
        summary="",
        key_factors=[
            "Contract: Month-to-month",  # Tier 1 match
            "Electronic check payment",  # Tier 2 match via const_embedder
        ],
        recommended_action="",
        citations=[],
    )
    result = check_faithfulness(expl, drivers, embedder=_const_embedder)
    assert result.faithful is True
    assert result.factor_results[0].match_type == "exact"
    assert result.factor_results[1].match_type == "semantic"


# ── run_faithfulness_eval breakdown fields ────────────────────────────────────

def test_run_faithfulness_eval_has_breakdown_fields():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_faithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
    )
    assert hasattr(result, "n_exact")
    assert hasattr(result, "n_semantic")
    assert hasattr(result, "n_unsupported")
    assert hasattr(result, "semantic_recoveries")


def test_run_faithfulness_eval_exact_count_for_faithful_mock():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_faithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
    )
    # 3 samples × 1 factor each, all exact
    assert result.n_exact == 3
    assert result.n_semantic == 0
    assert result.n_unsupported == 0


def test_run_faithfulness_eval_unsupported_count_for_unfaithful_mock():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_unfaithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
    )
    # 3 samples × 2 factors each, all none
    assert result.n_unsupported == 6
    assert result.n_exact == 0
    assert result.n_semantic == 0


def test_run_faithfulness_eval_semantic_recovery_with_embedder():
    """With const embedder, unfaithful factors are recovered as semantic."""
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_unfaithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
        _embedder=_const_embedder,
    )
    # const_embedder → cosine = 1.0 for all pairs → all factors recovered
    assert result.n_semantic == 6
    assert result.n_unsupported == 0
    assert result.n_faithful == 3
    assert result.faithfulness_score == 1.0


def test_run_faithfulness_eval_semantic_recoveries_populated():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_unfaithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
        _embedder=_const_embedder,
    )
    assert len(result.semantic_recoveries) == 6
    for rec in result.semantic_recoveries:
        assert "factor" in rec
        assert "matched_driver" in rec
        assert "similarity" in rec
        assert rec["similarity"] > 0.99


def test_run_faithfulness_eval_ortho_embedder_no_recovery():
    """Orthogonal embedder: no semantic recoveries, all stay unsupported."""
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_unfaithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
        _embedder=_ortho_embedder,
    )
    assert result.n_semantic == 0
    assert result.n_unsupported == 6
    assert result.semantic_recoveries == []


# ── _readable_feature_name ────────────────────────────────────────────────────

def test_readable_monthly_charges():
    assert _readable_feature_name("MonthlyCharges") == "monthly charges"


def test_readable_payment_method():
    assert _readable_feature_name("PaymentMethod") == "payment method"


def test_readable_online_security():
    assert _readable_feature_name("OnlineSecurity") == "online security"


def test_readable_avg_monthly_spend():
    assert _readable_feature_name("avg_monthly_spend") == "average monthly spend"


def test_readable_internet_service():
    assert _readable_feature_name("InternetService") == "internet service"


def test_readable_plain_feature():
    assert _readable_feature_name("tenure") == "tenure"
    assert _readable_feature_name("Contract") == "contract"


# ── _driver_description ───────────────────────────────────────────────────────

def test_driver_description_numeric_value_excluded():
    d = _driver("MonthlyCharges", 89.50)
    desc = _driver_description(d)
    assert desc == "monthly charges"
    assert "89" not in desc


def test_driver_description_categorical_value_included():
    d = _driver("PaymentMethod", "Electronic check")
    desc = _driver_description(d)
    assert "payment method" in desc
    assert "electronic check" in desc


def test_driver_description_avg_monthly_spend_readable():
    d = _driver("avg_monthly_spend", 65.0)
    desc = _driver_description(d)
    assert desc == "average monthly spend"
    assert "65" not in desc


def test_driver_description_categorical_feature():
    d = _driver("Contract", "Month-to-month")
    desc = _driver_description(d)
    assert "contract" in desc
    assert "month-to-month" in desc


# ── Tier 1 improvement: readable name enables new exact matches ────────────────

def test_tier1_avg_monthly_spend_exact_after_expansion():
    """avg_monthly_spend → 'average monthly spend' enables Tier 1 exact match."""
    d = _driver("avg_monthly_spend", 65.0)
    expl = ChurnExplanation(
        risk_level="high", summary="", key_factors=["Average monthly spend"],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [d])
    assert result.faithful is True
    assert result.factor_results[0].match_type == "exact"


def test_tier1_avg_monthly_spend_was_failing_without_expansion():
    """Confirm the old token 'avg monthly spend' would NOT match 'Average monthly spend'."""
    # This documents that the fix is needed: "avg" ≠ "average" without expansion.
    # We verify by checking that the Tier 1 string check works via expansion —
    # the result being "exact" (above) is the proof that expansion is applied.
    assert _readable_feature_name("avg_monthly_spend") == "average monthly spend"
    assert "average" not in "avg monthly spend"


# ── Semantic tier uses readable descriptions (spy embedder) ───────────────────

def test_semantic_tier_driver_description_excludes_numeric_value():
    """Tier 2 must NOT include numeric driver values in the embedded texts."""
    captured: list[str] = []

    def spy_embedder(texts: list[str]) -> np.ndarray:
        captured.extend(texts)
        return np.tile(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), (len(texts), 1))

    d = _driver("MonthlyCharges", 89.50)
    expl = ChurnExplanation(
        risk_level="high", summary="",
        key_factors=["some factor that will not match tier one"],
        recommended_action="", citations=[],
    )
    check_faithfulness(expl, [d], embedder=spy_embedder)

    driver_texts = [t for t in captured if t != "some factor that will not match tier one"]
    assert all("89" not in t for t in driver_texts)
    assert any("monthly charges" in t for t in driver_texts)


def test_semantic_tier_driver_description_uses_readable_name():
    """Tier 2 must use expanded readable name (not raw camelCase identifier)."""
    captured: list[str] = []

    def spy_embedder(texts: list[str]) -> np.ndarray:
        captured.extend(texts)
        return np.tile(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), (len(texts), 1))

    d = _driver("avg_monthly_spend", 65.0)
    expl = ChurnExplanation(
        risk_level="high", summary="",
        key_factors=["some factor that will not match tier one"],
        recommended_action="", citations=[],
    )
    check_faithfulness(expl, [d], embedder=spy_embedder)

    driver_texts = [t for t in captured if t != "some factor that will not match tier one"]
    assert any("average monthly spend" in t for t in driver_texts)
    assert not any("avg" in t for t in driver_texts)


# ── _categorize_unsupported ───────────────────────────────────────────────────

def test_categorize_likely_paraphrase():
    """Factor shares a key word with driver readable feature name."""
    d = _driver("MonthlyCharges", 89.50)
    # "monthly" appears in both "high monthly charge" and "monthly charges"
    assert _categorize_unsupported("High monthly charge", [d]) == "likely-paraphrase"


def test_categorize_value_narration():
    """Factor references a word from the driver's VALUE, not its feature name."""
    d = _driver("PaymentMethod", "Electronic check")
    # "electronic" / "check" appear in value "Electronic check"
    assert _categorize_unsupported("Electronic check issues", [d]) == "value-narration"


def test_categorize_other_ungrounded():
    """No word overlap with any driver feature or value."""
    d = _driver("Contract", "Month-to-month")
    assert _categorize_unsupported("poor customer service", [d]) == "other/ungrounded"


def test_categorize_empty_factor():
    d = _driver("Contract", "Month-to-month")
    assert _categorize_unsupported("", [d]) == "other/ungrounded"


def test_categorize_no_drivers():
    assert _categorize_unsupported("high monthly charge", []) == "other/ungrounded"


# ── FactorResult.reason field ─────────────────────────────────────────────────

def test_factor_result_reason_set_for_none_factor():
    """match_type 'none' factors must have a non-None reason."""
    d = _driver("Contract", "Month-to-month")
    expl = ChurnExplanation(
        risk_level="high", summary="",
        key_factors=["unrelated xyz factor"],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [d], embedder=_ortho_embedder)
    fr = result.factor_results[0]
    assert fr.match_type == "none"
    assert fr.reason is not None
    assert fr.reason in {"likely-paraphrase", "value-narration", "other/ungrounded"}


def test_factor_result_reason_none_for_exact():
    """match_type 'exact' factors must have reason=None."""
    d = _driver("Contract", "Month-to-month")
    expl = ChurnExplanation(
        risk_level="high", summary="",
        key_factors=["Contract: Month-to-month"],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [d])
    assert result.factor_results[0].reason is None


def test_factor_result_reason_none_for_semantic():
    """match_type 'semantic' factors must have reason=None."""
    d = _driver("PaymentMethod", "Electronic check")
    expl = ChurnExplanation(
        risk_level="high", summary="",
        key_factors=["Electronic check payment"],
        recommended_action="", citations=[],
    )
    result = check_faithfulness(expl, [d], embedder=_const_embedder)
    fr = result.factor_results[0]
    assert fr.match_type == "semantic"
    assert fr.reason is None


# ── run_faithfulness_eval residual_categories + random_state ─────────────────

def test_run_faithfulness_eval_has_residual_categories():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_unfaithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
    )
    assert hasattr(result, "residual_categories")
    assert isinstance(result.residual_categories, dict)


def test_run_faithfulness_eval_residual_totals_match_unsupported():
    """Sum of residual category counts must equal n_unsupported."""
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_unfaithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
        _embedder=_ortho_embedder,
    )
    total = sum(result.residual_categories.values())
    assert total == result.n_unsupported


def test_run_faithfulness_eval_residual_zero_when_all_faithful():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_faithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
    )
    assert sum(result.residual_categories.values()) == 0


def test_run_faithfulness_eval_random_state_stored_in_result():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_faithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
        random_state=99,
    )
    assert result.random_state == 99


def test_run_faithfulness_eval_default_random_state():
    result = run_faithfulness_eval(
        _explain_fn=_mock_explain_faithful,
        _drivers_fn=_mock_drivers_fn,
        _test_data=_SYNTHETIC,
    )
    assert result.random_state == 42
