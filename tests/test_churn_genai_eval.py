"""Tests for churn/genai/eval.py — fully offline; LLM and embedder mocked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from churn.genai.eval import (
    FaithfulnessEvalResult,
    FaithfulnessResult,
    FaithfulnessViolation,
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
