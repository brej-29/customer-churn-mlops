"""Tests for Tier 2 GenAI config fields in churn/config.py.

All tests are offline — no network calls, no real API keys.
"""

from __future__ import annotations

import pytest

from churn.config import Settings


@pytest.fixture(autouse=True)
def _clean_genai_env(monkeypatch):
    """Strip any GenAI env vars that might leak from the real environment."""
    for var in (
        "GEMINI_API_KEY", "GROQ_API_KEY",
        "CHURN_GEMINI_API_KEY", "CHURN_GROQ_API_KEY",
        "LLM_PROVIDER", "LLM_MODEL",
        "CHURN_LLM_PROVIDER", "CHURN_LLM_MODEL",
        "CHURN_EXPLANATION_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)


def _settings(**kwargs):
    """Instantiate Settings without reading any .env file."""
    return Settings(_env_file=None, **kwargs)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_genai_defaults():
    s = _settings()
    assert s.llm_provider == "gemini"
    assert s.llm_model is None
    assert s.gemini_api_key is None
    assert s.groq_api_key is None
    assert s.explanation_enabled is False
    from pathlib import Path
    assert s.rag_corpus_path == Path("data/playbooks")


# ---------------------------------------------------------------------------
# explanation_enabled auto-detection
# ---------------------------------------------------------------------------


def test_explanation_disabled_without_keys():
    s = _settings()
    assert s.explanation_enabled is False
    assert isinstance(s.explanation_enabled, bool)


def test_explanation_enabled_with_gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    s = _settings()
    assert s.gemini_api_key == "test-gemini-key"
    assert s.explanation_enabled is True


def test_explanation_enabled_with_groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    s = _settings()
    assert s.groq_api_key == "test-groq-key"
    assert s.explanation_enabled is True


def test_explanation_explicit_override_respected(monkeypatch):
    """CHURN_EXPLANATION_ENABLED=false disables even when a key is present."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("CHURN_EXPLANATION_ENABLED", "false")
    s = _settings()
    assert s.explanation_enabled is False


# ---------------------------------------------------------------------------
# llm_provider and llm_model via unprefixed env vars
# ---------------------------------------------------------------------------


def test_llm_provider_via_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    s = _settings()
    assert s.llm_provider == "groq"


def test_llm_model_via_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemma2-9b-it")
    s = _settings()
    assert s.llm_model == "gemma2-9b-it"


def test_llm_provider_invalid_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(Exception):
        _settings()
