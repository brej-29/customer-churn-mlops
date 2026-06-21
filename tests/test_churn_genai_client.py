"""Tests for churn/genai/client.py — fully offline, no real API calls."""

from __future__ import annotations

import pytest

from churn.genai.client import (
    _PROVIDER_CONFIGS,
    LLMError,
    _strip_fences,
    complete_with_fallback,
)

# ── _strip_fences ──────────────────────────────────────────────────────────────

def test_strip_fences_plain_json():
    assert _strip_fences('{"key": "value"}') == '{"key": "value"}'


def test_strip_fences_json_fence():
    assert _strip_fences('```json\n{"key": "value"}\n```') == '{"key": "value"}'


def test_strip_fences_plain_fence():
    assert _strip_fences('```\n{"key": "value"}\n```') == '{"key": "value"}'


def test_strip_fences_surrounding_whitespace():
    result = _strip_fences('  ```json\n{"k": 1}\n```  ')
    assert result == '{"k": 1}'


def test_strip_fences_no_newline_after_fence():
    assert _strip_fences('```json{"k": 1}```') == '{"k": 1}'


# ── Provider config constants ──────────────────────────────────────────────────

def test_gemini_default_model():
    assert _PROVIDER_CONFIGS["gemini"]["default_model"] == "gemini-2.5-flash-lite"


def test_groq_default_model():
    assert _PROVIDER_CONFIGS["groq"]["default_model"] == "llama-3.1-8b-instant"


# ── complete_with_fallback helpers ─────────────────────────────────────────────

@pytest.fixture
def _both_keys(monkeypatch):
    """Inject fake keys so both providers are eligible; set no custom model."""
    import churn.config
    monkeypatch.setattr(churn.config.settings, "gemini_api_key", "fake-gemini")
    monkeypatch.setattr(churn.config.settings, "groq_api_key", "fake-groq")
    monkeypatch.setattr(churn.config.settings, "llm_provider", "gemini")
    monkeypatch.setattr(churn.config.settings, "llm_model", None)


# ── complete_with_fallback — success ──────────────────────────────────────────

def test_complete_primary_succeeds(monkeypatch, _both_keys):
    canned = '{"summary": "ok"}'
    monkeypatch.setattr("churn.genai.client._create_completion", lambda *a, **kw: canned)

    result = complete_with_fallback("system msg", "user msg")

    assert result.text == canned
    assert result.provider == "gemini"
    assert result.model == _PROVIDER_CONFIGS["gemini"]["default_model"]


def test_custom_model_override_respected(monkeypatch, _both_keys):
    import churn.config
    monkeypatch.setattr(churn.config.settings, "llm_model", "my-custom-model")
    seen: list[str] = []

    def mock_create(client, model, messages, max_tokens, temperature, use_json_mode):
        seen.append(model)
        return '{"ok": true}'

    monkeypatch.setattr("churn.genai.client._create_completion", mock_create)
    complete_with_fallback("s", "u")
    assert seen[0] == "my-custom-model"


# ── complete_with_fallback — fallback ─────────────────────────────────────────

def test_fallback_used_when_primary_raises(monkeypatch, _both_keys):
    canned = '{"summary": "fallback ok"}'
    call_count = [0]

    def mock_create(*a, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("primary failed")
        return canned

    monkeypatch.setattr("churn.genai.client._create_completion", mock_create)
    result = complete_with_fallback("s", "u")

    assert result.text == canned
    assert result.provider == "groq"
    assert call_count[0] == 2


def test_provider_skipped_when_no_api_key(monkeypatch):
    """Primary provider is skipped if its key is absent; fallback is used instead."""
    import churn.config
    monkeypatch.setattr(churn.config.settings, "gemini_api_key", None)
    monkeypatch.setattr(churn.config.settings, "groq_api_key", "fake-groq")
    monkeypatch.setattr(churn.config.settings, "llm_provider", "gemini")
    monkeypatch.setattr(churn.config.settings, "llm_model", None)
    monkeypatch.setattr("churn.genai.client._create_completion", lambda *a, **kw: '{"ok": 1}')

    result = complete_with_fallback("s", "u")
    assert result.provider == "groq"


# ── complete_with_fallback — total failure ────────────────────────────────────

def test_both_providers_fail_raises_llm_error(monkeypatch, _both_keys):
    def _always_raise(*a, **kw):
        raise RuntimeError("network error")

    monkeypatch.setattr("churn.genai.client._create_completion", _always_raise)

    with pytest.raises(LLMError):
        complete_with_fallback("s", "u")


def test_no_keys_at_all_raises_llm_error(monkeypatch):
    import churn.config
    monkeypatch.setattr(churn.config.settings, "gemini_api_key", None)
    monkeypatch.setattr(churn.config.settings, "groq_api_key", None)
    monkeypatch.setattr(churn.config.settings, "llm_provider", "gemini")
    monkeypatch.setattr(churn.config.settings, "llm_model", None)

    with pytest.raises(LLMError):
        complete_with_fallback("s", "u")
