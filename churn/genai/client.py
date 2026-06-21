"""Provider-agnostic LLM wrapper using OpenAI-compatible endpoints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from churn.config import settings

_PROVIDER_CONFIGS: dict[str, dict[str, str]] = {
    "gemini": {
        "default_model": "gemini-2.5-flash-lite",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_attr": "gemini_api_key",
    },
    "groq": {
        "default_model": "llama-3.1-8b-instant",
        "base_url": "https://api.groq.com/openai/v1",
        "key_attr": "groq_api_key",
    },
}


@dataclass
class CompletionResult:
    text: str      # JSON string, markdown fences already stripped
    provider: str  # "gemini" | "groq"
    model: str     # actual model used


class LLMError(Exception):
    """Raised when all configured providers fail or have no API key."""


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _create_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    use_json_mode: bool,
) -> str:
    """Single OpenAI-compatible chat completion.

    Isolated so tests can monkeypatch ``churn.genai.client._create_completion``
    to avoid real network calls.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _call_provider(
    provider: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> CompletionResult:
    cfg = _PROVIDER_CONFIGS[provider]
    api_key = getattr(settings, cfg["key_attr"]) or ""
    model = settings.llm_model or cfg["default_model"]
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    raw = _create_completion(client, model, messages, max_tokens, temperature, use_json_mode=True)
    return CompletionResult(text=_strip_fences(raw), provider=provider, model=model)


def complete_with_fallback(
    system: str,
    user: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> CompletionResult:
    """Call the primary provider; fall back to the other on any error.

    Raises ``LLMError`` when both providers fail or neither has an API key.
    """
    primary = settings.llm_provider
    fallback = "groq" if primary == "gemini" else "gemini"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    errors: list[str] = []
    for provider in (primary, fallback):
        api_key = getattr(settings, _PROVIDER_CONFIGS[provider]["key_attr"]) or ""
        if not api_key:
            errors.append(f"{provider}: no API key configured")
            continue
        try:
            return _call_provider(provider, messages, max_tokens, temperature)
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
    raise LLMError("; ".join(errors))
