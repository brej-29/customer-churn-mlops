"""Session-wide fixtures shared across test modules."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

_FAKE_DIM = 8


def _make_fake_embed(dim: int = _FAKE_DIM):
    """Return a deterministic fake embed function of the given output dimension."""
    def _fake_embed(texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            seed = int(hashlib.sha256(text[:50].encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(dim).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            vecs.append(vec / norm if norm > 0 else vec)
        return np.array(vecs, dtype=np.float32)
    return _fake_embed


@pytest.fixture
def fake_embedder():
    """Deterministic fake embedder — sha256-seeded, no model download."""
    return _make_fake_embed(_FAKE_DIM)


@pytest.fixture(autouse=True)
def _disable_default_rag_embedder(monkeypatch):
    """Prevent sentence-transformers model download in ALL tests.

    Tests that need the RAG path must inject fake_embedder explicitly
    (pass it to explain_prediction via _rag_embedder= or directly to retrieve).
    """
    try:
        monkeypatch.setattr("churn.genai.rag._get_default_embedder", lambda: None)
    except Exception:
        pass
