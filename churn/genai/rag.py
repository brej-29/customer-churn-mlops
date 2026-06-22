"""RAG over the retention-playbook corpus using FAISS + sentence-transformers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from churn.config import settings

Embedder = Callable[[list[str]], np.ndarray]  # (texts) -> (n, d) float32 normalised


@dataclass
class PlaybookEntry:
    id: str
    title: str
    source: str  # filename, e.g. "contract_upgrade.md" — used directly as citation
    text: str


@dataclass
class RetrievalResult:
    entry: PlaybookEntry
    score: float   # cosine similarity in [0, 1] (unit-vector inner product)
    citation: str  # same as entry.source


# ── Corpus loader ─────────────────────────────────────────────────────────────────

def load_playbooks(path: Path | None = None) -> list[PlaybookEntry]:
    """Read all .md files (excluding README) from the corpus directory."""
    corpus_dir = Path(path) if path is not None else settings.rag_corpus_path
    entries: list[PlaybookEntry] = []
    for md_file in sorted(corpus_dir.glob("*.md")):
        if md_file.name.lower() == "readme.md":
            continue
        text = md_file.read_text(encoding="utf-8")
        m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = m.group(1).strip() if m else md_file.stem.replace("_", " ").title()
        entries.append(PlaybookEntry(
            id=md_file.stem,
            title=title,
            source=md_file.name,
            text=text,
        ))
    return entries


# ── Embedder ──────────────────────────────────────────────────────────────────────

_DEFAULT_EMBEDDER: Embedder | None = None


def _get_default_embedder() -> Embedder | None:
    """Lazy-load sentence-transformers all-MiniLM-L6-v2.

    Returns None if the package or model is unavailable.
    Tests replace this function via monkeypatch to avoid downloading the model.
    """
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            _model = SentenceTransformer("all-MiniLM-L6-v2")

            def _embed(texts: list[str]) -> np.ndarray:
                return np.array(
                    _model.encode(texts, normalize_embeddings=True),
                    dtype=np.float32,
                )

            _DEFAULT_EMBEDDER = _embed
        except Exception:
            return None
    return _DEFAULT_EMBEDDER


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


# ── Index ─────────────────────────────────────────────────────────────────────────

def build_index(entries: list[PlaybookEntry], embedder: Embedder | None = None) -> Any:
    """Build a FAISS IndexFlatIP over entry texts (cosine similarity for unit vectors)."""
    import faiss

    if embedder is None:
        embedder = _get_default_embedder()
        if embedder is None:
            raise RuntimeError(
                "No embedder available — install sentence-transformers and "
                "download all-MiniLM-L6-v2, or pass an explicit embedder."
            )

    vecs = _l2_normalize(embedder([e.text for e in entries]))
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    return index


# ── Retrieval ─────────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    entries: list[PlaybookEntry],
    index: Any,
    top_k: int = 3,
    embedder: Embedder | None = None,
) -> list[RetrievalResult]:
    """Return the top_k most relevant playbook entries for a query string."""
    if embedder is None:
        embedder = _get_default_embedder()
        if embedder is None:
            return []

    q_vec = _l2_normalize(embedder([query]))
    scores, indices = index.search(q_vec, min(top_k, len(entries)))

    results: list[RetrievalResult] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        entry = entries[int(idx)]
        results.append(RetrievalResult(entry=entry, score=float(score), citation=entry.source))
    return results


# ── Module-level cache ────────────────────────────────────────────────────────────

_RAG_CACHE: dict[str, tuple[list[PlaybookEntry], Any]] = {}


def get_rag_components(
    corpus_path: Path | None = None,
    embedder: Embedder | None = None,
) -> tuple[list[PlaybookEntry], Any]:
    """Return (entries, index); module cache is used only with the default embedder.

    When an explicit embedder is provided (e.g., a fake in tests) the cache is
    bypassed so each call builds a fresh index with the supplied embedder.
    """
    path = Path(corpus_path) if corpus_path is not None else settings.rag_corpus_path

    if not path.exists():
        return [], None

    if embedder is not None:
        entries = load_playbooks(path)
        if not entries:
            return [], None
        return entries, build_index(entries, embedder=embedder)

    cache_key = str(path)
    if cache_key not in _RAG_CACHE:
        entries = load_playbooks(path)
        if not entries:
            _RAG_CACHE[cache_key] = ([], None)
        else:
            try:
                index = build_index(entries)
                _RAG_CACHE[cache_key] = (entries, index)
            except Exception:
                _RAG_CACHE[cache_key] = (entries, None)

    return _RAG_CACHE[cache_key]
