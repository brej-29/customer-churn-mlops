"""Tests for churn/genai/rag.py — offline, fake embedder injected via conftest."""

from __future__ import annotations

from pathlib import Path

import pytest

from churn.genai.rag import (
    RetrievalResult,
    build_index,
    get_rag_components,
    load_playbooks,
    retrieve,
)

_CORPUS_PRESENT = pytest.mark.skipif(
    not Path("data/playbooks").exists()
    or not any(p for p in Path("data/playbooks").glob("*.md") if p.name.lower() != "readme.md"),
    reason="data/playbooks corpus not present",
)

_EXPECTED_CORPUS_SIZE = 8  # 8 tactic files; README is excluded


# ── load_playbooks ─────────────────────────────────────────────────────────────

@_CORPUS_PRESENT
def test_load_playbooks_count():
    entries = load_playbooks()
    assert len(entries) == _EXPECTED_CORPUS_SIZE


@_CORPUS_PRESENT
def test_playbook_entries_have_required_fields():
    for e in load_playbooks():
        assert e.id
        assert e.title
        assert e.source.endswith(".md")
        assert len(e.text) > 50


@_CORPUS_PRESENT
def test_playbook_entries_exclude_readme():
    ids = [e.id for e in load_playbooks()]
    assert "README" not in ids
    assert "readme" not in ids


@_CORPUS_PRESENT
def test_playbook_source_is_filename_not_path():
    for e in load_playbooks():
        assert "/" not in e.source, f"source should be filename only, got: {e.source!r}"
        assert "\\" not in e.source


# ── build_index ────────────────────────────────────────────────────────────────

@_CORPUS_PRESENT
def test_build_index_total_matches_corpus(fake_embedder):
    entries = load_playbooks()
    index = build_index(entries, embedder=fake_embedder)
    assert index.ntotal == len(entries)


# ── retrieve ──────────────────────────────────────────────────────────────────

@_CORPUS_PRESENT
def test_retrieve_returns_top_k(fake_embedder):
    entries = load_playbooks()
    index = build_index(entries, embedder=fake_embedder)
    results = retrieve(
        "month-to-month contract short tenure",
        entries, index, top_k=3, embedder=fake_embedder,
    )
    assert len(results) == 3


@_CORPUS_PRESENT
def test_retrieve_results_are_retrieval_result_instances(fake_embedder):
    entries = load_playbooks()
    index = build_index(entries, embedder=fake_embedder)
    results = retrieve("high monthly charges", entries, index, top_k=2, embedder=fake_embedder)
    for r in results:
        assert isinstance(r, RetrievalResult)


@_CORPUS_PRESENT
def test_retrieve_scores_in_valid_range(fake_embedder):
    entries = load_playbooks()
    index = build_index(entries, embedder=fake_embedder)
    results = retrieve("electronic check autopay", entries, index, top_k=3, embedder=fake_embedder)
    for r in results:
        assert -1e-5 <= r.score <= 1.0 + 1e-5


@_CORPUS_PRESENT
def test_retrieve_scores_descending(fake_embedder):
    entries = load_playbooks()
    index = build_index(entries, embedder=fake_embedder)
    results = retrieve("fiber optic internet quality", entries, index, top_k=4, embedder=fake_embedder)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@_CORPUS_PRESENT
def test_retrieve_exact_match_top_result(fake_embedder):
    """Querying with an entry's own text returns that entry with cosine score ≈ 1.0."""
    entries = load_playbooks()
    index = build_index(entries, embedder=fake_embedder)
    target = entries[0]
    results = retrieve(target.text, entries, index, top_k=1, embedder=fake_embedder)
    assert results[0].entry.id == target.id
    assert results[0].score > 0.99


@_CORPUS_PRESENT
def test_retrieve_citations_are_md_filenames(fake_embedder):
    entries = load_playbooks()
    index = build_index(entries, embedder=fake_embedder)
    results = retrieve("senior customer no dependents", entries, index, top_k=3, embedder=fake_embedder)
    for r in results:
        assert r.citation.endswith(".md")


@_CORPUS_PRESENT
def test_retrieve_top_k_capped_by_corpus_size(fake_embedder):
    """Requesting more results than corpus entries should not error."""
    entries = load_playbooks()
    index = build_index(entries, embedder=fake_embedder)
    results = retrieve("security addon tech support", entries, index, top_k=100, embedder=fake_embedder)
    assert len(results) == len(entries)


# ── get_rag_components ─────────────────────────────────────────────────────────

@_CORPUS_PRESENT
def test_get_rag_components_returns_correct_size(fake_embedder):
    entries, index = get_rag_components(embedder=fake_embedder)
    assert len(entries) == _EXPECTED_CORPUS_SIZE
    assert index is not None
    assert index.ntotal == _EXPECTED_CORPUS_SIZE


@_CORPUS_PRESENT
def test_get_rag_components_skips_cache_for_explicit_embedder(fake_embedder):
    """Each call with an explicit embedder builds a fresh index (no caching)."""
    _, index1 = get_rag_components(embedder=fake_embedder)
    _, index2 = get_rag_components(embedder=fake_embedder)
    assert index1 is not index2
