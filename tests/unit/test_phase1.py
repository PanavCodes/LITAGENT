"""Unit tests for Phase 1 data ingestion, cleaning, chunking, and idempotency."""

from __future__ import annotations

import tempfile

from scripts.build_corpus import build_corpus

from litagent.ingest.chunking import chunk_paper
from litagent.ingest.clean import deduplicate_papers
from litagent.ingest.keywords import extract_keywords_tfidf


def test_clean_and_deduplicate() -> None:
    raw_papers = [
        {"arxiv_id": "2001.0001", "title": "Paper 1", "abstract": "A valid abstract with enough text length to exceed fifty characters minimum requirement."},
        {"arxiv_id": "2001.0001", "title": "Paper 1 duplicate", "abstract": "Another abstract with enough text length to exceed fifty characters minimum requirement."},
        {"arxiv_id": "2001.0002", "title": "Paper 2", "abstract": "Short"},  # too short
        {"arxiv_id": "2001.0003", "title": "", "abstract": "No title here at all but length is fine."},
        {"arxiv_id": "2001.0004", "title": "Paper 4", "abstract": "A fourth valid abstract with enough text length to exceed fifty characters minimum requirement."},
    ]

    unique, stats = deduplicate_papers(raw_papers)
    assert len(unique) == 2
    assert stats["duplicate_arxiv_id"] == 1
    assert stats["missing_fields"] == 2
    assert unique[0]["arxiv_id"] == "2001.0001"
    assert unique[1]["arxiv_id"] == "2001.0004"


def test_chunking_abstract_c0000() -> None:
    abstract = "This is the abstract for DPR."
    body = "Section 1: Introduction.\n\nSection 2: Architecture and methods.\n\nSection 3: Evaluation results."

    chunks = chunk_paper("2004.04906", abstract=abstract, full_text=body, chunk_size=50, overlap=10)
    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "2004.04906_c0000"
    assert "Abstract:" in chunks[0].text
    assert chunks[0].scores["is_abstract"] == 1.0
    assert chunks[1].chunk_id == "2004.04906_c0001"


def test_keywords_tfidf() -> None:
    docs = [
        "dense passage retrieval and sparse retrieval for questions",
        "hybrid retrieval and reciprocal rank fusion for literature search",
        "cross-encoder reranking and reciprocal rank fusion in search engines",
    ]
    keywords = extract_keywords_tfidf(docs, top_k=3)
    assert len(keywords) == 3
    for kw in keywords:
        assert len(kw) <= 3
        assert len(kw) > 0


def test_build_corpus_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Run 1
        manifest1 = build_corpus(
            max_papers=30, output_dir=tmp_dir, use_sample_fallback=True, skip_network=True
        )
        # Run 2
        manifest2 = build_corpus(
            max_papers=30, output_dir=tmp_dir, use_sample_fallback=True, skip_network=True
        )

        assert manifest1["total_cleaned"] == manifest2["total_cleaned"]
        assert manifest1["total_chunks"] == manifest2["total_chunks"]
        assert manifest1["total_citations"] == manifest2["total_citations"]
        assert manifest1["citation_coverage_pct"] >= 60.0

