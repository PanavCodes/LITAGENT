"""Unit tests for Phase 2 stores: GraphStore, FAISSVectorStore, BM25Store."""

from __future__ import annotations

import tempfile
from pathlib import Path

from litagent.stores.graph_networkx import NetworkXGraphStore, normalise_author
from litagent.stores.sparse_bm25 import BM25Store
from litagent.stores.vector_faiss import FAISSVectorStore


def test_normalise_author() -> None:
    assert normalise_author("Vladimir Karpukhin, Jr.") == "vladimir karpukhin jr"
    assert normalise_author("  B.  Ding  ") == "b ding"


def test_networkx_graph_store() -> None:
    store = NetworkXGraphStore()
    papers = [
        {
            "arxiv_id": "2004.04906",
            "title": "Dense Passage Retrieval",
            "abstract": "We explore dense passage retrieval using dual-encoders.",
            "authors": ["Vladimir Karpukhin", "Barlas Oguz"],
            "year": 2020,
            "categories": ["cs.IR"],
            "keywords": ["dense retrieval", "dual encoder"],
        },
        {
            "arxiv_id": "2005.11401",
            "title": "Retrieval-Augmented Generation",
            "abstract": "RAG models combine parametric and non-parametric memory.",
            "authors": ["Patrick Lewis", "Ethan Perez"],
            "year": 2020,
            "categories": ["cs.CL"],
            "keywords": ["rag", "dense retrieval"],
        },
    ]
    citations = [{"source_id": "2005.11401", "target_id": "2004.04906"}]

    stats = store.load_corpus(papers, citations)
    assert stats["total_nodes"] > 0
    assert stats["total_edges"] > 0

    # Query authors
    authors_rows = store.run_read("MATCH (a:Author)-[:WROTE]->(p:Paper)", {"title": "Dense Passage Retrieval"})
    assert len(authors_rows) >= 2
    author_names = [r["author"] for r in authors_rows]
    assert "Vladimir Karpukhin" in author_names

    # Query citations
    cites_rows = store.run_read("MATCH (p:Paper)-[:CITES]->(q:Paper)", {"title": "Dense Passage Retrieval"})
    assert len(cites_rows) == 1
    assert cites_rows[0]["citing_id"] == "2005.11401"

    # Schema text
    schema = store.schema_text()
    assert "Paper" in schema
    assert "CITES" in schema


def test_bm25_store() -> None:
    chunks = [
        {"chunk_id": "c1", "paper_id": "p1", "text": "Dense passage retrieval and vector embeddings."},
        {"chunk_id": "c2", "paper_id": "p2", "text": "Reciprocal rank fusion for hybrid search."},
        {"chunk_id": "c3", "paper_id": "p3", "text": "Cross-encoder reranking algorithms."},
    ]
    store = BM25Store()
    count = store.build_index(chunks)
    assert count == 3

    hits = store.search("reciprocal rank fusion", top_k=2)
    assert len(hits) > 0
    assert hits[0][0] == "c2"


def test_faiss_vector_store() -> None:
    chunks = [
        {"chunk_id": "c1", "paper_id": "p1", "text": "Dense passage retrieval for question answering."},
        {"chunk_id": "c2", "paper_id": "p2", "text": "Knowledge graph reasoning and multi-hop queries."},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        idx_file = Path(tmp_dir) / "test.index"
        meta_file = Path(tmp_dir) / "test.json"

        store = FAISSVectorStore(index_path=idx_file, metadata_path=meta_file)
        count = store.add_chunks(chunks)
        assert count == 2

        hits = store.search("dense retrieval question answering", top_k=1)
        assert len(hits) == 1
        assert hits[0][0] == "c1"
        assert hits[0][1] > 0.0  # Cosine similarity > 0
