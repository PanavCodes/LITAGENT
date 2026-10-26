"""Unit tests for Phase 5: CRAG correction loop, termination invariant, and uncertainty signals."""

from __future__ import annotations

import pytest

from litagent.agent.uncertainty import (
    compute_uncertainty,
    evidence_grade_uncertainty,
    graph_flags_uncertainty,
    retrieval_margin,
)
from litagent.agent.workflow import build_litagent_graph
from litagent.llm import FakeLLM
from litagent.retrieval.rerank import CrossEncoderReranker
from litagent.schemas import ChunkHit
from litagent.stores.graph_networkx import NetworkXGraphStore
from litagent.stores.sparse_bm25 import BM25Store
from litagent.stores.vector_faiss import FAISSVectorStore


def test_uncertainty_signals() -> None:
    # S1: High margin -> low uncertainty
    assert retrieval_margin([0.9, 0.4, 0.3]) < retrieval_margin([0.5, 0.49, 0.48])

    # S2: High grade -> low uncertainty
    assert evidence_grade_uncertainty(0.9) == pytest.approx(0.1)
    assert evidence_grade_uncertainty(0.2) == pytest.approx(0.8)

    # S5: Status mapping
    assert graph_flags_uncertainty("ok") == 0.0
    assert graph_flags_uncertainty("error") == 1.0
    assert graph_flags_uncertainty("empty") == 0.8


def test_composite_uncertainty_computation() -> None:
    state = {
        "chunks": [ChunkHit(chunk_id="c1", paper_id="p1", text="...", scores={"rerank": 0.8})],
        "grade": 0.9,
        "cypher_status": "ok",
    }
    report = compute_uncertainty(state)  # type: ignore[arg-type]
    assert 0.0 <= report.score <= 1.0
    assert "S1" in report.signals
    assert "S2" in report.signals
    assert "S5" in report.signals
    assert not report.abstain


def test_workflow_termination_invariant() -> None:
    """AGENTS.md §4 Invariant: Workflow MUST terminate within iteration bound

    even when the fake grader consistently returns insufficient (grade=0.0).
    """
    graph = NetworkXGraphStore()
    faiss = FAISSVectorStore()
    bm25 = BM25Store()
    reranker = CrossEncoderReranker(use_model=False)

    # Fake LLM where grader ALWAYS returns grade 0.0
    fake_llm = FakeLLM(
        responses={
            "router": '{"route": "vector", "rationale": "Test query"}',
            "Evaluate if evidence answers": '{"grade": 0.0, "missing_info": ["Always missing"]}',
            "Reformulate this query": "rewritten search query",
        },
        default_response="Default generated answer [P1].",
    )

    max_iterations = 2
    app = build_litagent_graph(
        llm=fake_llm,
        graph_store=graph,
        faiss_store=faiss,
        bm25_store=bm25,
        reranker=reranker,
        grader_threshold=0.6,
        max_iterations=max_iterations,
        enable_loop=True,
    )

    # Run 10 times to verify termination invariant holds deterministically
    for _ in range(10):
        res = app.invoke({"question": "Test question with impossible evidence", "forced_route": "vector"})
        assert res.get("iteration") == max_iterations
        assert res.get("answer") is not None
        assert res.get("uncertainty") is not None
