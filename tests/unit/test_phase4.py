"""Unit tests for Phase 4 LangGraph Agent v1: routing, branches, and end-to-end execution."""

from __future__ import annotations

from litagent.agent.router import RouterNode
from litagent.agent.workflow import build_litagent_graph
from litagent.llm import FakeLLM
from litagent.retrieval.rerank import CrossEncoderReranker
from litagent.stores.graph_networkx import NetworkXGraphStore
from litagent.stores.sparse_bm25 import BM25Store
from litagent.stores.vector_faiss import FAISSVectorStore


def test_router_node() -> None:
    fake_llm = FakeLLM(
        responses={
            "Question: Which papers cite Dense Passage Retrieval": '{"route": "graph", "rationale": "Asks for citations in graph"}',
            "Question: how does colbert work": '{"route": "vector", "rationale": "Asks for algorithm details"}',
            "Question: broken json query": "invalid non-json output",
        }
    )
    router = RouterNode(llm=fake_llm)


    route, rationale = router.route_query("Which papers cite Dense Passage Retrieval?")
    assert route == "graph"

    route, rationale = router.route_query("how does colbert work?")
    assert route == "vector"

    # Fallback on broken json
    route, rationale = router.route_query("broken json query")
    assert route == "vector"
    assert "Fallback" in rationale


def test_workflow_end_to_end() -> None:
    # Setup test stores
    graph = NetworkXGraphStore()
    graph.load_corpus(
        papers=[{
            "arxiv_id": "2004.04906",
            "title": "Dense Passage Retrieval",
            "abstract": "Dense dual encoders outperform BM25 on open-domain QA.",
            "authors": ["Vladimir Karpukhin"],
            "year": 2020,
            "categories": ["cs.IR"],
            "keywords": ["dense retrieval"],
        }],
        citations=[],
    )

    chunks = [
        {"chunk_id": "2004.04906_c0000", "paper_id": "2004.04906", "text": "Dense dual encoders outperform BM25 on open-domain QA."}
    ]
    bm25 = BM25Store()
    bm25.build_index(chunks)
    faiss = FAISSVectorStore()
    faiss.add_chunks(chunks)
    reranker = CrossEncoderReranker(use_model=False)

    fake_llm = FakeLLM(
        responses={
            "router": '{"route": "vector", "rationale": "Content query"}',
            "cypher": "MATCH (p:Paper) RETURN p.title AS title LIMIT 10",
        },
        default_response="Dense passage retrieval uses dual-encoders [P1].",
    )

    app = build_litagent_graph(
        llm=fake_llm,
        graph_store=graph,
        faiss_store=faiss,
        bm25_store=bm25,
        reranker=reranker,
    )

    # 1. Vector Route Execution
    res = app.invoke({"question": "How does DPR compare to BM25?", "forced_route": "vector"})
    assert res["route"] == "vector"
    assert "Dense passage retrieval" in res["answer"]
    assert len(res["citations"]) > 0
    assert res["citations"][0].tag == "P1"

    # 2. Graph Route Execution
    res_graph = app.invoke({"question": "Who wrote DPR?", "forced_route": "graph"})
    assert res_graph["route"] == "graph"
    assert res_graph.get("cypher") is not None

    # 3. Abstain Route
    res_abstain = app.invoke({"question": "What is the weather?", "forced_route": "none"})
    assert res_abstain["route"] == "none"
    assert "outside the scope" in res_abstain["answer"]
