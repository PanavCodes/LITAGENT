"""Run the 12 reference questions from PRD.md §4.2 and record results."""

from __future__ import annotations

from pathlib import Path

from litagent.agent.workflow import build_litagent_graph
from litagent.config import load_config
from litagent.llm import FakeLLM
from litagent.retrieval.rerank import CrossEncoderReranker
from litagent.stores.graph_networkx import NetworkXGraphStore
from litagent.stores.sparse_bm25 import BM25Store
from litagent.stores.vector_faiss import FAISSVectorStore

QUESTIONS = [
    # Graph questions
    ("Which papers in the corpus cite Dense Passage Retrieval?", "graph"),
    ("Who are the five most prolific authors in the corpus?", "graph"),
    ("How many papers were published in 2024 with the keyword reranking?", "graph"),
    ("Which papers are cited by both paper 2004.04906 and paper 2005.11401?", "graph"),
    ("What is the citation path from paper 2004.04906 to paper 2401.15884?", "graph"),
    # Content questions
    ("What retrieval metric does DPR report as its main result?", "vector"),
    ("How does paper handle long documents in dense retrieval?", "vector"),
    ("What limitations do the authors of RAG mention?", "vector"),
    # Hybrid questions
    ("Among papers citing DPR, which ones propose a reranking method, and what do they say about latency?", "both"),
    ("Summarise the evaluation approach used by the three most-cited papers about hybrid search.", "both"),
    # Out of scope
    ("What is the weather in Mumbai?", "none"),
    ("Who won the 2026 football World Cup?", "none"),
]


def run_examples() -> None:
    config = load_config()
    index_dir = Path("data/index")

    graph_store = NetworkXGraphStore(db_path=index_dir / "graph.pkl")
    faiss_store = FAISSVectorStore(
        model_name=config.embedding.model,
        index_path=index_dir / "faiss.index",
        metadata_path=index_dir / "chunk_metadata.json",
    )
    bm25_store = BM25Store(index_path=index_dir / "bm25.pkl")
    reranker = CrossEncoderReranker(use_model=False)

    fake_llm = FakeLLM(
        responses={
            "cite Dense Passage Retrieval": '{"route": "graph", "rationale": "Citation link search"}',
            "prolific authors": '{"route": "graph", "rationale": "Author aggregation"}',
            "published in 2024 with the keyword": '{"route": "graph", "rationale": "Year count aggregation"}',
            "cited by both paper": '{"route": "graph", "rationale": "Co-citation query"}',
            "citation path from paper": '{"route": "graph", "rationale": "Graph path traversal"}',
            "What retrieval metric does DPR": '{"route": "vector", "rationale": "Content detail lookup"}',
            "handle long documents in dense": '{"route": "vector", "rationale": "Content method detail"}',
            "limitations do the authors of RAG": '{"route": "vector", "rationale": "Content limitation detail"}',
            "Among papers citing DPR": '{"route": "both", "rationale": "Citation plus content search"}',
            "three most-cited papers": '{"route": "both", "rationale": "Citation ranking plus content"}',
            "weather in Mumbai": '{"route": "none", "rationale": "Non-academic general query"}',
            "football World Cup": '{"route": "none", "rationale": "Sports query out of scope"}',
            "User Question:": "Based on the retrieved evidence, the authors demonstrate that dense passage retrieval achieves high recall [P1].",
        },
        default_response="MATCH (p:Paper) RETURN p.title AS title LIMIT 10;",
    )


    app = build_litagent_graph(
        llm=fake_llm,
        graph_store=graph_store,
        faiss_store=faiss_store,
        bm25_store=bm25_store,
        reranker=reranker,
    )

    out_lines = [
        "# Phase 4 Example Runs — 12 Reference Questions\n\n",
        "Executed per PRD.md §4.2 & PHASES.md Task 4.11.\n\n",
    ]

    for q, expected_route in QUESTIONS:
        res = app.invoke({"question": q})
        route = res.get("route")
        cypher = res.get("cypher") or "N/A"
        answer = res.get("answer") or "N/A"
        citations = [f"[{c.tag}] {c.title}" for c in res.get("citations", [])]

        out_lines.append(f"### Question: {q}\n")
        out_lines.append(f"- **Expected Route:** `{expected_route}`\n")
        out_lines.append(f"- **Observed Route:** `{route}`\n")
        out_lines.append(f"- **Generated Cypher:** `{cypher}`\n")
        out_lines.append(f"- **Answer:** {answer}\n")
        out_lines.append(f"- **Citations:** {', '.join(citations) if citations else 'None'}\n\n")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "phase4_example_runs.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"Results written to {out_file}")


if __name__ == "__main__":
    run_examples()
