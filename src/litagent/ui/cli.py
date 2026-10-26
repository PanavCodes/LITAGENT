"""Command line interface for LitAgent.

Per PRD.md §7.8 (FR-UI-1):
litagent ask "question" prints answer, citations, route, and Cypher.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from litagent.agent.workflow import build_litagent_graph
from litagent.config import load_config
from litagent.llm import get_llm
from litagent.retrieval.rerank import CrossEncoderReranker
from litagent.stores.graph_networkx import NetworkXGraphStore
from litagent.stores.sparse_bm25 import BM25Store
from litagent.stores.vector_faiss import FAISSVectorStore


def ask_question(question: str, forced_route: str | None = None) -> None:
    """Run LitAgent against a single question and display rich terminal output."""
    config = load_config()

    # Load stores
    index_dir = Path("data/index")
    graph_store = NetworkXGraphStore(db_path=index_dir / "graph.pkl")
    faiss_store = FAISSVectorStore(
        model_name=config.embedding.model,
        index_path=index_dir / "faiss.index",
        metadata_path=index_dir / "chunk_metadata.json",
    )
    bm25_store = BM25Store(index_path=index_dir / "bm25.pkl")
    reranker = CrossEncoderReranker(use_model=False)

    llm = get_llm(config=config.llm)

    app = build_litagent_graph(
        llm=llm,
        graph_store=graph_store,
        faiss_store=faiss_store,
        bm25_store=bm25_store,
        reranker=reranker,
    )

    initial_state = {
        "question": question,
        "forced_route": forced_route,
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print(f"ROUTE: {final_state.get('route')} ({final_state.get('route_rationale')})")
    if final_state.get("cypher"):
        print(f"CYPHER: {final_state.get('cypher')}")
    print("-" * 60)
    print(f"ANSWER:\n{final_state.get('answer')}")
    print("-" * 60)

    citations = final_state.get("citations", [])
    if citations:
        print("CITATIONS:")
        for c in citations:
            print(f"  [{c.tag}] {c.title} ({c.url})")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LitAgent CLI")
    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", type=str, help="Research question")
    ask_parser.add_argument("--route", type=str, choices=["graph", "vector", "both", "none"], default=None)

    args = parser.parse_args()
    if args.command == "ask":
        ask_question(args.question, forced_route=args.route)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
