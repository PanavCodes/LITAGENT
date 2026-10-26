"""Build all indexes: GraphStore, FAISS, and BM25.

Per PRD.md §7.2, §7.3 & PHASES.md Phase 2:
Builds FAISS index, BM25 index, and GraphStore from processed data.
Idempotent and logs build timings.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from litagent.config import load_config
from litagent.stores.graph_networkx import NetworkXGraphStore
from litagent.stores.sparse_bm25 import BM25Store
from litagent.stores.vector_faiss import FAISSVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items = []
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    return items


def build_indexes(
    processed_dir: str | Path = "data/processed",
    index_dir: str | Path = "data/index",
) -> dict[str, Any]:
    start_time = time.time()
    config = load_config()

    proc_path = Path(processed_dir)
    idx_path = Path(index_dir)
    idx_path.mkdir(parents=True, exist_ok=True)

    papers = load_jsonl(proc_path / "papers.jsonl")
    chunks = load_jsonl(proc_path / "chunks.jsonl")
    citations = load_jsonl(proc_path / "citations.jsonl")

    logger.info(f"Loaded {len(papers)} papers, {len(chunks)} chunks, {len(citations)} citations.")

    # 1. Build Graph Store (NetworkX or Neo4j)
    graph_start = time.time()
    graph_store = NetworkXGraphStore(db_path=idx_path / "graph.pkl")
    graph_stats = graph_store.load_corpus(papers, citations)
    graph_duration = time.time() - graph_start
    logger.info(f"GraphStore loaded in {graph_duration:.2f}s: {graph_stats}")

    # 2. Build BM25 Sparse Index
    bm25_start = time.time()
    bm25_store = BM25Store(index_path=idx_path / "bm25.pkl")
    bm25_count = bm25_store.build_index(chunks)
    bm25_duration = time.time() - bm25_start
    logger.info(f"BM25 index built with {bm25_count} chunks in {bm25_duration:.2f}s")

    # 3. Build FAISS Dense Index
    faiss_start = time.time()
    faiss_store = FAISSVectorStore(
        model_name=config.embedding.model,
        index_path=idx_path / "faiss.index",
        metadata_path=idx_path / "chunk_metadata.json",
    )
    faiss_count = faiss_store.add_chunks(chunks, batch_size=config.embedding.batch_size)
    faiss_duration = time.time() - faiss_start
    logger.info(f"FAISS index built with {faiss_count} chunks in {faiss_duration:.2f}s")

    total_time = time.time() - start_time

    # Record stats to docs/graph_stats.md
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    with open(docs_dir / "graph_stats.md", "w", encoding="utf-8") as f:
        f.write("# Graph and Index Statistics\n\n")
        f.write(f"- Total Papers: {len(papers)}\n")
        f.write(f"- Total Chunks: {len(chunks)}\n")
        f.write(f"- Total Graph Nodes: {graph_stats.get('total_nodes')}\n")
        f.write(f"- Total Graph Edges: {graph_stats.get('total_edges')}\n")
        f.write(f"- Total Build Duration: {total_time:.2f}s\n")

    return {
        "papers": len(papers),
        "chunks": len(chunks),
        "graph_stats": graph_stats,
        "bm25_chunks": bm25_count,
        "faiss_chunks": faiss_count,
        "total_duration_s": round(total_time, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LitAgent Indexes")
    parser.parse_args()
    build_indexes()
