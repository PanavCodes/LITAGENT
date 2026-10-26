"""Vector retrieval branch node for LangGraph.

Per PRD.md §7.5 (FR-AGT-5) & ARCHITECTURE.md §4.1:
BM25 + Dense + RRF + CrossEncoder rerank, outputting standardized EvidenceItems.
"""

from __future__ import annotations

import logging
from typing import Any

from litagent.agent.state import AgentState
from litagent.retrieval.dense import DenseRetriever
from litagent.retrieval.rerank import CrossEncoderReranker
from litagent.retrieval.rrf import reciprocal_rank_fusion
from litagent.retrieval.sparse import SparseRetriever
from litagent.schemas import ChunkHit, EvidenceItem
from litagent.stores.sparse_bm25 import BM25Store
from litagent.stores.vector_faiss import FAISSVectorStore

logger = logging.getLogger(__name__)


class VectorBranchNode:
    """Executes dense + sparse + RRF + reranking vector branch."""

    def __init__(
        self,
        faiss_store: FAISSVectorStore,
        bm25_store: BM25Store,
        reranker: CrossEncoderReranker | None = None,
        use_dense: bool = True,
        use_bm25: bool = True,
        use_rrf: bool = True,
        use_rerank: bool = True,
    ) -> None:
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store
        self.dense_retriever = DenseRetriever(faiss_store, top_k=20)
        self.sparse_retriever = SparseRetriever(bm25_store, top_k=20)
        self.reranker = reranker or CrossEncoderReranker(top_n=5)

        self.use_dense = use_dense
        self.use_bm25 = use_bm25
        self.use_rrf = use_rrf
        self.use_rerank = use_rerank

    def __call__(self, state: AgentState) -> AgentState:
        query = state.get("current_query") or state.get("question", "")

        ranked_lists: list[list[tuple[str, float]]] = []

        if self.use_dense:
            dense_hits = self.dense_retriever.retrieve(query)
            ranked_lists.append(dense_hits)

        if self.use_bm25:
            bm25_hits = self.sparse_retriever.retrieve(query)
            ranked_lists.append(bm25_hits)

        # Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(ranked_lists, k=60, top_n=20)

        # Retrieve chunk text for candidate items
        candidate_chunks: list[dict[str, Any]] = []
        for chunk_id, rrf_score, ranks in fused:
            chunk_data = self.faiss_store.get_chunk(chunk_id) or self.bm25_store.get_chunk(chunk_id)
            if chunk_data:
                chunk_copy = dict(chunk_data)
                chunk_copy["rrf_score"] = rrf_score
                chunk_copy["ranks"] = ranks
                candidate_chunks.append(chunk_copy)

        # Rerank top candidates with CrossEncoder
        reranked_tuples = self.reranker.rerank(query, candidate_chunks)

        # Keep top 5 reranked

        top_chunks: list[ChunkHit] = []
        evidence: list[EvidenceItem] = state.get("evidence", [])
        start_idx = len(evidence) + 1

        for idx, (chunk_id, r_score) in enumerate(reranked_tuples, start=start_idx):
            chunk_data = self.faiss_store.get_chunk(chunk_id) or self.bm25_store.get_chunk(chunk_id)
            if chunk_data:
                hit = ChunkHit(
                    chunk_id=chunk_id,
                    paper_id=chunk_data.get("paper_id", ""),
                    text=chunk_data.get("text", ""),
                    scores={"rerank": r_score},
                )
                top_chunks.append(hit)

                item = EvidenceItem(
                    id=f"E{idx}",
                    source="vector",
                    text=chunk_data.get("text", ""),
                    paper_ids=[chunk_data.get("paper_id", "")],
                    scores={"rerank": r_score},
                )
                evidence.append(item)

        state["chunks"] = top_chunks
        state["evidence"] = evidence
        return state
