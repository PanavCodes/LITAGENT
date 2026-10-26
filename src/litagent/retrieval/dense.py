"""Dense retriever wrapper using FAISS.

Per PRD.md §7.3 (FR-VEC-3) and ARCHITECTURE.md §4.1:
Returns ranked candidates (chunk_id, score).
"""

from __future__ import annotations

import logging

from litagent.stores.vector_faiss import FAISSVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Retrieves ranked chunks using FAISS cosine similarity."""

    def __init__(self, store: FAISSVectorStore, top_k: int = 20) -> None:
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) sorted descending by score."""
        k = top_k if top_k is not None else self.top_k
        return self.store.search(query, top_k=k)

