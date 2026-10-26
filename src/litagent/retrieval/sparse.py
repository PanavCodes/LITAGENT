"""Sparse retriever wrapper for BM25.

Per PRD.md §7.3 (FR-VEC-4) & ARCHITECTURE.md §4.1:
Queries BM25Store and returns ranked (chunk_id, score) pairs.
"""

from __future__ import annotations

import logging

from litagent.stores.sparse_bm25 import BM25Store

logger = logging.getLogger(__name__)


class SparseRetriever:
    """Retrieves ranked chunks using BM25Okapi scoring."""

    def __init__(self, store: BM25Store, top_k: int = 20) -> None:
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) sorted descending by score."""
        k = top_k if top_k is not None else self.top_k
        return self.store.search(query, top_k=k)

