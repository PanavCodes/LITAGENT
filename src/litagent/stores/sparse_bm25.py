"""BM25 sparse retrieval store.

Per PRD.md §7.3 (FR-VEC-4) & ARCHITECTURE.md §4.1:
BM25Okapi sparse index over processed chunks with ID mapping.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def tokenise_bm25(text: str) -> list[str]:
    """Tokenise string for BM25: lower-case, alphanumeric words, min len 2."""
    if not text:
        return []
    words = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
    return words


class BM25Store:
    """Sparse index wrapper around rank-bm25 BM25Okapi."""

    def __init__(self, index_path: str | Path | None = None) -> None:
        self.index_path = Path(index_path) if index_path else None
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.metadata: list[dict[str, Any]] = []

        if self.index_path and self.index_path.exists():
            self.load(self.index_path)

    def build_index(self, chunks: list[dict[str, Any]]) -> int:
        """Tokenise corpus chunks and fit BM25Okapi."""
        if not chunks:
            return 0

        self.metadata = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]

        corpus_tokens = [tokenise_bm25(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

        if self.index_path:
            self.save(self.index_path)

        return len(chunks)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Score query against BM25 index and return list of (chunk_id, bm25_score)."""
        if self.bm25 is None or not self.chunk_ids:
            return []

        q_tokens = tokenise_bm25(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)
        top_indices = scores.argsort()[::-1][:top_k]

        hits: list[tuple[str, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                hits.append((self.chunk_ids[idx], score))

        return hits

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Lookup chunk dictionary by chunk_id."""
        for c in self.metadata:
            if c["chunk_id"] == chunk_id:
                return c
        return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"bm25": self.bm25, "chunk_ids": self.chunk_ids, "metadata": self.metadata},
                f,
            )

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunk_ids = data["chunk_ids"]
            self.metadata = data["metadata"]
