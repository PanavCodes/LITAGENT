"""FAISS dense vector index store.

Per PRD.md §7.3 (FR-VEC-2, FR-VEC-3):
- CPU-based embeddings with sentence-transformers/all-MiniLM-L6-v2.
- Normalised vectors with IndexFlatIP (cosine similarity).
- Maps internal vector IDs to chunk IDs and metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS dense vector store with embedding cache and persistence."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        dimension: int = 384,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self._model: SentenceTransformer | None = None

        self.index: faiss.Index = faiss.IndexFlatIP(self.dimension)
        self.metadata: list[dict[str, Any]] = []


        if self.index_path and self.index_path.exists() and self.metadata_path and self.metadata_path.exists():
            self.load(self.index_path, self.metadata_path)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Generate normalised dense embeddings on CPU."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def add_chunks(self, chunks: list[dict[str, Any]], batch_size: int = 64) -> int:
        """Add chunks to FAISS index with metadata mapping."""
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        self.index.add(embeddings)
        self.metadata.extend(chunks)

        if self.index_path and self.metadata_path:
            self.save(self.index_path, self.metadata_path)

        return len(chunks)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Search top_k chunks by cosine similarity, returning list of (chunk_id, score)."""
        if self.index.ntotal == 0:
            return []

        q_vec = self.embed_texts([query])
        scores, indices = self.index.search(q_vec, min(top_k, self.index.ntotal))

        hits: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(self.metadata):
                chunk_id = self.metadata[idx]["chunk_id"]
                hits.append((chunk_id, float(score)))

        return hits

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Lookup chunk dictionary by chunk_id."""
        for c in self.metadata:
            if c["chunk_id"] == chunk_id:
                return c
        return None

    def save(self, index_file: Path, meta_file: Path) -> None:
        index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_file))
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def load(self, index_file: Path, meta_file: Path) -> None:
        self.index = faiss.read_index(str(index_file))
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
