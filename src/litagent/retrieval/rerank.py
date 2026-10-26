"""Cross-encoder reranking over fused candidates.

Per PRD.md §7.3 (FR-VEC-6) and ARCHITECTURE.md §4.1:
Takes top 20 fused candidates, scores (query, passage) pairs using
a cross-encoder, and returns top 5 with individual stage scores.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranker using CrossEncoder on CPU."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        use_model: bool = True,
    ) -> None:
        self.model_name = model_name
        self.top_n = top_n
        self.use_model = use_model
        self._model: Any = None

    @property
    def model(self) -> Any:
        if not self.use_model:
            return "fallback"
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading CrossEncoder: {self.model_name}")
                self._model = CrossEncoder(self.model_name, device="cpu")
            except Exception as e:
                logger.warning(f"Failed to load CrossEncoder {self.model_name}: {e}. Using word-overlap fallback.")
                self._model = "fallback"
        return self._model


    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[tuple[str, float]]:
        """Score (query, chunk_text) pairs and return top_n (chunk_id, score)."""
        if not candidates:
            return []

        pairs = [[query, c.get("text", "")] for c in candidates]

        if self.model != "fallback":
            try:
                scores = self.model.predict(pairs)
                ranked = []
                for idx, c in enumerate(candidates):
                    ranked.append((c["chunk_id"], float(scores[idx])))
                ranked.sort(key=lambda x: x[1], reverse=True)
                return ranked[: self.top_n]
            except Exception as e:
                logger.warning(f"CrossEncoder inference failed: {e}. Falling back.")

        # Fallback keyword overlap scoring
        q_words = set(query.lower().split())
        scored = []
        for c in candidates:
            c_words = set(c.get("text", "").lower().split())
            overlap = len(q_words & c_words) / max(len(q_words), 1)
            scored.append((c["chunk_id"], float(overlap)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_n]
