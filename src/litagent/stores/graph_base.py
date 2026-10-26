"""Abstract base protocol and interface for graph storage.

Per PRD.md §7.2 (FR-KG-4) and ARCHITECTURE.md §1:
GraphStore interface with Neo4j and NetworkX implementations.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphStore(Protocol):
    """Protocol for graph database backends (Neo4j and NetworkX)."""

    def load_corpus(
        self, papers: list[dict[str, Any]], citations: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Load papers, authors, categories, keywords, years and citations idempotently."""
        ...

    def run_read(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a read-only query and return list of row dictionaries."""
        ...

    def schema_text(self) -> str:
        """Return formatted schema string (node labels, properties, edge types)."""
        ...

    def stats(self) -> dict[str, Any]:
        """Return graph statistics (node counts, edge counts, density)."""
        ...
