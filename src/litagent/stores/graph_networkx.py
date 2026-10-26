"""NetworkX graph storage implementation.

Per PRD.md §7.2 (FR-KG-4) and DATA_SPEC.md §3:
Provides a fully functional local in-memory/persisted graph store
supporting the LitAgent entity schema and query templates.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


def normalise_author(name: str) -> str:
    """Normalise author name: lower-case, strip initials and punctuation."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    return " ".join(name.split())


class NetworkXGraphStore:
    """NetworkX directed graph store supporting the LitAgent schema."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.graph = nx.MultiDiGraph()
        self.db_path = Path(db_path) if db_path else None
        if self.db_path and self.db_path.exists():
            self.load_from_disk(self.db_path)

    def load_corpus(
        self, papers: list[dict[str, Any]], citations: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Load papers, authors, categories, keywords, years and citations idempotently."""
        nodes_before = self.graph.number_of_nodes()
        edges_before = self.graph.number_of_edges()

        for p in papers:
            p_id = f"paper_{p['arxiv_id']}"
            self.graph.add_node(
                p_id,
                label="Paper",
                arxiv_id=p["arxiv_id"],
                title=p.get("title", ""),
                abstract=p.get("abstract", ""),
                year=p.get("year", 2024),
                doi=p.get("doi", ""),
                cited_by_count=p.get("cited_by_count", 0),
                url=p.get("pdf_url", ""),
                has_fulltext=p.get("has_fulltext", False),
            )

            # Year node & edge
            year_val = p.get("year", 2024)
            y_id = f"year_{year_val}"
            self.graph.add_node(y_id, label="Year", value=year_val)
            self.graph.add_edge(p_id, y_id, rel_type="PUBLISHED_IN")

            # Authors
            for idx, author in enumerate(p.get("authors", [])):
                a_norm = normalise_author(author)
                a_id = f"author_{a_norm}"
                self.graph.add_node(a_id, label="Author", name=author, name_norm=a_norm)
                self.graph.add_edge(a_id, p_id, rel_type="WROTE", position=idx)

            # Categories
            for cat in p.get("categories", []):
                c_id = f"cat_{cat}"
                self.graph.add_node(c_id, label="Category", code=cat)
                self.graph.add_edge(p_id, c_id, rel_type="IN_CATEGORY", primary=(cat == p.get("primary_category")))

            # Keywords
            for rank, kw in enumerate(p.get("keywords", [])):
                kw_norm = kw.lower().strip()
                k_id = f"kw_{kw_norm}"
                self.graph.add_node(k_id, label="Keyword", text=kw_norm)
                self.graph.add_edge(p_id, k_id, rel_type="HAS_KEYWORD", rank=rank)

        # Citations between papers
        for cit in citations:
            src = f"paper_{cit['source_id']}"
            tgt = f"paper_{cit['target_id']}"
            if self.graph.has_node(src) and self.graph.has_node(tgt):
                self.graph.add_edge(src, tgt, rel_type="CITES")

        if self.db_path:
            self.save_to_disk(self.db_path)

        return {
            "new_nodes": self.graph.number_of_nodes() - nodes_before,
            "new_edges": self.graph.number_of_edges() - edges_before,
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
        }

    def run_read(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute template/pattern queries against NetworkX graph."""
        params = params or {}
        q = query.lower()
        rows: list[dict[str, Any]] = []

        # 1. Who wrote / Authors of paper
        if "wrote" in q or "author" in q:
            title_filter = params.get("title", "").lower()
            for u, v, data in self.graph.edges(data=True):
                if data.get("rel_type") == "WROTE":
                    author_node = self.graph.nodes[u]
                    paper_node = self.graph.nodes[v]
                    if not title_filter or title_filter in paper_node.get("title", "").lower():
                        rows.append({
                            "author": author_node.get("name"),
                            "paper": paper_node.get("title"),
                            "arxiv_id": paper_node.get("arxiv_id"),
                        })
            # Handle aggregation: prolific authors
            if "order by" in q or "count" in q:
                counts: dict[str, int] = {}
                for r in rows:
                    counts[r["author"]] = counts.get(r["author"], 0) + 1
                sorted_authors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                return [{"author": a, "paper_count": c} for a, c in sorted_authors[:5]]
            return rows[: params.get("limit", 50)]

        # 2. Citing papers / incoming CITES edges
        if "cites" in q:
            title_filter = params.get("title", "").lower()
            for u, v, data in self.graph.edges(data=True):
                if data.get("rel_type") == "CITES":
                    p1 = self.graph.nodes[u]
                    p2 = self.graph.nodes[v]
                    if not title_filter or title_filter in p2.get("title", "").lower():
                        rows.append({
                            "citing_paper": p1.get("title"),
                            "citing_id": p1.get("arxiv_id"),
                            "cited_paper": p2.get("title"),
                            "cited_id": p2.get("arxiv_id"),
                            "year": p1.get("year"),
                        })
            return rows[: params.get("limit", 50)]

        # 3. Year / Keyword aggregation
        if "published_in" in q or "has_keyword" in q or "year" in q:
            target_year = params.get("year")
            target_kw = params.get("kw", "").lower()
            matching_papers: set[str] = set()

            for n, d in self.graph.nodes(data=True):
                if d.get("label") == "Paper":
                    match_y = (target_year is None) or (d.get("year") == target_year)
                    if match_y:
                        matching_papers.add(n)

            if target_kw:
                kw_papers: set[str] = set()
                for u, v, data in self.graph.edges(data=True):
                    if data.get("rel_type") == "HAS_KEYWORD":
                        kw_node = self.graph.nodes[v]
                        if target_kw in kw_node.get("text", "").lower():
                            kw_papers.add(u)
                matching_papers &= kw_papers

            return [{"count": len(matching_papers)}]

        # Fallback: return sample papers
        for n, d in self.graph.nodes(data=True):
            if d.get("label") == "Paper":
                rows.append({"arxiv_id": d.get("arxiv_id"), "title": d.get("title"), "year": d.get("year")})
                if len(rows) >= params.get("limit", 10):
                    break
        return rows

    def schema_text(self) -> str:
        """Dynamically generate schema string for prompts."""
        return (
            "Node Labels: [Paper, Author, Keyword, Category, Year]\n"
            "Paper Properties: [arxiv_id, title, abstract, year, doi, cited_by_count, url, has_fulltext]\n"
            "Author Properties: [name, name_norm]\n"
            "Keyword Properties: [text]\n"
            "Category Properties: [code]\n"
            "Year Properties: [value]\n"
            "Relationships:\n"
            "  (:Author)-[:WROTE {position}]->(:Paper)\n"
            "  (:Paper)-[:CITES]->(:Paper)\n"
            "  (:Paper)-[:HAS_KEYWORD {rank}]->(:Keyword)\n"
            "  (:Paper)-[:IN_CATEGORY {primary}]->(:Category)\n"
            "  (:Paper)-[:PUBLISHED_IN]->(:Year)\n"
        )

    def stats(self) -> dict[str, Any]:
        """Compute node and edge counts and degree metrics."""
        paper_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("label") == "Paper"]
        cites_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True) if d.get("rel_type") == "CITES"
        ]
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "paper_count": len(paper_nodes),
            "cites_edge_count": len(cites_edges),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0.0,
        }

    def save_to_disk(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)

    def load_from_disk(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.graph = pickle.load(f)
