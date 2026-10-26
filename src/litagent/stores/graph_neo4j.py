"""Neo4j graph store implementation.

Per PRD.md §7.2 (FR-KG-1, FR-KG-2, FR-KG-3) and DATA_SPEC.md §3:
Manages Neo4j constraints, idempotent UNWIND MERGE batch loading,
and read-only transaction execution.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import GraphDatabase

from litagent.stores.graph_networkx import normalise_author

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """Neo4j graph store with constraints, batch MERGE loaders, and read transactions."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "litagentpassword",
        query_timeout_s: int = 10,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.query_timeout_s = query_timeout_s
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def create_constraints(self) -> None:
        """Apply constraints per DATA_SPEC.md §3."""
        ddl_statements = [
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE;",
            "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.name_norm IS UNIQUE;",
            "CREATE CONSTRAINT keyword_id IF NOT EXISTS FOR (k:Keyword) REQUIRE k.text IS UNIQUE;",
            "CREATE CONSTRAINT category_id IF NOT EXISTS FOR (c:Category) REQUIRE c.code IS UNIQUE;",
            "CREATE CONSTRAINT year_id IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE;",
            "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title);",
        ]
        with self.driver.session() as session:
            for ddl in ddl_statements:
                session.run(ddl)

    def load_corpus(
        self, papers: list[dict[str, Any]], citations: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Batch UNWIND MERGE loader for nodes and edges."""
        self.create_constraints()

        # Batch 1: Papers
        paper_rows = []
        author_edges = []
        cat_edges = []
        kw_edges = []

        for p in papers:
            p_id = p["arxiv_id"]
            paper_rows.append({
                "arxiv_id": p_id,
                "title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "year": p.get("year", 2024),
                "doi": p.get("doi", ""),
                "cited_by_count": p.get("cited_by_count", 0),
                "url": p.get("pdf_url", ""),
                "has_fulltext": p.get("has_fulltext", False),
            })

            for idx, a in enumerate(p.get("authors", [])):
                author_edges.append({
                    "arxiv_id": p_id,
                    "name": a,
                    "name_norm": normalise_author(a),
                    "position": idx,
                })

            for cat in p.get("categories", []):
                cat_edges.append({
                    "arxiv_id": p_id,
                    "code": cat,
                    "primary": (cat == p.get("primary_category")),
                })

            for r, kw in enumerate(p.get("keywords", [])):
                kw_edges.append({
                    "arxiv_id": p_id,
                    "text": kw.lower().strip(),
                    "rank": r,
                })

        with self.driver.session() as session:
            # Load Papers and Years
            session.run(
                """
                UNWIND $rows AS r
                MERGE (p:Paper {arxiv_id: r.arxiv_id})
                SET p.title = r.title, p.abstract = r.abstract, p.year = r.year,
                    p.doi = r.doi, p.cited_by_count = r.cited_by_count,
                    p.url = r.url, p.has_fulltext = r.has_fulltext
                MERGE (y:Year {value: r.year})
                MERGE (p)-[:PUBLISHED_IN]->(y)
                """,
                rows=paper_rows,
            )

            # Load Authors
            session.run(
                """
                UNWIND $rows AS r
                MERGE (a:Author {name_norm: r.name_norm})
                ON CREATE SET a.name = r.name
                WITH a, r
                MATCH (p:Paper {arxiv_id: r.arxiv_id})
                MERGE (a)-[w:WROTE]->(p)
                SET w.position = r.position
                """,
                rows=author_edges,
            )

            # Load Categories
            session.run(
                """
                UNWIND $rows AS r
                MERGE (c:Category {code: r.code})
                WITH c, r
                MATCH (p:Paper {arxiv_id: r.arxiv_id})
                MERGE (p)-[ic:IN_CATEGORY]->(c)
                SET ic.primary = r.primary
                """,
                rows=cat_edges,
            )

            # Load Keywords
            session.run(
                """
                UNWIND $rows AS r
                MERGE (k:Keyword {text: r.text})
                WITH k, r
                MATCH (p:Paper {arxiv_id: r.arxiv_id})
                MERGE (p)-[hk:HAS_KEYWORD]->(k)
                SET hk.rank = r.rank
                """,
                rows=kw_edges,
            )

            # Load Citations
            session.run(
                """
                UNWIND $rows AS r
                MATCH (s:Paper {arxiv_id: r.source_id})
                MATCH (t:Paper {arxiv_id: r.target_id})
                MERGE (s)-[:CITES]->(t)
                """,
                rows=citations,
            )

        return self.stats()

    def run_read(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run read-only query in explicit read transaction with timeout."""
        params = params or {}

        def _execute(tx: Any) -> list[dict[str, Any]]:
            result = tx.run(query, params)
            return [dict(record) for record in result]

        with self.driver.session() as session:
            return session.execute_read(_execute)

    def schema_text(self) -> str:
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
        with self.driver.session() as session:
            res = session.run(
                """
                MATCH (p:Paper)
                OPTIONAL MATCH (p)-[c:CITES]->(:Paper)
                RETURN count(DISTINCT p) AS paper_count, count(c) AS cites_count
                """
            ).single()
            return {
                "paper_count": res["paper_count"] if res else 0,
                "cites_edge_count": res["cites_count"] if res else 0,
            }
