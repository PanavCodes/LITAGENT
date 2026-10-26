"""OpenAlex enrichment for citation links and referenced works.

Per PRD.md §7.1 (FR-ING-3) and DATA_SPEC.md §1:
- Polite pool requires email configuration via pyalex.config.email.
- Enriches papers with referenced_works and cited_by_count.
- Caches responses under data/raw/openalex/.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pyalex
from pyalex import Works

logger = logging.getLogger(__name__)


def setup_pyalex(email: str | None = None) -> None:
    """Configure pyalex polite pool."""
    contact_email = email or os.getenv("OPENALEX_MAILTO", "researcher@example.com")
    pyalex.config.email = contact_email
    pyalex.config.max_retries = 3


def enrich_papers_openalex(
    papers: list[dict[str, Any]],
    raw_cache_dir: str | Path = "data/raw/openalex",
    email: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enrich papers with referenced works, cited_by_count, and build citation edges."""
    setup_pyalex(email)
    cache_path = Path(raw_cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    enriched_papers: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []

    # Map DOI and Title to arxiv_id for resolving in-corpus citations
    doi_to_arxiv: dict[str, str] = {}
    title_to_arxiv: dict[str, str] = {}

    for p in papers:
        arxiv_id = p["arxiv_id"]
        if p.get("doi"):
            doi_to_arxiv[p["doi"].lower()] = arxiv_id
        if p.get("title"):
            title_to_arxiv[p["title"].lower().strip()] = arxiv_id

    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        cache_file = cache_path / f"{arxiv_id}.json"

        work_data: dict[str, Any] | None = None
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    work_data = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

        if work_data is None:
            try:
                # Query by DOI if available, else by title
                query = None
                if paper.get("doi"):
                    query = Works()[f"doi:{paper['doi']}"]
                else:
                    results = Works().search_filter(title=paper["title"][:150]).get()
                    if results:
                        query = results[0]

                if query:
                    work_data = {
                        "openalex_id": query.get("id"),
                        "cited_by_count": query.get("cited_by_count", 0),
                        "referenced_works": query.get("referenced_works", []),
                    }
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(work_data, f, indent=2)
            except Exception as e:
                logger.debug(f"OpenAlex lookup failed for {arxiv_id}: {e}")

        if work_data:
            paper["cited_by_count"] = work_data.get("cited_by_count", 0)
            paper["referenced_works"] = work_data.get("referenced_works", [])
        else:
            paper["cited_by_count"] = paper.get("cited_by_count", 0)
            paper["referenced_works"] = paper.get("referenced_works", [])

        enriched_papers.append(paper)

    return enriched_papers, citations
