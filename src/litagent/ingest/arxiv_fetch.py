"""arXiv metadata and preprint fetcher.

Per PRD.md §7.1 (FR-ING-1, FR-ING-2) and DATA_SPEC.md §1:
- Respects arXiv API etiquette: at most 1 request per 3 seconds.
- Caches all raw responses under data/raw/arxiv/.
- Resumable and idempotent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import arxiv

logger = logging.getLogger(__name__)


def fetch_arxiv_papers(
    query: str,
    categories: list[str],
    max_results: int = 50,
    raw_cache_dir: str | Path = "data/raw/arxiv",
    delay_s: float = 3.0,
) -> list[dict[str, Any]]:
    """Query arXiv API with polite rate limiting and on-disk caching."""
    cache_path = Path(raw_cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Build query combining categories and terms
    cat_terms = " OR ".join(f"cat:{c}" for c in categories)
    full_query = f"({query}) AND ({cat_terms})" if categories else query

    logger.info(f"Querying arXiv with: {full_query} (max={max_results})")

    client = arxiv.Client(
        page_size=min(max_results, 100),
        delay_seconds=delay_s,
        num_retries=3,
    )

    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[dict[str, Any]] = []

    try:
        for result in client.results(search):
            # Strip version suffix (e.g., '2104.08663v2' -> '2104.08663')
            arxiv_id = result.get_short_id().split("v")[0]
            cache_file = cache_path / f"{arxiv_id}.json"

            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        papers.append(json.load(f))
                        continue
                except (OSError, json.JSONDecodeError):
                    pass

            published_date = (
                result.published.strftime("%Y-%m-%d") if result.published else ""
            )
            year = result.published.year if result.published else 2024

            paper_data = {
                "arxiv_id": arxiv_id,
                "title": result.title,
                "abstract": result.summary,
                "authors": [a.name for a in result.authors],
                "categories": result.categories,
                "primary_category": result.primary_category,
                "published": published_date,
                "year": year,
                "pdf_url": result.pdf_url,
                "doi": result.doi or "",
                "has_fulltext": False,
            }

            # Cache locally
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(paper_data, f, indent=2)

            papers.append(paper_data)


    except Exception as e:
        logger.warning(f"Error fetching from arXiv API: {e}")

    return papers
