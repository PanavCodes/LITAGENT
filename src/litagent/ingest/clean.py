"""Data cleaning and deduplication rules for LitAgent.

Per PRD.md §7.1 (FR-ING-4) and DATA_SPEC.md §2:
- Drop records with missing title or abstract.
- Normalise titles (lower-case, punctuation removal, whitespace collapse).
- Deduplicate by arXiv ID, DOI, and normalised title.
"""

from __future__ import annotations

import re
from typing import Any


def normalise_text(text: str | None) -> str:
    """Normalise string: lower-case, remove punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def is_valid_paper(record: dict[str, Any]) -> tuple[bool, str]:
    """Check if record meets minimum data quality standards."""
    title = str(record.get("title", "")).strip()
    abstract = str(record.get("abstract", "")).strip()

    if not title:
        return False, "missing_title"
    if not abstract:
        return False, "missing_abstract"
    if len(abstract) < 50:
        return False, "abstract_too_short"

    return True, "valid"


def deduplicate_papers(
    papers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deduplicate records by arxiv_id, doi, and normalised title."""
    seen_arxiv_ids: set[str] = set()
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()

    unique_papers: list[dict[str, Any]] = []
    dropped_stats = {
        "missing_fields": 0,
        "duplicate_arxiv_id": 0,
        "duplicate_doi": 0,
        "duplicate_title": 0,
    }

    for paper in papers:
        valid, _reason = is_valid_paper(paper)
        if not valid:
            dropped_stats["missing_fields"] += 1
            continue

        arxiv_id = str(paper.get("arxiv_id", "")).strip()
        doi = str(paper.get("doi", "")).strip()
        norm_title = normalise_text(paper.get("title", ""))

        if arxiv_id and arxiv_id in seen_arxiv_ids:
            dropped_stats["duplicate_arxiv_id"] += 1
            continue

        if doi and doi in seen_dois:
            dropped_stats["duplicate_doi"] += 1
            continue

        if norm_title and norm_title in seen_titles:
            dropped_stats["duplicate_title"] += 1
            continue

        if arxiv_id:
            seen_arxiv_ids.add(arxiv_id)
        if doi:
            seen_dois.add(doi)
        if norm_title:
            seen_titles.add(norm_title)

        unique_papers.append(paper)

    return unique_papers, dropped_stats
