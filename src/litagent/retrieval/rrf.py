"""Reciprocal Rank Fusion (RRF) implementation.

Per PRD.md §7.3 (FR-VEC-5) & ARCHITECTURE.md §4.1:
Combines ranked candidate lists using:
score(d) = sum(1 / (k + rank_i(d)))
where rank is 1-indexed (1, 2, ...), default k = 60.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_n: int = 20,
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse multiple ranked lists using RRF.

    Returns:
        List of (chunk_id, rrf_score, {list_idx: rank_1_indexed}) sorted descending.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for list_idx, candidate_list in enumerate(ranked_lists):
        source_name = f"rank_{list_idx}"
        for rank_zero, (chunk_id, _) in enumerate(candidate_list):
            rank = rank_zero + 1  # 1-indexed rank
            reciprocal_score = 1.0 / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + reciprocal_score

            if chunk_id not in ranks:
                ranks[chunk_id] = {}
            ranks[chunk_id][source_name] = rank

    sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    results: list[tuple[str, float, dict[str, int]]] = []
    for chunk_id, score in sorted_candidates[:top_n]:
        results.append((chunk_id, score, ranks.get(chunk_id, {})))

    return results
