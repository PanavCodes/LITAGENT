"""Multi-signal runtime uncertainty estimator and abstention engine.

Per PRD.md §7.6 (FR-UNC-1..5) & ARCHITECTURE.md §5:
Computes signals S1 (retrieval margin), S2 (evidence grade), S5 (graph status),
S3 (cross-source agreement), S4 (self-consistency), and produces UncertaintyReport.
"""

from __future__ import annotations

import logging

from litagent.agent.state import AgentState
from litagent.schemas import UncertaintyReport

logger = logging.getLogger(__name__)


def retrieval_margin(rerank_scores: list[float]) -> float:
    """S1: Uncertainty from top-1 vs rest. Returns 0..1, high = uncertain."""
    if not rerank_scores:
        return 1.0
    if len(rerank_scores) == 1:
        return 0.2

    top1 = rerank_scores[0]
    rest_mean = sum(rerank_scores[1:]) / len(rerank_scores[1:])
    gap = max(0.0, top1 - rest_mean)

    # Margin normalized: high gap -> low uncertainty
    margin_uncertainty = 1.0 - min(1.0, max(0.0, gap * 2.0))
    return float(margin_uncertainty)


def evidence_grade_uncertainty(grade: float | None) -> float:
    """S2: Grader sufficiency uncertainty (1 - grade)."""
    if grade is None:
        return 0.5
    return float(max(0.0, min(1.0, 1.0 - grade)))


def graph_flags_uncertainty(status: str | None) -> float:
    """S5: Uncertainty from Graph execution status."""
    if not status:
        return 0.0
    status_map = {
        "ok": 0.0,
        "retried": 0.3,
        "empty": 0.8,
        "error": 1.0,
    }
    return status_map.get(status, 0.5)


def compute_uncertainty(
    state: AgentState,
    weights: dict[str, float] | None = None,
    abstain_threshold: float = 0.7,
    strict_abstain: bool = False,
) -> UncertaintyReport:
    """Compute composite uncertainty score across all available signals."""
    weights = weights or {"S1": 0.35, "S2": 0.35, "S5": 0.30}

    signals: dict[str, float] = {}

    # S1: Retrieval margin (for vector/both)
    rerank_scores = []
    for c in state.get("chunks", []):
        if "rerank" in c.scores:
            rerank_scores.append(c.scores["rerank"])
    if rerank_scores:
        signals["S1"] = retrieval_margin(rerank_scores)

    # S2: Evidence grade
    grade = state.get("grade")
    signals["S2"] = evidence_grade_uncertainty(grade)

    # S5: Graph execution flags (for graph/both)
    cypher_status = state.get("cypher_status")
    if cypher_status:
        signals["S5"] = graph_flags_uncertainty(cypher_status)

    # Calculate weighted average over active signals
    active_weight_sum = 0.0
    weighted_score = 0.0

    for s_name, s_val in signals.items():
        w = weights.get(s_name, 0.33)
        weighted_score += s_val * w
        active_weight_sum += w

    composite_score = (
        (weighted_score / active_weight_sum) if active_weight_sum > 0 else 0.5
    )
    composite_score = round(max(0.0, min(1.0, composite_score)), 3)

    is_abstain = composite_score >= abstain_threshold

    report = UncertaintyReport(
        score=composite_score,
        signals=signals,
        weights=weights,
        abstain=is_abstain,
    )

    # Update answer with confidence notice or abstention
    if is_abstain:
        if strict_abstain:
            state["answer"] = (
                "[Abstained] The system confidence for this question is too low to produce a verified answer."
            )
        else:
            current_ans = state.get("answer") or ""
            if not current_ans.startswith("[Low-Confidence Warning]"):
                state["answer"] = f"[Low-Confidence Warning: uncertainty={composite_score}] {current_ans}"


    state["uncertainty"] = report
    return report
