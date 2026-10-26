"""Query rewrite node for CRAG bounded correction loop.

Per PRD.md §7.5 (FR-AGT-8) & ARCHITECTURE.md §3:
Rewrites retrieval query using missing information identified by the grader,
potentially switches route, and increments loop iteration counter.
"""

from __future__ import annotations

import logging

from litagent.agent.state import AgentState
from litagent.llm import BaseLLM

logger = logging.getLogger(__name__)


class RewriteNode:
    """Reformulates query to retrieve missing evidence."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> AgentState:
        question = state.get("question", "")
        missing_info = state.get("missing_info", [])
        current_route = state.get("route", "vector")
        cypher_status = state.get("cypher_status")

        # Route switching rule: if graph returned empty or error, escalate to both or vector
        if current_route == "graph" and cypher_status in {"empty", "error"}:
            logger.info("Switching route from graph to both due to empty/error graph outcome.")
            state["route"] = "both"

        missing_text = "; ".join(missing_info) if missing_info else "additional context"
        prompt = (
            f"Original research question: {question}\n"
            f"The initial retrieval was insufficient. Missing information: {missing_text}\n"
            "Reformulate this query into a concise search query to find the missing facts.\n"
            "Output ONLY the rewritten search query:"
        )

        rewritten = self.llm.invoke(prompt).strip()
        # Fallback if empty
        if not rewritten or len(rewritten) < 5:
            rewritten = f"{question} {missing_text}"

        state["current_query"] = rewritten
        state["iteration"] = state.get("iteration", 0) + 1
        state["llm_calls"] = state.get("llm_calls", 0) + 1
        return state
