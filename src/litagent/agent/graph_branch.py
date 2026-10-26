"""Graph retrieval branch node for LangGraph.

Per PRD.md §7.5 (FR-AGT-4) & ARCHITECTURE.md §4.2:
Runs text-to-Cypher and converts result rows into standardized EvidenceItems.
"""

from __future__ import annotations

import logging
from typing import Any

from litagent.agent.state import AgentState
from litagent.retrieval.text2cypher import Text2CypherRetriever
from litagent.schemas import EvidenceItem

logger = logging.getLogger(__name__)


def row_to_text(row: dict[str, Any]) -> str:
    """Format row dictionary into a concise academic sentence."""
    if "author" in row and "paper" in row:
        return f"Author {row['author']} wrote the paper '{row['paper']}'."
    if "citing_paper" in row and "cited_paper" in row:
        return f"The paper '{row['citing_paper']}' cites '{row['cited_paper']}'."
    if "shared_reference" in row:
        return f"Paper '{row['shared_reference']}' is cited by both queried works."
    if "paper_path" in row:
        path_str = " -> ".join(row["paper_path"])
        return f"Citation path: {path_str}."
    if "paper_count" in row:
        return f"Total count: {row['paper_count']} papers."
    if "count" in row:
        return f"Total count: {row['count']}."

    # Generic key-value formatting
    parts = [f"{k}: {v}" for k, v in row.items() if v]
    return ", ".join(parts) + "."


class GraphBranchNode:
    """Executes Graph retrieval branch in LangGraph workflow."""

    def __init__(self, text2cypher: Text2CypherRetriever) -> None:
        self.text2cypher = text2cypher

    def __call__(self, state: AgentState) -> AgentState:
        query = state.get("current_query") or state.get("question", "")
        result = self.text2cypher.query(query)

        state["cypher"] = result.cypher
        state["cypher_status"] = result.status
        state["rows"] = result.rows
        state["llm_calls"] = state.get("llm_calls", 0) + (2 if result.status == "retried" else 1)

        evidence: list[EvidenceItem] = state.get("evidence", [])
        start_idx = len(evidence) + 1

        for idx, row in enumerate(result.rows[:15], start=start_idx):
            paper_ids = []
            for k in ["arxiv_id", "citing_id", "cited_id"]:
                if row.get(k):
                    paper_ids.append(str(row[k]))

            item = EvidenceItem(
                id=f"E{idx}",
                source="graph",
                text=row_to_text(row),
                paper_ids=paper_ids,
                scores={"graph_status": 1.0 if result.status in {"ok", "retried"} else 0.0},
            )
            evidence.append(item)

        state["evidence"] = evidence
        return state
