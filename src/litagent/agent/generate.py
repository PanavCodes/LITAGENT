"""Answer generation node using strict evidence grounding and citation resolution.

Per PRD.md §7.5 (FR-AGT-9) & ARCHITECTURE.md §3:
Answers only from retrieved evidence, formats citations as [P#],
and treats evidence as untrusted data.
"""

from __future__ import annotations

import logging
from pathlib import Path

from litagent.agent.state import AgentState
from litagent.llm import BaseLLM
from litagent.schemas import Citation

logger = logging.getLogger(__name__)


def load_answer_template(path: str | Path = "prompts/answer.md") -> str:
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return (
        "Answer the question strictly from the evidence below.\n"
        "Evidence:\n{evidence_text}\n\n"
        "Question: {question}\nAnswer:"
    )


class GenerateNode:
    """Generates academic answers with citations strictly grounded in evidence."""

    def __init__(self, llm: BaseLLM, prompt_path: str | Path = "prompts/answer.md") -> None:
        self.llm = llm
        self.prompt_path = prompt_path

    def __call__(self, state: AgentState) -> AgentState:
        question = state.get("question", "")
        evidence_items = state.get("evidence", [])

        if not evidence_items:
            state["answer"] = "This information was not found in the corpus."
            state["citations"] = []
            return state

        # Format evidence blocks with untrusted data delimiters
        formatted_evidence = []
        paper_map: dict[str, str] = {}  # paper_id -> tag P1, P2...
        citations: list[Citation] = []

        for e in evidence_items:
            formatted_evidence.append(
                f'<evidence id="{e.id}" source="{e.source}">\n{e.text}\n</evidence>'
            )
            for pid in e.paper_ids:
                if pid and pid not in paper_map:
                    tag = f"P{len(paper_map) + 1}"
                    paper_map[pid] = tag
                    citations.append(
                        Citation(
                            tag=tag,
                            paper_id=pid,
                            title=f"Paper {pid}",
                            url=f"https://arxiv.org/abs/{pid}",
                        )
                    )

        evidence_text = "\n\n".join(formatted_evidence)
        template = load_answer_template(self.prompt_path)
        prompt = template.replace("{evidence_text}", evidence_text).replace("{question}", question)

        raw_answer = self.llm.invoke(prompt)

        state["answer"] = raw_answer.strip()
        state["citations"] = citations
        state["llm_calls"] = state.get("llm_calls", 0) + 1
        return state
