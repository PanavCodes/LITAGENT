"""Evidence sufficiency grader node for CRAG correction loop.

Per PRD.md §7.5 (FR-AGT-7) & ARCHITECTURE.md §3:
Evaluates retrieved evidence and returns a score [0.0, 1.0] with missing information.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from litagent.agent.state import AgentState
from litagent.llm import BaseLLM

logger = logging.getLogger(__name__)


def load_grader_template(path: str | Path = "prompts/grader.md") -> str:
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return (
        "Evaluate if evidence answers question: {question}\n"
        "Evidence: {evidence_text}\n"
        'Return JSON: {"grade": 0.8, "missing_info": []}'
    )


class GraderNode:
    """Grades retrieved evidence sufficiency."""

    def __init__(self, llm: BaseLLM, prompt_path: str | Path = "prompts/grader.md") -> None:
        self.llm = llm
        self.prompt_path = prompt_path

    def __call__(self, state: AgentState) -> AgentState:
        question = state.get("question", "")
        evidence = state.get("evidence", [])

        if not evidence:
            state["grade"] = 0.0
            state["missing_info"] = ["No evidence was retrieved."]
            return state

        ev_lines = [f"[{e.id}] ({e.source}): {e.text}" for e in evidence[:10]]
        evidence_text = "\n".join(ev_lines)

        template = load_grader_template(self.prompt_path)
        prompt = template.replace("{question}", question).replace("{evidence_text}", evidence_text)

        response = self.llm.invoke(prompt)

        # Parse JSON output
        grade = 0.7 if evidence else 0.0
        missing: list[str] = []

        try:
            clean_resp = re.sub(r"^```[a-zA-Z]*\n", "", response.strip())
            clean_resp = re.sub(r"\n```$", "", clean_resp).strip()
            match = re.search(r"\{.*\}", clean_resp, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                grade = float(parsed.get("grade", grade))
                missing = parsed.get("missing_info", [])
        except Exception as e:
            logger.warning(f"Failed to parse grader response JSON: {e}")

        state["grade"] = max(0.0, min(1.0, grade))
        state["missing_info"] = missing
        state["llm_calls"] = state.get("llm_calls", 0) + 1
        return state
