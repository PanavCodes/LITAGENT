"""Router node for directing queries to optimal retrieval branches.

Per PRD.md §7.5 (FR-AGT-2, FR-AGT-3):
Outputs structured JSON (route, rationale) with fallback to vector.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import yaml

from litagent.agent.state import AgentState
from litagent.llm import BaseLLM

logger = logging.getLogger(__name__)


def load_router_prompt(path: str | Path = "prompts/router_fewshot.yaml") -> tuple[str, str]:
    """Load system prompt and few-shot examples from YAML."""
    p = Path(path)
    if not p.exists():
        return "You are a query router. Return JSON with 'route' and 'rationale'.", ""

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    sys_prompt = data.get("system_prompt", "")
    examples = data.get("examples", [])
    ex_text = []
    for ex in examples:
        ex_text.append(f"Question: {ex['question']}")
        ex_text.append(f'{{"route": "{ex["route"]}", "rationale": "{ex["rationale"]}"}}\n')

    return sys_prompt, "\n".join(ex_text)


class RouterNode:
    """Routes questions into 'graph', 'vector', 'both', or 'none'."""

    def __init__(self, llm: BaseLLM, prompt_path: str | Path = "prompts/router_fewshot.yaml") -> None:
        self.llm = llm
        self.prompt_path = prompt_path

    def route_query(self, question: str, forced_route: str | None = None) -> tuple[str, str]:
        """Determine retrieval route. Returns (route, rationale)."""
        if forced_route in {"graph", "vector", "both", "none"}:
            return forced_route, f"Ablation forced route: {forced_route}"

        sys_prompt, examples = load_router_prompt(self.prompt_path)
        prompt = (
            f"{sys_prompt}\n\n"
            f"Few-Shot Examples:\n{examples}\n\n"
            f"Now classify this question:\nQuestion: {question}\nJSON:"
        )

        response = self.llm.invoke(prompt)

        # Parse JSON
        try:
            # Strip code fences if present
            clean_resp = re.sub(r"^```[a-zA-Z]*\n", "", response.strip())
            clean_resp = re.sub(r"\n```$", "", clean_resp).strip()
            # Find json block
            match = re.search(r"\{.*\}", clean_resp, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                route = parsed.get("route", "").lower().strip()
                rationale = parsed.get("rationale", "Classified by router")
                if route in {"graph", "vector", "both", "none"}:
                    return route, rationale
        except Exception as e:
            logger.warning(f"Router failed to parse JSON output: {e}. Falling back to 'vector'.")

        return "vector", "Fallback to vector retrieval due to parse failure"

    def __call__(self, state: AgentState) -> AgentState:
        question = state.get("question", "")
        forced = state.get("forced_route")
        route, rationale = self.route_query(question, forced_route=forced)

        state["route"] = route  # type: ignore[typeddict-item]
        state["route_rationale"] = rationale

        state["current_query"] = question
        state["iteration"] = state.get("iteration", 0)
        state["llm_calls"] = state.get("llm_calls", 0) + (0 if forced else 1)
        return state
