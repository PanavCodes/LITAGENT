"""Text-to-Cypher translation and execution with validation and single retry.

Per PRD.md §7.4 (FR-T2C-1, 3, 5):
Translates natural-language questions to Cypher using schema and few-shot bank,
validates via cypher_guard, executes against GraphStore, and retries once on error.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from litagent.llm import BaseLLM
from litagent.retrieval.cypher_guard import CypherSecurityError, validate_and_sanitize_cypher
from litagent.stores.graph_base import GraphStore

logger = logging.getLogger(__name__)


class CypherResult(BaseModel):
    cypher: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["ok", "empty", "error", "retried"] = "ok"
    error: str | None = None


def load_fewshot_examples(path: str | Path = "prompts/cypher_fewshot.yaml") -> str:
    """Load and format few-shot Cypher examples as prompt text."""
    p = Path(path)
    if not p.exists():
        return ""
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    lines = []
    for ex in data.get("examples", [])[:10]:
        lines.append(f"Question: {ex['question']}")
        lines.append(f"Cypher:\n{ex['cypher'].strip()}\n")
    return "\n".join(lines)


class Text2CypherRetriever:
    """Translates natural language questions to safe Cypher and executes them."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm: BaseLLM,
        fewshot_path: str | Path = "prompts/cypher_fewshot.yaml",
        default_limit: int = 50,
    ) -> None:
        self.graph_store = graph_store
        self.llm = llm
        self.fewshot_path = fewshot_path
        self.default_limit = default_limit

    def _build_prompt(self, question: str, error_feedback: str | None = None) -> str:
        schema = self.graph_store.schema_text()
        fewshots = load_fewshot_examples(self.fewshot_path)

        prompt = (
            "You are an expert Neo4j Cypher generator for scientific literature reviews.\n"
            f"Here is the database schema:\n{schema}\n\n"
            f"Few-Shot Examples:\n{fewshots}\n\n"
            "Instructions:\n"
            "1. Output ONLY a valid, read-only Cypher query (no conversational text or explanations).\n"
            "2. Never use write clauses like CREATE, MERGE, DELETE, SET, or DROP.\n"
            "3. Use case-insensitive string matching: toLower(p.title) CONTAINS toLower('...').\n"
            "4. Return node properties cleanly with aliases (e.g. RETURN p.title AS title).\n"
        )
        if error_feedback:
            prompt += f"\nNote: Your previous attempt failed with: {error_feedback}. Please fix the query.\n"

        prompt += f"\nUser Question: {question}\nCypher:"
        return prompt

    def query(self, question: str) -> CypherResult:
        """Generate Cypher, validate, run against graph, with 1 retry on error."""
        # Attempt 1
        prompt = self._build_prompt(question)
        raw_cypher = self.llm.invoke(prompt)

        try:
            safe_cypher = validate_and_sanitize_cypher(
                raw_cypher, default_limit=self.default_limit
            )
            rows = self.graph_store.run_read(safe_cypher)
            status: Literal["ok", "empty", "error", "retried"] = "ok" if rows else "empty"
            return CypherResult(cypher=safe_cypher, rows=rows, status=status)

        except (CypherSecurityError, Exception) as err1:
            logger.warning(f"Text2Cypher attempt 1 failed: {err1}. Retrying once...")

            # Attempt 2 (Retry with error feedback)
            retry_prompt = self._build_prompt(question, error_feedback=str(err1))
            retry_cypher_raw = self.llm.invoke(retry_prompt)

            try:
                safe_cypher2 = validate_and_sanitize_cypher(
                    retry_cypher_raw, default_limit=self.default_limit
                )
                rows2 = self.graph_store.run_read(safe_cypher2)
                status2: Literal["ok", "empty", "error", "retried"] = "retried" if rows2 else "empty"
                return CypherResult(cypher=safe_cypher2, rows=rows2, status=status2)
            except Exception as err2:
                logger.error(f"Text2Cypher attempt 2 failed: {err2}")
                return CypherResult(cypher=raw_cypher, rows=[], status="error", error=str(err2))
