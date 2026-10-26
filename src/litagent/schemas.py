"""Pydantic schemas and typed state definitions for LitAgent.

Follows ARCHITECTURE.md §2:
Defines EvidenceItem, Citation, UncertaintyReport, TraceEvent, and AgentState.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ChunkHit(BaseModel):
    chunk_id: str
    paper_id: str
    text: str
    scores: dict[str, float] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    id: str  # "E1", "E2", ...
    source: Literal["graph", "vector"]
    text: str  # Row rendered as a sentence, or chunk text
    paper_ids: list[str] = Field(default_factory=list)  # arXiv IDs
    scores: dict[str, float] = Field(default_factory=dict)


class Citation(BaseModel):
    tag: str  # "P1", "P2", ...
    paper_id: str  # arXiv ID
    title: str
    url: str  # https://arxiv.org/abs/<id>


class UncertaintyReport(BaseModel):
    score: float  # 0.0 certain .. 1.0 uncertain
    signals: dict[str, float] = Field(default_factory=dict)  # S1, S2, S3, S4, S5
    weights: dict[str, float] = Field(default_factory=dict)
    abstain: bool = False


class TraceEvent(BaseModel):
    node: str
    start: float
    duration: float
    input_summary: str
    output_summary: str
    error: str | None = None


class AgentState(TypedDict, total=False):
    question: str
    current_query: str
    route: Literal["graph", "vector", "both", "none"]
    route_rationale: str
    forced_route: str | None
    cypher: str | None
    cypher_status: Literal["ok", "empty", "error", "retried"] | None
    rows: list[dict[str, Any]]
    chunks: list[ChunkHit]
    evidence: list[EvidenceItem]
    grade: float | None
    missing_info: list[str]
    iteration: int
    llm_calls: int
    answer: str | None
    citations: list[Citation]
    uncertainty: UncertaintyReport | None
    trace: list[dict[str, Any]]
    errors: list[str]
