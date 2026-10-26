"""Configuration loader and schema definitions for LitAgent.

Follows AGENTS.md standards: typed settings via pydantic-settings,
default YAML loading, environment overrides, and zero hard-coded values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    provider: Literal["fake", "google-genai", "openai", "anthropic"] = "fake"
    answer_model: str = "gemini-2.5-flash"
    judge_model: str = "gemini-2.5-pro"
    temperature: float = 0.0
    cache_dir: str = "data/cache/llm"
    max_retries: int = 3
    timeout_s: int = 30


class DataConfig(BaseModel):
    query: str = "retrieval-augmented generation OR information retrieval"
    categories: list[str] = Field(default_factory=lambda: ["cs.IR", "cs.CL"])
    year_min: int = 2020
    max_papers: int = 600
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


class ChunkingConfig(BaseModel):
    size: int = 2000
    overlap: int = 50


class EmbeddingConfig(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64


class RetrievalConfig(BaseModel):
    use_bm25: bool = True
    use_dense: bool = True
    use_rrf: bool = True
    use_rerank: bool = True
    candidates_per_retriever: int = 20
    rrf_k: int = 60
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_in: int = 20
    rerank_top_out: int = 5


class GraphConfig(BaseModel):
    backend: Literal["neo4j", "networkx"] = "networkx"
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "litagentpassword"
    query_timeout_s: int = 10
    default_limit: int = 50
    db_path: str = "data/index/graph.pkl"


class AgentConfig(BaseModel):
    grader_threshold: float = 0.6
    max_iterations: int = 2
    max_llm_calls: int = 12
    timeout_s: int = 90
    forced_route: Literal["graph", "vector", "both", "none"] | None = None


class UncertaintyConfig(BaseModel):
    use_self_consistency: bool = False
    samples: int = 3
    abstain_threshold: float = 0.7
    strict_abstain: bool = False
    weights: dict[str, float] = Field(
        default_factory=lambda: {"S1": 0.35, "S2": 0.35, "S5": 0.30}
    )


class AppConfig(BaseSettings):
    seed: int = 42
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file, applying environment overrides."""
    if config_path is None:
        default_path = Path("configs/default.yaml")
        if default_path.exists():
            config_path = default_path

    yaml_data: dict[str, Any] = {}
    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    yaml_data = loaded

    return AppConfig(**yaml_data)


LitAgentConfig = AppConfig

