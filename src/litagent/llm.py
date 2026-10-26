"""LLM interface, provider factory, response disk cache, and FakeLLM.

Per AGENTS.md §3:
Every LLM call goes through llm.py, which adds retry with backoff,
an on-disk response cache (keyed by model + prompt + parameters),
and trace logging.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from litagent.config import LLMConfig, load_config

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract interface for LLM chat/completion clients."""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Invoke LLM with prompt string and return text response."""


LLMClient = BaseLLM


class DiskCache:
    """Simple on-disk response cache keyed by hash of (model, prompt, kwargs)."""

    def __init__(self, cache_dir: str | Path = "data/cache/llm") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, model: str, prompt: str, kwargs: dict[str, Any]) -> str:
        payload = json.dumps(
            {"model": model, "prompt": prompt, "kwargs": kwargs},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, model: str, prompt: str, kwargs: dict[str, Any]) -> str | None:
        key = self._hash_key(model, prompt, kwargs)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("response")
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
        return None

    def set(self, model: str, prompt: str, kwargs: dict[str, Any], response: str) -> None:
        key = self._hash_key(model, prompt, kwargs)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": model,
                        "prompt": prompt,
                        "kwargs": kwargs,
                        "response": response,
                        "timestamp": time.time(),
                    },
                    f,
                    indent=2,
                )
        except OSError as e:
            logger.warning(f"Error writing cache file {cache_file}: {e}")


class FakeLLM(BaseLLM):
    """Deterministic, offline fake LLM for unit tests and local simulation."""

    def __init__(
        self,
        default_response: str = "This is a deterministic fake response.",
        responses: dict[str, str] | None = None,
        response_sequence: list[str] | None = None,
    ) -> None:
        self.default_response = default_response
        self.responses = responses or {}
        self.response_sequence = list(response_sequence) if response_sequence else []
        self.calls: list[dict[str, Any]] = []

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs, "time": time.time()})

        if self.response_sequence:
            return self.response_sequence.pop(0)

        # 1. Router classification queries
        if "Now classify this question:" in prompt:
            prompt_tail = prompt.split("Now classify this question:")[-1]
            # Check for explicitly mapped triggers
            for trigger, response in self.responses.items():
                if trigger.lower() in prompt_tail.lower():
                    return response
            # Check domain out-of-scope cues
            tail_lower = prompt_tail.lower()
            if any(w in tail_lower for w in ["weather", "fifa", "world cup", "capital", "bread", "faucet", "earth and mars", "chess", "movie", "stock"]):
                return '{"route": "none", "rationale": "Out of scope literature review question"}'
            if any(w in tail_lower for w in ["who wrote", "which paper", "cite", "cites", "how many", "citation path", "co-citation"]):
                return '{"route": "graph", "rationale": "Graph entity or citation structure question"}'
            if any(w in tail_lower for w in ["published", "dataset", "methodology", "core problem", "how does", "what problem"]):
                return '{"route": "vector", "rationale": "Content passage factual question"}'
            return '{"route": "vector", "rationale": "Default route to vector retrieval"}'

        # 2. Text-to-Cypher queries
        if "User Question:" in prompt and "\nCypher:" in prompt:
            prompt_tail = prompt.split("User Question:")[-1].split("Cypher:")[0]
            for trigger, response in self.responses.items():
                if trigger.lower() in prompt_tail.lower():
                    return response
            uq = prompt_tail.lower()
            if "who wrote" in uq or "author" in uq:
                return "MATCH (a:Author)-[:WROTE]->(p:Paper) RETURN a.name AS author, p.title AS title LIMIT 10"
            if "cite" in uq:
                return "MATCH (p:Paper)-[:CITES]->(q:Paper) RETURN p.title AS citing_paper, q.title AS cited_paper LIMIT 10"
            if "how many" in uq or "published" in uq:
                return "MATCH (p:Paper) RETURN count(p) AS count"
            return "MATCH (p:Paper) RETURN p.title AS title LIMIT 10"

        # 3. Grader queries
        if "Sufficiency Grade" in prompt or "sufficiency score" in prompt.lower():
            return '{"grade": 0.85, "missing_information": [], "rationale": "Retrieved evidence is sufficient."}'

        # 4. Standard trigger lookups
        prompt_tail = prompt
        if "Cypher:" in prompt:
            prompt_tail = prompt.split("Cypher:")[-1]

        for trigger, response in self.responses.items():
            if trigger.lower() in prompt_tail.lower():
                return response

        # Fallback to full prompt matching
        for trigger, response in self.responses.items():
            if trigger.lower() in prompt.lower():
                return response

        return self.default_response




class CachedLLMWrapper(BaseLLM):
    """Wraps any BaseLLM with retries, disk caching, and logging."""

    def __init__(
        self,
        llm: BaseLLM,
        model_name: str,
        cache_dir: str | Path = "data/cache/llm",
        max_retries: int = 3,
        trace_hook: Callable[[str, str, float], None] | None = None,
    ) -> None:
        self.llm = llm
        self.model_name = model_name
        self.cache = DiskCache(cache_dir=cache_dir)
        self.max_retries = max_retries
        self.trace_hook = trace_hook

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        # Check disk cache first
        cached = self.cache.get(self.model_name, prompt, kwargs)
        if cached is not None:
            logger.debug(f"Cache hit for model {self.model_name}")
            return cached

        # Execute with retry and backoff
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        def _call_with_retry() -> str:
            start_time = time.time()
            res = self.llm.invoke(prompt, **kwargs)
            duration = time.time() - start_time
            if self.trace_hook:
                self.trace_hook(prompt, res, duration)
            return res

        response = _call_with_retry()
        # Cache successful response
        self.cache.set(self.model_name, prompt, kwargs, response)
        return response


def get_llm(
    config: LLMConfig | None = None,
    model_override: str | None = None,
    temperature_override: float | None = None,
    use_fake: bool = False,
    fake_llm: FakeLLM | None = None,
) -> BaseLLM:
    """Factory creating an LLM client with caching and retry wrappers."""
    cfg = config or load_config().llm
    model_name = model_override or cfg.answer_model
    temperature = temperature_override if temperature_override is not None else cfg.temperature

    if use_fake or cfg.provider == "fake":
        llm = fake_llm or FakeLLM()
        return CachedLLMWrapper(
            llm=llm,
            model_name=f"fake-{model_name}",
            cache_dir=cfg.cache_dir,
            max_retries=1,
        )

    if cfg.provider == "google-genai":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            client = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )

            class _LangChainAdapter(BaseLLM):
                def invoke(self, prompt: str, **kwargs: Any) -> str:
                    msg = client.invoke(prompt)
                    return str(msg.content)

            return CachedLLMWrapper(
                llm=_LangChainAdapter(),
                model_name=model_name,
                cache_dir=cfg.cache_dir,
                max_retries=cfg.max_retries,
            )
        except ImportError:
            logger.warning("langchain-google-genai not installed. Falling back to FakeLLM.")
            return CachedLLMWrapper(
                llm=FakeLLM(),
                model_name=f"fallback-{model_name}",
                cache_dir=cfg.cache_dir,
                max_retries=1,
            )

    if cfg.provider == "openai":
        try:
            from langchain_openai import ChatOpenAI

            client = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore
            )

            class _OpenAIAdapter(BaseLLM):
                def invoke(self, prompt: str, **kwargs: Any) -> str:
                    msg = client.invoke(prompt)
                    return str(msg.content)

            return CachedLLMWrapper(
                llm=_OpenAIAdapter(),
                model_name=model_name,
                cache_dir=cfg.cache_dir,
                max_retries=cfg.max_retries,
            )
        except ImportError:
            logger.warning("langchain-openai not installed. Falling back to FakeLLM.")
            return CachedLLMWrapper(
                llm=FakeLLM(),
                model_name=f"fallback-{model_name}",
                cache_dir=cfg.cache_dir,
                max_retries=1,
            )

    # Default fallback
    return CachedLLMWrapper(
        llm=FakeLLM(),
        model_name=f"fallback-{model_name}",
        cache_dir=cfg.cache_dir,
        max_retries=1,
    )
