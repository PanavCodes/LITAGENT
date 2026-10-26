"""Unit tests for Phase 0 setup, config, FakeLLM, and caching."""

from __future__ import annotations

import tempfile
from pathlib import Path

from litagent.config import load_config
from litagent.llm import CachedLLMWrapper, FakeLLM
from litagent.tracing import Tracer


def test_config_loads_defaults() -> None:
    config = load_config()
    assert config.seed == 42
    assert config.chunking.size == 2000
    assert config.chunking.overlap == 50
    assert config.retrieval.rrf_k == 60
    assert config.graph.default_limit == 50
    assert "cs.IR" in config.data.categories


def test_fake_llm_deterministic() -> None:
    fake = FakeLLM(
        default_response="Default answer",
        responses={"cypher": "MATCH (p:Paper) RETURN p LIMIT 5"},
        response_sequence=["First", "Second"],
    )
    # Sequential pop
    assert fake.invoke("any query") == "First"
    assert fake.invoke("any query 2") == "Second"
    # Fallback to dictionary pattern match
    assert fake.invoke("generate cypher for papers") == "MATCH (p:Paper) RETURN p LIMIT 5"
    # Fallback to default
    assert fake.invoke("other question") == "Default answer"
    assert len(fake.calls) == 4


def test_disk_cache_hit() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        underlying = FakeLLM(default_response="Cached answer")
        cached_llm = CachedLLMWrapper(
            llm=underlying,
            model_name="test-model",
            cache_dir=tmp_dir,
        )

        # Call 1: invokes underlying LLM
        res1 = cached_llm.invoke("What is DPR?")
        assert res1 == "Cached answer"
        assert len(underlying.calls) == 1

        # Call 2: should hit disk cache, underlying LLM calls must remain 1
        res2 = cached_llm.invoke("What is DPR?")
        assert res2 == "Cached answer"
        assert len(underlying.calls) == 1


def test_tracer_records_events() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracer = Tracer(run_id="test_run", traces_dir=tmp_dir)
        tracer.record(
            node="router",
            start_time=100.0,
            input_summary="Input test",
            output_summary="Output test",
        )
        summary = tracer.get_summary()
        assert summary["run_id"] == "test_run"
        assert summary["total_events"] == 1
        assert "router" in summary["nodes"]

        # Verify JSONL exists and content parses
        trace_file = Path(tmp_dir) / "test_run.jsonl"
        assert trace_file.exists()
        lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert "router" in lines[0]
