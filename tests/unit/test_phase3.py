"""Unit tests for Phase 3 retrieval components: RRF, CrossEncoder, Cypher guard, and Text2Cypher."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from litagent.retrieval.cypher_guard import (
    CypherSecurityError,
    validate_and_sanitize_cypher,
)
from litagent.retrieval.rerank import CrossEncoderReranker
from litagent.retrieval.rrf import reciprocal_rank_fusion
from litagent.stores.graph_networkx import NetworkXGraphStore


def test_rrf_worked_arithmetic() -> None:
    """Verifies RRF arithmetic against hand-calculated proof.

    Given:
      List 0: [("doc_A", 0.9), ("doc_B", 0.8)]
      List 1: [("doc_B", 0.95), ("doc_A", 0.7)]
    With k = 60:
      doc_A:
        List 0: rank 1 -> 1 / (60 + 1) = 1/61 = 0.0163934426
        List 1: rank 2 -> 1 / (60 + 2) = 1/62 = 0.0161290323
        Total score = 1/61 + 1/62 = 0.0325224749

      doc_B:
        List 0: rank 2 -> 1 / (60 + 2) = 1/62 = 0.0161290323
        List 1: rank 1 -> 1 / (60 + 1) = 1/61 = 0.0163934426
        Total score = 1/62 + 1/61 = 0.0325224749
    """
    list0 = [("doc_A", 0.9), ("doc_B", 0.8)]
    list1 = [("doc_B", 0.95), ("doc_A", 0.7)]

    fused = reciprocal_rank_fusion([list0, list1], k=60, top_n=2)
    assert len(fused) == 2

    expected_score = (1.0 / 61.0) + (1.0 / 62.0)
    for chunk_id, score, _ in fused:
        assert pytest.approx(score, rel=1e-5) == expected_score


def test_cypher_guard_adversarial() -> None:
    # 1. Valid query with write keyword inside a string literal MUST pass
    valid_with_literal = 'MATCH (p:Paper) WHERE p.title CONTAINS "We create a dataset" RETURN p.title;'
    sanitized = validate_and_sanitize_cypher(valid_with_literal)
    assert "MATCH" in sanitized
    assert "LIMIT" in sanitized

    # 2. Write clauses must raise CypherSecurityError
    attacks = [
        "MATCH (p:Paper) DETACH DELETE p;",
        "CREATE (p:Paper {title: 'Hacked'});",
        "MERGE (a:Author {name: 'Attacker'});",
        "MATCH (p:Paper) SET p.title = 'Corrupted';",
        "MATCH (p:Paper) REMOVE p.title;",
        "DROP INDEX paper_title;",
        "LOAD CSV FROM 'http://evil.com/data.csv' AS line RETURN line;",
        "CALL dbms.components() YIELD name RETURN name;",
        "CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'DELETE n', {});",
    ]

    for attack in attacks:
        with pytest.raises(CypherSecurityError):
            validate_and_sanitize_cypher(attack)

    # 3. Limit injection test
    no_limit = "MATCH (p:Paper) RETURN p.title"
    with_limit = validate_and_sanitize_cypher(no_limit, default_limit=25)
    assert "LIMIT 25;" in with_limit


def test_cross_encoder_rerank() -> None:
    reranker = CrossEncoderReranker(top_n=2, use_model=False)
    candidates = [
        {"chunk_id": "c1", "text": "Dense retrieval using dual-encoders for passage matching."},
        {"chunk_id": "c2", "text": "The football world cup took place in summer."},
        {"chunk_id": "c3", "text": "Deep neural networks for dense passage retrieval."},
    ]

    ranked = reranker.rerank("dense passage retrieval", candidates)
    assert len(ranked) == 2
    assert ranked[0][0] in {"c1", "c3"}


def test_all_fewshot_examples_execute() -> None:
    store = NetworkXGraphStore()
    store.load_corpus(
        papers=[{
            "arxiv_id": "2004.04906",
            "title": "Dense Passage Retrieval",
            "abstract": "Dense dual encoders.",
            "authors": ["Vladimir Karpukhin"],
            "year": 2024,
            "categories": ["cs.IR"],
            "keywords": ["reranking"],
            "cited_by_count": 25,
        }],
        citations=[],
    )

    fewshot_file = Path("prompts/cypher_fewshot.yaml")
    assert fewshot_file.exists()

    with open(fewshot_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for ex in data.get("examples", []):
        cypher = ex["cypher"]
        # Must pass guard
        safe = validate_and_sanitize_cypher(cypher)
        # Must execute against GraphStore
        rows = store.run_read(safe)
        assert isinstance(rows, list)
