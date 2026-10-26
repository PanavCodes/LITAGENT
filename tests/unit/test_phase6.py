"""Unit tests for Phase 6: Evaluation, Question Generation, Bootstrap, and Calibration.

Per AGENTS.md §4 and PHASES.md Phase 6:
- Tests question generators (graph, vector, hybrid, out-of-scope)
- Tests stratified 30/70 split
- Tests bootstrap confidence intervals (paper 12x20 & extended 1,000)
- Tests calibration (AUROC, ECE, dev-only logistic fit)
- Tests evaluation metric computation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from litagent.eval.bootstrap import (
    run_extended_bootstrap,
    run_paired_bootstrap,
    run_paper_bootstrap,
)
from litagent.eval.calibration import (
    compute_auroc,
    compute_ece,
    fit_logistic_dev,
)
from litagent.eval.qa_graph import generate_graph_qa_pairs
from litagent.eval.qa_vector import (
    create_eval_dataset_and_split,
    generate_hybrid_qa_pairs,
    generate_out_of_scope_qa_pairs,
    generate_vector_qa_pairs,
)
from litagent.eval.ragas_runner import evaluate_single_sample
from litagent.stores.graph_networkx import NetworkXGraphStore


def test_graph_qa_generation():
    """Verify graph QA generator creates valid questions with verification metadata."""
    store = NetworkXGraphStore()
    papers = [
        {"arxiv_id": "P01", "title": "Paper One", "year": 2023, "categories": ["cs.IR"], "authors": ["Alice"]},
        {"arxiv_id": "P02", "title": "Paper Two", "year": 2024, "categories": ["cs.CL"], "authors": ["Bob"]},
    ]
    citations = [{"source_id": "P01", "target_id": "P02"}]
    store.load_corpus(papers, citations)

    pairs = generate_graph_qa_pairs(store, max_questions=20)
    assert len(pairs) > 0
    # Verify at least one question has verified metadata
    verified_count = sum(1 for p in pairs if p.get("verified") is True)
    assert verified_count >= 1
    # Check T1 lookup
    t1 = [p for p in pairs if p["type"] == "T1_lookup"]
    if t1:
        assert "Alice" in t1[0]["gold_entities"]


def test_vector_hybrid_oos_qa_generation():
    """Verify vector, hybrid, and out-of-scope question generators."""
    chunks = [
        {"chunk_id": "c1", "title": "Paper One", "text": "Paper One proposes a dense retrieval method."},
        {"chunk_id": "c2", "title": "Paper Two", "text": "Paper Two uses cross-encoder reranking."},
    ]
    vec_pairs = generate_vector_qa_pairs(chunks, max_questions=10)
    assert len(vec_pairs) == 2
    # Gate 6 requires >= 25% verified
    verified_rate = sum(1 for p in vec_pairs if p.get("verified") is True) / len(vec_pairs)
    assert verified_rate >= 0.25

    hybrid_pairs = generate_hybrid_qa_pairs([{"title": "P1", "year": 2024}], max_questions=5)
    assert len(hybrid_pairs) == 1
    assert hybrid_pairs[0]["expected_route"] == "both"

    oos_pairs = generate_out_of_scope_qa_pairs()
    assert len(oos_pairs) >= 10
    assert oos_pairs[0]["expected_route"] == "out_of_scope"


def test_dataset_and_split_stratification():
    """Verify dataset split creates stratified 30/70 dev/test split."""
    chunks = [{"chunk_id": f"c_{i}", "title": f"Paper {i}", "text": f"Content {i}"} for i in range(10)]
    v_pairs = generate_vector_qa_pairs(chunks, max_questions=10)
    oos_pairs = generate_out_of_scope_qa_pairs()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        split_data = create_eval_dataset_and_split(
            graph_qa=[],
            vector_qa=v_pairs,
            hybrid_qa=[],
            oos_qa=oos_pairs,
            output_dir=tmp_path,
            seed=42,
        )
        assert (tmp_path / "questions.jsonl").exists()
        assert (tmp_path / "split.json").exists()
        assert split_data["dev_count"] > 0
        assert split_data["test_count"] > 0
        assert 0.20 <= split_data["dev_ratio"] <= 0.40



def test_bootstrap_statistics():
    """Verify bootstrap calculations produce valid means and non-negative standard deviations."""
    sample_records = [
        {"expected_route": "vector", "faithfulness": 0.8, "answer_relevancy": 0.85},
        {"expected_route": "vector", "faithfulness": 0.9, "answer_relevancy": 0.95},
        {"expected_route": "graph", "faithfulness": 0.7, "answer_relevancy": 0.75},
        {"expected_route": "graph", "faithfulness": 0.8, "answer_relevancy": 0.80},
    ]
    res = run_paper_bootstrap(sample_records, ["faithfulness", "answer_relevancy"], n_resamples=5, sample_size=4)
    assert "faithfulness" in res
    assert 0.0 <= res["faithfulness"]["mean"] <= 1.0
    assert res["faithfulness"]["sd"] >= 0.0

    ext_res = run_extended_bootstrap(sample_records, ["faithfulness"], n_resamples=50)
    assert "faithfulness" in ext_res
    assert ext_res["faithfulness"]["p025"] <= ext_res["faithfulness"]["p975"]

    paired = run_paired_bootstrap(sample_records, sample_records, "faithfulness", n_resamples=20)
    assert paired["mean_diff"] == 0.0


def test_calibration_and_metrics():
    """Verify AUROC, ECE, logistic dev fitting, and metric calculations."""
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    auroc = compute_auroc(labels, scores)
    assert auroc == 1.0

    ece = compute_ece(labels, scores, n_bins=5)
    assert 0.0 <= ece <= 1.0

    dev_recs = [
        {"signals": {"S1": 0.9, "S2": 0.8, "S5": 1.0}, "faithfulness": 0.9, "answer_relevancy": 0.9, "expected_route": "vector"},
        {"signals": {"S1": 0.1, "S2": 0.2, "S5": 0.0}, "faithfulness": 0.3, "answer_relevancy": 0.3, "expected_route": "vector"},
    ]
    weights = fit_logistic_dev(dev_recs)
    assert "w_S1" in weights
    assert "bias" in weights

    # Evaluation metric computation test
    m_in_scope = evaluate_single_sample(
        question="What is RAG?",
        answer="RAG combines retrieval and generation.",
        contexts=["RAG combines retrieval and generation."],
        reference="RAG combines retrieval and generation.",
        q_type="V1_factual",
        expected_route="vector",
        actual_route="vector",
    )
    assert m_in_scope["router_acc"] == 1.0
    assert m_in_scope["faithfulness"] >= 0.7

    m_abstain = evaluate_single_sample(
        question="What is the weather?",
        answer="I do not have sufficient information.",
        contexts=[],
        reference="Abstain",
        q_type="out_of_scope",
        expected_route="out_of_scope",
        actual_route="out_of_scope",
        is_abstain=True,
    )
    assert m_abstain["abstain_acc"] == 1.0
