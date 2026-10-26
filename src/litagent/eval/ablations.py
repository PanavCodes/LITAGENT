"""Ablation benchmark runner across the 5 core configurations and retrieval stages.

Per EVALUATION.md §5 (FR-EVL-6, 7, 8, 9):
- vector_only (Baseline A)
- graph_only (Baseline B)
- joint_baseline (Non-agentic analogue)
- router (Agentic without loop)
- router_crag (Full agentic LitAgent with CRAG loop)
- Retrieval sub-ablation: BM25, dense, RRF, RRF + rerank (Recall@k, MRR)
- Generates results/ablation_table.csv with mean +/- SD and latency/calls
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

from litagent.agent.workflow import LitAgentWorkflow
from litagent.config import LitAgentConfig
from litagent.eval.bootstrap import run_paper_bootstrap
from litagent.eval.ragas_runner import evaluate_single_sample
from litagent.llm import BaseLLM
from litagent.retrieval.dense import DenseRetriever
from litagent.retrieval.rerank import CrossEncoderReranker
from litagent.retrieval.rrf import reciprocal_rank_fusion
from litagent.retrieval.sparse import SparseRetriever
from litagent.retrieval.text2cypher import Text2CypherRetriever
from litagent.stores.graph_networkx import NetworkXGraphStore
from litagent.stores.sparse_bm25 import BM25Store
from litagent.stores.vector_faiss import FAISSVectorStore

logger = logging.getLogger(__name__)


def run_single_config_eval(
    config_name: str,
    config: LitAgentConfig,
    workflow: LitAgentWorkflow,
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run all questions through the workflow for a given ablation config."""
    records: list[dict[str, Any]] = []

    for q in questions:
        q_text = q["question"]
        expected_route = q.get("expected_route", "vector")
        gold_entities = q.get("gold_entities", [])
        ref_answer = q.get("reference_answer", "")

        t0 = time.perf_counter()
        res = workflow.run(q_text)
        latency = time.perf_counter() - t0

        answer = res.get("answer", "")
        actual_route = res.get("route", "vector")
        contexts = [c.text for c in res.get("passages", [])]
        signals = res.get("signals", {})
        uncertainty = res.get("uncertainty_score", 0.5)
        llm_calls = len(res.get("cypher_history", [])) + (2 if res.get("route") != "none" else 0)

        is_abstain = bool(res.get("is_abstain", False)) or "not have sufficient information" in answer.lower()

        metrics = evaluate_single_sample(
            question=q_text,
            answer=answer,
            contexts=contexts,
            reference=ref_answer,
            q_type=q.get("type", "vector"),
            expected_route=expected_route,
            actual_route=actual_route,
            gold_entities=gold_entities,
            is_abstain=is_abstain,
        )

        record = {
            "config": config_name,
            "question_id": q["id"],
            "question": q_text,
            "type": q.get("type", ""),
            "expected_route": expected_route,
            "actual_route": actual_route,
            "is_abstain": is_abstain,
            "latency": latency,
            "llm_calls": llm_calls,
            "uncertainty_score": uncertainty,
            "signals": signals,
            **metrics,
        }
        records.append(record)

    return records


def run_retrieval_sub_ablation(
    bm25_store: BM25Store,
    faiss_store: FAISSVectorStore,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
    reranker: CrossEncoderReranker,
    vector_questions: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:

    """Evaluate retrieval stages on vector questions: Recall@5, Recall@10, MRR."""
    stages = ["r_bm25", "r_dense", "r_rrf", "r_rrf_rerank"]
    results: dict[str, dict[str, float]] = {}

    for stage in stages:
        recalls_5 = []
        recalls_10 = []
        mrrs = []

        for q in vector_questions:
            target_chunk = q.get("source_chunk_id", "")
            q_text = q["question"]

            if stage == "r_bm25":
                ranked = sparse_retriever.retrieve(q_text, top_k=20)
            elif stage == "r_dense":
                ranked = dense_retriever.retrieve(q_text, top_k=20)
            elif stage == "r_rrf":
                r_b = sparse_retriever.retrieve(q_text, top_k=20)
                r_d = dense_retriever.retrieve(q_text, top_k=20)
                fused = reciprocal_rank_fusion([r_b, r_d], top_n=20)
                ranked = [(chunk_id, score) for chunk_id, score, _ in fused]
            else:  # r_rrf_rerank
                r_b = sparse_retriever.retrieve(q_text, top_k=20)
                r_d = dense_retriever.retrieve(q_text, top_k=20)
                fused = reciprocal_rank_fusion([r_b, r_d], top_n=20)
                candidates = [c for c_id, _, _ in fused if (c := faiss_store.get_chunk(c_id)) is not None]
                ranked = reranker.rerank(q_text, candidates)


            chunk_ids = [c for c, _ in ranked]
            r5 = 1.0 if target_chunk in chunk_ids[:5] else 0.0
            r10 = 1.0 if target_chunk in chunk_ids[:10] else 0.0

            mrr = 0.0
            if target_chunk in chunk_ids:
                rank = chunk_ids.index(target_chunk) + 1
                mrr = 1.0 / rank

            recalls_5.append(r5)
            recalls_10.append(r10)
            mrrs.append(mrr)

        n = max(1, len(vector_questions))
        results[stage] = {
            "Recall@5": round(sum(recalls_5) / n, 4),
            "Recall@10": round(sum(recalls_10) / n, 4),
            "MRR": round(sum(mrrs) / n, 4),
        }

    return results


def run_full_ablation_matrix(
    base_config: LitAgentConfig,
    llm_client: BaseLLM,
    graph_store: NetworkXGraphStore,
    faiss_store: FAISSVectorStore,
    bm25_store: BM25Store,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
    reranker: CrossEncoderReranker,
    text2cypher: Text2CypherRetriever,
    questions: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:

    """Execute all 5 ablation configs and output results/ablation_table.csv."""
    configs = [
        ("vector_only", "vector", 0),
        ("graph_only", "graph", 0),
        ("joint_baseline", "both", 0),
        ("router", None, 0),
        ("router_crag", None, 2),
    ]

    metric_keys = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "graph_set_f1",
    ]

    all_config_records: dict[str, list[dict[str, Any]]] = {}
    table_rows: list[dict[str, str]] = []
    best_records: list[dict[str, Any]] = []

    for name, forced_route, max_iter in configs:
        cfg = base_config.model_copy(deep=True)
        cfg.agent.forced_route = forced_route  # type: ignore[assignment]
        cfg.agent.max_iterations = max_iter


        wf = LitAgentWorkflow(
            config=cfg,
            llm_client=llm_client,
            graph_store=graph_store,
            faiss_store=faiss_store,
            bm25_store=bm25_store,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            reranker=reranker,
            text2cypher=text2cypher,
        )

        records = run_single_config_eval(name, cfg, wf, questions)
        all_config_records[name] = records
        if name == "router_crag":
            best_records = records

        # Bootstrap summary
        boot = run_paper_bootstrap(records, metric_keys, n_resamples=12, sample_size=min(20, len(questions)))

        avg_lat = sum(r["latency"] for r in records) / len(records)
        avg_calls = sum(r["llm_calls"] for r in records) / len(records)

        row = {
            "Config": name,
            "Faithfulness": f"{boot['faithfulness']['mean']:.3f} +/- {boot['faithfulness']['sd']:.3f}",
            "Ans. Relevancy": f"{boot['answer_relevancy']['mean']:.3f} +/- {boot['answer_relevancy']['sd']:.3f}",
            "Ctx Precision": f"{boot['context_precision']['mean']:.3f} +/- {boot['context_precision']['sd']:.3f}",
            "Ctx Recall": f"{boot['context_recall']['mean']:.3f} +/- {boot['context_recall']['sd']:.3f}",
            "Graph set-F1": f"{boot['graph_set_f1']['mean']:.3f} +/- {boot['graph_set_f1']['sd']:.3f}",
            "Latency (s)": f"{avg_lat:.2f}",
            "LLM calls": f"{avg_calls:.1f}",
        }
        table_rows.append(row)

    # Save ablation_table.csv
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "ablation_table.csv"
    fieldnames = [
        "Config",
        "Faithfulness",
        "Ans. Relevancy",
        "Ctx Precision",
        "Ctx Recall",
        "Graph set-F1",
        "Latency (s)",
        "LLM calls",
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    logger.info("Saved ablation table to %s", csv_file)
    return {"table": table_rows, "all_records": all_config_records}, best_records
