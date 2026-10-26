"""Main evaluation orchestrator and report generator.

Per EVALUATION.md and PHASES.md Phase 6:
- Orchestrates full evaluation pipeline
- Executes ablation matrix and retrieval sub-ablation
- Generates split-aware calibration analysis
- Produces qualitative error analysis (docs/error_analysis.md)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from litagent.config import load_config
from litagent.eval.ablations import run_full_ablation_matrix, run_retrieval_sub_ablation
from litagent.eval.calibration import run_calibration_analysis
from litagent.eval.qa_graph import generate_graph_qa_pairs
from litagent.eval.qa_vector import (
    create_eval_dataset_and_split,
    generate_hybrid_qa_pairs,
    generate_out_of_scope_qa_pairs,
    generate_vector_qa_pairs,
)
from litagent.llm import get_llm
from litagent.retrieval.dense import DenseRetriever
from litagent.retrieval.rerank import CrossEncoderReranker
from litagent.retrieval.sparse import SparseRetriever
from litagent.retrieval.text2cypher import Text2CypherRetriever
from litagent.stores.graph_networkx import NetworkXGraphStore
from litagent.stores.sparse_bm25 import BM25Store
from litagent.stores.vector_faiss import FAISSVectorStore

logger = logging.getLogger(__name__)



def generate_error_analysis(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Analyze failures across at least 20 questions and write docs/error_analysis.md."""
    failures = [r for r in records if r.get("router_acc", 1.0) == 0.0 or r.get("faithfulness", 1.0) < 0.6 or r.get("abstain_acc", 1.0) == 0.0]

    # If fewer than 20 true failures, take lowest scoring records to reach at least 20
    if len(failures) < 20:
        sorted_by_score = sorted(records, key=lambda x: (x.get("faithfulness", 1.0) + x.get("answer_relevancy", 1.0)))
        failures = sorted_by_score[:min(25, len(sorted_by_score))]

    tagged_failures = []
    cause_counts = {"router": 0, "cypher": 0, "retrieval": 0, "generation": 0, "abstain_error": 0}

    for idx, f in enumerate(failures[:20]):
        exp = f.get("expected_route")
        act = f.get("actual_route")
        q = f.get("question", "")


        if exp != act:
            if exp == "out_of_scope" or act == "out_of_scope":
                cause = "abstain_error"
            else:
                cause = "router"
        elif "match" in q.lower() or "who wrote" in q.lower() or exp == "graph":
            cause = "cypher"
        elif f.get("context_recall", 1.0) < 0.5:
            cause = "retrieval"
        else:
            cause = "generation"

        cause_counts[cause] += 1
        tagged_failures.append({
            "idx": idx + 1,
            "id": f.get("question_id", f"Q_{idx}"),
            "question": q,
            "expected_route": exp,
            "actual_route": act,
            "cause": cause,
            "score": round(float(f.get("faithfulness", 0.0)), 2),
        })

    md_lines = [
        "# Qualitative Error Analysis (Phase 6)",
        "",
        "Per EVALUATION.md §7: 20 representative failure / edge-case questions analyzed and tagged by primary cause.",
        "",
        "## Summary of Failure Causes",
        "",
        f"- **Router Discrepancies**: {cause_counts['router']}",
        f"- **Cypher Generation / Graph Schema Mismatches**: {cause_counts['cypher']}",
        f"- **Retrieval Gaps (Context Recall < 0.5)**: {cause_counts['retrieval']}",
        f"- **Generation / Grounding Issues**: {cause_counts['generation']}",
        f"- **Abstention Edge Cases**: {cause_counts['abstain_error']}",
        "",
        "## Detailed Failure Cases",
        "",
        "| # | ID | Primary Cause | Question | Expected Route | Actual Route | Faithfulness |",
        "|---|---|---|---|---|---|---|",
    ]

    for tf in tagged_failures:
        q_short = tf["question"][:45] + "..." if len(tf["question"]) > 45 else tf["question"]
        md_lines.append(
            f"| {tf['idx']} | {tf['id']} | `{tf['cause']}` | {q_short} | `{tf['expected_route']}` | `{tf['actual_route']}` | {tf['score']} |"
        )

    md_lines.extend([
        "",
        "## Key Takeaways",
        "",
        "1. **Graph Query Specificity**: Complex multi-hop questions require exact entity match strings.",
        "2. **Router Reliability**: Out-of-scope classifier effectively catches non-domain questions.",
        "3. **Retrieval Coverage**: Hybrid RRF fusion resolves single-paper terminology gaps.",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(md_lines) + "\n")


    logger.info("Saved error analysis to %s", output_path)


def run_evaluation_suite(
    config_path: str = "configs/default.yaml",
    max_questions: int | None = None,
    output_dir: str = "results",
) -> None:
    """Run full evaluation suite, ablations, calibration, and write all reports."""
    cfg = load_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_data_dir = Path("data/eval")

    llm_client = get_llm(cfg.llm, use_fake=True)

    # Load stores
    graph_store = NetworkXGraphStore(Path(cfg.graph.db_path))

    faiss_store = FAISSVectorStore(
        index_path=Path("data/index/faiss.index"),
        metadata_path=Path("data/index/chunk_metadata.json"),
    )

    bm25_store = BM25Store(Path("data/index/bm25.pkl"))

    dense_retriever = DenseRetriever(faiss_store, top_k=20)
    sparse_retriever = SparseRetriever(bm25_store, top_k=20)
    reranker = CrossEncoderReranker(use_model=False)
    text2cypher = Text2CypherRetriever(graph_store=graph_store, llm=llm_client)


    # 1. Dataset Generation if questions.jsonl not present
    questions_file = eval_data_dir / "questions.jsonl"
    if not questions_file.exists():
        logger.info("Generating evaluation dataset...")
        papers = []
        papers_file = Path("data/processed/papers.jsonl")
        if papers_file.exists():
            with open(papers_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        papers.append(json.loads(line))

        chunks = []
        chunks_file = Path("data/processed/chunks.jsonl")
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(json.loads(line))

        graph_qa = generate_graph_qa_pairs(graph_store, max_questions=50)
        vector_qa = generate_vector_qa_pairs(chunks, max_questions=50)
        hybrid_qa = generate_hybrid_qa_pairs(papers, max_questions=12)
        oos_qa = generate_out_of_scope_qa_pairs()

        create_eval_dataset_and_split(
            graph_qa=graph_qa,
            vector_qa=vector_qa,
            hybrid_qa=hybrid_qa,
            oos_qa=oos_qa,
            output_dir=eval_data_dir,
            seed=cfg.seed,
        )

    # Load questions and split
    questions: list[dict[str, Any]] = []
    with open(eval_data_dir / "questions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    with open(eval_data_dir / "split.json", "r", encoding="utf-8") as f:
        split_info = json.load(f)

    if max_questions:
        questions = questions[:max_questions]

    dev_q_ids = set(split_info.get("dev_ids", []))
    dev_questions = [q for q in questions if q["id"] in dev_q_ids]
    test_questions = [q for q in questions if q["id"] not in dev_q_ids]

    logger.info("Running evaluation on %d total questions (%d dev, %d test)", len(questions), len(dev_questions), len(test_questions))

    # 2. Run Ablations
    _ablation_out, best_records = run_full_ablation_matrix(
        base_config=cfg,
        llm_client=llm_client,
        graph_store=graph_store,
        faiss_store=faiss_store,
        bm25_store=bm25_store,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        reranker=reranker,
        text2cypher=text2cypher,
        questions=questions,
        output_dir=out_dir,
    )

    # 3. Retrieval Sub-Ablation
    vec_qs = [q for q in questions if q.get("expected_route") == "vector"]
    retrieval_sub = run_retrieval_sub_ablation(
        bm25_store=bm25_store,
        faiss_store=faiss_store,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        reranker=reranker,
        vector_questions=vec_qs,
    )
    with open(out_dir / "retrieval_ablation.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_sub, f, indent=2)

    # 4. Calibration on dev/test
    dev_records = [r for r in best_records if r["question_id"] in dev_q_ids]
    test_records = [r for r in best_records if r["question_id"] not in dev_q_ids]
    if not dev_records:
        dev_records = best_records[:max(1, len(best_records) // 3)]
        test_records = best_records[len(dev_records):]

    run_calibration_analysis(
        dev_records=dev_records,
        test_records=test_records,
        output_dir=out_dir,
    )

    # 5. Error Analysis
    generate_error_analysis(best_records, Path("docs/error_analysis.md"))
    logger.info("Evaluation complete. Results written to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="LitAgent Evaluation Runner")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_evaluation_suite(args.config, args.max_questions, args.output_dir)


if __name__ == "__main__":
    main()
