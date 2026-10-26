# CHANGELOG.md — LitAgent Project Progress and History

## [v1.0] - 2026-10-26

### Phase 0: Setup and Core Architecture
- Initialized git repository on branch `phase-0-setup`.
- Created `.gitignore`, `.env.example`, and directory structure matching `AGENTS.md` §2.
- Created Python 3.12 virtual environment (`.venv`) using `uv` with all 159 dependencies.
- Implemented `src/litagent/config.py` (Pydantic-Settings), `src/litagent/llm.py` (CachedLLMWrapper with on-disk response cache and backoff retry), `schemas.py`, and `tracing.py`.
- Resolved open design questions Q1–Q6 in `docs/decisions.md`.
- Implemented `scripts/clone_repos.sh`, `third_party/SOURCES.txt`, and `THIRD_PARTY.md`.
- Added unit tests in `tests/unit/test_phase0.py` and passed Gate 0 (`docs/gates/phase-0.md`).

### Phase 1: Data Ingestion and Corpus Construction
- Implemented `src/litagent/ingest/clean.py` (deduplication, date parsing, sanitization).
- Implemented `src/litagent/ingest/chunking.py` (sentence-aware sliding window chunker).
- Implemented `src/litagent/ingest/keywords.py` (TF-IDF keyword extraction).
- Implemented `src/litagent/ingest/pdf_parse.py` (PyMuPDF section extraction).
- Implemented `src/litagent/ingest/arxiv_fetch.py` and `openalex_enrich.py` with polite rate-limiting and local cache.
- Built idempotent corpus runner `scripts/build_corpus.py` producing 35 papers, 35 chunks, 70 citations, and 100% citation coverage manifest.
- Added unit tests in `tests/unit/test_phase1.py` and passed Gate 1 (`docs/gates/phase-1.md`).

### Phase 2: Knowledge Graph and Vector Stores
- Implemented `src/litagent/stores/graph_base.py`, `graph_networkx.py`, and `graph_neo4j.py`.
- Implemented `src/litagent/stores/vector_faiss.py` (IndexFlatIP cosine similarity on CPU) and `sparse_bm25.py` (BM25Okapi).
- Implemented `scripts/build_index.py` indexing 175 graph nodes, 455 edges, and 35 chunks in 1.98 seconds.
- Added unit tests in `tests/unit/test_phase2.py` and passed Gate 2 (`docs/gates/phase-2.md`).

### Phase 3: Retrieval Components
- Implemented `src/litagent/retrieval/dense.py`, `sparse.py`, and `rrf.py` ($k=60$).
- Implemented `src/litagent/retrieval/rerank.py` (CrossEncoder reranker on CPU with fast fallback).
- Implemented `src/litagent/retrieval/cypher_guard.py` (AST comment stripping, read-only clause validation, label validation, and automatic `LIMIT 50` injection).
- Implemented `src/litagent/retrieval/text2cypher.py` with retry feedback loop.
- Created `prompts/cypher_fewshot.yaml` with 10 verified schema queries.
- Added unit tests in `tests/unit/test_phase3.py` and passed Gate 3 (`docs/gates/phase-3.md`).

### Phase 4: LangGraph Agent v1
- Implemented `src/litagent/agent/state.py`, `router.py`, `graph_branch.py`, `vector_branch.py`, and `generate.py`.
- Created `prompts/router_fewshot.yaml` and `prompts/answer.md` with inline citations and evidence constraints.
- Implemented `src/litagent/agent/workflow.py` assembling compiled LangGraph StateGraph.
- Implemented CLI in `src/litagent/ui/cli.py`.
- Evaluated 12 reference questions in `results/phase4_example_runs.md`.
- Added unit tests in `tests/unit/test_phase4.py` and passed Gate 4 (`docs/gates/phase-4.md`).

### Phase 5: CRAG Correction Loop and Uncertainty
- Implemented `src/litagent/agent/grader.py` and `prompts/grader.md` (sufficiency score $[0, 1]$ and missing info).
- Implemented `src/litagent/agent/rewrite.py` (query transformation and branch switching).
- Implemented `src/litagent/agent/uncertainty.py` computing signals S1 (margin), S2 (grade), S5 (graph status) and composite score with low-confidence warning.
- Verified loop termination invariant in `tests/unit/test_phase5.py`.
- Passed Gate 5 (`docs/gates/phase-5.md`).

### Phase 6: Rigorous Evaluation, Ablations, and Calibration
- Implemented `src/litagent/eval/qa_graph.py` and `qa_vector.py` creating 103 evaluation questions across 8 stratified types (30% vector and 20% graph verified).
- Created stratified 28% dev / 72% test split (`data/eval/split.json`).
- Implemented `src/litagent/eval/ragas_runner.py` and `bootstrap.py` (paper protocol 12x20 and extended 1,000 resamples).
- Implemented `src/litagent/eval/ablations.py` evaluating 5 configurations and 4 retrieval stages to `results/ablation_table.csv` and `results/retrieval_ablation.json`.
- Implemented `src/litagent/eval/calibration.py` fitting logistic regression on dev only, achieving AUROC 0.727 and ECE 0.110 on test, plotted in `results/figures/calibration_curve.png`.
- Analyzed 20 failure cases in `docs/error_analysis.md`.
- Added unit tests in `tests/unit/test_phase6.py` and passed Gate 6 (`docs/gates/phase-6.md`).

### Phase 7: UI, Tracing, and Polish
- Implemented interactive Web UI in `src/litagent/ui/app_gradio.py` with dark-themed Gradio 6.x blocks and real-time inspector panels.
- Implemented trace pretty-printer CLI in `scripts/view_trace.py`.
- Updated `README.md` with one-command workflow and hardware instructions.
- Achieved 100% clean passes on `ruff check src tests scripts`, `mypy src`, and all 28 unit tests in `pytest tests/unit/`.
- Passed Gate 7 (`docs/gates/phase-7.md`).

### Phase 8: Report and Release
- Authored comprehensive 13-section technical report in `docs/report.md` with measured metrics and Section 6 deviations table.
- Verified Gate 8 checklist in `docs/gates/phase-8.md`.
- Tagged release `v1.0`.
