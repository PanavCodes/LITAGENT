# Architecture and Design Decisions (Phase 0)

This document records the resolved design decisions for LitAgent as required by PRD.md §15.

---

### Q1: LLM Provider and Models
* **Decision:**
  - **Answer Model:** `gemini-2.5-flash` (or provider equivalent configured via `.env`). Highly cost-effective, fast, low latency for the multi-step agent and retry loops.
  - **Judge Model:** `gemini-2.5-pro` (or stronger evaluation model). Higher reasoning fidelity to reduce self-preference bias during RAGAS evaluations.
  - **Testing / Offline:** `FakeLLM` in `src/litagent/llm.py` provides deterministic, zero-cost, zero-latency execution for all unit tests and offline workflows.
* **Rationale:** Adheres to PRD §2.3 and NFR-4 (modest budget, fast CPU/API execution).

---

### Q2: Vector Store Selection (FAISS vs Pinecone)
* **Decision:** `faiss-cpu` with local on-disk persistence (`data/index/faiss.index`) and metadata mapping (`data/index/chunk_metadata.parquet` / `.pkl`).
* **Rationale:** Course requires local execution without GPU; avoids cloud subscription latency, rate limits, and external network dependency.

---

### Q3: Tracing & Observability
* **Decision:** Local JSONL file traces (`results/traces/<run_id>.jsonl`) enabled by default without requiring external credentials. Optional hooks for LangSmith or Langfuse if API keys are set in `.env`.
* **Rationale:** NFR-10, FR-OBS-1. Completely self-contained, auditable, and can be inspected via `scripts/view_trace.py`.

---

### Q4: Subfield, Scope, and Date Range
* **Decision:**
  - **Seed Query:** `"retrieval-augmented generation OR information retrieval OR dense retrieval OR reranking"`
  - **Categories:** `cs.IR`, `cs.CL`
  - **Publication Years:** 2020 to 2026
  - **Corpus Target:** 300 to 600 papers
* **Rationale:** High-density citation network with well-defined evaluation benchmarks, overlapping research authors, and immediate relevance.

---

### Q5: Graph Backend (Neo4j vs NetworkX)
* **Decision:** Dual `GraphStore` architecture. Neo4j is the primary backend when Docker Desktop is available (`bolt://localhost:7687`). When Docker is unavailable or for rapid CPU-only unit tests, `NetworkXGraphStore` acts as a seamless local fallback utilizing serialized graphs (`data/index/graph.pkl`).
* **Rationale:** Docker daemon status on developer workstations must never block testing or core agent development (PRD §5.3, FR-KG-4).

---

### Q6: Ownership and Roles
* **Decision:**
  - **Dev A Track:** Data ingestion, GraphStore (Neo4j & NetworkX), text-to-Cypher, Cypher security guard.
  - **Dev B Track:** Vector retrieval (FAISS, BM25, RRF, CrossEncoder), LangGraph agent workflow, evaluation pipeline, calibration.
  - **Antigravity (Coding Agent):** Core implementation, offline unit test suites, typing, linting, packaging.
  - **Both Developers:** Gate approvals, error analysis, manual QA verification, and final report.
