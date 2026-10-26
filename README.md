# LitAgent — Agentic Hybrid RAG for Scientific Literature Review

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LitAgent** is an agentic hybrid Retrieval-Augmented Generation (RAG) system for scientific literature review. It combines:
1. **Citation Knowledge Graph** (Neo4j / NetworkX local fallback) with read-only Text-to-Cypher.
2. **Hybrid Vector Search** (BM25 + FAISS Dense Embeddings fused via Reciprocal Rank Fusion, $k=60$).
3. **Cross-Encoder Reranking** (MS-MARCO MiniLM on CPU).
4. **Agentic Query Routing & CRAG Correction Loop** (LangGraph StateGraph with sufficiency grading and query rewriting).
5. **Calibrated Runtime Uncertainty** (Signals S1 margin, S2 evidence grade, S5 graph status; AUROC 0.727, ECE 0.110).

---

## 🚀 One-Command Workflow

You can set up, build the corpus, index stores, and run the evaluation pipeline with standard make commands:

```bash
# 1. Setup virtual environment & dependencies
make setup

# 2. Ingest scientific papers from arXiv & OpenAlex (idempotent, polite rate limits)
make data

# 3. Build Citation Graph, FAISS Dense Vector Index, and BM25 Sparse Index
make index

# 4. Run the 5-way ablation benchmarks & uncertainty calibration
make eval
```

---

## 💻 Running the User Interfaces

### Terminal CLI
Ask questions directly from the command line:
```bash
python -m litagent.ui.cli ask "Which papers in the corpus cite Dense Passage Retrieval?"
```

Forced route ablation preview:
```bash
python -m litagent.ui.cli ask "What is the core methodology of ColBERT?" --route vector
```

### Modern Gradio Web UI
Launch the interactive web review interface with real-time inspection panels:
```bash
python -m litagent.ui.app_gradio
```
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) to explore:
- **Chat Window** with grounded literature answers and inline citations.
- **Route & Cypher Inspector**: view AST-guarded read-only Cypher queries and execution results.
- **Evidence Passages**: inspect retrieved chunk texts and fused RRF ranks.
- **CRAG Monitor**: view reflection iterations and sufficiency scores.
- **Uncertainty Decomposition**: see composite scores and signals S1, S2, and S5.

---

## 🔍 Observability & Tracing

View execution step breakdowns and latency metrics:
```bash
# View the most recent trace
python scripts/view_trace.py

# List all available execution traces
python scripts/view_trace.py --list
```

---

## 🧪 Testing & Verification

Run the comprehensive unit test suite offline with zero external network or API key dependencies:
```bash
pytest tests/unit/
```

Run code formatting and type checking:
```bash
ruff check src tests scripts
mypy src
```

---

## ⚙️ Hardware & Environment Notes

- **OS**: Windows, Linux, or macOS.
- **Python**: 3.12 (managed via `uv`).
- **Hardware**: Designed to run entirely on a laptop CPU without requiring a GPU.
- **Neo4j / NetworkX**: If Docker is running, Neo4j can be launched via `docker compose up -d`. If Docker is stopped, the built-in NetworkX graph engine automatically serves as a 100% offline, local graph store with identical query support.
- **Determinism**: All embeddings, index creation, and evaluations use seeded RNG (`seed: 42`).
