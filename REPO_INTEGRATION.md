# REPO_INTEGRATION.md — Which GitHub repos to clone, what to take, how to combine

This file answers: *"Which existing repositories can we `git clone` to speed up development, and how do we combine them into one project?"*

All URLs below were checked with `git ls-remote` on 2026-10-26 and the ones marked **inspected** were also shallow-cloned and their folder layout and licences were read. Licences are as seen in each repo's LICENSE file; **re-check before the final submission**.

---

## 1. Strategy in one paragraph

Do **not** merge other projects' source trees into yours. Instead:

1. **Install libraries with pip** wherever a library exists (`langgraph`, `neo4j-graphrag`, `ragas`, `langchain-neo4j`, `neo4j`, and so on). This is the real speed-up and has no licence copying.
2. **Clone a few repos into `third_party/`** as *read-only reference* (examples, prompts, datasets, experiment design).
3. **Copy-and-adapt specific files** from permissively licensed repos into `src/litagent/` only when they save real time, keeping the licence header and logging it in `THIRD_PARTY.md`.
4. **Write the glue and the novel parts yourselves**: router, grader, rewrite loop, uncertainty, evaluation, ingestion for your corpus.

```
your repo (litagent/)
├── src/litagent/        ← your code (adapted snippets carry attribution)
├── third_party/         ← git-ignored; shallow clones for reading only
│   ├── dsa-public/
│   ├── langgraph-ai/
│   ├── neo4j-graphrag-python/
│   ├── langchain-neo4j/
│   ├── text2cypher/
│   ├── langgraph/
│   └── ragas/
└── THIRD_PARTY.md       ← what you copied, from where, under which licence
```

---

## 2. Repository catalogue

### 2.1 Clone these (reference or code reuse)

| # | Repo | Licence (seen) | Status | Role | Clone? | Use as |
|---|---|---|---|---|---|---|
| 1 | `https://github.com/PMRA-DSA/dsa-public` | MIT at root; `LitChat/` has its own `LICENCE.txt` (MIT, Government of Canada) | **inspected** | Closest public project: literature chatbot with vector, graph and hybrid agents (LlamaIndex + Gradio). Related paper arXiv 2602.17856 | Yes | Reference for system-prompt rules, UI layout, test-set format, and a **ready PDF smoke-test set** |
| 2 | `https://github.com/neo4j/neo4j-graphrag-python` | Apache-2.0 | **inspected** | Official Neo4j GraphRAG package: `Text2CypherRetriever`, `HybridRetriever`, examples | Yes (and `pip install neo4j-graphrag`) | Library plus reference |
| 3 | `https://github.com/langchain-ai/langchain-neo4j` | MIT | **inspected** | `GraphCypherQAChain` and `cypher_utils.py` (query corrector) | Yes | Reference for Cypher validation ideas |
| 4 | `https://github.com/neo4j-labs/text2cypher` | CC0 1.0 | **inspected** | Text-to-Cypher datasets, evaluation notebook (Jaccard), fine-tuning scripts | Yes | Mine **few-shot examples** and borrow the evaluation idea. Skip fine-tuning |
| 5 | `https://github.com/piyushagni5/langgraph-ai` | MIT | **inspected** | LangGraph cookbook with `rag/agentic-rag` and `rag/rag-agents` notebooks | Yes | LangGraph skeleton, CRAG-style loop patterns |
| 6 | `https://github.com/langchain-ai/langgraph` | MIT | **inspected** | The agent framework | Install via pip; clone only for `examples/` | Library |
| 7 | `https://github.com/explodinggradients/ragas` | Apache-2.0 | **inspected** | Evaluation framework | Install via pip; clone to read metric implementations | Library |

### 2.2 Install from PyPI only (no clone needed)

| Package | Purpose | Checked |
|---|---|---|
| `langgraph`, `langchain-core`, plus one provider package | Agent and LLM wrapper | repo exists |
| `neo4j` (official driver) | Neo4j access | repo `neo4j/neo4j-python-driver` exists |
| `neo4j-graphrag` | Text2CypherRetriever | repo exists |
| `ragas` | Metrics | repo exists |
| `sentence-transformers` | Embeddings and cross-encoder | repo `UKPLab/sentence-transformers` exists |
| `faiss-cpu` | Dense index | upstream `facebookresearch/faiss` exists |
| `rank-bm25` | Sparse retrieval | repo `dorianbrown/rank_bm25` exists |
| `arxiv` | arXiv API client | repo `lukasschwab/arxiv.py` exists |
| `pyalex` | OpenAlex client | repo `J535D165/pyalex` exists |
| `pymupdf` | PDF text extraction | repo `pymupdf/PyMuPDF` exists |
| `networkx` | Graph fallback | — |
| `gradio` | UI | — |

### 2.3 Optional, not recommended for the first pass

| Repo | Why optional |
|---|---|
| `https://github.com/neo4j-labs/llm-graph-builder` | Exists, but it builds graphs by LLM entity extraction, which is expensive and not needed for a metadata-plus-citations graph |
| `https://github.com/langfuse/langfuse` | Exists; self-hosting needs extra services. Prefer local JSONL traces; use hosted LangSmith or Langfuse only if the course requires it |
| `https://github.com/run-llama/llama_index` | Exists; LitChat uses it, but mixing two orchestration stacks would slow you down. Read, don't adopt |

### 2.4 Found-and-dropped

- `neo4j-contrib/text2cypher` and `neo4j-labs/text2cypher-agent` could not be read anonymously (they returned an authentication prompt). They may not exist or may be private. The correct, working repo is **`neo4j-labs/text2cypher`** (row 4 above).
- No public code was found for the target paper, arXiv 2508.05660.

---

## 3. Exactly what to take from each repo

### 3.1 `PMRA-DSA/dsa-public` → `LitChat/`

| Take | Where it goes | How |
|---|---|---|
| The **four system-prompt rules** (answer only from context; inline page-number citations; always retrieve even for repeated questions; say "not in the documents" for out-of-scope questions) | `prompts/answer.md` | Rewrite in your own words, keep the intent |
| Test-set structure (`testset_single_paper.xlsx`, `testset_multi_paper.xlsx`): single-paper vs multi-paper question types | `data/eval/` design | Mirror the idea: your question taxonomy has "single-paper" and "multi-paper" content questions |
| `testset_input_pdfs/` (about 50 real PDFs, pesticide and herbicide domain) | `data/smoke/` | Use **only** to test PDF parsing and chunking in Phase 1 before the arXiv download finishes. Keep local; do not commit or redistribute (these are third-party papers) |
| Gradio UI idea: retrieval-parameter controls (top-k, graph depth, minimum relevance) | `ui/app_gradio.py` | Re-implement; do not copy the LlamaIndex code |
| Chunking defaults (512 / 50 in LitChat; paper uses about 2,024 / 50) | `configs/default.yaml` | Make chunk size a tunable; run both in the retrieval ablation |

Leave: the LlamaIndex `PropertyGraphIndex` and LLM-extracted triplets (costly, different approach).

### 3.2 `neo4j/neo4j-graphrag-python`

| Take | Where it goes | How |
|---|---|---|
| `Text2CypherRetriever(driver, llm, neo4j_schema, examples, custom_prompt, result_formatter, neo4j_database)` | `retrieval/text2cypher.py` | **Baseline implementation.** Wrap it: your wrapper adds the read-only guard, `EXPLAIN` validation, retry-once, timeout and `LIMIT` injection. Check the installed version's signature first |
| `examples/retrieve/text2cypher_search.py` | reference | Copy the usage pattern |
| The default text-to-Cypher prompt template | `prompts/cypher_system.md` | Read it, then write your own schema-specific version |
| `HybridRetriever` | not used for VectorRAG | It fuses Neo4j full-text and vector indexes inside Neo4j. LitAgent needs **BM25 + FAISS + RRF** that you implement and ablate. You may mention it in the report as an alternative |

Design note: the retriever needs an `LLMInterface` from `neo4j_graphrag.llm`. If wiring your LangChain model into it takes more than about an hour, **write your own thin `text2cypher.py`** and keep the library only as a reference. Decide this in Phase 3, Day 1.

### 3.3 `langchain-ai/langchain-neo4j`

| Take | Where it goes | How |
|---|---|---|
| `chains/graph_qa/cypher_utils.py` (Cypher query corrector idea: check relationship directions against the schema) | `retrieval/cypher_guard.py` | Read, then implement a smaller version for your schema. If you copy a function, keep the MIT header and log it in `THIRD_PARTY.md` |
| `chains/graph_qa/cypher.py` and `prompts.py` | reference | Compare prompt wording |
| `pip install langchain-neo4j` | optional | Only if you want `Neo4jGraph` schema introspection; otherwise use the plain driver |

### 3.4 `neo4j-labs/text2cypher` (CC0)

| Take | Where it goes | How |
|---|---|---|
| `datasets/` (natural language ↔ Cypher pairs from many demo databases) | `prompts/cypher_fewshot_seed.yaml` | Pick 10 to 15 **generic pattern** examples (counts, top-k by degree, two-hop, filters by year). **Rewrite every example for your schema** (Paper, Author, Keyword, CITES …) and run each one in a test |
| `evaluations/evaluating_cypher_jaccard.ipynb` | `eval/qa_graph.py` | Reuse the idea: compare **result sets** with Jaccard / set-F1, not Cypher strings |
| `finetuning/` | not used | GPU work, out of scope |

CC0 means no attribution is legally required, but log it anyway.

### 3.5 `piyushagni5/langgraph-ai`

| Take | Where it goes | How |
|---|---|---|
| `rag/agentic-rag/` patterns (agentic workflow, systems) | `agent/workflow.py` | Skeleton for state, conditional edges, loop bound |
| `rag/rag-agents/*.ipynb` (RAG as a tool inside LangGraph agents) | `agent/` | Pattern for tools and routing |
| `langgraph-cookbook/agentic-patterns/` | reference | Router and reflection patterns |

Check each file's imports against your pinned LangGraph version: tutorial code often uses old APIs. This repo is MIT, copyright Piyush Agnihotri.

### 3.6 `langchain-ai/langgraph` and `explodinggradients/ragas`

Install both from PyPI. Clone only to read the documentation examples (CRAG-style and router examples in LangGraph; metric implementations and the testset generator in RAGAS). Pin versions in `pyproject.toml` because both APIs change quickly.

---

## 4. How to combine everything (step by step)

| Step | Action | Output |
|---|---|---|
| 1 | Create your own repo `litagent`; `git init`; copy the spec files in | Clean base |
| 2 | Run `scripts/clone_repos.sh` | `third_party/` with 7 shallow clones; `third_party/SOURCES.txt` with commit SHAs |
| 3 | `pip install` the libraries in `pyproject.toml` | Working environment |
| 4 | Skim each repo's relevant folder (Section 3) and write 3 to 5 lines of "what I will reuse" in `docs/reuse_notes.md` | Shared understanding |
| 5 | Copy **only** the specific functions or prompts you need into `src/litagent/…`; adapt the names, types and tests | Attributed, tested code |
| 6 | Add one line per copied item to `THIRD_PARTY.md` (source URL, commit SHA, licence, files, what you changed) | Licence compliance |
| 7 | Never import from `third_party/` | No hidden dependency on untracked code |
| 8 | Re-run the unit tests after every adaptation | No regressions |

### `THIRD_PARTY.md` template

```markdown
| Item | Source repo | Commit | Licence | Files taken | Modified? | Used in |
|---|---|---|---|---|---|---|
| Cypher corrector idea | langchain-ai/langchain-neo4j | <sha> | MIT | cypher_utils.py (logic only) | Rewritten for our schema | src/litagent/retrieval/cypher_guard.py |
```

---

## 5. Integration map: which repo helps which phase

| Phase | Helped by |
|---|---|
| 1 Ingestion | `dsa-public` smoke PDFs; `arxiv`, `pyalex`, `pymupdf` packages |
| 2 Stores | `neo4j` driver; Neo4j Docker; (optional) `langchain-neo4j` schema helpers |
| 3 Retrieval | `sentence-transformers`, `faiss-cpu`, `rank-bm25`; `neo4j-graphrag` Text2CypherRetriever; `text2cypher` datasets |
| 4 Agent | `langgraph`, `langgraph-ai` patterns; `dsa-public` prompt rules |
| 5 Uncertainty and loop | `langgraph-ai` agentic-rag (CRAG patterns); your own code |
| 6 Evaluation | `ragas`; `text2cypher` evaluation idea |
| 7 UI and report | `dsa-public` UI ideas; Gradio |

---

## 6. Licence notes and cautions

- **MIT / Apache-2.0 / CC0** code may be reused with attribution (Apache-2.0 also requires keeping the NOTICE where applicable and stating changes). Still keep a record.
- `dsa-public/LitChat/testset_input_pdfs/` are third-party papers. They are fine for local testing; do not redistribute them in your repository or report.
- Tutorial notebooks age fast. If a notebook fails on your pinned versions, update the code to the current docs instead of pinning an old version of the library.
- `neo4j-labs/text2cypher` is an experimental Neo4j Labs project: treat it as a source of examples, not a supported library.
- Always re-verify a licence file before the final submission; the table above reflects what was seen on 2026-10-26.

---

## 7. Time saved (honest estimate)

| Component | Build from scratch | With reuse | Where the saving comes from |
|---|---|---|---|
| LangGraph skeleton | about 1 day | about 3 hours | `langgraph-ai` patterns |
| Text-to-Cypher | about 2 days | about 1 day | `Text2CypherRetriever` and dataset examples |
| Cypher validation | about 1 day | about 4 hours | `cypher_utils.py` ideas |
| Evaluation harness | about 3 days | about 2 days | RAGAS metrics, Jaccard idea |
| PDF parsing test set | about 1 day | about 2 hours | LitChat sample PDFs |

These are estimates, not measurements.
