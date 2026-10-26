# PRD: LitAgent — Agentic Hybrid RAG for Scientific Literature Review

| Field | Value |
|---|---|
| Document | Product Requirements Document (PRD) |
| Project codename | **LitAgent** |
| Version | 1.0 |
| Date | 2026-10-26 |
| Target builder | Google Antigravity (agentic IDE) plus two human developers |
| Target duration | About 4 weeks for two people (an estimate, not a measurement) |
| Hardware assumption | **No GPU required.** Laptop CPU, 16 GB RAM, Docker available |
| Companion files | `PHASES.md`, `ARCHITECTURE.md`, `DATA_SPEC.md`, `EVALUATION.md`, `REPO_INTEGRATION.md`, `AGENTS.md`, `ANTIGRAVITY_PROMPTS.md`, `REPORT_OUTLINE.md` |

> **How to read this PRD.** This file says *what* to build and *why*. `PHASES.md` says *in what order*. `ARCHITECTURE.md` and `DATA_SPEC.md` say *how*. `REPO_INTEGRATION.md` says *which existing GitHub code to clone and reuse*. `AGENTS.md` is the rulebook the coding agent must follow.

---

## 1. Executive summary

LitAgent is a question-answering system over a corpus of scientific papers. A **LangGraph agent** reads each question and chooses the best retrieval strategy:

- **GraphRAG** for structural questions ("Which papers cite X?", "Who co-authored with Y?", "How many papers in 2024 mention Z?"). The question is translated to **Cypher** and run against a **Neo4j** knowledge graph of papers, authors, keywords and citations.
- **VectorRAG** for content questions ("What method does paper X use for reranking?"). It runs **BM25 + dense retrieval, fuses them with Reciprocal Rank Fusion (RRF), then reranks** with a small cross-encoder.
- **Both**, when the question needs structure and content together.

After retrieval, an **evidence grader** checks whether the evidence is good enough. If not, a **bounded rewrite loop** (CRAG-style: grade, rewrite, retry) runs. The final answer carries **paper citations** and an **uncertainty score** so that low-confidence answers are flagged or abstained.

The project is evaluated with **RAGAS** metrics and a **bootstrapped protocol** (mean ± SD over resamples), plus an **ablation table** comparing vector-only, graph-only, non-agentic baseline, router, and router plus correction loop.

The project replicates and extends the design in arXiv 2508.05660, *Open-Source Agentic Hybrid RAG Framework for Scientific Literature Review*, using lighter components so that it runs on a laptop.

---

## 2. Background and reference paper

### 2.1 What the paper does (verified from the paper text)

| Aspect | Paper's design |
|---|---|
| Data sources | PubMed, arXiv and Google Scholar records; open-access full-text PDFs |
| Cleaning | Drop records with missing values; de-duplicate by DOI and title |
| Keyword filter | 5 TF-IDF keywords per article; keep articles whose keyword similarity to the query is in the top 25% |
| Graph | Neo4j; main node is a Publication (DOI, title, abstract) linked to Author, Database, Keyword, Publication Year and Citation nodes |
| Vector store | FAISS; recursive chunks of about 2,024 characters with 50-character overlap; all-MiniLM-L6-v2 embeddings |
| GraphRAG | LLM translates the question to Cypher using the schema plus 30 few-shot examples |
| VectorRAG | BM25 (top 5) plus dense L2 retrieval (top 5), merged and reranked with Cohere rerank-english-v3.0 |
| Agent | Llama-3.3-70B chooses the tool, with 5 few-shot examples per tool |
| Generator | Mistral-7B-Instruct-v0.3, optionally DPO-tuned on 15 preference pairs |
| Evaluation | 40 synthetic QA pairs (20 vector, 20 graph); Faithfulness, Answer Relevance, Context Precision, Context Recall; bootstrap with 12 resamples of 20 questions |

### 2.2 What the paper does not give us

- It does **not name the Neo4j edge types**. We define our own (see `DATA_SPEC.md`).
- Its methods section does **not describe a runtime uncertainty estimator**; it reports bootstrap uncertainty over *metrics* and mentions runtime flagging only at a high level. **We design our own runtime uncertainty score** (Section 7.6) and say so in the report.
- It reports **deltas, not absolute scores**, in the text. We cannot compare our absolute numbers to the paper's. We compare *relative* effects (agentic vs baseline).
- **No public code was found** for 2508.05660. We implement from the paper.

### 2.3 Deliberate deviations (must be stated openly in the final report)

| Paper | LitAgent | Reason |
|---|---|---|
| Llama-3.3-70B agent | Any API LLM, prompting only | No local GPU |
| Mistral-7B generator, DPO-tuned | Same API LLM, prompt-engineered | No fine-tuning hardware |
| Cohere rerank-english-v3.0 | Local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on top 20 | Free, CPU-friendly |
| PubMed + arXiv + Scholar | arXiv (papers) + OpenAlex (citations) | Free, open, scriptable |
| Healthcare MLLM topic | One IR/RAG subfield in cs.IR / cs.CL | Builders can judge answers themselves |
| 40 QA pairs | 80 to 120 QA pairs | Tighter confidence intervals |
| Joint-search baseline | Same, plus vector-only, graph-only | Richer ablation |
| No correction loop | CRAG-style grade-and-rewrite loop added | Goes beyond the paper |

---

## 3. Goals and non-goals

### 3.1 Goals

| ID | Goal | Measure |
|---|---|---|
| G1 | A working end-to-end agentic hybrid RAG system on a 300 to 1,000 paper corpus | Demo answers questions of all types with citations |
| G2 | Show that the agentic router beats single-strategy baselines on a mixed question set | Ablation table with mean ± SD, router better than vector-only on graph questions and better than graph-only on content questions |
| G3 | Provide a usable uncertainty score | AUROC of uncertainty vs answer correctness on the dev set, reported honestly even if modest |
| G4 | Rigorous, reproducible evaluation | One command regenerates every table and figure from stored artifacts |
| G5 | Runs on CPU with an API LLM | Documented laptop run, no GPU steps |
| G6 | Reuse existing code to cut build time | At least 60% of "plumbing" comes from cloned or installed libraries; the custom code is the router, grader, uncertainty, evaluation and glue |
| G7 | Honest reporting | Deviations, failures and limits written in the report |

### 3.2 Non-goals

- No fine-tuning (no DPO, no LoRA).
- No local 70B models.
- No web crawling beyond the arXiv API and OpenAlex API.
- No multi-user auth, no production deployment, no horizontal scaling.
- No building a knowledge-graph extractor from raw text (entity-relation extraction by LLM). The graph is built from **structured metadata and citation data**, which is cheaper and verifiable.
- No claim of state-of-the-art results.

---

## 4. Users and use cases

### 4.1 Personas

| Persona | Need |
|---|---|
| **Student researcher** | Quickly understand a subfield: key papers, who cites whom, what methods exist |
| **Course evaluator** | Verify rigor: ablations, bootstrapped metrics, traces, reproducibility |
| **The two developers** | A buildable, well-scoped plan with clear acceptance tests per phase |

### 4.2 Example questions (the system must handle all of these types)

**Structural / graph questions**
1. "Which papers in the corpus cite *Dense Passage Retrieval*?"
2. "Who are the five most prolific authors in the corpus?"
3. "How many papers were published in 2024 with the keyword *reranking*?"
4. "Which papers are cited by both paper A and paper B?" (co-citation / shared references)
5. "What is the citation path from paper A to paper B?"

**Content / vector questions**
6. "What retrieval metric does paper X report as its main result?"
7. "How does paper Y handle long documents?"
8. "What limitations do the authors of paper Z mention?"

**Hybrid questions**
9. "Among papers citing *DPR*, which ones propose a reranking method, and what do they say about latency?"
10. "Summarise the evaluation approach used by the three most-cited papers about hybrid search."

**Out of scope questions (must abstain or say so)**
11. "What is the weather in Mumbai?" (not in corpus)
12. "Who won the 2026 football World Cup?" (not in corpus)

---

## 5. Scope

### 5.1 In scope

1. Ingestion pipeline: arXiv metadata and PDFs, OpenAlex citations, cleaning, keyword extraction.
2. Knowledge graph in Neo4j (Docker) with a **NetworkX fallback** behind one interface.
3. Vector store (FAISS) and BM25 index over chunked full text.
4. Retrieval: dense, sparse, RRF, cross-encoder rerank, text-to-Cypher.
5. LangGraph agent: router, two retrieval branches, evidence grader, rewrite loop, answer generator, uncertainty scorer.
6. Evaluation harness: question generation, gold answers, RAGAS, bootstrapping, ablations, calibration.
7. Tracing of every run.
8. A minimal UI (Gradio or Streamlit) and a CLI.
9. Final report materials: tables, figures, written deviations.

### 5.2 Out of scope

See non-goals (3.2).

### 5.3 Scope dials (to cut if time slips)

| Dial | Default | Reduced |
|---|---|---|
| Corpus size | 600 papers | 300 papers |
| Full-text coverage | All with PDFs | Abstracts plus full text for a 200-paper subset |
| QA set | 100 questions | 60 questions |
| Uncertainty signals | 4 signals | 2 signals |
| UI | Gradio chat | CLI only |
| Graph backend | Neo4j | NetworkX |

---

## 6. System overview

```
                        +-----------------------------+
 arXiv API ───────────► |  Ingestion                  |
 OpenAlex API ────────► |  fetch → clean → parse PDF  |
                        |  → keywords → chunk         |
                        +--------------+--------------+
                                       |
              +------------------------+------------------------+
              v                                                 v
     +-----------------+                              +--------------------+
     |  Neo4j / NetworkX|                             |  FAISS (dense)     |
     |  Paper, Author,  |                             |  BM25 (sparse)     |
     |  Keyword, Year,  |                             |  chunk metadata    |
     |  CITES, WROTE... |                             +--------------------+
     +--------+--------+                                        |
              |                                                 |
              +------------------+------------------------------+
                                 v
                       +--------------------+
   Question ─────────► |  LangGraph agent   |  ─────────► Answer + citations
                       |  router → branches |             + uncertainty score
                       |  grader → rewrite  |             + trace
                       +--------------------+
```

Full detail is in `ARCHITECTURE.md`.

---

## 7. Functional requirements

Priority key: **M** = must have, **S** = should have, **C** = could have.

### 7.1 Data ingestion (FR-ING)

| ID | Requirement | Pri |
|---|---|---|
| FR-ING-1 | Fetch paper metadata from arXiv for a configured query and category list (title, abstract, authors, categories, published date, arXiv ID, PDF URL, DOI if present) | M |
| FR-ING-2 | Respect arXiv API etiquette: no more than one request every 3 seconds, a descriptive user agent, local caching of every response | M |
| FR-ING-3 | Enrich each paper through OpenAlex (via `pyalex`) with referenced works and DOI, matching by DOI or title | M |
| FR-ING-4 | Clean records: drop missing title/abstract, de-duplicate by arXiv ID, DOI and normalised title | M |
| FR-ING-5 | Download open-access PDFs from arXiv with caching and resume; log failures without crashing | M |
| FR-ING-6 | Parse PDFs with PyMuPDF; strip headers, footers and the reference section from the *chunking* text, but keep the reference section for citation extraction if OpenAlex lacks data | M |
| FR-ING-7 | Extract 5 keywords per paper by TF-IDF (unigrams and bigrams, lemmatised, lower-cased) as in the paper | S |
| FR-ING-8 | Optional relevance filter: keep papers in the top 25% keyword similarity to the seed query, as in the paper | C |
| FR-ING-9 | Produce a **corpus manifest** (`data/processed/corpus_manifest.json`) with counts: papers fetched, with PDF, with citation data, dropped and why | M |
| FR-ING-10 | Ingestion is **idempotent**: re-running never duplicates nodes, chunks or vectors | M |

### 7.2 Knowledge graph (FR-KG)

| ID | Requirement | Pri |
|---|---|---|
| FR-KG-1 | Create the schema in `DATA_SPEC.md` with constraints on unique IDs | M |
| FR-KG-2 | Load Paper, Author, Keyword, Category and Year nodes and WROTE, HAS_KEYWORD, IN_CATEGORY, PUBLISHED_IN, CITES edges | M |
| FR-KG-3 | CITES edges are created **between corpus papers** from OpenAlex referenced works | M |
| FR-KG-4 | A `GraphStore` interface with two implementations: `Neo4jGraphStore` and `NetworkXGraphStore`, selectable by config | M |
| FR-KG-5 | A schema description string (node labels, properties, edge types, direction) generated from the live graph for use in prompts | M |
| FR-KG-6 | Graph statistics report: node counts, edge counts, average degree, share of papers with at least one CITES edge | M |
| FR-KG-7 | Author name normalisation (case, punctuation, initials) to limit duplicate author nodes | S |

### 7.3 Vector and sparse retrieval (FR-VEC)

| ID | Requirement | Pri |
|---|---|---|
| FR-VEC-1 | Recursive character chunking, default about 2,000 characters with 50-character overlap (paper setting), configurable; also test 1,000 characters | M |
| FR-VEC-2 | Embed chunks with `sentence-transformers/all-MiniLM-L6-v2` on CPU (384 dimensions), in batches, cached to disk | M |
| FR-VEC-3 | FAISS index with chunk-to-metadata mapping (paper ID, title, section if known, page) | M |
| FR-VEC-4 | BM25 index (`rank-bm25`) over the same chunks with the same IDs | M |
| FR-VEC-5 | Reciprocal Rank Fusion: `score(d) = Σ 1 / (k + rank_i(d))`, default k = 60, configurable | M |
| FR-VEC-6 | Cross-encoder reranker over the top 20 fused candidates, returning top 5 | M |
| FR-VEC-7 | Every retrieved chunk returns paper ID, chunk ID, and scores from each stage (BM25 rank, dense rank, RRF score, rerank score) so traces and ablations can use them | M |
| FR-VEC-8 | Each stage can be switched off by config (needed for retrieval ablations) | M |

### 7.4 Text-to-Cypher (FR-T2C)

| ID | Requirement | Pri |
|---|---|---|
| FR-T2C-1 | Generate Cypher from a natural-language question using the live schema string plus 15 to 30 few-shot examples | M |
| FR-T2C-2 | **Validate** generated Cypher before execution: read-only guard (reject `CREATE`, `MERGE`, `DELETE`, `SET`, `REMOVE`, `DROP`, `CALL` with write procedures), syntax check with `EXPLAIN`, and label/relationship check against the schema | M |
| FR-T2C-3 | **Retry once** with the error message fed back to the LLM; then give up gracefully | M |
| FR-T2C-4 | Query timeout and `LIMIT` enforcement (default timeout 10 s, default `LIMIT 50` injected if absent) | M |
| FR-T2C-5 | Return the Cypher, the raw rows, and a flag (`ok`, `empty`, `error`, `retried`) | M |
| FR-T2C-6 | When the backend is NetworkX, a documented reduced mode: a fixed set of parameterised query templates instead of free Cypher | S |
| FR-T2C-7 | Few-shot example bank stored as YAML, versioned, with a unit test that **every example's Cypher runs** on the sample graph | M |

### 7.5 Agent (FR-AGT)

| ID | Requirement | Pri |
|---|---|---|
| FR-AGT-1 | Implement the agent as a LangGraph `StateGraph` with typed state (see `ARCHITECTURE.md`) | M |
| FR-AGT-2 | **Router node** outputs one of `graph`, `vector`, `both`, `none` with a short rationale, using few-shot examples (about 5 per route) | M |
| FR-AGT-3 | Router returns structured output (JSON schema) and falls back to `vector` on parse failure | M |
| FR-AGT-4 | **Graph branch**: text-to-Cypher, then rows to evidence passages | M |
| FR-AGT-5 | **Vector branch**: BM25 + dense + RRF + rerank, then chunks to evidence passages | M |
| FR-AGT-6 | **Both**: run branches (in parallel if feasible), merge evidence with source labels | M |
| FR-AGT-7 | **Evidence grader**: an LLM judge rates the evidence for sufficiency (0 to 1) and lists missing information | M |
| FR-AGT-8 | **Bounded rewrite loop**: if the grade is below threshold, rewrite the query (and optionally switch route) and retry. Maximum 2 iterations (configurable). Always terminates | M |
| FR-AGT-9 | **Answer generator**: answers only from evidence; inline citations like `[P12]` mapped to paper titles; says "not found in the corpus" when evidence is insufficient | M |
| FR-AGT-10 | **Uncertainty scorer** (7.6) runs after generation and attaches a score and the contributing signals | M |
| FR-AGT-11 | **Abstain policy**: if uncertainty is above the abstain threshold, the answer is prefixed with a low-confidence warning, or replaced with an abstention in strict mode | S |
| FR-AGT-12 | Every node writes to a trace (inputs, outputs, latency, token counts if available) | M |
| FR-AGT-13 | A `--route` override flag to force `vector` or `graph` for ablations | M |
| FR-AGT-14 | Hard limits: maximum total LLM calls per question (default 12) and wall-clock timeout (default 90 s) | M |

### 7.6 Uncertainty estimation (FR-UNC)

Because the paper does not specify a runtime estimator, LitAgent defines its own. The score is in [0, 1], where **higher means more uncertain**.

| Signal | How it is computed | Applies to |
|---|---|---|
| **S1 Retrieval margin** | Normalised gap between the top-1 and top-k rerank scores, plus score spread | Vector |
| **S2 Evidence grade** | `1 − grader sufficiency score` | All |
| **S3 Cross-source agreement** | When route is `both`, similarity between graph-derived facts and vector-derived facts (entity overlap and embedding similarity of the two partial answers) | Both |
| **S4 Self-consistency** | Generate k = 3 answers at temperature > 0; mean pairwise embedding similarity; `1 − mean` | All |
| **S5 Graph execution flags** | Empty result, retry used, or validation failure raises uncertainty | Graph |

Combination:

1. Start with fixed, documented weights (equal weights over the signals that apply).
2. After the dev QA set exists, fit a logistic regression of "answer was correct/faithful" on the signals and report the learned weights.
3. Report **AUROC**, a reliability diagram and **Expected Calibration Error (ECE)** on the held-out test split.
4. Report honestly if the score is only weakly predictive.

| ID | Requirement | Pri |
|---|---|---|
| FR-UNC-1 | Implement S1, S2, S5 | M |
| FR-UNC-2 | Implement S4 with a config switch (it triples generation cost) | S |
| FR-UNC-3 | Implement S3 for the `both` route | S |
| FR-UNC-4 | Calibration script producing AUROC, ECE and a reliability plot | M |
| FR-UNC-5 | Uncertainty score and signals shown in the UI and written to the trace | M |

### 7.7 Evaluation (FR-EVL)

Details in `EVALUATION.md`.

| ID | Requirement | Pri |
|---|---|---|
| FR-EVL-1 | Graph questions: generated from **templates**; gold answers computed by running the template Cypher, with no LLM involved | M |
| FR-EVL-2 | Vector questions: generated by an LLM from random chunks; **at least 25% hand-verified** (more if time allows) | M |
| FR-EVL-3 | Split into dev (30%) and test (70%); thresholds and weights tuned only on dev | M |
| FR-EVL-4 | Metrics: RAGAS Faithfulness, Answer Relevancy, Context Precision, Context Recall; plus exact-match or set-F1 for graph questions with scripted gold answers | M |
| FR-EVL-5 | Bootstrap: 12 resamples of 20 questions (paper protocol) **and** a larger run with 1,000 resamples of the full set; report mean ± SD and a 95% interval | M |
| FR-EVL-6 | Ablation table: vector-only, graph-only, non-agentic joint baseline, router, router + correction loop | M |
| FR-EVL-7 | Retrieval ablation: BM25 only, dense only, RRF, RRF + rerank, reporting Recall@k and MRR against chunk-level gold | S |
| FR-EVL-8 | Per-question-type breakdown (lookup, multi-hop, aggregation, co-citation, path, content) | M |
| FR-EVL-9 | Cost and latency report per configuration (LLM calls, seconds per question) | S |
| FR-EVL-10 | All results written to `results/` as CSV/JSON with config hashes so runs are reproducible | M |
| FR-EVL-11 | A judge-LLM different from the answer LLM where possible, to reduce self-preference bias | S |

### 7.8 Interfaces (FR-UI)

| ID | Requirement | Pri |
|---|---|---|
| FR-UI-1 | CLI: `litagent ask "question"` prints answer, citations, route, uncertainty, Cypher (if any) | M |
| FR-UI-2 | Gradio chat UI with a side panel showing route, Cypher, retrieved chunks, grader score, uncertainty | S |
| FR-UI-3 | Streaming of node-level progress (router → retrieve → grade → answer) | C |
| FR-UI-4 | An "evidence" tab linking each citation to its arXiv page | S |

### 7.9 Tracing and observability (FR-OBS)

| ID | Requirement | Pri |
|---|---|---|
| FR-OBS-1 | Local JSONL trace for every question, always on, no external service needed | M |
| FR-OBS-2 | Optional LangSmith or Langfuse integration through environment variables | C |
| FR-OBS-3 | A script that prints a trace as a readable tree | S |

---

## 8. Non-functional requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Performance | Median question latency under 20 s with an API LLM on a laptop for the `vector` route; under 30 s for `both`. (Targets, to be measured and reported, not promised) |
| NFR-2 | Hardware | Runs on a CPU-only machine with 16 GB RAM. Full index build for 1,000 papers completes in under 2 hours |
| NFR-3 | Reproducibility | Pinned dependencies (`uv.lock` or `requirements.lock`), fixed random seeds, cached API responses, config-hash on every result file |
| NFR-4 | Cost | A full evaluation run (all ablations) fits within a modest API budget; the README states the measured token usage. A `--max-questions` flag allows cheap smoke runs |
| NFR-5 | Safety | Cypher is read-only; LLM output never executes shell or file operations; API keys only from environment variables |
| NFR-6 | Maintainability | Type hints, ruff and mypy clean, modules under about 300 lines, docstrings on public functions |
| NFR-7 | Testability | Unit tests run offline with no network and no LLM using fakes; integration tests are marked and optional |
| NFR-8 | Licensing | Every reused file is attributed; licences checked (see `REPO_INTEGRATION.md`) |
| NFR-9 | Privacy | No personal data beyond public author names in public metadata |
| NFR-10 | Failure behaviour | Any node failure degrades gracefully (fallback route, "could not retrieve" message) and is logged, never a crash |

---

## 9. Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11 or 3.12 | Avoid 3.13 until every wheel (FAISS, torch) is confirmed available |
| Package manager | `uv` (preferred) or `pip` with a venv | |
| Agent framework | **LangGraph** | Course requirement |
| LLM access | LangChain chat models behind one `llm.py` factory; provider chosen by `.env` | Any API model; provider-agnostic |
| Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` | CPU |
| Vector store | FAISS (`faiss-cpu`) | Pinecone only if the course demands it |
| Sparse | `rank-bm25` | |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` CrossEncoder | Top 20 only |
| Graph DB | Neo4j (Docker) with `neo4j` Python driver | NetworkX fallback |
| Graph helpers | `neo4j-graphrag` (Text2CypherRetriever), `langchain-neo4j` (reference) | See `REPO_INTEGRATION.md` |
| Data | `arxiv`, `pyalex`, `pymupdf` | |
| Evaluation | `ragas`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib` | |
| UI | Gradio | |
| Config | `pydantic-settings` plus YAML | |
| Testing | `pytest`, `pytest-mock` | |
| Lint | `ruff`, `mypy` | |
| Containers | Docker Compose for Neo4j | |

---

## 10. Data requirements

Summarised here; full detail in `DATA_SPEC.md`.

- **Seed query:** a single subfield, default *retrieval-augmented generation* and *information retrieval*, categories `cs.IR` and `cs.CL`, 2020 to 2026.
- **Size:** target 600 papers, minimum 300, maximum 1,000.
- **Selection rule for good citation coverage:** prefer papers with an OpenAlex record and at least 5 referenced works. Report the share of papers dropped by this rule.
- **Storage layout:** `data/raw/` (cached API responses, PDFs), `data/processed/` (clean tables, chunks), `data/index/` (FAISS, BM25), `data/eval/` (question sets, gold answers).
- **Git hygiene:** `data/` is git-ignored except small samples; a `make data` command rebuilds it.

---

## 11. Success metrics and acceptance criteria

### 11.1 Project-level acceptance (the "definition of done")

| # | Criterion |
|---|---|
| A1 | `make setup && make data && make index && make eval` runs from a clean clone and produces the report tables |
| A2 | The agent answers all 12 example questions in Section 4.2 sensibly, including abstaining on the two out-of-scope ones |
| A3 | The ablation table has all five configurations, each with mean ± SD from the bootstrap |
| A4 | The router configuration is not worse than the better single-strategy baseline on the **mixed** question set. If this is not true, the report explains why |
| A5 | Uncertainty calibration plot and AUROC are produced and discussed honestly |
| A6 | Every deviation from the paper is listed in the report |
| A7 | Unit tests pass offline; lint and type checks pass |
| A8 | All reused code is attributed in `THIRD_PARTY.md` with its licence |

### 11.2 Target (not guaranteed) quality bars

| Metric | Target | Comment |
|---|---|---|
| Graph questions, set-F1 vs scripted gold | at least 0.70 | Depends on text-to-Cypher accuracy |
| Valid Cypher rate after one retry | at least 90% | |
| Router accuracy vs hand-labelled route | at least 85% | On a 40-question labelled set |
| Faithfulness (RAGAS), vector route | at least 0.75 | Depends on the judge LLM |
| Out-of-scope abstention rate | at least 80% | |

These are goals to measure against, not promises. The report states what was actually achieved.

---

## 12. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Text-to-Cypher produces wrong or invalid queries | High | High | Small schema, 15 to 30 verified few-shots, validation, retry once, template fallback, unit-tested example bank |
| R2 | Sparse citation links between corpus papers (OpenAlex gaps) | Medium | High | Choose a subfield with good coverage; require at least 5 referenced works; add co-citation through shared external references; report coverage statistics early (Phase 1 gate) |
| R3 | arXiv or OpenAlex rate limits or downtime | Medium | Medium | Cache everything, polite delays, resumable downloads, small initial pull |
| R4 | PDF parsing noise (two-column text, formulas) | High | Medium | PyMuPDF block sorting; drop short noisy chunks; test on 20 PDFs first; abstract-only fallback |
| R5 | LLM-judged RAGAS scores are noisy and costly | High | Medium | Bootstrap, fixed seeds, judge-model different from answer model, report SD, cap question count |
| R6 | Router errors cascade into bad answers | Medium | High | Labelled router test set, few-shot tuning on dev, `both` fallback when router confidence is low |
| R7 | API cost or rate limits during evaluation | Medium | Medium | Response caching, `--max-questions`, small smoke runs, retries with backoff |
| R8 | Neo4j or Docker setup blocks a teammate | Medium | Medium | NetworkX fallback behind the same interface; Phase 0 gate checks Docker |
| R9 | Hand-checking workload | High | Medium | Scripted graph gold; verify only a stratified 25% of vector questions; split verification between two people |
| R10 | Scope creep | High | High | Scope dials (5.3), phase gates, "should/could" items cut first |
| R11 | Coding agent (Antigravity) hallucinates APIs or invents files | Medium | Medium | `AGENTS.md` rules, version pins, tests per phase, review diffs, never accept unverified imports |
| R12 | Licence problems from copied code | Low | Medium | Prefer pip installs; copy only MIT/Apache/CC0 code with attribution |
| R13 | Python 3.13 wheel gaps for faiss/torch | Medium | Low | Use 3.11 or 3.12 |
| R14 | Self-consistency (S4) is expensive | Medium | Low | Off by default; run only in the uncertainty experiment |

---

## 13. Assumptions and dependencies

**Assumptions**
- Two developers, each able to give roughly 15 to 20 hours per week (assumed from the 4-week estimate; adjust if wrong).
- Internet access to arXiv, OpenAlex, PyPI, GitHub and one LLM API.
- Docker Desktop (or Docker Engine) works on at least one machine.
- The LLM API supports structured/JSON output or at least reliable instruction following.

**External dependencies**
- arXiv API terms of use; OpenAlex API (polite pool: provide a contact email in config).
- LLM provider quotas.
- Neo4j Community Edition Docker image.

---

## 14. Deliverables

| # | Deliverable | Location |
|---|---|---|
| D1 | Source code | `src/litagent/` |
| D2 | Tests | `tests/` |
| D3 | Data build scripts and manifest | `scripts/`, `data/processed/corpus_manifest.json` |
| D4 | Evaluation question sets and gold answers | `data/eval/` |
| D5 | Result tables and figures | `results/` |
| D6 | Few-shot banks (router, text-to-Cypher, grader) | `prompts/` |
| D7 | Third-party attribution | `THIRD_PARTY.md` |
| D8 | Written report | `docs/report.md` (outline in `REPORT_OUTLINE.md`) |
| D9 | Demo (UI or recorded CLI session) | `docs/demo/` |
| D10 | README with one-command reproduction | `README.md` |

---

## 15. Open questions (resolve in Phase 0)

| # | Question | Default if unanswered |
|---|---|---|
| Q1 | Which LLM provider and model will be used (answer model and judge model)? | One provider, two model sizes: a stronger one for judging, a cheaper one for answering |
| Q2 | Is Pinecone required by the course, or is FAISS acceptable? | FAISS |
| Q3 | Is LangSmith or Langfuse required for tracing? | Local JSONL traces, optional LangSmith |
| Q4 | Which subfield and date range? | RAG / IR, `cs.IR` + `cs.CL`, 2020 to 2026 |
| Q5 | Is Neo4j mandatory, or is NetworkX acceptable? | Neo4j primary, NetworkX fallback |
| Q6 | Who does what? | Dev A: data, graph, text-to-Cypher. Dev B: vector, agent, evaluation. Both: integration and report |

---

## 16. Glossary

| Term | Meaning |
|---|---|
| RAG | Retrieval-Augmented Generation |
| GraphRAG | Retrieval from a knowledge graph, here by generating Cypher |
| VectorRAG | Retrieval from text chunks by similarity |
| BM25 | A sparse, keyword-based ranking function |
| Dense retrieval | Embedding-based similarity search |
| RRF | Reciprocal Rank Fusion, a rank-merging method |
| Cross-encoder | A model scoring (query, passage) pairs jointly, used for reranking |
| CRAG | Corrective RAG: grade retrieved evidence, then correct or rewrite |
| RAGAS | An evaluation framework with LLM-judged RAG metrics |
| Bootstrap | Resampling the question set to estimate the spread of a metric |
| ECE | Expected Calibration Error |
| AUROC | Area under the ROC curve, here for predicting correctness from uncertainty |
| Cypher | Neo4j's graph query language |
