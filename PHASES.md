# PHASES.md — Development phases for LitAgent

Total plan: **8 phases over about 4 weeks for two developers** (an estimate; re-plan at the Phase 1 gate once you know how slow the data pull really is).

- **Dev A** = data, graph, text-to-Cypher.
- **Dev B** = vector retrieval, agent, evaluation.
- **Antigravity** = the coding agent that does most typing, working from `AGENTS.md`.
- Every phase ends with a **gate**: a checklist whose real command output is pasted into `docs/gates/phase-N.md`. Do not start the next phase until the gate passes.

## Timeline overview

| Phase | Name | Calendar (suggested) | Main owner | Gate in one line |
|---|---|---|---|---|
| 0 | Setup and decisions | Days 1 to 2 | Both | Environment runs; open questions answered |
| 1 | Data ingestion | Days 3 to 7 | Dev A | Clean corpus manifest, citation coverage known |
| 2 | Stores (graph, vector, sparse) | Days 6 to 10 | A graph, B vector | Both indexes built and queryable |
| 3 | Retrieval components | Days 9 to 14 | A Cypher, B hybrid | Each retriever passes its quality checks alone |
| 4 | LangGraph agent v1 | Days 13 to 17 | Dev B | End-to-end answer with route and citations |
| 5 | Correction loop and uncertainty | Days 16 to 20 | Both | Loop terminates; uncertainty score produced |
| 6 | Evaluation | Days 18 to 25 | Dev B, A helps | All ablation rows with mean ± SD |
| 7 | UI, tracing, polish | Days 24 to 27 | Both | Demo works from a clean clone |
| 8 | Report and buffer | Days 26 to 28 | Both | Report complete; deviations listed |

Overlaps are deliberate. Buffer: 2 to 3 days are included in Phase 8; if you are behind, cut using the dials in `PRD.md` §5.3.

---

## Phase 0 — Setup and decisions (Days 1 to 2)

### Objectives
Working environment, repo skeleton, decisions made, reference repos cloned.

### Tasks

| # | Task | Owner | Notes |
|---|---|---|---|
| 0.1 | Create repo, `git init`, copy spec files in, add `.gitignore` (`data/`, `third_party/`, `.env`, `*.faiss`, `.venv/`) | A | |
| 0.2 | Create the folder layout from `AGENTS.md` §2 with empty `__init__.py` files | Antigravity | |
| 0.3 | Write `pyproject.toml` with pinned versions; create venv with Python 3.11 or 3.12 | A | Use `uv` if available |
| 0.4 | Run `scripts/clone_repos.sh` | A | Produces `third_party/SOURCES.txt` |
| 0.5 | Start Neo4j with `docker compose up -d`; open the browser UI on port 7474 and log in | A | If Docker is a blocker, switch to NetworkX now and note it |
| 0.6 | Create `.env` from `.env.example`: LLM key, OpenAlex contact email, Neo4j password | Both | Never commit |
| 0.7 | Implement `config.py` and `llm.py` (provider factory, retry, disk cache, trace hook) | B | Include a **fake LLM** for tests |
| 0.8 | Hello-world smoke tests: one LLM call; embed one sentence; one Cypher `RETURN 1` | Both | |
| 0.9 | Answer the open questions Q1 to Q6 in `PRD.md` §15 and record in `docs/decisions.md` | Both | |
| 0.10 | Set up `ruff`, `mypy`, `pytest`, `Makefile` targets (`setup`, `lint`, `test`, `data`, `index`, `eval`) | Antigravity | |

### Deliverables
Repo skeleton, `docs/decisions.md`, `third_party/SOURCES.txt`, passing smoke tests.

### Gate 0
- [ ] `make lint` and `make test` pass on the empty skeleton
- [ ] One real LLM call returns text and is cached on the second call
- [ ] Neo4j reachable **or** NetworkX chosen and recorded in `decisions.md`
- [ ] Q1 to Q6 answered
- [ ] All 7 repos cloned; SHAs recorded

### Risks
Docker issues (R8), API key problems, Python 3.13 wheel gaps (R13).

---

## Phase 1 — Data ingestion (Days 3 to 7)

### Objectives
A clean, de-duplicated corpus of 300 to 1,000 papers with metadata, full text, and citation links, plus a manifest that says exactly what you have.

### Tasks

| # | Task | Owner | Notes |
|---|---|---|---|
| 1.1 | `ingest/arxiv_fetch.py`: query arXiv by seed query and categories; paginate; cache raw Atom responses; 3-second delay; descriptive user agent | A | FR-ING-1, 2 |
| 1.2 | `ingest/clean.py`: drop missing title or abstract; de-duplicate by arXiv ID, DOI, normalised title | A | FR-ING-4 |
| 1.3 | `ingest/openalex_enrich.py`: for each paper find the OpenAlex work (by DOI, else by title and year); store `referenced_works`, `cited_by_count`, concepts | A | FR-ING-3; set contact email for the polite pool |
| 1.4 | **Citation coverage check**: compute the share of papers with at least one **in-corpus** CITES edge and the average in-corpus out-degree | A | Early warning for R2 |
| 1.5 | `ingest/pdf_parse.py`: download PDFs with cache and resume; parse with PyMuPDF; sort blocks for two-column layouts; remove headers, footers, page numbers; split off the references section | B | FR-ING-5, 6 |
| 1.6 | Smoke-test the parser on the ~50 PDFs from `third_party/dsa-public/LitChat/testset_input_pdfs/` (local only) | B | Inspect 10 outputs by eye |
| 1.7 | `ingest/keywords.py`: TF-IDF 5 keywords per paper (unigram and bigram, lemmatised) | B | FR-ING-7 |
| 1.8 | `ingest/chunking.py`: recursive splitter with configurable size and overlap; keep paper ID and page | B | Two configs: about 2,000 and about 1,000 characters |
| 1.9 | `scripts/build_corpus.py`: runs everything, idempotent, writes `data/processed/*.parquet` or JSONL and `corpus_manifest.json` | A | FR-ING-9, 10 |
| 1.10 | Unit tests: cleaning rules, dedupe, idempotency, parser on a tiny fixture PDF | Both | Offline tests only |

### Deliverables
`data/processed/papers.jsonl`, `chunks.jsonl`, `citations.jsonl`, `corpus_manifest.json`, coverage report.

### Gate 1
- [ ] Manifest shows at least 300 papers with full text (or the reduced-scope decision is written down)
- [ ] At least 60% of papers have at least 1 in-corpus citation edge **or** the seed query was changed and the pull repeated
- [ ] Re-running `build_corpus.py` leaves counts unchanged (idempotent)
- [ ] 10 hand-inspected PDFs parse with readable text
- [ ] Unit tests pass

### Decision point
If coverage is poor, you have three options, in this order: (1) narrow to a denser subfield, (2) widen the year range, (3) add `CITES_EXTERNAL` edges to non-corpus works (enables co-citation questions even when direct links are few).

### Risks
R2, R3, R4.

---

## Phase 2 — Stores: graph, vector, sparse (Days 6 to 10)

### Objectives
Three queryable stores built from the processed data, behind clean interfaces.

### Tasks

| # | Task | Owner | Notes |
|---|---|---|---|
| 2.1 | `stores/graph_base.py`: `GraphStore` protocol (`load`, `run_read(query, params)`, `schema_text()`, `stats()`) | A | FR-KG-4 |
| 2.2 | `stores/graph_neo4j.py`: constraints and indexes; batch `UNWIND ... MERGE` loaders; idempotent | A | FR-KG-1, 2, 3 |
| 2.3 | `stores/graph_networkx.py`: same interface; Cypher unsupported, template queries only | A | Fallback (FR-T2C-6) |
| 2.4 | `schema_text()`: generated from the live graph (labels, properties, relationship types and directions) | A | FR-KG-5 |
| 2.5 | Author normalisation (lower-case, strip punctuation, handle initials) | A | FR-KG-7 |
| 2.6 | Graph stats report (nodes, edges, degree, coverage) | A | FR-KG-6 |
| 2.7 | `stores/vector_faiss.py`: embed chunks with all-MiniLM-L6-v2 in batches; save index and ID map; load | B | FR-VEC-2, 3 |
| 2.8 | `stores/sparse_bm25.py`: tokenise (lower-case, simple stop-word removal); build and persist | B | FR-VEC-4 |
| 2.9 | `scripts/build_index.py` builds all stores; idempotent; timing logged | Both | NFR-2 |
| 2.10 | Tests: tiny fixture graph and 20-chunk fixture index; `load` twice leaves counts unchanged | Both | |

### Deliverables
Populated Neo4j (or NetworkX pickle), `data/index/faiss.index`, `data/index/bm25.pkl`, `docs/graph_stats.md`.

### Gate 2
- [ ] Neo4j Browser shows expected node and edge counts matching `papers.jsonl` and `citations.jsonl`
- [ ] `MATCH (p:Paper)-[:CITES]->(q:Paper) RETURN p.title, q.title LIMIT 5` returns real rows
- [ ] FAISS and BM25 return the same chunk IDs for an exact-title query
- [ ] Index build finished in under 2 hours on a laptop (NFR-2); otherwise reduce corpus
- [ ] Idempotency tests pass

### Risks
Embedding time on CPU (batch size, cache), Neo4j memory settings.

---

## Phase 3 — Retrieval components (Days 9 to 14)

### Objectives
Each retriever works and is measured **on its own**, before any agent exists.

### Tasks — vector side (Dev B)

| # | Task | Notes |
|---|---|---|
| 3.1 | `retrieval/dense.py` and `retrieval/sparse.py` returning ranked `(chunk_id, score)` | |
| 3.2 | `retrieval/rrf.py` with k = 60, tests against a hand-worked example | Show the arithmetic in the test |
| 3.3 | `retrieval/rerank.py`: CrossEncoder on top 20 fused candidates, return top 5; batching; CPU timing | FR-VEC-6 |
| 3.4 | Stage toggles via config; every result carries stage scores | FR-VEC-7, 8 |
| 3.5 | Quick retrieval check: 30 hand-written (query, expected paper) pairs; compute Recall@5 and MRR for BM25, dense, RRF, RRF + rerank | Early ablation preview (FR-EVL-7) |

### Tasks — graph side (Dev A)

| # | Task | Notes |
|---|---|---|
| 3.6 | **Decision (Day 1 of phase):** wrap `neo4j_graphrag.Text2CypherRetriever` or write own thin module | Timebox 1 hour to try wrapping |
| 3.7 | Few-shot bank `prompts/cypher_fewshot.yaml`: 15 to 30 examples written for **your** schema, seeded from `third_party/text2cypher/datasets` patterns | FR-T2C-1 |
| 3.8 | `retrieval/cypher_guard.py`: read-only check, `EXPLAIN` syntax check, label and relationship check, `LIMIT` injection, timeout | FR-T2C-2, 4 |
| 3.9 | Retry-once with error feedback; result flags (`ok`, `empty`, `error`, `retried`) | FR-T2C-3, 5 |
| 3.10 | Test: **every few-shot example's Cypher executes** on the fixture graph and returns a non-error result | FR-T2C-7 |
| 3.11 | Quick graph check: 20 template questions with scripted gold answers; compute valid-Cypher rate and set-F1 | Preview of FR-EVL-1 |
| 3.12 | NetworkX template queries (if fallback in use) | |

### Deliverables
`retrieval/*` modules, `prompts/cypher_fewshot.yaml`, `results/phase3_retrieval_check.csv`.

### Gate 3
- [ ] RRF unit test passes with a worked example
- [ ] Reranker improves or matches MRR vs RRF alone on the 30-pair check (if it does not, record that honestly)
- [ ] Valid Cypher rate at least 80% after one retry on the 20-question check (target 90% by Phase 6)
- [ ] Every few-shot example executes
- [ ] No write clause can pass the guard (adversarial test with `CREATE`, `DELETE`, `CALL dbms...`)

### Risks
R1, wrong relationship directions in generated Cypher, slow CPU reranking (limit to top 20).

---

## Phase 4 — LangGraph agent v1 (Days 13 to 17)

### Objectives
End-to-end question answering with routing, two branches, answer generation and citations. No loop or uncertainty yet.

### Tasks

| # | Task | Notes |
|---|---|---|
| 4.1 | `agent/state.py`: typed state (`question`, `route`, `rationale`, `cypher`, `rows`, `chunks`, `evidence`, `grade`, `iteration`, `answer`, `citations`, `uncertainty`, `trace`) | See `ARCHITECTURE.md` |
| 4.2 | `prompts/router_fewshot.yaml`: about 5 examples per route (graph, vector, both, none) | |
| 4.3 | `agent/router.py`: structured JSON output; fallback to `vector` on parse failure | FR-AGT-2, 3 |
| 4.4 | `agent/graph_branch.py` and `agent/vector_branch.py`: convert rows or chunks to uniform **evidence items** `{id, source, text, paper_ids, scores}` | FR-AGT-4, 5 |
| 4.5 | `agent/generate.py`: answer only from evidence; `[P#]` citations; "not found in the corpus" behaviour; evidence marked as data, not instructions | FR-AGT-9 |
| 4.6 | `agent/workflow.py`: build the `StateGraph`; conditional edges from router; `both` runs both branches and merges | FR-AGT-1, 6 |
| 4.7 | `--route` override flag | FR-AGT-13 |
| 4.8 | Hard limits: max LLM calls, wall-clock timeout | FR-AGT-14 |
| 4.9 | `ui/cli.py`: `litagent ask "..."` prints answer, route, Cypher, citations | FR-UI-1 |
| 4.10 | Router test set: 40 hand-labelled questions (10 per route); measure accuracy | Tune few-shots on the dev half only |
| 4.11 | Run the 12 example questions from `PRD.md` §4.2 and save outputs | |
| 4.12 | Tests with fake LLM: routing, branch merging, failure fallbacks | |

### Deliverables
Working CLI, `results/phase4_example_runs.md`, router accuracy number.

### Gate 4
- [ ] The 12 example questions all return an answer or an abstention with a route shown
- [ ] Router accuracy at least 80% on the dev half (target 85% by Phase 6)
- [ ] Every sentence-level claim in sample answers has a citation or is a "not found" statement (spot-check 10)
- [ ] Out-of-scope questions do not get hallucinated answers (spot-check 2)
- [ ] Failure injection tests pass (Neo4j down, empty retrieval, LLM parse error)

### Risks
R6, prompt injection from paper text, answers that cite the wrong paper.

---

## Phase 5 — Correction loop and uncertainty (Days 16 to 20)

### Objectives
Add the CRAG-style grade-and-rewrite loop and the uncertainty score.

### Tasks

| # | Task | Notes |
|---|---|---|
| 5.1 | `prompts/grader.md` and `agent/grader.py`: sufficiency score 0 to 1 plus missing-information list, structured output | FR-AGT-7 |
| 5.2 | `agent/rewrite.py`: rewrite the query using the grader's missing-information list; optionally switch route (for example, `graph` to `both` when the graph returned empty) | FR-AGT-8 |
| 5.3 | Add conditional edges: `grade ≥ threshold` → generate; else → rewrite → retrieve; loop bound default 2 | |
| 5.4 | Termination test: fake grader always says "insufficient"; workflow still ends within the bound | AGENTS.md §4 |
| 5.5 | `agent/uncertainty.py`: S1 retrieval margin, S2 evidence grade, S5 graph flags | FR-UNC-1 |
| 5.6 | S4 self-consistency (k = 3, config switch) and S3 cross-source agreement | FR-UNC-2, 3 |
| 5.7 | Abstain policy and low-confidence warning text | FR-AGT-11 |
| 5.8 | Write signals and score into the trace | FR-UNC-5 |
| 5.9 | Choose the grader threshold on the **dev** split only (try 0.5, 0.6, 0.7) | |
| 5.10 | Tests for each signal with fake inputs and known expected values | |

### Deliverables
Agent v2 with loop and uncertainty; `docs/uncertainty_design.md` describing signals, formulas, defaults.

### Gate 5
- [ ] Loop never exceeds the iteration bound across 100 randomised fake-LLM runs
- [ ] Uncertainty score present on every answer, in [0, 1], with the contributing signals listed
- [ ] On 20 dev questions, answers marked low-confidence include more wrong answers than high-confidence ones (a first sanity check, not the final calibration)
- [ ] Average LLM calls per question measured and under the cap

### Risks
Loop adds latency and cost (R7, R14); grader too lenient or too strict; self-consistency cost.

---

## Phase 6 — Evaluation (Days 18 to 25)

### Objectives
Full, reproducible evaluation: question sets, gold answers, RAGAS, bootstrap, ablations, calibration. Details in `EVALUATION.md`.

### Tasks

| # | Task | Owner | Notes |
|---|---|---|---|
| 6.1 | `eval/qa_graph.py`: template question generator and gold from Cypher results (no LLM); about 50 questions across 5 types | A | FR-EVL-1 |
| 6.2 | `eval/qa_vector.py`: LLM-generated questions from random chunks; store source chunk as gold context; about 50 questions | B | FR-EVL-2 |
| 6.3 | **Human verification** of a stratified at least 25% of vector questions and 10% of graph questions; record verifier and verdict | Both | R9 |
| 6.4 | 10 to 15 out-of-scope and 10 to 15 hybrid questions (hand-written) | Both | |
| 6.5 | Dev/test split (30/70), fixed seed; store split IDs | B | FR-EVL-3 |
| 6.6 | `eval/ragas_runner.py`: Faithfulness, Answer Relevancy, Context Precision, Context Recall with a judge LLM distinct from the answer LLM where possible | B | FR-EVL-4, 11 |
| 6.7 | `eval/bootstrap.py`: paper protocol (12 × 20) and large protocol (1,000 resamples) | B | FR-EVL-5 |
| 6.8 | `eval/ablations.py`: configs in `configs/ablations/*.yaml` — `vector_only`, `graph_only`, `joint_baseline`, `router`, `router_crag` | Both | FR-EVL-6 |
| 6.9 | Retrieval ablation: BM25, dense, RRF, RRF + rerank; Recall@k and MRR | B | FR-EVL-7 |
| 6.10 | Per-question-type breakdown, latency and cost columns | B | FR-EVL-8, 9 |
| 6.11 | `eval/calibration.py`: fit uncertainty weights on dev; AUROC, ECE, reliability plot on test | A | FR-UNC-4 |
| 6.12 | Smoke run (`--max-questions 10`), then a full run; cache every LLM response | Both | R7 |
| 6.13 | Error analysis: read 20 failures, tag by cause (router, Cypher, retrieval, generation, gold error) | Both | Goes into the report |

### Deliverables
`data/eval/*.jsonl`, `results/ablation_table.csv`, `results/figures/*.png`, `results/calibration.json`, `docs/error_analysis.md`.

### Gate 6
- [ ] Ablation table has all five configs with mean ± SD
- [ ] Question sets have a recorded hand-verification rate (at least 25% vector, 10% graph)
- [ ] Test split untouched by any tuning (checked against git history of configs and prompts)
- [ ] `make eval` reproduces tables from cache without new API calls
- [ ] Calibration outputs exist (AUROC, ECE, plot)

### Risks
R5, R7, R9; gold-answer errors; judge bias.

---

## Phase 7 — UI, tracing, polish (Days 24 to 27)

| # | Task | Owner |
|---|---|---|
| 7.1 | Gradio chat UI with side panel: route, Cypher, evidence chunks, grader score, uncertainty (FR-UI-2, 4) | B |
| 7.2 | Trace pretty-printer script (FR-OBS-3); optional LangSmith or Langfuse via env vars (FR-OBS-2) | A |
| 7.3 | README with the one-command path: `make setup data index eval`; hardware notes; measured token usage and timings | A |
| 7.4 | Fresh-clone test on a second machine or clean directory | Both |
| 7.5 | Lint, type check, test coverage report; remove dead code | Both |
| 7.6 | `THIRD_PARTY.md` completed and checked against actual copied code | A |
| 7.7 | Record a 2 to 3 minute demo covering: a graph question, a vector question, a hybrid question, an out-of-scope abstention, the trace | Both |

### Gate 7
- [ ] A person who has never seen the project reaches a working answer in under 30 minutes using only the README
- [ ] Demo recorded
- [ ] `THIRD_PARTY.md` matches the code
- [ ] All CI-style checks pass

---

## Phase 8 — Report and buffer (Days 26 to 28)

| # | Task |
|---|---|
| 8.1 | Write the report using `REPORT_OUTLINE.md`; every number comes from `results/` |
| 8.2 | Deviations section (PRD §2.3), limitations, threats to validity |
| 8.3 | Re-run the final evaluation once from clean state; freeze the results with config hashes and commit SHA |
| 8.4 | Buffer for fixes |

### Gate 8
- [ ] Every table in the report is regenerated by a script
- [ ] Deviations, failures and limitations are written down
- [ ] Final commit tagged `v1.0`

---

## Dependency graph

```
P0 ──► P1 ──► P2 ──► P3 ──► P4 ──► P5 ──► P6 ──► P7 ──► P8
              │       ▲      ▲
              └───────┴──────┘  (vector and graph tracks run in parallel in P2 and P3)
```

## Cut order if you fall behind (do these in order)

1. Drop S3 cross-source agreement and S4 self-consistency (keep S1, S2, S5).
2. Drop the retrieval ablation (FR-EVL-7).
3. Reduce QA set to 60 questions.
4. Gradio UI → CLI only.
5. Corpus 600 → 300 papers; Neo4j → NetworkX.
6. **Never cut:** read-only Cypher guard, bootstrap with SD, the five-row ablation, the deviations section.

## Weekly rhythm

| Day | Activity |
|---|---|
| Daily, 10 minutes | Stand-up: what the agent did, what broke, what needs a human decision |
| End of each phase | Gate review; paste evidence into `docs/gates/phase-N.md` |
| Weekly | Review the `THIRD_PARTY.md` and `docs/CHANGELOG.md` diff |
