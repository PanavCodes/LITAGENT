# ARCHITECTURE.md — LitAgent technical design

## 1. Component view

```
┌─────────────────────────── Offline build (make data, make index) ───────────────────────────┐
│ arXiv API ─► fetch ─► clean ─┬─► OpenAlex enrich ─► citations ──────► GraphStore (Neo4j/NX)   │
│                              ├─► PDF download ─► parse ─► chunk ─┬──► FAISS (dense, MiniLM)   │
│                              └─► TF-IDF keywords                 └──► BM25 (sparse)           │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────── Online (per question) ───────────────────────────────────────────┐
│ question                                                                                       │
│    │                                                                                           │
│    ▼                                                                                           │
│  ROUTER ──graph──► TEXT2CYPHER ► guard ► run ► rows ─┐                                         │
│    │ ├──vector─► BM25 + DENSE ► RRF ► RERANK ► chunks ─┤                                       │
│    │ ├──both───► (both of the above, merged) ─────────┤                                        │
│    │ └──none───► ABSTAIN (out of scope) ──────────────┼──► final                              │
│                                                       ▼                                         │
│                                              EVIDENCE GRADER                                    │
│                                    sufficient │            │ insufficient                       │
│                                               │            ▼                                    │
│                                               │      REWRITE (≤ 2 loops, may switch route)      │
│                                               │            │                                    │
│                                               ▼            └──► back to retrieval               │
│                                          GENERATE (cited)                                       │
│                                               ▼                                                 │
│                                     UNCERTAINTY SCORER ─► answer + citations + score + trace    │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Agent state (LangGraph)

Use a `TypedDict` or Pydantic model. Fields:

| Field | Type | Meaning |
|---|---|---|
| `question` | str | Original user question |
| `current_query` | str | Possibly rewritten query used for retrieval |
| `route` | Literal["graph","vector","both","none"] | Router decision |
| `route_rationale` | str | One-line reason |
| `forced_route` | Optional[str] | Ablation override |
| `cypher` | Optional[str] | Final Cypher that ran |
| `cypher_status` | Optional[Literal["ok","empty","error","retried"]] | Outcome flag |
| `rows` | list[dict] | Graph result rows |
| `chunks` | list[ChunkHit] | Vector hits with stage scores |
| `evidence` | list[EvidenceItem] | Unified evidence list |
| `grade` | Optional[float] | Sufficiency 0 to 1 |
| `missing_info` | list[str] | What the grader says is missing |
| `iteration` | int | Rewrite loop counter |
| `llm_calls` | int | Budget counter |
| `answer` | Optional[str] | Final text |
| `citations` | list[Citation] | Map `[P#]` to paper metadata |
| `uncertainty` | Optional[UncertaintyReport] | Score and signals |
| `trace` | list[TraceEvent] | Node-level events |
| `errors` | list[str] | Non-fatal failures |

### Shared types

```python
class EvidenceItem(BaseModel):
    id: str                      # "E1", "E2", ...
    source: Literal["graph", "vector"]
    text: str                    # row rendered as a sentence, or the chunk text
    paper_ids: list[str]         # arXiv IDs
    scores: dict[str, float]     # bm25_rank, dense_rank, rrf, rerank, etc.

class Citation(BaseModel):
    tag: str                     # "P3"
    paper_id: str                # arXiv ID
    title: str
    url: str                     # https://arxiv.org/abs/<id>

class UncertaintyReport(BaseModel):
    score: float                 # 0 certain .. 1 very uncertain
    signals: dict[str, float]    # S1, S2, S3, S4, S5 as available
    weights: dict[str, float]
    abstain: bool
```

## 3. Nodes and edges

| Node | Input | Output | LLM calls | Notes |
|---|---|---|---|---|
| `router` | question | route, rationale | 1 | Few-shot JSON; fallback `vector` |
| `graph_branch` | current_query | cypher, rows, status, evidence | 1 (2 with retry) | Uses guard |
| `vector_branch` | current_query | chunks, evidence | 0 | BM25 + dense + RRF + rerank |
| `merge_evidence` | both branches | evidence | 0 | Re-number IDs; keep source label |
| `grader` | question, evidence | grade, missing_info | 1 | Structured JSON |
| `rewrite` | question, missing_info, route | current_query, maybe new route, iteration+1 | 1 | |
| `generate` | question, evidence | answer, citations | 1 (k with S4) | Evidence is data |
| `uncertainty` | all above | UncertaintyReport | 0 (plus S4 samples) | |
| `abstain` | question | fixed answer | 0 | For route `none` |

### Edges

```
START → router
router → graph_branch | vector_branch | both_branches | abstain     (conditional)
graph_branch / vector_branch / both_branches → grader               (merge if both)
grader → generate                    if grade ≥ τ  or iteration ≥ max_iter
grader → rewrite                     otherwise
rewrite → graph_branch | vector_branch | both_branches              (conditional)
generate → uncertainty → END
abstain → END
```

Budget guard: every node increments `llm_calls`; a wrapper short-circuits to `generate` (with whatever evidence exists) if `llm_calls ≥ max_llm_calls` or wall-clock exceeds the timeout.

## 4. Retrieval pipeline detail

### 4.1 Vector branch

1. Embed `current_query` with MiniLM (same model as the index).
2. Dense top 20 from FAISS, BM25 top 20.
3. RRF: `score(d) = Σ_{r ∈ {bm25, dense}} 1/(k + rank_r(d))`, k = 60, over the union.
4. Take the top 20 fused; run CrossEncoder on `(query, chunk)` pairs; keep top 5.
5. Return chunk ID, paper ID, text, and every stage score.

The paper retrieved 5 from each retriever and reranked; we use 20 for more headroom and ablate it.

### 4.2 Graph branch

1. Build the prompt: system instructions, **live schema text**, 15 to 30 few-shot (question, Cypher) pairs, the user question.
2. LLM returns Cypher only (strip code fences).
3. **Guard:** read-only token check → `EXPLAIN` → label and relationship-direction check → `LIMIT` injection → run in a read transaction with a timeout.
4. On error, retry once, passing the error text.
5. Render rows as short sentences: `"Paper 'X' (2301.01234) cites 'Y' (2005.11401)."`
6. Status flags: `ok`, `empty`, `error`, `retried`.

### 4.3 Evidence rendering rules

- One evidence item per row for small results; group rows if there are more than 15.
- Always include arXiv ID so citations can be resolved.
- Truncate chunk text to a configured maximum before sending to the LLM.

## 5. Uncertainty

See `PRD.md` §7.6 for definitions. Implementation notes:

```python
def retrieval_margin(rerank_scores: list[float]) -> float:
    """S1: uncertainty from top-1 vs rest. Returns 0..1, high = uncertain."""
    # normalise scores with a sigmoid or min-max over the candidate list,
    # margin = top1 - mean(top2..k); uncertainty = 1 - clip(margin, 0, 1)

def evidence_grade_uncertainty(grade: float) -> float:   # S2
    return 1.0 - grade

def graph_flags_uncertainty(status: str) -> float:       # S5
    return {"ok": 0.0, "retried": 0.3, "empty": 0.8, "error": 1.0}[status]

def self_consistency(answers: list[str], embed) -> float:  # S4
    # mean pairwise cosine similarity of answer embeddings; return 1 - mean

def cross_source_agreement(graph_answer: str, vector_answer: str, embed) -> float:  # S3
    # 1 - cosine(graph_partial, vector_partial)
```

Combination: weighted mean over available signals. Default weights equal. After dev data exists, `eval/calibration.py` fits logistic regression and writes `configs/uncertainty_weights.yaml`. The abstain threshold is picked on dev to hit a chosen precision target and then frozen before touching test.

## 6. Configuration

`configs/default.yaml`:

```yaml
seed: 42
llm:
  provider: ${LLM_PROVIDER}        # from .env
  answer_model: ${ANSWER_MODEL}
  judge_model: ${JUDGE_MODEL}
  temperature: 0.0
  cache_dir: data/cache/llm
data:
  query: "retrieval-augmented generation OR information retrieval"
  categories: [cs.IR, cs.CL]
  year_min: 2020
  max_papers: 600
chunking: {size: 2000, overlap: 50}
embedding: {model: sentence-transformers/all-MiniLM-L6-v2, batch_size: 64}
retrieval:
  use_bm25: true
  use_dense: true
  use_rrf: true
  use_rerank: true
  candidates_per_retriever: 20
  rrf_k: 60
  rerank_top_in: 20
  rerank_top_out: 5
graph:
  backend: neo4j                    # or networkx
  uri: bolt://localhost:7687
  query_timeout_s: 10
  default_limit: 50
agent:
  grader_threshold: 0.6
  max_iterations: 2
  max_llm_calls: 12
  timeout_s: 90
uncertainty:
  use_self_consistency: false
  samples: 3
  abstain_threshold: 0.7
  strict_abstain: false
```

Ablation configs override only the relevant keys, for example `configs/ablations/vector_only.yaml` sets `agent.forced_route: vector`.

## 7. Tracing

Every node wraps its function with a decorator that appends a `TraceEvent` (node, start, duration, input summary, output summary, token counts if available, error) and writes JSONL to `results/traces/<run_id>.jsonl`. Optional LangSmith or Langfuse callbacks attach through environment variables only.

## 8. Failure handling matrix

| Failure | Detection | Behaviour |
|---|---|---|
| Router JSON parse error | Pydantic validation fails | Route = `vector`; log |
| Cypher invalid after retry | Guard or Neo4j error | Status `error`; if route was `graph`, rewrite switches to `both`; else continue with vector evidence |
| Neo4j down | Driver exception | Disable graph branch for the run; answer with vector evidence; set uncertainty S5 = 1 |
| Empty retrieval | No chunks | Grader gets empty evidence → low grade → rewrite → likely "not found in the corpus" |
| LLM rate limit | HTTP 429 | Backoff, retry up to 3 times, then fail the node |
| Loop never satisfied | iteration ≥ max | Proceed to generate with best evidence and high uncertainty |
| Prompt injection in chunk text | Pattern checks plus prompt rule | Evidence is wrapped in delimiters and labelled as untrusted data |

## 9. Security design

- Graph access uses a **read-only** database user if you create one, plus a read transaction.
- Disallowed Cypher tokens list is enforced after stripping comments and string literals.
- Secrets only in `.env`; `.env` is in `.gitignore`; `.env.example` has no real values.
- Evidence delimiters: `<evidence id="E1" source="vector">…</evidence>`, with an instruction that nothing inside is a command.

## 10. Performance budget (targets to measure)

| Step | Expected cost |
|---|---|
| Query embedding (CPU) | tens of milliseconds |
| FAISS + BM25 | tens of milliseconds |
| Cross-encoder over 20 pairs (CPU) | well under a few seconds |
| Cypher run | milliseconds |
| LLM calls | dominate latency (seconds each) |

Typical question: 3 to 5 LLM calls (router, optional Cypher, grader, generate). With the loop and S4, up to 12.

## 11. Testing strategy

| Level | What | How |
|---|---|---|
| Unit | RRF, chunking, cleaning, Cypher guard, uncertainty signals, bootstrap statistics | Pure functions, fixed inputs |
| Component | Retrievers on fixture index and graph | Tiny fixtures checked into `tests/fixtures/` |
| Workflow | LangGraph termination, routing, failure fallbacks | Fake LLM returning scripted outputs |
| Integration (marked) | Real Neo4j, real LLM on 3 questions | Skipped by default |
| Evaluation sanity | Bootstrap code on synthetic data with known SD | Compare with closed form |

## 12. Extension ideas (after v1.0, not required)

- Replace TF-IDF keywords with KeyBERT.
- Add `CITES_EXTERNAL` nodes for co-citation analysis at scale.
- Section-aware chunking (abstract, methods, results).
- Learn the router with a small classifier on logged decisions.
- DPO-style generator tuning, if hardware becomes available (the paper's approach).
