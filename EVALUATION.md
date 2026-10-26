# EVALUATION.md — Evaluation protocol for LitAgent

The evaluation follows the reference paper's protocol where it is specified (four RAGAS metrics, bootstrapping with 12 resamples of 20 questions, agentic vs non-agentic comparison) and extends it where we have budget (larger question set, more ablations, calibration of the uncertainty score).

## 1. Research questions

| ID | Question | Evidence |
|---|---|---|
| RQ1 | Does an agentic router beat vector-only and graph-only on a mixed question set? | Ablation table, per-type breakdown |
| RQ2 | Does the grade-and-rewrite loop improve answer quality, and at what cost? | `router` vs `router_crag`: metrics, latency, LLM calls |
| RQ3 | Does the reranker, and does RRF, improve retrieval over BM25 or dense alone? | Retrieval ablation: Recall@k, MRR |
| RQ4 | Is the uncertainty score predictive of wrong or unfaithful answers? | AUROC, ECE, reliability diagram |
| RQ5 | How reliable is text-to-Cypher on our schema? | Valid-Cypher rate, set-F1 vs scripted gold |

## 2. Question sets

### 2.1 Composition

| Set | Count (default) | How gold is produced | Verification |
|---|---|---|---|
| Graph questions | 50 | Template Cypher executed on the graph; answer = result set | Automatic; hand-check 10% |
| Vector questions | 50 | LLM generates question and reference answer from a random chunk; source chunk stored as gold context | Hand-check at least 25%, stratified |
| Hybrid questions | 10 to 15 | Hand-written; gold from graph query plus chunk | Fully hand-checked |
| Out-of-scope | 10 to 15 | Hand-written (weather, sports, unrelated science, questions about papers not in corpus) | Gold = abstention |

Total about 120 to 130 questions. The paper used 40, so confidence intervals will be tighter.

### 2.2 Graph question taxonomy

The paper adapts five question types from PolyG but does not list them in the text we read, so this taxonomy is our own.

| Type | Template example | Gold from |
|---|---|---|
| T1 Lookup | "Who wrote *{title}*?" | `WROTE` edges |
| T2 One-hop citation | "Which papers cite *{title}*?" | `CITES` incoming |
| T3 Aggregation | "How many papers were published in {year}?" / "Top 5 authors by paper count" | `count`, `ORDER BY` |
| T4 Co-citation | "Which papers are cited by both *{A}* and *{B}*?" | shared `CITES` targets |
| T5 Path or multi-hop | "Is there a citation path from *{A}* to *{B}* within 3 hops?" | `shortestPath` |

About 10 questions per type. Pick entities (titles, years) by seeded random sampling, **requiring non-empty gold** so a question is never trivially "none".

### 2.3 Vector question types

| Type | Example |
|---|---|
| V1 Single-paper factual | "What dataset does *{title}* use for evaluation?" |
| V2 Single-paper method | "How does *{title}* handle long documents?" |
| V3 Multi-paper comparison | "How do *{A}* and *{B}* differ in their reranking approach?" |

Generation prompt rules: question must be answerable from the chunk alone, must not mention "the chunk" or "the text", must be self-contained, and must name the paper when ambiguous. Reject questions that contain document-specific pronouns.

### 2.4 Splits

- Dev 30%, test 70%, stratified by question type, fixed seed, saved in `data/eval/split.json`.
- **Dev** is used for: router few-shots, grader threshold, uncertainty weights, abstain threshold, prompt edits.
- **Test** is used once per final configuration. Do not tune on it. Record any exception in the report.

## 3. Metrics

### 3.1 RAGAS (LLM-judged)

| Metric | What it measures | Needs |
|---|---|---|
| Faithfulness | Are the claims in the answer supported by the retrieved context? | question, answer, contexts |
| Answer Relevancy | Does the answer address the question? | question, answer |
| Context Precision | Are relevant contexts ranked high among retrieved ones? | question, contexts, reference |
| Context Recall | Does the retrieved context cover the reference answer? | contexts, reference |

For the graph route the "contexts" are the rendered rows. Note that RAGAS metrics were designed for text contexts; report graph-route RAGAS numbers **alongside** the scripted set-F1, because set-F1 is the more trustworthy signal there.

Implementation notes:
- Check the installed RAGAS version's API before coding; the library changes fast.
- Use a **judge model different from the answer model** where possible (reduces self-preference bias).
- Judge temperature 0; cache all judge calls.
- Report the number of judge failures (NaN) and how they were handled (exclude and count).

### 3.2 Scripted metrics

| Metric | Applies to | Definition |
|---|---|---|
| Set precision, recall, F1 | Graph questions | Compare the set of entities in the agent's answer (or in its executed Cypher result) with the gold set. Report both "answer-level" (parsed from text) and "query-level" (rows from the executed Cypher) |
| Valid Cypher rate | Graph route | Share of generated queries that pass the guard and run without error (before and after retry) |
| Router accuracy | Router | Agreement with hand labels on the 40-question labelled set |
| Abstention precision/recall | Out-of-scope | Out-of-scope questions answered with abstention; in-scope questions wrongly abstained |
| Recall@k, MRR | Retrieval ablation | Gold chunk or gold paper in the top k |
| Latency, LLM calls, tokens | All | From traces |

### 3.3 Correctness label for calibration

For RQ4 we need a binary "answer was good" label per question:

- Graph questions: set-F1 ≥ 0.8.
- Vector questions: Faithfulness ≥ 0.7 **and** Answer Relevancy ≥ 0.7 (thresholds fixed on dev, written down, never changed after seeing test).
- Out-of-scope: good = abstained.

## 4. Bootstrap protocol

### 4.1 Paper protocol (reproduction)

- 12 resamples; each resample draws 20 questions, 10 per retriever type (vector and graph), as described in the paper.
- For each resample compute each metric; report the mean and standard deviation across the 12 resamples.
- 95% margin of error with the t-distribution, α = 0.05, `t_{0.975, 11} × SD / √12`.
- Sampling is with or without replacement per the paper's text, which we did not find stated precisely; **we sample without replacement within a resample and with replacement across resamples**, and say so in the report.

### 4.2 Extended protocol (ours)

- 1,000 resamples of the full test set (with replacement), reporting mean, SD and a 95% percentile interval.
- Paired bootstrap for comparing two configurations: resample question indices once, compute the metric difference for both configs on the same indices, report the mean difference and the share of resamples where config A > config B.

### 4.3 Reporting rules

- Always show **mean ± SD** (and n).
- Never report a single run without an interval.
- If two configs' intervals overlap, do not claim one is better; say "no clear difference".

## 5. Ablation matrix

| Config ID | Route | Retrieval | Loop | Uncertainty | Purpose |
|---|---|---|---|---|---|
| `vector_only` | forced vector | BM25 + dense + RRF + rerank | no | yes | Baseline A |
| `graph_only` | forced graph | text-to-Cypher | no | yes | Baseline B |
| `joint_baseline` | both always, no router | both, merged | no | yes | The paper's **non-agentic baseline** analogue |
| `router` | agent chooses | per route | no | yes | Agentic |
| `router_crag` | agent chooses | per route | grade → rewrite (≤ 2) | yes | Our extension |

Retrieval sub-ablation (vector questions only):

| Config | BM25 | Dense | RRF | Rerank |
|---|---|---|---|---|
| `r_bm25` | ✓ | | | |
| `r_dense` | | ✓ | | |
| `r_rrf` | ✓ | ✓ | ✓ | |
| `r_rrf_rerank` | ✓ | ✓ | ✓ | ✓ |

Chunk-size sub-ablation: 1,000 vs 2,000 characters on `r_rrf_rerank`.

### Expected table shape

| Config | Faithfulness | Ans. Relevancy | Ctx Precision | Ctx Recall | Graph set-F1 | Latency (s) | LLM calls |
|---|---|---|---|---|---|---|---|
| `vector_only` | m ± sd | | | | | | |
| `graph_only` | | | | | | | |
| `joint_baseline` | | | | | | | |
| `router` | | | | | | | |
| `router_crag` | | | | | | | |

(Empty by design: fill only with measured values.)

## 6. Uncertainty calibration

1. Run the best config on dev and test; record signals S1 to S5 and the correctness label.
2. Fit logistic regression on **dev only**: label ~ signals. Save weights.
3. On **test**, compute:
   - AUROC of uncertainty score vs "answer was wrong".
   - ECE with 10 equal-width bins on predicted correctness probability.
   - Reliability diagram.
   - Selective-answering curve: accuracy of answered questions as a function of abstention rate.
4. Compare the fixed equal-weight score with the fitted score.
5. Ablate signals one at a time (drop S1, drop S2, and so on) and report AUROC change.
6. Be honest: if AUROC is near 0.5, say the score is not predictive and discuss why.

## 7. Error analysis

Read at least 20 failed answers. Tag each failure with one primary cause:

| Tag | Meaning |
|---|---|
| `router` | Wrong route chosen |
| `cypher` | Invalid, wrong or empty query |
| `retrieval` | Right route, relevant evidence not retrieved |
| `generation` | Evidence was fine, answer wrong or unsupported |
| `grader` | Grader approved weak evidence or rejected good evidence |
| `gold` | The gold answer was wrong |
| `judge` | RAGAS verdict looks wrong |

Report counts and two or three illustrative examples (question, route, evidence, answer, why it failed).

## 8. Reproducibility checklist

- [ ] Seeds fixed and recorded
- [ ] Every LLM call cached, so reruns need no new API calls
- [ ] Config hash and git commit SHA written into every results file
- [ ] `make eval` regenerates all tables and figures
- [ ] Question sets, splits, and hand-verification records committed (small files)
- [ ] Versions of `ragas`, `langgraph`, models and embedding model recorded

## 9. Cost control

| Technique | Effect |
|---|---|
| `--max-questions N` | Cheap smoke runs |
| LLM response cache | Free reruns |
| Run S4 self-consistency only in the calibration experiment | Avoids tripling cost everywhere |
| Cheaper answer model, stronger judge on fewer samples | Balance quality and cost |
| Evaluate the five configs on **the same** cached retrieval where possible | Avoids duplicate calls |

Record measured token usage and put it in the README so others can budget.

## 10. Threats to validity (write these in the report)

- LLM-generated questions may favour what the generator model finds easy.
- The judge LLM can be biased or inconsistent; mitigated with a different judge model and bootstrapped intervals.
- A single subfield and corpus limits generality.
- Graph questions come from templates, so they are cleaner than real user questions.
- Small hand-verified fraction; verification done by the authors, not independent annotators.
- RAGAS metrics were designed for text contexts and are less meaningful on graph rows.
