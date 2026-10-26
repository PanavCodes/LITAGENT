# REPORT_OUTLINE.md — Structure of the final report

Target length: 8 to 12 pages plus appendices. Every number must come from `results/`.

| # | Section | Content | Source files |
|---|---|---|---|
| 1 | Abstract | Problem, approach, main numbers with ± SD, one-sentence limitation | `results/ablation_table.csv` |
| 2 | Introduction | Literature review is slow; why hybrid retrieval; contributions (router, correction loop, uncertainty score, bootstrapped evaluation) | |
| 3 | Related work | Reference paper (arXiv 2508.05660); LitChat paper (arXiv 2602.17856); CRAG; RAGAS; text-to-Cypher | |
| 4 | Data | arXiv + OpenAlex, selection rules, manifest numbers, graph statistics, coverage | `corpus_manifest.json`, `docs/graph_stats.md` |
| 5 | System | Architecture figure, router, vector pipeline (BM25 + dense + RRF + rerank), text-to-Cypher with guard, grader and rewrite loop, uncertainty signals | `ARCHITECTURE.md` |
| 6 | Deviations from the reference paper | Table from PRD §2.3, stated plainly | `PRD.md` |
| 7 | Experimental setup | Question sets, splits, hand-verification rates, metrics, bootstrap protocol, models and versions | `EVALUATION.md`, `data/eval/` |
| 8 | Results | Ablation table (mean ± SD); per-type breakdown; retrieval ablation; chunk-size ablation; latency and cost | `results/` |
| 9 | Uncertainty analysis | AUROC, ECE, reliability diagram, selective-answering curve, signal ablation | `results/calibration.json`, figures |
| 10 | Error analysis | Failure tags with counts; 2 or 3 worked examples | `docs/error_analysis.md` |
| 11 | Discussion | What worked, what did not, why; where the router helps and where it hurts | |
| 12 | Limitations and threats to validity | From `EVALUATION.md` §10; single subfield; generated questions; judge bias; author-only verification | |
| 13 | Conclusion and future work | Short; section-aware chunking, learned router, larger corpus, fine-tuning as in the paper | |
| A | Appendix: prompts | Router, Cypher, grader, answer prompts | `prompts/` |
| B | Appendix: example traces | One per route | `results/traces/` |
| C | Appendix: reproducibility | Commands, versions, seeds, config hash, commit SHA | |
| D | Appendix: third-party code | Copy of `THIRD_PARTY.md` | |

## Rules for the writers

1. No number without a file behind it.
2. If the router does not beat the best single strategy, say so and explain; that is still a valid result.
3. State every deviation from the paper in section 6.
4. Report SDs everywhere; say "no clear difference" when intervals overlap.
5. Include at least one failure case per route.
