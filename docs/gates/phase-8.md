# Gate 8 Verification: Report, Deviations, and Release

Executed on: 2026-10-26
Platform: Windows (Python 3.12.13)
Branch: phase-0-setup
Release Tag: v1.0

## Gate Checklist

- [x] Comprehensive 13-section report written in `docs/report.md` following `REPORT_OUTLINE.md`
- [x] Every table and score is regenerated from stored artifacts via `make eval` / `python -m litagent.eval.report`
- [x] Section 6 explicitly details all paper deviations (API LLM, local cross-encoder, arXiv/OpenAlex data, no fine-tuning)
- [x] Qualitative error analysis documented across 20 representative failures in `docs/error_analysis.md`
- [x] All 8 test suites pass offline (`28 passed in 30.57s`)
- [x] `ruff check src tests scripts` passes with zero warnings
- [x] `mypy src` passes with zero type errors across 46 source files
- [x] Git repository committed and tagged `v1.0`

## Summary of Empirical Achievements

| Metric / Check | Measured Target | Verified Result | Status |
|---|---|---|---|
| Vector Chunks Indexed | >= 30 | 35 | PASS |
| Graph Nodes & Edges | >= 100 nodes, >= 300 edges | 175 nodes, 455 edges | PASS |
| Citation Coverage | >= 80% | 100.0% | PASS |
| AST Cypher Security | 100% write-rejection | 100% (AST guard + read-only transaction) | PASS |
| CRAG Loop Termination | Invariant bound <= 2 | Proven across 100 random runs | PASS |
| Full Agent Faithfulness | Baseline > single vector | 0.484 ± 0.003 vs 0.130 ± 0.034 | PASS |
| Full Agent Answer Relevancy | Baseline > single vector | 0.494 ± 0.006 vs 0.109 ± 0.029 | PASS |
| Test AUROC (Fitted Uncertainty) | >= 0.70 | 0.7271 | PASS |
| Test ECE (Expected Calibration Error) | <= 0.15 | 0.1097 | PASS |
| Unit Test Pass Rate | 100% offline | 28 / 28 passed | PASS |
| Static Lint & Type Analysis | 0 errors | 0 ruff errors, 0 mypy errors | PASS |
