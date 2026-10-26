# Gate 5 Checklist and Observed Evidence

## Checklist
- [x] CRAG evidence sufficiency grader outputs score [0.0, 1.0] and missing facts
- [x] Query rewriter reformulates query using missing info and increments iteration
- [x] AGENTS.md §4 Termination Invariant holds: workflow strictly terminates at `max_iterations = 2` even when the grader always rejects evidence
- [x] Uncertainty report attaches S1 (margin), S2 (grader), S5 (graph status) signals to every response
- [x] Abstain threshold flags low-confidence answers with warnings
- [x] `docs/uncertainty_design.md` completed and committed
- [x] Unit tests pass offline (`tests/unit/test_phase5.py`)

## Observed Test Output
```
tests\unit\test_phase5.py ...                                            [100%]
3 passed in 19.80s
```

- Gate 5 Verdict: **PASS**
