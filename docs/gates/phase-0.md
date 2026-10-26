# Gate 0 Checklist and Observed Evidence

## Checklist
- [x] `make lint` (`ruff check src tests`) passes with 0 errors
- [x] `pytest tests/unit` passes offline on fake LLM, config, and caching
- [x] Virtual environment running Python 3.12 with all dependencies locked
- [x] Neo4j reachable or NetworkX fallback recorded in `docs/decisions.md`
- [x] Q1 to Q6 answered in `docs/decisions.md`
- [x] All 7 repos referenced in `third_party/SOURCES.txt` with attribution in `THIRD_PARTY.md`

## Evidence
- Ruff output:
  `Found 4 errors (4 fixed, 0 remaining).`
- Pytest output:
  `tests\unit\test_phase0.py .... [100%]`
  `4 passed in 0.52s`
- Decisions recorded: `docs/decisions.md`
- Gate 0 Verdict: **PASS**
