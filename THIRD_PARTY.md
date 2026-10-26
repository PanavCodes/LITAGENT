# THIRD_PARTY.md — Third-Party Code and Attribution Record

All code imported from external libraries or adapted from third-party repositories is tracked here.

| Source Repository | Licence | What was borrowed / adapted | Destination in LitAgent | Notes |
|---|---|---|---|---|
| `PMRA-DSA/dsa-public` (`LitChat`) | MIT | 4 system prompt rules (cite inline, evidence-only, retrieve always, abstain) | `prompts/answer.md` | Rewritten in our words; no code copied directly |
| `neo4j/neo4j-graphrag-python` | Apache-2.0 | Text-to-Cypher prompt structure and schema rendering logic | `retrieval/text2cypher.py` | Wrapped with our custom read-only `cypher_guard.py` |
| `neo4j-labs/text2cypher` | CC0 1.0 | Few-shot query patterns | `prompts/cypher_fewshot.yaml` | Adapted to LitAgent's specific schema |
| `piyushagni5/langgraph-ai` | MIT | CRAG rewrite and grader node loop flow pattern | `agent/workflow.py`, `agent/grader.py` | Adapted to our TypedDict state |

## Direct Pip Dependencies (No Code Copied)
- `langgraph`, `langchain-core` (MIT)
- `sentence-transformers` (Apache-2.0)
- `faiss-cpu` (MIT)
- `rank-bm25` (Apache-2.0)
- `pymupdf` (AGPL-3.0 / Commercial; used as library via PyPI)
- `pyalex` (MIT)
- `arxiv` (MIT)
- `ragas` (Apache-2.0)
- `networkx` (BSD-3-Clause)
- `neo4j` (Apache-2.0)
- `gradio` (Apache-2.0)
