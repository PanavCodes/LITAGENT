# Phase 4 Example Runs — 12 Reference Questions

Executed per PRD.md §4.2 & PHASES.md Task 4.11.

### Question: Which papers in the corpus cite Dense Passage Retrieval?
- **Expected Route:** `graph`
- **Observed Route:** `graph`
- **Generated Cypher:** `{"route": "graph", "rationale": "Citation link search"}`
- **Answer:** This information was not found in the corpus.
- **Citations:** None

### Question: Who are the five most prolific authors in the corpus?
- **Expected Route:** `graph`
- **Observed Route:** `graph`
- **Generated Cypher:** `{"route": "graph", "rationale": "Author aggregation"}`
- **Answer:** This information was not found in the corpus.
- **Citations:** None

### Question: How many papers were published in 2024 with the keyword reranking?
- **Expected Route:** `graph`
- **Observed Route:** `graph`
- **Generated Cypher:** `{"route": "graph", "rationale": "Year count aggregation"}`
- **Answer:** This information was not found in the corpus.
- **Citations:** None

### Question: Which papers are cited by both paper 2004.04906 and paper 2005.11401?
- **Expected Route:** `graph`
- **Observed Route:** `graph`
- **Generated Cypher:** `{"route": "graph", "rationale": "Co-citation query"}`
- **Answer:** This information was not found in the corpus.
- **Citations:** None

### Question: What is the citation path from paper 2004.04906 to paper 2401.15884?
- **Expected Route:** `graph`
- **Observed Route:** `graph`
- **Generated Cypher:** `{"route": "graph", "rationale": "Graph path traversal"}`
- **Answer:** This information was not found in the corpus.
- **Citations:** None

### Question: What retrieval metric does DPR report as its main result?
- **Expected Route:** `vector`
- **Observed Route:** `vector`
- **Generated Cypher:** `N/A`
- **Answer:** {"route": "vector", "rationale": "Content detail lookup"}
- **Citations:** [P1] Paper 2305.06983, [P2] Paper 2401.10034, [P3] Paper 2401.10016, [P4] Paper 2401.10026, [P5] Paper 2401.15884

### Question: How does paper handle long documents in dense retrieval?
- **Expected Route:** `vector`
- **Observed Route:** `vector`
- **Generated Cypher:** `N/A`
- **Answer:** {"route": "vector", "rationale": "Content method detail"}
- **Citations:** [P1] Paper 2401.10030, [P2] Paper 2401.10020, [P3] Paper 2212.10496, [P4] Paper 2004.04906, [P5] Paper 2401.10010

### Question: What limitations do the authors of RAG mention?
- **Expected Route:** `vector`
- **Observed Route:** `vector`
- **Generated Cypher:** `N/A`
- **Answer:** {"route": "vector", "rationale": "Content limitation detail"}
- **Citations:** [P1] Paper 2310.11511, [P2] Paper 2401.10015, [P3] Paper 2401.10025, [P4] Paper 2312.10997, [P5] Paper 2004.12832

### Question: Among papers citing DPR, which ones propose a reranking method, and what do they say about latency?
- **Expected Route:** `both`
- **Observed Route:** `both`
- **Generated Cypher:** `{"route": "both", "rationale": "Citation plus content search"}`
- **Answer:** {"route": "both", "rationale": "Citation plus content search"}
- **Citations:** [P1] Paper 2407.02485, [P2] Paper 2401.10028, [P3] Paper 2401.10018, [P4] Paper 2401.10012, [P5] Paper 2305.06983

### Question: Summarise the evaluation approach used by the three most-cited papers about hybrid search.
- **Expected Route:** `both`
- **Observed Route:** `both`
- **Generated Cypher:** `{"route": "both", "rationale": "Citation ranking plus content"}`
- **Answer:** {"route": "both", "rationale": "Citation ranking plus content"}
- **Citations:** [P1] Paper 2401.10022, [P2] Paper 2305.06983, [P3] Paper 2004.12832, [P4] Paper 2401.10032, [P5] Paper 2401.10012

### Question: What is the weather in Mumbai?
- **Expected Route:** `none`
- **Observed Route:** `none`
- **Generated Cypher:** `N/A`
- **Answer:** This question is outside the scope of the scientific literature review corpus. LitAgent only answers questions related to retrieval-augmented generation and information retrieval.
- **Citations:** None

### Question: Who won the 2026 football World Cup?
- **Expected Route:** `none`
- **Observed Route:** `none`
- **Generated Cypher:** `N/A`
- **Answer:** This question is outside the scope of the scientific literature review corpus. LitAgent only answers questions related to retrieval-augmented generation and information retrieval.
- **Citations:** None

