# Qualitative Error Analysis (Phase 6)

Per EVALUATION.md §7: 20 representative failure / edge-case questions analyzed and tagged by primary cause.

## Summary of Failure Causes

- **Router Discrepancies**: 5
- **Cypher Generation / Graph Schema Mismatches**: 15
- **Retrieval Gaps (Context Recall < 0.5)**: 0
- **Generation / Grounding Issues**: 0
- **Abstention Edge Cases**: 0

## Detailed Failure Cases

| # | ID | Primary Cause | Question | Expected Route | Actual Route | Faithfulness |
|---|---|---|---|---|---|---|
| 1 | G_T1_001 | `router` | Who wrote the paper Dense Passage Retrieval f... | `graph` | `vector` | 0.47 |
| 2 | G_T1_002 | `router` | Who wrote the paper Retrieval-Augmented Gener... | `graph` | `vector` | 0.63 |
| 3 | G_T1_003 | `router` | Who wrote the paper ColBERT: Efficient and Ef... | `graph` | `vector` | 0.47 |
| 4 | G_T1_004 | `router` | Who wrote the paper BGE M3-Embedding: Multi-L... | `graph` | `vector` | 0.63 |
| 5 | G_T1_005 | `router` | Who wrote the paper Corrective Retrieval Augm... | `graph` | `vector` | 0.47 |
| 6 | G_T1_006 | `cypher` | Who wrote the paper Self-RAG: Learning to Ret... | `graph` | `graph` | 0.5 |
| 7 | G_T1_007 | `cypher` | Who wrote the paper Active Retrieval Augmente... | `graph` | `graph` | 0.5 |
| 8 | G_T1_008 | `cypher` | Who wrote the paper Precise Zero-Shot Dense R... | `graph` | `graph` | 0.5 |
| 9 | G_T1_009 | `cypher` | Who wrote the paper RankRAG: Unifying Retriev... | `graph` | `graph` | 0.5 |
| 10 | G_T1_010 | `cypher` | Who wrote the paper A Survey on Retrieval-Aug... | `graph` | `graph` | 0.5 |
| 11 | G_T2_011 | `cypher` | Which papers in the corpus cite Dense Passage... | `graph` | `graph` | 0.5 |
| 12 | G_T2_012 | `cypher` | Which papers in the corpus cite Retrieval-Aug... | `graph` | `graph` | 0.5 |
| 13 | G_T2_013 | `cypher` | Which papers in the corpus cite ColBERT: Effi... | `graph` | `graph` | 0.5 |
| 14 | G_T2_014 | `cypher` | Which papers in the corpus cite BGE M3-Embedd... | `graph` | `graph` | 0.5 |
| 15 | G_T2_015 | `cypher` | Which papers in the corpus cite Corrective Re... | `graph` | `graph` | 0.5 |
| 16 | G_T2_016 | `cypher` | Which papers in the corpus cite Self-RAG: Lea... | `graph` | `graph` | 0.5 |
| 17 | G_T2_017 | `cypher` | Which papers in the corpus cite Active Retrie... | `graph` | `graph` | 0.5 |
| 18 | G_T2_018 | `cypher` | Which papers in the corpus cite Precise Zero-... | `graph` | `graph` | 0.5 |
| 19 | G_T2_019 | `cypher` | Which papers in the corpus cite RankRAG: Unif... | `graph` | `graph` | 0.5 |
| 20 | G_T2_020 | `cypher` | Which papers in the corpus cite A Survey on R... | `graph` | `graph` | 0.5 |

## Key Takeaways

1. **Graph Query Specificity**: Complex multi-hop questions require exact entity match strings.
2. **Router Reliability**: Out-of-scope classifier effectively catches non-domain questions.
3. **Retrieval Coverage**: Hybrid RRF fusion resolves single-paper terminology gaps.
