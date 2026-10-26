# DATA_SPEC.md — Data sources, schemas, and storage

## 1. Sources

| Source | What we take | Access | Etiquette |
|---|---|---|---|
| **arXiv API** (`arxiv` package) | Title, abstract, authors, categories, published date, arXiv ID, PDF URL, DOI if present | Free, no key | At most 1 request every 3 seconds; descriptive user agent; cache every response |
| **arXiv PDFs** | Full text of open-access preprints | Direct PDF link from the API result | Same delay; cache; never re-download |
| **OpenAlex** (`pyalex`) | `referenced_works`, `cited_by_count`, DOI, concepts | Free; set a contact email for the polite pool | Cache; batch lookups by DOI where possible |

No other sources. No scraping.

## 2. Seed query and selection

```
Query:      retrieval-augmented generation OR information retrieval OR dense retrieval OR reranking
Categories: cs.IR, cs.CL
Years:      2020 to 2026
Target:     600 papers (min 300, max 1000)
```

Selection rules, applied in order:

1. Has title, abstract and PDF URL.
2. Not a duplicate (arXiv ID, DOI, normalised title).
3. Has an OpenAlex match.
4. Has at least 5 referenced works in OpenAlex (a coverage rule; report how many papers this removes).
5. Optional: top 25% keyword similarity to the seed query (the paper's filter).

Record the number removed at each step in `corpus_manifest.json`.

## 3. Entity schema (graph)

### Node labels

| Label | Key property | Other properties |
|---|---|---|
| `Paper` | `arxiv_id` (unique) | `title`, `abstract`, `doi`, `year`, `published`, `cited_by_count`, `url`, `has_fulltext` |
| `Author` | `name_norm` (unique) | `name` (display form) |
| `Keyword` | `text` (unique) | |
| `Category` | `code` (unique) | e.g. `cs.IR` |
| `Year` | `value` (unique) | integer |

The paper's graph used Publication, Author, Database, Keyword, Publication Year and Citation nodes. We drop `Database` (only arXiv here) and model citations as `CITES` edges between papers.

### Relationship types (our own naming; the paper does not name edges)

| Relationship | Direction | Meaning |
|---|---|---|
| `WROTE` | `(Author)-[:WROTE]->(Paper)` | Authorship; property `position` (author order) |
| `CITES` | `(Paper)-[:CITES]->(Paper)` | The first paper's reference list includes the second; **both are in the corpus** |
| `HAS_KEYWORD` | `(Paper)-[:HAS_KEYWORD]->(Keyword)` | TF-IDF keyword; property `rank` |
| `IN_CATEGORY` | `(Paper)-[:IN_CATEGORY]->(Category)` | arXiv category; property `primary` (boolean) |
| `PUBLISHED_IN` | `(Paper)-[:PUBLISHED_IN]->(Year)` | Publication year |
| `CITES_EXTERNAL` *(optional)* | `(Paper)-[:CITES_EXTERNAL]->(ExternalWork)` | Reference to a work outside the corpus, used for co-citation |

### Cypher DDL

```cypher
CREATE CONSTRAINT paper_id    IF NOT EXISTS FOR (p:Paper)    REQUIRE p.arxiv_id  IS UNIQUE;
CREATE CONSTRAINT author_id   IF NOT EXISTS FOR (a:Author)   REQUIRE a.name_norm IS UNIQUE;
CREATE CONSTRAINT keyword_id  IF NOT EXISTS FOR (k:Keyword)  REQUIRE k.text      IS UNIQUE;
CREATE CONSTRAINT category_id IF NOT EXISTS FOR (c:Category) REQUIRE c.code      IS UNIQUE;
CREATE CONSTRAINT year_id     IF NOT EXISTS FOR (y:Year)     REQUIRE y.value     IS UNIQUE;
CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title);
```

### Idempotent loading pattern

```cypher
UNWIND $rows AS r
MERGE (p:Paper {arxiv_id: r.arxiv_id})
  SET p.title = r.title, p.abstract = r.abstract, p.year = r.year, p.doi = r.doi,
      p.cited_by_count = r.cited_by_count, p.url = r.url, p.has_fulltext = r.has_fulltext
```

Load nodes first, then edges, each in batches of about 1,000 rows.

### Example queries the system must support

```cypher
// Papers citing a given paper
MATCH (p:Paper)-[:CITES]->(q:Paper)
WHERE toLower(q.title) CONTAINS toLower($title)
RETURN p.title AS citing_paper, p.year AS year LIMIT 50;

// Most prolific authors
MATCH (a:Author)-[:WROTE]->(p:Paper)
RETURN a.name AS author, count(p) AS papers ORDER BY papers DESC LIMIT 5;

// Count by year and keyword
MATCH (p:Paper)-[:PUBLISHED_IN]->(:Year {value: $year}), (p)-[:HAS_KEYWORD]->(:Keyword {text: $kw})
RETURN count(p) AS n;

// Papers cited by both A and B
MATCH (a:Paper {arxiv_id: $a})-[:CITES]->(x:Paper)<-[:CITES]-(b:Paper {arxiv_id: $b})
RETURN x.title AS shared_reference LIMIT 50;

// Shortest citation path
MATCH path = shortestPath((a:Paper {arxiv_id: $a})-[:CITES*..4]->(b:Paper {arxiv_id: $b}))
RETURN [n IN nodes(path) | n.title] AS titles;
```

## 4. Chunk schema (vector and sparse)

```json
{
  "chunk_id": "2005.11401_c0007",
  "arxiv_id": "2005.11401",
  "title": "…",
  "page": 4,
  "section": null,
  "text": "…",
  "char_start": 12000,
  "char_end": 14000
}
```

| Parameter | Default | Notes |
|---|---|---|
| Splitter | Recursive character splitter (paragraph, sentence, word) | |
| `size` | 2,000 characters | Matches the paper's about 2,024; also test 1,000 |
| `overlap` | 50 characters | Paper's setting |
| Minimum chunk length | 200 characters | Drop shorter, they are mostly noise |
| Excluded | Headers, footers, page numbers, reference list | Reference list kept separately for citation fallback |
| Always included | Abstract as chunk `c0000` | Guarantees every paper is retrievable even when the PDF fails |

Embedding: `all-MiniLM-L6-v2`, 384 dimensions, L2-normalised; FAISS `IndexFlatIP` (inner product on normalised vectors, equivalent to cosine). The paper used L2 distance; on normalised vectors the rankings are equivalent.

Note: MiniLM-L6 truncates input at about 256 word pieces by default, so a 2,000-character chunk is partly cut off at embedding time. Check `model.max_seq_length`. This is a known limitation worth testing (1,000-character chunks may do better), and worth a sentence in the report.

## 5. Storage layout

```
data/
├── raw/
│   ├── arxiv/            # cached API responses (Atom/JSON), one file per page
│   ├── openalex/         # cached works JSON
│   └── pdfs/             # <arxiv_id>.pdf
├── processed/
│   ├── papers.jsonl
│   ├── authors.jsonl
│   ├── citations.jsonl           # {"src": "...", "dst": "..."}
│   ├── keywords.jsonl
│   ├── chunks.jsonl
│   └── corpus_manifest.json
├── index/
│   ├── faiss.index
│   ├── faiss_ids.json
│   └── bm25.pkl
├── eval/
│   ├── graph_questions.jsonl
│   ├── vector_questions.jsonl
│   ├── hybrid_questions.jsonl
│   ├── oos_questions.jsonl
│   ├── router_labels.jsonl
│   └── split.json
├── cache/llm/            # response cache
└── smoke/                # PDFs from dsa-public for parser tests (local only)
```

### `corpus_manifest.json` fields

```json
{
  "built_at": "<ISO timestamp>",
  "config_hash": "<sha256>",
  "seed_query": "...",
  "counts": {
    "fetched": 0, "after_cleaning": 0, "with_openalex": 0,
    "with_5plus_refs": 0, "with_pdf": 0, "with_fulltext_parsed": 0, "final": 0
  },
  "dropped": {"missing_fields": 0, "duplicates": 0, "no_openalex": 0, "few_refs": 0, "pdf_failed": 0},
  "graph": {"papers": 0, "authors": 0, "keywords": 0, "cites_edges": 0,
            "papers_with_in_corpus_cites_pct": 0.0, "avg_out_degree": 0.0},
  "chunks": {"total": 0, "avg_chars": 0.0}
}
```

(All zeros are placeholders; the build script writes real values.)

## 6. Data quality checks (run in Phase 1 and Phase 2)

| Check | Pass condition |
|---|---|
| No duplicate arXiv IDs | 0 duplicates |
| Every chunk's `arxiv_id` exists in papers | 100% |
| Every CITES edge endpoints exist | 100% |
| Share of papers with at least one in-corpus CITES edge | at least 60% (see Phase 1 gate) |
| Empty or tiny chunks | under 2% |
| Author duplicates after normalisation (sample 50 names) | under 5% by eye |
| PDF text readable (10 random papers by eye) | 10 of 10 readable |

## 7. Ethics and licensing

- arXiv metadata is openly accessible; PDFs are used for local indexing only. Do not redistribute PDFs; cite arXiv IDs instead.
- Authors' names are public bibliographic data; no personal data beyond that is collected.
- Respect arXiv's API terms of use and OpenAlex's usage guidance.
