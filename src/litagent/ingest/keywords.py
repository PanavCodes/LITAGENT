"""Keyword extraction using TF-IDF for scientific papers.

Per PRD.md §7.1 (FR-ING-7):
Extracts 5 keywords per paper using TF-IDF (unigram and bigram, lower-cased).
"""

from __future__ import annotations

import logging
import re

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def extract_keywords_tfidf(
    documents: list[str],
    top_k: int = 5,
    min_df: int = 1,
) -> list[list[str]]:
    """Extract top_k TF-IDF keywords (unigrams + bigrams) for a corpus of documents."""
    if not documents:
        return []

    # Clean text lightly for TF-IDF
    cleaned_docs = [
        re.sub(r"[^\w\s]", " ", doc.lower()) if doc else "" for doc in documents
    ]

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
            min_df=min_df,
        )
        tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
        feature_names = vectorizer.get_feature_names_out()

        all_keywords: list[list[str]] = []
        for row_idx in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix.getrow(row_idx)
            # Get indices sorted by score descending
            sorted_indices = row.indices[row.data.argsort()[::-1]]
            doc_keywords: list[str] = []
            for idx in sorted_indices[:top_k]:
                doc_keywords.append(str(feature_names[idx]))
            all_keywords.append(doc_keywords)

        return all_keywords
    except Exception as e:
        logger.warning(f"TF-IDF keyword extraction error: {e}")
        # Fallback to simple word frequency
        fallback: list[list[str]] = []
        for doc in documents:
            words = [w for w in doc.lower().split() if len(w) > 4][:top_k]
            fallback.append(words)
        return fallback
