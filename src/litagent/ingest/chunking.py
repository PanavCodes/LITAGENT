"""Recursive character chunking for scientific documents.

Per PRD.md §7.3 (FR-VEC-1) and ARCHITECTURE.md §4.1:
- Configurable chunk size (default 2,000 characters) and overlap (default 50 characters).
- Preserves paper ID, chunk index, and page number.
- Chunk c0000 is reserved for the paper abstract.
"""

from __future__ import annotations

from litagent.schemas import ChunkHit


def split_text_recursive(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursively split text using paragraph, sentence, and word boundaries."""
    if not text:
        return []

    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if len(text) <= chunk_size:
        return [text.strip()]

    # Pick the highest priority separator present in the text
    sep = separators[-1]
    for s in separators:
        if s in text:
            sep = s
            break

    splits = text.split(sep) if sep else list(text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for split in splits:
        split_len = len(split) + len(sep)
        if current_len + split_len > chunk_size and current_chunk:
            combined = sep.join(current_chunk).strip()
            if combined:
                chunks.append(combined)

            # Keep overlap by retaining the tail of current_chunk
            overlap_chunk: list[str] = []
            overlap_len = 0
            for item in reversed(current_chunk):
                if overlap_len + len(item) + len(sep) <= overlap:
                    overlap_chunk.insert(0, item)
                    overlap_len += len(item) + len(sep)
                else:
                    break
            current_chunk = overlap_chunk
            current_len = overlap_len

        current_chunk.append(split)
        current_len += split_len

    if current_chunk:
        combined = sep.join(current_chunk).strip()
        if combined:
            chunks.append(combined)

    return chunks


def chunk_paper(
    paper_id: str,
    abstract: str,
    full_text: str | None = None,
    chunk_size: int = 2000,
    overlap: int = 50,
) -> list[ChunkHit]:
    """Chunk paper into standardized ChunkHit objects. Abstract is chunk c0000."""
    chunks: list[ChunkHit] = []

    # Abstract is always chunk c0000
    if abstract:
        chunks.append(
            ChunkHit(
                chunk_id=f"{paper_id}_c0000",
                paper_id=paper_id,
                text=f"Abstract: {abstract.strip()}",
                scores={"is_abstract": 1.0},
            )
        )

    # Chunk full text if available
    if full_text:
        text_chunks = split_text_recursive(full_text, chunk_size=chunk_size, overlap=overlap)
        for i, text in enumerate(text_chunks, start=1):
            chunks.append(
                ChunkHit(
                    chunk_id=f"{paper_id}_c{i:04d}",
                    paper_id=paper_id,
                    text=text,
                    scores={"is_abstract": 0.0},
                )
            )

    return chunks
