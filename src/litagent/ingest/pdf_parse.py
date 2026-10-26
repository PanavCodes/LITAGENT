"""PDF parsing for scientific preprints using PyMuPDF.

Per PRD.md §7.1 (FR-ING-5, FR-ING-6):
- Two-column layout block sorting.
- Strips running headers, footers, and page numbers.
- Splits references section from main body text for chunking.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def parse_pdf(
    pdf_path: str | Path,
    header_margin: float = 50.0,
    footer_margin: float = 50.0,
) -> dict[str, Any]:
    """Parse a scientific PDF file, returning body text and references separately."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found at {path}")

    doc = fitz.open(str(path))
    body_paragraphs: list[str] = []
    reference_paragraphs: list[str] = []
    in_references = False

    ref_heading_pattern = re.compile(
        r"^(references|bibliography|works cited)\s*$", re.IGNORECASE
    )

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            rect = page.rect
            page_height = rect.height

            # Extract text blocks: (x0, y0, x1, y1, text, block_no, block_type)
            blocks = page.get_text("blocks")

            # Sort blocks top-to-bottom, left-to-right (handles multi-column layout)
            # Sort by horizontal column first (x0 < rect.width / 2), then y0
            sorted_blocks = sorted(
                blocks,
                key=lambda b: (0 if b[0] < rect.width * 0.45 else 1, b[1]),
            )

            for b in sorted_blocks:
                _x0, y0, _x1, y1, text, _block_no, b_type = b
                if b_type != 0:  # 0 is text block
                    continue

                # Strip headers and footers
                if y0 < header_margin or y1 > page_height - footer_margin:
                    continue

                clean_b_text = text.strip()
                if not clean_b_text:
                    continue

                # Check for reference section transition
                first_line = clean_b_text.splitlines()[0].strip()
                if ref_heading_pattern.match(first_line):
                    in_references = True

                if in_references:
                    reference_paragraphs.append(clean_b_text)
                else:
                    body_paragraphs.append(clean_b_text)

    finally:
        doc.close()

    return {
        "body_text": "\n\n".join(body_paragraphs),
        "references_text": "\n\n".join(reference_paragraphs),
        "total_pages": len(doc),
        "has_references": bool(reference_paragraphs),
    }
