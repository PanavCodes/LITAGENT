"""Cypher query security guard and sanitizer.

Per AGENTS.md §5, PRD.md §7.4 (FR-T2C-2, 4) & Trail of Bits/Strix standards:
1. Strips comments and string literals before keyword checks.
2. Rejects write clauses: CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP, LOAD CSV, CALL dbms.*.
3. Enforces label validation against known schema.
4. Injects mandatory LIMIT clause if absent.
"""

from __future__ import annotations

import re


class CypherSecurityError(ValueError):
    """Raised when generated Cypher violates read-only security constraints."""


DISALLOWED_CLAUSES = [
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bDETACH\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bDROP\b",
    r"\bLOAD\s+CSV\b",
    r"\bCALL\s+dbms\b",
    r"\bCALL\s+apoc\b",
    r"\bALTER\b",
]

ALLOWED_LABELS = {"Paper", "Author", "Keyword", "Category", "Year"}


def strip_comments_and_literals(cypher: str) -> str:
    """Remove single-line comments, block comments, and string literals."""
    # Remove single-line comments // ...
    text = re.sub(r"//.*$", "", cypher, flags=re.MULTILINE)
    # Remove multi-line comments /* ... */
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Remove double-quoted strings "..."
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    # Remove single-quoted strings '...'
    text = re.sub(r"'(?:\\.|[^'\\])*'", "''", text)
    return text


def validate_and_sanitize_cypher(
    cypher: str,
    allowed_labels: set[str] | None = None,
    default_limit: int = 50,
) -> str:
    """Validate query for read-only safety, label correctness, and inject LIMIT."""
    clean_cypher = cypher.strip()
    # Strip markdown code fences if present
    if clean_cypher.startswith("```"):
        clean_cypher = re.sub(r"^```[a-zA-Z]*\n", "", clean_cypher)
        clean_cypher = re.sub(r"\n```$", "", clean_cypher).strip()

    if not clean_cypher:
        raise CypherSecurityError("Empty Cypher query.")

    # Check for write keywords on sanitized query (excluding literals and comments)
    tokens_only = strip_comments_and_literals(clean_cypher)

    for pattern in DISALLOWED_CLAUSES:
        if re.search(pattern, tokens_only, re.IGNORECASE):
            matched = re.search(pattern, tokens_only, re.IGNORECASE)
            clause = matched.group(0) if matched else "WRITE_CLAUSE"
            raise CypherSecurityError(
                f"Disallowed write or administrative clause detected: '{clause}'. Queries must be strictly read-only."
            )

    # Must contain MATCH or RETURN
    if not re.search(r"\b(MATCH|RETURN)\b", tokens_only, re.IGNORECASE):
        raise CypherSecurityError("Cypher query must contain a MATCH or RETURN clause.")

    # Label validation
    valid_labels = allowed_labels or ALLOWED_LABELS
    found_labels = re.findall(r":([a-zA-Z0-9_]+)", tokens_only)
    for lbl in found_labels:
        # Ignore relationship types which appear in brackets like [:CITES]
        if lbl in valid_labels or lbl in {"CITES", "WROTE", "HAS_KEYWORD", "IN_CATEGORY", "PUBLISHED_IN"}:
            continue
        # Also ignore variable aliases or syntax
        if lbl.isupper():  # likely relationship
            continue
        raise CypherSecurityError(f"Unknown or unauthorized node label: ':{lbl}'. Allowed: {valid_labels}")

    # Inject LIMIT if absent
    if not re.search(r"\bLIMIT\s+\d+\b", tokens_only, re.IGNORECASE):
        # Remove trailing semicolon if present
        if clean_cypher.endswith(";"):
            clean_cypher = clean_cypher[:-1].strip()
        clean_cypher = f"{clean_cypher} LIMIT {default_limit};"

    return clean_cypher
