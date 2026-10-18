"""
PII redaction: a defense-in-depth safety net that runs on generated answers.

RBAC filtering (rag_chain.retrieve_with_scores) is the PRIMARY control -- a
non-HR/non-C-level role should never even retrieve HR chunks. This module
only matters if that primary control is ever bypassed (e.g. a prompt
injection attempt, or a future retrieval bug), which is why it's scoped
narrowly: it targets patterns that are specifically personal (emails, phone
numbers, employee IDs, labeled salary figures) rather than "any number",
so it doesn't accidentally redact legitimate aggregate figures that
finance/marketing roles are supposed to see (e.g. "$30M in vendor costs").
"""

import re

# Roles allowed to see PII outright -- redaction is a no-op for these.
PII_EXEMPT_ROLES = {"hr", "c_level"}

PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "employee_id": re.compile(r"\bFINEMP\d{4,}\b"),
    "salary": re.compile(r"(?i)\b(annual\s+salary|salary)\s*:?\s*[\d,]+(?:\.\d+)?"),
}


def detect_pii(text: str) -> dict[str, list[str]]:
    """Return {category: [matches]} for any PII patterns found -- useful for logging/tests."""
    found = {}
    for name, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            found[name] = matches
    return found


def redact_pii(text: str, role: str) -> str:
    """Redact PII patterns from text, unless the role is exempt (HR / C-level)."""
    if role in PII_EXEMPT_ROLES:
        return text

    redacted = text
    for name, pattern in PII_PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{name.upper()}]", redacted)
    return redacted
