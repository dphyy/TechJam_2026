"""Conservative recognition of slate feedback, independent of product preferences."""
from __future__ import annotations

import re

from .product_features import terms

REFERENT = (r"(?:(?:these|those)(?:\s+(?:options|recommendations|products|items|choices|"
            r"suggestions|results|ones))?|(?:the|your)\s+(?:(?:shown|displayed|suggested)\s+)?"
            r"(?:options|recommendations|products|items|choices|suggestions|results|list|slate))")
REJECTION_PATTERNS = (
    re.compile(r"\b(?:none|neither)\s+of\s+" + REFERENT + r"\b"
               r"(?=\s*(?:[,.!?;]|$|(?:works?|fits?|appeals?)\b|(?:is|are)\s+(?:right|suitable|what)\b))", re.I),
    re.compile(r"\b" + REFERENT + r"\s+(?:(?:are|is|do|does)\s+not|aren't|isn't|don't|doesn't)\s+"
               r"(?:quite\s+)?(?:right|suitable|what\b|fit\b|work\b|match\b|appeal\b)", re.I),
    re.compile(r"\bi\s+(?:do\s+not|don't|dont)\s+(?:want|like|need)\s+(?:any\s+of\s+)?"
               + REFERENT + r"\b(?=\s*(?:[,.!?;]|$|anymore\b|at all\b|for me\b))", re.I),
    re.compile(r"\b(?:reject|skip|discard)\s+(?:all\s+of\s+)?" + REFERENT + r"\b", re.I),
    re.compile(r"\b" + REFERENT + r"\s+(?:are|is)\s+(?:all\s+)?(?:wrong|unsuitable|unacceptable)\b", re.I),
)
QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\x27[^\x27\n]+\x27(?!\w)')
HYPOTHETICAL = re.compile(r"\b(?:if|whether|maybe|perhaps|might)\b|"
                          r"\b(?:not saying|not rejecting|didn't say|do not mean|don't mean)\b", re.I)
PREFERENCE_PREFIX = re.compile(
    r"^(?:(?:for that|actually)[,:]?\s*)?(?:i\s+)?(?:would\s+)?"
    r"(?:prefer|need|want|require|have a preference for)\s+", re.I,
)


def explicit_slate_rejection(message: str) -> bool:
    """Require an actual negative statement about the displayed group."""
    unquoted = QUOTED.sub("", message[:8000])
    for clause in re.split(r"[.!?;\n]|\bbut\b", unquoted, flags=re.I):
        if HYPOTHETICAL.search(clause):
            continue
        if REJECTION_PATTERNS[0].search(clause):
            return True
        if re.search(r"\b(?:none|neither)\s+of\s+" + REFERENT + r"\b", clause, re.I):
            continue
        if any(pattern.search(clause) for pattern in REJECTION_PATTERNS[1:]):
            return True
    return False


def _semantic_value(text: str) -> tuple[str, ...]:
    cleaned = PREFERENCE_PREFIX.sub("", text.strip())
    return tuple(terms(cleaned))


def preference_content(message: str) -> str:
    """Remove actual slate feedback while preserving separately stated needs.

    Rejection describes the displayed results, not a positive product feature.
    Split only at explicit clause boundaries so mixed feedback such as
    "Those aren't right, I need cotton" still supplies its requirement.
    """
    if not explicit_slate_rejection(message):
        return message
    clauses = re.split(
        r"(?<!\d)\.|\.(?!\d)|[!?;\n]|\bbut\b|,\s+(?=[a-z])|(?:\band\b|\bbecause\b)\s*(?=(?:i\b|no\b|without\b|"
        r"avoid\b|actually\b|correction\b|please\b|make that\b))", message, flags=re.I,
    )
    kept = [clause.strip(" ,") for clause in clauses
            if clause.strip(" ,") and not explicit_slate_rejection(clause)]
    return "; ".join(kept)
