"""Catalog-grounded category recognition for asserted shopping requests."""
from __future__ import annotations

import re

from .product_features import FACET_PATTERNS, terms


def category_terms(tokens) -> tuple[str, ...]:
    tokens = tuple(tokens)
    end = 0
    while end < len(tokens) and tokens[end] in {"clothing", "shoes", "jewelry"}:
        end += 1
    if end >= 2:
        tokens = tokens[end:]
    aliases = {"women": "woman", "womens": "woman", "men": "man", "mens": "man", "jewellery": "jewelry"}
    return tuple(aliases.get(token, token[:-1] if len(token) > 3 and token.endswith("s")
                             and not token.endswith(("ss", "us")) else token)
                 for token in tokens if len(token) > 2)


REQUEST = re.compile(
    r"^(?:(?:actually|instead|i changed my mind)[,:]?\s*)?"
    r"(?:(?:i(?:'d| would)?\s+)?(?:want|need|prefer|would like)|"
    r"i(?:'m| am)\s+looking for|looking for|(?:please\s+)?show me)\s+", re.I,
)


def category_choices(value: str) -> tuple[str, ...]:
    parts = tuple(re.sub(r"^either\s+", "", part.strip(), flags=re.I)
                  for part in re.split(r"\s+or\s+", value, flags=re.I))
    return parts if 2 <= len(parts) <= 4 and all(parts) and not re.search(r'["“”]', value) else (value,)


def asserted_category(message: str, names: frozenset[tuple[str, ...]], *,
                      correction: bool = False, answering: bool = False) -> tuple[str, str] | None:
    """Return a known category and separate modifiers, never infer from a mention.

    A bare taxonomy name is accepted only as an explicit correction or an answer
    to a category question. Other statements must start with a positive request.
    """
    if not names:
        return None
    request = REQUEST.match(message)
    if not (request or correction or answering):
        return None
    body = message[request.end():] if request else message
    body = re.sub(r"^(?:a pair of|pair of|a|an|some)\s+", "", body, flags=re.I)
    body = re.sub(r"\s+instead(?=[.!?;]|$)", "", body, flags=re.I)
    # Keep numeric separators inside the remainder; only a category clause ends here.
    parts = re.split(r"[.!?;]|,\s+|\b(?:with|under|below|over|above|for|that|having)\b", body, maxsplit=1, flags=re.I)
    head = parts[0].strip()
    remainder = body[len(parts[0]):].strip(" .!?;,")
    if re.search(r"[\"“”]|\b(?:not|no(?![- ]show\b)|without|avoid|if|maybe|perhaps|might|keep|instead of)\b", head, re.I):
        return None
    choices = category_choices(head)
    if len(choices) > 1:
        return (head, remainder) if all(category_terms(terms(choice)) in names for choice in choices) else None
    if category_terms(terms(head)) in names:
        return head, remainder
    # Recognize 'blue leather shoes' without making blue/leather taxonomy gates.
    words = list(re.finditer(r"\S+", head))
    for index in range(1, min(len(words), 12)):
        category = head[words[index].start():]
        if category_terms(terms(category)) not in names:
            continue
        modifiers = head[:words[index].start()].strip()
        leftover = modifiers
        for pattern in FACET_PATTERNS.values():
            leftover = pattern.sub("", leftover)
        if not terms(leftover):
            return category, "; ".join(value for value in (modifiers, remainder) if value)
    return None
