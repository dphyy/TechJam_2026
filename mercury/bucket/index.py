from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[^\W_]+(?:\.[0-9]+)?", re.UNICODE)
FILLER = frozenset({
    "a", "an", "the", "i", "am", "im", "s", "for", "to", "of", "with", "and",
    "or", "either", "is", "are", "it", "my", "me", "please", "would", "like", "prefer",
    "need", "want", "looking", "searching", "something", "some", "but", "still", "exploring",
})


def tokens(value: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(unicodedata.normalize("NFKC", value).casefold()))


def normalized(value: str) -> str:
    return " ".join(tokens(value))


def category_tokens(value: str) -> tuple[str, ...]:
    def singular(word: str) -> str:
        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith("sses"):
            return word[:-2]
        return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word
    return tuple(singular(word) for word in tokens(value) if word not in FILLER)


def finite_number(value: object, *, minimum: float = 0.0) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= minimum else None


def flatten(value: object) -> list[str]:
    """Preserve complete atomic values and their nested field labels."""
    if value is None:
        return []
    if isinstance(value, dict):
        result = []
        for key, child in value.items():
            for text in flatten(child):
                result.extend((f"{key}: {text}", text))
        return list(dict.fromkeys(result))
    if isinstance(value, list):
        return [text for item in value for text in flatten(item)]
    return [str(value)] if str(value).strip() else []


def positive_tokens(text: str) -> frozenset[str]:
    result = set()
    for clause in re.split(r"[.!?;\n]|,\s*", text):
        words = tokens(clause)
        for index, word in enumerate(words):
            before, after = words[max(0, index - 3):index], words[index + 1:index + 3]
            denied = any(term in {"no", "without", "neither"} for term in before)
            denied = denied or ("not" in before and "only" not in before)
            denied = denied or (after[:1] == ("free",)) or tuple(before[-2:]) == ("free", "of")
            if not denied:
                result.add(word)
    return frozenset(result)


@dataclass(frozen=True, slots=True)
class Product:
    identifier: str
    raw_json: str
    atoms: frozenset[str]
    fields: tuple[tuple[str, str], ...]
    affirmed_fields: tuple[frozenset[str], ...]
    positive: frozenset[str]
    price: float | None
    quality: tuple[float, float]

    def raw(self) -> dict:
        return json.loads(self.raw_json)


class CatalogIndex:
    def __init__(self, path: str | Path) -> None:
        self.products: dict[str, Product] = {}
        self.categories: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self.atomic: dict[str, set[str]] = defaultdict(set)
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute(
            "CREATE VIRTUAL TABLE documents USING fts5(identifier UNINDEXED, title, categories, body)"
        )
        digest = hashlib.sha256()
        try:
            with Path(path).open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    row = json.loads(line)
                    identifier = row.get("parent_asin") if isinstance(row, dict) else None
                    if not isinstance(identifier, str) or not identifier.strip() or identifier in self.products:
                        raise ValueError("catalog IDs must be unique nonempty strings")
                    fields = [(str(name), text) for name, value in row.items() if name != "parent_asin"
                              for text in flatten(value)]
                    fields = [(name, text) for name, text in fields if tokens(text)]
                    normalized_fields = tuple((name, normalized(text)) for name, text in fields)
                    affirmed_fields = tuple(positive_tokens(text) for _, text in fields)
                    atoms = frozenset(text for _, text in normalized_fields)
                    positive = frozenset(word for _, text in fields for word in positive_tokens(text))
                    count = finite_number(row.get("rating_number")) or 0.0
                    rating = min(finite_number(row.get("average_rating")) or 0.0, 5.0)
                    product = Product(identifier, line.decode("utf-8"), atoms, normalized_fields, affirmed_fields,
                                      positive, finite_number(row.get("price")), (math.log1p(count), rating))
                    self.products[identifier] = product
                    for atom in atoms:
                        self.atomic[atom].add(identifier)
                    parts = [part for value in flatten(row.get("categories"))
                             for part in re.split(r"[,>/|]", value) if category_tokens(part)]
                    for start in range(len(parts)):
                        for end in range(start + 1, len(parts) + 1):
                            self.categories[category_tokens(" ".join(parts[start:end]))].add(identifier)
                    self.connection.execute("INSERT INTO documents VALUES (?, ?, ?, ?)", (
                        identifier, " ".join(flatten(row.get("title"))),
                        " ".join(flatten(row.get("categories"))), "\n".join(text for _, text in fields),
                    ))
            self.connection.commit()
        except Exception:
            self.connection.close()
            raise
        self.sha256 = digest.hexdigest()
        self._category_keys = sorted(self.categories, key=lambda key: (-len(key), key))
        self._quality_order = sorted(self.products, key=lambda key: (
            -self.products[key].quality[0], -self.products[key].quality[1], key,
        ))

    def close(self) -> None:
        self.connection.close()
        self.products.clear()
        self.categories.clear()
        self.atomic.clear()
        self._category_keys.clear()
        self._quality_order.clear()

    def resolve_category(self, text: str, *, explicit: bool) -> tuple[set[str], str]:
        query = category_tokens(text)
        if not query:
            return set(), "unresolved"
        if query in self.categories:
            return set(self.categories[query]), "exact"
        padded = " " + " ".join(query) + " "
        matches = [key for key in self._category_keys if " " + " ".join(key) + " " in padded]
        if matches:
            width = len(matches[0])
            return set().union(*(self.categories[key] for key in matches if len(key) == width)), "union"
        if explicit:
            wanted = set(query)
            matches = [key for key in self._category_keys if wanted <= set(key)]
            if matches:
                return set().union(*(self.categories[key] for key in matches)), "token_union"
        return set(), "unresolved"

    def lexical(self, words: tuple[str, ...], limit: int) -> list[str]:
        unique = tuple(dict.fromkeys(word for word in words if word not in FILLER))[:64]
        if not unique:
            return []
        expression = " OR ".join('"' + word.replace('"', '""') + '"' for word in unique)
        return [row[0] for row in self.connection.execute(
            "SELECT identifier FROM documents WHERE documents MATCH ? "
            "ORDER BY bm25(documents, 0.0, 1.0, 1.0, 1.0), identifier LIMIT ?", (expression, limit),
        )]

    def candidates(self, category: str, query: str, phrases: tuple[str, ...],
                   limit: int, lexical_limit: int) -> tuple[list[str], set[str], dict]:
        category_ids, mode = self.resolve_category(category or query, explicit=bool(category))
        lexical_ids = self.lexical(tokens(query), lexical_limit)
        exact_counts: dict[str, int] = defaultdict(int)
        for phrase in phrases:
            for identifier in self.atomic.get(phrase, ()):
                exact_counts[identifier] += 1
        union = category_ids | set(lexical_ids) | set(exact_counts)
        if not union:
            union.update(self._quality_order[:limit])
        if len(union) < min(10, len(self.products)):
            union.update(self._quality_order[:10])
        lexical_rank = {identifier: rank for rank, identifier in enumerate(lexical_ids)}
        selected = sorted(union, key=lambda identifier: (
            -int(identifier in category_ids), -exact_counts.get(identifier, 0),
            lexical_rank.get(identifier, lexical_limit),
            -self.products[identifier].quality[0], identifier,
        ))[:limit]
        return selected, category_ids, {
            "category_mode": mode, "category_count": len(category_ids),
            "lexical_count": len(lexical_ids), "atomic_count": len(exact_counts),
            "union_count": len(union), "candidate_count": len(selected), "candidate_limit": limit,
        }
