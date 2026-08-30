from __future__ import annotations

import re
import sqlite3

from mercury.catalog import Catalog, FIELD_NAMES


TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOPWORDS = frozenset("a an and are as at be but by for from i in is it me my of on or please "
                      "some that the this to want with would you looking need like prefer "
                      "something any more options choices find show should must can could".split())


def terms(text: str) -> list[str]:
    return list(dict.fromkeys(token.lower() for token in TOKEN_RE.findall(text)
                             if (len(token) > 1 or token.isdigit()) and token.lower() not in STOPWORDS))[:64]


class SparseIndex:
    """Field-weighted BM25, with safe tokens and optional broad category scoping."""

    def __init__(self, catalog: Catalog) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='porter unicode61 remove_diacritics 2')"
        )
        self.connection.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
            ((product.parent_asin, *(product.fields[field] for field in FIELD_NAMES))
             for product in catalog.products),
        )
        self.connection.commit()

    def search(self, query: str, limit: int, categories: list[str] | None = None) -> list[str]:
        tokens = terms(query)
        if not tokens or limit <= 0:
            return []
        expression = "(" + " OR ".join(f'"{token}"' for token in tokens) + ")"
        category_tokens = terms(" ".join(categories or []))
        if category_tokens:
            expression += " AND {title categories} : (" + " OR ".join(
                f'"{token}"' for token in category_tokens) + ")"
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0), rowid LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [row[0] for row in rows]

    def search_factored(self, categories: list[str], requirements: list[str], limit: int) -> list[str]:
        """Search ordinary category/object and requirement fields as separate clauses."""
        category_tokens = terms(" ".join(categories))
        requirement_tokens = terms(" ".join(requirements))
        if limit <= 0 or (not category_tokens and not requirement_tokens):
            return []

        def expression(tokens: list[str], fields: str) -> str:
            return "{" + fields + "} : (" + " OR ".join(f'"{token}"' for token in tokens) + ")"

        clauses = []
        if category_tokens:
            clauses.append(expression(category_tokens, "title categories"))
        if requirement_tokens:
            clauses.append(expression(requirement_tokens, "title features details description"))
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0), rowid LIMIT ?",
            (" AND ".join(clauses), limit),
        ).fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        connection = getattr(self, "connection", None)
        if connection is not None:
            self.connection = None
            connection.close()

    def __del__(self) -> None:
        # The public agent contract has no lifecycle callback, so retain explicit
        # close() while also preventing an abandoned session from leaking SQLite.
        try:
            self.close()
        except Exception:
            pass


def fuse_routes(routes: dict[str, list[str]], weights: dict[str, float]) -> list[tuple[str, float, dict[str, float]]]:
    """Weighted reciprocal ranks avoid treating incomparable model scores as probabilities."""
    scores: dict[str, float] = {}
    parts: dict[str, dict[str, float]] = {}
    for route, identifiers in routes.items():
        weight = weights.get(route, 0.0)
        for rank, identifier in enumerate(dict.fromkeys(identifiers), 1):
            contribution = weight * 61.0 / (60.0 + rank)
            scores[identifier] = scores.get(identifier, 0.0) + contribution
            parts.setdefault(identifier, {})[route] = contribution
    return [(identifier, scores[identifier], parts[identifier])
            for identifier in sorted(scores, key=lambda key: (-scores[key], key))]
