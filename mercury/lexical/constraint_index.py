from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from .product_features import (
    BUDGET_RE, FACET_PATTERNS, affirmed_terms, alternative_values, denied_terms, terms,
)
from .vector_index import catalog_sha256


CATALOG_INDEX_SCHEMA_VERSION = "1"
INDEX_NAMES = (
    "constraint_to_asins",
    "category_to_asins",
    "material_to_asins",
    "color_to_asins",
    "feature_to_asins",
    "detail_to_asins",
)


def normalize_constraint(value: object) -> str:
    """Normalize catalog and dialogue values to the same exact-match key."""
    return " ".join(terms(value))


def _values(value: object) -> Iterable[str]:
    if isinstance(value, list):
        return (str(item) for item in value if item not in (None, ""))
    if value not in (None, ""):
        return (str(value),)
    return ()


def _asserted_property(value: str) -> bool:
    # Preserve literal absence properties, but never let dropped 'not' create
    # the same exact key as an affirmative property.
    return not denied_terms(value) or bool(set(terms(value)) & {"no", "without", "free"})


class ConstraintIndex:
    """In-memory inverted indices derived only from public catalog fields."""

    def __init__(self) -> None:
        self.constraint_to_asins: dict[str, set[str]] = defaultdict(set)
        self.category_to_asins: dict[str, set[str]] = defaultdict(set)
        self.material_to_asins: dict[str, set[str]] = defaultdict(set)
        self.color_to_asins: dict[str, set[str]] = defaultdict(set)
        self.feature_to_asins: dict[str, set[str]] = defaultdict(set)
        self.detail_to_asins: dict[str, set[str]] = defaultdict(set)

    @staticmethod
    def _add(index: dict[str, set[str]], value: object, parent_asin: str) -> str:
        key = normalize_constraint(value)
        if key:
            index[key].add(parent_asin)
        return key

    def add_product(self, product: Mapping[str, object]) -> None:
        parent_asin = str(product["parent_asin"])

        categories = [
            key
            for value in _values(product.get("categories"))
            if (key := normalize_constraint(value))
        ]
        # A request may describe a product using the last two taxonomy nodes.
        # Store every suffix so both that phrase and a more specific leaf match.
        for start in range(len(categories)):
            self._add(
                self.category_to_asins,
                " ".join(categories[start:]),
                parent_asin,
            )
        for category in categories:
            self._add(self.category_to_asins, category, parent_asin)

        for value in _values(product.get("features")):
            key = self._add(self.feature_to_asins, value, parent_asin)
            if key and _asserted_property(value):
                self.constraint_to_asins[key].add(parent_asin)

        details = product.get("details")
        if isinstance(details, Mapping):
            for name, value in details.items():
                if value in (None, "", []):
                    continue
                rendered = f"{name}: {value}"
                key = self._add(self.detail_to_asins, rendered, parent_asin)
                if key and _asserted_property(rendered):
                    self.constraint_to_asins[key].add(parent_asin)

        searchable = " ".join(" ".join(affirmed_terms(value)) for value in
            [
                str(product.get("title") or ""),
                *list(_values(product.get("features"))),
                *(
                    [f"{key} {value}" for key, value in details.items()]
                    if isinstance(details, Mapping)
                    else []
                ),
                *list(_values(product.get("description"))),
                *list(_values(product.get("categories"))),
                str(product.get("store") or ""),
            ]
        )
        for attribute, target in (
            ("material", self.material_to_asins),
            ("color", self.color_to_asins),
        ):
            pattern = FACET_PATTERNS[attribute]
            for match in pattern.finditer(searchable):
                value = match.group(1)
                key = self._add(target, value, parent_asin)
                if key:
                    self.constraint_to_asins[key].add(parent_asin)
                    if attribute == "color":
                        self._add(
                            self.constraint_to_asins,
                            f"color {value}",
                            parent_asin,
                        )

    def exact_intersection(
        self,
        category: str,
        constraints: Iterable[str],
    ) -> set[str]:
        """Return products satisfying every catalog-recognized exact value."""
        matched_sets: list[set[str]] = []
        for value in constraints:
            if BUDGET_RE.search(value):
                continue
            matches = set().union(*(self.constraint_to_asins.get(normalize_constraint(branch), set())
                                    for branch in alternative_values(value)))
            if matches:
                matched_sets.append(matches)
        if not matched_sets:
            return set()

        result = set.intersection(*(set(matches) for matches in matched_sets))
        category_matches = self.category_to_asins.get(normalize_constraint(category))
        if category_matches:
            result.intersection_update(category_matches)
        return result

    def iter_entries(self) -> Iterable[tuple[str, str, str]]:
        """Yield the six named maps in a persistence-friendly representation."""
        for index_name in INDEX_NAMES:
            index = getattr(self, index_name)
            for value, parent_asins in index.items():
                for parent_asin in parent_asins:
                    yield index_name, value, parent_asin


class SQLiteConstraintIndex:
    """Read exact constraint intersections from a preprocessed SQLite artifact."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def _matches(self, index_name: str, value: str) -> set[str]:
        return {
            str(row[0])
            for row in self.connection.execute(
                "SELECT parent_asin FROM constraint_entries "
                "WHERE index_name = ? AND value = ?",
                (index_name, value),
            )
        }

    def exact_intersection(
        self,
        category: str,
        constraints: Iterable[str],
    ) -> set[str]:
        matched_sets: list[set[str]] = []
        for value in constraints:
            if BUDGET_RE.search(value):
                continue
            matches = set().union(*(self._matches("constraint_to_asins", normalize_constraint(branch))
                                    for branch in alternative_values(value)))
            if matches:
                matched_sets.append(matches)
        if not matched_sets:
            return set()

        result = set.intersection(*matched_sets)
        category_matches = self._matches(
            "category_to_asins", normalize_constraint(category)
        )
        if category_matches:
            result.intersection_update(category_matches)
        return result


def default_catalog_index_path(catalog_path: str | Path) -> Path:
    catalog = Path(catalog_path)
    return catalog.with_name(f"{catalog.stem}_index.sqlite3")


def _catalog_text(value: object) -> str:
    """Keep structured entries separate without changing their searchable words."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return "; ".join(f"{key} {_catalog_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "; ".join(_catalog_text(item) for item in value)
    return str(value)


def _validate_catalog_contents(connection: sqlite3.Connection, catalog: Path) -> None:
    """Verify every persisted field and membership against the actual catalog once.

    The ordered cursors keep Python memory bounded by one product's metadata and
    memberships; SQLite handles the external membership sort at initialization.
    """
    fields = ("parent_asin", "title", "categories", "features", "details", "store",
              "description", "price", "average_rating", "rating_number")
    products = iter(connection.execute(f"SELECT {','.join(fields)} FROM products ORDER BY rowid"))
    entries = iter(connection.execute(
        "SELECT c.index_name, c.value, c.parent_asin FROM constraint_entries AS c "
        "JOIN product_rows AS r ON r.parent_asin = c.parent_asin "
        "ORDER BY r.row_id, c.index_name, c.value, c.parent_asin"
    ))
    with catalog.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            serialized = tuple(_catalog_text(product.get(field)) for field in fields)
            if next(products, None) != serialized:
                raise ValueError("catalog index product content mismatch")
            expected = ConstraintIndex()
            expected.add_product(product)
            for entry in sorted(expected.iter_entries()):
                if next(entries, None) != entry:
                    raise ValueError("catalog index constraint content mismatch")
    if next(products, None) is not None or next(entries, None) is not None:
        raise ValueError("catalog index contains extra data")


def open_catalog_index(
    catalog_path: str | Path,
    index_path: str | Path,
) -> sqlite3.Connection | None:
    """Open a matching artifact read-only, or return None for a safe rebuild."""
    catalog = Path(catalog_path)
    artifact = Path(index_path)
    if not artifact.is_file():
        return None
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            artifact.resolve().as_uri() + "?mode=ro&immutable=1",
            uri=True,
        )
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if (
            metadata.get("schema_version") != CATALOG_INDEX_SCHEMA_VERSION
            or metadata.get("catalog_sha256") != catalog_sha256(catalog)
        ):
            connection.close()
            return None
        expected_rows = int(metadata.get("catalog_rows", "-1"))
        required_columns = {
            "products": {"parent_asin", "title", "categories", "features", "details", "store", "description", "price", "average_rating", "rating_number"},
            "product_rows": {"parent_asin", "row_id"},
            "constraint_entries": {"index_name", "value", "parent_asin"},
        }
        for table, required in required_columns.items():
            columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
            if not required.issubset(columns):
                raise ValueError("incomplete catalog index schema")
        counts = connection.execute(
            "SELECT (SELECT COUNT(*) FROM products), "
            "(SELECT COUNT(DISTINCT parent_asin) FROM products), "
            "(SELECT COUNT(*) FROM product_rows), "
            "(SELECT COUNT(DISTINCT row_id) FROM product_rows)"
        ).fetchone()
        if expected_rows < 0 or any(count != expected_rows for count in counts):
            raise ValueError("catalog index count mismatch")
        if connection.execute(
            "SELECT 1 FROM product_rows AS r LEFT JOIN products AS p ON p.rowid = r.row_id "
            "WHERE p.rowid IS NULL OR p.parent_asin != r.parent_asin LIMIT 1"
        ).fetchone():
            raise ValueError("catalog index row identity mismatch")
        placeholders = ",".join("?" for _ in INDEX_NAMES)
        if connection.execute(
            "SELECT 1 FROM constraint_entries AS c LEFT JOIN product_rows AS r "
            "ON r.parent_asin = c.parent_asin WHERE r.parent_asin IS NULL "
            f"OR c.index_name NOT IN ({placeholders}) OR c.value = '' LIMIT 1", INDEX_NAMES,
        ).fetchone():
            raise ValueError("invalid catalog constraint membership")
        _validate_catalog_contents(connection, catalog)
        return connection
    except (OSError, sqlite3.DatabaseError, ValueError, KeyError, TypeError):
        if connection is not None:
            connection.close()
        return None
