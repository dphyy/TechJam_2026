from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from .product_features import BUDGET_RE, FACET_PATTERNS, terms
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
            if key:
                self.constraint_to_asins[key].add(parent_asin)

        details = product.get("details")
        if isinstance(details, Mapping):
            for name, value in details.items():
                if value in (None, "", []):
                    continue
                rendered = f"{name}: {value}"
                key = self._add(self.detail_to_asins, rendered, parent_asin)
                if key:
                    self.constraint_to_asins[key].add(parent_asin)

        searchable = " ".join(
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
            key = normalize_constraint(value)
            matches = self.constraint_to_asins.get(key)
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
            matches = self._matches(
                "constraint_to_asins", normalize_constraint(value)
            )
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
        return connection
    except (OSError, sqlite3.DatabaseError):
        if connection is not None:
            connection.close()
        return None
