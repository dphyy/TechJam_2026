"""Bounded field searches whose candidate union survives through scoring."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, replace
from math import isfinite
from typing import TYPE_CHECKING

from mercury.catalog import FIELD_NAMES, Catalog
from mercury.ranking import preference_evidence
from mercury.state import SessionState
from mercury.types import Candidate, Preference, Product

if TYPE_CHECKING:
    from mercury.multiview.raw_state import RawEvidenceState


# Title, category, feature, structured-detail, store, description weights.
VIEWS = {
    "identity": (8.0, 6.0, 2.0, 3.0, 2.0, 1.0),
    "structured": (3.0, 8.0, 3.0, 8.0, 3.0, 1.0),
    "descriptive": (2.0, 2.0, 8.0, 4.0, 1.0, 6.0),
}
ATTRIBUTE_WEIGHTS = {
    "category": 1.1, "material": 0.65, "color": 0.55, "use_case": 0.65,
    "style": 0.45, "feature": 0.65, "size": 0.3, "brand": 0.5, "budget": 0.5,
}


@dataclass(frozen=True, slots=True)
class Config:
    state_mode: str = "typed"
    views: tuple[str, ...] = ("identity", "structured", "descriptive")
    route_limit: int = 64
    constraint_limit: int = 32
    category_limit: int = 96
    max_constraints: int = 24
    max_query_terms: int = 64
    rrf_constant: float = 40.0
    evidence_weight: float = 1.0
    field_routes: bool = True
    constraint_routes: bool = True
    category_rescue: bool = True
    fullwidth: bool = True
    max_sessions: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.state_mode, str) or self.state_mode not in {"typed", "raw"}:
            raise ValueError("state_mode must be typed or raw")
        for name, maximum in (
            ("route_limit", 512), ("constraint_limit", 256), ("category_limit", 512),
            ("max_constraints", 64), ("max_query_terms", 128), ("max_sessions", 1024),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        for name, maximum in (("rrf_constant", 10000), ("evidence_weight", 10)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be finite and in [0, {maximum}]")
        if not isinstance(self.views, tuple) or any(
                not isinstance(view, str) or view not in VIEWS for view in self.views) \
                or len(set(self.views)) != len(self.views):
            raise ValueError("views must be a tuple of unique known view names")
        for name in ("field_routes", "constraint_routes", "category_rescue", "fullwidth"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


def _tokens(text: str, limit: int) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return list(dict.fromkeys(re.findall(r"[^\W_]+", normalized)))[:limit]


def _expression(text: str, limit: int, *, conjunction: bool = False) -> str:
    # Only generated, quoted word tokens enter MATCH; user syntax stays inert.
    operator = " AND " if conjunction else " OR "
    return operator.join(f'"{token}"' for token in _tokens(text, limit))


def preference_groups(preferences: list[Preference]) -> list[list[Preference]]:
    """Explicit alternatives share one vote; independent requirements do not."""
    groups: dict[tuple, list[Preference]] = {}
    for index, preference in enumerate(preferences):
        if not preference.active or preference.polarity == 0:
            continue
        key = ("item", index)
        if preference.polarity == 1 and preference.alternative_group is not None:
            key = ("alternative", preference.attribute, preference.alternative_group,
                   preference.scope)
        groups.setdefault(key, []).append(preference)
    return list(groups.values())


def evidence_score(product: Product, preferences: list[Preference]) -> float:
    return _assess(product, preferences)[0]


def _assess(product: Product, preferences: list[Preference]) -> tuple[float, bool]:
    score = 0.0
    violation = False
    for group in preference_groups(preferences):
        votes = []
        violations = []
        for preference in group:
            signal = preference_evidence(product, preference)
            violations.append(preference.hard and signal < 0)
            strength = 1.25 if preference.hard else preference.confidence
            if signal < 0:
                strength *= 2.0
            votes.append(signal * strength * ATTRIBUTE_WEIGHTS.get(preference.attribute, 0.4))
        score += max(votes)
        violation |= all(violations)
    return score, violation


def explain(product: Product, preferences: list[Preference]) -> list[dict]:
    """Report observed fields; absence and conflicting evidence remain unknown."""
    result = []
    for preference in preferences:
        if not preference.active or preference.polarity == 0:
            continue
        signal = preference_evidence(product, preference)
        fields = []
        if signal and preference.attribute == "budget":
            fields = ["price"]
        elif signal:
            for field in FIELD_NAMES:
                fragment = replace(
                    product, fields={field: product.fields[field]},
                    evidence=tuple(item for item in product.evidence
                                   if item.source.split(".", 1)[0] == field),
                )
                field_signal = preference_evidence(fragment, preference)
                if field_signal * signal > 0:
                    fields.append(field)
        result.append({
            "attribute": preference.attribute, "value": preference.value,
            "polarity": preference.polarity,
            "status": "supported" if signal > 0 else "contradicted" if signal < 0 else "unknown",
            "signal": signal, "fields": fields,
        })
    return result


@dataclass(frozen=True, slots=True)
class SearchResult:
    candidates: list[Candidate]
    routes: dict[str, list[str]]
    queries: dict[str, str]
    constraints_omitted: int


class MultiViewIndex:
    def __init__(self, catalog: Catalog, config: Config) -> None:
        self.catalog = catalog
        self.config = config
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "title, categories, features, details, store, description, "
            "tokenize='porter unicode61 remove_diacritics 2')"
        )
        # All raw metadata participates. No length cap or preselected vocabulary.
        self.connection.executemany(
            "INSERT INTO products(rowid, title, categories, features, details, store, description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ((index, *(product.fields[field] for field in FIELD_NAMES))
             for index, product in enumerate(catalog.products, 1)),
        )
        self.connection.commit()
        self._fallback = sorted(catalog.by_id)[:10]

    def close(self) -> None:
        self.connection.close()

    def _search(self, expression: str, weights: tuple[float, ...], limit: int) -> list[str]:
        if not expression:
            return []
        placeholders = ", ".join("?" for _ in weights)
        rows = self.connection.execute(
            f"SELECT rowid FROM products WHERE products MATCH ? "
            f"ORDER BY bm25(products, {placeholders}), rowid LIMIT ?",
            (expression, *weights, limit),
        )
        return [self.catalog.products[rowid - 1].parent_asin for (rowid,) in rows]

    def search(self, state: SessionState | RawEvidenceState) -> SearchResult:
        config = self.config
        preferences = state.active_preferences()
        query = _expression(state.query() + " " + state.source_alias_query(), config.max_query_terms)
        routes: dict[str, list[str]] = {}
        queries: dict[str, str] = {}
        family_weights: dict[str, float] = {}
        route_families: dict[str, str] = {}

        def add(name: str, expression: str, weights: tuple[float, ...], limit: int,
                family: str, weight: float) -> None:
            if not expression:
                return
            queries[name] = expression
            routes[name] = self._search(expression, weights, limit)
            route_families[name] = family
            family_weights[family] = weight

        for name in config.views:
            add(f"view:{name}", query, VIEWS[name], config.route_limit, f"view:{name}", 1.0)
        if config.field_routes:
            for field in FIELD_NAMES:
                add(f"field:{field}", f"{field} : ({query})" if query else "",
                    VIEWS["structured"], config.route_limit, f"field:{field}", 0.5)

        categories = [preference.value for preference in preferences
                      if preference.polarity == 1 and preference.attribute == "category"]
        category = _expression(" ".join(categories), config.max_query_terms)
        category_expression = f"{{title categories}} : ({category})" if category else ""
        if config.category_rescue:
            add("category", category_expression, VIEWS["identity"],
                config.category_limit, "category", 1.0)

        constraints = [group for group in preference_groups(preferences)
                       if group[0].polarity == 1 and group[0].attribute not in {"category", "budget"}]
        # Prefer explicit hard and recent constraints if a very long conversation
        # exceeds the query budget. This affects admission only, never scoring.
        constraints.sort(key=lambda group: (
            -int(any(item.hard for item in group)), -max(item.source_turn for item in group),
            group[0].attribute, tuple(sorted(item.value for item in group)),
        ))
        omitted = (max(0, len(constraints) - config.max_constraints)
                   if config.constraint_routes else len(constraints))
        if config.constraint_routes:
            for index, group in enumerate(constraints[:config.max_constraints]):
                expressions = [_expression(item.value, config.max_query_terms, conjunction=True)
                               for item in group]
                expression = " OR ".join(f"({item})" for item in expressions if item)
                family = f"constraint:{index}"
                add(family, expression, VIEWS["descriptive"], config.constraint_limit, family, 1.0)
                if category_expression and expression:
                    add(family + ":category", f"({category_expression}) AND ({expression})",
                        VIEWS["structured"], config.constraint_limit, family, 1.0)

        # Retain every route member. Alternative/scoped routes use the maximum
        # vote within a family, avoiding an extra vote for matching two choices.
        votes: dict[str, dict[str, float]] = {}
        for route, identifiers in routes.items():
            family = route_families[route]
            for rank, identifier in enumerate(identifiers, 1):
                score = family_weights[family] * (config.rrf_constant + 1) / (config.rrf_constant + rank)
                parts = votes.setdefault(identifier, {})
                parts[family] = max(parts.get(family, 0.0), score)
        denominator = sum(family_weights.values()) or 1.0
        # A full-width control stays full-width even for sparse/no lexical hits.
        # Stable ID order is only a fallback, never a catalog quality assumption.
        for identifier in self._fallback:
            if len(votes) >= min(10, len(self.catalog.products)):
                break
            votes.setdefault(identifier, {})
        candidates = []
        for identifier, parts in votes.items():
            product = self.catalog.by_id[identifier]
            fusion = sum(parts.values()) / denominator
            evidence, violation = _assess(product, preferences)
            candidates.append(Candidate(product, fusion + config.evidence_weight * evidence,
                                        {**parts, "fusion": fusion, "evidence": evidence,
                                         "semantic_violation": float(violation)}))
        # Lexical agreement cannot compensate for an observed hard contradiction.
        # Shift the violating tier below the lowest unshifted score, preserving
        # relative order and every candidate. Unknowns never enter this tier.
        if candidates:
            penalty = max(item.score for item in candidates) - min(item.score for item in candidates) + 1.0
            for candidate in candidates:
                if candidate.route_scores["semantic_violation"]:
                    candidate.score -= penalty
                    candidate.route_scores["violation_penalty"] = -penalty
        candidates.sort(key=lambda item: (-item.score, item.product.parent_asin))
        return SearchResult(candidates, routes, queries, omitted)
