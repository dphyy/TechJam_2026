from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field as dataclass_field, replace
from functools import lru_cache
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol


TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
BUDGET_RE = re.compile(
    r"\b(?P<mode>under|below|maximum|max|around|about|budget(?:\s+around)?)?\s*"
    r"\$\s*(?P<amount>\d+(?:\.\d+)?)",
    re.I,
)
STOPWORDS = {
    "a", "about", "additional", "am", "an", "and", "are", "as", "at", "be",
    "but", "by", "do", "for", "from", "have", "i", "in", "is", "it", "looking",
    "me", "my", "need", "not", "of", "on", "or", "please", "preference", "some",
    "still", "that", "the", "these", "this", "those", "to", "want", "what", "with",
    "would", "you", "your",
}
FIELD_WEIGHTS = {
    "title": 4.0,
    "categories": 3.0,
    "features": 2.8,
    "details": 2.8,
    "store": 1.5,
    "description": 1.3,
}
FIELD_ORDER = tuple(FIELD_WEIGHTS)
FACET_PATTERNS = {
    "material": re.compile(
        r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|linen|"
        r"denim|fleece|suede|canvas|rubber|synthetic|acrylic|fabric)\b", re.I
    ),
    "color": re.compile(
        r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|"
        r"orange|beige|navy|gold|silver|multicolor)\b", re.I
    ),
    "size": re.compile(
        r"\b(x{0,3}s|x{0,4}l|small|medium|large|wide|narrow|petite|plus size)\b", re.I
    ),
    "style": re.compile(
        r"\b(casual|formal|classic|modern|vintage|slim|regular|relaxed|fitted|"
        r"loose|athletic|crew neck|v-neck|long sleeve|short sleeve)\b", re.I
    ),
    "use_case": re.compile(
        r"\b(running|hiking|walking|work|office|gym|workout|sports|travel|"
        r"winter|outdoor|wedding|party|sleep|swimming|cycling)\b", re.I
    ),
}
FACET_ORDER = tuple(FACET_PATTERNS)
FIELD_SEPARATOR = "\x1f"
COMPONENT_RE = re.compile(r"\b(lining|upper|outsole|midsole|insole|sole|shell|sleeves?|pockets?|collar|hood)\b", re.I)


@lru_cache(maxsize=4096)
def component_scope(value: str) -> str | None:
    matches = list(COMPONENT_RE.finditer(value))
    owners = {match.group().lower().rstrip("s") for match in matches}
    if len(owners) != 1:
        return None
    match = matches[0]
    before, after = value[:match.start()].strip(), value[match.end():].strip()
    explicit_label = after.startswith(":")
    named_facet = re.search(r"\b(?:material|color|colour)\b", after, re.I)
    prefix_value = not before and any(pattern.search(after) for name, pattern in FACET_PATTERNS.items() if name in {"material", "color"})
    suffix_value = any(re.search(pattern.pattern + r"\s*$", before, re.I) for name, pattern in FACET_PATTERNS.items() if name in {"material", "color"})
    return next(iter(owners)) if explicit_label or named_facet or prefix_value or suffix_value else None


def component_value(value: str) -> str:
    return COMPONENT_RE.sub(" ", value).strip(" :;,-")


def affirmed_terms(value: str) -> tuple[str, ...]:
    """Only explicit positive wording may prove an excluded value is present."""
    value = re.sub(r"\bnot (?:only|just)\b", "", value, flags=re.I)
    value = re.sub(r"\b\w+[- ]free\b", "", value, flags=re.I)
    value = re.sub(r"\b(?:no|not|without|free of|doesn't contain|does not contain)\b[^,;.!?]*", "", value, flags=re.I)
    return tuple(terms(value))


@lru_cache(maxsize=4096)
def alternative_values(value: str) -> tuple[str, ...]:
    """Recognize bounded, same-facet alternatives without splitting ordinary phrases."""
    parts = tuple(re.sub(r"^\s*either\s+", "", part, flags=re.I).strip()
                  for part in re.split(r"\s+or\s+", value, flags=re.I))
    if not 2 <= len(parts) <= 4 or any(not part for part in parts):
        return (value,)
    if re.search(r"\b(?:not|no|without|maybe)\b|[\"“”]", value, re.I):
        return (value,)
    facets = [{name for name, pattern in FACET_PATTERNS.items() if pattern.search(part)}
              for part in parts]
    if any(len(names) != 1 for names in facets) or any(names != facets[0] for names in facets):
        return (value,)
    owners = {component_scope(part) for part in parts} - {None}
    if len(owners) > 1:
        return (value,)
    owner = next(iter(owners), None)
    return tuple(f"{owner}: {component_value(part)}" if owner else part for part in parts)


def _component_fields(fields: Mapping[str, str]) -> Mapping[str, tuple[str, ...]]:
    """Keep component values within their own field and clause boundaries."""
    scoped: dict[str, list[str]] = {}
    for field_index, name in enumerate(FIELD_ORDER):
        raw = fields.get(name, "")
        for clause in re.split(r"[;\n.!?]", raw):
            matches = list(COMPONENT_RE.finditer(clause))
            if not matches:
                continue
            # Prefix records include flattened structured details: lining cotton upper leather.
            prefix = not clause[:matches[0].start()].strip(" ,:")
            for index, match in enumerate(matches):
                owner = match.group().lower().rstrip("s")
                if prefix or clause[match.end():].lstrip().startswith(":"):
                    end = matches[index + 1].start() if index + 1 < len(matches) else len(clause)
                    value = clause[match.end():end].strip(" ,:")
                    rendered = f"{match.group()} {value}"
                else:
                    start = matches[index - 1].end() if index else 0
                    value = clause[start:match.start()].strip(" ,:")
                    rendered = f"{value} {match.group()}"
                if value:
                    slots = scoped.setdefault(owner, [""] * len(FIELD_ORDER))
                    slots[field_index] += " " + rendered
    return MappingProxyType({owner: tuple(values) for owner, values in scoped.items()})


class EvidenceLike(Protocol):
    text: str
    weight: float


def terms(value: object, *, min_length: int = 2) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(str(value or ""))
        if len(token) >= min_length and token.lower() not in STOPWORDS
    ]


def _compact(value: object, limit: int = 32) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit].rstrip()


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return 0


@dataclass(frozen=True, slots=True)
class ProductFeatures:
    parent_asin: str
    token_weights: Mapping[str, float]
    normalized_text: str
    field_sequences: tuple[tuple[str, ...], ...]
    feature_tokens: frozenset[str]
    category_tokens: tuple[str, ...]
    price: float | None
    brand: str
    average_rating: float
    rating_number: int
    component_fields: Mapping[str, tuple[str, ...]] = dataclass_field(default_factory=dict)
    affirmed_sequences: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class ProductQuestionFeatures:
    facets: tuple[tuple[str, ...], ...]

    def facet_values(self, attribute: str) -> tuple[str, ...]:
        try:
            return self.facets[FACET_ORDER.index(attribute)]
        except ValueError:
            return ()


@dataclass(frozen=True, slots=True)
class CompiledEvidence:
    tokens: tuple[str, ...]
    normalized_query: str
    weight: float
    source: str
    attribute: str | None
    facets: tuple[tuple[str, tuple[str, ...]], ...]
    is_budget: bool
    scope: str | None = None
    alternatives: tuple[CompiledEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class BudgetPreference:
    mode: str
    amount: float
    weight: float


@dataclass(frozen=True, slots=True)
class CompiledQuery:
    evidence: tuple[CompiledEvidence, ...]
    preference_tokens: tuple[str, ...]
    budgets: tuple[BudgetPreference, ...]


def evidence_product(product: ProductFeatures, evidence: CompiledEvidence) -> ProductFeatures:
    if evidence.scope is None:
        return product
    values = product.component_fields.get(evidence.scope, ("",) * len(FIELD_ORDER))
    sequences = tuple(affirmed_terms(value) for value in values)
    weights: dict[str, float] = {}
    for name, sequence in zip(FIELD_ORDER, sequences):
        for token in sequence:
            weights[token] = max(weights.get(token, 0.0), FIELD_WEIGHTS[name])
    return replace(product, token_weights=MappingProxyType(weights),
                   normalized_text=FIELD_SEPARATOR.join(" ".join(sequence) for sequence in sequences),
                   field_sequences=sequences,
                   affirmed_sequences=sequences,
                   feature_tokens=frozenset(sequences[FIELD_ORDER.index("features")]))


def resolve_query(product: ProductFeatures, query: CompiledQuery) -> CompiledQuery:
    """Count an OR group once, using its strongest available lexical witness."""
    if not any(item.alternatives for item in query.evidence):
        return query
    def support(item: CompiledEvidence) -> tuple[bool, float, float]:
        view = evidence_product(product, item)
        return (bool(item.tokens) and item.normalized_query in view.normalized_text,
                sum(token in view.token_weights for token in item.tokens) / max(1, len(item.tokens)),
                sum(view.token_weights.get(token, 0.0) for token in item.tokens) / max(1, len(item.tokens)))
    return replace(query, evidence=tuple(max(item.alternatives, key=support) if item.alternatives else item
                                         for item in query.evidence))


@dataclass(frozen=True, slots=True)
class FeatureCacheInfo:
    hits: int
    misses: int
    evictions: int
    current_size: int
    max_size: int


class ProductFeatureStore:
    """Bounded LRU store of immutable, tokenized product representations."""

    def __init__(self, max_size: int = 12_000) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._products: OrderedDict[str, ProductFeatures] = OrderedDict()
        self._question_features: OrderedDict[str, ProductQuestionFeatures] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._question_hits = 0
        self._question_misses = 0
        self._question_evictions = 0

    def add(
        self,
        parent_asin: str,
        fields: Mapping[str, str],
        *,
        price: object = None,
        average_rating: object = None,
        rating_number: object = None,
    ) -> ProductFeatures:
        if parent_asin in self._products:
            raise ValueError(f"duplicate parent_asin in catalog: {parent_asin}")

        sequences: dict[str, tuple[str, ...]] = {}
        token_weights: dict[str, float] = {}
        normalized_fields: list[str] = []
        for field in FIELD_ORDER:
            field_terms = terms(fields.get(field, ""))
            field_tokens = tuple(field_terms)
            sequences[field] = field_tokens
            normalized_fields.append(" ".join(field_terms))
            field_weight = FIELD_WEIGHTS[field]
            for token in field_tokens:
                token_weights[token] = max(
                    field_weight, token_weights.get(token, 0.0)
                )

        feature_tokens = frozenset(
            token for token in sequences["features"] if len(token) > 2
        )
        category_tokens = tuple(
            token for token in sequences["categories"] if len(token) > 2
        )
        parsed_rating = _optional_float(average_rating) or 0.0
        features = ProductFeatures(
            parent_asin=parent_asin,
            token_weights=MappingProxyType(token_weights),
            normalized_text=FIELD_SEPARATOR.join(normalized_fields),
            field_sequences=tuple(sequences[field] for field in FIELD_ORDER),
            feature_tokens=feature_tokens,
            category_tokens=category_tokens,
            price=_optional_float(price),
            brand=_compact(fields.get("store", "")).casefold(),
            average_rating=parsed_rating,
            rating_number=_non_negative_int(rating_number),
            component_fields=_component_fields(fields),
            affirmed_sequences=tuple(affirmed_terms(fields.get(name, "")) for name in FIELD_ORDER),
        )
        self._insert(parent_asin, features)
        return features

    def get(self, parent_asin: str) -> ProductFeatures:
        features = self._products[parent_asin]
        self._products.move_to_end(parent_asin)
        return features

    def get_or_add(
        self,
        parent_asin: str,
        fields: Mapping[str, str],
        *,
        price: object = None,
        average_rating: object = None,
        rating_number: object = None,
    ) -> ProductFeatures:
        existing = self._products.get(parent_asin)
        if existing is not None:
            self._hits += 1
            self._products.move_to_end(parent_asin)
            return existing
        self._misses += 1
        return self.add(
            parent_asin,
            fields,
            price=price,
            average_rating=average_rating,
            rating_number=rating_number,
        )

    def _insert(self, parent_asin: str, features: ProductFeatures) -> None:
        self._products[parent_asin] = features
        if len(self._products) > self.max_size:
            self._products.popitem(last=False)
            self._evictions += 1

    def question_features(self, product: Mapping[str, object]) -> ProductQuestionFeatures:
        parent_asin = str(product["parent_asin"])
        existing = self._question_features.get(parent_asin)
        if existing is not None:
            self._question_hits += 1
            self._question_features.move_to_end(parent_asin)
            return existing

        self._question_misses += 1
        searchable = " ".join(
            str(product.get(field) or "")
            for field in ("title", "features", "details", "description")
        )
        features = ProductQuestionFeatures(
            facets=tuple(
                tuple(
                    sorted({match.lower() for match in pattern.findall(searchable)})
                )
                for pattern in FACET_PATTERNS.values()
            )
        )
        self._question_features[parent_asin] = features
        if len(self._question_features) > self.max_size:
            self._question_features.popitem(last=False)
            self._question_evictions += 1
        return features

    def cache_info(self) -> FeatureCacheInfo:
        return FeatureCacheInfo(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            current_size=len(self._products),
            max_size=self.max_size,
        )

    def question_cache_info(self) -> FeatureCacheInfo:
        return FeatureCacheInfo(
            hits=self._question_hits,
            misses=self._question_misses,
            evictions=self._question_evictions,
            current_size=len(self._question_features),
            max_size=self.max_size,
        )

    def compile_query(
        self,
        evidence: Iterable[EvidenceLike],
        user_profile: Mapping[str, object] | None = None,
    ) -> CompiledQuery:
        compiled_evidence: list[CompiledEvidence] = []
        budgets: list[BudgetPreference] = []
        for item in evidence:
            def compile_value(value: str) -> CompiledEvidence:
                scope = component_scope(value)
                scoped_value = component_value(value) if scope and getattr(item, "source", "") == "exclusion" else value
                unique_terms = tuple(dict.fromkeys(terms(scoped_value)))
                return CompiledEvidence(
                    tokens=unique_terms,
                    normalized_query=" ".join(unique_terms),
                    weight=item.weight,
                    source=str(getattr(item, "source", "")),
                    attribute=getattr(item, "attribute", None),
                    facets=tuple(
                        (
                            attribute,
                            tuple(
                                sorted(
                                    {
                                        match.lower()
                                        for match in pattern.findall(scoped_value)
                                    }
                                )
                            ),
                        )
                        for attribute, pattern in FACET_PATTERNS.items()
                        if pattern.search(scoped_value)
                    ),
                    is_budget=bool(BUDGET_RE.search(item.text)),
                    scope=scope,
                )
            compiled = compile_value(item.text)
            alternatives = alternative_values(item.text)
            if len(alternatives) > 1:
                compiled = replace(compiled, alternatives=tuple(compile_value(value) for value in alternatives))
            compiled_evidence.append(compiled)
            match = BUDGET_RE.search(item.text)
            if match and getattr(item, "source", "") != "exclusion" and math.isfinite(float(match.group("amount"))):
                budgets.append(
                    BudgetPreference(
                        mode=(match.group("mode") or "around").lower(),
                        amount=float(match.group("amount")),
                        weight=item.weight,
                    )
                )

        excluded_tokens = {
            token
            for item in compiled_evidence
            if item.source == "exclusion"
            for token in item.tokens
        }
        raw_tags = user_profile.get("preference_tags") if user_profile else None
        if isinstance(raw_tags, list):
            preference_terms = tuple(
                dict.fromkeys(
                    token for tag in raw_tags for token in terms(tag)
                    if token not in excluded_tokens
                )
            )
        else:
            preference_terms = ()
        return CompiledQuery(
            evidence=tuple(compiled_evidence),
            preference_tokens=preference_terms,
            budgets=tuple(budgets),
        )

    def __len__(self) -> int:
        return len(self._products)
