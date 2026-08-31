from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
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
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _non_negative_int(value: object) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
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
            unique_terms = tuple(dict.fromkeys(terms(item.text)))
            compiled_evidence.append(
                CompiledEvidence(
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
                                        for match in pattern.findall(item.text)
                                    }
                                )
                            ),
                        )
                        for attribute, pattern in FACET_PATTERNS.items()
                        if pattern.search(item.text)
                    ),
                    is_budget=bool(BUDGET_RE.search(item.text)),
                )
            )
            match = BUDGET_RE.search(item.text)
            if match and getattr(item, "source", "") != "exclusion":
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
