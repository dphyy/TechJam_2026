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
NEGATION_RE = re.compile(
    r"\b(?:no(?![- ]show\b)|not|without|free of|doesn't contain|does not contain)\b"
    r"[^,;.!?]*?(?=$|[,;.!?]|\b(?:but|however|whereas|while)\b|"
    r"\band\s+(?:the|its|a|an|with)\b)", re.I,
)
FREE_RE = re.compile(r"\b(?P<value>\w+)[- ]free\b", re.I)


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
    value = _negation_text(value)
    return tuple(terms(NEGATION_RE.sub(" ", FREE_RE.sub(" ", value))))


def _negation_text(value: str) -> str:
    value = re.sub(r"\bnot (?:only|just|exclusively)\b", "", value, flags=re.I)
    # Uncertain composition is neither an affirmative witness nor an explicit denial.
    return re.sub(
        r"\b(?:not (?:necessarily|always)|may not|might not|"
        r"not (?:sure|clear|known|certain)(?: whether| if)?)\b[^,;.!?]*",
        "", value, flags=re.I,
    )


def denied_terms(value: str) -> tuple[tuple[str, ...], ...]:
    """Preserve local explicit absences separately from missing catalog facts."""
    value = _negation_text(value)
    denied = [tuple(terms(match.group("value"))) for match in FREE_RE.finditer(value)]
    value = FREE_RE.sub(" ", value)
    for match in NEGATION_RE.finditer(value):
        wording = re.sub(
            r"^(?:no|not|without|free of|doesn't contain|does not contain)\b",
            "", match.group(), flags=re.I,
        )
        if tokens := tuple(terms(wording)):
            denied.append(tokens)
    return tuple(denied)


@lru_cache(maxsize=4096)
def exclusive_facet_values(value: str, attribute: str) -> frozenset[str]:
    """Recognize a complete, explicit facet restriction, never an incidental 'only'."""
    pattern = FACET_PATTERNS.get(attribute)
    if pattern is None or re.search(r"[\"'“”‘’]", value):
        return frozenset()
    body = value.strip().rstrip(".!?").strip()
    label = "(?:color|colour)" if attribute == "color" else re.escape(attribute)
    body = re.sub(rf"^{label}\s*:\s*", "", body, flags=re.I)
    if re.match(r"^(?:only|exclusively)\s+", body, re.I):
        body = re.sub(r"^(?:only|exclusively)\s+", "", body, flags=re.I)
    elif re.search(r"\s+(?:only|exclusively)$", body, re.I):
        body = re.sub(r"\s+(?:only|exclusively)$", "", body, flags=re.I)
    else:
        return frozenset()
    values = frozenset(match.group().lower() for match in pattern.finditer(body))
    remainder = pattern.sub("", body)
    if values and not re.sub(r"\b(?:and|or)\b|[\s,/&]+", "", remainder, flags=re.I):
        return values
    return frozenset()


def _evidence_value(value: str) -> tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]:
    exclusive = tuple(
        (attribute, tuple(sorted(values)))
        for attribute in FACET_ORDER
        if (values := exclusive_facet_values(value, attribute))
    )
    if exclusive:
        # The operator is semantic; it is not a catalog keyword requirement.
        value = re.sub(r"\b(?:only|exclusively)\b", "", value, flags=re.I)
        value = re.sub(r"^\s*(?:" + "|".join(FACET_ORDER) + r"|colour)\s*:\s*", "", value, flags=re.I)
    return value, exclusive


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
            # Both flattened labels and natural subject/predicate prose are supported.
            first_prefix = bool(re.fullmatch(r"\s*(?:(?:the|its|a|an)\s+)?", clause[:matches[0].start()], re.I))
            prefix_modes = []
            for index, match in enumerate(matches):
                start = matches[index - 1].end() if index else 0
                before = re.split(r"\b(?:and|but|with)\b|,", clause[start:match.start()], flags=re.I)[-1]
                local_prefix = bool(re.fullmatch(r"\s*(?:(?:the|its|a|an)\s+)?", before, re.I))
                end = matches[index + 1].start() if index + 1 < len(matches) else len(clause)
                after = clause[match.end():end].strip()
                explicit_prefix = bool(re.match(r"(?::|(?:is|are|was|were|material|color|colour)\b)", after, re.I))
                has_value = bool(terms(after))
                prefix_modes.append(explicit_prefix or ((first_prefix or local_prefix) and has_value))
            for index, match in enumerate(matches):
                owner = match.group().lower().rstrip("s")
                if prefix_modes[index]:
                    end = matches[index + 1].start() if index + 1 < len(matches) else len(clause)
                    value = clause[match.end():end].strip(" ,:")
                    if index + 1 < len(matches) and not prefix_modes[index + 1]:
                        # A mixed record: 'lining cotton and leather upper'.
                        value = re.split(r"\b(?:and|but|with)\b|,", value, flags=re.I)[0]
                    value = re.sub(r"\b(?:and|but|with)\s*(?:(?:the|its|a|an)\s*)?$", "", value, flags=re.I)
                    value = re.sub(r"^(?:is|are|was|were)\s+", "", value, flags=re.I).strip()
                    rendered = f"{match.group()} {value}"
                else:
                    start = matches[index - 1].end() if index else 0
                    value = clause[start:match.start()].strip(" ,:")
                    value = re.split(r"\b(?:and|but|with)\b|,", value, flags=re.I)[-1].strip()
                    value = re.sub(r"^(?:the|its|a|an)\s+", "", value, flags=re.I)
                    rendered = f"{value} {match.group()}"
                if value:
                    slots = scoped.setdefault(owner, [""] * len(FIELD_ORDER))
                    slots[field_index] += " " + rendered
    return MappingProxyType({owner: tuple(values) for owner, values in scoped.items()})


class EvidenceLike(Protocol):
    text: str
    weight: float


def terms(value: object, *, min_length: int = 1) -> list[str]:
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
    denied_sequences: tuple[tuple[str, ...], ...] = ()


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
    exclusive_facets: tuple[tuple[str, tuple[str, ...]], ...] = ()
    literal_absence: bool = False


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
    if evidence.scope:
        values = product.component_fields.get(evidence.scope, ("",) * len(FIELD_ORDER))
        sequences = tuple(tuple(terms(value)) if evidence.literal_absence else affirmed_terms(value)
                          for value in values)
        denied = tuple(sequence for value in values for sequence in denied_terms(value))
    else:
        sequences = product.field_sequences if evidence.literal_absence else product.affirmed_sequences
        if not sequences or sequences == product.field_sequences:
            return product
        denied = product.denied_sequences
    weights: dict[str, float] = {}
    for name, sequence in zip(FIELD_ORDER, sequences):
        for token in sequence:
            weights[token] = max(weights.get(token, 0.0), FIELD_WEIGHTS[name])
    return replace(product, token_weights=MappingProxyType(weights),
                   normalized_text=FIELD_SEPARATOR.join(" ".join(sequence) for sequence in sequences),
                   field_sequences=sequences,
                   affirmed_sequences=sequences,
                   denied_sequences=denied,
                   feature_tokens=frozenset(sequences[FIELD_ORDER.index("features")]))


def _sequence_match(tokens: tuple[str, ...], sequences: tuple[tuple[str, ...], ...]) -> bool:
    return bool(tokens) and any(
        sequence[start:start + len(tokens)] == tokens
        for sequence in sequences
        for start in range(len(sequence) - len(tokens) + 1)
    )


def evidence_contradiction(product: ProductFeatures, evidence: CompiledEvidence) -> bool:
    """Require explicit adverse evidence; an unmentioned attribute is unknown."""
    return _view_contradiction(evidence_product(product, evidence), evidence)


def _view_contradiction(view: ProductFeatures, evidence: CompiledEvidence) -> bool:
    if not evidence.literal_absence:
        wanted = tuple(terms(component_value(evidence.normalized_query))) if evidence.scope else evidence.tokens
        denied = _sequence_match(wanted, view.denied_sequences)
        # A short material denial also contradicts a longer requested description.
        for _, expected in evidence.facets:
            denied |= any(_sequence_match(tuple(terms(value)), view.denied_sequences)
                          and not _sequence_match(tuple(terms(value)), view.affirmed_sequences)
                          for value in expected)
        if denied and not _sequence_match(wanted, view.affirmed_sequences):
            return True
    for attribute, allowed in evidence.exclusive_facets:
        actual = set(FACET_PATTERNS[attribute].findall(view.normalized_text))
        if actual - set(allowed):
            return True
    if evidence.scope:
        for attribute, expected in evidence.facets:
            actual = set(FACET_PATTERNS[attribute].findall(view.normalized_text))
            if actual and actual.isdisjoint(expected):
                return True
    return False


@lru_cache(maxsize=4096)
def _compile_hard_evidence(value: str) -> CompiledEvidence:
    value, exclusive = _evidence_value(value)
    scope = component_scope(value)
    searchable = component_value(value) if scope else value
    tokens = tuple(dict.fromkeys(terms(searchable)))
    return CompiledEvidence(
        tokens=tokens, normalized_query=" ".join(tokens), weight=1.0,
        source="hard_constraint", attribute=None,
        facets=tuple((name, tuple(sorted(set(pattern.findall(searchable.lower())))))
                     for name, pattern in FACET_PATTERNS.items() if pattern.search(searchable)),
        is_budget=False, scope=scope, exclusive_facets=exclusive,
        literal_absence=bool(denied_terms(value)),
    )


def hard_evidence_match(product: ProductFeatures, value: str) -> bool:
    evidence = _compile_hard_evidence(value)
    for branch in alternative_values(value):
        item = _compile_hard_evidence(branch)
        if evidence.exclusive_facets:
            item = replace(item, exclusive_facets=evidence.exclusive_facets)
        view = evidence_product(product, item)
        if _sequence_match(item.tokens, view.field_sequences) and not _view_contradiction(view, item):
            return True
    return False


def resolve_query(product: ProductFeatures, query: CompiledQuery) -> CompiledQuery:
    """Count an OR group once, using its strongest available lexical witness."""
    if not any(item.alternatives for item in query.evidence):
        return query
    def support(item: CompiledEvidence) -> tuple[bool, bool, float, float]:
        view = evidence_product(product, item)
        compatible = item.source == "exclusion" or not _view_contradiction(view, item)
        return (_sequence_match(item.tokens, view.field_sequences) and compatible,
                compatible,
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
            denied_sequences=tuple(sequence for name in FIELD_ORDER
                                   for sequence in denied_terms(fields.get(name, ""))),
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
                value, exclusive = _evidence_value(value)
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
                    exclusive_facets=exclusive,
                    literal_absence=bool(denied_terms(value)),
                )
            compiled = compile_value(item.text)
            alternatives = alternative_values(item.text)
            if len(alternatives) > 1:
                branches = tuple(compile_value(value) for value in alternatives)
                if compiled.exclusive_facets:
                    branches = tuple(replace(branch, exclusive_facets=compiled.exclusive_facets) for branch in branches)
                compiled = replace(compiled, alternatives=branches)
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
