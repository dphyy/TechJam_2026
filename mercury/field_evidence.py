"""Optional rare-phrase admission and scoring from bounded raw field witnesses.

This module never changes base membership or its order. The caller decides where
to apply the returned score deltas; admission and scoring can be tested separately.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

from mercury.catalog import Catalog
from mercury.ranking import preference_evidence
from mercury.retrieval import STOPWORDS, TOKEN_RE, SparseIndex
from mercury.types import Preference, Product


Arm = Literal["off", "admission_only", "scoring_only", "admission_and_scoring"]
FIELDS = ("title", "categories", "features", "details", "description")
_NEGATIVE = re.compile(
    r"\b(?:no|not|never|without|avoid(?:ing)?|exclud(?:e|ing)|hate|dislike|"
    r"don't|dont|do not|rather than|instead of)\b", re.I,
)
_UNCERTAIN = re.compile(r"\b(?:maybe|perhaps|may|might|could|possibly|unsure|uncertain|considering)\b", re.I)
_NEGATIVE_SUFFIX = re.compile(
    r"^\s*(?:[- ]free\b|(?:is|are)\s+(?:not|unwanted)\b|(?:isn't|aren't)\b)", re.I,
)
_CLAUSE = re.compile(r"[.;!?\n]|\b(?:but|however)\b", re.I)


@dataclass(frozen=True, slots=True)
class FieldEvidenceConfig:
    max_phrases: int = 8
    max_postings: int = 64
    max_admissions: int = 32
    max_phrase_tokens: int = 16
    max_field_characters: int = 12_000
    max_document_fraction: float = 0.01
    minimum_confidence: float = 0.6
    score_cap: float = 0.08

    def __post_init__(self) -> None:
        limits = ((self.max_phrases, 32), (self.max_postings, 256),
                  (self.max_admissions, 128), (self.max_phrase_tokens, 32),
                  (self.max_field_characters, 64_000))
        if any(type(value) is not int or not 1 <= value <= ceiling for value, ceiling in limits):
            raise ValueError("phrase work limits must be positive integers within resource ceilings")
        if self.max_phrase_tokens < 2:
            raise ValueError("phrases require at least two tokens")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (
                self.max_document_fraction, self.minimum_confidence, self.score_cap)):
            raise ValueError("evidence gates must be finite numbers")
        if not 0 < self.max_document_fraction <= 1:
            raise ValueError("document fraction must be within (0, 1]")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum confidence must be within [0, 1]")
        if not 0 <= self.score_cap <= 1:
            raise ValueError("score cap must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class FieldWitness:
    preference: str
    source: str
    span: str
    start: int
    end: int
    document_frequency: int
    strength: float
    scope: str | None = None


@dataclass(slots=True)
class FieldEvidenceResult:
    candidate_ids: list[str]
    admitted_ids: list[str] = field(default_factory=list)
    score_deltas: dict[str, float] = field(default_factory=dict)
    witnesses: dict[str, FieldWitness] = field(default_factory=dict)
    diagnostics: dict[str, int] = field(default_factory=dict)


def _pattern(value: str) -> re.Pattern[str]:
    literal = r"\s+".join(re.escape(part) for part in value.strip().split())
    return re.compile(r"(?<!\w)" + literal + r"(?!\w)", re.I)


def _span_status(text: str, start: int, end: int) -> int:
    """Direct assertion = 1, negation = -1, uncertainty/cross-clause = 0."""
    if re.search(r"[.;!?\n]", text[start:end]):
        return 0
    prefix = _CLAUSE.split(text[max(0, start - 96):start])[-1]
    prefix = re.sub(r"[\"'“”‘’]", "", prefix)
    if _NEGATIVE.search(prefix) or _NEGATIVE_SUFFIX.search(text[end:end + 40]):
        return -1
    if _UNCERTAIN.search(prefix):
        return 0
    return 1


def _scope_matches(text: str, start: int, end: int, scope: str | None) -> bool:
    if scope is None:
        return True
    owner = re.escape(scope.rstrip("s")) + r"s?"
    before = text[max(0, start - 64):start]
    after = text[end:end + 64]
    links = r"(?:is|are|made|from|of|with|has|uses|in|the|must|be|not|no|never|without)"
    return bool(re.search(r"\b" + owner + r"\b(?:\s+" + links + r"){0,3}\s*[:=-]?\s*$", before, re.I)
                or re.match(r"^\s+(?:on|in|for|of)\s+(?:the\s+)?" + owner + r"\b", after, re.I)
                or re.match(r"^\s+" + owner + r"\b\s*(?:$|[.;!?\n])", after, re.I))


def _raw_assertion(product: Product, preference: Preference, pattern: re.Pattern[str],
                   character_limit: int) -> tuple[tuple[str, re.Match[str]] | None, int]:
    supported: tuple[str, re.Match[str]] | None = None
    contradicted = False
    for source in FIELDS:
        text = product.fields.get(source, "")[:character_limit]
        for match in pattern.finditer(text):
            if not _scope_matches(text, match.start(), match.end(), preference.scope):
                continue
            status = _span_status(text, match.start(), match.end())
            contradicted |= status < 0
            if status > 0 and supported is None:
                supported = (source, match)
    if contradicted:
        return None, 0 if supported is not None else -1
    return supported, int(supported is not None)


def _guarded(product: Product, preferences: list[Preference], character_limit: int) -> bool:
    """Check all active constraints; confidence gates only positive phrase queries."""
    # Truncating a conflicting assertion could turn uncertain metadata into
    # positive evidence. Oversized records conservatively receive no new signal.
    if any(len(value) > character_limit for value in product.fields.values()):
        return True
    groups: dict[tuple[str, str] | int, list[tuple[Preference, float]]] = {}
    for index, preference in enumerate(preferences):
        if preference.polarity == 0 or (preference.polarity == 1 and not preference.hard
                                        and preference.alternative_group is None):
            continue
        if preference.scope is not None:
            signal = (_raw_assertion(product, preference, _pattern(preference.value), character_limit)[1]
                      if preference.value.strip() else 0) * preference.polarity
        else:
            signal = preference_evidence(product, preference)
        if preference.polarity == -1 and signal < 0:
            return True
        if preference.polarity == 1 and preference.alternative_group is not None:
            key = (preference.attribute, preference.alternative_group)
        elif preference.hard:
            key = index
        else:
            continue
        groups.setdefault(key, []).append((preference, signal))
    return any(any(item.hard for item, _ in group) and max(signal for _, signal in group) < 0
               for group in groups.values())


def field_phrase_evidence(catalog: Catalog, index: SparseIndex, base_ids: list[str],
                          preferences: list[Preference], *, arm: Arm = "admission_only",
                          config: FieldEvidenceConfig = FieldEvidenceConfig()) -> FieldEvidenceResult:
    """Collect rare literal phrase support without scanning the catalog per turn.

    At most ``max_phrases`` index probes each fetch ``max_postings + 1`` rows.
    Overflow detects common phrases without a full posting count. Verification
    examines only these rows and bounded fields. Constraint work also scales
    with the active preference count (capped at 128); oversized inputs or raw
    records receive no new signal. Frequency uses the broader stemmed posting
    list, making rarity conservative. Unknown metadata
    never excludes a base candidate. One strongest witness supplies the entire
    score delta, so repeated fields and alternative members cannot accumulate.
    """
    if arm not in {"off", "admission_only", "scoring_only", "admission_and_scoring"}:
        raise ValueError("unknown field evidence arm")
    result = FieldEvidenceResult(list(dict.fromkeys(base_ids)), diagnostics={
        "queries": 0, "posting_rows": 0, "verified_products": 0,
        "skipped_preferences": 0, "common_phrases": 0, "guarded_products": 0,
        "resource_skips": 0,
    })
    if arm == "off":
        return result
    if len(preferences) > 128:
        result.diagnostics["resource_skips"] += 1
        return result
    active = [p for p in preferences if p.active and math.isfinite(p.confidence)]
    if any(len(p.value) > config.max_field_characters
           or (p.scope is not None and len(p.scope) > config.max_field_characters) for p in active):
        result.diagnostics["resource_skips"] += 1
        return result
    owners = {(p.attribute, p.value) for p in active if p.polarity == 1}
    phrases: list[tuple[Preference, list[str], re.Pattern[str]]] = []
    for preference in active:
        tokens = TOKEN_RE.findall(preference.value.lower())
        content = {token for token in tokens if token not in STOPWORDS and len(token) > 1}
        if (preference.polarity != 1 or preference.attribute == "budget"
                or preference.confidence < config.minimum_confidence
                or len(preference.source_text) > config.max_field_characters
                or _NEGATIVE.search(preference.value) or _UNCERTAIN.search(preference.value)
                or not 2 <= len(tokens) <= config.max_phrase_tokens or len(content) < 2
                or (preference.depends_on is not None and preference.depends_on not in owners)):
            result.diagnostics["skipped_preferences"] += 1
            continue
        pattern = _pattern(preference.value)
        if not any(_span_status(preference.source_text, match.start(), match.end()) == 1
                   and _scope_matches(preference.source_text, match.start(), match.end(), preference.scope)
                   for match in pattern.finditer(preference.source_text)):
            result.diagnostics["skipped_preferences"] += 1
            continue
        phrases.append((preference, tokens, pattern))
    phrases.sort(key=lambda item: (-len(item[1]), -item[0].confidence,
                                  item[0].attribute, item[0].value, item[0].scope or ""))
    count = len(catalog.products)
    max_frequency = min(config.max_postings, max(1, int(count * config.max_document_fraction)))
    postings: dict[str, list[str]] = {}
    guards: dict[str, bool] = {}
    seen: set[tuple[str, str | None]] = set()
    for preference, tokens, pattern in phrases:
        phrase = " ".join(tokens)
        if (phrase, preference.scope) in seen:
            continue
        seen.add((phrase, preference.scope))
        if len(seen) > config.max_phrases:
            break
        if phrase not in postings:
            postings[phrase] = index.search_phrase(phrase, max_frequency + 1)
            result.diagnostics["queries"] += 1
            result.diagnostics["posting_rows"] += len(postings[phrase])
        hits = postings[phrase]
        if not hits or len(hits) > max_frequency:
            result.diagnostics["common_phrases"] += bool(hits)
            continue
        rarity = math.log((count + 1) / (len(hits) + 1)) / math.log(count + 1)
        strength = rarity * min(1.0, preference.confidence)
        for identifier in hits:
            product = catalog.by_id.get(identifier)
            if product is None:
                continue
            result.diagnostics["verified_products"] += 1
            if identifier not in guards:
                guards[identifier] = _guarded(product, active, config.max_field_characters)
            if guards[identifier]:
                result.diagnostics["guarded_products"] += 1
                continue
            witness, _ = _raw_assertion(product, preference, pattern, config.max_field_characters)
            previous = result.witnesses.get(identifier)
            if witness is not None and strength > 0 and (previous is None or strength > previous.strength):
                source, match = witness
                result.witnesses[identifier] = FieldWitness(
                    preference.value, source, match.group(), match.start(), match.end(),
                    len(hits), strength, preference.scope,
                )
    base = set(result.candidate_ids)
    if arm in {"admission_only", "admission_and_scoring"}:
        result.admitted_ids = sorted(
            (identifier for identifier in result.witnesses if identifier not in base),
            key=lambda identifier: (-result.witnesses[identifier].strength, identifier),
        )[:config.max_admissions]
        result.candidate_ids.extend(result.admitted_ids)
    eligible = set(result.candidate_ids)
    result.witnesses = {identifier: witness for identifier, witness in result.witnesses.items()
                        if identifier in eligible}
    if arm in {"scoring_only", "admission_and_scoring"}:
        result.score_deltas = {identifier: config.score_cap * witness.strength
                               for identifier, witness in result.witnesses.items()}
    return result
