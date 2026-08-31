from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum

from .budgets import NEGATIVE_SPEND, parse_budgets, separate_budget
from .categories import asserted_category
from .memory import LongTermUserProfile
from .feedback import preference_content
from .product_features import (
    FACET_PATTERNS, alternative_values, component_scope, component_value, exclusive_facet_values,
)


OVERRIDE_RE = re.compile(
    r"\b(actually|instead|changed my mind|ignore|no longer|rather than)\b", re.I
)
CORRECTION_ACTION = r"(?:make that|change(?: that| it)? to)"
CORRECTION_PREFIX_RE = re.compile(
    rf"^(?:correction\s*[:,]\s*(?:{CORRECTION_ACTION}\s+)?|(?:please\s+)?{CORRECTION_ACTION}\s+|"
    r"(?:let me\s+)?correct that\s*[:,]\s*)(.+)$", re.I,
)
CORRECTION_MENTION_RE = re.compile(rf"\b(?:correction|{CORRECTION_ACTION}|correct that)\b", re.I)
QUOTED_TEXT_RE = re.compile(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\x27[^\x27\n]+\x27(?!\w)')
NO_PREFERENCE_RE = re.compile(
    r"\b(?:do not|don't|dont|no)\s+(?:have\s+)?(?:an?\s+)?(?:additional\s+)?"
    r"(?:(?:material|colo[u]?r|size|style|fit|budget|lining|upper)\s+){0,2}preference\b",
    re.I,
)
LOOKING_FOR_RE = re.compile(r"\blooking for\s+(.+?)(?:[,.!?;]|$)", re.I)
NEED_RE = re.compile(r"\bwhat i need is\s*:\s*(.+)$", re.I)
REQUIREMENT_RE = re.compile(r"\bkey requirement is\s*:\s*(.+)$", re.I)
MATTERS_RE = re.compile(r"\bwhat matters is\s*:\s*(.+)$", re.I)
NEGATIVE_PREFERENCE_RE = re.compile(r"\b(?:do not|don't|dont|no longer)\s+(?:want|like|need|prefer)\s+(.+)$", re.I)
NEGATIVE_CLAUSE_RE = re.compile(
    r"^(?:(?:i\s+)?(?:do not|don't|dont|no longer)\s+(?:want|like|need|prefer)\s+|"
    r"(?:(?:i\s+)?(?:want|need|prefer)\s+)?(?:no|not|without|avoid|anything but)\s+)(.+)$", re.I,
)
UNCERTAIN_RE = re.compile(r"^(?:i(?:'m| am)?\s+)?(?:maybe|perhaps|not sure|unsure|undecided)\b", re.I)
DURABLE_PREFERENCE_RE = re.compile(r"\b(?:always|generally|usually)\b", re.I)
EXPLORING_RE = re.compile(r"\bi(?:'m| am) still exploring\b", re.I)
QUESTION_BOILERPLATE_RE = re.compile(
    r"\b(?:options are not quite right|ask me about|closest matches (?:differ|vary)|"
    r"which .+ best fits what you need)\b",
    re.I,
)
GENERIC_PREFIX_RE = re.compile(
    r"^(?:(?:for that|actually),?\s+)?(?:i\s+)?(?:would\s+)?"
    r"(?:prefer|need|want|am looking for|(?:'m|am) looking for)\s+",
    re.I,
)
GENERIC_CATEGORY_ONLY = {"shoe", "shoes", "jewelry", "jewellery"}
MATERIAL_TERMS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "linen", "suede", "denim", "fabric",
}
COLOR_TERMS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "navy",
}


class PreferenceOperation(str, Enum):
    """How a newly observed value changes the active set for its attribute."""

    REPLACE = "replace"
    APPEND = "append"
    EXCLUDE = "exclude"


@dataclass(frozen=True)
class Evidence:
    text: str
    weight: float
    source: str
    turn: int
    attribute: str | None = None
    operation: PreferenceOperation = PreferenceOperation.APPEND
    raw_chunk: str | None = None
    derivation: str | None = None


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.")


def _split_constraints(value: str) -> list[str]:
    return [cleaned for part in value.split(";") if (cleaned := _clean(part))]


def _derived(item: Evidence, value: str, derivation: str) -> Evidence:
    raw_chunk = item.raw_chunk if item.derivation is not None else item.text
    return replace(item, text=value, raw_chunk=raw_chunk, derivation=derivation)


def _unasserted_correction(value: str) -> bool:
    if not (CORRECTION_MENTION_RE.search(value) or re.search(r"\b(?:looking for|want|need|prefer)\b", value, re.I)):
        return False
    return bool(QUOTED_TEXT_RE.fullmatch(value) or re.match(
        r"^(?:if|suppose|supposing|imagine|hypothetically|maybe|perhaps|"
        r"i (?:might|may|would)|i(?:'m| am)? not|(?:please )?(?:do not|don't|dont|no)|"
        r"(?:the )?(?:label|description|listing|package|tag) (?:says|reads|states))\b",
        value, re.I,
    ))


def _correction_payload(value: str) -> tuple[list[str], list[str]]:
    """Remove control clauses while retaining each replacement phrase verbatim."""
    spans = [match.span() for match in QUOTED_TEXT_RE.finditer(value)]
    parts, start = [], 0
    for boundary in re.finditer(r";|,?\s+(?:but|and)\s+(?=keep\b)", value, re.I):
        if any(left <= boundary.start() < right for left, right in spans):
            continue
        parts.append(_clean(value[start:boundary.start()]))
        start = boundary.end()
    parts.append(_clean(value[start:]))
    replacements, options = [], []
    for part in parts:
        if match := re.fullmatch(r"keep\s+(?:the\s+)?(.+?)\s+as\s+(?:(?:an?|another)\s+)?option(?:\s+too)?", part, re.I):
            options.append(match.group(1))
        elif (part and not re.match(r"^keep\b", part, re.I)
              and not UNCERTAIN_RE.match(part) and not _unasserted_correction(part)):
            replacements.append(part)
    return replacements, options


def _with_kept_options(values: list[str], options: list[str]) -> tuple[list[str], dict[str, str]]:
    """Broaden one unambiguous facet without weakening the other requirements."""
    values = list(values)
    derived = {}
    for option in options:
        facets = [name for name, pattern in FACET_PATTERNS.items() if pattern.search(option)]
        if len(facets) != 1:
            continue
        attribute = facets[0]
        pattern = FACET_PATTERNS[attribute]
        owner = component_scope(option)
        matching = [index for index, value in enumerate(values)
                    if pattern.search(value) and (owner is None or component_scope(value) == owner)
                    and len(alternative_values(value)) == 1]
        if len(matching) != 1:
            continue
        index = matching[0]
        value = values[index]
        matches = list(pattern.finditer(value))
        if len(matches) != 1:
            continue
        owner = component_scope(value)
        option_value = component_value(option) if component_scope(option) else option
        group = f"{matches[0].group()} or {option_value}"
        if owner:
            group = f"{owner}: {group}"
        remainder = _clean(pattern.sub("", value))
        remainder = _clean(re.sub(r"^(?:and|or)\b|\b(?:and|or)$", "", remainder, flags=re.I))
        remainder_value = component_value(remainder) if owner else remainder
        remainder_value = re.sub(rf"^(?:{attribute}|colour)\s*:\s*", "", remainder_value, flags=re.I)
        replacement = [group]
        derived[group] = "kept_facet_alternative"
        if re.search(r"[a-z0-9]", remainder_value, re.I):
            replacement.append(remainder)
            derived[remainder] = "remaining_replacement"
        values[index:index + 1] = replacement
    return values, derived


def _infer_attribute(value: str, fallback: str | None = None) -> str:
    """Infer the facet named by a value, preserving a specific asked facet."""
    lowered = value.casefold()
    tokens = set(re.findall(r"[a-z0-9]+", lowered))
    label = re.match(r"^([a-z][a-z ]{0,48})\s*:", lowered)
    if label and not component_scope(value):
        name = "_".join(label.group(1).split())
        aliases = {"fabric_type": "material", "fabric": "material", "colour": "color", "price": "budget"}
        if name not in {"feature", "features", "requirement"}:
            return aliases.get(name, name)
    if "budget" in tokens or parse_budgets(value):
        return "budget"
    if FACET_PATTERNS["material"].search(value) or "material" in tokens:
        return "material"
    if tokens & COLOR_TERMS or "color" in tokens or "colour" in tokens:
        return "color"
    if tokens & {"size", "sizing", "width", "wide", "narrow"}:
        return "size"
    if tokens & {"department", "style", "fit", "sleeve", "neck", "casual", "formal"}:
        return "style"
    if tokens & {"hiking", "running", "gym", "winter", "outdoor", "work"}:
        return "use_case"
    if fallback and fallback != "other":
        return fallback
    return "feature" if fallback is not None else "other"


@dataclass
class SessionState:
    user_profile: dict
    evidence: list[Evidence] = field(default_factory=list)
    asked_attributes: list[str] = field(default_factory=list)
    no_preference_attributes: set[str] = field(default_factory=set)
    messages: list[str] = field(default_factory=list)
    category_text: str = ""
    last_turn: int = 0
    long_term_profile: LongTermUserProfile | None = None

    def observe(self, message: str, turn: int, *, category_names: frozenset[tuple[str, ...]] = frozenset()) -> None:
        """Convert the latest customer message into weighted preference evidence."""
        if turn <= self.last_turn:
            return
        self.last_turn = turn
        raw_message = str(message)
        message = _clean(raw_message)
        self.messages.append(message)
        message = _clean(preference_content(message))
        if not message:
            return

        if _unasserted_correction(message):
            self._observe_special_clauses(message, turn, source="clarification",
                                          weight=2.5 if turn > 1 else 2.0,
                                          allow_control_prefixes=False)
            return
        correction = CORRECTION_PREFIX_RE.match(message)
        correction_values, correction_derivations = [], {}
        if correction:
            correction_values, options = _correction_payload(correction.group(1))
            if not LOOKING_FOR_RE.search(correction.group(1)):
                correction_values, correction_derivations = _with_kept_options(correction_values, options)
            if not correction_values:
                return
            message = "; ".join(correction_values)
        is_override = bool(correction or OVERRIDE_RE.search(message))
        inferred_category = asserted_category(message, category_names, correction=bool(correction),
                                              answering=self._answer_attribute() == "category")
        if inferred_category:
            name, remainder = inferred_category
            is_override |= bool(self.category_text and self.category_text != name)
            message = f"I'm looking for {name}. {remainder}"
        if is_override and re.search(r"ignore my earlier preference", message, re.I):
            # "Ignore my earlier preference" explicitly retires the opening
            # preference. The replacement operation below additionally clears
            # clarification evidence for each attribute named by the new value.
            self.evidence = [
                item for item in self.evidence
                if item.source != "initial_preference"
            ]

        category_match = LOOKING_FOR_RE.search(message)
        if category_match and (not self.category_text or is_override):
            next_category = re.sub(r"\s+instead$", "", category_match.group(1), flags=re.I)
            self.category_text = _clean(next_category)
            if self.category_text:
                self._apply_values(
                    [self.category_text], "category", 1.4, "category", turn,
                    PreferenceOperation.REPLACE,
                )

        special_message = message
        source, weight = (("override", 6.0) if correction else ("clarification", 2.5 if turn > 1 else 2.0))
        if category_match and not any(pattern.search(message) for pattern in (NEED_RE, REQUIREMENT_RE, MATTERS_RE)):
            special_message = _clean(message[category_match.end():])
            source, weight = "initial_preference", 1.8
        if self._observe_special_clauses(special_message, turn, source=source, weight=weight):
            self._annotate_derived(turn, raw_message, correction_derivations)
            return

        if correction and not category_match:
            self._apply_grouped_values(correction_values, 6.0, "override", turn,
                                       PreferenceOperation.REPLACE)
            self._annotate_derived(turn, raw_message, correction_derivations)
            return

        match = NEED_RE.search(message)
        if match:
            self._apply_grouped_values(
                _split_constraints(match.group(1)), 6.0, "override", turn,
                PreferenceOperation.REPLACE,
            )
            return

        match = REQUIREMENT_RE.search(message)
        if match:
            self._apply_grouped_values(
                _split_constraints(match.group(1)), 3.8, "hard_constraint", turn,
                PreferenceOperation.APPEND,
            )
            return

        match = MATTERS_RE.search(message)
        if match:
            self._apply_grouped_values(
                _split_constraints(match.group(1)), 3.3, "clarification", turn,
                PreferenceOperation.APPEND,
            )
            return

        if category_match:
            remainder = message[category_match.end():]
            remainder = re.sub(
                r"^(?:\s*but\s+)?i(?:'m| am) still exploring$", "", remainder, flags=re.I
            )
            remainder = _clean(remainder)
            if remainder:
                self._apply_grouped_values(
                    _split_constraints(remainder), 6.0 if is_override else 1.8,
                    "override" if is_override else "initial_preference", turn,
                    PreferenceOperation.REPLACE if is_override else PreferenceOperation.APPEND,
                )
            return

        if not re.search(r"options are not quite right|ask me about", message, re.I):
            value = message
            if is_override:
                value = re.sub(r"^(?:actually|instead|i changed my mind)[,:]?\s*|\s+instead$", "", value, flags=re.I)
            self._apply_grouped_values(
                [value], 2.5 if turn > 1 else 2.0, "clarification", turn,
                PreferenceOperation.REPLACE if is_override else PreferenceOperation.APPEND,
            )

    def _observe_special_clauses(self, message: str, turn: int, *, source: str, weight: float,
                                 allow_control_prefixes: bool = True) -> bool:
        """Handle local polarity and retractions without rewriting ordinary positives."""
        content = message
        for pattern, candidate_source, candidate_weight in (
            (NEED_RE, "override", 6.0), (REQUIREMENT_RE, "hard_constraint", 3.8),
            (MATTERS_RE, "clarification", 3.3),
        ):
            if allow_control_prefixes and (match := pattern.search(content)):
                content, source, weight = match.group(1), candidate_source, candidate_weight
                break
        contrast = re.match(r"(.+?)\s+(?P<link>rather than|instead of)\s+(.+)$", content, re.I)
        if (allow_control_prefixes and contrast and not any(
                quoted.start() <= contrast.start("link") < quoted.end()
                for quoted in QUOTED_TEXT_RE.finditer(content))):
            content = f"not {contrast.group(3)}; {contrast.group(1)}"
        reported = re.compile(r"\b(?:label|description|listing|package|tag)\s+(?:says|reads|states)\b", re.I)
        quote = QUOTED_TEXT_RE
        quoted_spans = [match.span() for match in quote.finditer(content)]
        boundaries = re.finditer(
            r"\s*(?:;|\bbut\b|\band\s+(?=(?:no|not|without)\b)|"
            r",\s*(?=(?:no|not|without|maybe|perhaps)\b)|"
            r"[.!?]\s+(?=(?:i\s+)?(?:have\s+no|do not|don't|dont|no|not|without|avoid|anything but)\b)|"
            r"(?:,\s*(?:and\s+)?|\band\s+|[.!?]\s+)(?=i\s+(?:want|need|prefer|require)\b))\s*",
            content, re.I,
        )
        clauses, start = [], 0
        for boundary in boundaries:
            if any(left <= boundary.start() < right for left, right in quoted_spans):
                continue
            clauses.append(_clean(content[start:boundary.start()]))
            start = boundary.end()
        clauses.append(_clean(content[start:]))
        explicit_requirement = source in {"hard_constraint", "override"}
        if not any(NO_PREFERENCE_RE.search(quote.sub("", clause)) or NEGATIVE_CLAUSE_RE.match(clause)
                   or not explicit_requirement and (UNCERTAIN_RE.match(clause) or reported.search(clause)
                                                     or _unasserted_correction(clause))
                   for clause in clauses):
            return False
        for raw in clauses:
            clause = _clean(raw)
            if not clause:
                continue
            if not explicit_requirement and (UNCERTAIN_RE.match(clause) or reported.search(clause)
                                             or _unasserted_correction(clause)):
                continue
            if neutral := NO_PREFERENCE_RE.search(quote.sub("", clause)):
                attribute = _infer_attribute(clause)
                if attribute == "other":
                    named = re.search(r"\bpreference\s+(?:for|about)\s+(other|features?)\b", clause, re.I)
                    attribute = (named.group(1).casefold().removesuffix("s") if named
                                 else self._answer_attribute() or "other")
                owner = component_scope(clause)
                # Declining another requirement does not withdraw an existing one.
                if not re.search(r"\badditional\b", neutral.group(0), re.I):
                    self._retire(attribute, owner)
                if owner is None:
                    self.no_preference_attributes.add(attribute)
                continue
            negative = NEGATIVE_CLAUSE_RE.match(clause)
            if parse_budgets(clause) and re.match(r"^(?:(?:no|not)\s+(?:more|less|under|below|over|above)\b|" + NEGATIVE_SPEND + ")", clause, re.I):
                self._apply_values([clause], "budget", weight, source, turn,
                                   PreferenceOperation.REPLACE if source == "override" else PreferenceOperation.APPEND)
                continue
            if negative:
                value = _clean(re.sub(r",?\s+please$", "", negative.group(1), flags=re.I))
                self._apply_values([value], _infer_attribute(value, self._answer_attribute()),
                                   3.8, "exclusion", turn, PreferenceOperation.EXCLUDE)
                continue
            if re.search(r"options are not quite right|ask me about", clause, re.I):
                continue
            if NO_PREFERENCE_RE.search(message) and not re.search(
                r"\b(?:want|need|prefer|require)\b", clause, re.I
            ) and _infer_attribute(clause) == "other":
                continue
            operation = PreferenceOperation.REPLACE if source == "override" else PreferenceOperation.APPEND
            self._apply_grouped_values([clause], weight, source, turn, operation)
        return True

    def _retire(self, attribute: str | None, owner: str | None,
                replacement_values: list[str] | None = None) -> None:
        retained: list[Evidence] = []
        for item in self.evidence:
            if item.source == "category":
                if attribute != "category":
                    retained.append(item)
                continue
            if component_scope(item.text) != owner:
                retained.append(item)
                continue
            inferred = _infer_attribute(item.text, item.attribute)
            pattern = FACET_PATTERNS.get(attribute or "")
            if inferred != attribute and not (pattern and pattern.search(item.text)):
                retained.append(item)
                continue
            if replacement_values is not None:
                if attribute in {"feature", "other", None}:
                    retained.append(item)
                    continue
                if pattern:
                    old_values = {value.lower() for value in pattern.findall(item.text)}
                    new_values = {value.lower() for value in pattern.findall(" ".join(replacement_values))}
                    overlap = bool(old_values & new_values)
                    exclusive = frozenset().union(*(exclusive_facet_values(value, attribute)
                                                    for value in replacement_values))
                    narrowed_values = bool(exclusive and old_values - exclusive)
                    changed_percentage = False
                    if attribute == "material":
                        def percentages(text: str) -> dict[str, float]:
                            values = {}
                            for match in pattern.finditer(text):
                                amount = re.search(r"(\d+(?:\.\d+)?)\s*%\s*$", text[:match.start()])
                                if amount:
                                    values[match.group().lower()] = float(amount.group(1))
                            return values
                        old_percentages = percentages(item.text)
                        new_percentages = percentages(" ".join(replacement_values))
                        changed_percentage = any(old_percentages[value] != new_percentages[value]
                                                 for value in old_percentages.keys() & new_percentages.keys())
                    if (item.source == "exclusion" and old_values and new_values and not overlap
                            or item.source != "exclusion" and overlap and not changed_percentage and not narrowed_values
                            and len(alternative_values(item.text)) == 1
                            and all(len(alternative_values(value)) == 1 for value in replacement_values)):
                        retained.append(item)
                        continue
            other_facets = [name for name, candidate in FACET_PATTERNS.items()
                            if name != attribute and candidate.search(item.text)]
            if pattern and other_facets:
                remainder = _clean(pattern.sub("", item.text))
                remainder = _clean(re.sub(r"^(?:and|or)\b|\b(?:and|or)$", "", remainder, flags=re.I))
                if remainder:
                    retained.append(_derived(item, remainder, "facet_removed"))
        self.evidence = retained

    def _apply_grouped_values(
        self,
        values: list[str],
        weight: float,
        source: str,
        turn: int,
        operation: PreferenceOperation,
    ) -> None:
        """Apply one operation to each affected attribute as a value set."""
        values = [part for value in values for part in separate_budget(value)]
        grouped: dict[str | None, list[str]] = {}
        replacements: dict[tuple[str | None, str | None], list[str]] = {}
        fallback = self._answer_attribute()
        for value in values:
            attribute = (
                _infer_attribute(value, fallback)
                if operation == PreferenceOperation.REPLACE
                else fallback
            )
            grouped.setdefault(attribute, []).append(value)
            if operation == PreferenceOperation.REPLACE:
                owner = component_scope(value)
                attributes = {attribute}
                if attribute in {*FACET_PATTERNS, "budget", "feature", "other", None} or owner:
                    attributes.update(name for name, pattern in FACET_PATTERNS.items() if pattern.search(value))
                for affected in sorted(attributes, key=lambda item: item or ""):
                    replacements.setdefault((affected, owner), []).append(value)
        # Retire against the complete replacement set before adding any new
        # chunk, so a later facet cannot accidentally prune an earlier new one.
        for (attribute, owner), replacement_values in replacements.items():
            prior = list(self.evidence)
            self._retire(attribute, owner, replacement_values)
            if self.long_term_profile is not None and owner is None:
                remaining = set(self.evidence)
                for old in prior:
                    if old not in remaining and old.source not in {"category", "exclusion"}:
                        self.long_term_profile.reject(attribute or "other", old.text)
        for attribute, attribute_values in grouped.items():
            self._apply_values(
                attribute_values, attribute, weight, source, turn, operation,
                retire_existing=operation != PreferenceOperation.REPLACE,
            )

    def _apply_values(
        self,
        values: list[str],
        attribute: str | None,
        weight: float,
        source: str,
        turn: int,
        operation: PreferenceOperation,
        *,
        retire_existing: bool = True,
    ) -> None:
        """Replace, append to, or exclude from one attribute's active set."""
        cleaned = [value for raw in values if (value := _clean(raw))]
        if not cleaned:
            return
        if operation == PreferenceOperation.REPLACE and retire_existing:
            for owner in {component_scope(value) for value in cleaned}:
                self._retire(attribute, owner, [value for value in cleaned if component_scope(value) == owner])
        elif operation == PreferenceOperation.EXCLUDE:
            for value in cleaned:
                retained: list[Evidence] = []
                for item in self.evidence:
                    if item.source in {"category", "exclusion"} or not self._conflicts_with_exclusion(item, value, attribute):
                        retained.append(item)
                        continue
                    branches = alternative_values(item.text)
                    if len(branches) > 1:
                        remaining = [branch for branch in branches
                                     if not self._conflicts_with_exclusion(replace(item, text=branch), value, attribute)]
                        if remaining:
                            retained.append(_derived(item, " or ".join(remaining), "excluded_alternative_removed"))
                    else:
                        phrase = component_value(value) if component_scope(value) else value
                        remainder = _clean(re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", "", item.text, flags=re.I))
                        if remainder != item.text and any(pattern.search(remainder) for pattern in FACET_PATTERNS.values()):
                            retained.append(_derived(item, remainder, "excluded_phrase_removed"))
                self.evidence = retained

        for value in cleaned:
            self._add(value, weight, source, turn, attribute, operation)
            if self.long_term_profile and operation == PreferenceOperation.EXCLUDE:
                self.long_term_profile.reject(attribute, value)

    @staticmethod
    def _conflicts_with_exclusion(
        item: Evidence, value: str, attribute: str
    ) -> bool:
        """Return whether a rejection supersedes an earlier positive preference."""
        attributes_overlap = (
            item.attribute in {None, "other"}
            or attribute == "other"
            or item.attribute == attribute
        )
        if not attributes_overlap:
            return False
        owner = component_scope(value)
        if owner is not None and component_scope(item.text) != owner:
            return False
        excluded = (component_value(value) if owner else value).casefold()
        positive = (component_value(item.text) if component_scope(item.text) else item.text).casefold()
        return excluded in positive or positive in excluded

    def _answer_attribute(self) -> str | None:
        return self.asked_attributes[-1] if self.asked_attributes else None

    def _annotate_derived(self, turn: int, raw_chunk: str, derivations: dict[str, str]) -> None:
        if derivations:
            self.evidence = [replace(item, raw_chunk=raw_chunk, derivation=derivations[item.text])
                             if item.turn == turn and item.text in derivations else item
                             for item in self.evidence]

    def forget_provenance(self) -> None:
        """Keep active semantics and routing history, but discard source chunks."""
        self.evidence = [replace(item, raw_chunk=None) if item.raw_chunk is not None else item
                         for item in self.evidence]

    def _add(
        self,
        text: str,
        weight: float,
        source: str,
        turn: int,
        attribute: str | None = None,
        operation: PreferenceOperation = PreferenceOperation.APPEND,
    ) -> None:
        text = _clean(text)
        if not text:
            return
        key = text.casefold()
        if any(item.text.casefold() == key and item.source == source for item in self.evidence):
            return
        self.evidence.append(
            Evidence(
                text=text, weight=weight, source=source, turn=turn,
                attribute=attribute, operation=operation,
            )
        )
        if self.long_term_profile:
            self.long_term_profile.observe(
                attribute or "other", text, turn,
                durable=bool(DURABLE_PREFERENCE_RE.search(text)),
                replacement=operation == PreferenceOperation.REPLACE,
            )

    def record_question(self, attribute: str) -> None:
        self.asked_attributes.append(attribute)

    @property
    def latest_evidence(self) -> Evidence | None:
        return self.evidence[-1] if self.evidence else None

    def semantic_query(self) -> str | None:
        """Return one concise intent query, or None when intent is only generic."""
        category = _clean(self.category_text)
        required: list[str] = []
        intended_use: list[str] = []
        seen: set[str] = {category.casefold()} if category else set()

        for item in self.evidence:
            if item.source == "category":
                if not category:
                    category = _clean(item.text)
                    if category:
                        seen.add(category.casefold())
                continue
            if item.source == "exclusion":
                continue
            value = _clean(GENERIC_PREFIX_RE.sub("", item.text))
            if (
                not value
                or NO_PREFERENCE_RE.search(value)
                or EXPLORING_RE.search(value)
                or QUESTION_BOILERPLATE_RE.search(value)
                or (not category and value.casefold() in GENERIC_CATEGORY_ONLY)
                or value.casefold() in seen
            ):
                continue
            seen.add(value.casefold())
            target = intended_use if item.attribute == "use_case" else required
            target.append(value)

        # A category alone describes a catalog aisle, not a semantic target.
        if not required and not intended_use:
            return None

        lines: list[str] = []
        if category:
            lines.append(f"Product category: {category}")
        if required:
            lines.append(f"Required features: {'; '.join(required)}")
        if intended_use:
            lines.append(f"Intended use: {'; '.join(intended_use)}")
        return "\n".join(lines) or None
