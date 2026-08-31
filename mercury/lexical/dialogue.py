from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum

from .memory import LongTermUserProfile
from .product_features import FACET_PATTERNS, alternative_values, component_scope, component_value


OVERRIDE_RE = re.compile(
    r"\b(actually|instead|changed my mind|ignore|no longer|rather than)\b", re.I
)
NO_PREFERENCE_RE = re.compile(
    r"\b(?:do not|don't|dont|no)\s+(?:have\s+)?(?:an?\s+)?(?:additional\s+)?"
    r"(?:(?:material|colo[u]?r|size|style|fit|budget|lining|upper)\s+){0,2}preference\b",
    re.I,
)
LOOKING_FOR_RE = re.compile(r"\blooking for\s+(.+?)(?:[,.]|$)", re.I)
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


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.")


def _split_constraints(value: str) -> list[str]:
    return [cleaned for part in value.split(";") if (cleaned := _clean(part))]


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
    if "budget" in tokens or re.search(r"(?:\$|<=|under|below|around)\s*\$?\d", lowered):
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

    def observe(self, message: str, turn: int) -> None:
        """Convert the latest customer message into weighted preference evidence."""
        if turn <= self.last_turn:
            return
        self.last_turn = turn
        message = _clean(str(message))
        self.messages.append(message)

        is_override = bool(OVERRIDE_RE.search(message))
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
        source, weight = "clarification", 2.5 if turn > 1 else 2.0
        if category_match and not any(pattern.search(message) for pattern in (NEED_RE, REQUIREMENT_RE, MATTERS_RE)):
            special_message = _clean(message[category_match.end():])
            source, weight = "initial_preference", 1.8
        if self._observe_special_clauses(special_message, turn, source=source, weight=weight):
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
                    [remainder], 1.8, "initial_preference", turn,
                    PreferenceOperation.APPEND,
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

    def _observe_special_clauses(self, message: str, turn: int, *, source: str, weight: float) -> bool:
        """Handle local polarity and retractions without rewriting ordinary positives."""
        content = message
        for pattern, candidate_source, candidate_weight in (
            (NEED_RE, "override", 6.0), (REQUIREMENT_RE, "hard_constraint", 3.8),
            (MATTERS_RE, "clarification", 3.3),
        ):
            if match := pattern.search(content):
                content, source, weight = match.group(1), candidate_source, candidate_weight
                break
        contrast = re.match(r"(.+?)\s+(?:rather than|instead of)\s+(.+)$", content, re.I)
        if contrast:
            content = f"not {contrast.group(2)}; {contrast.group(1)}"
        clauses = [_clean(value) for value in re.split(
            r"\s*(?:;|\bbut\b|\band\s+(?=(?:no|not|without)\b)|,\s*(?=(?:no|not|without|maybe|perhaps)\b))\s*",
            content, flags=re.I,
        )]
        reported = re.compile(r"\b(?:label|description|listing|package|tag)\s+(?:says|reads|states)\b", re.I)
        quote = re.compile(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\x27[^\x27\n]+\x27(?!\w)')
        explicit_requirement = source in {"hard_constraint", "override"}
        if not any(NO_PREFERENCE_RE.search(quote.sub("", clause)) or NEGATIVE_CLAUSE_RE.match(clause)
                   or not explicit_requirement and (UNCERTAIN_RE.match(clause) or reported.search(clause))
                   for clause in clauses):
            return False
        for raw in clauses:
            clause = _clean(raw)
            if not clause:
                continue
            if not explicit_requirement and (UNCERTAIN_RE.match(clause) or reported.search(clause)):
                continue
            if NO_PREFERENCE_RE.search(quote.sub("", clause)):
                attribute = _infer_attribute(clause, self._answer_attribute())
                owner = component_scope(clause)
                self._retire(attribute, owner)
                if owner is None:
                    self.no_preference_attributes.add(attribute)
                continue
            negative = NEGATIVE_CLAUSE_RE.match(clause)
            if negative:
                value = _clean(negative.group(1))
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
                            or item.source != "exclusion" and overlap and not changed_percentage
                            and len(alternative_values(item.text)) == 1):
                        retained.append(item)
                        continue
            other_facets = [name for name, candidate in FACET_PATTERNS.items()
                            if name != attribute and candidate.search(item.text)]
            if pattern and other_facets:
                remainder = _clean(pattern.sub("", item.text))
                if remainder:
                    retained.append(replace(item, text=remainder))
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
        grouped: dict[str | None, list[str]] = {}
        fallback = self._answer_attribute()
        for value in values:
            attribute = (
                _infer_attribute(value, fallback)
                if operation == PreferenceOperation.REPLACE
                else fallback
            )
            grouped.setdefault(attribute, []).append(value)
        for attribute, attribute_values in grouped.items():
            self._apply_values(
                attribute_values, attribute, weight, source, turn, operation
            )

    def _apply_values(
        self,
        values: list[str],
        attribute: str | None,
        weight: float,
        source: str,
        turn: int,
        operation: PreferenceOperation,
    ) -> None:
        """Replace, append to, or exclude from one attribute's active set."""
        cleaned = [value for raw in values if (value := _clean(raw))]
        if not cleaned:
            return
        if operation == PreferenceOperation.REPLACE:
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
                            retained.append(replace(item, text=" or ".join(remaining)))
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
