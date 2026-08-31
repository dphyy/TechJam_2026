from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .memory import LongTermUserProfile


OVERRIDE_RE = re.compile(
    r"\b(actually|instead|changed my mind|ignore|no longer|rather than)\b", re.I
)
NO_PREFERENCE_RE = re.compile(
    r"\b(?:do not|don't|dont|no)\s+(?:have\s+)?(?:an?\s+)?(?:additional\s+)?preference\b",
    re.I,
)
LOOKING_FOR_RE = re.compile(r"\blooking for\s+(.+?)(?:[,.]|$)", re.I)
NEED_RE = re.compile(r"\bwhat i need is\s*:\s*(.+)$", re.I)
REQUIREMENT_RE = re.compile(r"\bkey requirement is\s*:\s*(.+)$", re.I)
MATTERS_RE = re.compile(r"\bwhat matters is\s*:\s*(.+)$", re.I)
NEGATIVE_PREFERENCE_RE = re.compile(r"\b(?:do not|don't|dont|no longer)\s+(?:want|like|need|prefer)\s+(.+)$", re.I)
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
    if "budget" in tokens or re.search(r"(?:\$|<=|under|below|around)\s*\$?\d", lowered):
        return "budget"
    if tokens & MATERIAL_TERMS:
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

        if NO_PREFERENCE_RE.search(message):
            if self.asked_attributes:
                self.no_preference_attributes.add(self.asked_attributes[-1])
            return

        negative = NEGATIVE_PREFERENCE_RE.search(message)
        if negative:
            value = _clean(negative.group(1))
            attribute = _infer_attribute(value, self._answer_attribute())
            self._apply_values(
                [value], attribute, 3.8, "exclusion", turn,
                PreferenceOperation.EXCLUDE,
            )
            return

        is_override = bool(OVERRIDE_RE.search(message))
        if is_override:
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
            self._apply_grouped_values(
                [message], 2.5 if turn > 1 else 2.0, "clarification", turn,
                PreferenceOperation.APPEND,
            )

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
            self.evidence = [
                item for item in self.evidence if item.attribute != attribute
            ]
        elif operation == PreferenceOperation.EXCLUDE:
            for value in cleaned:
                self.evidence = [
                    item for item in self.evidence
                    if item.source == "category"
                    or item.source == "exclusion"
                    or not self._conflicts_with_exclusion(item, value, attribute)
                ]

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
        excluded = value.casefold()
        positive = item.text.casefold()
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
