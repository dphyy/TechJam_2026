"""Conservative direct support for explicit material-composition requests."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from mercury.catalog import FIELD_NAMES, VOCABULARY
from mercury.types import Preference, Product


KNOWN_MATERIALS = frozenset(VOCABULARY["material"])
_MATERIAL_VALUE = "|".join(re.escape(value) for value in sorted(KNOWN_MATERIALS, key=len, reverse=True))
_COMPOSITION = re.compile(
    rf"^(?P<number>\d+(?:\.\d+)?)\s*%\s+(?P<material>{_MATERIAL_VALUE})$",
    re.IGNORECASE,
)
_COMPOSITION_SCORE = 0.010


@dataclass(frozen=True, slots=True)
class CompositionEvidenceWitness:
    """One local catalog phrase that directly supports a composition fact."""

    preference: str
    material: str
    source: str
    span: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class CompositionEvidenceResult:
    """A single bounded adjustment and its local source witnesses."""

    score: float
    witnesses: tuple[CompositionEvidenceWitness, ...]


def _qualified_composition(preference: Preference, active_materials: set[str]) -> tuple[str, str] | None:
    if not (preference.active and preference.polarity > 0 and preference.attribute == "other"):
        return None
    match = _COMPOSITION.fullmatch(" ".join(preference.value.lower().split()))
    if match is None:
        return None
    number, material = match.group("number"), match.group("material")
    if preference.depends_on != ("material", material) or material not in active_materials:
        return None
    return number, material


def _first_witness(product: Product, number: str, material: str, preference: str) -> CompositionEvidenceWitness | None:
    pattern = re.compile(
        rf"(?<!\w){re.escape(number)}\s*%\s+{re.escape(material)}(?!\w)",
        re.IGNORECASE,
    )
    for source in FIELD_NAMES:
        text = product.fields.get(source)
        if not isinstance(text, str):
            continue
        match = pattern.search(text)
        if match is not None:
            return CompositionEvidenceWitness(
                preference=preference,
                material=material,
                source=source,
                span=match.group(),
                start=match.start(),
                end=match.end(),
            )
    return None


def composition_evidence(product: Product, preferences: Sequence[Preference]) -> CompositionEvidenceResult:
    """Return one post-reranker boost for direct, current composition support.

    The shopper must have a positive dependent quantity fact and a currently active
    matching material fact. Product support requires that exact percentage/material
    phrase in one catalog field. Missing, cross-field, material-only, and stale facts
    remain unknown rather than becoming support or a constraint.
    """
    active_materials = {
        preference.value.lower()
        for preference in preferences
        if preference.active and preference.polarity > 0 and preference.attribute == "material"
    }
    witnesses: list[CompositionEvidenceWitness] = []
    seen: set[tuple[str, str]] = set()
    for preference in preferences:
        qualified = _qualified_composition(preference, active_materials)
        if qualified is None or qualified in seen:
            continue
        number, material = qualified
        witness = _first_witness(product, number, material, preference.value)
        if witness is not None:
            seen.add(qualified)
            witnesses.append(witness)
    score = _COMPOSITION_SCORE if witnesses else 0.0
    if not math.isfinite(score):
        score = 0.0
    return CompositionEvidenceResult(score=score, witnesses=tuple(witnesses))
