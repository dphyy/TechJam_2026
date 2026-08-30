"""Conservative, inspectable support for material-qualified whole-product roles."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from mercury.catalog import FIELD_NAMES
from mercury.types import Preference, Product


# These are canonical material values the conversation state can emit.  The
# helper intentionally does not try to resolve aliases, attributes, or free
# text at ranking time.
KNOWN_MATERIALS = frozenset({
    "acrylic",
    "bamboo",
    "canvas",
    "cashmere",
    "cotton",
    "denim",
    "faux leather",
    "fleece",
    "gold",
    "leather",
    "linen",
    "mesh",
    "merino wool",
    "nylon",
    "polyester",
    "rayon",
    "rubber",
    "satin",
    "silicone",
    "silk",
    "spandex",
    "stainless steel",
    "sterling silver",
    "suede",
    "velvet",
    "wool",
})
WHOLE_PRODUCT_ROLES = ("body", "outer shell", "exterior", "shell", "band")
COMPONENT_ROLES = ("lining", "handle", "cuff", "collar", "accent", "insert", "strap", "patch")

_ROLE_VALUE = "|".join(re.escape(role) for role in sorted(WHOLE_PRODUCT_ROLES, key=len, reverse=True))
_MATERIAL_VALUE = "|".join(re.escape(material) for material in sorted(KNOWN_MATERIALS, key=len, reverse=True))
_PREFERENCE = re.compile(
    rf"^(?P<material>{_MATERIAL_VALUE})\s+(?P<role>{_ROLE_VALUE})$",
    re.IGNORECASE,
)
_SOURCE_PHRASE = re.compile(
    rf"(?<!\w)(?P<material>{_MATERIAL_VALUE})\s+(?P<role>{_ROLE_VALUE})(?!\w)",
    re.IGNORECASE,
)

_PER_PREFERENCE_SCORE = 0.15
_MAX_SCORE = 0.30


@dataclass(frozen=True, slots=True)
class RoleEvidenceWitness:
    """One direct catalog phrase that supports one active shopper preference."""

    preference: str
    material: str
    role: str
    source: str
    span: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class RoleEvidenceResult:
    """Finite soft support plus the source spans that earned it."""

    score: float
    witnesses: tuple[RoleEvidenceWitness, ...]


def _qualified_role(value: str) -> tuple[str, str] | None:
    match = _PREFERENCE.fullmatch(" ".join(value.lower().split()))
    if match is None:
        return None
    return match.group("material"), match.group("role")


def _first_witness(product: Product, material: str, role: str, preference: str) -> RoleEvidenceWitness | None:
    for source in FIELD_NAMES:
        text = product.fields.get(source)
        if not isinstance(text, str):
            continue
        for match in _SOURCE_PHRASE.finditer(text):
            if (match.group("material").lower(), match.group("role").lower()) != (material, role):
                continue
            return RoleEvidenceWitness(
                preference=preference,
                material=material,
                role=role,
                source=source,
                span=match.group(),
                start=match.start(),
                end=match.end(),
            )
    return None


def role_evidence(product: Product, preferences: Sequence[Preference]) -> RoleEvidenceResult:
    """Return direct, local whole-product support for active positive ``other`` values.

    A preference must be exactly a known material immediately followed by an
    allowed whole-product role.  A product must repeat that direct phrase in
    one ordinary catalog field.  Cross-field matches, component-only claims,
    and all other text therefore remain unknown rather than becoming support.
    """
    active_materials = {
        preference.value.lower()
        for preference in preferences
        if preference.active and preference.polarity > 0 and preference.attribute == "material"
    }
    witnesses: list[RoleEvidenceWitness] = []
    seen: set[tuple[str, str]] = set()
    for preference in preferences:
        if not (preference.active and preference.polarity > 0 and preference.attribute == "other"):
            continue
        qualified = _qualified_role(preference.value)
        if qualified is None or qualified in seen:
            continue
        material, role = qualified
        if material not in active_materials:
            continue
        witness = _first_witness(product, material, role, preference.value)
        if witness is None:
            continue
        seen.add(qualified)
        witnesses.append(witness)

    score = min(_MAX_SCORE, _PER_PREFERENCE_SCORE * len(witnesses))
    # Both constants and the count are finite, but retain a closed boundary if
    # this helper is ever refactored to receive external score configuration.
    if not math.isfinite(score):
        score = 0.0
    return RoleEvidenceResult(score=score, witnesses=tuple(witnesses))
