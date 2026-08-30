from __future__ import annotations

import re
from dataclasses import dataclass

from mercury.types import Product


@dataclass(frozen=True, slots=True)
class ProductType:
    object_type: str | None
    role: str
    compatible_with: tuple[str, ...] = ()
    confidence: float = 0.0


_FAMILIES = {
    "footwear": ("shoes", "shoe", "sneakers", "sneaker", "boots", "boot", "sandals", "sandal"),
    "bag": ("bags", "bag", "handbag", "purse", "backpack", "backpacks", "tote"),
    "watch": ("watch", "watches"),
    "jewelry": ("ring", "rings", "necklace", "necklaces", "earring", "earrings",
                "bracelet", "bracelets", "jewelry", "jewellery"),
    "clothing": ("shirt", "shirts", "dress", "dresses", "jacket", "jackets", "coat", "coats",
                 "pants", "jeans", "sweater", "sweaters", "hoodie", "hoodies"),
}
_COMPONENTS = {
    "footwear": ("laces", "lace", "insoles", "insole", "shoe inserts", "heel grips"),
    "bag": ("replacement strap", "bag strap", "purse strap", "handles", "bag handles"),
    "watch": ("watch band", "watch strap", "replacement band"),
}
_ACCESSORY = re.compile(r"\b(?:accessor(?:y|ies)|replacement|spare|attachment|add-on)\b", re.I)


def requested_family(value: str) -> str | None:
    normalized = value.lower()
    for family, aliases in _FAMILIES.items():
        if any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", normalized) for alias in aliases):
            return family
    return None


def classify_product(product: Product) -> ProductType:
    category_text = product.fields.get("categories", "").lower()
    title_text = product.title.lower()
    text = f"{category_text} {title_text}"
    for family, markers in _COMPONENTS.items():
        if any(re.search(r"(?<!\w)" + re.escape(marker) + r"(?!\w)", text) for marker in markers):
            role = "component" if _ACCESSORY.search(text) or "laces" in text or "insoles" in text else "accessory"
            return ProductType(None, role, (family,), 0.95)
    if _ACCESSORY.search(category_text):
        compatible = tuple(family for family, aliases in _FAMILIES.items()
                           if any(alias in text for alias in aliases))
        return ProductType(None, "accessory", compatible, 0.9)
    for family, aliases in _FAMILIES.items():
        if any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", category_text) for alias in aliases):
            return ProductType(family, "object", (), 0.9)
    return ProductType(None, "unknown", (), 0.0)


def accessory_mismatch(product: Product, requested: tuple[str, ...]) -> bool:
    families = {family for value in requested if (family := requested_family(value)) is not None}
    product_type = classify_product(product)
    return bool(families and product_type.role in {"accessory", "component"}
                and families.intersection(product_type.compatible_with))


_SCOPES = ("body", "lining", "handle", "handles", "strap", "straps", "zipper", "sole",
           "lace", "laces", "sleeve", "sleeves", "collar", "cuff", "cuffs", "pocket", "pockets")


def scoped_value_evidence(product: Product, value: str, scope: str) -> float:
    """Return three-state evidence for a value owned by a named component."""
    text = product.text.lower()
    value_matches = list(re.finditer(r"(?<!\w)" + re.escape(value.lower()) + r"(?!\w)", text))
    scope_matches = list(re.finditer(r"(?<!\w)" + re.escape(scope.lower()) + r"(?!\w)", text))
    if not value_matches or not scope_matches:
        return 0.0
    relation = re.compile(
        r"(?:\b" + re.escape(value.lower()) + r"\b(?:\s+\w+){0,1}\s+\b" + re.escape(scope.lower())
        + r"\b|\b" + re.escape(scope.lower()) + r"\b(?:\s+\w+){0,1}\s+\b"
        + re.escape(value.lower()) + r"\b)"
    )
    if relation.search(text):
        return 0.8
    other_components = [match for component in _SCOPES if component != scope.lower()
                        for match in re.finditer(r"(?<!\w)" + re.escape(component) + r"(?!\w)", text)]
    if any(abs(value_match.start() - component.start()) <= len(value) + 12
           for value_match in value_matches for component in other_components):
        return -0.8
    return 0.0
