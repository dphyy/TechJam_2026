from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from mercury.types import FacetEvidence, Product


FIELD_NAMES = ("title", "categories", "features", "details", "store", "description")
VOCABULARY = {
    "material": ("cotton", "leather", "polyester", "nylon", "wool", "silk", "linen",
                 "denim", "suede", "rubber", "canvas", "mesh", "fleece", "spandex",
                 "elastane", "rayon", "viscose", "cashmere", "velvet", "satin", "acrylic",
                 "stainless steel", "sterling silver", "gold", "silicone"),
    "color": ("black", "white", "blue", "navy", "red", "green", "yellow", "pink",
              "purple", "brown", "beige", "grey", "gray", "orange", "silver", "gold",
              "burgundy", "khaki", "cream", "tan", "teal", "multicolor"),
    "style": ("casual", "formal", "athletic", "vintage", "classic", "slim fit",
              "relaxed fit", "loose fit", "bohemian", "minimalist", "elegant"),
    "use_case": ("running", "walking", "hiking", "swimming", "wedding", "work",
                 "travel", "yoga", "cycling", "gym", "winter", "summer", "outdoor"),
    "feature": ("waterproof", "breathable", "lightweight", "adjustable", "stretch",
                "pockets", "arch support", "slip resistant", "machine washable",
                "quick dry", "insulated", "padded", "reversible", "seamless"),
    "category": ("shoes", "boots", "sandals", "sneakers", "slippers", "flats", "heels",
                 "shirts", "t-shirts", "tops", "pants", "jeans", "shorts", "leggings",
                 "skirts", "dresses", "jackets", "coats", "sweaters", "hoodies", "socks",
                 "underwear", "bras", "swimwear", "hats", "gloves", "scarves", "belts",
                 "bags", "backpacks", "wallets", "watches", "rings", "necklaces",
                 "earrings", "bracelets", "jewelry"),
}
PATTERNS = {
    attribute: re.compile(r"(?<!\w)(?:" + "|".join(
        re.escape(value) for value in sorted(values, key=len, reverse=True)
    ) + r")(?!\w)", re.I)
    for attribute, values in VOCABULARY.items()
}
STRUCTURED_KEYS = {"color": "color", "colour": "color", "material": "material",
                   "fabric type": "material", "style": "style", "size": "size",
                   "brand": "brand", "brand name": "brand"}


def flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {flatten(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(flatten(item) for item in value)
    return str(value)


def _price(value: object) -> tuple[float | None, bool]:
    lower_bound = False
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(from\s+)?\$?([\d,]+(?:\.\d+)?)\s*", value, re.I)
        if not match:
            return None, False
        lower_bound = bool(match.group(1))
        value = float(match.group(2).replace(",", ""))
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None, False
    return float(value), lower_bound


def negated_match(text: str, start: int, end: int) -> bool:
    """Whether this matched span is directly negated or materially qualified."""
    before, after = text[max(0, start - 25):start], text[end:end + 12]
    return bool(re.search(r"\b(?:no|not|non|without|faux|imitation|synthetic|avoid(?:s|ing)?|avoidance\s+of)\s*[- ]?$", before, re.I)
                or re.match(r"[- ]free\b", after, re.I))


def product_from_dict(row: dict) -> Product:
    identifier = row.get("parent_asin")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Catalog product requires a nonempty string parent_asin")
    fields = {name: flatten(row.get(name)) for name in FIELD_NAMES}
    found: dict[str, set[str]] = {}
    evidence: list[FacetEvidence] = []

    def add(attribute: str, value: str, source: str, confidence: float) -> None:
        normalized = re.sub(r"\s+", " ", value.lower()).strip()
        if not normalized:
            return
        if normalized == "gray":
            normalized = "grey"
        found.setdefault(attribute, set()).add(normalized)
        item = FacetEvidence(attribute, normalized, source, confidence)
        if item not in evidence:
            evidence.append(item)

    details = row.get("details")
    if isinstance(details, dict):
        for key, value in details.items():
            attribute = STRUCTURED_KEYS.get(str(key).strip().lower())
            if attribute and isinstance(value, (str, int, float)):
                add(attribute, str(value), f"details.{key}", 0.95)
    for source, content in fields.items():
        # Word matches provide soft evidence only; source strings remain available.
        for attribute, pattern in PATTERNS.items():
            for match in pattern.finditer(content):
                if not negated_match(content, match.start(), match.end()):
                    add(attribute, match.group(), source, 0.65 if source == "title" else 0.5)
    price, lower_bound = _price(row.get("price"))
    return Product(identifier, fields["title"], fields,
                   {key: tuple(sorted(values)) for key, values in found.items()},
                   tuple(evidence), price, lower_bound,
                   int(row.get("rating_number") or 0))


class Catalog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.products: list[Product] = []
        self.by_id: dict[str, Product] = {}
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for raw in handle:
                digest.update(raw)
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("Catalog rows must be objects")
                product = product_from_dict(row)
                if product.parent_asin in self.by_id:
                    raise ValueError(f"Duplicate catalog ID: {product.parent_asin}")
                self.products.append(product)
                self.by_id[product.parent_asin] = product
        if not self.products:
            raise ValueError("Catalog is empty")
        self.sha256 = digest.hexdigest()
