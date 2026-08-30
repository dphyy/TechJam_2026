"""Pinned catalog-derived aliases for unresolved shopping-language spans."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "mercury-catalog-vocabulary-v1"


@dataclass(frozen=True, slots=True)
class VocabularyMatch:
    attribute: str
    canonical: str
    start: int
    end: int
    confidence: float
    provenance: str


class CatalogVocabulary:
    """Validated immutable alias matcher bound to exact catalog bytes."""

    def __init__(self, path: str | Path, catalog_sha256: str) -> None:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            raise ValueError("Unsupported catalog vocabulary")
        if payload.get("catalog_sha256") != catalog_sha256:
            raise ValueError("Catalog vocabulary hash mismatch")
        version = payload.get("version")
        minimum_support = payload.get("minimum_support")
        minimum_confidence = payload.get("minimum_confidence")
        if not isinstance(version, str) or not version \
                or type(minimum_support) is not int or minimum_support < 2 \
                or type(minimum_confidence) not in (int, float) or not 0.5 <= minimum_confidence <= 1:
            raise ValueError("Invalid catalog vocabulary thresholds")
        aliases = payload.get("aliases")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError("Catalog vocabulary aliases must be nonempty")
        records = []
        seen = set()
        for row in aliases:
            if not isinstance(row, dict) or set(row) != {
                "alias", "attribute", "canonical", "support", "confidence", "method",
            }:
                raise ValueError("Malformed catalog vocabulary alias")
            alias, canonical = row["alias"], row["canonical"]
            if not isinstance(alias, str) or not alias or not isinstance(canonical, str) or not canonical \
                    or row["attribute"] not in {"category", "material", "color", "style", "feature"} \
                    or type(row["support"]) is not int or row["support"] < minimum_support \
                    or type(row["confidence"]) not in (int, float) or row["confidence"] < minimum_confidence \
                    or not isinstance(row["method"], str) or not row["method"]:
                raise ValueError("Catalog vocabulary alias violates its thresholds")
            if alias in seen:
                raise ValueError("Catalog vocabulary aliases must be unambiguous")
            seen.add(alias)
            records.append(row)
        self.version = version
        self.minimum_support = minimum_support
        self.minimum_confidence = float(minimum_confidence)
        self.model_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        taxonomy = payload.get("taxonomy", [])
        if not isinstance(taxonomy, list) or any(
            not isinstance(row, dict) or set(row) != {"canonical", "role", "support", "method"}
            or not isinstance(row["canonical"], str) or not row["canonical"]
            or row["role"] not in {"object", "accessory", "component"}
            or type(row["support"]) is not int or row["support"] < minimum_support
            or row["method"] != "category_path"
            for row in taxonomy
        ):
            raise ValueError("Malformed catalog-derived taxonomy")
        self._taxonomy = {row["canonical"]: row["role"] for row in taxonomy}
        self._records = {row["alias"]: row for row in records}
        self._pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(row["alias"]).replace(r"\ ", r"[-\s]+") for row in sorted(
                records, key=lambda item: (-len(item["alias"]), item["alias"]),
            )) + r")(?!\w)"
        )

    def category_role(self, canonical: str) -> str | None:
        return self._taxonomy.get(canonical)

    def find(self, text: str, occupied: list[tuple[int, int]] | None = None) -> list[VocabularyMatch]:
        occupied = list(occupied or ())
        matches = []
        for match in self._pattern.finditer(text):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            alias = re.sub(r"[-\s]+", " ", match.group())
            row = self._records[alias]
            matches.append(VocabularyMatch(
                row["attribute"], row["canonical"], match.start(), match.end(),
                float(row["confidence"]), f"catalog_alias:{self.version}:{row['method']}",
            ))
            occupied.append((match.start(), match.end()))
        return matches
