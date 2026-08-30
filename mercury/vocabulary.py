"""Pinned catalog-derived aliases for unresolved shopping-language spans."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "mercury-catalog-vocabulary-v1"
SCHEMA_V2 = "mercury-catalog-vocabulary-v2"


@dataclass(frozen=True, slots=True)
class VocabularyMatch:
    attribute: str
    canonical: str
    start: int
    end: int
    confidence: float
    provenance: str
    lane: str = "state"
    role: str | None = None


class CatalogVocabulary:
    """Validated immutable alias matcher bound to exact catalog bytes."""

    def __init__(self, path: str | Path, catalog_sha256: str) -> None:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") not in {SCHEMA, SCHEMA_V2}:
            raise ValueError("Unsupported catalog vocabulary")
        self.schema = payload["schema"]
        self.dual_lane = self.schema == SCHEMA_V2
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
        v1_fields = {"alias", "attribute", "canonical", "support", "confidence", "method"}
        v2_fields = v1_fields | {"ambiguity_margin", "state_eligible", "role"}
        for row in aliases:
            expected_fields = v2_fields if self.dual_lane else v1_fields
            if not isinstance(row, dict) or set(row) != expected_fields:
                raise ValueError("Malformed catalog vocabulary alias")
            alias, canonical = row["alias"], row["canonical"]
            if not isinstance(alias, str) or not alias or not isinstance(canonical, str) or not canonical \
                    or row["attribute"] not in {"category", "material", "color", "style", "feature"} \
                    or type(row["support"]) is not int or row["support"] < minimum_support \
                    or type(row["confidence"]) not in (int, float) or row["confidence"] < minimum_confidence \
                    or not isinstance(row["method"], str) or not row["method"]:
                raise ValueError("Catalog vocabulary alias violates its thresholds")
            if self.dual_lane and (
                    type(row["ambiguity_margin"]) not in (int, float)
                    or not 0 <= row["ambiguity_margin"] <= 1
                    or type(row["state_eligible"]) is not bool
                    or row["role"] not in {None, "object", "accessory", "component"}):
                raise ValueError("Catalog vocabulary v2 alias has invalid lane evidence")
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
        """Return state-lane proposals; v2 requires an explicit local cue."""
        occupied = list(occupied or ())
        matches = []
        for match in self._pattern.finditer(text):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            alias = re.sub(r"[-\s]+", " ", match.group())
            row = self._records[alias]
            if self.dual_lane and (
                    not row["state_eligible"]
                    or not self._state_cued(text, match.start(), match.end(), row)):
                continue
            matches.append(VocabularyMatch(
                row["attribute"], row["canonical"], match.start(), match.end(),
                float(row["confidence"]), f"catalog_alias:{self.version}:{row['method']}",
                "state", row.get("role"),
            ))
            occupied.append((match.start(), match.end()))
        return matches

    def find_expansions(self, text: str, occupied: list[tuple[int, int]] | None = None,
                        limit: int = 8) -> list[VocabularyMatch]:
        """Return bounded exact query-local expansions without persisting facts."""
        if not self.dual_lane:
            return []
        occupied = list(occupied or ())
        matches = []
        for match in self._pattern.finditer(text):
            if len(matches) >= limit:
                break
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            prefix = text[max(0, match.start() - 32):match.start()]
            suffix = text[match.end():match.end() + 24]
            if re.search(r"\b(?:no|not|without|avoid|exclude|excluding|hate|dislike)\b[^,;.!?]*$", prefix):
                continue
            if re.search(r"\b(?:does not|doesn't|do not|don't) matter\b|\bno longer\b", suffix):
                continue
            alias = re.sub(r"[-\s]+", " ", match.group())
            row = self._records[alias]
            matches.append(VocabularyMatch(
                row["attribute"], row["canonical"], match.start(), match.end(),
                float(row["confidence"]),
                f"catalog_expansion:{self.version}:{row['method']}",
                "retrieval", row.get("role"),
            ))
            occupied.append((match.start(), match.end()))
        return matches

    @staticmethod
    def _state_cued(text: str, start: int, end: int, row: dict) -> bool:
        window = text[max(0, start - 48):min(len(text), end + 32)]
        prefix = text[max(0, start - 32):start]
        if re.search(r"\b(?:no|not|without|avoid|exclude|excluding|hate|dislike)\b[^,;.!?]*$", prefix):
            return False
        attribute_cues = {
            "material": r"\b(?:material|fabric|made (?:of|from))\b",
            "color": r"\bcolou?r\b",
            "style": r"\b(?:style|pattern|design)\b",
            "feature": r"\b(?:feature|must have|with)\b",
        }
        if row["attribute"] == "category":
            if row.get("role") != "object":
                return False
            return bool(re.search(
                r"\b(?:need|want|find|show|buy|browse|search|looking for|shopping for)\b",
                window,
            ))
        pattern = attribute_cues.get(row["attribute"])
        return bool(pattern and re.search(pattern, window))
