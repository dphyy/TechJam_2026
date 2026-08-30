"""Build the conservative dual-lane catalog vocabulary v2 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from experiments.catalog_vocabulary_prepare import (
    GENERIC,
    MINIMUM_SUPPORT,
    STRUCTURED,
    normalize,
    variants,
)
from mercury.model_assets import file_sha256
from mercury.state import _LOOKUPS
from mercury.vocabulary import SCHEMA_V2


VERSION = "catalog-vocabulary-20260830-v2"
SEED = "catalog-vocabulary-cases-20260830-v2"
MINIMUM_CONFIDENCE = 0.80
MINIMUM_MARGIN = 0.20
STATE_CONFIDENCE = 0.95
STATE_MARGIN = 0.50
GENERIC_STATE_WORDS = frozenset(
    "active classic casual formal fashion premium basic standard everyday summer winter "
    "outdoor work travel party beach sport sports accessories accessory products".split()
)


def _role(canonical: str) -> str:
    if re.search(r"\b(?:replacement|parts?|components?|laces?|straps?|insoles?)\b", canonical):
        return "component"
    if re.search(r"\baccessor(?:y|ies)\b", canonical):
        return "accessory"
    return "object"


def build(rows: list[dict], catalog_sha256: str) -> dict:
    claims: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    methods: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    category_support: Counter[str] = Counter()
    for row in rows:
        title = normalize(row.get("title") or "")
        categories = row.get("categories") if isinstance(row.get("categories"), list) else []
        for raw in categories:
            canonical = normalize(raw)
            if not canonical or set(canonical.split()) <= GENERIC:
                continue
            category_support[canonical] += 1
            for alias in variants(canonical):
                claims[alias][("category", canonical)] += 1
                methods[(alias, "category", canonical)].add("category_path")
                if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", title):
                    methods[(alias, "category", canonical)].add("title_confirmation")
        details = row.get("details")
        if isinstance(details, dict):
            for key, raw in details.items():
                attribute = STRUCTURED.get(normalize(str(key)))
                canonical = normalize(str(raw)) if attribute and isinstance(raw, (str, int, float)) else ""
                if not canonical or not 1 <= len(canonical.split()) <= 4 or len(canonical) > 48:
                    continue
                for alias in variants(canonical):
                    claims[alias][(attribute, canonical)] += 1
                    methods[(alias, attribute, canonical)].add("structured_field")

    taxonomy = [
        {"canonical": canonical, "role": _role(canonical), "support": support,
         "method": "category_path"}
        for canonical, support in sorted(category_support.items())
        if support >= MINIMUM_SUPPORT
    ]
    roles = {row["canonical"]: row["role"] for row in taxonomy}
    aliases = []
    for alias, candidates in sorted(claims.items()):
        total = sum(candidates.values())
        ordered = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
        (attribute, canonical), count = ordered[0]
        runner_up = ordered[1][1] if len(ordered) > 1 else 0
        confidence = count / total
        margin = (count - runner_up) / total
        if count < MINIMUM_SUPPORT or confidence < MINIMUM_CONFIDENCE \
                or margin < MINIMUM_MARGIN or len(alias) < 3:
            continue
        role = roles.get(canonical) if attribute == "category" else None
        generic = len(alias.split()) == 1 and alias in GENERIC_STATE_WORDS
        state_eligible = bool(
            confidence >= STATE_CONFIDENCE
            and margin >= STATE_MARGIN
            and not generic
            and (
                (attribute == "category" and role == "object" and len(alias.split()) >= 2)
                or (attribute != "category" and len(alias.split()) >= 1)
            )
        )
        aliases.append({
            "alias": alias,
            "attribute": attribute,
            "canonical": canonical,
            "support": count,
            "confidence": round(confidence, 6),
            "ambiguity_margin": round(margin, 6),
            "state_eligible": state_eligible,
            "role": role,
            "method": "+".join(sorted(methods[(alias, attribute, canonical)])),
        })
    return {
        "schema": SCHEMA_V2,
        "version": VERSION,
        "catalog_sha256": catalog_sha256,
        "minimum_support": MINIMUM_SUPPORT,
        "minimum_confidence": MINIMUM_CONFIDENCE,
        "minimum_ambiguity_margin": MINIMUM_MARGIN,
        "state_minimum_confidence": STATE_CONFIDENCE,
        "state_minimum_ambiguity_margin": STATE_MARGIN,
        "normalization": "exact normalized token boundaries; conservative final-token inflection",
        "aliases": aliases,
        "taxonomy": taxonomy,
    }


def build_cases(payload: dict, state_count: int = 48, retrieval_count: int = 48,
                adversarial_count: int = 48) -> dict:
    static = {alias for values in _LOOKUPS.values() for alias in values}
    static_pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(
            re.escape(alias) for alias in sorted(static, key=len, reverse=True)
        ) + r")(?!\w)"
    )
    eligible = [
        row for row in payload["aliases"]
        if row["alias"] not in static and not static_pattern.search(row["alias"])
    ]

    def order(row: dict, lane: str) -> str:
        return hashlib.sha256(
            f"{SEED}\0{lane}\0{row['attribute']}\0{row['canonical']}\0{row['alias']}".encode()
        ).hexdigest()

    state_rows = sorted(
        (row for row in eligible if row["state_eligible"]), key=lambda row: order(row, "state"),
    )[:state_count]
    retrieval_rows = sorted(
        (row for row in eligible if not row["state_eligible"]),
        key=lambda row: order(row, "retrieval"),
    )[:retrieval_count]
    if len(state_rows) < state_count or len(retrieval_rows) < retrieval_count:
        raise ValueError("Not enough catalog aliases for the v2 frozen word suite")
    adversarial_source = sorted(state_rows + retrieval_rows, key=lambda row: order(row, "adversarial"))
    adversarial_rows = [adversarial_source[index % len(adversarial_source)]
                        for index in range(adversarial_count)]
    cases = []
    for index, row in enumerate(state_rows):
        cue = {
            "category": f"I need {row['alias']}.",
            "material": f"Material: {row['alias']}.",
            "color": f"Color: {row['alias']}.",
            "style": f"Style: {row['alias']}.",
            "feature": f"A must have feature is {row['alias']}.",
        }[row["attribute"]]
        cases.append({
            "id": f"state_{index + 1:03d}", "kind": "state_positive", "message": cue,
            "attribute": row["attribute"], "canonical": row["canonical"],
            "expect_state": True, "expect_expansion": True,
        })
    for index, row in enumerate(retrieval_rows):
        cases.append({
            "id": f"retrieval_{index + 1:03d}", "kind": "retrieval_positive",
            "message": f"Please explore {row['alias']} options.",
            "attribute": row["attribute"], "canonical": row["canonical"],
            "expect_state": False, "expect_expansion": True,
        })
    for index, row in enumerate(adversarial_rows):
        if index % 2 == 0:
            message = f"Please avoid {row['alias']}."
            kind = "negated_alias"
        else:
            message = f"The phrase {row['alias']} is just background conversation."
            kind = "ordinary_context"
        cases.append({
            "id": f"adversarial_{index + 1:03d}", "kind": kind, "message": message,
            "attribute": row["attribute"], "canonical": row["canonical"],
            "expect_state": False, "expect_expansion": kind == "ordinary_context",
        })
    return {
        "schema": "mercury-catalog-vocabulary-cases-v2",
        "seed": SEED,
        "evidence_kind": "catalog-derived dual-lane cases frozen without runtime rankings",
        "cases": cases,
    }


def prepare(catalog: Path, model: Path, cases: Path) -> dict:
    rows = []
    with catalog.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Catalog rows must be objects")
                rows.append(row)
    payload = build(rows, file_sha256(catalog))
    fixture = build_cases(payload)
    model.parent.mkdir(parents=True, exist_ok=True)
    cases.parent.mkdir(parents=True, exist_ok=True)
    model.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cases.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "catalog_sha256": file_sha256(catalog),
        "model_sha256": file_sha256(model),
        "cases_sha256": file_sha256(cases),
        "alias_count": len(payload["aliases"]),
        "state_alias_count": sum(row["state_eligible"] for row in payload["aliases"]),
        "case_count": len(fixture["cases"]),
        "attribute_counts": dict(Counter(row["attribute"] for row in payload["aliases"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dual-lane catalog vocabulary v2")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("models/catalog_vocabulary_v2.json"))
    parser.add_argument("--cases", type=Path, default=Path("data/catalog_vocabulary_cases_v2.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.catalog, args.model, args.cases), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
