"""Build a conservative catalog-derived vocabulary and frozen slot cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from mercury.model_assets import file_sha256
from mercury.state import _LOOKUPS
from mercury.vocabulary import SCHEMA


VERSION = "catalog-vocabulary-20260830-v1"
SEED = "catalog-vocabulary-cases-20260830-v1"
MINIMUM_SUPPORT = 5
MINIMUM_CONFIDENCE = 0.80
GENERIC = frozenset("clothing shoes jewelry women men girls boys accessories fashion products department".split())
STRUCTURED = {
    "color": "color", "colour": "color", "material": "material",
    "fabric type": "material", "style": "style", "pattern": "style",
}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def variants(value: str) -> set[str]:
    result = {value, value.replace("-", " ")}
    words = value.split()
    if words:
        last = words[-1]
        if last.endswith("ies") and len(last) > 4:
            result.add(" ".join(words[:-1] + [last[:-3] + "y"]))
        elif last.endswith("s") and not last.endswith("ss") and len(last) > 2:
            result.add(" ".join(words[:-1] + [last[:-1]]))
        elif not last.endswith("s"):
            result.add(" ".join(words[:-1] + [last + "s"]))
    return {normalize(item) for item in result if normalize(item)}


def build(rows: list[dict], catalog_sha256: str) -> dict:
    support: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    method: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        title = normalize(row.get("title") or "")
        categories = row.get("categories") if isinstance(row.get("categories"), list) else []
        for raw in categories:
            canonical = normalize(raw)
            if not canonical or set(canonical.split()) <= GENERIC:
                continue
            for alias in variants(canonical):
                support[("category", alias)][canonical] += 1
                method[("category", alias, canonical)].add("category_path")
                if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", title):
                    method[("category", alias, canonical)].add("title_confirmation")
        details = row.get("details")
        if isinstance(details, dict):
            for key, raw in details.items():
                attribute = STRUCTURED.get(normalize(str(key)))
                canonical = normalize(str(raw)) if attribute and isinstance(raw, (str, int, float)) else ""
                if not canonical or not 1 <= len(canonical.split()) <= 4 or len(canonical) > 48:
                    continue
                for alias in variants(canonical):
                    support[(attribute, alias)][canonical] += 1
                    method[(attribute, alias, canonical)].add("structured_field")

    aliases = []
    for (attribute, alias), candidates in sorted(support.items()):
        total = sum(candidates.values())
        canonical, count = min(candidates.items(), key=lambda item: (-item[1], item[0]))
        confidence = count / total
        if count < MINIMUM_SUPPORT or confidence < MINIMUM_CONFIDENCE or len(alias) < 3:
            continue
        aliases.append({
            "alias": alias,
            "attribute": attribute,
            "canonical": canonical,
            "support": count,
            "confidence": round(confidence, 6),
            "method": "+".join(sorted(method[(attribute, alias, canonical)])),
        })
    alias_counts = Counter(row["alias"] for row in aliases)
    aliases = [row for row in aliases if alias_counts[row["alias"]] == 1]
    category_support = Counter()
    for (attribute, _), candidates in support.items():
        if attribute == "category":
            category_support.update(candidates)
    taxonomy = []
    for canonical, count in sorted(category_support.items()):
        if count < MINIMUM_SUPPORT:
            continue
        if re.search(r"\b(?:replacement|parts?|components?|laces?|straps?|insoles?)\b", canonical):
            role = "component"
        elif re.search(r"\baccessor(?:y|ies)\b", canonical):
            role = "accessory"
        else:
            role = "object"
        taxonomy.append({"canonical": canonical, "role": role, "support": count, "method": "category_path"})
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "catalog_sha256": catalog_sha256,
        "minimum_support": MINIMUM_SUPPORT,
        "minimum_confidence": MINIMUM_CONFIDENCE,
        "normalization": "lowercase; punctuation/hyphen to spaces; conservative final-token singular/plural",
        "aliases": aliases,
        "taxonomy": taxonomy,
    }


def build_cases(payload: dict, count: int = 48) -> dict:
    static = {alias for values in _LOOKUPS.values() for alias in values}
    static_pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(alias) for alias in sorted(static, key=len, reverse=True)) + r")(?!\w)"
    )
    eligible = [
        row for row in payload["aliases"]
        if row["alias"] not in static and row["attribute"] == "category"
        and 2 <= len(row["alias"].split()) <= 5 and not static_pattern.search(row["alias"])
    ]
    ordered = sorted(
        eligible,
        key=lambda row: (hashlib.sha256(f"{SEED}\0{row['canonical']}\0{row['alias']}".encode()).hexdigest(), row["alias"]),
    )
    chosen = []
    seen_canonical = set()
    for row in ordered:
        if row["canonical"] in seen_canonical:
            continue
        chosen.append(row)
        seen_canonical.add(row["canonical"])
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise ValueError("Not enough disjoint catalog-derived category cases")
    templates = (
        "I need {alias}.", "Could you show me some {alias}?", "I'm browsing for {alias}.",
        "Please help me find {alias}.",
    )
    return {
        "schema": "mercury-catalog-vocabulary-cases-v1",
        "seed": SEED,
        "evidence_kind": "catalog-derived slot cases authored without runtime ranking or evaluator outcomes",
        "cases": [
            {
                "id": f"catalog_word_{index + 1:03d}",
                "message": templates[index % len(templates)].format(alias=row["alias"]),
                "attribute": row["attribute"],
                "canonical": row["canonical"],
                "alias": row["alias"],
                "support": row["support"],
                "paraphrase_family": f"catalog-template-{index % len(templates) + 1}",
            }
            for index, row in enumerate(chosen)
        ],
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
        "case_count": len(fixture["cases"]),
        "attribute_counts": dict(Counter(row["attribute"] for row in payload["aliases"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pinned catalog vocabulary and word cases")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("models/catalog_vocabulary_v1.json"))
    parser.add_argument("--cases", type=Path, default=Path("data/catalog_vocabulary_cases_v1.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.catalog, args.model, args.cases), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
