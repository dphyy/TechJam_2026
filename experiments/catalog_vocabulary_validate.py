"""Measure frozen catalog-word slot recall without ranking target labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mercury.catalog import Catalog
from mercury.model_assets import file_sha256
from mercury.state import SessionState
from mercury.vocabulary import CatalogVocabulary


def _preferences(state: SessionState) -> list[dict]:
    return [
        {"attribute": item.attribute, "value": item.value, "polarity": item.polarity,
         "source_kind": item.source_kind}
        for item in state.active_preferences()
    ]


def evaluate(catalog: Path, model: Path, cases: Path) -> dict:
    fixture = json.loads(cases.read_text(encoding="utf-8"))
    if fixture.get("schema") != "mercury-catalog-vocabulary-cases-v1" \
            or not isinstance(fixture.get("cases"), list) or not fixture["cases"]:
        raise ValueError("Unsupported catalog vocabulary cases")
    catalog_object = Catalog(catalog)
    vocabulary = CatalogVocabulary(model, catalog_object.sha256)
    arms = {"selected": None, "catalog_vocabulary": vocabulary}
    results = []
    for name, source in arms.items():
        tp = fp = fn = 0
        rows = []
        for case in fixture["cases"]:
            state = SessionState({}, "ledger", "grouped", False, source)
            state.update(case["message"], 1)
            preferences = _preferences(state)
            proposed = {
                (row["attribute"], row["value"])
                for row in preferences if row["polarity"] == 1
                and row["attribute"] == case["attribute"]
                and (name == "selected" or row["source_kind"].startswith("catalog_alias:"))
            }
            expected = (case["attribute"], case["canonical"])
            matched = expected in proposed
            tp += int(matched)
            fn += int(not matched)
            fp += len(proposed - {expected})
            rows.append({"id": case["id"], "matched": matched, "proposal_count": len(proposed)})
        results.append({
            "name": name,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "cases": rows,
        })
    return {
        "schema": "mercury-catalog-vocabulary-result-v1",
        "catalog_sha256": file_sha256(catalog),
        "model_sha256": file_sha256(model),
        "cases_sha256": file_sha256(cases),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen catalog-derived word extraction")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("models/catalog_vocabulary_v1.json"))
    parser.add_argument("--cases", type=Path, default=Path("data/catalog_vocabulary_cases_v1.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.catalog, args.model, args.cases)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
