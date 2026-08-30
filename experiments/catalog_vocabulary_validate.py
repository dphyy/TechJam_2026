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
    if fixture.get("schema") not in {
        "mercury-catalog-vocabulary-cases-v1", "mercury-catalog-vocabulary-cases-v2",
    } \
            or not isinstance(fixture.get("cases"), list) or not fixture["cases"]:
        raise ValueError("Unsupported catalog vocabulary cases")
    catalog_object = Catalog(catalog)
    vocabulary = CatalogVocabulary(model, catalog_object.sha256)
    if fixture["schema"] == "mercury-catalog-vocabulary-cases-v2":
        return evaluate_v2(catalog, model, cases, fixture, vocabulary)
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


def evaluate_v2(catalog: Path, model: Path, cases: Path, fixture: dict,
                vocabulary: CatalogVocabulary) -> dict:
    state_tp = state_fp = state_fn = 0
    coverage_tp = coverage_fp = coverage_fn = 0
    rows = []
    for case in fixture["cases"]:
        state = SessionState({}, "ledger", "grouped", False, vocabulary, True)
        state.update(case["message"], 1)
        state_matches = {
            (item.attribute, item.value)
            for item in state.active_preferences()
            if item.polarity == 1 and item.source_kind.startswith("catalog_alias:")
        }
        expansion_matches = {
            (item.attribute, item.canonical) for item in state.last_vocabulary_expansions
        }
        expected = (case["attribute"], case["canonical"])
        state_matched = expected in state_matches
        expansion_matched = expected in expansion_matches
        expect_state = bool(case["expect_state"])
        expect_expansion = bool(case["expect_expansion"])
        state_tp += int(expect_state and state_matched)
        state_fn += int(expect_state and not state_matched)
        state_fp += int(not expect_state and bool(state_matches)) + len(state_matches - {expected})
        expected_coverage = expect_state or expect_expansion
        covered = state_matched or expansion_matched
        coverage_tp += int(expected_coverage and covered)
        coverage_fn += int(expected_coverage and not covered)
        coverage_fp += int(not expected_coverage and covered)
        rows.append({
            "id": case["id"], "kind": case["kind"],
            "state_matched": state_matched, "expansion_matched": expansion_matched,
            "state_proposal_count": len(state_matches),
            "expansion_count": len(expansion_matches),
        })
    return {
        "schema": "mercury-catalog-vocabulary-result-v2",
        "catalog_sha256": file_sha256(catalog),
        "model_sha256": file_sha256(model),
        "cases_sha256": file_sha256(cases),
        "state": {
            "true_positive": state_tp, "false_positive": state_fp, "false_negative": state_fn,
            "precision": state_tp / (state_tp + state_fp) if state_tp + state_fp else 1.0,
            "recall": state_tp / (state_tp + state_fn) if state_tp + state_fn else 0.0,
        },
        "dual_lane_coverage": {
            "true_positive": coverage_tp, "false_positive": coverage_fp,
            "false_negative": coverage_fn,
            "precision": coverage_tp / (coverage_tp + coverage_fp)
            if coverage_tp + coverage_fp else 1.0,
            "recall": coverage_tp / (coverage_tp + coverage_fn)
            if coverage_tp + coverage_fn else 0.0,
        },
        "cases": rows,
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
