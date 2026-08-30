"""Evaluate target-independent dialogue properties over authored variants."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import tempfile
from pathlib import Path

from mercury.agent import Agent
from mercury.catalog import product_from_dict
from mercury.config import Config
from mercury.ranking import preference_evidence
from mercury.types import Preference


SCHEMA = "mercury-metamorphic-dialogues-v1"
PROPERTIES = {
    "equivalent_active_state", "equivalent_candidate_membership", "legal_output",
    "equivalent_retrieval_plan", "override_detected", "no_override",
    "unknown_not_contradicted",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pack(pack: object) -> dict:
    if not isinstance(pack, dict) or pack.get("schema") != SCHEMA:
        raise ValueError("Unsupported metamorphic fixture schema")
    cases = pack.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Metamorphic fixtures require cases")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"] or case["id"] in seen:
            raise ValueError("Metamorphic case IDs must be unique nonempty strings")
        seen.add(case["id"])
        catalog = case.get("catalog")
        if not isinstance(catalog, list) or not catalog:
            raise ValueError("Metamorphic cases require a catalog")
        ids = set()
        for row in catalog:
            product = product_from_dict(row) if isinstance(row, dict) else None
            if product is None or product.parent_asin in ids:
                raise ValueError("Metamorphic catalog IDs must be unique")
            ids.add(product.parent_asin)
        variants = case.get("variants")
        if not isinstance(variants, list) or len(variants) < 2:
            raise ValueError("Metamorphic cases require at least two variants")
        if any(not isinstance(messages, list) or not 1 <= len(messages) <= 10
               or any(not isinstance(message, str) or not message.strip() for message in messages)
               for messages in variants):
            raise ValueError("Metamorphic variants require one to ten nonempty messages")
        properties = case.get("properties")
        if not isinstance(properties, list) or not properties or set(properties) - PROPERTIES:
            raise ValueError("Unsupported metamorphic property")
        for check in case.get("unknown_checks", []):
            if not isinstance(check, dict) or set(check) != {"product_id", "attribute", "value"} \
                    or check["product_id"] not in ids:
                raise ValueError("Invalid unknown-metadata check")
    return pack


def _state_signature(diagnostics: dict) -> tuple[tuple, ...]:
    semantic = diagnostics.get("semantic_state_signature")
    if isinstance(semantic, list):
        return tuple(sorted(json.dumps(item, sort_keys=True) for item in semantic))
    return tuple(sorted(
        (
            item.get("attribute"), item.get("value"), item.get("polarity"),
            item.get("hard"), item.get("scope"), item.get("alternative_group"),
        )
        for item in diagnostics.get("preferences", [])
        if isinstance(item, dict) and item.get("active", True)
    ))


def _legal(response: dict, catalog_ids: set[str]) -> bool:
    recommendations = response.get("recommendations") if isinstance(response, dict) else None
    if not isinstance(recommendations, list) or len(recommendations) > 10:
        return False
    ids = [row.get("parent_asin") for row in recommendations if isinstance(row, dict)]
    return len(ids) == len(recommendations) and len(ids) == len(set(ids)) and set(ids) <= catalog_ids


def _run_variant(catalog: Path, config: Config, messages: list[str]) -> dict:
    agent = Agent(catalog, config)
    try:
        agent.reset("metamorphic", {})
        response: dict = {}
        for turn, message in enumerate(messages, 1):
            response = agent.respond("metamorphic", message, turn, 10)
        diagnostics = agent.last_diagnostics
        return {
            "state": _state_signature(diagnostics),
            "candidate_membership": tuple(sorted(diagnostics.get("ranked_ids", []))),
            "ranked_ids": tuple(diagnostics.get("ranked_ids", [])),
            "retrieval_plan_sha256": diagnostics.get("retrieval_plan_sha256"),
            "override": bool(diagnostics.get("override", {}).get("detected")),
            "response": response,
            "diagnostics": diagnostics,
        }
    finally:
        agent.close()


def evaluate_pack(pack: dict, config: Config) -> dict:
    validate_pack(pack)
    cases = []
    with tempfile.TemporaryDirectory(prefix="mercury-metamorphic-") as directory:
        root = Path(directory)
        for case in pack["cases"]:
            catalog = root / f"{case['id']}.jsonl"
            catalog.write_text("".join(json.dumps(row) + "\n" for row in case["catalog"]), encoding="utf-8")
            outcomes = [_run_variant(catalog, config, messages) for messages in case["variants"]]
            ids = {row["parent_asin"] for row in case["catalog"]}
            checks = []
            for prop in case["properties"]:
                if prop == "equivalent_active_state":
                    passed = len({outcome["state"] for outcome in outcomes}) == 1
                elif prop == "equivalent_candidate_membership":
                    passed = len({outcome["candidate_membership"] for outcome in outcomes}) == 1
                elif prop == "equivalent_retrieval_plan":
                    passed = len({outcome["retrieval_plan_sha256"] for outcome in outcomes}) == 1
                elif prop == "legal_output":
                    passed = all(_legal(outcome["response"], ids) for outcome in outcomes)
                elif prop == "override_detected":
                    passed = all(outcome["override"] for outcome in outcomes)
                elif prop == "no_override":
                    passed = all(not outcome["override"] for outcome in outcomes)
                elif prop == "unknown_not_contradicted":
                    passed = all(
                        preference_evidence(
                            product_from_dict(next(row for row in case["catalog"] if row["parent_asin"] == check["product_id"])),
                            Preference(check["attribute"], check["value"], 0, "metamorphic"),
                        ) == 0
                        for check in case.get("unknown_checks", [])
                    )
                else:  # pragma: no cover - validated above
                    raise AssertionError(prop)
                checks.append({"property": prop, "passed": passed})
            reference = outcomes[0]["ranked_ids"]
            comparisons = [_ranking_comparison(reference, outcome["ranked_ids"])
                           for outcome in outcomes[1:]]
            cases.append({
                "id": case["id"], "checks": checks,
                "invariance": {
                    "minimum_top120_jaccard": min(
                        (row["top120_jaccard"] for row in comparisons), default=1.0,
                    ),
                    "minimum_top10_overlap": min(
                        (row["top10_overlap"] for row in comparisons), default=1.0,
                    ),
                    "minimum_rank_correlation": min(
                        (row["rank_correlation"] for row in comparisons), default=1.0,
                    ),
                },
                "passed": all(row["passed"] for row in checks),
            })
    passed = sum(case["passed"] for case in cases)
    return {
        "schema": "mercury-metamorphic-result-v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "case_count": len(cases),
        "passed_cases": passed,
        "failed_cases": len(cases) - passed,
        "cases": cases,
    }


def _ranking_comparison(left: tuple[str, ...], right: tuple[str, ...]) -> dict[str, float]:
    left120, right120 = set(left[:120]), set(right[:120])
    union = left120 | right120
    common = [identifier for identifier in left[:120] if identifier in right120]
    right_rank = {identifier: index for index, identifier in enumerate(right[:120])}
    if len(common) < 2:
        correlation = 1.0 if left[:120] == right[:120] else 0.0
    else:
        left_positions = list(range(len(common)))
        right_positions = [right_rank[identifier] for identifier in common]
        left_mean = sum(left_positions) / len(common)
        right_mean = sum(right_positions) / len(common)
        numerator = sum(
            (a - left_mean) * (b - right_mean)
            for a, b in zip(left_positions, right_positions, strict=True)
        )
        denominator = (
            sum((value - left_mean) ** 2 for value in left_positions)
            * sum((value - right_mean) ** 2 for value in right_positions)
        ) ** 0.5
        correlation = numerator / denominator if denominator else 1.0
    return {
        "top120_jaccard": len(left120 & right120) / len(union) if union else 1.0,
        "top10_overlap": len(set(left[:10]) & set(right[:10])) / max(1, min(10, len(left), len(right))),
        "rank_correlation": correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run authored target-independent metamorphic dialogue checks")
    parser.add_argument("--fixture", type=Path, default=Path("data/metamorphic_robustness_v1.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/selected.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pack = json.loads(args.fixture.read_text(encoding="utf-8"))
    result = evaluate_pack(pack, Config.load(args.config))
    result["fixture_sha256"] = file_sha256(args.fixture)
    result["config_sha256"] = file_sha256(args.config)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
