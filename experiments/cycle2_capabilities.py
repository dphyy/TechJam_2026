"""Strict, evaluation-only checks for the pre-authored capability fixtures."""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import platform
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

from experiments.run import source_hashes
from experiments.run import peak_rss_bytes
from mercury.agent import Agent
from mercury.catalog import product_from_dict
from mercury.config import Config
from mercury.model_assets import file_sha256
from mercury.ranking import preference_evidence
from mercury.types import Preference


REPOSITORY = Path(__file__).resolve().parents[1]
STATUSES = ("passed", "failed", "unverified")
ATTRIBUTES = {"category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case", "other"}
ASSERTION_KEYS = {
    "rank_before": {"kind", "preferred", "other"},
    "active_preference": {"kind", "attribute", "value", "polarity"},
    "absent_positive_preference": {"kind", "attribute", "value"},
    "not_hard_excluded": {"kind", "product_id"},
    "unknown_fact": {"kind", "product_id", "attribute", "value"},
}


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_fixture(pack: object, split: str) -> dict:
    if not isinstance(pack, dict) or pack.get("schema") != "cycle2-capability-fixtures-v1":
        raise ValueError("Unsupported capability fixture schema")
    if split not in {"development", "validation"} or pack.get("split") != split:
        raise ValueError("Capability fixture split does not match the declared evaluation")
    cases = pack.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Capability cases must be a nonempty list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not _text(case.get("id")) or case["id"] in seen or not _text(case.get("group")):
            raise ValueError("Capability case IDs must be unique, with a nonempty group")
        seen.add(case["id"])
        rows = case.get("catalog")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Capability catalog must be a nonempty list")
        identifiers = set()
        for row in rows:
            if not isinstance(row, dict) or not _text(row.get("parent_asin")) or row["parent_asin"] in identifiers:
                raise ValueError("Capability catalog IDs must be unique normalized strings")
            if set(row) - {"parent_asin", "title", "categories", "features", "details", "price", "description", "store"}:
                raise ValueError("Only ordinary catalog fields may enter capability inference")
            product_from_dict(row)
            identifiers.add(row["parent_asin"])
        turns = case.get("turns")
        if not isinstance(turns, list) or not 1 <= len(turns) <= 10:
            raise ValueError("Capability conversations require one to ten turns")
        for turn in turns:
            if not isinstance(turn, dict) or not _text(turn.get("message")) or len(turn["message"]) > 8000:
                raise ValueError("Invalid capability message")
            assertions = turn.get("assertions")
            if not isinstance(assertions, list):
                raise ValueError("Every turn must declare an assertion list")
            for assertion in assertions:
                if not isinstance(assertion, dict) or not isinstance(assertion.get("kind"), str) \
                        or assertion["kind"] not in ASSERTION_KEYS:
                    raise ValueError("Unsupported capability assertion")
                if set(assertion) != ASSERTION_KEYS[assertion["kind"]]:
                    raise ValueError("Capability assertion fields do not match its kind")
                for key in ("preferred", "other", "product_id"):
                    if key in assertion and (not _text(assertion[key]) or assertion[key] not in identifiers):
                        raise ValueError("Capability assertion references an unknown catalog ID")
                if assertion.get("kind") == "rank_before" and assertion["preferred"] == assertion["other"]:
                    raise ValueError("Ranking comparisons require two different IDs")
                if "attribute" in assertion and (not isinstance(assertion["attribute"], str)
                                                 or assertion["attribute"] not in ATTRIBUTES or not _text(assertion["value"])):
                    raise ValueError("Invalid capability attribute/value")
                if "polarity" in assertion and (type(assertion["polarity"]) is not int or assertion["polarity"] not in {-1, 0, 1}):
                    raise ValueError("Capability preference polarity must be -1, 0, or 1")
        if not any(turn["assertions"] for turn in turns):
            raise ValueError("Every capability case must include at least one assertion")
    return pack


def check_assertion(assertion: dict, diagnostics: object, products: dict) -> dict:
    def outcome(status, reason):
        return {"assertion": copy.deepcopy(assertion), "status": status, "reason": reason}

    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    kind = assertion["kind"]
    if kind == "unknown_fact":
        try:
            product = products[assertion["product_id"]]
            value = preference_evidence(product, Preference(assertion["attribute"], assertion["value"], 0, ""))
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
            return outcome("unverified", f"Evidence adapter unavailable: {type(error).__name__}")
        if not _finite(value):
            return outcome("unverified", "Evidence adapter returned a non-finite signal")
        return outcome("passed" if value == 0 else "failed", f"Actual positive-preference evidence: {value}")
    if kind in {"active_preference", "absent_positive_preference"}:
        preferences = diagnostics.get("preferences")
        if not isinstance(preferences, list) or any(
                not isinstance(item, dict) or not _text(item.get("attribute")) or not _text(item.get("value"))
                or type(item.get("polarity")) is not int or item["polarity"] not in {-1, 0, 1}
                or ("active" in item and type(item["active"]) is not bool) for item in preferences):
            return outcome("unverified", "Active preference diagnostics unavailable or malformed")
        present = any(item.get("active", True) and item["attribute"] == assertion["attribute"]
                      and item["value"] == assertion["value"] and item["polarity"] == assertion.get("polarity", 1)
                      for item in preferences)
        passed = present if kind == "active_preference" else not present
        return outcome("passed" if passed else "failed", f"Matching active preference present: {present}")
    ranked = diagnostics.get("ranked_ids")
    if not isinstance(ranked, list) or any(not isinstance(item, str) or item not in products for item in ranked) \
            or len(ranked) != len(set(ranked)):
        return outcome("unverified", "Full ranked candidate diagnostics unavailable or malformed")
    if kind == "rank_before":
        if assertion["preferred"] not in ranked or assertion["other"] not in ranked:
            return outcome("failed", "Ranking comparison requires both candidates; at least one was not retrieved/retained")
        before = ranked.index(assertion["preferred"]) < ranked.index(assertion["other"])
        return outcome("passed" if before else "failed", "Compared both candidates in the full ranked list")
    if kind == "not_hard_excluded":
        identifier = assertion["product_id"]
        if identifier not in ranked:
            return outcome("failed", "Candidate is absent from the retained ranked pool")
        fallbacks = diagnostics.get("fallbacks")
        if not isinstance(fallbacks, list) or any(not isinstance(item, str) for item in fallbacks) \
                or {"constraints", "ranking"} & set(fallbacks):
            return outcome("unverified", "Constraint/evidence guard health is not verified")
        penalties = diagnostics.get("constraint_penalties")
        if not isinstance(penalties, dict) or identifier not in penalties \
                or not _finite(penalties[identifier]) or penalties[identifier] < 0:
            return outcome("unverified", "Explicit finite nonnegative constraint penalty unavailable")
        return outcome("passed" if penalties[identifier] == 0 else "failed",
                       f"Observed constraint penalty: {penalties[identifier]}")
    return outcome("unverified", "No adapter implements this assertion")


def validate_response(response: object, identifiers: set[str]) -> dict:
    problems = []
    if not isinstance(response, dict):
        return {"status": "failed", "reasons": ["Response must be an object"]}
    if set(response) - {"message", "ask_attribute", "recommendations", "usage"}:
        problems.append("Unexpected response fields")
    if not isinstance(response.get("message"), str):
        problems.append("message must be text")
    ask = response.get("ask_attribute")
    if "ask_attribute" not in response or (ask is not None and (not isinstance(ask, str) or ask not in ATTRIBUTES)):
        problems.append("ask_attribute must be an allowed attribute or null")
    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list) or len(recommendations) > 10:
        problems.append("recommendations must contain at most ten items")
    else:
        seen = set()
        for item in recommendations:
            if not isinstance(item, dict) or set(item) - {"parent_asin", "score"} \
                    or not isinstance(item.get("parent_asin"), str) or item["parent_asin"] not in identifiers:
                problems.append("Recommendation must identify an ordinary catalog item")
                continue
            if item["parent_asin"] in seen:
                problems.append("Duplicate recommendation")
            seen.add(item["parent_asin"])
            if "score" in item and not _finite(item["score"]):
                problems.append("Recommendation score must be finite")
    usage = response.get("usage")
    if not isinstance(usage, dict) or set(usage) != {"prompt_tokens", "completion_tokens"} \
            or any(type(value) is not int or value < 0 for value in usage.values()):
        problems.append("Measured usage requires nonnegative integer token counts")
    return {"status": "failed" if problems else "passed", "reasons": problems}


def _health(diagnostics: object, inner: object, config: Config) -> dict:
    startup = getattr(inner, "startup_fallbacks", None)
    if not isinstance(diagnostics, dict) or not isinstance(diagnostics.get("fallbacks"), list) \
            or any(not isinstance(value, str) for value in diagnostics["fallbacks"]) or not isinstance(startup, dict):
        return {"status": "unverified", "reason": "Fallback instrumentation unavailable"}
    missing = [name for enabled, name in ((config.neural_rerank, "reranker"), (config.dense, "dense"),
                                         (config.contrast, "contrast")) if enabled and getattr(inner, name, None) is None]
    return {"status": "failed" if startup or diagnostics["fallbacks"] or missing else "passed",
            "startup_fallbacks": startup, "turn_fallbacks": diagnostics["fallbacks"], "missing_declared_models": missing}


def _json_safe(value: object) -> object:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError):
        return {"invalid_serialization": repr(value)}


def evaluate_fixture(pack: dict, config: Config, *, agent_factory=Agent) -> dict:
    validate_fixture(pack, pack.get("split"))
    cases, latencies, cold_starts = [], [], []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for case in pack["cases"]:
        products = {row["parent_asin"]: product_from_dict(row) for row in case["catalog"]}
        recorded = {"id": case["id"], "group": case["group"], "turns": []}
        inner = None
        startup_error = None
        with tempfile.TemporaryDirectory(prefix="cycle2-capability-") as directory:
            catalog = Path(directory) / "catalog.jsonl"
            with catalog.open("x", encoding="utf-8") as handle:
                for row in case["catalog"]:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
            started = time.perf_counter()
            try:
                inner = agent_factory(catalog, config)
                inner.reset("capability-session", {})
            except Exception as error:
                startup_error = repr(error)
            cold_starts.append(time.perf_counter() - started)
            recorded["cold_start_seconds"] = cold_starts[-1]
            recorded["startup_fallbacks"] = _json_safe(getattr(inner, "startup_fallbacks", None))
            for number, turn in enumerate(case["turns"], 1):
                started = time.perf_counter()
                response = diagnostics = None
                error_text = startup_error
                if error_text is None:
                    try:
                        response = inner.respond("capability-session", turn["message"], number, 10)
                        diagnostics = getattr(inner, "last_diagnostics", None)
                    except Exception as error:
                        error_text = repr(error)
                elapsed = time.perf_counter() - started
                latencies.append(elapsed)
                contract = validate_response(response, set(products))
                checks = [check_assertion(assertion, diagnostics, products)
                          if error_text is None and contract["status"] == "passed" else {
                    "assertion": copy.deepcopy(assertion), "status": "unverified",
                    "reason": "Inference failed or returned an invalid response"} for assertion in turn["assertions"]]
                health = _health(diagnostics, inner, config)
                recorded["turns"].append({"turn": number, "message": turn["message"],
                                           "response": _json_safe(response), "diagnostics": _json_safe(diagnostics),
                                           "assertions": checks, "response_contract": contract, "health": health,
                                           "latency_seconds": elapsed, "error": error_text})
                if isinstance(response, dict) and isinstance(response.get("usage"), dict):
                    for key in usage:
                        value = response["usage"].get(key)
                        if type(value) is int and value >= 0:
                            usage[key] += value
            if inner is not None and hasattr(inner, "close"):
                try:
                    inner.close()
                except Exception as error:
                    recorded["close_error"] = repr(error)
        cases.append(recorded)
    turns = [turn for case in cases for turn in case["turns"]]
    counts = Counter(check["status"] for turn in turns for check in turn["assertions"])
    ordered = sorted(latencies)
    return {"schema": "cycle2-capability-results-v1", "split": pack["split"], "cases": cases,
            "case_count": len(cases), "turn_count": len(turns),
            "assertion_counts": {status: counts[status] for status in STATUSES},
            "all_passed": not counts["failed"] and not counts["unverified"] and all(
                turn["response_contract"]["status"] == turn["health"]["status"] == "passed" and not turn["error"]
                for turn in turns) and not any("close_error" in case for case in cases),
            "api_error_turns": sum(turn["response_contract"]["status"] != "passed" for turn in turns),
            "fallback_turns": sum(turn["health"]["status"] == "failed" for turn in turns),
            "health_unverified_turns": sum(turn["health"]["status"] == "unverified" for turn in turns),
            "usage": usage, "p50_seconds": statistics.median(ordered),
            "p95_seconds": ordered[min(len(ordered) - 1, int(.95 * len(ordered)))],
            "max_seconds": max(ordered), "cold_start_seconds_by_case": cold_starts,
            "max_rss_bytes": peak_rss_bytes(),
            "resource_note": "Tiny authored catalogs; cold starts reload the declared model per case. Not 50000-product latency evidence.",
            "paid_cost_usd": 0.0}


def paired_capability_comparison(control: dict, candidate: dict) -> dict:
    def indexed(report):
        result = {}
        seen = set()
        for case in report["cases"]:
            if case["id"] in seen:
                raise ValueError("Duplicate capability case ID")
            seen.add(case["id"])
            for turn in case["turns"]:
                for index, check in enumerate(turn["assertions"]):
                    key = (case["id"], turn["turn"], index)
                    if key in result or check["status"] not in STATUSES:
                        raise ValueError("Invalid or duplicate capability assertion outcome")
                    result[key] = (case["group"], check)
        return result

    left, right = indexed(control), indexed(candidate)
    if not left or left.keys() != right.keys() or not _text(control.get("dataset_sha256")) \
            or control["dataset_sha256"] != candidate.get("dataset_sha256"):
        raise ValueError("Capability comparisons require identical nonempty fixture/assertion sets")
    improved, regressed, newly_failed = [], [], []
    for key in sorted(left):
        group, before = left[key]
        other_group, after = right[key]
        if group != other_group or before["assertion"] != after["assertion"]:
            raise ValueError("Capability assertion identity changed across paired runs")
        record = {"case_id": key[0], "turn": key[1], "assertion_index": key[2],
                  "control": before["status"], "candidate": after["status"]}
        if before["status"] != "passed" and after["status"] == "passed":
            improved.append(record)
        if before["status"] == "passed" and after["status"] != "passed":
            regressed.append(record)
        if before["status"] != "failed" and after["status"] == "failed":
            newly_failed.append(record)
    return {"assertion_count": len(left), "improved": improved, "regressed": regressed,
            "newly_failed": newly_failed, "no_lost_passes": not regressed,
            "note": "Paired authored assertions, not independent shopper or organizer-test performance."}


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_capabilities(dataset: Path, config_path: Path, output: Path, split: str, *, provenance: dict) -> dict:
    """Core runner. Validation callers must first consume the manifest-owned guard."""
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    inputs = {"dataset_sha256": file_sha256(dataset), "config_sha256": file_sha256(config_path)}
    try:
        pack = validate_fixture(json.loads(dataset.read_text(encoding="utf-8")), split)
        config = Config.load(config_path)
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result = evaluate_fixture(pack, config)
        result.update(inputs)
        _write_json(output / "result.json", result)
        changed = sources != source_hashes() or inputs != {
            "dataset_sha256": file_sha256(dataset), "config_sha256": file_sha256(config_path)}
        _write_json(output / "manifest.json", {"schema": "cycle2-capability-run-v1", **inputs,
                    "dataset": str(dataset.resolve()), "config_path": str(config_path.resolve()),
                    "config": config.to_dict(), "source_hashes": sources, "provenance": provenance,
                    "source_or_input_changed_during_run": changed, "started_at_utc": started,
                    "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "python": platform.python_version(), "platform": platform.platform(),
                    "evidence_kind": pack.get("evidence_kind"), "authoring": pack.get("authoring"),
                    "interpretation": pack.get("interpretation")})
        if changed:
            raise ValueError("Capability source/config/input changed during the run")
        return {key: value for key, value in result.items() if key != "cases"}
    except BaseException as error:
        _write_json(output / "failure.json", {"error": repr(error), **inputs})
        raise


def run_development(config_path: Path, output: Path) -> dict:
    pack_root = REPOSITORY / "artifacts/cycle2"
    dataset = pack_root / "capability-development.json"
    lock = json.loads((pack_root / "capability-manifest.json").read_text(encoding="utf-8"))
    if lock.get("schema") != "cycle2-capability-lock-v1" or file_sha256(dataset) != lock.get("sha256", {}).get(dataset.name):
        raise ValueError("Development capability lock mismatch")
    return run_capabilities(dataset, config_path, output, "development", provenance={
        "split": "development", "capability_manifest_sha256": file_sha256(pack_root / "capability-manifest.json")})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only the locked development capability fixture; validation uses cycle2_evaluate")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_development(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
