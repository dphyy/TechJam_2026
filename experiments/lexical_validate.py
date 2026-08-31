"""Authored conversation checks with explicit diagnostic availability.

The embedded catalogs and expectations are independent fixtures. Only catalog
rows, visible messages, and ordinary profiles enter the agent. Expectations
remain in this runner; neither mode loads models or evaluation transcripts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Callable

from mercury.lexical.agent import Agent
from mercury.lexical.config import DEFAULT_AGENT_CONFIG, FULL_WIDTH_CONFIG


ROOT = Path(__file__).resolve().parents[1]
MODES = {"fullwidth": FULL_WIDTH_CONFIG, "current-default": DEFAULT_AGENT_CONFIG}
STAGES = ("retrieval_union", "question_context", "ranked_prefix", "returned")
SCHEMA = "authored-lexical-validation-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _contains(value: object, phrase: str) -> bool:
    tokens, expected = _tokens(value), _tokens(phrase)
    return bool(expected) and any(tokens[index:index + len(expected)] == expected
                                  for index in range(len(tokens) - len(expected) + 1))


def _row(identifier: str, category: str = "shirts", **fields) -> dict:
    return {"parent_asin": identifier, "title": category, "categories": [category],
            "price": 20, "average_rating": 4, "rating_number": 10, **fields}


def _say(text: str, **expect) -> dict:
    return {"op": "respond", "text": text, "expect": expect}


def _variant(identifier: str, *steps: dict, **options) -> dict:
    return {"id": identifier, "steps": list(steps), **options}


def authored_pack() -> dict:
    """Return fresh fixtures, including every equally supported leading choice."""
    shirts = [_row("A", features=["blue", "cotton"]),
              _row("B", features=["red", "wool"]),
              _row("C", features=["blue", "cotton"])]
    opening = "I'm looking for shirts."
    constraints = "A key requirement is: blue; cotton."
    common = {"active": ["blue", "cotton"], "absent": ["red", "wool"], "leaders": ["A", "C"]}
    cases = [
        {"id": "paraphrase", "catalog": shirts, "variants": [
            _variant("structured", _say(opening), _say(constraints, **common)),
            _variant("natural", _say(opening), _say("I would prefer a blue cotton shirt.", **common)),
        ], "compare": [{"kind": "state_projection", "phrases": ["blue", "cotton"]},
                       {"kind": "union_membership"}]},
        {"id": "constraint_order", "catalog": shirts, "variants": [
            _variant("forward", _say(opening), _say(constraints, **common)),
            _variant("reverse", _say(opening), _say("A key requirement is: cotton; blue.", **common)),
            _variant("separate", _say(opening), _say("A key requirement is: cotton."),
                     _say("A key requirement is: blue.", **common)),
        ], "compare": [{"kind": "state_projection", "phrases": ["blue", "cotton"]},
                       {"kind": "union_membership"}]},
        {"id": "natural_correction", "catalog": [
            _row("A", features=["black", "leather", "adjustable strap"]),
            _row("B", features=["blue", "canvas", "adjustable strap"]),
            _row("C", features=["blue", "canvas", "fixed strap"]),
            _row("D", features=["black", "cotton"]),
            _row("E", features=["blue", "cotton"]),
        ], "variants": [
            _variant("multiple_facets", _say(opening),
                     _say("A key requirement is: black; leather; adjustable strap."),
                     _say("Correction: make that blue and canvas, but keep the adjustable strap.",
                          active=["blue", "canvas", "adjustable strap"], absent=["black", "leather"],
                          retired=["black", "leather"], leaders=["B"])),
            _variant("keep_material", _say(opening), _say("I want a black cotton shirt."),
                     _say("Correction: make that blue, but keep the cotton.",
                          active=["blue", "cotton"], absent=["black"], retired=["black"], leaders=["E"])),
        ]},
        {"id": "neutrality", "catalog": shirts, "variants": [
            _variant("retract_color", _say(opening), _say(constraints),
                     _say("I have no preference for color.", active=["cotton"], absent=["blue"],
                          retired=["blue"], leaders=["A", "C"])),
            _variant("no_addition", _say(opening), _say(constraints),
                     _say("I have no additional preference for color.", **common)),
            _variant("add_feature", _say(opening), _say("A key requirement is: cotton."),
                     _say("I also prefer blue.", **common)),
        ]},
        {"id": "component_ownership", "catalog": [
            _row("A", "jackets", details={"upper": "leather", "lining": "cotton"}),
            _row("B", "jackets", details={"upper": "cotton", "lining": "leather"}),
            _row("C", "jackets", details={"upper": "cotton", "lining": "silk"}),
        ], "variants": [
            _variant("scoped", _say("I'm looking for jackets."),
                     _say("A key requirement is: upper: cotton; lining: leather.",
                          active=[{"phrase": "cotton", "scope": "upper"},
                                  {"phrase": "leather", "scope": "lining"}], leaders=["B"],
                          statuses=[{"id": "A", "phrase": "cotton", "status": "contradicted"},
                                    {"id": "B", "phrase": "cotton", "status": "supported"}]),
                     _say("Actually, what I need is: lining: silk.",
                          active=[{"phrase": "cotton", "scope": "upper"},
                                  {"phrase": "silk", "scope": "lining"}],
                          absent=[{"phrase": "leather", "scope": "lining"}], leaders=["C"])),
        ]},
        {"id": "alternatives_exclusion", "catalog": [
            _row("A", features=["cotton", "blue"]),
            _row("B", features=["linen", "blue"]),
            _row("C", features=["cotton", "red"]),
            _row("D", features=["wool", "blue"]),
        ], "variants": [
            _variant("positive_or", _say(opening),
                     _say("A key requirement is: cotton or linen; no red.",
                          active=["cotton or linen", {"phrase": "red", "polarity": -1}],
                          leaders=["A", "B"], ahead={"better": ["A", "B"], "worse": ["C", "D"]},
                          statuses=[{"id": "A", "phrase": "cotton or linen", "status": "supported"},
                                    {"id": "B", "phrase": "cotton or linen", "status": "supported"},
                                    {"id": "C", "phrase": "red", "status": "contradicted"}])),
            _variant("negative_or", _say(opening),
                     _say("No red or blue.", active=[{"phrase": "red or blue", "polarity": -1}],
                          statuses=[{"id": "A", "phrase": "red or blue", "status": "contradicted"},
                                    {"id": "C", "phrase": "red or blue", "status": "contradicted"}])),
        ]},
        {"id": "quoted_phrases", "catalog": [
            _row("A", features=["cotton", 'graphic reading "maybe later"']),
            _row("B", features=["wool", 'graphic reading "ready now"']),
        ], "variants": [
            _variant("reported", _say(opening),
                     _say('The label says "wool, and I need silk", and I need cotton.',
                          active=["cotton"], absent=["wool", "silk"], leaders=["A"])),
            _variant("requested", _say(opening),
                     _say('A key requirement is: a graphic reading "maybe later".',
                          active=["maybe later"], leaders=["A"],
                          statuses=[{"id": "A", "phrase": "maybe later", "status": "supported"}])),
        ]},
        {"id": "numeric_sizes", "catalog": [
            _row("A", "shoes", details={"Size": "8", "Width": "2 mm"}),
            _row("B", "shoes", details={"Size": "9", "Width": "3 mm"}),
            _row("C", "shoes", details={"Size": "8", "Width": "2 mm"}),
        ], "variants": [
            _variant("digits", _say("I'm looking for shoes."),
                     _say("A key requirement is: size 8; 2 mm.", active=["size 8", "2 mm"],
                          leaders=["A", "C"],
                          statuses=[{"id": "A", "phrase": "size 8", "status": "supported"},
                                    {"id": "C", "phrase": "2 mm", "status": "supported"}])),
        ]},
        {"id": "numeric_budgets", "catalog": [
            _row("A", price=0), _row("B", price=25),
            _row("C", price=26), _row("D", price=None),
        ], "variants": [
            _variant("inclusive_maximum", _say(opening),
                     _say("A key requirement is: maximum $25.", active=["maximum 25"],
                          leaders=["A", "B"],
                          statuses=[{"id": "A", "phrase": "maximum 25", "status": "supported"},
                                    {"id": "B", "phrase": "maximum 25", "status": "supported"},
                                    {"id": "C", "phrase": "maximum 25", "status": "contradicted"},
                                    {"id": "D", "phrase": "maximum 25", "status": "unknown"}])),
        ]},
        {"id": "unknown_contradiction", "catalog": [
            _row("A", features=["plain finish"]), _row("B", features=["leather"]),
            _row("C", features=["no leather"]),
        ], "variants": [
            _variant("absence", _say(opening),
                     _say("No leather.", active=[{"phrase": "leather", "polarity": -1}],
                          leaders=["A", "C"], ahead={"better": ["A", "C"], "worse": ["B"]},
                          statuses=[{"id": "A", "phrase": "leather", "status": "unknown"},
                                    {"id": "B", "phrase": "leather", "status": "contradicted"},
                                    {"id": "C", "phrase": "leather", "status": "supported"}])),
        ]},
        {"id": "reset_retry", "catalog": shirts, "variants": [
            _variant("replay_then_reset", _say(opening), _say(constraints, **common),
                     {"op": "retry", "mutate_returned_copy": True},
                     {"op": "conflict", "text": "I want red wool."}, {"op": "retry"},
                     {"op": "reset"}, _say("I'm looking for shirts. A key requirement is: red; wool.",
                                              active=["red", "wool"], absent=["blue", "cotton"], leaders=["B"])),
            _variant("clean", _say("I'm looking for shirts. A key requirement is: red; wool.",
                                   active=["red", "wool"], absent=["blue", "cotton"], leaders=["B"])),
        ], "compare": [{"kind": "response", "variants": ["replay_then_reset", "clean"]}]},
        {"id": "profile_privacy", "catalog": shirts, "variants": [
            _variant("isolated", _say("I'm looking for shirts. I always prefer cotton."),
                     {"op": "reset", "session": "T", "profile": {"profile_id": "profile-sentinel"}},
                     {**_say(opening, absent=["cotton"]), "session": "T"},
                     profile={"profile_id": "profile-sentinel", "private_note": "private-sentinel"}),
            _variant("clean", _say(opening), profile={"profile_id": "profile-sentinel"}),
            _variant("forgotten", _say(opening), {"op": "forget", "profile_id": "profile-sentinel"},
                     _say(opening), profile={"profile_id": "profile-sentinel", "preference_tags": ["cotton"],
                                            "private_note": "private-sentinel"}),
            _variant("unseeded", _say(opening), _say(opening)),
        ], "compare": [{"kind": "response", "variants": ["isolated", "clean"]},
                       {"kind": "ranked_prefix", "variants": ["isolated", "clean"]},
                       {"kind": "response", "variants": ["forgotten", "unseeded"]},
                       {"kind": "ranked_prefix", "variants": ["forgotten", "unseeded"]}]},
    ]
    return {"schema": SCHEMA, "cases": cases}


def _check(name: str, passed: bool | None, observed: object = None) -> dict:
    return {"check": name, "status": "unavailable" if passed is None else "pass" if passed else "fail",
            "passed": passed is True, "observed": observed}


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _json_safe(value: object) -> object:
    if type(value) is float and not math.isfinite(value):
        return {"invalid_number": str(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or type(value) in {str, bool, int, float}:
        return value
    return {"invalid_json_type": type(value).__name__}


def _status(checks: list[dict]) -> str:
    statuses = {row["status"] for row in checks}
    return next((status for status in ("error", "fail", "unavailable") if status in statuses), "pass")


def _evidence_rows(diagnostics: dict, key: str = "active") -> list[dict] | None:
    receipt = diagnostics.get("evidence")
    if not isinstance(receipt, dict) or receipt.get(f"{key}_complete") is not True:
        return None
    rows = receipt.get(key)
    if not isinstance(rows, list) or receipt.get(f"{key}_count") != len(rows):
        return None
    required = {"evidence_id", "value", "source_turn", "source_kind", "polarity", "scope", "active"}
    return rows if all(isinstance(row, dict) and required <= row.keys() for row in rows) else None


def _matches(row: dict, rule: str | dict) -> bool:
    rule = {"phrase": rule} if isinstance(rule, str) else rule
    return (_contains(row.get("value", ""), rule["phrase"])
            and row.get("polarity") == rule.get("polarity", 1)
            and ("scope" not in rule or row.get("scope") == rule["scope"]))


def _stage(diagnostics: dict, name: str) -> list[str] | None:
    receipt = _mapping(diagnostics.get("stage_receipts")).get(name)
    if not isinstance(receipt, dict) or receipt.get("available") is not True or receipt.get("complete") is not True:
        return None
    values = receipt.get("ids")
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        return None
    return values


def _flatten(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten(value[key])}" for key in sorted(value))
    if isinstance(value, list):
        return " ".join(_flatten(item) for item in value)
    return "" if value is None else str(value)


def _witness_valid(witness: dict, product: dict) -> bool:
    field = witness.get("field")
    kind = witness.get("match_kind")
    if field not in product or kind not in {"numeric_value", "normalized_phrase", "scoped_value", "explicit_absence"}:
        return False
    if kind == "numeric_value":
        return (type(product[field]) in {int, float} and math.isfinite(product[field])
                and witness.get("catalog_value") == product[field]
                and type(witness.get("requested_value")) in {int, float}
                and math.isfinite(witness["requested_value"]))
    raw = witness.get("raw_value")
    actual_tokens = _tokens(_flatten(product[field]))
    if not isinstance(raw, str) or not raw:
        return False
    raw_tokens = _tokens(raw)
    if (not raw_tokens or actual_tokens[:len(raw_tokens)] != raw_tokens
            or witness.get("raw_value_complete") is True and actual_tokens != raw_tokens):
        return False
    phrase = witness.get("normalized_phrase")
    if kind in {"normalized_phrase", "scoped_value"} and (not isinstance(phrase, str) or not phrase):
        return False
    if witness.get("scope") and isinstance(product[field], dict):
        owned = product[field].get(witness["scope"])
        if owned is not None and phrase and not _contains(owned, phrase):
            return False
    return phrase is None or _contains(_flatten(product[field]), phrase)


def check_turn(response: object, diagnostics: object, *, catalog: list[dict], catalog_digest: str,
               mode: str, turn: int, sources: dict[int, str], expect: dict | None = None) -> list[dict]:
    """Validate observed receipts; missing, partial, and empty are distinct states."""
    expect = expect or {}
    products = {row["parent_asin"]: row for row in catalog}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    checks = []
    rows = response.get("recommendations") if isinstance(response, dict) else None
    legal = (isinstance(rows, list) and len(rows) <= 10 and isinstance(response.get("message"), str)
             and (response.get("ask_attribute") is None or isinstance(response.get("ask_attribute"), str))
             and isinstance(response.get("usage"), dict)
             and {"prompt_tokens", "completion_tokens"} <= response["usage"].keys()
             and all(type(value) is int and value >= 0 for value in response["usage"].values()))
    if legal:
        legal = all(isinstance(row, dict) and isinstance(row.get("parent_asin"), str)
                    and row["parent_asin"] in products and type(row.get("score")) in {int, float}
                    and math.isfinite(row["score"]) for row in rows)
    returned = [row["parent_asin"] for row in rows] if legal else []
    checks.append(_check("legal_response", bool(legal and len(returned) == len(set(returned)))))
    checks.append(_check("successful_transaction", None if not diagnostics else
                         diagnostics.get("request_succeeded") is True and diagnostics.get("state_committed") is True))
    stages = {name: _stage(diagnostics, name) for name in STAGES}
    for name, ids in stages.items():
        receipt = _mapping(diagnostics.get("stage_receipts")).get(name, {})
        checks.append(_check(f"stage:{name}", None if ids is None else
                             len(ids) == len(set(ids)) and set(ids) <= products.keys()
                             and receipt.get("count") == len(ids) and receipt.get("sha256") == _digest(ids)
                             and _mapping(diagnostics.get("stage_ids")).get(name) == ids
                             and _mapping(diagnostics.get("stage_counts")).get(name) == len(ids), receipt))
    complete = all(value is not None for value in stages.values())
    checks.append(_check("stage_membership", None if not complete else
                         all(set(stages[right]) <= set(stages[left]) for left, right in zip(STAGES, STAGES[1:]))
                         and stages["returned"] == returned))
    # All authored rows belong to the visible category and are below route limits.
    checks.append(_check("catalog_membership_preserved", None if stages["retrieval_union"] is None else
                         set(stages["retrieval_union"]) == products.keys(), stages["retrieval_union"]))
    width = diagnostics.get("output_width")
    checks.append(_check("presentation_receipt", None if not isinstance(width, dict) else
                         width.get("requested") == 10 and width.get("returned") == len(returned)
                         and width.get("full_width") is (mode == "fullwidth")
                         and isinstance(width.get("ambiguity_deferred"), bool) and bool(width.get("reason")), width))
    if mode == "fullwidth":
        checks.append(_check("fullwidth_raw_prefix", None if not complete else
                             returned == stages["ranked_prefix"] and len(returned) == len(products)))

    active, retired = _evidence_rows(diagnostics), _evidence_rows(diagnostics, "retired")
    checks.append(_check("active_evidence_available", None if active is None else
                         diagnostics.get("preferences") == active))
    checks.append(_check("retired_evidence_available", None if retired is None else
                         diagnostics.get("retired_preferences") == retired))
    source_rows = diagnostics.get("evidence_sources")
    source_valid = None if not isinstance(source_rows, list) else all(
        isinstance(row, dict) and row.get("source_turn") in sources
        and row.get("text") == sources[row["source_turn"]][:8000]
        and row.get("message_sha256") == hashlib.sha256(sources[row["source_turn"]].encode()).hexdigest()
        and row.get("received_characters") == len(sources[row["source_turn"]])
        and row.get("accepted_characters") == min(8000, len(sources[row["source_turn"]]))
        and row.get("complete") is (len(sources[row["source_turn"]]) <= 8000)
        for row in source_rows)
    if source_valid:
        source_valid = {row["source_turn"] for row in source_rows} == sources.keys()
    checks.append(_check("message_receipts", source_valid))
    checks.append(_check("evidence_turns", None if active is None else
                         all(type(row["source_turn"]) is int and 1 <= row["source_turn"] <= turn
                             and row["active"] is True
                             for row in active)))
    source_turns = {row.get("source_turn") for row in source_rows or [] if isinstance(row, dict)}
    covered = active is not None and all(row["source_turn"] in source_turns for row in active)
    checks.append(_check("active_evidence_source_coverage", True if covered else None,
                         {"retained_source_turns": sorted(source_turns)}))
    for key, observed in (("active", active), ("absent", active), ("retired", retired)):
        for rule in expect.get(key, []):
            found = observed is not None and any(_matches(row, rule) for row in observed)
            checks.append(_check(f"{key}:{json.dumps(rule, sort_keys=True)}", None if observed is None else
                                 not found if key == "absent" else found, observed))

    ranked = stages["ranked_prefix"]
    if "leaders" in expect:
        checks.append(_check("acceptable_leader", None if ranked is None or not ranked else
                             ranked[0] in expect["leaders"], {"allowed": expect["leaders"], "ranked": ranked}))
    if "ahead" in expect:
        better, worse = expect["ahead"]["better"], expect["ahead"]["worse"]
        checks.append(_check("safe_tier_order", None if ranked is None else
                             all(key in ranked for key in better + worse)
                             and max(ranked.index(key) for key in better) < min(ranked.index(key) for key in worse), ranked))
    constraint_rows = diagnostics.get("constraint_checks")
    constraint_valid = isinstance(constraint_rows, list) and all(isinstance(row, dict) for row in constraint_rows)
    checks.append(_check("returned_evidence_coverage", None if not constraint_valid or active is None else
                         {row.get("parent_asin") for row in constraint_rows} == set(returned)
                         and all(isinstance(row.get("evidence"), list)
                                 and {item.get("evidence_id") for item in row["evidence"]} ==
                                 {item["evidence_id"] for item in active} for row in constraint_rows)))
    for rule in expect.get("statuses", []):
        matching = [item for row in (constraint_rows if constraint_valid else []) if row.get("parent_asin") == rule["id"]
                    for item in row.get("evidence", []) if _contains(item.get("value", ""), rule["phrase"])]
        checks.append(_check(f"constraint:{rule['id']}:{rule['phrase']}", None if not matching else
                             all(item.get("status") == rule["status"] for item in matching),
                             {"expected": rule["status"], "evidence": matching,
                              "coverage": "observed" if matching else "not_displayed" if rule["id"] not in returned
                              else "receipt_missing"}))
    if constraint_valid:
        witnesses = [(row.get("parent_asin"), item) for row in constraint_rows for item in row.get("evidence", [])]
        checks.append(_check("catalog_witnesses", None if not witnesses else all(
            identifier in products and item.get("status") in {"supported", "contradicted", "unknown"}
            and isinstance(item.get("witnesses"), list)
            and (item["status"] != "supported" or bool(item["witnesses"]))
            and all(isinstance(witness, dict) and _witness_valid(witness, products[identifier])
                    for witness in item["witnesses"]) for identifier, item in witnesses)))
    else:
        checks.append(_check("catalog_witnesses", None))

    identity = diagnostics.get("identity")
    if not isinstance(identity, dict):
        checks.append(_check("runtime_binding", None))
    else:
        configuration = identity.get("configuration")
        package, components = identity.get("runtime_hashes"), identity.get("implementations")
        checks.append(_check("runtime_binding", identity.get("catalog_sha256") == catalog_digest
                             and identity.get("requested_catalog_sha256") == catalog_digest
                             and identity.get("catalog_count") == len(catalog)
                             and isinstance(configuration, dict) and configuration.get("agent") == asdict(MODES[mode])
                             and identity.get("config_sha256") == _digest(configuration)
                             and isinstance(package, dict) and bool(package) and isinstance(components, dict)
                             and identity.get("runtime_source_sha256") == _digest({"package": package, "components": components})
                             and _mapping(identity.get("binding")).get("complete") is True, identity))
        actual = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted((ROOT / "mercury" / "lexical").glob("*.py"))}
        checks.append(_check("runtime_source_files", None if not isinstance(package, dict) else package == actual))
    capabilities = _mapping(diagnostics.get("effective_capabilities")).get("components")
    checks.append(_check("model_free", None if not isinstance(capabilities, dict) else all(
        isinstance(capabilities.get(name), dict)
        and all(capabilities[name].get(flag) is False for flag in ("requested", "loaded", "effective"))
        for name in ("vector_rerank", "neural_rerank")), capabilities))
    current_call = diagnostics.get("current_call")
    checks.append(_check("model_free_execution", None if not isinstance(current_call, dict) else
                         current_call.get("inference_executed") is False, current_call))
    encoded = json.dumps(_json_safe(diagnostics), sort_keys=True, allow_nan=False)
    checks.append(_check("diagnostic_json_values", _json_safe(diagnostics) == diagnostics))
    checks.append(_check("profile_fields_private", None if not diagnostics else not any(value in encoded for value in
                                                           ("profile-sentinel", "private-sentinel"))))
    return checks


def _read_diagnostics(agent) -> dict:
    value = getattr(agent, "last_diagnostics", None)
    return deepcopy(value) if isinstance(value, dict) else {}


def _error(exc: Exception, expected: bool = False) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)[:512], "expected": expected}


def _run_variant(catalog_path: Path, case: dict, variant: dict, mode: str, agent_factory: Callable) -> dict:
    checks, operations = [], []
    agent = None
    sessions: dict[str, dict] = {}
    catalog_digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    try:
        agent = agent_factory(catalog_path, config=MODES[mode], share_profile_memory=False)
        agent.reset("S", deepcopy(variant.get("profile", {})))
        sessions["S"] = {"turn": 0, "sources": {}, "last": None,
                         "profile_id": variant.get("profile", {}).get("profile_id", "S")}
        for step in variant["steps"]:
            event = {"op": step["op"], "checks": []}
            operations.append(event)
            session = step.get("session", "S")
            try:
                if step["op"] == "reset":
                    agent.reset(session, deepcopy(step.get("profile", {})))
                    sessions[session] = {"turn": 0, "sources": {}, "last": None,
                                         "profile_id": step.get("profile", {}).get("profile_id", session)}
                    event["checks"].append(_check("reset_clears_last_receipt", _read_diagnostics(agent) == {}))
                    continue
                if step["op"] == "forget":
                    agent.forget_profile(step["profile_id"])
                    event["checks"].append(_check("forgotten_profile", agent.export_profile(step["profile_id"]) is None))
                    # Accepted conversation state remains, but source/cache receipts are intentionally cleared.
                    for state in sessions.values():
                        if state["profile_id"] == step["profile_id"]:
                            state["sources"] = {}
                            state["last"] = None
                    continue
                state = sessions[session]
                if step["op"] == "conflict":
                    message = step["text"]
                    try:
                        agent.respond(session, message, state["turn"], 10)
                    except Exception as exc:
                        event["error"] = _error(exc, isinstance(exc, ValueError))
                        failed = _read_diagnostics(agent)
                        event["diagnostics"] = failed
                        event["checks"].append(_check("conflicting_retry_rejected", isinstance(exc, ValueError)))
                        event["checks"].append(_check("failed_transaction_receipt", None if not failed else
                                                     failed.get("request_succeeded") is False
                                                     and failed.get("state_committed") is False))
                    else:
                        event["checks"].append(_check("conflicting_retry_rejected", False))
                    continue
                retry = step["op"] == "retry"
                if retry:
                    prior = state["last"]
                    turn, message = prior["turn"], prior["message"]
                    if step.get("mutate_returned_copy"):
                        prior["returned_object"]["recommendations"].clear()
                elif step["op"] == "respond":
                    turn, message = state["turn"] + 1, step["text"]
                else:
                    raise ValueError("unsupported fixture operation")
                event.update(turn=turn, message=message)
                response = agent.respond(session, message, turn, 10)
                diagnostics = _read_diagnostics(agent)
                accepted = {**state["sources"], turn: message}
                event.update(response=deepcopy(response), diagnostics=diagnostics)
                event["checks"].extend(check_turn(response, diagnostics, catalog=case["catalog"],
                                                  catalog_digest=catalog_digest, mode=mode, turn=turn,
                                                  sources=accepted, expect=step.get("expect")))
                if retry:
                    event["checks"].append(_check("identical_detached_response", response == prior["response"]))
                    current_call = diagnostics.get("current_call")
                    event["checks"].append(_check("retry_no_work", None if not isinstance(current_call, dict) else
                                                 diagnostics.get("cache_hit") is True
                                                 and current_call.get("search_executed") is False
                                                 and current_call.get("inference_executed") is False))
                    event["checks"].append(_check("retry_state_unchanged", None if _evidence_rows(diagnostics) is None else
                                                 diagnostics.get("evidence") == prior["diagnostics"].get("evidence")))
                else:
                    state.update(turn=turn, sources=accepted,
                                 last={"turn": turn, "message": message, "response": deepcopy(response),
                                       "returned_object": response, "diagnostics": diagnostics})
            except Exception as exc:
                event["error"] = _error(exc)
                event["diagnostics"] = _read_diagnostics(agent)
                event["checks"].append({"check": "operation_completed", "status": "error", "passed": False,
                                        "observed": event["error"]})
    except Exception as exc:
        checks.append({"check": "variant_setup", "status": "error", "passed": False, "observed": _error(exc)})
    finally:
        if agent is not None:
            try:
                agent.close()
            except Exception as exc:
                checks.append({"check": "agent_closed", "status": "error", "passed": False, "observed": _error(exc)})
    checks.extend(check for event in operations for check in event["checks"])
    return {"id": variant["id"], "status": _status(checks), "operations": operations,
            "checks": checks, "catalog_sha256": catalog_digest}


def _last_success(variant: dict) -> dict | None:
    return next((event for event in reversed(variant["operations"]) if "response" in event and "error" not in event), None)


def _compare(variants: list[dict], spec: dict) -> dict:
    selected = [variant for variant in variants if "variants" not in spec or variant["id"] in spec["variants"]]
    events = [_last_success(variant) for variant in selected]
    if len(events) < 2 or any(event is None for event in events):
        return _check(f"invariance:{spec['kind']}", None)
    values = []
    for event in events:
        diagnostics = event["diagnostics"]
        if spec["kind"] == "response":
            value = event["response"]
        elif spec["kind"] in {"union_membership", "ranked_prefix"}:
            stage = _stage(diagnostics, "retrieval_union" if spec["kind"] == "union_membership" else "ranked_prefix")
            value = (sorted(stage) if spec["kind"] == "union_membership" else stage) if stage is not None else None
        elif spec["kind"] == "state_projection":
            rows = _evidence_rows(diagnostics)
            value = sorted({(phrase, row["polarity"], row["scope"] or "")
                            for phrase in spec["phrases"] for row in rows or []
                            if _contains(row["value"], phrase)}) if rows is not None else None
        else:
            raise ValueError("unsupported fixture comparison")
        values.append(value)
    return _check(f"invariance:{spec['kind']}", None if any(value is None for value in values) else
                  all(value == values[0] for value in values[1:]), values)


def _source_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "mercury" / "lexical").glob("*.py")) + [Path(__file__)]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _validate_pack(pack: dict) -> None:
    if not isinstance(pack, dict) or pack.get("schema") != SCHEMA or not isinstance(pack.get("cases"), list) or not pack["cases"]:
        raise ValueError("invalid authored fixture pack")
    seen = set()
    for case in pack["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"] or case["id"] in seen:
            raise ValueError("fixture cases require unique identifiers")
        seen.add(case["id"])
        catalog = case.get("catalog")
        if not isinstance(catalog, list) or not catalog or not all(
                isinstance(row, dict) and isinstance(row.get("parent_asin"), str) and row["parent_asin"] for row in catalog):
            raise ValueError("fixture catalog requires identified rows")
        identifiers = {row["parent_asin"] for row in catalog}
        if len(identifiers) != len(catalog):
            raise ValueError("fixture catalog identifiers must be unique")
        variants = case.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError("fixture requires variants")
        variant_ids = set()
        for variant in variants:
            if not isinstance(variant, dict) or not isinstance(variant.get("id"), str) or not variant["id"] \
                    or variant["id"] in variant_ids:
                raise ValueError("fixture variants require unique identifiers")
            variant_ids.add(variant["id"])
            steps = variant.get("steps")
            if not isinstance(steps, list) or not steps or not all(isinstance(step, dict) for step in steps) \
                    or not any(step.get("op") == "respond" for step in steps):
                raise ValueError("fixture requires observed turns")
            for step in steps:
                if not isinstance(step, dict) or step.get("op") not in {"respond", "retry", "conflict", "reset", "forget"}:
                    raise ValueError("unsupported fixture operation")
                if step["op"] in {"respond", "conflict"} and not isinstance(step.get("text"), str):
                    raise ValueError("fixture turn requires text")
                expect = step.get("expect", {})
                if not isinstance(expect, dict) or set(expect) - {"active", "absent", "retired", "leaders", "ahead", "statuses"}:
                    raise ValueError("unsupported fixture expectation")
                expected_ids = list(expect.get("leaders", [])) + [rule["id"] for rule in expect.get("statuses", [])]
                for values in expect.get("ahead", {}).values():
                    expected_ids.extend(values)
                if not set(expected_ids) <= identifiers or "leaders" in expect and not expect["leaders"]:
                    raise ValueError("fixture expectation must use nonempty catalog choices")
        for spec in case.get("compare", []):
            selected = spec.get("variants", variant_ids)
            if spec.get("kind") not in {"response", "union_membership", "ranked_prefix", "state_projection"} \
                    or len(set(selected)) < 2 or not set(selected) <= variant_ids:
                raise ValueError("invalid fixture comparison")


def evaluate_pack(mode: str = "fullwidth", *, pack: dict | None = None, agent_factory: Callable = Agent) -> dict:
    if mode not in MODES:
        raise ValueError("unsupported validation mode")
    pack = deepcopy(authored_pack() if pack is None else pack)
    _validate_pack(pack)
    sources = _source_hashes()
    cases = []
    with tempfile.TemporaryDirectory(prefix="authored-validation-") as directory:
        for index, case in enumerate(pack["cases"]):
            path = Path(directory) / f"catalog-{index}.jsonl"
            path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                                    for row in case["catalog"]), encoding="utf-8")
            variants = [_run_variant(path, case, variant, mode, agent_factory) for variant in case["variants"]]
            comparisons = [_compare(variants, spec) for spec in case.get("compare", [])]
            checks = [check for variant in variants for check in variant["checks"]] + comparisons
            cases.append({"id": case["id"], "status": _status(checks), "variants": variants, "checks": comparisons})
    source_check = _check("source_stable_during_run", sources == _source_hashes())
    all_checks = [source_check] + [check for case in cases for check in case["checks"]] + [
        check for case in cases for variant in case["variants"] for check in variant["checks"]]
    counts = Counter(check["status"] for check in all_checks)
    errors = [event["error"] for case in cases for variant in case["variants"]
              for event in variant["operations"] if "error" in event]
    errors.extend(check["observed"] for case in cases for variant in case["variants"] for check in variant["checks"]
                  if check["check"] in {"variant_setup", "agent_closed"} and check["status"] == "error")
    availability = {name: Counter(_mapping(_mapping(event.get("diagnostics", {}).get("stage_receipts")).get(name)).get(
        "available", False) is True for case in cases for variant in case["variants"]
        for event in variant["operations"] if "response" in event) for name in STAGES}
    return _json_safe({"schema": SCHEMA, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": mode, "purpose": "authored engineering validation; no benchmark success estimate",
            "fixture_sha256": _digest(pack), "fixture": pack, "source_hashes": sources,
            "config": asdict(MODES[mode]), "config_sha256": _digest(asdict(MODES[mode])),
            "status": _status(all_checks), "case_count": len(cases),
            "case_status_counts": dict(Counter(case["status"] for case in cases)),
            "check_status_counts": dict(counts), "checks": [source_check], "cases": cases,
            "errors": {"expected": sum(item["expected"] for item in errors),
                       "unexpected": sum(not item["expected"] for item in errors), "details": errors},
            "stage_availability": {name: {"available": count[True], "unavailable": count[False]}
                                   for name, count in availability.items()},
            "limitations": ["Small authored catalogs do not estimate unseen-catalog accuracy.",
                            "Constraint witnesses cover displayed items; missing coverage is unavailable, never a pass.",
                            "No optional model, external service, shared-profile mode, or concurrent-request coverage."]})


def write_report(report: dict, output: Path) -> None:
    """Create a report only inside this checkout's ignored artifact directory."""
    output = output.resolve()
    artifact_root = (ROOT / "artifacts").resolve()
    if not output.is_relative_to(artifact_root) or output == artifact_root:
        raise ValueError("report must be inside the ignored artifacts directory")
    ignored = subprocess.run(["git", "check-ignore", "-q", "--", str(output)], cwd=ROOT, check=False)
    if ignored.returncode != 0:
        raise ValueError("report path is not ignored")
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the embedded authored conversation checks")
    parser.add_argument("--mode", choices=tuple(MODES), default="fullwidth")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    report = evaluate_pack(args.mode)
    write_report(report, args.output)
    print(json.dumps({key: report[key] for key in ("mode", "status", "case_count", "case_status_counts", "check_status_counts")}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
