"""Record the public agent's actual conversation and portable evidence receipts."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import html
import json
import math
from pathlib import Path
import socket
import time
from unittest.mock import patch

from experiments.submission_evaluate import _public_agent, source_receipt, verified_metrics
from mercury.lexical.config import DEFAULT_AGENT_CONFIG
from mercury.lexical.diagnostics import signature
from mercury.lexical.vector_index import catalog_sha256
from mercury.model_assets import file_sha256


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "submission-demo-v1"
MESSAGES = (
    "I'm looking for bags. A key requirement is: black; leather; adjustable strap.",
    "Correction: make that blue and canvas, but keep the adjustable strap.",
    "I have no preference for color. No leather, please.",
    "Those options aren't right.",
    "None of these work for me.",
)
ATTRIBUTES = {None, "category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case", "other"}
EVIDENCE_FIELDS = ("evidence_id", "attribute", "value", "source_turn", "source_kind", "operation",
                   "scope", "polarity", "hard", "active", "retired_turn", "retirement_reason",
                   "raw_chunk_available", "derivation")
WITNESS_FIELDS = ("field", "match_kind", "scope", "normalized_phrase", "catalog_value", "requested_value", "mode")
CAPABILITY_FLAGS = ("requested", "loaded", "effective", "attempted", "contributed", "returned_results", "score_called")


def _fields(value: dict, fields: tuple[str, ...]) -> dict:
    return {key: deepcopy(value[key]) for key in fields if key in value}


def _sources() -> dict:
    return {**source_receipt(), "demo/submission.py": file_sha256(Path(__file__))}


def _identifiers(catalog: Path) -> set[str]:
    identifiers = set()
    with catalog.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            identifier = row.get("parent_asin") if isinstance(row, dict) else None
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                raise ValueError("Catalog requires unique nonempty identifiers")
            identifiers.add(identifier)
    if not identifiers:
        raise ValueError("Catalog is empty")
    return identifiers


def _response(response: object, identifiers: set[str]) -> dict:
    if not isinstance(response, dict) or not isinstance(response.get("message"), str) \
            or "ask_attribute" not in response or response.get("ask_attribute") not in ATTRIBUTES:
        raise ValueError("Invalid public response")
    rows = response.get("recommendations")
    if not isinstance(rows, list) or len(rows) > 10:
        raise ValueError("Invalid recommendation width")
    shown = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("parent_asin"), str) or row["parent_asin"] not in identifiers \
                or type(row.get("score")) not in (int, float) or not math.isfinite(row["score"]):
            raise ValueError("Response contains an illegal recommendation")
        shown.append(row["parent_asin"])
    if len(shown) != len(set(shown)):
        raise ValueError("Response contains duplicate identifiers")
    usage = response.get("usage")
    if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0
                                          for key in ("prompt_tokens", "completion_tokens")):
        raise ValueError("Invalid response usage")
    return {"message": response["message"], "ask_attribute": response["ask_attribute"],
            "recommendations": [_fields(row, ("parent_asin", "score")) for row in rows],
            "usage": _fields(usage, ("prompt_tokens", "completion_tokens"))}


def _diagnostics(raw: object, response: dict, runtime_catalog_digest: str, count: int) -> dict:
    if not isinstance(raw, dict) or raw.get("request_succeeded") is not True or raw.get("state_committed") is not True:
        raise ValueError("Response has no successful transaction receipt")
    identity = raw.get("identity", {})
    if (identity.get("catalog_sha256") != runtime_catalog_digest or identity.get("catalog_count") != count
            or identity.get("configuration", {}).get("agent") != asdict(DEFAULT_AGENT_CONFIG)
            or identity.get("config_sha256") != signature(identity.get("configuration"))
            or identity.get("binding", {}).get("complete") is not True):
        raise ValueError("Live receipt does not bind the selected configuration and catalog")
    shown = [row["parent_asin"] for row in response["recommendations"]]
    returned = raw.get("stage_receipts", {}).get("returned", {})
    if (returned.get("available") is not True or returned.get("complete") is not True
            or returned.get("ids") != shown or returned.get("count") != len(shown)
            or returned.get("sha256") != signature(shown)):
        raise ValueError("Returned-stage receipt does not match the actual response")
    paging = raw.get("paging", {})
    if paging.get("enabled") is not True or paging.get("returned_ids") != shown:
        raise ValueError("Paging receipt does not match the actual public response")
    evidence = raw.get("evidence")
    if not isinstance(evidence, dict) or any(not isinstance(evidence.get(key), list) for key in ("active", "retired")):
        raise ValueError("Active and retired evidence receipts are required")
    capabilities = raw.get("effective_capabilities", {})
    components = capabilities.get("components")
    required = ("sparse_retrieval", "exact_constraint_index", "vector_rerank", "neural_rerank")
    if not isinstance(components, dict) or any(not isinstance(components.get(name), dict) or any(
            type(components[name].get(flag)) is not bool for flag in ("requested", "loaded", "effective")) for name in required):
        raise ValueError("Requested, loaded and effective capability receipts are required")
    constraints = raw.get("constraint_checks")
    if not isinstance(constraints, list) or {row.get("parent_asin") for row in constraints} != set(shown):
        raise ValueError("Shown products require constraint receipts")
    return {
        "identity": _fields(identity, ("catalog_sha256", "catalog_count", "config_sha256", "runtime_source_sha256")),
        "request_succeeded": True, "state_committed": True, "cache_hit": raw.get("cache_hit"),
        "evidence": {**_fields(evidence, ("active_count", "active_complete", "retired_count", "retired_complete")),
                     "active": [_fields(row, EVIDENCE_FIELDS) for row in evidence["active"]],
                     "retired": [_fields(row, EVIDENCE_FIELDS) for row in evidence["retired"]]},
        "stage_receipts": {name: _fields(receipt, ("count", "sha256", "available", "complete"))
                           for name, receipt in raw["stage_receipts"].items()},
        "output_width": _fields(raw.get("output_width", {}),
                                ("requested", "returned", "full_width", "ambiguity_deferred", "policy_limit", "reason")),
        "question": _fields(raw.get("question", {}), ("attribute", "information_gain", "answerability", "expected_value")),
        "paging": _fields(paging, ("enabled", "triggered", "reset", "stable_head", "reason", "advances",
                                   "prior_seen", "new_exposures", "repeated_exposures", "base_ids", "returned_ids",
                                   "width_preserved", "violation_quota_preserved", "reset_replayed_base")),
        "capabilities": {name: _fields(components[name], CAPABILITY_FLAGS) for name in required},
        "fallbacks": list(raw.get("fallbacks", [])), "ranking_faults": list(capabilities.get("ranking_faults", [])),
        "constraint_checks": [{"parent_asin": row["parent_asin"], "evidence": [
            {**_fields(item, ("evidence_id", "value", "source_turn", "status")),
             "witnesses": [_fields(witness, WITNESS_FIELDS) for witness in item.get("witnesses", [])]}
            for item in row.get("evidence", [])]} for row in constraints],
    }


def _healthy(turn: dict) -> bool:
    receipt = turn["diagnostics"]
    return not receipt["fallbacks"] and not receipt["ranking_faults"] and all(
        not flags["requested"] or flags["loaded"] for flags in receipt["capabilities"].values())


def _transcript(report: dict) -> str:
    lines = ["Recorded public API conversation", "Authored messages; no target or per-conversation success score.",
             f"Runtime status: {report['status']}", f"Catalog rows: {report['catalog_count']}"]
    for turn in report["turns"]:
        response, diagnostics = turn["response"], turn["diagnostics"]
        identifiers = [row["parent_asin"] for row in response["recommendations"]]
        lines.extend(("", f"Turn {turn['turn']}", f"User: {turn['user_message']}", f"Agent: {response['message']}",
                      f"Shown IDs: {', '.join(identifiers) or '(none)'}",
                      f"Measured response time: {turn['latency_seconds']:.6f} seconds",
                      "Active evidence: " + json.dumps(diagnostics["evidence"]["active"], ensure_ascii=False),
                      "Capabilities: " + json.dumps(diagnostics["capabilities"], sort_keys=True),
                      "Paging: " + json.dumps(diagnostics["paging"], sort_keys=True),
                      "Fallbacks: " + json.dumps(diagnostics["fallbacks"])))
    if report["evaluation"] is not None:
        lines.extend(("", "Separate verified aggregate evaluation", report["evaluation"]["scope"],
                      json.dumps(report["evaluation"]["metrics"], sort_keys=True)))
    return "\n".join(lines) + "\n"


def _html(report: dict, transcript: str) -> str:
    evidence = html.escape(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            "<title>Recorded conversation</title><style>body{max-width:72rem;margin:3rem auto;padding:0 1.2rem;"
            "font:16px/1.6 system-ui;background:#101820;color:#e9eef4}h1{font-size:2rem}pre{white-space:pre-wrap;"
            "overflow-wrap:anywhere;padding:1.2rem;background:#192735;border-radius:.5rem}summary{cursor:pointer}"
            "a{color:#8ed9ee}</style><h1>Recorded conversation</h1><p>Actual public API responses and measured timings. "
            "This authored conversation has no target or success score.</p>"
            f"<p>Runtime status: {html.escape(report['status'])}</p><pre>{html.escape(transcript)}</pre>"
            f"<details><summary>Sanitized evidence receipt</summary><pre>{evidence}</pre></details></html>")


def _deny_network(*args, **kwargs):
    raise RuntimeError("Network access is disabled for this recording")


@patch.object(socket, "create_connection", _deny_network)
@patch.object(socket.socket, "connect", _deny_network)
def build_demo(output: Path, catalog: Path, *, messages: tuple[str, ...] = MESSAGES,
               evaluation_report: Path | None = None, evaluation_sha256: str | None = None) -> dict:
    output, catalog = Path(output), Path(catalog)
    if output.exists():
        raise FileExistsError("Recording destination already exists")
    if not 1 <= len(messages) <= 10 or any(not isinstance(message, str) or not message.strip() or len(message) > 8000
                                         for message in messages):
        raise ValueError("Supply one to ten nonempty authored messages of at most 8000 characters")
    if (evaluation_report is None) != (evaluation_sha256 is None):
        raise ValueError("Evaluation report and expected checksum must be supplied together")
    agent_class = _public_agent()
    before, digest = _sources(), file_sha256(catalog)
    identifiers, runtime_digest = _identifiers(catalog), catalog_sha256(catalog)
    evaluation = verified_metrics(evaluation_report, evaluation_sha256, digest) if evaluation_report else None
    started = time.perf_counter()
    agent = agent_class(catalog)
    cold = time.perf_counter() - started
    turns = []
    try:
        if asdict(agent.config) != asdict(DEFAULT_AGENT_CONFIG):
            raise ValueError("Public entry point did not use the selected default")
        agent.reset("recorded-conversation", {})
        for number, message in enumerate(messages, 1):
            started = time.perf_counter()
            response = _response(agent.respond("recorded-conversation", message, number, 10), identifiers)
            latency = time.perf_counter() - started
            diagnostics = _diagnostics(agent.last_diagnostics, response, runtime_digest, len(identifiers))
            turns.append({"turn": number, "user_message": message, "response": response,
                          "diagnostics": diagnostics, "latency_seconds": latency})
    finally:
        agent.close()
    if before != _sources() or digest != file_sha256(catalog):
        raise ValueError("Runtime or catalog changed during recording")
    if evaluation_report and file_sha256(evaluation_report) != evaluation_sha256:
        raise ValueError("Evaluation report changed during recording")
    status = "healthy" if all(_healthy(turn) for turn in turns) else "degraded"
    if evaluation is not None and status != "healthy":
        raise ValueError("A degraded recording cannot represent a healthy evaluation")
    report = {"schema": SCHEMA, "status": status, "catalog_sha256": digest, "catalog_count": len(identifiers),
              "source_hashes": before, "config": asdict(DEFAULT_AGENT_CONFIG),
              "cold_start_seconds": cold, "evaluation": evaluation, "turns": turns,
              "scope": "Authored conversation only; aggregate evaluation is separate.",
              "sanitization": "Raw catalog fields, profiles and evaluation-session records are omitted; visible responses are unchanged."}
    transcript = _transcript(report)
    files = {"evidence.json": json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
             "transcript.txt": transcript, "index.html": _html(report, transcript)}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        with (output / name).open("x", encoding="utf-8") as handle:
            handle.write(value)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--message", action="append", dest="messages")
    parser.add_argument("--evaluation-report", type=Path)
    parser.add_argument("--evaluation-sha256")
    args = parser.parse_args()
    report = build_demo(args.output, args.catalog, messages=tuple(args.messages) if args.messages else MESSAGES,
                        evaluation_report=args.evaluation_report, evaluation_sha256=args.evaluation_sha256)
    print(json.dumps({"status": report["status"], "turn_count": len(report["turns"]),
                      "verified_evaluation_attached": report["evaluation"] is not None}))


if __name__ == "__main__":
    main()
