"""Verify the public entry point with the unchanged evaluator and local receipts."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import socket
import statistics
import time
from unittest.mock import patch

from experiments.run import peak_rss_bytes
from mercury.lexical import Agent as LexicalAgent, FULL_WIDTH_CONFIG
from mercury.lexical.config import DEFAULT_AGENT_CONFIG
from mercury.model_assets import file_sha256


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mercury-submission-evaluation-v1"
EVALUATOR_SHA256 = "79a5ea06f9a1b8c5036f30efa85dc1f36b8f6b06eb8feb8f545dfa767bc45564"


def source_receipt() -> dict[str, str]:
    paths = sorted((ROOT / "mercury").rglob("*.py")) + [
        ROOT / "agent.py", ROOT / "starter/agent.py", ROOT / "evaluator/local_evaluator.py",
        ROOT / "experiments/submission_evaluate.py",
    ]
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}


def _public_agent():
    module = importlib.import_module("agent")
    compatibility = importlib.import_module("starter.agent")
    if (Path(module.__file__).resolve() != ROOT / "agent.py"
            or Path(compatibility.__file__).resolve() != ROOT / "starter/agent.py"
            or module.Agent is not LexicalAgent or compatibility.Agent is not module.Agent
            or Path(inspect.getfile(module.Agent)).resolve() != ROOT / "mercury/lexical/agent.py"):
        raise ValueError("Public entry point does not select the expected implementation")
    return module.Agent


def _deny_network(*args, **kwargs):
    raise RuntimeError("Network access is disabled for submission verification")


def _aggregate(sessions: object) -> dict:
    """Recompute outcomes independently before publishing aggregate metrics."""
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("Completed evaluation sessions are required")
    identifiers, hits, ranks, turns = set(), [], [], []
    for row in sessions:
        if not isinstance(row, dict):
            raise ValueError("Invalid evaluation outcome")
        identifier = row.get("sample_id")
        hit, rank, turn = row.get("hit"), row.get("best_rank"), row.get("first_hit_turn")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Evaluation session IDs must be unique nonempty strings")
        identifiers.add(identifier)
        if type(hit) is not bool or (hit and (
            type(rank) is not int or not 1 <= rank <= 10 or type(turn) is not int or not 1 <= turn <= 10
        )) or (not hit and (rank is not None or turn is not None)):
            raise ValueError("Invalid hit, rank or turn")
        reciprocal = 1 / rank if hit else 0.0
        if type(row.get("reciprocal_rank")) not in (int, float) or not math.isclose(
            row["reciprocal_rank"], reciprocal, rel_tol=0, abs_tol=1e-12,
        ):
            raise ValueError("Reciprocal rank does not match the recorded rank")
        hits.append(hit)
        ranks.append(reciprocal)
        turns.append(turn if hit else 11)
    mrr, mttc, hit_rate = (round(statistics.fmean(values), 6) for values in (ranks, turns, hits))
    efficiency = max(0.0, min(1.0, (11 - mttc) / 10))
    return {"sample_count": len(sessions), "hit_rate_at_10": hit_rate, "mrr": mrr, "mttc": mttc,
            "efficiency": round(efficiency, 6),
            "recommended_technical_score": round(.5 * hit_rate + .3 * mrr + .2 * efficiency, 6)}


def verified_metrics(path: Path, expected_sha256: str, catalog_sha256: str) -> dict:
    """Export only metrics bound to the current default, code and catalog."""
    _public_agent()
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Evaluation report checksum mismatch")
    report = json.loads(raw)
    if (not isinstance(report, dict) or report.get("schema") != SCHEMA
            or report.get("mode") != "selected" or report.get("valid") is not True
            or report.get("catalog_sha256") != catalog_sha256
            or report.get("source_hashes") != source_receipt()
            or report.get("config") != asdict(DEFAULT_AGENT_CONFIG)
            or report.get("evaluator_sha256") != EVALUATOR_SHA256):
        raise ValueError("Evaluation identity does not match the current default")
    if source_receipt()["evaluator/local_evaluator.py"] != EVALUATOR_SHA256:
        raise ValueError("Evaluator has changed")
    measurement = report.get("measurement")
    if not isinstance(measurement, dict) or any(measurement.get(name) is not False for name in (
        "source_changed", "catalog_changed", "dataset_changed",
    )) or any(type(measurement.get(name)) is not int or measurement[name] != 0 for name in (
        "errors", "fallback_turns",
    )) or measurement.get("startup_fallbacks") != []:
        raise ValueError("Evaluation was not a healthy unchanged run")
    result = report.get("result")
    if not isinstance(result, dict):
        raise ValueError("Evaluation outcomes are missing")
    aggregate = _aggregate(result.get("sessions"))
    if any(type(result.get(key)) not in (int, float) or not math.isfinite(result[key])
           or result[key] != value for key, value in aggregate.items()):
        raise ValueError("Aggregate metrics do not match the outcomes")
    return {"report_sha256": expected_sha256, "metrics": aggregate,
            "scope": "verified local evaluation; no private-test or placement claim"}


class ObservedAgent:
    def __init__(self, inner, identifiers: set[str], full_width: bool):
        self.inner, self.identifiers, self.full_width = inner, identifiers, full_width
        self.traces: list[list[dict]] = []
        self.errors = 0
        self.latencies: list[float] = []
        self.widths: Counter = Counter()
        self.candidate_shortfalls = 0

    def reset(self, session_id, profile):
        self.traces.append([])
        self.inner.reset(session_id, profile)

    def respond(self, session_id, message, turn, top_k):
        started = time.perf_counter()
        try:
            response = self.inner.respond(session_id, message, turn, top_k)
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                raise ValueError("Response must contain a message")
            from evaluator.local_evaluator import ALLOWED_ATTRIBUTES

            attribute = response.get("ask_attribute")
            if attribute is not None and (not isinstance(attribute, str) or attribute not in ALLOWED_ATTRIBUTES):
                raise ValueError("Invalid clarification attribute")
            usage = response.get("usage")
            if not isinstance(usage, dict) or any(type(usage.get(name)) is not int or usage[name] < 0
                                                  for name in ("prompt_tokens", "completion_tokens")):
                raise ValueError("Invalid token usage")
            recommendations = response.get("recommendations")
            if not isinstance(recommendations, list) or len(recommendations) > min(top_k, 10):
                raise ValueError("Response exceeds the requested width")
            if any(not isinstance(row, dict) or not isinstance(row.get("parent_asin"), str)
                   or type(row.get("score")) not in (int, float) or not math.isfinite(row["score"])
                   for row in recommendations):
                raise ValueError("Invalid recommendation")
            returned = [row["parent_asin"] for row in recommendations]
            if len(returned) != len(set(returned)) or not set(returned) <= self.identifiers:
                raise ValueError("Response contains illegal catalog identifiers")
            diagnostics = deepcopy(self.inner.last_diagnostics)
            if diagnostics.get("state_committed") is not True:
                raise ValueError("Successful response lacks a committed-state receipt")
            ranked = diagnostics.get("stage_receipts", {}).get("ranked_prefix", {})
            if self.full_width:
                context = diagnostics.get("stage_receipts", {}).get("question_context", {})
                count = context.get("count")
                if (context.get("available") is not True or type(count) is not int or count < 0
                        or ranked.get("available") is not True or ranked.get("complete") is not True
                        or ranked.get("count") != len(ranked.get("ids", []))
                        or len(returned) != min(top_k, 10, count)
                        or returned != ranked.get("ids", [])[:top_k]):
                    raise ValueError("Full-width control does not expose the raw ranked prefix")
                self.candidate_shortfalls += count < min(top_k, 10)
            self.widths[len(returned)] += 1
            self.traces[-1].append({"turn": turn, "message": message,
                                   "response": deepcopy(response), "diagnostics": diagnostics})
            return response
        except Exception as error:
            self.errors += 1
            self.traces[-1].append({"turn": turn, "error_type": type(error).__name__})
            raise
        finally:
            self.latencies.append(time.perf_counter() - started)


def _write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


@patch.object(socket, "create_connection", _deny_network)
@patch.object(socket.socket, "connect", _deny_network)
def evaluate_submission(catalog: Path, dataset: Path, output: Path, *, full_width: bool = False) -> dict:
    catalog, dataset, output = Path(catalog), Path(dataset), Path(output)
    if type(full_width) is not bool:
        raise ValueError("full_width must be boolean")
    if output.exists():
        raise FileExistsError("Evaluation destination already exists")
    agent_class = _public_agent()
    official = importlib.import_module("evaluator.local_evaluator")
    before = source_receipt()
    if (Path(official.__file__).resolve() != ROOT / "evaluator/local_evaluator.py"
            or before["evaluator/local_evaluator.py"] != EVALUATOR_SHA256):
        raise ValueError("Official evaluator identity mismatch")
    config = FULL_WIDTH_CONFIG if full_width else DEFAULT_AGENT_CONFIG
    identity = {"schema": SCHEMA, "mode": "full_width" if full_width else "selected",
                "source_hashes": before, "catalog_sha256": file_sha256(catalog),
                "dataset_sha256": file_sha256(dataset), "config": asdict(config),
                "evaluator_sha256": EVALUATOR_SHA256, "network": "socket connections denied",
                "scope": "local evaluation; public data is development evidence"}
    samples = official.load_jsonl(dataset)
    if not samples:
        raise ValueError("Evaluation dataset is empty")
    identifiers, categories, products = official.catalog_index(catalog)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "registration.json", identity)
    started = time.perf_counter()
    inner = agent_class(catalog, config=config) if full_width else agent_class(catalog)
    cold = time.perf_counter() - started
    observed = ObservedAgent(inner, identifiers, full_width)
    try:
        if asdict(inner.config) != asdict(config):
            raise ValueError("Public entry point used an unexpected configuration")
        started = time.perf_counter()
        result = official.evaluate(observed, samples, identifiers, categories, products)
        elapsed = time.perf_counter() - started
        latencies = sorted(observed.latencies)
        measurement = {"cold_start_seconds": cold, "evaluation_seconds": elapsed,
                       "p50_seconds": statistics.median(latencies),
                       "p95_seconds": latencies[int((len(latencies) - 1) * .95)],
                       "peak_rss_bytes": peak_rss_bytes(), "errors": observed.errors,
                       "candidate_shortfall_turns": observed.candidate_shortfalls,
                       "widths": dict(observed.widths), "startup_fallbacks": list(inner.startup_fallbacks),
                       "fallback_turns": sum(bool(turn.get("diagnostics", {}).get("fallbacks"))
                                             for session in observed.traces for turn in session),
                       "source_changed": before != source_receipt(),
                       "catalog_changed": identity["catalog_sha256"] != file_sha256(catalog),
                       "dataset_changed": identity["dataset_sha256"] != file_sha256(dataset)}
        aggregate = _aggregate(result["sessions"])
        valid = not any(measurement[name] for name in (
            "errors", "startup_fallbacks", "fallback_turns", "source_changed", "catalog_changed", "dataset_changed",
        )) and aggregate == {key: result[key] for key in aggregate}
        report = {**identity, "valid": valid, "result": result, "measurement": measurement}
        _write(output / "report.json", report)
        _write(output / "traces.json", observed.traces)
        if not valid:
            raise RuntimeError("Submission verification failed; inspect the receipt")
        return report
    finally:
        inner.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/public_set.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-width", action="store_true")
    args = parser.parse_args()
    report = evaluate_submission(args.catalog, args.dataset, args.output, full_width=args.full_width)
    print(json.dumps({key: report["result"][key] for key in
                      ("recommended_technical_score", "hit_rate_at_10", "mrr", "mttc")}))


if __name__ == "__main__":
    main()
