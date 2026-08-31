# Historical research demo. Current public pipeline: python -m demo.submission.
"""Create a portable replay of real responses with separately verified metrics."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import statistics
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from mercury.agent import Agent
from mercury.catalog import negated_match
from mercury.config import Config
from mercury.model_assets import file_sha256
from mercury.ranking import preference_evidence
from mercury.types import Preference, Product


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SHA256 = "79a5ea06f9a1b8c5036f30efa85dc1f36b8f6b06eb8feb8f545dfa767bc45564"
MESSAGES = (
    "I need a black leather shoulder bag with an adjustable strap.",
    "Correction: make that blue, made of canvas, but keep the adjustable strap.",
    "I no longer have a color preference. No leather, please.",
)
METRICS = ("sample_count", "hit_rate_at_10", "mrr", "mttc", "efficiency", "technical_score")


def source_receipt() -> dict[str, str]:
    paths = sorted((ROOT / "mercury").rglob("*.py")) + [ROOT / "evaluator/local_evaluator.py"]
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}


def _aggregate(sessions: object) -> dict:
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("Evaluation must contain completed sessions")
    ranks, turns, hits = [], [], []
    identifiers = set()
    for row in sessions:
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
            raise ValueError("Invalid evaluation session")
        if row["sample_id"] in identifiers:
            raise ValueError("Duplicate evaluation session")
        identifiers.add(row["sample_id"])
        hit, rank, turn = row.get("hit"), row.get("best_rank"), row.get("first_hit_turn")
        if type(hit) is not bool or (hit and (
            type(rank) is not int or not 1 <= rank <= 10 or type(turn) is not int or not 1 <= turn <= 10
        )) or (not hit and (rank is not None or turn is not None)):
            raise ValueError("Invalid evaluation outcome")
        reciprocal = 1 / rank if hit else 0.0
        if type(row.get("reciprocal_rank")) not in (int, float) or not math.isclose(
            row["reciprocal_rank"], reciprocal, rel_tol=0, abs_tol=1e-12,
        ):
            raise ValueError("Edited reciprocal rank")
        hits.append(hit)
        ranks.append(reciprocal)
        turns.append(turn if hit else 11)
    result = {"sample_count": len(sessions), "hit_rate_at_10": round(sum(hits) / len(hits), 6),
              "mrr": round(statistics.fmean(ranks), 6), "mttc": round(statistics.fmean(turns), 6)}
    efficiency = max(0.0, min(1.0, (11 - result["mttc"]) / 10))
    return {**result, "efficiency": round(efficiency, 6),
            "technical_score": round(.5 * result["hit_rate_at_10"] + .3 * result["mrr"] + .2 * efficiency, 6)}


def validated_metrics(path: Path, expected_sha256: str, catalog_sha256: str, config: Config) -> dict:
    """Validate a checksum-bound evaluation file; never accept a metrics dict."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Evaluation report checksum mismatch")
    report = json.loads(raw)
    current = source_receipt()
    if current["evaluator/local_evaluator.py"] != EVALUATOR_SHA256:
        raise ValueError("Evaluator is not unchanged")
    if (not isinstance(report, dict) or report.get("schema") != "mercury-evaluation-suite-v1"
            or report.get("source_changed_during_run") is not False
            or report.get("catalog_sha256") != catalog_sha256):
        raise ValueError("Evaluation identity mismatch")
    recorded = report.get("source_hashes")
    if not isinstance(recorded, dict):
        raise ValueError("Evaluation source receipt missing")
    runtime = {key: value for key, value in recorded.items()
               if key.startswith("mercury/") and key.endswith(".py") or key == "evaluator/local_evaluator.py"}
    if runtime != current:
        raise ValueError("Evaluation runtime source mismatch")
    matches = [run for run in report.get("runs", []) if isinstance(run, dict) and run.get("kind") == "config"
               and isinstance(run.get("config"), dict)
               and Config.from_dict(run["config"]).to_dict() == config.to_dict()]
    if len(matches) != 1:
        raise ValueError("Evaluation requires one matching selected configuration")
    run = matches[0]
    measured = run.get("metrics")
    aggregate = _aggregate(run.get("sessions"))
    if not isinstance(measured, dict) or any(
        type(measured.get(key)) not in (int, float) or not math.isfinite(measured[key])
        or measured[key] != value for key, value in aggregate.items()
    ):
        raise ValueError("Evaluation aggregate metrics do not match recorded outcomes")
    if measured.get("startup_fallbacks") != {} or measured.get("fallback_turns") != 0:
        raise ValueError("Evaluation fallback run cannot certify healthy execution")
    return {"report_sha256": digest, "metrics": aggregate,
            "classification": "matching evaluation receipt; replay itself has no score"}


def _preference(preference: Preference) -> dict:
    return {key: getattr(preference, key) for key in (
        "attribute", "value", "polarity", "hard", "active", "confidence", "scope", "alternative_group", "depends_on",
    )}


def field_evidence(product: Product, preference: Preference) -> dict:
    """Show literal field witnesses; absence and conflicting text remain unknown."""
    witnesses, signals = [], set()
    if preference.active and preference.polarity and preference.scope is None:
        if preference.attribute == "budget" and product.price is not None:
            signal = preference_evidence(product, preference)
            if signal:
                signals.add(1 if signal > 0 else -1)
                witnesses.append({"field": "price", "value": product.price,
                                  "lower_bound": product.price_lower_bound})
        else:
            pattern = re.compile(r"(?<!\w)" + re.escape(preference.value) + r"(?!\w)", re.I)
            for field, content in product.fields.items():
                for match in pattern.finditer(content):
                    if re.search(r"\b(?:faux|imitation|synthetic)\s*[- ]?$", content[max(0, match.start() - 25):match.start()], re.I):
                        continue
                    signal = (-1 if negated_match(content, match.start(), match.end()) else 1) * preference.polarity
                    signals.add(signal)
                    witnesses.append({"field": field, "value": match.group(),
                                      "status": "supported" if signal > 0 else "contradicted"})
    status = "supported" if signals == {1} else "contradicted" if signals == {-1} else "unknown"
    return {"attribute": preference.attribute, "value": preference.value, "polarity": preference.polarity,
            "status": status, "fields": witnesses,
            "explanation": "Literal field evidence only; missing, conflicting or unverified scope stays unknown."}


def _record(catalog: Path, config: Config, messages: tuple[str, ...], intentional_missing: bool) -> dict:
    started = time.perf_counter()
    agent = Agent(catalog, config)
    try:
        startup = time.perf_counter() - started
        session = "portable-replay"
        agent.reset(session, {})
        turns = []
        for turn, message in enumerate(messages, 1):
            response = agent.respond(session, message, turn, 10)
            recommendations = response.get("recommendations") if isinstance(response, dict) else None
            if (not isinstance(recommendations, list) or len(recommendations) > 10
                    or not isinstance(response.get("message"), str)
                    or any(not isinstance(row, dict) or not isinstance(row.get("parent_asin"), str)
                           for row in recommendations)):
                raise ValueError("Replay response violates the contract")
            ids = [row["parent_asin"] for row in recommendations]
            if len(ids) != len(set(ids)) or not set(ids) <= set(agent.catalog.by_id):
                raise ValueError("Replay response contains illegal catalog IDs")
            diagnostics = agent.last_diagnostics
            capabilities = diagnostics["effective_capabilities"]
            neural = capabilities["components"]["neural_rerank"]
            if intentional_missing and not (
                neural["requested"] is True and neural["loaded"] is False and neural["effective"] is False
                and "neural_rerank" in agent.startup_fallbacks
            ):
                raise ValueError("Missing-model replay did not demonstrate the intended fallback")
            state = agent.sessions[session]
            active = state.active_preferences()
            turns.append({"turn": turn, "user_message": message,
                          "response": {"message": response["message"], "ask_attribute": response.get("ask_attribute"),
                                       "recommendations": [{"parent_asin": value} for value in ids]},
                          "active_preferences": [_preference(p) for p in active],
                          "retired_preferences": [_preference(p) for p in state.preferences if not p.active],
                          "runtime": capabilities, "fallbacks": list(diagnostics["fallbacks"]),
                          "latency_seconds": diagnostics["latency_seconds"],
                          "products": [{"parent_asin": value, "rank": rank,
                                        "evidence": [field_evidence(agent.catalog.by_id[value], p) for p in active]}
                                       for rank, value in enumerate(ids, 1)]})
        healthy = all(not turn["runtime"]["ranking_faults"] and all(
            not component["requested"] or component["loaded"] and component["effective"]
            for component in turn["runtime"]["components"].values()
        ) for turn in turns)
        return {"kind": "intentional_missing_model" if intentional_missing else "selected_configuration",
                "status": "intentional_fallback" if intentional_missing else "healthy" if healthy else "degraded",
                "catalog_sha256": agent.catalog.sha256, "catalog_count": len(agent.catalog.by_id),
                "cold_start_seconds": startup, "turns": turns}
    finally:
        agent.close()


def _terminal(report: dict) -> tuple[str, list]:
    lines = ["ACTUAL API EVIDENCE REPLAY", "Presentation pacing: 180 seconds; recorded latency is measured separately."]
    if report["evaluation"]:
        lines.append("Verified evaluation aggregates: " + json.dumps(report["evaluation"]["metrics"], sort_keys=True))
    else:
        lines.append("No evaluation score supplied or claimed.")
    for run in report["runs"]:
        lines.append(f"\n{run['kind']} | {run['status']}")
        for turn in run["turns"]:
            lines.extend((f"Turn {turn['turn']}: {turn['user_message']}",
                          "Response: " + turn["response"]["message"],
                          "IDs: " + ", ".join(row["parent_asin"] for row in turn["response"]["recommendations"]),
                          "Active: " + ", ".join(f"{p['attribute']}={p['value']} ({p['polarity']})" for p in turn["active_preferences"]),
                          "Retired: " + ", ".join(f"{p['attribute']}={p['value']}" for p in turn["retired_preferences"]),
                          "Neural requested/loaded/effective: " + "/".join(str(turn["runtime"]["components"]["neural_rerank"][key])
                                                                         for key in ("requested", "loaded", "effective"))))
    lines.append("Replay complete. Missing catalog evidence is unknown; this replay has no target or success score.")
    events = [[round(index * 179 / max(1, len(lines) - 1), 3), "o", line + "\r\n"] for index, line in enumerate(lines)]
    return "\n".join(lines) + "\n", events


def _html(report: dict, transcript: str) -> str:
    sections = []
    for run in report["runs"]:
        sections.append(f"<h2>{html.escape(run['kind'])}: {html.escape(run['status'])}</h2>")
        for turn in run["turns"]:
            sections.append(f"<h3>Turn {turn['turn']}</h3><p>{html.escape(turn['user_message'])}</p>"
                            f"<p>{html.escape(turn['response']['message'])}</p>")
            for product in turn["products"]:
                sections.append(f"<details><summary>{product['rank']}. {html.escape(product['parent_asin'])}</summary><ul>")
                for item in product["evidence"]:
                    fields = ", ".join(sorted({w["field"] for w in item["fields"]})) or "no verified field"
                    sections.append(f"<li>{html.escape(item['attribute'])}: {html.escape(item['value'])} — "
                                    f"<b>{item['status']}</b> ({html.escape(fields)})</li>")
                sections.append("</ul></details>")
    return ("<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Recorded API evidence</title><style>body{max-width:960px;margin:40px auto;padding:0 24px;font:16px/1.5 system-ui;background:#f7f8fa;color:#182333}"
            "pre{white-space:pre-wrap;background:#fff;padding:20px;border:1px solid #ddd}details{padding:8px;background:#fff;margin:6px 0}</style>"
            "<h1>Recorded API evidence</h1><p>Three-minute presentation replay. Timing below is narration pacing, not inference latency.</p>"
            f"<pre>{html.escape(transcript)}</pre>{''.join(sections)}"
            "<p>Complete active and retired facts, runtime capabilities and actual responses are in evidence.json. "
            "The terminal recording is replay.cast. No evaluation session traces are included.</p></html>")


def build_release(output: Path, catalog: Path, config_path: Path = ROOT / "configs/selected.json", *,
                  evaluation_report: Path | None = None, evaluation_sha256: str | None = None,
                  messages: tuple[str, ...] = MESSAGES) -> dict:
    output, catalog, config_path = Path(output), Path(catalog), Path(config_path)
    if output.exists():
        raise FileExistsError("Replay destination already exists")
    if not messages or len(messages) > 10 or any(not isinstance(m, str) or not m.strip() for m in messages):
        raise ValueError("Replay requires one to ten nonempty messages")
    if (evaluation_report is None) != (evaluation_sha256 is None):
        raise ValueError("Evaluation report and expected checksum must be supplied together")
    config = Config.load(config_path)
    sources, catalog_sha = source_receipt(), file_sha256(catalog)
    if sources["evaluator/local_evaluator.py"] != EVALUATOR_SHA256:
        raise ValueError("Evaluator is not unchanged")
    evaluation = validated_metrics(evaluation_report, evaluation_sha256, catalog_sha, config) if evaluation_report else None
    selected = _record(catalog, config, messages, False)
    if evaluation and selected["status"] != "healthy":
        raise ValueError("A degraded replay cannot represent the healthy evaluation")
    with tempfile.TemporaryDirectory(prefix="missing-model-") as missing:
        fallback = _record(catalog, replace(config, neural_rerank=True, artifact_dir=missing), messages, True)
    if (source_receipt() != sources or file_sha256(catalog) != catalog_sha
            or Config.load(config_path).to_dict() != config.to_dict()
            or any(run["catalog_sha256"] != catalog_sha for run in (selected, fallback))):
        raise ValueError("Replay inputs changed during execution")
    if evaluation_report and file_sha256(evaluation_report) != evaluation_sha256:
        raise ValueError("Evaluation report changed during replay")
    report = {"schema": "recorded-release-v1", "actual_agent_responses": True, "presentation_seconds": 180,
              "catalog_sha256": catalog_sha, "runtime_hashes": sources,
              "config_sha256": hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest(),
              "evaluation": evaluation, "runs": [selected, fallback]}
    transcript, events = _terminal(report)
    output.mkdir(parents=True, exist_ok=False)
    files = {"evidence.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
             "index.html": _html(report, transcript), "transcript.txt": transcript,
             "replay.cast": "\n".join(json.dumps(row) for row in [
                 {"version": 2, "width": 120, "height": 40, "duration": 180}, *events,
             ]) + "\n"}
    for name, text in files.items():
        with (output / name).open("x", encoding="utf-8") as handle:
            handle.write(text)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Record actual API responses, verified aggregates and a separate missing-model replay.")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/selected.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path)
    parser.add_argument("--evaluation-sha256")
    args = parser.parse_args()
    report = build_release(args.output, args.catalog, args.config, evaluation_report=args.evaluation_report,
                           evaluation_sha256=args.evaluation_sha256)
    print(_terminal(report)[0], end="")


if __name__ == "__main__":
    main()
