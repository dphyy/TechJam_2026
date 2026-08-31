# Historical research demo. Current public pipeline: python -m demo.submission.
"""Record a bounded source-span capability proof through the ordinary Agent API."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

from experiments.cycle2_capabilities import validate_response
from experiments.cycle2_evaluate import model_file_hashes
from experiments.run import source_hashes
from mercury.agent import Agent
from mercury.config import Config
from mercury.model_assets import file_sha256


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY / "configs/cycle4_role_evidence.json"
PROBE = (
    "I need a leather outer shell, not merely leather elbow patches.",
    "Actually, canvas outer shell.",
    "Actually, I have no material preference.",
)
INVENTED_ROWS = (
    {"parent_asin": "WHOLE", "title": "Field Jacket",
     "description": "A jacket with a leather outer shell."},
    {"parent_asin": "COMPONENT", "title": "Field Jacket",
     "features": ["Leather elbow patches."]},
    {"parent_asin": "CROSS_FIELD", "title": "Leather jacket",
     "description": "Canvas outer shell."},
    {"parent_asin": "UNKNOWN", "title": "Field Jacket",
     "description": "Canvas outer shell."},
)
CLAIM_BOUNDARY = (
    "Authored micro-catalog capability demonstration. It proves bounded source-span behavior and "
    "correction safety only; it is not technical-score, hidden-set, shopper, or organizer-private "
    "performance evidence."
)


def _safe(value: object) -> object:
    return json.loads(json.dumps(value, allow_nan=False))


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _configurations() -> dict[str, dict]:
    raw = CONFIG_PATH.read_bytes()
    enabled = Config.from_dict(json.loads(raw))
    if not enabled.role_evidence:
        raise ValueError("The registered role-evidence configuration must enable role_evidence")
    control_values = enabled.to_dict()
    control_values["role_evidence"] = False
    control = Config.from_dict(control_values)
    if {**enabled.to_dict(), "role_evidence": False} != control.to_dict():
        raise ValueError("Role-evidence replay control differs outside role_evidence")
    control_bytes = (json.dumps(control.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode()
    return {
        "enabled": {"config": enabled, "sha256": hashlib.sha256(raw).hexdigest(),
                    "source": str(CONFIG_PATH), "values": enabled.to_dict()},
        "control": {"config": control, "sha256": hashlib.sha256(control_bytes).hexdigest(),
                    "source": "derived:configs/cycle4_role_evidence.json", "values": control.to_dict()},
    }


def _inventory(catalog: Path, configurations: dict[str, dict]) -> dict:
    return {
        "source": source_hashes(),
        "catalog_sha256": file_sha256(catalog),
        "configs": {name: item["sha256"] for name, item in configurations.items()},
        "models": model_file_hashes(configurations["enabled"]["config"]),
    }


def _normalized(diagnostics: object) -> dict:
    if not isinstance(diagnostics, dict):
        raise RuntimeError("Capability replay requires diagnostics")
    normalized = {key: diagnostics.get(key) for key in ("preferences", "role_evidence", "ranked_ids")}
    if not isinstance(normalized["preferences"], list) or not isinstance(normalized["role_evidence"], dict) \
            or not isinstance(normalized["ranked_ids"], list):
        raise RuntimeError("Capability replay diagnostics are malformed")
    return _safe(normalized)


def _record_run(label: str, catalog: Path, config: Config, report: dict, manifest: dict, agent_factory) -> None:
    agent = None
    receipt = {"label": label, "config": config.to_dict(), "closed": False}
    manifest["agents"].append(receipt)
    try:
        started = time.perf_counter()
        agent = agent_factory(catalog, config)
        receipt["cold_start_seconds"] = time.perf_counter() - started
        receipt["catalog_sha256"] = agent.catalog.sha256
        receipt["product_count"] = len(agent.catalog.by_id)
        if receipt["catalog_sha256"] != manifest["before"]["catalog_sha256"]:
            raise RuntimeError("Capability replay catalog changed before agent initialization")
        if getattr(agent, "startup_fallbacks", None) != {}:
            raise RuntimeError("Capability replay startup health failed")
        agent.reset(f"role-evidence-{label}", {})
        for turn, message in enumerate(PROBE, 1):
            record = {"label": label, "turn": turn, "user_message": message, "response": None,
                      "diagnostics": None, "response_contract": None, "error": None}
            report["records"].append(record)
            started = time.perf_counter()
            try:
                response = agent.respond(f"role-evidence-{label}", message, turn, 10)
                record["latency_seconds"] = time.perf_counter() - started
                record["response"] = _safe(response)
                record["diagnostics"] = _safe(agent.last_diagnostics)
                record["response_contract"] = validate_response(response, set(agent.catalog.by_id))
                if record["response_contract"]["status"] != "passed":
                    raise RuntimeError("Capability replay response contract failed")
                if record["diagnostics"].get("fallbacks") != []:
                    raise RuntimeError("Capability replay turn health failed")
                _normalized(record["diagnostics"])
            except BaseException as error:
                record.setdefault("latency_seconds", time.perf_counter() - started)
                record["error"] = repr(error)
                raise
    finally:
        if agent is not None:
            try:
                agent.close()
                receipt["closed"] = True
            except Exception as error:
                receipt["close_error"] = repr(error)
                raise


def _records_by_label(records: list[dict], label: str) -> list[dict]:
    selected = [record for record in records if record["label"] == label]
    if len(selected) != len(PROBE) or [record["turn"] for record in selected] != list(range(1, len(PROBE) + 1)):
        raise RuntimeError(f"Capability replay is missing complete {label} records")
    return selected


def verify_report(report: dict) -> None:
    """Reject malformed, non-deterministic, or unsupported proof observations."""
    records = report.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Capability replay records are malformed")
    enabled_first = _records_by_label(records, "enabled-first")
    enabled_second = _records_by_label(records, "enabled-second")
    control = _records_by_label(records, "control")
    for first, second in zip(enabled_first, enabled_second, strict=True):
        if _normalized(first["diagnostics"]) != _normalized(second["diagnostics"]):
            raise RuntimeError("Capability replay enabled diagnostics are not deterministic")
    for enabled, baseline in zip(enabled_first, control, strict=True):
        if enabled["user_message"] != baseline["user_message"]:
            raise RuntimeError("Capability replay controls received different messages")
        left, right = (enabled["diagnostics"], baseline["diagnostics"])
        for key in ("preferences", "retrieved_ids", "fallbacks"):
            if left.get(key) != right.get(key):
                raise RuntimeError(f"Capability replay controls diverged on {key}")
        if enabled["response_contract"] != baseline["response_contract"]:
            raise RuntimeError("Capability replay controls diverged on response contract")
    witnesses = enabled_first[0]["diagnostics"].get("role_evidence")
    expected = {"WHOLE": [{"preference": "leather outer shell", "material": "leather", "role": "outer shell",
                            "source": "description", "span": "leather outer shell", "start": 16, "end": 35}]}
    if witnesses != expected:
        raise RuntimeError("Capability replay lacks the required direct whole-product witness")
    for record in [*enabled_first[1:], *enabled_second[1:]]:
        if record["diagnostics"].get("role_evidence") != {}:
            raise RuntimeError("Capability replay retained evidence after material correction")


def render_transcript(report: dict) -> str:
    lines = ["MERCURY | Component-qualified source evidence", CLAIM_BOUNDARY, ""]
    for label in ("enabled-first", "control"):
        lines.append(label.upper())
        for record in _records_by_label(report["records"], label):
            diagnostics = record["diagnostics"]
            lines.extend((f"Turn {record['turn']}: {record['user_message']}",
                          f"Active state: {json.dumps(diagnostics['preferences'], sort_keys=True)}",
                          f"Role evidence: {json.dumps(diagnostics['role_evidence'], sort_keys=True)}",
                          f"Ranked IDs: {json.dumps(diagnostics['ranked_ids'])}",
                          f"Contract: {record['response_contract']['status']} | fallbacks: {diagnostics['fallbacks']}",
                          ""))
    lines.append("The enabled replay runs twice; normalized state, role evidence, and ranking must match exactly.")
    lines.append("The control differs only by role_evidence=false. No score or hidden-set claim is made.")
    return "\n".join(lines) + "\n"


def run_replay(output: Path, *, agent_factory=Agent) -> dict:
    """Create one fresh, audited capability replay directory from the repository root."""
    if Path.cwd().resolve() != REPOSITORY:
        raise ValueError("Run role-evidence replay from the submission repository root")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog = output / "invented-catalog.jsonl"
    report = {"schema": "role-evidence-capability-replay-v1", "status": "failed", "records": [],
              "claim_boundary": CLAIM_BOUNDARY}
    manifest = {"schema": "role-evidence-capability-replay-v1", "status": "failed", "error": None,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "claim_boundary": CLAIM_BOUNDARY,
                "paid_calls": 0, "output_is_video": False, "agents": []}
    failure = None
    try:
        with catalog.open("x", encoding="utf-8") as handle:
            for row in INVENTED_ROWS:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
        configurations = _configurations()
        manifest["configurations"] = {
            name: {key: value for key, value in item.items() if key != "config"}
            for name, item in configurations.items()
        }
        manifest["before"] = _inventory(catalog, configurations)
        _record_run("enabled-first", catalog, configurations["enabled"]["config"], report, manifest, agent_factory)
        _record_run("enabled-second", catalog, configurations["enabled"]["config"], report, manifest, agent_factory)
        _record_run("control", catalog, configurations["control"]["config"], report, manifest, agent_factory)
        verify_report(report)
        report["status"] = "completed"
    except BaseException as error:
        failure = error
        manifest["error"] = repr(error)
    finally:
        try:
            if "before" in manifest:
                manifest["after"] = _inventory(catalog, configurations)
                if manifest["before"] != manifest["after"]:
                    raise RuntimeError("Capability replay source, configuration, catalog, or model changed")
            if failure is None:
                manifest["status"] = "completed"
        except BaseException as error:
            if failure is None:
                failure = error
                manifest["error"] = repr(error)
            else:
                manifest["verification_error"] = repr(error)
        if failure is not None:
            report["status"] = "failed"
            manifest["status"] = "failed"
        try:
            _write_json(output / "responses.json", report)
        except BaseException as error:
            if failure is None:
                failure = error
                manifest["status"] = "failed"
                report["status"] = "failed"
        try:
            (output / "transcript.txt").write_text(render_transcript(report), encoding="utf-8")
        except BaseException as error:
            manifest["presentation_error"] = repr(error)
            if failure is None:
                failure = error
                manifest["error"] = repr(error)
                manifest["status"] = "failed"
                report["status"] = "failed"
        try:
            _write_json(output / "manifest.json", manifest)
        except BaseException as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a source-span role-evidence capability replay.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_replay(args.output)


if __name__ == "__main__":
    main()
