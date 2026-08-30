"""Record the preregistered alternatives probes through the ordinary shopping API."""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import textwrap
import time
from dataclasses import asdict
from pathlib import Path

from experiments.cycle2_capabilities import validate_response
from experiments.cycle2_evaluate import MODES, model_file_hashes
from experiments.run import source_hashes
from mercury.agent import Agent
from mercury.config import Config
from mercury.model_assets import file_sha256
from mercury.ranking import preference_evidence
from mercury.types import Preference


REPOSITORY = Path(__file__).resolve().parents[1]
REAL_PROBES = (
    ("bag", ("I need a bag. It must be canvas or leather.",
             "Actually, canvas only. Keep the bag requirement.")),
    ("shirt", ("I need a shirt. It must be cotton or linen.", "Actually, no linen.")),
    ("jacket", ("I need a jacket. It must be waterproof or insulated.", "Actually, waterproof only.")),
)
SYNTHETIC_PROBES = (("three-state", ("I need a shirt. It must be cotton or linen.", "Actually, no linen.")),)
SYNTHETIC_ROWS = (
    {"parent_asin": "INVENTED_COTTON", "title": "Everyday cotton shirt", "categories": ["Clothing", "Shirts"],
     "features": ["Cotton fabric. Linen-free."], "details": {"Fabric type": "cotton"}},
    {"parent_asin": "INVENTED_UNKNOWN", "title": "Everyday shirt", "categories": ["Clothing", "Shirts"],
     "features": ["Button closure."]},
    {"parent_asin": "INVENTED_NEITHER", "title": "Everyday shirt", "categories": ["Clothing", "Shirts"],
     "features": ["Cotton-free fabric. Linen-free construction."]},
)


def _finite(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _safe(value: object) -> object:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError):
        return {"invalid_serialization": repr(value)}


def _preferences(diagnostics: dict) -> list[dict] | None:
    preferences = diagnostics.get("preferences")
    if not isinstance(preferences, list) or any(not isinstance(item, dict) for item in preferences):
        return None
    return [{key: value for key, value in item.items() if key not in {"alternative_group", "active"}}
            for item in preferences if item.get("active", True)]


def _rank(identifiers: list[str], identifier: str) -> int | None:
    return identifiers.index(identifier) + 1 if identifier in identifiers else None


def select_witness(records: list[dict], catalog_kind: str = "real") -> dict | None:
    """Select a common-pool guard intervention, never infer a missing penalty is zero."""
    indexed = {}
    for record in records:
        if record["catalog_kind"] == catalog_kind:
            key = (record["mode"], record["probe_id"], record["turn"])
            if key in indexed:
                raise ValueError("Duplicate replay turn identity")
            indexed[key] = record
    probes = REAL_PROBES if catalog_kind == "real" else SYNTHETIC_PROBES
    for probe_id, messages in probes:
        for number in range(1, len(messages) + 1):
            before, after = (indexed.get((mode, probe_id, number)) for mode in ("parse", "grouped"))
            if not before or not after or any(record.get("error") or
                    record.get("response_contract", {}).get("status") != "passed" for record in (before, after)):
                continue
            left, right = (record.get("diagnostics") for record in (before, after))
            if any(not isinstance(item, dict) or item.get("fallbacks") != [] for item in (left, right)):
                continue
            if not isinstance(left.get("query"), str) or left["query"] != right.get("query") \
                    or not isinstance(left.get("retrieved_ids"), list) or not left["retrieved_ids"] \
                    or left["retrieved_ids"] != right.get("retrieved_ids") \
                    or _preferences(left) is None or _preferences(left) != _preferences(right):
                continue
            if any(not isinstance(item.get("ranked_ids"), list) for item in (left, right)):
                continue
            groups = {}
            for preference in right["preferences"]:
                if preference.get("active", True) and preference.get("polarity") == 1 \
                        and isinstance(preference.get("alternative_group"), str) and preference["alternative_group"]:
                    groups.setdefault((preference["attribute"], preference["alternative_group"]), []).append(preference)
            for identifier in right["ranked_ids"]:
                if identifier not in left["ranked_ids"] or identifier not in left["retrieved_ids"]:
                    continue
                penalties = [item.get("constraint_penalties") for item in (left, right)]
                if any(not isinstance(item, dict) or not _finite(item.get(identifier)) for item in penalties) \
                        or penalties[0][identifier] <= 0 or penalties[1][identifier] != 0:
                    continue
                entry = after.get("evidence", {}).get(identifier, {})
                source = entry.get("source")
                if not isinstance(source, dict) or source.get("parent_asin") != identifier \
                        or any(not isinstance(source.get(key), str) for key in ("title", "features", "details")):
                    continue
                for group in groups.values():
                    if len(group) < 2 or not any(item.get("hard") is True for item in group):
                        continue
                    options = [option for option in entry.get("options", [])
                               if any({key: value for key, value in option.items() if key != "signal"} == item
                                      for item in group)]
                    if len(options) != len(group) or any(not _finite(item.get("signal")) for item in options):
                        continue
                    signals = [option["signal"] for option in options]
                    if max(signals) < 0 or min(signals) >= 0:
                        continue
                    return {"catalog_kind": catalog_kind, "probe_id": probe_id, "turn": number,
                            "parent_asin": identifier, "query": right["query"], "source": copy.deepcopy(source),
                            "option_evidence": copy.deepcopy(options),
                            "group_evidence": "supported" if max(signals) > 0 else "unknown",
                            "parse_penalty": penalties[0][identifier], "grouped_penalty": penalties[1][identifier],
                            "parse_rank": _rank(left["ranked_ids"], identifier),
                            "grouped_rank": _rank(right["ranked_ids"], identifier),
                            "parse_top10_rank": _rank([item["parent_asin"] for item in before["response"]["recommendations"]], identifier),
                            "grouped_top10_rank": _rank([item["parent_asin"] for item in after["response"]["recommendations"]], identifier)}
    return None


def _evidence(diagnostics: dict, products: dict) -> dict:
    preferences = [item for item in diagnostics["preferences"]
                   if item.get("active", True) and item["polarity"] == 1]
    result = {}
    for identifier in diagnostics["ranked_ids"]:
        product = products[identifier]
        options = []
        for item in preferences:
            preference = Preference(item["attribute"], item["value"], item["source_turn"], "",
                                    hard=item["hard"], alternative_group=item.get("alternative_group"))
            signal = preference_evidence(product, preference)
            if not _finite(signal):
                raise ValueError("Evidence adapter returned a non-finite signal")
            options.append({**item, "signal": signal})
        result[identifier] = {"source": {"parent_asin": identifier, "title": product.title,
                                         "features": product.fields.get("features", ""),
                                         "details": product.fields.get("details", ""),
                                         "fields": dict(product.fields),
                                         "facet_evidence": [asdict(item) for item in product.evidence]},
                              "options": options}
    return result


def render_replay(report: dict, selected_mode: str) -> tuple[str, list[list]]:
    """Pace recorded observations for narration without changing measured timings."""
    lines, events = [], []

    def emit(position, message):
        message = "\n".join("\n".join(textwrap.wrap(line, width=112)) for line in message.split("\n"))
        lines.append(message)
        events.append([position, "o", message.replace("\n", "\r\n") + "\r\n"])

    emit(0.0, "MERCURY | Explicit shopping alternatives")
    emit(0.5, "Recorded API calls. Narration pacing is not a latency measurement.")
    emit(1.0, f"Narrated control: {selected_mode}; every fixed control is retained in responses.json.")
    emit(2.0, f"Execution status: {report.get('status', 'unverified')}. Invented catalog examples are labeled.")
    narrated = [record for record in report["records"] if record["mode"] == selected_mode]
    for index, record in enumerate(narrated):
        position = 5.0 + index * 16.0
        diagnostics = record.get("diagnostics") if isinstance(record.get("diagnostics"), dict) else {}
        response = record.get("response") if isinstance(record.get("response"), dict) else {}
        emit(position, f"\n{record['catalog_kind'].upper()} | {record['probe_id']} | turn {record['turn']}")
        emit(position + 1, f"Shopper: {record['user_message']}")
        emit(position + 2, f"Actual query: {diagnostics.get('query', '(unavailable)')}")
        preferences = diagnostics.get("preferences")
        state = []
        for item in preferences if isinstance(preferences, list) else []:
            if not isinstance(item, dict) or not item.get("active", True):
                continue
            polarity = "exclude " if item.get("polarity") == -1 else "any " if item.get("polarity") == 0 else ""
            group = f" [OR {item['alternative_group']}]" if item.get("alternative_group") else ""
            state.append(f"{polarity}{item.get('attribute')}={item.get('value')}{group}")
        emit(position + 3, "Active state: " + ("; ".join(state) or "(none recorded)"))
        emit(position + 4, f"Response: {response.get('message', '(unavailable)')}")
        recommendations = response.get("recommendations")
        emit(position + 5, "Top recommendations (complete response retained in responses.json):")
        for rank, item in enumerate(recommendations[:3] if isinstance(recommendations, list) else [], 1):
            identifier = item.get("parent_asin") if isinstance(item, dict) else "(invalid)"
            entry = record.get("evidence", {}).get(identifier, {}) if isinstance(identifier, str) else {}
            title = entry.get("source", {}).get("title", "(source unavailable)")
            emit(position + 5 + rank / 4, f"  {rank}. {identifier} | {title}")
        emit(position + 7, f"Measured response: {record['latency_seconds']:.3f}s | "
             f"contract: {record['response_contract']['status']} | fallbacks: {diagnostics.get('fallbacks')}")
    synthetic = next((record for record in report["records"] if record["catalog_kind"] == "invented"
                      and record["mode"] == "grouped" and record["turn"] == 1), None)
    if synthetic:
        emit(137.0, "\nINVENTED catalog: actual positive material evidence, not facts inferred from missing words:")
        for index, (identifier, entry) in enumerate(synthetic.get("evidence", {}).items()):
            options = [item for item in entry["options"] if item["attribute"] == "material"]
            signals = [item["signal"] for item in options]
            state = "unverified" if not signals else "supported" if max(signals) > 0 else "unknown" if max(signals) == 0 else "contradicted"
            signals_text = ", ".join(f"{item['value']}={item['signal']:+.2f}" for item in options)
            emit(138.0 + index, f"  {identifier}: {state}; {signals_text}")
    witness = report.get("real_witness")
    if witness is None:
        emit(149.0, "\nReplay failed or is incomplete; no verified intervention is selected." if report.get("status") == "failed"
             else "\nNo real-catalog intervention observed in the retained, successfully checked records.")
        witness = report.get("invented_witness")
    if witness:
        emit(152.0, f"{witness['catalog_kind'].upper()} guard observation: {witness['parent_asin']} | "
             f"parse penalty {witness['parse_penalty']} -> grouped {witness['grouped_penalty']}; "
             f"retained rank {witness['parse_rank']} -> {witness['grouped_rank']}.")
        emit(156.0, "Observed option evidence: " + ", ".join(
            f"{item['attribute']}={item['value']}: {item['signal']:+.2f}" for item in witness["option_evidence"]))
        emit(158.0, "Catalog title: " + witness["source"]["title"])
        for position, field in ((160.0, "features"), (162.0, "details")):
            value = witness["source"][field]
            excerpt = value if len(value) <= 180 else value[:180] + "... [excerpt]"
            emit(position, f"Catalog {field}: {excerpt or '(not provided)'}")
    emit(169.0, "Same query and retrieval IDs are required for a reported guard intervention; missing penalties are not zero.")
    emit(171.0, "No score, conversion or hidden-set gain is claimed. These are authored probes, not independent shopper tests.")
    emit(179.0, "End of paced terminal replay. This is an asciicast, not an encoded submission video.")
    return "\n".join(lines) + "\n", events


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _provenance(catalog: Path, synthetic: Path, configs: dict) -> dict:
    return {"source": source_hashes(), "catalog_sha256": file_sha256(catalog),
            "invented_catalog_sha256": file_sha256(synthetic),
            "configs": {mode: file_sha256(Path(item["path"])) for mode, item in configs.items()},
            "models": model_file_hashes(Config.from_dict(configs["frozen"]["values"]))}


def _record_agent(catalog, config, mode, kind, probes, report, manifest, agent_factory):
    agent = None
    failed = False
    receipt = {"mode": mode, "catalog_kind": kind}
    manifest["agents"].append(receipt)
    started = time.perf_counter()
    try:
        agent = agent_factory(catalog, config)
        receipt["cold_start_seconds"] = time.perf_counter() - started
        receipt["catalog_sha256"] = agent.catalog.sha256
        receipt["product_count"] = len(agent.catalog.by_id)
        receipt["startup_fallbacks"] = _safe(getattr(agent, "startup_fallbacks", None))
        if receipt["startup_fallbacks"] != {}:
            raise RuntimeError("Startup model health failed or is unverified")
        expected = manifest["before"]["catalog_sha256" if kind == "real" else "invented_catalog_sha256"]
        if agent.catalog.sha256 != expected:
            raise RuntimeError("Catalog changed before Agent initialization")
        for probe_id, messages in probes:
            session = "replay-" + probe_id
            agent.reset(session, {})
            for number, message in enumerate(messages, 1):
                record = {"catalog_kind": kind, "mode": mode, "probe_id": probe_id, "turn": number,
                          "user_message": message, "response": None, "diagnostics": None, "evidence": {}, "error": None}
                report["records"].append(record)
                started = time.perf_counter()
                try:
                    response = agent.respond(session, message, number, 10)
                    record["latency_seconds"] = time.perf_counter() - started
                    record["response"] = _safe(response)
                    record["diagnostics"] = _safe(agent.last_diagnostics)
                    record["response_contract"] = validate_response(response, set(agent.catalog.by_id))
                    if record["response_contract"]["status"] != "passed":
                        raise RuntimeError("Response contract failed")
                    if record["diagnostics"].get("fallbacks") != []:
                        raise RuntimeError("Turn health failed or is unverified")
                    record["evidence"] = _evidence(record["diagnostics"], agent.catalog.by_id)
                except BaseException as error:
                    record.setdefault("latency_seconds", time.perf_counter() - started)
                    record.setdefault("response_contract", validate_response(record["response"], set(agent.catalog.by_id)))
                    record["error"] = repr(error)
                    raise
    except BaseException as error:
        receipt.setdefault("cold_start_seconds", time.perf_counter() - started)
        receipt["error"] = repr(error)
        failed = True
        raise
    finally:
        if agent is not None:
            try:
                agent.close()
                receipt["closed"] = True
            except Exception as error:
                receipt["close_error"] = repr(error)
                if not failed:
                    raise


def run_replay(catalog: Path, output: Path, *, selected_mode: str = "grouped", agent_factory=Agent) -> dict:
    if Path.cwd().resolve() != REPOSITORY or selected_mode not in MODES:
        raise ValueError("Run from the repository root with a registered narration mode")
    catalog, output = Path(catalog).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"records": [], "real_witness": None, "invented_witness": None, "status": "failed"}
    manifest = {"schema": "cycle2-alternatives-replay-v1", "status": "failed", "error": None,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "catalog_path": str(catalog), "selected_narration_mode": selected_mode, "configs": {}, "agents": [],
                "agent_factory": f"{agent_factory.__module__}.{agent_factory.__qualname__}",
                "paid_calls": 0, "output_is_video": False, "presentation_duration_seconds": 180.0,
                "note": "All fixed controls and probes retained; authored demonstrations are not shopper or score evidence."}
    synthetic = output / "invented-catalog.jsonl"
    failure = None
    try:
        with synthetic.open("x", encoding="utf-8") as handle:
            handle.write("".join(json.dumps(row, allow_nan=False) + "\n" for row in SYNTHETIC_ROWS))
        common = None
        for mode, expected_mode in MODES.items():
            path = REPOSITORY / "configs" / f"cycle2_{mode}.json"
            raw = path.read_bytes()
            config = Config.from_dict(json.loads(raw))
            values = config.to_dict()
            mode_value = values.pop("alternatives_mode")
            if mode_value != expected_mode or (common is not None and values != common):
                raise ValueError("Replay controls must differ only in registered alternatives mode")
            common = values
            manifest["configs"][mode] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                                          "values": config.to_dict()}
        manifest["before"] = _provenance(catalog, synthetic, manifest["configs"])
        if manifest["before"]["configs"] != {mode: item["sha256"] for mode, item in manifest["configs"].items()}:
            raise RuntimeError("Configuration changed during replay preparation")
        for mode, item in manifest["configs"].items():
            config = Config.from_dict(item["values"])
            for kind, path, probes in (("real", catalog, REAL_PROBES), ("invented", synthetic, SYNTHETIC_PROBES)):
                _record_agent(path, config, mode, kind, probes, report, manifest, agent_factory)
        manifest["status"] = "completed"
    except BaseException as error:
        failure = error
        manifest["error"] = repr(error)
    finally:
        try:
            manifest["after"] = _provenance(catalog, synthetic, manifest["configs"])
            if manifest.get("before") != manifest["after"]:
                raise RuntimeError("Source, configuration, catalog or model inventory changed during replay")
        except Exception as error:
            manifest["verification_error"] = repr(error)
            if failure is None:
                failure = error
                manifest["error"] = repr(error)
        if failure is not None:
            manifest["status"] = "failed"
        report["status"] = manifest["status"]
        try:
            if manifest["status"] == "completed":
                report["real_witness"] = select_witness(report["records"])
                report["invented_witness"] = select_witness(report["records"], "invented")
            transcript, events = render_replay(report, selected_mode)
            with (output / "transcript.txt").open("x", encoding="utf-8") as handle:
                handle.write(transcript)
            header = {"version": 2, "width": 120, "height": 40, "duration": 180.0, "timestamp": int(time.time()),
                      "title": "Mercury: actual alternatives and correction replay", "env": {"TERM": "xterm-256color"}}
            with (output / "replay.cast").open("x", encoding="utf-8") as handle:
                handle.write("\n".join(json.dumps(row, allow_nan=False) for row in [header, *events]) + "\n")
        except BaseException as error:
            manifest["presentation_error"] = repr(error)
            if failure is None:
                failure = error
                manifest["error"] = repr(error)
            manifest["status"] = "failed"
            report.update(status="failed", real_witness=None, invented_witness=None)
        finally:
            _write_json(output / "responses.json", report)
            manifest["output_sha256"] = {name: file_sha256(output / name)
                                         for name in ("responses.json", "transcript.txt", "replay.cast", "invented-catalog.jsonl")
                                         if (output / name).is_file()}
            _write_json(output / "manifest.json", manifest)
    if failure is not None:
        raise failure
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Record all preregistered real and invented alternatives probes.")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-mode", choices=tuple(MODES), default="grouped",
                        help="Narration only; all three fixed controls still run.")
    args = parser.parse_args()
    report = run_replay(args.catalog, args.output, selected_mode=args.selected_mode)
    print(json.dumps({"output": str(args.output), "status": report["status"],
                      "real_witness": report["real_witness"], "invented_witness": report["invented_witness"]}, indent=2))


if __name__ == "__main__":
    main()
