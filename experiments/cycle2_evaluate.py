"""One source/configuration freeze and six persistently consumed validation runs."""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

from experiments.cycle2_prepare import verify_lock
from experiments.run import source_hashes
from mercury.config import Config
from mercury.model_assets import file_sha256, verify_model


REPOSITORY = Path(__file__).resolve().parents[1]
FREEZE_DIRECTORY = Path("artifacts/cycle2/alternatives-freeze")
MODES = {"frozen": "off", "parse": "parse", "grouped": "grouped"}
KINDS = {"targets", "capabilities"}
PARITY_DIAGNOSTICS = ("query", "revision", "cache_hit", "preferences", "routes", "retrieved_ids", "ranked_ids", "fallbacks", "policy")
INPUT_PATHS = {
    "catalog": "data/catalog.jsonl", "public_development": "data/public_set.jsonl",
    "target_development": "artifacts/cycle2/synthetic-targets/development.jsonl",
    "target_validation": "artifacts/cycle2/synthetic-targets/validation.jsonl",
    "target_manifest": "artifacts/cycle2/synthetic-targets/manifest.json",
    "original_generator": "artifacts/cycle2/provenance/cycle2_prepare-original.py",
    "capability_development": "artifacts/cycle2/capability-development.json",
    "capability_validation": "artifacts/cycle2/capability-validation.json",
    "capability_manifest": "artifacts/cycle2/capability-manifest.json",
    "protocol": "docs/CYCLE2_EXPERIMENT_PROTOCOL.md",
    "alternatives_protocol": "docs/CYCLE2_ALTERNATIVES_PROTOCOL.md",
    "selected_config": "configs/selected.json",
}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def model_file_hashes(config: Config) -> dict[str, str]:
    artifacts = (REPOSITORY / config.artifact_dir).resolve()
    result = {}
    for enabled, kind in ((config.neural_rerank, "reranker"), (config.dense, "embedding")):
        if not enabled:
            continue
        directory = artifacts / "models" / kind
        manifest = verify_model(directory, kind)
        for name in (*manifest["files"], "asset_manifest.json"):
            path = (directory / name).resolve()
            result[str(path)] = file_sha256(path)
    if config.contrast:
        raise ValueError("The registered alternatives comparison does not include contrast assets")
    return result


def behavior_parity(control: Path, candidate: Path) -> dict:
    """Compare actual control behavior, excluding timing and additive diagnostics only."""
    def load(directory):
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        traces = json.loads((directory / "traces.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("source_changed_during_run") is not False:
            raise ValueError("Parity requires unchanged-source run manifests")
        sessions = result.get("sessions")
        if not isinstance(sessions, list) or not sessions or not isinstance(traces, list) or len(sessions) != len(traces):
            raise ValueError("Parity requires nonempty aligned session and trace lists")
        indexed = {}
        for session, turns in zip(sessions, traces, strict=True):
            identifier = session.get("sample_id") if isinstance(session, dict) else None
            if not isinstance(identifier, str) or not identifier or identifier in indexed:
                raise ValueError("Parity requires unique nonempty session IDs")
            if not isinstance(turns, list) or not 1 <= len(turns) <= 10:
                raise ValueError("Parity requires all recorded turns")
            normalized = []
            for number, turn in enumerate(turns, 1):
                if not isinstance(turn, dict) or turn.get("turn") != number or not isinstance(turn.get("message"), str) \
                        or not isinstance(turn.get("response"), dict) or turn.get("error"):
                    raise ValueError("Parity requires successful complete turn records")
                diagnostics = turn.get("diagnostics")
                if not isinstance(diagnostics, dict) or any(key not in diagnostics for key in PARITY_DIAGNOSTICS):
                    raise ValueError("Parity trace is missing a required diagnostic field")
                normalized.append({"turn": number, "message": turn["message"], "response": turn["response"],
                                   **{key: diagnostics[key] for key in PARITY_DIAGNOSTICS}})
            indexed[identifier] = {"session": session, "turns": normalized}
        return {key: value for key, value in result.items() if key != "sessions"}, indexed

    left_summary, left = load(control)
    right_summary, right = load(candidate)
    if left.keys() != right.keys():
        raise ValueError("Parity requires identical session ID sets")
    mismatches = []
    if left_summary != right_summary:
        mismatches.append({"field": "official_result_aggregate"})
    count = 0
    for identifier in sorted(left):
        before, after = left[identifier], right[identifier]
        if before["session"] != after["session"]:
            mismatches.append({"session_id": identifier, "field": "official_session_result"})
        count += len(before["turns"])
        if len(before["turns"]) != len(after["turns"]):
            mismatches.append({"session_id": identifier, "field": "turn_count"})
            continue
        for original, current in zip(before["turns"], after["turns"], strict=True):
            for field in original:
                if original[field] != current[field]:
                    mismatches.append({"session_id": identifier, "turn": original["turn"], "field": field})
    return {"matched": not mismatches, "session_count": len(left), "turn_count": count, "mismatches": mismatches,
            "compared": ["complete official results", "messages", "complete responses including usage", *PARITY_DIAGNOSTICS],
            "excluded": ["timing", "new diagnostic fields absent from the original baseline"]}


def _frozen_state() -> dict:
    if Path.cwd().resolve() != REPOSITORY.resolve():
        raise ValueError("Run cycle2 evaluation from the submission repository root")
    inputs = {name: {"path": str((REPOSITORY / relative).resolve()),
                     "sha256": file_sha256(REPOSITORY / relative)} for name, relative in INPUT_PATHS.items()}
    verify_lock(Path(inputs["catalog"]["path"]), Path(inputs["public_development"]["path"]),
                (REPOSITORY / "artifacts/cycle2/synthetic-targets"), Path(inputs["original_generator"]["path"]))
    capability_lock = json.loads(Path(inputs["capability_manifest"]["path"]).read_text(encoding="utf-8"))
    if capability_lock.get("schema") != "cycle2-capability-lock-v1":
        raise ValueError("Unsupported capability lock")
    for split in ("development", "validation"):
        item = inputs[f"capability_{split}"]
        if capability_lock.get("sha256", {}).get(Path(item["path"]).name) != item["sha256"]:
            raise ValueError("Capability lock hash mismatch")
    configs = {}
    baseline = Config.load(REPOSITORY / "configs/selected.json").to_dict()
    if baseline["alternatives_mode"] != "off":
        raise ValueError("The selected frozen baseline must retain alternatives off")
    for mode, alternative in MODES.items():
        path = REPOSITORY / f"configs/cycle2_{mode}.json"
        value = Config.load(path).to_dict()
        if value["alternatives_mode"] != alternative or {**value, "alternatives_mode": "off"} != baseline:
            raise ValueError("All three configurations must match frozen budgets except alternatives_mode")
        configs[mode] = {"path": str(path.resolve()), "sha256": file_sha256(path), "config": value}
    return {"source_hashes": source_hashes(), "configs": configs, "inputs": inputs,
            "model_file_hashes": model_file_hashes(Config.from_dict(baseline))}


def create_freeze(reason: str) -> dict:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A development-only selection reason is required")
    output = (REPOSITORY / FREEZE_DIRECTORY).resolve()
    if output.exists():
        raise FileExistsError(f"The one-time alternatives freeze already exists: {output}")
    state = _frozen_state()
    output.mkdir(parents=True, exist_ok=False)
    (output / "consumption").mkdir()
    for relative, checksum in state["source_hashes"].items():
        source = (REPOSITORY / relative).resolve()
        if not source.is_relative_to(REPOSITORY.resolve()) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Invalid source snapshot path")
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if file_sha256(destination) != checksum:
            raise ValueError("Source changed while snapshotting")
    manifest_path = output / "manifest.json"
    manifest = {"schema": "cycle2-alternatives-freeze-v1", "frozen_at_utc": _now(), "reason": reason.strip(),
                "repository": str(REPOSITORY.resolve()), "manifest_path": str(manifest_path),
                "consumption_directory": str(output / "consumption"), "frozen": state,
                "validation_content_opened_by_freeze": False,
                "policy": "Each fixed mode/kind is consumed before inference. Failure remains consumed; no automatic recovery."}
    if state != _frozen_state():
        raise ValueError("Source/configuration/input changed while freezing")
    _write_json(manifest_path, manifest)
    _write_json(output / "manifest-integrity.json", {"sha256": file_sha256(manifest_path)})
    return manifest


def verify_freeze(manifest_path: Path) -> dict:
    expected = (REPOSITORY / FREEZE_DIRECTORY / "manifest.json").resolve()
    if manifest_path.resolve() != expected:
        raise ValueError("Use the fixed repository-owned alternatives freeze; copied manifests are not validation budgets")
    try:
        integrity = json.loads((expected.parent / "manifest-integrity.json").read_text(encoding="utf-8"))
        if integrity.get("sha256") != file_sha256(expected):
            raise ValueError("Frozen manifest integrity mismatch")
        manifest = json.loads(expected.read_text(encoding="utf-8"))
        if manifest.get("schema") != "cycle2-alternatives-freeze-v1" or manifest.get("manifest_path") != str(expected) \
                or manifest.get("repository") != str(REPOSITORY.resolve()) \
                or manifest.get("consumption_directory") != str(expected.parent / "consumption"):
            raise ValueError("Invalid alternatives freeze identity")
        if manifest.get("frozen") != _frozen_state():
            raise ValueError("Frozen source/configuration/data/model hashes changed")
        for relative, checksum in manifest["frozen"]["source_hashes"].items():
            if file_sha256(expected.parent / "source" / relative) != checksum:
                raise ValueError("Frozen source snapshot changed")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot verify alternatives freeze: {error}") from error


def _execute(job: dict) -> dict:
    output = Path(job["output"])
    if job["kind"] == "targets":
        command = [sys.executable, "-m", "experiments.run", "--name", output.name,
                   "--config", job["config_path"], "--catalog", job["catalog"],
                   "--dataset", job["dataset"], "--output-root", str(output.parent)]
        result = subprocess.run(command, cwd=REPOSITORY, text=True, capture_output=True, check=True)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    from experiments.cycle2_capabilities import run_capabilities
    return run_capabilities(Path(job["dataset"]), Path(job["config_path"]), output, "validation", provenance={
        "freeze_sha256": job["freeze_sha256"], "consumption_path": job["consumption_path"], "mode": job["mode"]})


def run_validation(manifest_path: Path, mode: str, kind: str, output: Path, *, runner=None) -> dict:
    if mode not in MODES or kind not in KINDS:
        raise ValueError("Validation requires one registered mode and targets/capabilities kind")
    manifest = verify_freeze(manifest_path)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Validation output already exists: {output}")
    if output == REPOSITORY.resolve() or (REPOSITORY / FREEZE_DIRECTORY).resolve().is_relative_to(output) \
            or output.is_relative_to((REPOSITORY / FREEZE_DIRECTORY).resolve()):
        raise ValueError("Validation output must not contain or live inside the freeze")
    ledger = Path(manifest["consumption_directory"])
    marker = ledger / f"{mode}-{kind}.json"
    frozen = manifest["frozen"]
    dataset = frozen["inputs"]["target_validation" if kind == "targets" else "capability_validation"]
    job = {"mode": mode, "kind": kind, "output": str(output), "dataset": dataset["path"],
           "dataset_sha256": dataset["sha256"], "config_path": frozen["configs"][mode]["path"],
           "config_sha256": frozen["configs"][mode]["sha256"], "catalog": frozen["inputs"]["catalog"]["path"],
           "freeze_sha256": file_sha256(manifest_path), "consumption_path": str(marker)}
    _write_json(marker, {**job, "status": "consumed", "started_at_utc": _now()})
    receipt = {"mode": mode, "kind": kind, "output": str(output), "freeze_sha256": job["freeze_sha256"]}
    try:
        result = (runner or _execute)(job)
        verify_freeze(manifest_path)
        receipt.update(status="completed", result=result)
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        if isinstance(error, subprocess.CalledProcessError):
            receipt.update(stdout=error.stdout, stderr=error.stderr, returncode=error.returncode)
        raise
    finally:
        receipt["finished_at_utc"] = _now()
        _write_json(ledger / f"{mode}-{kind}.result.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--reason", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--freeze", type=Path, default=FREEZE_DIRECTORY / "manifest.json")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--freeze", type=Path, default=FREEZE_DIRECTORY / "manifest.json")
    validate.add_argument("--mode", choices=MODES, required=True)
    validate.add_argument("--kind", choices=sorted(KINDS), required=True)
    validate.add_argument("--output", type=Path, required=True)
    parity = subparsers.add_parser("parity")
    parity.add_argument("control", type=Path)
    parity.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        manifest = create_freeze(args.reason)
        result = {"manifest_path": manifest["manifest_path"], "modes": list(MODES), "validation_runs": 6}
    elif args.action == "verify":
        manifest = verify_freeze(args.freeze)
        result = {"verified": True, "manifest_path": manifest["manifest_path"]}
    elif args.action == "parity":
        result = behavior_parity(args.control, args.candidate)
    else:
        result = run_validation(args.freeze, args.mode, args.kind, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
