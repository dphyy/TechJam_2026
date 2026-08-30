from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from mercury.config import Config
from mercury.model_assets import file_sha256


class ObservedAgent:
    """Evaluation-only instrumentation; labels are never sent to the agent."""

    def __init__(self, inner):
        self.inner = inner
        self.traces: list[list[dict]] = []

    def reset(self, session_id, profile):
        self.traces.append([])
        if len(self.traces) % 20 == 0:
            print(f"Starting session {len(self.traces)}", file=sys.stderr, flush=True)
        self.inner.reset(session_id, profile)

    def respond(self, session_id, message, turn, top_k):
        started = time.perf_counter()
        try:
            response = self.inner.respond(session_id, message, turn, top_k)
        except Exception as error:
            self.traces[-1].append({"turn": turn, "message": message, "error": repr(error),
                                    "latency_seconds": time.perf_counter() - started})
            raise
        self.traces[-1].append({"turn": turn, "message": message, "response": response,
                                "diagnostics": getattr(self.inner, "last_diagnostics", {}),
                                "latency_seconds": time.perf_counter() - started})
        return response


def source_hashes() -> dict[str, str]:
    paths = []
    for directory in ("mercury", "experiments", "baselines", "starter", "evaluator", "tests", "demo"):
        paths.extend(sorted(Path(directory).glob("*.py")))
    paths.extend([Path("agent.py"), Path("docs/evaluation_config.json"), Path("docs/agent_api_contract.json"),
                  Path("requirements.txt"), Path("requirements-dev.txt"),
                  Path("requirements-neural.txt"), Path("requirements-neural.lock.txt")])
    return {str(path): file_sha256(path) for path in paths if path.is_file()}


def summarize_traces(traces: list[list[dict]], samples: list[dict], result_sessions: list[dict]) -> dict:
    sample_ids = [sample.get("sample_id") for sample in samples]
    result_ids = [session.get("sample_id") for session in result_sessions]
    for name, identifiers in (("Samples", sample_ids), ("Result sessions", result_ids)):
        if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise ValueError(f"{name} must have nonempty string sample_id values")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{name} contain duplicate sample_id values")
    if set(sample_ids) != set(result_ids):
        raise ValueError("Result session sample_id values do not match samples")
    if any(not isinstance(session.get("hit"), bool) for session in result_sessions):
        raise ValueError("Result sessions must contain official boolean hit values")
    if len(traces) != len(samples):
        raise ValueError("Trace session count does not match samples")
    results_by_id = {session["sample_id"]: session for session in result_sessions}
    latencies = sorted(item["latency_seconds"] for session in traces for item in session)
    route_recalls: dict[str, int] = {}
    recalls = {depth: 0 for depth in (10, 30, 60, 120, 300)}
    failures = {"not_retrieved": 0, "ranking_or_policy": 0, "agent_error_turns": 0}
    fallback_turns = 0
    cached_turns = 0
    instrumented = any("ranked_ids" in item.get("diagnostics", {}) for session in traces for item in session)
    for sample, session in zip(samples, traces, strict=True):
        target = sample["ground_truth"]["parent_asin"]
        seen_routes: set[str] = set()
        seen_depths: set[int] = set()
        retrieved = False
        for item in session:
            diagnostic = item.get("diagnostics", {})
            fallback_turns += bool(diagnostic.get("fallbacks"))
            cached_turns += bool(diagnostic.get("cache_hit"))
            failures["agent_error_turns"] += "error" in item
            retrieved |= target in diagnostic.get("retrieved_ids", [])
            for route, identifiers in diagnostic.get("routes", {}).items():
                if target in identifiers:
                    seen_routes.add(route)
            identifiers = diagnostic.get("ranked_ids", [])
            for depth in recalls:
                if target in identifiers[:depth]:
                    seen_depths.add(depth)
        for route in seen_routes:
            route_recalls[route] = route_recalls.get(route, 0) + 1
        for depth in seen_depths:
            recalls[depth] += 1
        if instrumented and not results_by_id[sample["sample_id"]]["hit"]:
            failures["ranking_or_policy" if retrieved else "not_retrieved"] += 1
    count = len(samples)
    return {"turn_count": len(latencies), "p50_seconds": statistics.median(latencies),
            "p95_seconds": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
            "max_seconds": max(latencies), "fallback_turns": fallback_turns, "cached_turns": cached_turns,
            "ever_ranked_recall": {str(key): value / count for key, value in recalls.items()} if instrumented else None,
            "ever_route_recall": {key: value / count for key, value in route_recalls.items()} if instrumented else None,
            "failure_diagnostics": failures if instrumented else {"agent_error_turns": failures["agent_error_turns"]},
            "recall_note": "Ever observed over each policy-dependent session; not an independent fixed-turn benchmark."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unchanged official evaluator with isolated diagnostics.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/splits/development.jsonl"))
    parser.add_argument("--freeze", type=Path, help="Required frozen-finalist manifest for reserved evaluation")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be one safe path component")
    config = Config.load(args.config) if args.config else Config()
    dataset_hash = file_sha256(args.dataset)
    reserved = Path("artifacts/splits/reserved.jsonl")
    is_reserved = reserved.exists() and dataset_hash == file_sha256(reserved)
    if is_reserved:
        if args.freeze is None:
            parser.error("Reserved evaluation requires a frozen-finalist manifest")
        freeze = json.loads(args.freeze.read_text())
        if freeze["reserved_sha256"] != dataset_hash or config.to_dict() not in freeze["configs"]:
            parser.error("Configuration/dataset is not in the frozen final comparison")
        if freeze["source_hashes"] != source_hashes():
            parser.error("Code changed after finalist freeze")
        for existing in args.output_root.glob("*/manifest.json"):
            previous = json.loads(existing.read_text())
            if previous.get("reserved_evaluation") and previous["config"] == config.to_dict():
                parser.error("This frozen configuration already has a reserved evaluation")
    output = args.output_root / args.name
    output.mkdir(parents=True, exist_ok=False)
    frozen_sources = source_hashes()
    for source, checksum in frozen_sources.items():
        destination = output / "source" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if file_sha256(destination) != checksum:
            raise RuntimeError(f"Source changed while snapshotting: {source}")
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.perf_counter()
    if args.baseline:
        from baselines.official import Agent
        inner = Agent(args.catalog)
    else:
        from mercury.agent import Agent
        inner = Agent(args.catalog, config)
    cold_start = time.perf_counter() - started
    samples = load_jsonl(args.dataset)
    ids, categories, products = catalog_index(args.catalog)
    observed = ObservedAgent(inner)
    evaluation_start = time.perf_counter()
    result = evaluate(observed, samples, ids, categories, products)
    elapsed = time.perf_counter() - evaluation_start
    diagnostics = summarize_traces(observed.traces, samples, result["sessions"])
    manifest = {
        "name": args.name, "baseline": args.baseline, "config": config.to_dict(),
        "started_at_utc": timestamp,
        "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "dataset_sha256": dataset_hash, "catalog_sha256": file_sha256(args.catalog),
        "dataset": str(args.dataset), "reserved_evaluation": is_reserved,
        "source_hashes": frozen_sources, "source_changed_during_run": frozen_sources != source_hashes(),
        "python": platform.python_version(),
        "machine": platform.machine(), "platform": platform.platform(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "cold_start_seconds": cold_start, "evaluation_seconds": elapsed,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
        "paid_cost_usd": 0.0, "cost_basis": "Existing local hardware; public model downloads; no paid APIs.",
        "startup_fallbacks": getattr(inner, "startup_fallbacks", {}),
    }
    manifest["package_versions"] = {}
    for name in ("numpy", "torch", "transformers", "sentence-transformers", "huggingface-hub"):
        try:
            manifest["package_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest["asset_manifests"] = {
        str(path): file_sha256(path) for path in Path(config.artifact_dir).glob("**/*manifest.json")
        if ".cache" not in path.parts
    }
    for filename, value in (("result.json", result), ("manifest.json", manifest),
                            ("diagnostics.json", diagnostics), ("traces.json", observed.traces)):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))
    print(json.dumps({**diagnostics, "cold_start_seconds": cold_start,
                      "evaluation_seconds": elapsed, "max_rss_bytes": manifest["max_rss_bytes"]}, indent=2))
    if hasattr(inner, "close"):
        inner.close()


if __name__ == "__main__":
    main()
