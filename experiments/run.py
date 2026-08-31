from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import platform
try:
    import resource
except ModuleNotFoundError:  # Windows exposes peak memory through psapi instead.
    resource = None
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


def peak_rss_bytes() -> int:
    """Peak resident set size in bytes, or 0 where the platform cannot report it."""
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak) * (1 if platform.system() == "Darwin" else 1024)
    import ctypes
    from ctypes import wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = wintypes.HANDLE
        current_process.argtypes = []
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.restype = wintypes.BOOL
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
        measured = query(current_process(), ctypes.byref(counters), counters.cb)
    except (AttributeError, OSError):
        return 0
    return int(counters.PeakWorkingSetSize) if measured else 0


def source_hashes() -> dict[str, str]:
    paths = []
    for directory in ("mercury", "experiments", "baselines", "starter", "evaluator", "tests", "demo"):
        paths.extend(sorted(Path(directory).glob("*.py")))
    paths.extend([Path("agent.py"), Path("docs/evaluation_config.json"), Path("docs/agent_api_contract.json"),
                  Path("requirements.txt"), Path("requirements-dev.txt"),
                  Path("requirements-neural.txt"), Path("requirements-neural.lock.txt")])
    return {path.as_posix(): file_sha256(path) for path in paths if path.is_file()}


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
    route_depth_recalls: dict[str, dict[int, int]] = {}
    recalls = {depth: 0 for depth in (10, 30, 60, 120, 300)}
    rerank_prefix_recall = 0
    failures = {"not_retrieved": 0, "ranking_or_policy": 0, "agent_error_turns": 0}
    taxonomy = {"state_or_intent_signal": 0, "retrieval_miss": 0, "rerank_admission_miss": 0,
                "ranking_or_policy_miss": 0, "dialogue_signal": 0, "constraint_signal": 0,
                "runtime_error_or_fallback": 0}
    fallback_turns = 0
    cached_turns = 0
    instrumented = any("ranked_ids" in item.get("diagnostics", {}) for session in traces for item in session)
    for sample, session in zip(samples, traces, strict=True):
        target = sample["ground_truth"]["parent_asin"]
        seen_routes: set[str] = set()
        seen_depths: set[int] = set()
        seen_rerank_prefix = False
        retrieved = False
        session_fallback = False
        question_goals = []
        for item in session:
            diagnostic = item.get("diagnostics", {})
            fallback_turns += bool(diagnostic.get("fallbacks"))
            session_fallback |= bool(diagnostic.get("fallbacks")) or "error" in item
            cached_turns += bool(diagnostic.get("cache_hit"))
            failures["agent_error_turns"] += "error" in item
            retrieved |= target in diagnostic.get("retrieved_ids", [])
            for route, identifiers in diagnostic.get("routes", {}).items():
                if target in identifiers:
                    seen_routes.add(route)
                route_depth_recalls.setdefault(route, {depth: 0 for depth in (30, 60, 120)})
            identifiers = diagnostic.get("ranked_ids", [])
            for depth in recalls:
                if target in identifiers[:depth]:
                    seen_depths.add(depth)
            seen_rerank_prefix |= target in diagnostic.get("rerank_prefix_ids", [])
            goal = diagnostic.get("question", {}).get("goal")
            if isinstance(goal, str):
                question_goals.append(goal)
        for route in seen_routes:
            route_recalls[route] = route_recalls.get(route, 0) + 1
        for route in route_depth_recalls:
            route_hits = {depth for item in session for depth in (30, 60, 120)
                          if target in item.get("diagnostics", {}).get("routes", {}).get(route, [])[:depth]}
            for depth in route_hits:
                route_depth_recalls[route][depth] += 1
        for depth in seen_depths:
            recalls[depth] += 1
        rerank_prefix_recall += seen_rerank_prefix
        if instrumented and not results_by_id[sample["sample_id"]]["hit"]:
            failures["ranking_or_policy" if retrieved else "not_retrieved"] += 1
            taxonomy["retrieval_miss" if not retrieved else "ranking_or_policy_miss"] += 1
            if retrieved and not seen_rerank_prefix:
                taxonomy["rerank_admission_miss"] += 1
            if session_fallback:
                taxonomy["runtime_error_or_fallback"] += 1
            if any(goal in question_goals[:index] for index, goal in enumerate(question_goals)) \
                    or any(item.get("diagnostics", {}).get("policy", {}).get("previous_answer_productivity")
                           in {"neutral", "unresolved"} for item in session):
                taxonomy["dialogue_signal"] += 1
            if any(item.get("diagnostics", {}).get("constraint_penalties", {}).get(target, 0) > 0
                   for item in session):
                taxonomy["constraint_signal"] += 1
            intents = [item.get("diagnostics", {}).get("intent", {}) for item in session]
            if ((intents and all(intent.get("confidence", 1.0) < .5 for intent in intents))
                    or all(not item.get("diagnostics", {}).get("query") for item in session)):
                taxonomy["state_or_intent_signal"] += 1
    count = len(samples)
    checks = [check for session in traces for item in session
              for check in item.get("diagnostics", {}).get("constraint_checks", [])]
    audit = {
        stage: {"calls": sum(check["stage"] == stage for check in checks),
                "reordered_calls": sum(check["stage"] == stage and check["reordered"] for check in checks),
                "seconds": sum(check["seconds"] for check in checks if check["stage"] == stage)}
        for stage in ("pre", "post")
    }
    audit["returned_contradictions"] = sum(
        len(item.get("diagnostics", {}).get("returned_constraint_contradictions", []))
        for session in traces for item in session
    )
    return {"turn_count": len(latencies), "constraint_audit": audit, "p50_seconds": statistics.median(latencies),
            "p95_seconds": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
            "max_seconds": max(latencies), "fallback_turns": fallback_turns, "cached_turns": cached_turns,
            "ever_ranked_recall": {str(key): value / count for key, value in recalls.items()} if instrumented else None,
            "ever_rerank_prefix_recall": rerank_prefix_recall / count if instrumented else None,
            "ever_route_recall": {key: value / count for key, value in route_recalls.items()} if instrumented else None,
            "ever_route_depth_recall": {
                route: {str(depth): hits / count for depth, hits in depths.items()}
                for route, depths in route_depth_recalls.items()
            } if instrumented else None,
            "failure_diagnostics": failures if instrumented else {"agent_error_turns": failures["agent_error_turns"]},
            "failure_taxonomy": taxonomy if instrumented else None,
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
        "max_rss_bytes": peak_rss_bytes(),
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
