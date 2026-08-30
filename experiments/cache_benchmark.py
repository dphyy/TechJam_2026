from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import resource
import statistics
import time
from pathlib import Path

from experiments.run import source_hashes
from mercury.agent import Agent
from mercury.config import Config
from mercury.model_assets import file_sha256


DEFAULT_MESSAGE = "blue cotton shirt"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))]


def _semantic_response(response: dict) -> dict:
    return {
        "message": response["message"],
        "ask_attribute": response["ask_attribute"],
        "recommendations": response["recommendations"],
        "completion_tokens": response["usage"]["completion_tokens"],
    }


def run_cache_benchmark(catalog: Path, config_path: Path, sessions: int = 40,
                        message: str = DEFAULT_MESSAGE) -> dict:
    if type(sessions) is not int or sessions < 1:
        raise ValueError("sessions must be a positive integer")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be nonempty text")
    catalog = Path(catalog)
    config_path = Path(config_path)
    catalog_hash = file_sha256(catalog)
    config_hash = file_sha256(config_path)
    source_before = source_hashes()
    started = time.perf_counter()
    agent = Agent(catalog, Config.load(config_path))
    cold_start = time.perf_counter() - started
    latencies = []
    responses = []
    prompt_tokens = 0
    fallback_turns = 0
    try:
        for index in range(sessions):
            session_id = f"cache-benchmark-{index:04d}"
            agent.reset(session_id, {})
            turn_started = time.perf_counter()
            response = agent.respond(session_id, message, 1, 10)
            latencies.append(time.perf_counter() - turn_started)
            responses.append(_semantic_response(response))
            prompt_tokens += response["usage"]["prompt_tokens"]
            fallback_turns += bool(agent.last_diagnostics.get("fallbacks"))
        cache_stats = {
            key: value for key, value in agent.last_diagnostics["neural_logit_cache"].items()
            if key != "turn"
        }
        startup_fallbacks = dict(agent.startup_fallbacks)
    finally:
        agent.close()
    if catalog_hash != file_sha256(catalog) or config_hash != file_sha256(config_path):
        raise RuntimeError("Catalog or configuration changed during cache benchmark")
    semantic_bytes = json.dumps(
        responses, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    rss_scale = 1 if platform.system() == "Darwin" else 1024
    return {
        "schema": "mercury-exact-pair-cache-benchmark-v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "catalog": str(catalog),
        "catalog_sha256": catalog_hash,
        "config": str(config_path),
        "config_sha256": config_hash,
        "source_changed_during_run": source_before != source_hashes(),
        "sessions": sessions,
        "message": message,
        "semantic_responses_sha256": hashlib.sha256(semantic_bytes).hexdigest(),
        "prompt_tokens": prompt_tokens,
        "cache": cache_stats,
        "fallback_turns": fallback_turns,
        "startup_fallbacks": startup_fallbacks,
        "cold_start_seconds": cold_start,
        "p50_seconds": statistics.median(latencies),
        "p95_seconds": _percentile(latencies, .95),
        "max_seconds": max(latencies),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact cross-session neural pair-logit reuse.",
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_cache_benchmark(args.catalog, args.config, args.sessions, args.message)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
