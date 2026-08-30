from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import platform
import resource
import statistics
import time
from dataclasses import replace
from pathlib import Path

from experiments.run import source_hashes
from mercury.agent import Agent
from mercury.config import Config
from mercury.model_assets import file_sha256
from mercury.neural import NeuralRanker, fuse_neural_logits
from mercury.types import Candidate


QUERY = "blue cotton shirt"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))]


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()


def _fixed_candidates(catalog: Path, base: Config) -> list[Candidate]:
    retrieval = Agent(catalog, replace(
        base, neural_rerank=False, neural_logit_cache=False,
        page_local_rerank=False, progressive_frontier_rerank=False,
        seen_aware_slate=False,
    ))
    try:
        retrieval.reset("batch-benchmark", {})
        retrieval.respond("batch-benchmark", QUERY, 1, 10)
        identifiers = retrieval.last_diagnostics["ranked_ids"][:base.rerank_limit]
        return [
            Candidate(retrieval.catalog.by_id[identifier], float(len(identifiers) - index))
            for index, identifier in enumerate(identifiers)
        ]
    finally:
        retrieval.close()


def run_batch_benchmark(catalog: Path, config_path: Path, threads: int,
                        batch_size: int, repetitions: int = 20) -> dict:
    if type(threads) is not int or threads not in {2, 4, 6, 8}:
        raise ValueError("threads must be one of 2, 4, 6, or 8")
    if type(batch_size) is not int or batch_size not in {16, 30, 32}:
        raise ValueError("batch_size must be one of 16, 30, or 32")
    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("repetitions must be an integer of at least two")
    catalog = Path(catalog)
    config_path = Path(config_path)
    catalog_hash = file_sha256(catalog)
    config_hash = file_sha256(config_path)
    source_before = source_hashes()
    base = Config.load(config_path)
    candidates = _fixed_candidates(catalog, base)
    started = time.perf_counter()
    ranker = NeuralRanker(
        Path(base.artifact_dir), base.device, threads, base.reranker_model,
        batch_size=batch_size,
    )
    cold_start = time.perf_counter() - started
    for _ in range(2):
        ranker.score(QUERY, candidates)
    ranker.prompt_tokens = 0
    latencies = []
    vectors = []
    for _ in range(repetitions):
        turn_started = time.perf_counter()
        logits = ranker.score(QUERY, candidates)
        latencies.append(time.perf_counter() - turn_started)
        vectors.append([logits[item.product.parent_asin] for item in candidates])
    reference = vectors[0]
    maximum_drift = max(
        abs(value - expected)
        for vector in vectors[1:]
        for value, expected in zip(vector, reference, strict=True)
    )
    if not all(math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("Benchmark produced non-finite logits")
    ranking = fuse_neural_logits(
        candidates,
        {item.product.parent_asin: value for item, value in zip(candidates, reference, strict=True)},
        base.neural_weight,
    )
    if catalog_hash != file_sha256(catalog) or config_hash != file_sha256(config_path):
        raise RuntimeError("Catalog or configuration changed during batch benchmark")
    rss_scale = 1 if platform.system() == "Darwin" else 1024
    return {
        "schema": "mercury-neural-batch-thread-benchmark-v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "catalog_sha256": catalog_hash,
        "config_sha256": config_hash,
        "source_changed_during_run": source_before != source_hashes(),
        "query": QUERY,
        "candidate_count": len(candidates),
        "candidate_ids_sha256": _digest([item.product.parent_asin for item in candidates]),
        "threads": threads,
        "batch_size": batch_size,
        "warmups": 2,
        "repetitions": repetitions,
        "logits_sha256": _digest(reference),
        "ranking_sha256": _digest([item.product.parent_asin for item in ranking]),
        "max_repeat_logit_drift": maximum_drift,
        "prompt_tokens": ranker.prompt_tokens,
        "evaluated_pairs": len(candidates) * repetitions,
        "cold_start_seconds": cold_start,
        "p50_seconds": statistics.median(latencies),
        "p95_seconds": _percentile(latencies, .95),
        "max_seconds": max(latencies),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MiniLM batch size and CPU threads.")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/selected.json"))
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_batch_benchmark(
        args.catalog, args.config, args.threads, args.batch_size, args.repetitions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
