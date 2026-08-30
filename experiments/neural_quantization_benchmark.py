"""Bounded feasibility benchmark for dynamic int8 MiniLM quantization."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import platform
import resource
import statistics
import tempfile
import time
from pathlib import Path

from experiments.neural_batch_benchmark import QUERY, _fixed_candidates
from experiments.run import source_hashes
from mercury.config import Config
from mercury.model_assets import file_sha256
from mercury.neural import NeuralRanker, fuse_neural_logits


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))]


def run(catalog: Path, config_path: Path, quantized: bool, repetitions: int = 20) -> dict:
    if type(quantized) is not bool:
        raise ValueError("quantized must be a boolean")
    if type(repetitions) is not int or repetitions < 2:
        raise ValueError("repetitions must be an integer of at least two")
    catalog, config_path = Path(catalog), Path(config_path)
    catalog_hash, config_hash = file_sha256(catalog), file_sha256(config_path)
    source_before = source_hashes()
    config = Config.load(config_path)
    candidates = _fixed_candidates(catalog, config)
    started = time.perf_counter()
    ranker = NeuralRanker(
        Path(config.artifact_dir), config.device, config.threads,
        config.reranker_model, batch_size=config.neural_batch_size,
    )
    backend = None
    serialized_bytes = (Path(config.artifact_dir) / "models" / config.reranker_model / "model.safetensors").stat().st_size
    if quantized:
        import torch

        if "qnnpack" not in torch.backends.quantized.supported_engines:
            raise RuntimeError("QNNPACK dynamic quantization is unavailable")
        torch.backends.quantized.engine = "qnnpack"
        backend = torch.backends.quantized.engine
        ranker.model.model = torch.ao.quantization.quantize_dynamic(
            ranker.model.model, {torch.nn.Linear}, dtype=torch.qint8,
        )
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save(ranker.model.model.state_dict(), handle.name)
            serialized_bytes = Path(handle.name).stat().st_size
    cold_start = time.perf_counter() - started
    for _ in range(2):
        ranker.score(QUERY, candidates)
    ranker.prompt_tokens = 0
    latencies, vectors = [], []
    for _ in range(repetitions):
        turn_started = time.perf_counter()
        logits = ranker.score(QUERY, candidates)
        latencies.append(time.perf_counter() - turn_started)
        vectors.append([logits[item.product.parent_asin] for item in candidates])
    if not all(math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("Quantization benchmark produced non-finite logits")
    reference = vectors[0]
    ranking = fuse_neural_logits(
        candidates,
        {item.product.parent_asin: value for item, value in zip(candidates, reference, strict=True)},
        config.neural_weight,
    )
    rss_scale = 1 if platform.system() == "Darwin" else 1024
    return {
        "schema": "mercury-neural-quantization-benchmark-v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arm": "dynamic_int8_qnnpack" if quantized else "float32_control",
        "catalog_sha256": catalog_hash,
        "config_sha256": config_hash,
        "source_changed_during_run": source_before != source_hashes(),
        "candidate_ids_sha256": _digest([item.product.parent_asin for item in candidates]),
        "candidate_count": len(candidates),
        "backend": backend,
        "threads": config.threads,
        "batch_size": config.neural_batch_size,
        "warmups": 2,
        "repetitions": repetitions,
        "logits": reference,
        "logits_sha256": _digest(reference),
        "ranking_sha256": _digest([item.product.parent_asin for item in ranking]),
        "serialized_model_bytes": serialized_bytes,
        "prompt_tokens": ranker.prompt_tokens,
        "cold_start_seconds": cold_start,
        "p50_seconds": statistics.median(latencies),
        "p95_seconds": _percentile(latencies, .95),
        "max_seconds": max(latencies),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale,
    }


def compare(control: dict, candidate: dict) -> dict:
    keys = ("schema", "catalog_sha256", "config_sha256", "candidate_ids_sha256", "candidate_count", "repetitions")
    if any(control.get(key) != candidate.get(key) for key in keys):
        raise ValueError("Quantization reports do not share fixed inputs")
    if control.get("arm") != "float32_control" or candidate.get("arm") != "dynamic_int8_qnnpack":
        raise ValueError("Quantization comparison requires control and int8 arms")
    left, right = control.get("logits"), candidate.get("logits")
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        raise ValueError("Quantization reports have invalid logits")
    return {
        "ranking_equal": control["ranking_sha256"] == candidate["ranking_sha256"],
        "maximum_logit_drift": max(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)),
        "p95_reduction": 1.0 - candidate["p95_seconds"] / control["p95_seconds"],
        "serialized_size_reduction": 1.0 - candidate["serialized_model_bytes"] / control["serialized_model_bytes"],
        "rss_change": candidate["max_rss_bytes"] / control["max_rss_bytes"] - 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark float32 or dynamic-int8 MiniLM")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/selected.json"))
    parser.add_argument("--arm", choices=("control", "quantized"))
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-control", type=Path)
    parser.add_argument("--compare-candidate", type=Path)
    args = parser.parse_args()
    if args.compare_control or args.compare_candidate:
        if not args.compare_control or not args.compare_candidate:
            parser.error("comparison requires both report paths")
        result = compare(
            json.loads(args.compare_control.read_text(encoding="utf-8")),
            json.loads(args.compare_candidate.read_text(encoding="utf-8")),
        )
    else:
        if args.arm is None or args.output is None:
            parser.error("a benchmark requires --arm and --output")
        result = run(args.catalog, args.config, args.arm == "quantized", args.repetitions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
