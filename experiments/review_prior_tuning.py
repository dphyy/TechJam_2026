"""Fresh family-disjoint development search with a one-shot reserved comparison."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import replace
import datetime
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl, metric_summary
from experiments.cycle3_prepare import _catalog_rows, _order, _read_rows, title_key
from experiments.cycle5_prepare import band_label, band_of, rating_number
from experiments.run import ObservedAgent, source_hashes, summarize_traces
from mercury.config import Config
from mercury.model_assets import file_sha256


SEED = "review-prior-tuning-20260831-v2"
PACK = Path("artifacts/review-prior-tuning-v2")
OUTPUT = Path("runs/review-prior-tuning-v2")
PROTOCOL = Path("docs/REVIEW_PRIOR_TUNING_PROTOCOL.md")
RATIOS = (0.0, .25, .50, .75, 1.0)
PRE_WEIGHTS = (.10, .20, .30)
POST_WEIGHTS = (0.0, .01, .02)
SCENARIOS = ("buying",) * 8 + ("browsing",) * 8 + ("intent_override",) * 3 + ("boundary",)


def write_json(path: Path, value: object) -> None:
    """Evidence is append-only: no accidental replacement of a prior outcome."""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def discover_inventory(catalog: Path, output: Path) -> tuple[set[str], dict[str, str]]:
    by_id = _catalog_rows(_read_rows(catalog))
    paths = set()
    for root in (Path("data"), Path("artifacts")):
        paths.update(path.resolve() for path in root.rglob("*.jsonl")
                     if not path.resolve().is_relative_to(output.resolve())
                     and path.resolve() != catalog.resolve())
    # Include external/temporary datasets referenced by past scored runs too.
    for pattern in ("*/manifest.json", "*/report.json"):
        for path in Path("runs").glob(pattern):
            value = json.loads(path.read_text()).get("dataset")
            if isinstance(value, str):
                target = Path(value).resolve()
                if not target.exists():
                    raise ValueError(f"Cannot audit missing historical dataset: {value}")
                paths.add(target)
    excluded: set[str] = set()
    sources = {}
    for path in sorted(paths):
        rows = _read_rows(path) if path.suffix == ".jsonl" else json.loads(path.read_text())
        if not isinstance(rows, list):
            continue  # Authored mini-catalog capability packs are not target sets.
        targets = set()
        for row in rows:
            truth = row.get("ground_truth") if isinstance(row, dict) else None
            target = truth.get("parent_asin") if isinstance(truth, dict) else None
            if isinstance(target, str) and target in by_id:
                targets.add(target)
        if targets:
            excluded.update(targets)
            sources[str(path)] = file_sha256(path)
    if not excluded or str(Path("data/public_set.jsonl").resolve()) not in sources:
        raise ValueError("Inventory must include public and historical target packs")
    return excluded, sources


def build_fresh_pack(products: list[dict], excluded_ids: set[str], per_band: int = 40) -> dict:
    if type(per_band) is not int or per_band < 1:
        raise ValueError("Positive per-band sample count required")
    by_id = _catalog_rows(products)
    excluded_families = {title_key(by_id[key].get("title") or "", True) for key in excluded_ids}
    groups: dict[str, list[str]] = defaultdict(list)
    for key, row in by_id.items():
        family = title_key(row.get("title") or "", True)
        if family and family not in excluded_families:
            groups[family].append(key)
    buckets: dict[int, list[str]] = defaultdict(list)
    for ids in groups.values():
        key = min(ids, key=lambda value: (_order(SEED, "member", value), value))
        buckets[band_of(rating_number(by_id[key]))].append(key)
    assigned = {"development": [], "reserved": []}
    for band in range(4):
        ordered = sorted(buckets[band], key=lambda key: (_order(SEED, f"band-{band}", key), key))
        if len(ordered) < 2 * per_band:
            raise ValueError(f"Insufficient fresh families in {band_label(band)}; do not silently rebalance")
        for offset, key in enumerate(ordered[:2 * per_band]):
            assigned[("development", "reserved")[offset % 2]].append(key)
    datasets = {}
    for split, ids in assigned.items():
        ids.sort(key=lambda key: (_order(SEED, f"scenario-{split}", key), key))
        datasets[split] = [
            {"sample_id": f"review_tuning_v2_{split}_{index + 1:04d}",
             "scenario_type": SCENARIOS[index % len(SCENARIOS)],
             "ground_truth": {"parent_asin": key}, "user_profile": {},
             "evidence_kind": "fresh family-disjoint synthetic target recovery"}
            for index, key in enumerate(ids)
        ]
    families = {split: {title_key(by_id[key].get("title") or "", True) for key in ids}
                for split, ids in assigned.items()}
    overlap = {
        "cross_split_targets": len(set(assigned["development"]) & set(assigned["reserved"])),
        "cross_split_families": len(families["development"] & families["reserved"]),
        "prior_targets": len(set().union(*map(set, assigned.values())) & excluded_ids),
        "prior_families": len(set().union(*families.values()) & excluded_families),
    }
    if any(overlap.values()):
        raise ValueError(f"Dataset leakage: {overlap}")
    return {"datasets": datasets, "audit": {
        "excluded_targets": len(excluded_ids), "excluded_families": len(excluded_families),
        "eligible_families_by_band": {band_label(k): len(v) for k, v in sorted(buckets.items())},
        "overlap": overlap,
        "counts": {split: len(rows) for split, rows in datasets.items()},
        "scenarios": {split: dict(Counter(row["scenario_type"] for row in rows))
                      for split, rows in datasets.items()},
        "bands": {split: dict(Counter(band_label(band_of(rating_number(by_id[key]))) for key in ids))
                  for split, ids in assigned.items()},
    }}


def prepare(pack: Path = PACK, catalog: Path = Path("data/catalog.jsonl")) -> dict:
    if pack.exists():
        raise FileExistsError("Pack is already locked; use verify, never regenerate a reserve")
    excluded, inventory = discover_inventory(catalog, pack)
    built = build_fresh_pack(_read_rows(catalog), excluded)
    pack.mkdir(parents=True, exist_ok=False)
    for split, rows in built["datasets"].items():
        with (pack / f"{split}.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema": SEED, "created_at_utc": now(), "catalog": str(catalog),
        "catalog_sha256": file_sha256(catalog), "inventory": inventory,
        "file_sha256": {split: file_sha256(pack / f"{split}.jsonl") for split in built["datasets"]},
        "audit": built["audit"], "baseline_config": Config.load("configs/selected.json").to_dict(),
        "baseline_config_sha256": file_sha256(Path("configs/selected.json")),
        "protocol_sha256": file_sha256(PROTOCOL), "source_hashes": source_hashes(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "reserved_status": "sealed; procedural access lock, not encryption",
    }
    write_json(pack / "manifest.json", manifest)
    return manifest


def verify(pack: Path = PACK) -> dict:
    manifest = json.loads((pack / "manifest.json").read_text())
    if manifest["schema"] != SEED or manifest["protocol_sha256"] != file_sha256(PROTOCOL):
        raise ValueError("Protocol/manifest drift")
    for split, expected in manifest["file_sha256"].items():
        if file_sha256(pack / f"{split}.jsonl") != expected:
            raise ValueError(f"Frozen {split} data changed")
    for path, expected in {**manifest["inventory"], manifest["catalog"]: manifest["catalog_sha256"]}.items():
        if file_sha256(Path(path)) != expected:
            raise ValueError(f"Inventory/catalog drift: {path}")
    if source_hashes() != manifest["source_hashes"]:
        raise ValueError("Source changed since dataset lock")
    if file_sha256(Path("configs/selected.json")) != manifest["baseline_config_sha256"]:
        raise ValueError("Production baseline changed since dataset lock")
    return manifest


def grid(base: Config) -> list[tuple[str, Config]]:
    return [
        (f"a{round(ratio * 100):03d}_pre{round(pre * 100):03d}_post{round(post * 100):03d}",
         replace(base, review_prior_mode="mixed", review_prior_count_fraction=ratio,
                 review_prior_pre_weight=pre, review_prior_post_weight=post))
        for ratio, pre, post in itertools.product(RATIOS, PRE_WEIGHTS, POST_WEIGHTS)
    ]


class PredictionMemo:
    """Development-only memo; fixed model, hashed inputs, uncached-equivalent tokens."""

    def __init__(self, ranker, capacity: int = 100_000):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Memo capacity must be positive")
        self.ranker, self.original, self.capacity = ranker, ranker._predict_logits, capacity
        self.cache: OrderedDict[tuple, tuple[float, int]] = OrderedDict()
        self.hits = self.physical_pairs = self.physical_tokens = 0

    def predict(self, query: str, documents: list[str]) -> list[float]:
        keys = [(hashlib.sha256(query.encode()).digest(), hashlib.sha256(doc.encode()).digest())
                for doc in documents]
        # Restore hit values before LRU insertion can evict an earlier hit.
        found = {key: self.cache[key] for key in keys if key in self.cache}
        self.hits += sum(key in found for key in keys)
        missing = {}
        for key, document in zip(keys, documents):
            if key not in found:
                missing.setdefault(key, document)
        before = self.ranker.prompt_tokens
        if missing:
            logits = self.original(query, list(missing.values()))
            lengths = self.ranker._last_prediction_token_lengths
            if len(logits) != len(missing) or len(lengths) != len(missing):
                raise ValueError("Malformed prediction memo input")
            self.physical_pairs += len(missing)
            self.physical_tokens += self.ranker.prompt_tokens - before
            for key, value, tokens in zip(missing, logits, lengths):
                found[key] = (value, tokens)
        for key in keys:
            self.cache[key] = found[key]
            self.cache.move_to_end(key)
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        lengths = [found[key][1] for key in keys]
        self.ranker._last_prediction_token_lengths = lengths
        self.ranker.prompt_tokens = before + sum(lengths)
        return [found[key][0] for key in keys]


def run_arm(agent, config: Config, name: str, samples: list[dict], index: tuple) -> dict:
    agent.config = config  # All model, retrieval, and state settings are fixed across this grid.
    observed = ObservedAgent(agent)
    started = time.perf_counter()
    result = evaluate(observed, samples, *index)
    diagnostics = summarize_traces(observed.traces, samples, result["sessions"])
    if agent.startup_fallbacks or diagnostics["fallback_turns"] or diagnostics["failure_diagnostics"]["agent_error_turns"]:
        raise RuntimeError("Tuning requires healthy neural runs; fallback results cannot select a finalist")
    print(json.dumps({"arm": name, "technical_score": result["recommended_technical_score"],
                      "hit_rate": result["hit_rate_at_10"], "tokens": result["reported_token_usage"]["total_tokens"]}),
          flush=True)
    return {"name": name, "config": config.to_dict(), "result": result,
            "breakdowns": breakdowns(samples, result["sessions"], index[2]),
            "diagnostics": diagnostics, "seconds": time.perf_counter() - started}


def breakdowns(samples: list[dict], sessions: list[dict], products: dict) -> dict:
    targets = {row["sample_id"]: row["ground_truth"]["parent_asin"] for row in samples}
    if len(targets) != len(samples) or set(targets) != {row["sample_id"] for row in sessions}:
        raise ValueError("Breakdown sessions must match samples exactly")
    grouped = {"review_band": defaultdict(list), "scenario": defaultdict(list)}
    for row in sessions:
        label = band_label(band_of(rating_number(products[targets[row["sample_id"]]])))
        grouped["review_band"][label].append(row)
        grouped["scenario"][row["scenario_type"]].append(row)
    result = {}
    for dimension, groups in grouped.items():
        result[dimension] = {}
        for label, rows in sorted(groups.items()):
            metrics = metric_summary(rows)
            metrics["recommended_technical_score"] = round(
                .5 * metrics["hit_rate_at_10"] + .3 * metrics["mrr"] + .02 * (11 - metrics["mttc"]), 6)
            result[dimension][label] = metrics
    return result


def selection_key(run: dict) -> tuple:
    config = run["config"]
    return (-run["result"]["recommended_technical_score"], config["review_prior_post_weight"],
            config["review_prior_pre_weight"], abs(config["review_prior_count_fraction"] - .5),
            config["review_prior_count_fraction"])


def eligible(run: dict, baseline: dict) -> bool:
    return (run["result"]["hit_rate_at_10"] >= baseline["result"]["hit_rate_at_10"]
            and run["diagnostics"]["constraint_audit"]["returned_contradictions"]
            <= baseline["diagnostics"]["constraint_audit"]["returned_contradictions"])


def search(pack: Path = PACK, output: Path = OUTPUT) -> dict:
    from mercury.agent import Agent

    manifest = verify(pack)
    if output.exists():
        raise FileExistsError("Development run already exists; do not replace its selection history")
    base = Config.from_dict(manifest["baseline_config"])
    if not base.neural_rerank or base.neural_logit_cache or base.turn_budget_seconds:
        raise ValueError("This experiment requires fixed uncached neural work without a time-adaptive budget")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "development-started.json", {"at": now(), "manifest_sha256": file_sha256(pack / "manifest.json")})
    samples = load_jsonl(pack / "development.jsonl")
    index = catalog_index(manifest["catalog"])
    agent = Agent(manifest["catalog"], base)
    try:
        baseline = run_arm(agent, base, "control_uncached", samples, index)
        write_json(output / "control-uncached.json", baseline)
        memo = PredictionMemo(agent.reranker)
        agent.reranker._predict_logits = memo.predict
        choices = grid(base)
        control_name = next(name for name, config in choices if config == base)
        choices.sort(key=lambda row: row[0] != control_name)
        runs = []
        for name, config in choices:
            run = run_arm(agent, config, name, samples, index)
            write_json(output / f"{name}.json", run)
            runs.append(run)
            if name == control_name and run["result"] != baseline["result"]:
                raise RuntimeError("Memoized control failed full outcome/token parity")
        others = [run for run in runs if run["name"] != control_name]
        allowed = [run for run in others if eligible(run, baseline)]
        winner = min(allowed or others, key=selection_key)
        agent.reranker._predict_logits = memo.original
        finalist = run_arm(agent, Config.from_dict(winner["config"]), "finalist_uncached", samples, index)
        write_json(output / "finalist-uncached.json", finalist)
        if finalist["result"] != winner["result"]:
            raise RuntimeError("Memoized finalist failed full outcome/token parity; reserved remains closed")
        verify(pack)
        report = {"schema": SEED, "control_name": control_name, "winner_name": winner["name"],
                  "runs": runs, "uncached_control": baseline, "uncached_finalist": finalist,
                  "memo_parity": True, "memo_hits": memo.hits, "physical_pairs": memo.physical_pairs,
                  "physical_tokens": memo.physical_tokens,
                  "memo_latency_note": "Development timings include cross-arm memoization; not production latency.",
                  "source_hashes": source_hashes(), "created_at_utc": now()}
        write_json(output / "development-report.json", report)
        freeze = {"candidate_name": winner["name"], "candidate_config": winner["config"],
                  "baseline_config": base.to_dict(), "source_hashes": source_hashes(),
                  "manifest_sha256": file_sha256(pack / "manifest.json"),
                  "development_report_sha256": file_sha256(output / "development-report.json"),
                  "development_eligible": bool(allowed),
                  "development_delta": finalist["result"]["recommended_technical_score"]
                  - baseline["result"]["recommended_technical_score"], "frozen_at_utc": now()}
        write_json(output / "finalist-freeze.json", freeze)
        return freeze
    finally:
        agent.close()


def paired_interval(baseline: dict, candidate: dict) -> dict:
    import numpy as np

    left, right = baseline["result"]["sessions"], candidate["result"]["sessions"]
    if not left or [row["sample_id"] for row in left] != [row["sample_id"] for row in right]:
        raise ValueError("Paired outcomes must have aligned session IDs")
    def value(row):
        return .5 * row["hit"] + .3 * row["reciprocal_rank"] + .02 * (11 - (row["first_hit_turn"] or 11))
    delta = np.array([value(b) - value(a) for a, b in zip(left, right)])
    indices = np.random.default_rng(20260831).integers(0, len(delta), size=(10000, len(delta)))
    return {"mean": float(delta.mean()), "ci95": np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()}


def reserve(pack: Path = PACK, output: Path = OUTPUT) -> dict:
    from mercury.agent import Agent

    manifest = verify(pack)
    freeze = json.loads((output / "finalist-freeze.json").read_text())
    if (freeze["source_hashes"] != source_hashes()
            or freeze["manifest_sha256"] != file_sha256(pack / "manifest.json")
            or freeze["development_report_sha256"] != file_sha256(output / "development-report.json")):
        raise ValueError("Finalist freeze drift; do not open reserved outcomes")
    # Consume before model loading or any outcome access, including a failed run.
    write_json(output / "reserved-consumed.json", {
        "opened_at_utc": now(), "finalist_freeze_sha256": file_sha256(output / "finalist-freeze.json"),
        "reserved_sha256": manifest["file_sha256"]["reserved"], "policy": "one frozen pair only",
    })
    samples = load_jsonl(pack / "reserved.jsonl")
    index = catalog_index(manifest["catalog"])
    runs = []
    for label in ("baseline", "candidate"):
        config = Config.from_dict(freeze[f"{label}_config"])
        agent = Agent(manifest["catalog"], config)
        try:
            run = run_arm(agent, config, label, samples, index)
            write_json(output / f"reserved-{label}.json", run)
            runs.append(run)
        finally:
            agent.close()
    verify(pack)
    baseline, candidate = runs
    interval = paired_interval(baseline, candidate)
    delta = candidate["result"]["recommended_technical_score"] - baseline["result"]["recommended_technical_score"]
    tokens = [run["result"]["reported_token_usage"]["total_tokens"] for run in runs]
    gates = {"development_gain": freeze["development_eligible"] and freeze["development_delta"] >= .005,
             "reserved_gain": delta >= .005, "hit_and_constraint_preservation": eligible(candidate, baseline),
             "paired_interval_above_zero": interval["ci95"][0] > 0,
             "token_budget": tokens[1] <= 1.05 * tokens[0]}
    result = {"runs": runs, "paired_delta": interval, "gates": gates,
              "passes_fresh_proxy_gate": all(gates.values()),
              "production_changed": False,
              "scope": "Fresh lower/medium-popularity proxy only; no new high-popularity evidence.",
              "finalist_freeze_sha256": file_sha256(output / "finalist-freeze.json"),
              "source_hashes": source_hashes(), "finished_at_utc": now()}
    write_json(output / "reserved-report.json", result)
    return {key: value for key, value in result.items() if key != "runs"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "search", "reserve"))
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.pack)
        print(json.dumps({"audit": result["audit"], "hashes": result["file_sha256"]}, indent=2))
    elif args.action == "verify":
        print(json.dumps(verify(args.pack)["audit"], indent=2))
    else:
        result = search(args.pack, args.output) if args.action == "search" else reserve(args.pack, args.output)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
