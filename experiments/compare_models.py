"""Compare one local reranker variant under identical code, data, and pair caps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.compare import paired_comparison


def _load(run: Path) -> tuple[dict, dict, dict, list]:
    return tuple(
        json.loads((run / name).read_text(encoding="utf-8"))
        for name in ("result.json", "manifest.json", "diagnostics.json", "traces.json")
    )


def _max_pairs(traces: list) -> int:
    return max((turn.get("diagnostics", {}).get("neural_scores", {}).get("scored_pairs", 0)
                for session in traces for turn in session), default=0)


def _asset_bytes(config: dict) -> int | None:
    root = Path(config.get("artifact_dir", "")) / "models" / "reranker"
    if not root.is_dir():
        return None
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def compare_model_runs(control: Path, candidate: Path, max_pairs: int = 60) -> dict:
    if type(max_pairs) is not int or not 1 <= max_pairs <= 60:
        raise ValueError("Model comparison pair cap must be in [1, 60]")
    control_result, control_manifest, control_diagnostics, control_traces = _load(control)
    candidate_result, candidate_manifest, candidate_diagnostics, candidate_traces = _load(candidate)
    if control_manifest.get("dataset_sha256") != candidate_manifest.get("dataset_sha256"):
        raise ValueError("Model comparisons require identical datasets")
    if control_manifest.get("source_hashes") != candidate_manifest.get("source_hashes"):
        raise ValueError("Model comparisons require identical runtime sources")
    left_config = dict(control_manifest.get("config") or {})
    right_config = dict(candidate_manifest.get("config") or {})
    for config in (left_config, right_config):
        config.pop("artifact_dir", None)
    if left_config != right_config:
        raise ValueError("Only the local model artifact may differ in a model comparison")
    pair_counts = {"control": _max_pairs(control_traces), "candidate": _max_pairs(candidate_traces)}
    if max(pair_counts.values()) > max_pairs:
        raise ValueError("Observed reranker work exceeds the fixed pair cap")
    return {
        "schema": "mercury-bounded-model-comparison-v1",
        "paired_metrics": paired_comparison(control_result["sessions"], candidate_result["sessions"]),
        "max_observed_pairs": pair_counts,
        "pair_cap": max_pairs,
        "resources": {
            "control": {
                "p95_seconds": control_diagnostics.get("p95_seconds"),
                "max_rss_bytes": control_manifest.get("max_rss_bytes"),
                "asset_bytes": _asset_bytes(control_manifest.get("config") or {}),
                "tokens": control_result.get("reported_token_usage"),
            },
            "candidate": {
                "p95_seconds": candidate_diagnostics.get("p95_seconds"),
                "max_rss_bytes": candidate_manifest.get("max_rss_bytes"),
                "asset_bytes": _asset_bytes(candidate_manifest.get("config") or {}),
                "tokens": candidate_result.get("reported_token_usage"),
            },
        },
        "interpretation": "Paired development evidence under identical code/data/config and a fixed neural-pair cap.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exactly one local reranker artifact change")
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-pairs", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(compare_model_runs(args.control, args.candidate, args.max_pairs), indent=2))


if __name__ == "__main__":
    main()
