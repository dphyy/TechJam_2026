from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def session_score(session: dict) -> float:
    if not session["hit"]:
        return 0.0
    return 0.5 + 0.3 * session["reciprocal_rank"] + 0.02 * (11 - session["first_hit_turn"])


def paired_comparison(control: list[dict], candidate: list[dict], resamples: int = 10000) -> dict:
    left = {item["sample_id"]: item for item in control}
    right = {item["sample_id"]: item for item in candidate}
    if not left or set(left) != set(right) or len(left) != len(control) or len(right) != len(candidate):
        raise ValueError("Comparisons require unique, identical nonempty sample ID sets")
    deltas = [session_score(right[key]) - session_score(left[key]) for key in sorted(left)]
    hit_deltas = [int(right[key]["hit"]) - int(left[key]["hit"]) for key in sorted(left)]
    rng = random.Random(20260826)
    bootstrap = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(resamples))
    return {
        "sample_count": len(deltas), "technical_score_delta": statistics.fmean(deltas),
        "technical_score_95pct_ci": [bootstrap[int(0.025 * resamples)], bootstrap[min(resamples - 1, int(0.975 * resamples))]],
        "hit_rate_delta": statistics.fmean(hit_deltas), "bootstrap_resamples": resamples,
        "bootstrap_seed": 20260826, "practical_gain": statistics.fmean(deltas) >= 0.01,
        "hit_loss_within_limit": statistics.fmean(hit_deltas) >= -0.02,
        "note": "Paired public-session bootstrap; uncertainty does not predict private-test performance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired frozen-protocol comparison of two completed runs.")
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    control = json.loads((args.control / "result.json").read_text())
    candidate = json.loads((args.candidate / "result.json").read_text())
    print(json.dumps(paired_comparison(control["sessions"], candidate["sessions"]), indent=2))


if __name__ == "__main__":
    main()
