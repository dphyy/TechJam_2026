from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from mercury.catalog import Catalog
from mercury.diversity import diversify_candidates
from mercury.types import Candidate


STRENGTHS = (0.10, 0.20, 0.30, 0.40, 0.50)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_oracle(catalog: Catalog, samples: list[dict], traces: list[list[dict]]) -> dict:
    """Evaluator-only browsing-label ceiling; never a valid runtime policy."""
    results = {}
    for strength in STRENGTHS:
        better = worse = same = top10_gain = top10_loss = 0
        reciprocal_rank_deltas = []
        for sample, session in zip(samples, traces, strict=True):
            if sample.get("scenario_type") != "browsing":
                continue
            target = sample["ground_truth"]["parent_asin"]
            for turn in session:
                identifiers = turn.get("diagnostics", {}).get("ranked_ids", [])
                if target not in identifiers:
                    continue
                candidates = [Candidate(catalog.by_id[identifier], -float(index))
                              for index, identifier in enumerate(identifiers)
                              if identifier in catalog.by_id]
                diversified = diversify_candidates(candidates, strength)
                diversified_ids = [item.product.parent_asin for item in diversified]
                old_rank = identifiers.index(target) + 1
                new_rank = diversified_ids.index(target) + 1
                better += new_rank < old_rank
                worse += new_rank > old_rank
                same += new_rank == old_rank
                top10_gain += old_rank > 10 >= new_rank
                top10_loss += old_rank <= 10 < new_rank
                reciprocal_rank_deltas.append(1.0 / new_rank - 1.0 / old_rank)
        results[f"{strength:.2f}"] = {
            "observations": len(reciprocal_rank_deltas),
            "rank_better": better,
            "rank_worse": worse,
            "rank_same": same,
            "top10_gain": top10_gain,
            "top10_loss": top10_loss,
            "mean_reciprocal_rank_delta": round(
                statistics.fmean(reciprocal_rank_deltas), 9,
            ) if reciprocal_rank_deltas else 0.0,
        }
    return {
        "warning": "Evaluator-only oracle using scenario labels; prohibited in runtime.",
        "browsing_strengths": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_oracle(
        Catalog(args.catalog), load_jsonl(args.dataset),
        json.loads(args.traces.read_text(encoding="utf-8")),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
