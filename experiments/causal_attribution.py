"""Target-independent stage receipts and offline paired causal attribution.

Runtime diagnostics contain only live state and candidate IDs.  This module is
an evaluator-side consumer: it may join those receipts to frozen target labels
after a run, but none of its outputs are available to the shopping agent.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from evaluator.local_evaluator import load_jsonl


STAGES = (
    "runtime", "state_or_intent", "retrieval", "admission", "ranking",
    "guard", "question", "paging", "success", "unattributed",
)


def _target_in(turn: dict, key: str, target: str, depth: int | None = None) -> bool:
    values = turn.get("diagnostics", {}).get("stage_ids", {}).get(key, [])
    if not isinstance(values, list):
        return False
    return target in (values[:depth] if depth is not None else values)


def attribute_session(trace: list[dict], result: dict, target: str) -> dict:
    """Assign the earliest observable stage that prevented a successful slate."""
    if not isinstance(trace, list) or not isinstance(result, dict) or not isinstance(target, str):
        raise ValueError("Attribution requires a trace, result, and target ID")
    if any("error" in turn for turn in trace) or any(
        turn.get("diagnostics", {}).get("fallbacks") for turn in trace
    ):
        stage = "runtime"
    elif result.get("hit") is True:
        stage = "success"
    else:
        ever_retrieved = any(
            target in turn.get("diagnostics", {}).get("retrieved_ids", [])
            for turn in trace
        )
        ever_admitted = any(
            _target_in(turn, "admission_selected", target)
            or target in turn.get("diagnostics", {}).get("rerank_prefix_ids", [])
            for turn in trace
        )
        ever_neural_top10 = any(_target_in(turn, "neural_ranked", target, 10) for turn in trace)
        ever_guard_top10 = any(_target_in(turn, "guarded_after_rerank", target, 10) for turn in trace)
        ever_final_top10 = any(_target_in(turn, "final_ranked", target, 10) for turn in trace)
        ever_returned = any(_target_in(turn, "returned_page", target) for turn in trace)
        state_empty = all(
            not turn.get("diagnostics", {}).get("semantic_state_signature")
            for turn in trace
        )
        asked = any(turn.get("diagnostics", {}).get("question", {}).get("attribute") for turn in trace)
        if not ever_retrieved:
            stage = "state_or_intent" if state_empty else "retrieval"
        elif not ever_admitted:
            stage = "admission"
        elif not ever_neural_top10:
            stage = "ranking"
        elif ever_neural_top10 and not ever_guard_top10:
            stage = "guard"
        elif not ever_final_top10:
            stage = "ranking"
        elif ever_final_top10 and not ever_returned:
            stage = "paging"
        elif asked:
            stage = "question"
        elif ever_returned:
            # The official evaluator can delay eligibility in override sessions.
            stage = "paging"
        else:
            stage = "unattributed"
    return {
        "sample_id": result.get("sample_id"),
        "hit": result.get("hit"),
        "stage": stage,
        "first_hit_turn": result.get("first_hit_turn"),
        "first_hit_rank": result.get("first_hit_rank"),
    }


def timing_reconciliation(traces: list[list[dict]], absolute_tolerance: float = 0.02,
                          relative_tolerance: float = 0.10) -> dict:
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("Timing tolerances must be nonnegative")
    rows = []
    for session in traces:
        for turn in session:
            observed = turn.get("latency_seconds")
            parts = turn.get("diagnostics", {}).get("component_latency_seconds", {})
            if type(observed) not in (int, float) or not math.isfinite(observed) \
                    or not isinstance(parts, dict):
                continue
            measured = sum(
                float(value) for value in parts.values()
                if type(value) in (int, float) and math.isfinite(value) and value >= 0
            )
            delta = abs(float(observed) - measured)
            tolerance = max(absolute_tolerance, relative_tolerance * float(observed))
            rows.append({"observed": float(observed), "components": measured,
                         "delta": delta, "within_tolerance": delta <= tolerance})
    deltas = sorted(row["delta"] for row in rows)
    return {
        "turn_count": len(rows),
        "within_tolerance": sum(row["within_tolerance"] for row in rows),
        "reconciled_fraction": (
            sum(row["within_tolerance"] for row in rows) / len(rows) if rows else 1.0
        ),
        "delta_p50_seconds": statistics.median(deltas) if deltas else 0.0,
        "delta_p95_seconds": deltas[min(len(deltas) - 1, int(0.95 * len(deltas)))] if deltas else 0.0,
        "absolute_tolerance_seconds": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }


def summarize_run(dataset: list[dict], results: list[dict], traces: list[list[dict]]) -> dict:
    if len(dataset) != len(results) or len(dataset) != len(traces):
        raise ValueError("Dataset, results, and traces must have equal lengths")
    result_by_id = {row.get("sample_id"): row for row in results}
    if len(result_by_id) != len(results):
        raise ValueError("Result sample IDs must be unique")
    rows = []
    for sample, trace in zip(dataset, traces, strict=True):
        sample_id = sample.get("sample_id")
        result = result_by_id.get(sample_id)
        if result is None:
            raise ValueError(f"Missing result for {sample_id}")
        target = sample.get("ground_truth", {}).get("parent_asin")
        if not isinstance(target, str) or not target:
            raise ValueError("Dataset targets must be nonempty strings")
        row = attribute_session(trace, result, target)
        row["scenario"] = sample.get("scenario_type", "unknown")
        rows.append(row)
    return {
        "session_count": len(rows),
        "stage_counts": dict(Counter(row["stage"] for row in rows)),
        "scenario_stage_counts": {
            scenario: dict(Counter(row["stage"] for row in values))
            for scenario, values in _group(rows, "scenario").items()
        },
        "timing_reconciliation": timing_reconciliation(traces),
        "sessions": rows,
    }


def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return dict(grouped)


def paired_diff(control: dict, candidate: dict) -> dict:
    left = {row["sample_id"]: row for row in control.get("sessions", [])}
    right = {row["sample_id"]: row for row in candidate.get("sessions", [])}
    if set(left) != set(right):
        raise ValueError("Paired runs must contain identical sample IDs")
    transitions = Counter()
    gained = []
    lost = []
    unchanged = []
    for sample_id in sorted(left):
        before, after = left[sample_id], right[sample_id]
        transitions[(before["stage"], after["stage"])] += 1
        row = {
            "sample_id": sample_id,
            "scenario": before.get("scenario"),
            "control_stage": before["stage"],
            "candidate_stage": after["stage"],
        }
        if not before["hit"] and after["hit"]:
            gained.append(row)
        elif before["hit"] and not after["hit"]:
            lost.append(row)
        else:
            unchanged.append(row)
    return {
        "paired_sessions": len(left),
        "gained_count": len(gained),
        "lost_count": len(lost),
        "net_hits": len(gained) - len(lost),
        "gained": gained,
        "lost": lost,
        "unchanged_count": len(unchanged),
        "stage_transitions": {
            f"{before}->{after}": count
            for (before, after), count in sorted(transitions.items())
        },
    }


def load_run(run: Path, dataset: Path) -> dict:
    run = Path(run)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    traces = json.loads((run / "traces.json").read_text(encoding="utf-8"))
    return summarize_run(load_jsonl(dataset), result["sessions"], traces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute stage failures and paired gains/losses")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    control = load_run(args.control_run, args.dataset)
    report = {"schema": "mercury-causal-attribution-v1", "control": control}
    if args.candidate_run:
        candidate = load_run(args.candidate_run, args.dataset)
        report["candidate"] = candidate
        report["paired_diff"] = paired_diff(control, candidate)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
