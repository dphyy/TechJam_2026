from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiments.compare import paired_comparison


def _ids(turn: dict) -> list[str]:
    return [item["parent_asin"] for item in turn["response"]["recommendations"]]


def paging_behavior(traces: list[list[dict]]) -> dict:
    eligible = duplicates = advanced = changed_resets = override_resets = 0
    coverage: dict[int, list[int]] = {1: [], 2: [], 3: []}
    for session in traces:
        seen: set[str] = set()
        for number in coverage:
            for turn in session[:number]:
                seen.update(_ids(turn))
            coverage[number].append(len(seen))
            seen.clear()
        for index, turn in enumerate(session):
            diagnostics = turn["diagnostics"]
            current = _ids(turn)
            override = diagnostics["slate_selection"].get("override_reset", False)
            if override and diagnostics["slate_page"] == 0:
                override_resets += 1
            if index == 0 or not current:
                continue
            previous_turn = session[index - 1]
            previous = _ids(previous_turn)
            limit = len(current)
            current_ranked = diagnostics.get("ranked_ids", [])
            previous_ranked = previous_turn["diagnostics"].get("ranked_ids", [])
            stable = bool(
                previous
                and limit
                and set(current_ranked[:limit]) == set(previous_ranked[:limit])
            )
            if stable and not override:
                eligible += 1
                duplicates += int(set(current) == set(previous))
                advanced += int(
                    diagnostics["slate_selection"].get("reason") == "highest_ranked_unseen"
                )
            elif not stable and not override and diagnostics["slate_page"] == 0:
                changed_resets += 1
    return {
        "session_count": len(traces),
        "eligible_stable_head_turns": eligible,
        "exact_adjacent_duplicate_slates": duplicates,
        "highest_ranked_unseen_selections": advanced,
        "changed_head_resets": changed_resets,
        "override_resets": override_resets,
        "mean_unique_products_through_turn": {
            str(number): round(statistics.fmean(values), 6)
            for number, values in coverage.items()
        },
    }


def compare_runs(control: Path, candidate: Path) -> dict:
    control_result = json.loads((control / "result.json").read_text())
    candidate_result = json.loads((candidate / "result.json").read_text())
    control_traces = json.loads((control / "traces.json").read_text())
    candidate_traces = json.loads((candidate / "traces.json").read_text())
    if len(control_traces) != len(control_result["sessions"]):
        raise ValueError("Control trace and result session counts differ")
    if len(candidate_traces) != len(candidate_result["sessions"]):
        raise ValueError("Candidate trace and result session counts differ")
    return {
        "paired_official": paired_comparison(
            control_result["sessions"], candidate_result["sessions"]
        ),
        "control_behavior": paging_behavior(control_traces),
        "candidate_behavior": paging_behavior(candidate_traces),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare repeat-driven paging runs")
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.control, args.candidate), indent=2))


if __name__ == "__main__":
    main()
