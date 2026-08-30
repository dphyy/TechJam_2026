from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mercury.agent import Agent
from mercury.config import Config


MESSAGES = (
    "I'm looking for a black leather shoulder bag with a zipper.",
    "Actually, no leather. I would prefer canvas, and blue is fine.",
    "An adjustable strap would be useful. I have no additional preferences about color.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a real multi-turn correction; record actual API outputs.")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/selected.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = Config.load(args.config)
    started = time.perf_counter()
    agent = Agent(args.catalog, config)
    cold_start = time.perf_counter() - started
    agent.reset("recorded-demo", {})
    transcript = []
    events = []
    plain_lines = []

    def emit(text: str, position: float) -> None:
        print(text, flush=True)
        plain_lines.append(text)
        events.append([position, "o", text.replace("\n", "\r\n") + "\r\n"])

    emit("MERCURY | Evidence-driven shopping search", 0.0)
    emit("Live API outputs; replay pacing is for narration, not a latency measurement.", 0.5)
    emit(f"Catalog: {len(agent.catalog.products):,} products | cold start: {cold_start:.2f}s", 1.0)
    for turn, message in enumerate(MESSAGES, 1):
        response_start = time.perf_counter()
        response = agent.respond("recorded-demo", message, turn, 10)
        elapsed = time.perf_counter() - response_start
        diagnostic = agent.last_diagnostics
        position = 10.0 + (turn - 1) * 45.0
        emit(f"\nTURN {turn} | User: {message}", position)
        emit(f"Active query: {diagnostic['query']}", position + 2)
        exclusions = [p["value"] for p in diagnostic["preferences"] if p["polarity"] == -1]
        emit(f"Explicit exclusions: {', '.join(exclusions) or '(none)'}", position + 3)
        emit(f"Agent: {response['message']}", position + 4)
        for rank, item in enumerate(response["recommendations"][:3], 1):
            product = agent.catalog.by_id[item["parent_asin"]]
            title = product.title.replace("\n", " ")[:95]
            emit(f"  {rank}. {product.parent_asin} | {title}", position + 4 + rank)
        emit(f"Measured response: {elapsed:.3f}s | returned: {len(response['recommendations'])} | "
             f"fallbacks: {diagnostic['fallbacks']}", position + 8)
        transcript.append({"turn": turn, "user_message": message, "response": response,
                           "latency_seconds": elapsed, "diagnostics": diagnostic})
    emit("\nVerified IDs, source-aware corrections, and an offline fallback.", 150.0)
    emit("Benchmark evidence is reported separately. This replay has no hidden target or conversion claim.", 155.0)
    emit("End of three-minute presentation replay.", 179.0)
    header = {"version": 2, "width": 120, "height": 40, "duration": 180.0,
              "timestamp": int(time.time()), "title": "Mercury: live API correction replay",
              "env": {"TERM": "xterm-256color", "SHELL": "/bin/sh"}}
    (args.output / "replay.cast").write_text("\n".join(json.dumps(row) for row in [header, *events]) + "\n")
    (args.output / "transcript.txt").write_text("\n".join(plain_lines) + "\n")
    (args.output / "responses.json").write_text(json.dumps(transcript, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps({"config": config.to_dict(),
        "catalog_sha256": agent.catalog.sha256, "cold_start_seconds": cold_start,
        "presentation_duration_seconds": 180, "output_is_video": False,
        "note": "Asciicast terminal replay of actual responses; timings are paced for narration."}, indent=2) + "\n")
    agent.close()


if __name__ == "__main__":
    main()
