from __future__ import annotations

import json
import re
from pathlib import Path

from evaluator.local_evaluator import load_jsonl
from mercury.catalog import VOCABULARY
from mercury.model_assets import file_sha256


def main() -> None:
    products = {row["parent_asin"]: row for row in load_jsonl("data/catalog.jsonl")}
    development = load_jsonl("artifacts/splits/development.jsonl")
    reserved = load_jsonl("artifacts/splits/reserved.jsonl")
    colors = set(VOCABULARY["color"])

    def key(row: dict, loose: bool = False) -> str:
        tokens = re.findall(r"[a-z0-9]+", str(row.get("title", "")).lower())
        if loose:
            tokens = [token for token in tokens if token not in colors and not token.isdigit()]
        return " ".join(tokens)

    counts = {}
    for loose in (False, True):
        dev_keys = {key(products[sample["ground_truth"]["parent_asin"]], loose) for sample in development}
        count = sum(key(products[sample["ground_truth"]["parent_asin"]], loose) in dev_keys for sample in reserved)
        counts["color_number_stripped_title_overlap" if loose else "normalized_exact_title_overlap"] = count
    report = {
        "development_count": len(development), "reserved_count": len(reserved),
        "reserved_targets_with_development_title_family": counts,
        "method": "Case/punctuation-normalized titles; loose heuristic additionally removes color tokens and numbers.",
        "limitation": "No manufacturer family identifier exists; this is a conservative overlap diagnostic, not proof of family independence.",
        "outcomes_accessed": False,
        "split_manifest_sha256": file_sha256(Path("artifacts/splits/manifest.json")),
    }
    path = Path("artifacts/splits/family_audit.json")
    if path.exists():
        if json.loads(path.read_text()) != report:
            raise ValueError("Existing family audit differs; refusing overwrite")
    else:
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
