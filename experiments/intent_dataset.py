from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from mercury.model_assets import file_sha256


LABELS = ("buying", "browsing", "mixed")
SPLIT_GROUPS_PER_LABEL = {"train": 14, "validation": 3, "sealed_test": 3}
REQUIRED_SLICES = frozenset({"direct", "indirect", "mixed", "correction", "override", "vague",
                             "conflicting", "oov"})


def _group_key(group: dict) -> tuple[str, str, str, str]:
    return (group["author_id"], group["paraphrase_family"], group["intent_card"], group["product_family"])


def load_authored(path: Path) -> tuple[dict, list[dict]]:
    source = json.loads(path.read_text())
    if source.get("version") != "intent-authored-v1" or source.get("authored_without_mercury_predictions") is not True:
        raise ValueError("Unexpected intent dataset protocol")
    groups = source.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Intent source must contain groups")
    ids, keys = set(), set()
    label_counts = Counter()
    seen_slices = set()
    for group in groups:
        required = {"group_id", "author_id", "label", "paraphrase_family", "intent_card",
                    "product_family", "slices", "history", "utterances"}
        if not required <= group.keys() or group["label"] not in LABELS:
            raise ValueError("Invalid authored intent group")
        if group["group_id"] in ids or _group_key(group) in keys:
            raise ValueError("Duplicate intent group identity")
        if not isinstance(group["utterances"], list) or len(group["utterances"]) < 2:
            raise ValueError("Each group needs at least two independently authored utterances")
        if any(not isinstance(text, str) or not text.strip() for text in group["history"] + group["utterances"]):
            raise ValueError("Intent text must be nonempty")
        ids.add(group["group_id"])
        keys.add(_group_key(group))
        label_counts[group["label"]] += 1
        seen_slices.update(group["slices"])
    if label_counts != Counter({label: 20 for label in LABELS}):
        raise ValueError(f"Expected 20 groups per label, got {dict(label_counts)}")
    if not REQUIRED_SLICES <= seen_slices:
        raise ValueError(f"Missing authored slices: {sorted(REQUIRED_SLICES - seen_slices)}")
    return source, groups


def _allocate(groups: list[dict], seed: int) -> dict[str, list[dict]]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for group in groups:
        by_label[group["label"]].append(group)
    result = {name: [] for name in SPLIT_GROUPS_PER_LABEL}
    for label in LABELS:
        ordered = sorted(by_label[label], key=lambda group: group["group_id"])
        random.Random(f"intent-v1\0{seed}\0{label}").shuffle(ordered)
        offset = 0
        for split, count in SPLIT_GROUPS_PER_LABEL.items():
            result[split].extend(ordered[offset:offset + count])
            offset += count
    return result


def _expand(groups: list[dict]) -> list[dict]:
    rows = []
    for group in sorted(groups, key=lambda item: item["group_id"]):
        for index, message in enumerate(group["utterances"], 1):
            sample_id = hashlib.sha256(f"{group['group_id']}\0{index}\0{message}".encode()).hexdigest()[:16]
            rows.append({
                "sample_id": f"intent_{sample_id}", "group_id": group["group_id"],
                "author_id": group["author_id"], "paraphrase_family": group["paraphrase_family"],
                "intent_card": group["intent_card"], "product_family": group["product_family"],
                "slices": group["slices"], "history": group["history"], "message": message,
                "label": group["label"],
            })
    return rows


def validate_splits(splits: dict[str, list[dict]]) -> dict:
    expected = set(SPLIT_GROUPS_PER_LABEL)
    if set(splits) != expected:
        raise ValueError("Intent splits must contain train, validation, and sealed_test")
    group_sets, author_sets, family_card_product_sets = {}, {}, {}
    summary = {}
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"{name} is empty")
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} has duplicate sample IDs")
        group_sets[name] = {row["group_id"] for row in rows}
        author_sets[name] = {row["author_id"] for row in rows}
        family_card_product_sets[name] = {
            (row["paraphrase_family"], row["intent_card"], row["product_family"]) for row in rows
        }
        groups_by_label = Counter()
        for group_id in group_sets[name]:
            labels = {row["label"] for row in rows if row["group_id"] == group_id}
            if len(labels) != 1:
                raise ValueError("A group cannot contain multiple labels")
            groups_by_label[next(iter(labels))] += 1
        expected_counts = {label: SPLIT_GROUPS_PER_LABEL[name] for label in LABELS}
        if dict(groups_by_label) != expected_counts:
            raise ValueError(f"Unexpected {name} group allocation: {dict(groups_by_label)}")
        summary[name] = {"rows": len(rows), "groups": len(group_sets[name]),
                         "labels": dict(Counter(row["label"] for row in rows))}
    names = tuple(splits)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            for dimension, values in (("group", group_sets), ("author", author_sets),
                                      ("family/card/product", family_card_product_sets)):
                if values[first] & values[second]:
                    raise ValueError(f"Intent {dimension} leakage between {first} and {second}")
    all_rows = [row for rows in splits.values() for row in rows]
    if len({row["sample_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Sample leakage between intent splits")
    return summary


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def prepare(source_path: Path, output: Path, seed: int = 20260829) -> dict:
    source, groups = load_authored(source_path)
    allocated = _allocate(groups, seed)
    splits = {name: _expand(values) for name, values in allocated.items()}
    summary = validate_splits(splits)
    output.mkdir(parents=True, exist_ok=False)
    filenames = {"train": "train.jsonl", "validation": "validation.jsonl",
                 "sealed_test": "sealed-test.jsonl"}
    for name, filename in filenames.items():
        _write_jsonl(output / filename, splits[name])
    hashes = {name: file_sha256(output / filename) for name, filename in filenames.items()}
    manifest = {
        "protocol": source["version"], "seed": seed, "source": str(source_path),
        "source_sha256": file_sha256(source_path), "split_policy": "grouped-stratified-70-15-15",
        "grouping_fields": ["author_id", "paraphrase_family", "intent_card", "product_family"],
        "excluded_sources": ["data/public_set.jsonl", "artifacts/unseen-v1"],
        "files": filenames, "sha256": hashes, "summary": summary,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the independently authored Phase 15 intent dataset")
    parser.add_argument("--source", type=Path, default=Path("data/intent_authored_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
