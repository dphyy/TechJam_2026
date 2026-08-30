from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mercury.intent import IntentWeights, decide_intent
from mercury.model_assets import file_sha256
from mercury.state import SessionState


LABELS = ("buying", "browsing", "mixed")
GRID = tuple(value / 20 for value in range(21))
PARAMETERS = (
    "object", "slots", "hard", "buying_language", "browsing_language",
    "use_case_without_object", "unresolved", "sparse_request",
    "buying_threshold", "browsing_threshold",
)
LEGACY = {
    "object": 0.42, "slots": 0.12, "hard": 0.18,
    "buying_language": 0.28, "browsing_language": 0.52,
    "use_case_without_object": 0.25, "unresolved": 0.15,
    "sparse_request": 0.15, "buying_threshold": 0.55,
    "browsing_threshold": 0.55,
}


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(row.get("label") not in LABELS for row in rows):
        raise ValueError(f"Invalid intent rows: {path}")
    return rows


def verify_protocol_inputs(train: Path, validation: Path, manifest_path: Path | None = None) -> dict:
    """Accept only the frozen train/validation files declared by the split manifest."""
    if manifest_path is None:
        if train.parent.resolve() != validation.parent.resolve():
            raise ValueError("Intent train and validation files must share one manifested data root")
        manifest_path = train.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "intent-authored-v1":
        raise ValueError("Unexpected intent dataset manifest protocol")
    files = manifest.get("files")
    hashes = manifest.get("sha256")
    if not isinstance(files, dict) or not isinstance(hashes, dict):
        raise ValueError("Intent dataset manifest is missing files or hashes")
    root = manifest_path.parent
    expected = {
        "train": root / str(files.get("train", "")),
        "validation": root / str(files.get("validation", "")),
    }
    selected = {"train": train, "validation": validation}
    for name, path in selected.items():
        if path.resolve() != expected[name].resolve():
            raise ValueError(f"Intent tuner only accepts the manifested {name} split")
        expected_hash = hashes.get(name)
        if not isinstance(expected_hash, str) or file_sha256(path) != expected_hash:
            raise ValueError(f"Intent {name} split does not match its frozen manifest hash")
    if expected["train"].resolve() == expected["validation"].resolve():
        raise ValueError("Intent train and validation splits must be distinct")
    return manifest


def _state(row: dict) -> SessionState:
    state = SessionState({}, mode="ledger", alternatives_mode="grouped", scoped_preferences=True)
    for turn, message in enumerate(row.get("history", []), 1):
        state.update(message, turn)
    state.update(row["message"], len(row.get("history", [])) + 1)
    return state


def _predict(row: dict, parameters: dict[str, float]) -> str:
    weights = IntentWeights(**{key: parameters[key] for key in PARAMETERS[:8]})
    return decide_intent(
        _state(row), row["message"], parameters["buying_threshold"],
        parameters["browsing_threshold"], weights=weights,
    ).mode


def _report(rows: list[dict], parameters: dict[str, float]) -> dict:
    predictions = [_predict(row, parameters) for row in rows]
    per_class = {}
    scores = []
    for label in LABELS:
        true_positive = sum(row["label"] == label and prediction == label
                            for row, prediction in zip(rows, predictions, strict=True))
        false_positive = sum(row["label"] != label and prediction == label
                             for row, prediction in zip(rows, predictions, strict=True))
        false_negative = sum(row["label"] == label and prediction != label
                             for row, prediction in zip(rows, predictions, strict=True))
        denominator = 2 * true_positive + false_positive + false_negative
        score = 2 * true_positive / denominator if denominator else 0.0
        scores.append(score)
        per_class[label] = {
            "precision": true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0,
            "recall": true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0,
        }
    return {
        "count": len(rows), "macro_f1": sum(scores) / len(scores), "per_class": per_class,
        "confusion": {
            truth: {predicted: sum(row["label"] == truth and prediction == predicted
                                   for row, prediction in zip(rows, predictions, strict=True))
                    for predicted in LABELS}
            for truth in LABELS
        },
    }


def _folds(rows: list[dict], count: int = 5) -> dict[str, int]:
    result = {}
    for label in LABELS:
        groups = sorted(
            {row["group_id"] for row in rows if row["label"] == label},
            key=lambda value: hashlib.sha256(f"intent-rule-v1\0{value}".encode()).hexdigest(),
        )
        for index, group in enumerate(groups):
            result[group] = index % count
    return result


def _cv_score(rows: list[dict], folds: dict[str, int], parameters: dict[str, float]) -> float:
    scores = [_report([row for row in rows if folds[row["group_id"]] == fold], parameters)["macro_f1"]
              for fold in range(5)]
    return sum(scores) / len(scores)


def _coordinate_search(rows: list[dict], folds: dict[str, int], start: dict[str, float],
                       order: tuple[str, ...]) -> dict[str, float]:
    selected = dict(start)
    for _ in range(8):
        before = dict(selected)
        for key in order:
            incumbent = selected[key]
            candidates = []
            for value in GRID:
                candidate = {**selected, key: value}
                candidates.append((
                    _cv_score(rows, folds, candidate),
                    _report(rows, candidate)["macro_f1"],
                    -abs(value - incumbent), -value, value,
                ))
            selected[key] = max(candidates)[-1]
        if selected == before:
            break
    return selected


def tune(train: list[dict], validation: list[dict]) -> dict:
    train_groups = {row["group_id"] for row in train}
    validation_groups = {row["group_id"] for row in validation}
    if train_groups & validation_groups:
        raise ValueError("Intent groups cross train and validation")
    folds = _folds(train)
    starts = (
        LEGACY,
        {**LEGACY, **{key: 0.25 for key in PARAMETERS[:8]},
         "buying_threshold": 0.50, "browsing_threshold": 0.50},
    )
    candidates = [
        _coordinate_search(train, folds, start, order)
        for start in starts for order in (PARAMETERS, tuple(reversed(PARAMETERS)))
    ]
    selected = max(candidates, key=lambda item: (
        _cv_score(train, folds, item), _report(train, item)["macro_f1"],
        -sum(item.values()), tuple(-item[key] for key in PARAMETERS),
    ))
    return {
        "protocol": "intent-rules-grouped-cv-v1",
        "selection_data": "train only; validation is acceptance evidence; sealed test excluded",
        "grid": list(GRID), "fold_count": 5,
        "legacy": {"parameters": LEGACY, "train_cv_macro_f1": _cv_score(train, folds, LEGACY),
                   "train": _report(train, LEGACY), "validation": _report(validation, LEGACY)},
        "selected": {"parameters": selected,
                     "train_cv_macro_f1": _cv_score(train, folds, selected),
                     "train": _report(train, selected), "validation": _report(validation, selected)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune interpretable intent rules without evaluator targets")
    parser.add_argument("--train", type=Path, default=Path("artifacts/intent-v1/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("artifacts/intent-v1/validation.jsonl"))
    parser.add_argument("--manifest", type=Path,
                        help="Frozen split manifest; defaults to manifest.json beside the train split")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_protocol_inputs(args.train, args.validation, args.manifest)
    report = tune(_load(args.train), _load(args.validation))
    report["input_sha256"] = {
        "train": file_sha256(args.train), "validation": file_sha256(args.validation),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["selected"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
