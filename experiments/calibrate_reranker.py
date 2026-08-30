"""Group-safe calibration for observed cross-encoder top-margin diagnostics."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

from evaluator.local_evaluator import load_jsonl
from mercury.model_assets import file_sha256


SLOPES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
INTERCEPTS = tuple(value / 2 for value in range(-12, 13))


def _probability(margin: float, slope: float, intercept: float) -> float:
    value = max(-40.0, min(40.0, slope * margin + intercept))
    return 1.0 / (1.0 + math.exp(-value))


def _log_loss(records: list[dict], slope: float, intercept: float) -> float:
    losses = []
    for record in records:
        probability = min(1 - 1e-12, max(1e-12, _probability(record["margin"], slope, intercept)))
        losses.append(-(record["label"] * math.log(probability)
                        + (1 - record["label"]) * math.log(1 - probability)))
    return sum(losses) / len(losses)


def fit_platt(records: list[dict]) -> dict:
    if not records or any(record.get("label") not in {0, 1}
                          or type(record.get("margin")) not in (int, float)
                          or not math.isfinite(record["margin"]) or record["margin"] < 0
                          for record in records):
        raise ValueError("Calibration needs finite nonnegative margins and binary labels")
    slope, intercept = min(
        ((slope, intercept) for slope in SLOPES for intercept in INTERCEPTS),
        key=lambda pair: (_log_loss(records, *pair), pair),
    )
    return {"slope": slope, "intercept": intercept,
            "training_log_loss": _log_loss(records, slope, intercept)}


def calibration_metrics(records: list[dict], probabilities: list[float], bins: int = 10) -> dict:
    if not records or len(records) != len(probabilities):
        raise ValueError("Metrics require one probability per record")
    brier = sum((probability - record["label"]) ** 2
                for record, probability in zip(records, probabilities, strict=True)) / len(records)
    log_loss = sum(
        -(record["label"] * math.log(min(1 - 1e-12, max(1e-12, probability)))
          + (1 - record["label"]) * math.log(min(1 - 1e-12, max(1e-12, 1 - probability))))
        for record, probability in zip(records, probabilities, strict=True)
    ) / len(records)
    ece = 0.0
    bin_rows = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [(record, probability) for record, probability in zip(records, probabilities, strict=True)
                   if low <= probability < high or index == bins - 1 and probability == 1]
        if not members:
            continue
        confidence = sum(probability for _, probability in members) / len(members)
        accuracy = sum(record["label"] for record, _ in members) / len(members)
        ece += len(members) / len(records) * abs(confidence - accuracy)
        bin_rows.append({"low": low, "high": high, "count": len(members),
                         "mean_confidence": confidence, "accuracy": accuracy})
    return {"count": len(records), "brier": brier, "log_loss": log_loss,
            "expected_calibration_error": ece, "bins": bin_rows}


def grouped_calibration(records: list[dict], folds: int = 5) -> dict:
    groups = {record.get("group") for record in records}
    if not records or None in groups or len(groups) < folds or folds < 2:
        raise ValueError("Grouped calibration needs at least one nonempty group per fold")
    assignments = {
        group: int(hashlib.sha256(str(group).encode()).hexdigest(), 16) % folds for group in groups
    }
    probabilities = [None] * len(records)
    fold_reports = []
    for fold in range(folds):
        training = [record for record in records if assignments[record["group"]] != fold]
        validation = [(index, record) for index, record in enumerate(records)
                      if assignments[record["group"]] == fold]
        if not training or not validation:
            raise ValueError("Stable group assignment produced an empty fold; use fewer folds")
        fitted = fit_platt(training)
        fold_probabilities = [_probability(record["margin"], fitted["slope"], fitted["intercept"])
                              for _, record in validation]
        for (index, _), probability in zip(validation, fold_probabilities, strict=True):
            probabilities[index] = probability
        fold_reports.append({"fold": fold, "training_count": len(training),
                             "validation_count": len(validation), "parameters": fitted,
                             "metrics": calibration_metrics([record for _, record in validation],
                                                            fold_probabilities)})
    if any(probability is None for probability in probabilities):
        raise AssertionError("Out-of-fold prediction missing")
    final = fit_platt(records)
    threshold = max(0.0, -final["intercept"] / final["slope"]) if final["slope"] > 0 else None
    return {"folds": folds, "group_count": len(groups), "fold_reports": fold_reports,
            "out_of_fold_metrics": calibration_metrics(records, probabilities),
            "final_parameters": final, "probability_0_5_margin": threshold}


def extract_records(run: Path, dataset: Path) -> list[dict]:
    traces = json.loads((run / "traces.json").read_text(encoding="utf-8"))
    samples = load_jsonl(dataset)
    if len(traces) != len(samples):
        raise ValueError("Trace and dataset session counts differ")
    records = []
    for sample, session in zip(samples, traces, strict=True):
        target = sample["ground_truth"]["parent_asin"]
        group = sample.get("user_group_id") or target
        for turn in session:
            margin = turn.get("diagnostics", {}).get("neural_scores", {}).get("logit_margin")
            response = turn.get("response", {})
            recommendations = response.get("recommendations", []) if isinstance(response, dict) else []
            identifiers = {item.get("parent_asin") for item in recommendations if isinstance(item, dict)}
            if type(margin) in (int, float) and math.isfinite(margin) and margin >= 0:
                records.append({"sample_id": sample["sample_id"], "turn": turn.get("turn"),
                                "group": group, "margin": margin, "label": int(target in identifiers)})
    if not records:
        raise ValueError("Run contains no finite neural margin diagnostics")
    return records


def calibrate_run(run: Path, dataset: Path, output: Path, folds: int = 5) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_sha256 = file_sha256(dataset)
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Calibration dataset does not match the evaluated run")
    if manifest.get("reserved_evaluation") is True:
        raise ValueError("Reserved evaluation outcomes cannot be used for calibration")
    records = extract_records(run, dataset)
    report = {
        "schema": "mercury-reranker-calibration-v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "interpretation": "Grouped development calibration; not organizer-private evidence.",
        "run": str(run.resolve()),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": dataset_sha256,
        "run_manifest_sha256": file_sha256(manifest_path),
        "traces_sha256": file_sha256(run / "traces.json"),
        "record_count": len(records),
        "calibration": grouped_calibration(records, folds),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate reranker margins with target/user-grouped folds")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(calibrate_run(args.run, args.dataset, args.output, args.folds), indent=2))


if __name__ == "__main__":
    main()
