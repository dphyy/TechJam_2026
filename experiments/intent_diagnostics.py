from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mercury.intent import decide_intent
from mercury.model_assets import file_sha256, verify_model
from mercury.neural import load_encoder
from mercury.state import SessionState


LABELS = ("buying", "browsing", "mixed")
FEATURE_VERSION = "intent-structural-v1"
STRUCTURAL_FEATURES = (
    "known_object", "specified_attribute_count", "hard_constraint_count", "negative_count",
    "use_case_without_object", "unresolved_count", "prior_preference_count",
    "preference_added_count", "preference_removed_count", "feedback_scope_changed",
)
SOURCE_PATHS = (
    Path("data/intent_authored_v1.json"), Path("experiments/intent_dataset.py"),
    Path("experiments/intent_diagnostics.py"), Path("mercury/intent.py"), Path("mercury/state.py"),
)
REGULARIZATION_GRID = (0.01, 0.1, 1.0, 10.0)
TEMPERATURE_GRID = (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0)
ABSTENTION_GRID = tuple(round(value, 2) for value in np.arange(0.34, 0.86, 0.02))


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _signature(state: SessionState) -> set[tuple]:
    return {(item.attribute, item.value, item.polarity, item.hard, item.scope, item.depends_on)
            for item in state.active_preferences()}


def structural_vector(row: dict) -> np.ndarray:
    state = SessionState({}, mode="ledger", alternatives_mode="grouped", scoped_preferences=True)
    for turn, message in enumerate(row.get("history", []), 1):
        state.update(message, turn)
    before = _signature(state)
    state.update(row["message"], len(row.get("history", [])) + 1)
    after = _signature(state)
    active = state.active_preferences()
    positives = [item for item in active if item.polarity == 1]
    known_object = any(item.attribute == "category" for item in positives)
    use_case = any(item.attribute == "use_case" for item in positives)
    return np.asarray((
        float(known_object),
        float(len({item.attribute for item in positives if item.attribute != "other"})),
        float(sum(item.hard for item in active)),
        float(sum(item.polarity == -1 for item in active)),
        float(use_case and not known_object),
        float(sum(item.attribute == "other" and item.polarity == 1 for item in active)),
        float(len(before)),
        float(len(after - before)),
        float(len(before - after)),
        float(state.last_feedback.scope != "none"),
    ), dtype=np.float64)


def _state_for_rules(row: dict) -> SessionState:
    state = SessionState({}, mode="ledger", alternatives_mode="grouped", scoped_preferences=True)
    for turn, message in enumerate(row.get("history", []), 1):
        state.update(message, turn)
    state.update(row["message"], len(row.get("history", [])) + 1)
    return state


def rules_probabilities(rows: list[dict]) -> np.ndarray:
    probabilities = []
    for row in rows:
        decision = decide_intent(_state_for_rules(row), row["message"])
        selected = min(0.999, max(1 / len(LABELS), decision.confidence))
        remainder = (1.0 - selected) / (len(LABELS) - 1)
        values = np.full(len(LABELS), remainder, dtype=np.float64)
        values[LABELS.index(decision.mode)] = selected
        probabilities.append(values)
    return np.asarray(probabilities)


def _semantic_vectors(rows: list[dict], encoder) -> np.ndarray:
    vectors = encoder.encode([row["message"] for row in rows], normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
    result = np.asarray(vectors, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != len(rows) or not np.isfinite(result).all():
        raise ValueError("Intent encoder returned invalid vectors")
    return result


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    adjusted = logits / temperature
    adjusted -= adjusted.max(axis=1, keepdims=True)
    values = np.exp(adjusted)
    return values / values.sum(axis=1, keepdims=True)


def _probability_logits(probabilities: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probabilities, 1e-12, 1.0))


def _targets(rows: list[dict]) -> np.ndarray:
    return np.asarray([LABELS.index(row["label"]) for row in rows], dtype=np.int64)


def _macro_f1(targets: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for label in range(len(LABELS)):
        true_positive = int(np.sum((targets == label) & (predictions == label)))
        false_positive = int(np.sum((targets != label) & (predictions == label)))
        false_negative = int(np.sum((targets == label) & (predictions != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _ece(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            result += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return result


def metrics(rows: list[dict], probabilities: np.ndarray, threshold: float,
            latency_seconds: float = 0.0, include_slices: bool = True) -> dict:
    targets = _targets(rows)
    predictions = probabilities.argmax(axis=1)
    abstained = probabilities.max(axis=1) < threshold
    effective = predictions.copy()
    effective[abstained] = -1
    matrix = [[int(np.sum((targets == truth) & (effective == predicted)))
               for predicted in range(-1, len(LABELS))] for truth in range(len(LABELS))]
    per_class = {}
    for index, label in enumerate(LABELS):
        true_positive = int(np.sum((targets == index) & (effective == index)))
        predicted_count = int(np.sum(effective == index))
        actual_count = int(np.sum(targets == index))
        per_class[label] = {
            "precision": true_positive / predicted_count if predicted_count else 0.0,
            "recall": true_positive / actual_count if actual_count else 0.0,
            "support": actual_count,
        }
    selected = probabilities[np.arange(len(rows)), targets]
    one_hot = np.eye(len(LABELS))[targets]
    covered = ~abstained
    report = {
        "count": len(rows), "macro_f1": _macro_f1(targets, effective), "per_class": per_class,
        "confusion_matrix": {"rows": list(LABELS), "columns": ["abstain", *LABELS], "values": matrix},
        "log_loss": float(-np.log(np.clip(selected, 1e-12, 1.0)).mean()),
        "brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "expected_calibration_error": _ece(targets, probabilities),
        "abstention_threshold": threshold, "abstention_rate": float(abstained.mean()),
        "covered_accuracy": float((predictions[covered] == targets[covered]).mean()) if covered.any() else None,
        "latency_seconds": latency_seconds, "milliseconds_per_row": 1000 * latency_seconds / len(rows),
    }
    if include_slices:
        names = sorted({name for row in rows for name in row.get("slices", [])})
        report["slices"] = {
            name: metrics([row for row in rows if name in row.get("slices", [])], probabilities[
                [name in row.get("slices", []) for row in rows]], threshold, include_slices=False)
            for name in names
        }
    return report


def _temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    return min(TEMPERATURE_GRID, key=lambda value: float(
        -np.log(np.clip(_softmax(logits, value)[np.arange(len(targets)), targets], 1e-12, 1.0)).mean()
    ))


def _abstention_threshold(probabilities: np.ndarray, targets: np.ndarray) -> float:
    candidates = []
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    for threshold in ABSTENTION_GRID:
        covered = confidence >= threshold
        coverage = float(covered.mean())
        if coverage < 0.8:
            continue
        accuracy = float((predictions[covered] == targets[covered]).mean()) if covered.any() else 0.0
        candidates.append((accuracy, -float(1.0 - coverage), threshold))
    return max(candidates)[2] if candidates else 1 / len(LABELS)


def _serialize_linear(model: LogisticRegression, scaler: StandardScaler, regularization: float,
                      temperature: float, threshold: float, feature_names: list[str]) -> dict:
    return {
        "kind": "multinomial_logistic_regression", "regularization_c": regularization,
        "feature_names": feature_names, "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(), "coefficients": model.coef_.tolist(),
        "intercepts": model.intercept_.tolist(), "temperature": temperature,
        "abstention_threshold": threshold,
    }


def _linear_logits(artifact: dict, values: np.ndarray) -> np.ndarray:
    scaled = (values - np.asarray(artifact["scaler_mean"])) / np.asarray(artifact["scaler_scale"])
    with np.errstate(all="ignore"):
        logits = (np.ascontiguousarray(scaled) @ np.asarray(artifact["coefficients"]).T
                  + np.asarray(artifact["intercepts"]))
    if not np.isfinite(logits).all():
        raise ValueError("Intent linear model produced non-finite predictions")
    return logits


def _fit_linear(train_values: np.ndarray, train_targets: np.ndarray, validation_values: np.ndarray,
                validation_targets: np.ndarray, feature_names: list[str], seed: int) -> tuple[dict, np.ndarray]:
    candidates = []
    for regularization in REGULARIZATION_GRID:
        scaler = StandardScaler().fit(train_values)
        model = LogisticRegression(C=regularization, max_iter=3000, random_state=seed, solver="lbfgs")
        # Accelerate/vecLib can retain floating-point flags across an otherwise
        # finite arm64 matmul. Validate outputs explicitly instead of surfacing
        # those stale flags as optimizer warnings.
        with np.errstate(all="ignore"):
            model.fit(np.ascontiguousarray(scaler.transform(train_values)), train_targets)
            logits = model.decision_function(np.ascontiguousarray(scaler.transform(validation_values)))
        if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all() \
                or not np.isfinite(logits).all():
            raise ValueError("Intent linear model produced non-finite parameters")
        temperature = _temperature(logits, validation_targets)
        probabilities = _softmax(logits, temperature)
        score = _macro_f1(validation_targets, probabilities.argmax(axis=1))
        loss = float(-np.log(np.clip(probabilities[np.arange(len(validation_targets)), validation_targets],
                                     1e-12, 1.0)).mean())
        candidates.append((score, -loss, -regularization, model, scaler, temperature, probabilities))
    _, _, negative_regularization, model, scaler, temperature, probabilities = max(
        candidates, key=lambda item: item[:3]
    )
    threshold = _abstention_threshold(probabilities, validation_targets)
    artifact = _serialize_linear(model, scaler, -negative_regularization, temperature, threshold, feature_names)
    return artifact, probabilities


def _source_hashes() -> dict[str, str]:
    return {str(path): file_sha256(path) for path in SOURCE_PATHS}


def _artifact_hash(value: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _verify_data(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    for name, filename in manifest["files"].items():
        if file_sha256(root / filename) != manifest["sha256"][name]:
            raise ValueError(f"Changed intent split: {name}")
    return manifest


def fit(data_root: Path, output: Path, model_root: Path, seed: int = 20260829,
        encoder=None) -> dict:
    manifest = _verify_data(data_root)
    train = _load_jsonl(data_root / manifest["files"]["train"])
    validation = _load_jsonl(data_root / manifest["files"]["validation"])
    train_targets, validation_targets = _targets(train), _targets(validation)
    train_structural = np.vstack([structural_vector(row) for row in train])
    validation_structural = np.vstack([structural_vector(row) for row in validation])
    encoder_manifest = verify_model(model_root, "embedding") if encoder is None else {"revision": "test-encoder"}
    actual_encoder = load_encoder(model_root) if encoder is None else encoder
    train_semantic = _semantic_vectors(train, actual_encoder)
    validation_semantic = _semantic_vectors(validation, actual_encoder)

    models, validation_probabilities = {}, {}
    raw_rules = rules_probabilities(validation)
    rules_logits = _probability_logits(raw_rules)
    rules_temperature = _temperature(rules_logits, validation_targets)
    validation_probabilities["rules_only"] = _softmax(rules_logits, rules_temperature)
    models["rules_only"] = {
        "kind": "current_runtime_rules", "temperature": rules_temperature,
        "abstention_threshold": _abstention_threshold(validation_probabilities["rules_only"], validation_targets),
    }
    for name, train_values, validation_values, feature_names in (
        ("structural", train_structural, validation_structural, list(STRUCTURAL_FEATURES)),
        ("semantic_linear", train_semantic, validation_semantic,
         [f"embedding_{index}" for index in range(train_semantic.shape[1])]),
        ("hybrid_linear", np.hstack((train_structural, train_semantic)),
         np.hstack((validation_structural, validation_semantic)),
         [*STRUCTURAL_FEATURES, *[f"embedding_{index}" for index in range(train_semantic.shape[1])]]),
    ):
        models[name], validation_probabilities[name] = _fit_linear(
            train_values, train_targets, validation_values, validation_targets, feature_names, seed,
        )
    validation_reports = {
        name: metrics(validation, probabilities, models[name]["abstention_threshold"])
        for name, probabilities in validation_probabilities.items()
    }
    for model in models.values():
        model["sha256"] = _artifact_hash(model)
    freeze = {
        "protocol": "phase15-intent-diagnostics-v1", "created_before_sealed_test": True,
        "data_root": str(data_root), "dataset_manifest_sha256": file_sha256(data_root / "manifest.json"),
        "split_hashes": manifest["sha256"], "seed": seed, "feature_version": FEATURE_VERSION,
        "source_hashes": _source_hashes(), "encoder": {
            "kind": "embedding", "revision": encoder_manifest["revision"],
            "manifest_sha256": file_sha256(model_root / "asset_manifest.json") if encoder is None else "test-encoder",
        },
        "selection": {
            "regularization": "validation macro F1, then validation log loss, then simpler regularization",
            "calibration": "validation log-loss temperature scaling",
            "uncertainty": "confidence abstention selected on validation at >=80% coverage; no explicit uncertain class",
        },
        "models": models, "validation_reports": validation_reports,
    }
    freeze["coefficient_and_calibration_sha256"] = _artifact_hash(models)
    output.mkdir(parents=True, exist_ok=False)
    (output / "model-freeze.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    return freeze


def _predict_linear(artifact: dict, values: np.ndarray) -> np.ndarray:
    return _softmax(_linear_logits(artifact, values), artifact["temperature"])


def evaluate_sealed(freeze_path: Path, output: Path, model_root: Path, encoder=None) -> dict:
    freeze = json.loads(freeze_path.read_text())
    data_root = Path(freeze["data_root"])
    manifest = _verify_data(data_root)
    if file_sha256(data_root / "manifest.json") != freeze["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed after fitting")
    if freeze["source_hashes"] != _source_hashes():
        raise ValueError("Diagnostic source changed after fitting")
    if freeze["coefficient_and_calibration_sha256"] != _artifact_hash(freeze["models"]):
        raise ValueError("Frozen coefficients or calibration changed")
    if encoder is None:
        verify_model(model_root, "embedding")
        if file_sha256(model_root / "asset_manifest.json") != freeze["encoder"]["manifest_sha256"]:
            raise ValueError("Intent encoder manifest changed")
    if output.exists():
        raise FileExistsError(output)
    consumed = data_root / "sealed-test-consumed.json"
    with consumed.open("x") as handle:
        json.dump({"freeze": str(freeze_path), "freeze_sha256": file_sha256(freeze_path),
                   "output": str(output), "opened_at_unix": time.time()}, handle, indent=2)
        handle.write("\n")
    output.mkdir(parents=True, exist_ok=False)
    rows = _load_jsonl(data_root / manifest["files"]["sealed_test"])
    started = time.perf_counter()
    rules = rules_probabilities(rows)
    rules_seconds = time.perf_counter() - started
    structural = np.vstack([structural_vector(row) for row in rows])
    structural_seconds = time.perf_counter() - started - rules_seconds
    actual_encoder = load_encoder(model_root) if encoder is None else encoder
    encoding_started = time.perf_counter()
    semantic = _semantic_vectors(rows, actual_encoder)
    encoding_seconds = time.perf_counter() - encoding_started
    models = freeze["models"]
    probabilities = {
        "rules_only": _softmax(_probability_logits(rules), models["rules_only"]["temperature"]),
        "structural": _predict_linear(models["structural"], structural),
        "semantic_linear": _predict_linear(models["semantic_linear"], semantic),
        "hybrid_linear": _predict_linear(models["hybrid_linear"], np.hstack((structural, semantic))),
    }
    latency = {"rules_only": rules_seconds, "structural": structural_seconds,
               "semantic_linear": encoding_seconds, "hybrid_linear": encoding_seconds + structural_seconds}
    reports = {name: metrics(rows, values, models[name]["abstention_threshold"], latency[name])
               for name, values in probabilities.items()}
    best = max(reports, key=lambda name: reports[name]["macro_f1"])
    report = {
        "protocol": freeze["protocol"], "sealed_test_opened_once": True,
        "sealed_test_sha256": manifest["sha256"]["sealed_test"], "count": len(rows),
        "reports": reports, "best_by_macro_f1": best,
        "runtime_routing_changed": False, "selected_config_changed": False,
        "next_gate": "A separate joint candidate on a new downstream unseen pack is required before routing changes.",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        "paid_cost_usd": 0.0, "network_inference": False},
    }
    (output / "sealed-test-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate Phase 15 intent classifier diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--data-root", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--model-root", type=Path, default=Path("artifacts/models/embedding"))
    fit_parser.add_argument("--seed", type=int, default=20260829)
    test_parser = subparsers.add_parser("evaluate-sealed")
    test_parser.add_argument("--freeze", type=Path, required=True)
    test_parser.add_argument("--output", type=Path, required=True)
    test_parser.add_argument("--model-root", type=Path, default=Path("artifacts/models/embedding"))
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.command == "fit":
        result = fit(args.data_root, args.output, args.model_root, args.seed)
        printable = {
            "protocol": result["protocol"], "output": str(args.output),
            "coefficient_and_calibration_sha256": result["coefficient_and_calibration_sha256"],
            "validation": {name: {key: report[key] for key in (
                "macro_f1", "log_loss", "brier_score", "expected_calibration_error", "abstention_rate"
            )} for name, report in result["validation_reports"].items()},
        }
    else:
        result = evaluate_sealed(args.freeze, args.output, args.model_root)
        printable = {
            "protocol": result["protocol"], "output": str(args.output),
            "best_by_macro_f1": result["best_by_macro_f1"],
            "reports": {name: {key: report[key] for key in (
                "macro_f1", "log_loss", "brier_score", "expected_calibration_error", "abstention_rate",
                "milliseconds_per_row",
            )} for name, report in result["reports"].items()},
        }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
