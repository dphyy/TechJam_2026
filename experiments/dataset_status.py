"""Audit recorded dataset exposure without loading labels or running an agent.

Names and preparation manifests are not live consumption status. Match completed
runs, consumption markers, and attempted runs by dataset content hash instead.
Absence of a local receipt cannot establish that a holdout is untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mercury.model_assets import file_sha256


ROOT = Path(__file__).resolve().parents[1]
_RECEIPT_NAMES = {"manifest.json", "registration.json", "report.json", "consumption-ledger.json"}
_SKIP_DIRECTORIES = {"source", "models", ".git", ".venv", "__pycache__"}


def _has_outcomes(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    sessions = value.get("sessions")
    if isinstance(sessions, list) and sessions:
        return True
    # Early suite reports retained aggregate metrics but not per-session rows.
    return (type(value.get("sample_count")) is int and value["sample_count"] > 0
            and type(value.get("hit_rate_at_10")) in (int, float))


def _receipt_paths(roots: list[Path], warnings: list[str]):
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            warnings.append(f"Evidence root missing: {root}")
            continue
        if not root.is_dir():
            warnings.append(f"Evidence root is not a directory: {root}")
            continue
        for directory, subdirs, filenames in os.walk(
            root, onerror=lambda error: warnings.append(str(error)), followlinks=False,
        ):
            subdirs[:] = sorted(name for name in subdirs if name not in _SKIP_DIRECTORIES)
            for name in sorted(filenames):
                if name not in _RECEIPT_NAMES and not name.endswith(("-report.json", "-consumed.json")):
                    continue
                path = Path(directory) / name
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield path


def _events(path: Path, receipt: dict, digest: str):
    """Interpret explicit receipt fields, never arbitrary nested hash matches."""
    if path.name == "consumption-ledger.json":
        entries = receipt.get("entries")
        if isinstance(entries, dict):
            for split, entry in entries.items():
                if not isinstance(entry, dict):
                    continue
                events = entry.get("events")
                if not isinstance(events, list):
                    continue
                for event in events:
                    if isinstance(event, dict) and event.get("dataset_sha256") == digest:
                        yield {"kind": "consumption_event", "split": split,
                               "opened_at": event.get("opened_at_utc")}
        return

    if path.name.endswith("-consumed.json"):
        if any(receipt.get(key) == digest for key in (
            "dataset_sha256", "reserved_sha256", "sealed_test_sha256",
        )):
            yield {"kind": "consumption_marker", "opened_at": receipt.get("opened_at_utc")}
        return

    if receipt.get("dataset_sha256") != digest:
        return
    # A registration is conservative evidence of an attempted run, even if the
    # run failed before it could emit outcomes. Do not claim it completed.
    if path.name == "registration.json":
        yield {"kind": "evaluation_registered", "completed": False}
    elif path.name == "manifest.json" and "started_at_utc" in receipt:
        yield {"kind": "evaluation_attempt", "completed": "finished_at_utc" in receipt}
    elif path.name == "report.json" or path.name.endswith("-report.json"):
        result = receipt.get("result", receipt)
        runs = receipt.get("runs")
        has_runs = isinstance(runs, list) and any(
            isinstance(run, dict) and (_has_outcomes(run) or _has_outcomes(run.get("metrics")))
            for run in runs
        )
        if _has_outcomes(result) or has_runs:
            yield {"kind": "evaluation_report", "completed": True,
                   "recorded_at": receipt.get("created_at_utc")}


def audit_dataset(dataset: Path, evidence_roots: list[Path] | None = None) -> dict:
    dataset = Path(dataset).resolve()
    digest = file_sha256(dataset)
    roots = [Path(root).resolve() for root in evidence_roots] if evidence_roots is not None else [
        ROOT / "runs", ROOT / "output", ROOT / "artifacts",
    ]
    warnings: list[str] = []
    evidence: list[dict] = []
    for path in _receipt_paths(roots, warnings):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as error:
            warnings.append(f"Cannot inspect {path}: {error}")
            continue
        if isinstance(receipt, dict):
            evidence.extend({"receipt": str(path), **event} for event in _events(path, receipt, digest))
    unchanged = digest == file_sha256(dataset)
    if not unchanged:
        warnings.append("Dataset changed during the audit; rerun against stable bytes")
    confirmed = any(event["kind"] in {
        "consumption_marker", "consumption_event", "evaluation_report",
    } or event.get("completed") is True for event in evidence)
    status = "consumed" if confirmed else "attempt_recorded" if evidence else "unknown"
    if not unchanged:
        status = "unknown"
    return {
        "schema": "mercury-dataset-exposure-audit-v1",
        "dataset": str(dataset), "dataset_sha256": digest,
        "status": status,
        "untouched_holdout_verified": False,
        "evidence_roots": [str(root) for root in roots],
        "scan_complete": not warnings,
        "evidence": evidence, "warnings": warnings,
        "interpretation": (
            "Status describes recorded exposure of these exact bytes across configurations. "
            "A new branch, filename, or output directory does not restore an untouched holdout. "
            "Unknown means no supported matching receipt was found, not unseen. "
            "Remote runs, manual inspection, unrecorded runs, and overlapping/reformatted datasets "
            "are outside this audit; catalog indexing is not evaluation exposure."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, action="append",
                        help="Directory of receipts; repeat to replace the default runs/output/artifacts roots")
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.dataset, args.evidence_root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
