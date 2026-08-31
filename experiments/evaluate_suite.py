from __future__ import annotations

import argparse
import datetime
import json
import platform
import resource
import time
from dataclasses import dataclass
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.run import ObservedAgent, source_hashes, summarize_traces
from mercury.config import Config
from mercury.model_assets import file_sha256


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SPECS = (
    ("official_baseline", "baseline", None),
    ("mercury_sparse_fallback", "config", REPOSITORY / "configs/sparse_fallback.json"),
    ("mercury_selected", "config", REPOSITORY / "configs/selected.json"),
)


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    kind: str
    config_path: Path | None = None


def _safe_name(value: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("Run names must be safe path components")
    return value


def parse_config_spec(value: str) -> SuiteSpec:
    if "=" not in value:
        raise ValueError("Candidate config specs must be NAME=PATH")
    name, path = value.split("=", 1)
    return SuiteSpec(_safe_name(name), "config", Path(path))


def default_specs() -> list[SuiteSpec]:
    return [SuiteSpec(name, kind, path) for name, kind, path in DEFAULT_SPECS]


def _agent_for(spec: SuiteSpec, catalog: Path):
    if spec.kind == "baseline":
        from baselines.official import Agent

        return Agent(catalog), None
    if spec.kind != "config" or spec.config_path is None:
        raise ValueError(f"Unsupported suite spec: {spec}")
    from mercury.agent import Agent

    config = Config.load(spec.config_path)
    return Agent(catalog, config), config


def _metric_row(result: dict, diagnostics: dict, startup_fallbacks: dict,
                cold_start: float, elapsed: float, max_rss_bytes: int) -> dict:
    return {
        "sample_count": result["sample_count"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "technical_score": result["recommended_technical_score"],
        "latency_p50_seconds": diagnostics["p50_seconds"],
        "latency_p95_seconds": diagnostics["p95_seconds"],
        "latency_max_seconds": diagnostics["max_seconds"],
        "cold_start_seconds": cold_start,
        "evaluation_seconds": elapsed,
        "fallback_turns": diagnostics["fallback_turns"],
        "startup_fallbacks": startup_fallbacks,
        "token_usage": result["reported_token_usage"],
        "scenario_metrics": result["scenario_metrics"],
        "failure_diagnostics": diagnostics["failure_diagnostics"],
        "ever_ranked_recall": diagnostics["ever_ranked_recall"],
        "ever_route_depth_recall": diagnostics["ever_route_depth_recall"],
        "max_rss_bytes": max_rss_bytes,
        "constraint_audit": diagnostics.get("constraint_audit", {}),
    }


def evaluate_suite(specs: list[SuiteSpec], catalog: Path, dataset: Path) -> dict:
    if not specs:
        raise ValueError("At least one run spec is required")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Run names must be unique")
    catalog_sha256 = file_sha256(catalog)
    dataset_sha256 = file_sha256(dataset)
    samples = load_jsonl(dataset)
    ids, categories, products = catalog_index(catalog)
    source_before = source_hashes()
    runs = []
    for spec in specs:
        inner = None
        evaluation_error = None
        try:
            started = time.perf_counter()
            inner, config = _agent_for(spec, catalog)
            cold_start = time.perf_counter() - started
            observed = ObservedAgent(inner)
            eval_started = time.perf_counter()
            result = evaluate(observed, samples, ids, categories, products)
            elapsed = time.perf_counter() - eval_started
            diagnostics = summarize_traces(observed.traces, samples, result["sessions"])
            startup_fallbacks = getattr(inner, "startup_fallbacks", {})
            runs.append({
                "name": spec.name,
                "kind": spec.kind,
                "config_path": str(spec.config_path) if spec.config_path else None,
                "config": config.to_dict() if config is not None else None,
                "sessions": result["sessions"],
                "metrics": _metric_row(
                    result,
                    diagnostics,
                    startup_fallbacks if isinstance(startup_fallbacks, dict) else {},
                    cold_start,
                    elapsed,
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if platform.system() == "Darwin" else 1024),
                ),
            })
        except BaseException as error:
            evaluation_error = error
            raise
        finally:
            if inner is not None and hasattr(inner, "close"):
                try:
                    inner.close()
                except Exception:
                    if evaluation_error is None:
                        raise
    if catalog_sha256 != file_sha256(catalog) or dataset_sha256 != file_sha256(dataset):
        raise RuntimeError("Catalog or dataset changed during evaluation")
    return {
        "schema": "mercury-evaluation-suite-v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "catalog": str(catalog),
        "dataset": str(dataset),
        "catalog_sha256": catalog_sha256,
        "dataset_sha256": dataset_sha256,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "suite_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if platform.system() == "Darwin" else 1024),
        "source_changed_during_run": source_before != source_hashes(),
        "source_hashes": source_before,
        "runs": runs,
        "interpretation": (
            "Public and synthetic scores are development evidence, not private-test performance. "
            "A selected neural config with missing model assets is a fallback run, not neural reproduction."
        ),
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# Mercury Evaluation Suite",
        "",
        f"- Catalog SHA-256: `{report['catalog_sha256']}`",
        f"- Dataset SHA-256: `{report['dataset_sha256']}`",
        f"- Source changed during run: `{report['source_changed_during_run']}`",
        "",
        "| Run | HitRate@10 | MRR | MTTC | TechnicalScore | p50 | p95 | Fallback Turns | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        metrics = run["metrics"]
        usage = metrics["token_usage"]
        lines.append(
            f"| {run['name']} | {metrics['hit_rate_at_10']:.6f} | {metrics['mrr']:.6f} | "
            f"{metrics['mttc']:.6f} | {metrics['technical_score']:.6f} | "
            f"{metrics['latency_p50_seconds']:.3f}s | {metrics['latency_p95_seconds']:.3f}s | "
            f"{metrics['fallback_turns']} | {usage['total_tokens']} |"
        )
    lines.extend(["", "## Scenario Breakdown", ""])
    for run in report["runs"]:
        lines.extend([
            f"### {run['name']}",
            "",
            "| Scenario | n | HitRate@10 | MRR | MTTC |",
            "|---|---:|---:|---:|---:|",
        ])
        for scenario, metrics in run["metrics"]["scenario_metrics"].items():
            lines.append(
                f"| {scenario} | {metrics['sample_count']} | {metrics['hit_rate_at_10']:.6f} | "
                f"{metrics['mrr']:.6f} | {metrics['mttc']:.6f} |"
            )
        fallbacks = run["metrics"]["startup_fallbacks"]
        if fallbacks:
            lines.extend(["", f"Startup fallbacks: `{json.dumps(fallbacks, sort_keys=True)}`"])
        lines.append("")
    lines.extend(["## Interpretation", "", report["interpretation"], ""])
    return "\n".join(lines)


def write_report(report: dict, output: Path) -> None:
    report_json = json.dumps(report, indent=2, allow_nan=False) + "\n"
    report_markdown = markdown_report(report)
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "report.json").write_text(report_json, encoding="utf-8")
        (output / "report.md").write_text(report_markdown, encoding="utf-8")
    except BaseException:
        for name in ("report.json", "report.md"):
            try:
                (output / name).unlink(missing_ok=True)
            except OSError:
                pass
        try:
            output.rmdir()
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline, Mercury fallback, selected, and candidate configs.")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-defaults", action="store_true")
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH",
                        help="Additional candidate config to evaluate.")
    args = parser.parse_args()
    specs = [] if args.no_defaults else default_specs()
    specs.extend(parse_config_spec(value) for value in args.candidate)
    report = evaluate_suite(specs, args.catalog, args.dataset)
    write_report(report, args.output)
    print(markdown_report(report))


if __name__ == "__main__":
    main()
